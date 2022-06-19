import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from .agent_attention import AttentionModule
from torch_geometric.data import Batch
from .graph_layer import GraphConvolution
import onpolicy
from onpolicy.envs.habitat.model.PCL.resnet_pcl import resnet18
import torch.nn as nn
from .util import init
import copy
from .distributions import Categorical

def init_(m):
    init_method = nn.init.orthogonal_
    gain = nn.init.calculate_gain('relu')
    return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

def MLP(channels, do_bn=False):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z

class MLPAttention(nn.Module):
    def __init__(self, desc_dim):
        super().__init__()
        self.mlp = MLP([desc_dim * 3, desc_dim, 1])

    def forward(self, query, key, value, dist, mask):
        '''query: 1 x 128 x n_agent
        key: 1 x 128 x n_frontier
        dist: 1 x 128 x (n_agent x n_frontier)

        cat: 1 x 384 x (n_agent x n_frontier)

        value: 1 x 128 x n_frontier

        scores: 1 x n_agent x n_frontier

        output: n_agent x 128'''
        batch = query.shape[0]
        nq, nk = query.size(-1), key.size(-1)
        scores = self.mlp(torch.cat((
            query.view(batch, -1, nq, 1).repeat(1, 1, 1, nk).view(batch, -1, nq * nk),
            key.view(batch, -1, 1, nk).repeat(1, 1, nq, 1).view(batch, -1, nq * nk),
            dist), dim=1)).view(batch, nq, nk)
        if mask is not None:
            if type(mask) is float:
                scores_detach = scores.detach()
                scale = torch.clamp(mask / (scores_detach.max(2).values - scores_detach.median(2).values), 1., 1e3)
                scores = scores * scale.unsqueeze(-1).repeat(1, 1, nk)
            else:
                scores = scores + (scores.min().detach() - 20) * (~mask).float().view(1, nq, nk)
        prob = scores.softmax(dim=-1)
        return torch.einsum('bnm,bdm->bdn', prob, value), scores
        
class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([copy.deepcopy(self.merge) for _ in range(3)])

    def attention(self, query, key, value, mask):
        dim = query.shape[1]
        scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5
        if mask is not None:
            scores = scores + (scores.min().detach() - 20) * (~mask).float().unsqueeze(0).unsqueeze(0).repeat(1, self.num_heads, 1, 1)
        prob = torch.nn.functional.softmax(scores, dim=-1)
        return torch.einsum('bhnm,bdhm->bdhn', prob, value), scores

    def forward(self, query, key, value, dist, mask):
        batch = query.shape[0]
        query, key, value = [l(x).view(batch, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, scores = self.attention(query, key, value, mask)
        return self.merge(x.contiguous().view(batch, self.dim*self.num_heads, -1)), scores.mean(1)

class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int, type: str):
        super().__init__()
        self.attn = MLPAttention(feature_dim) if type == 'cross' else MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source, dist, mask):
        message, weights = self.attn(x, source, source, dist, mask)
        
        return self.mlp(torch.cat([x, message], dim=1)), weights

class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list, num_agents: int):
        super().__init__()
        self.attn = nn.ModuleList([AttentionalPropagation(feature_dim, 4, type) for type in layer_names])
        self.phattn = nn.ModuleList([AttentionalPropagation(feature_dim, 4, 'self') for type in layer_names])
        self.ghattn = nn.ModuleList([AttentionalPropagation(feature_dim, 4, 'self') for type in layer_names])
        # self.attn = MLP([feature_dim, 1])
        self.score_layer = MLP([2*feature_dim, feature_dim, 1])
        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)
        self.names = layer_names
        self.num_agents = num_agents

    def forward(self, desc0, desc1, desc2, desc3, dist, mask):
        # desc0: frontier
        # desc1: agent
        # desc2: agent_history
        # desc3: goal_history
        # fidx: n_frontier x 2
        batch = dist.shape[0]
        dist0 = dist.reshape(batch, -1, desc1.size(-1) * desc0.size(-1))
        dist1 = dist.transpose(1, 2).reshape(batch, -1, desc1.size(-1) * desc0.size(-1))
        #view(-1, desc1.size(-1) desc0.size(-1)).transpose(1, 2).reshape(1, -1, desc1.size(-1) * desc0.size(-1))

        for idx, attn, phattn, ghattn, name in zip(range(len(self.names)), self.attn, self.phattn, self.ghattn, self.names):
           
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:
                src0, src1 = desc0, desc1
            
            delta0, score0 = attn(desc0, src0, dist0, None)
            delta1, score1 = attn(desc1, src1, dist1, None)

            if name == 'cross':
                if desc2 is not None:
                    delta21, _ = phattn(desc2, desc1, None, None)
                    delta12, _ = phattn(desc1, desc2, None, None)
                    desc2 = desc2 + delta21
                else:
                    delta12 = 0
                if desc3 is not None:
                    delta03, _ = ghattn(desc0, desc3, None, None)
                    delta30, _ = ghattn(desc3, desc0, None, None)
                    desc3 = desc3 + delta30
                else:
                    delta03 = 0
                desc0, desc1 = (desc0 + delta0 + delta03), (desc1 + delta1 + delta12)
            else:  # if name == 'self':
                if desc2 is not None:
                    delta2, _ = phattn(desc2, desc2, None, None)
                    desc2 = desc2 + delta2
                if desc3 is not None:
                    delta3, _ = ghattn(desc3, desc3, None, None)
                    desc3 = desc3 + delta3
                desc0, desc1 = (desc0 + delta0), (desc1 + delta1)

        # weights1: n_agent x n_frontier
        # fidx: n_agent x n_frontier x 2
        # assert (~invalid).any(1).all()
        
        scores = score1
        scores = log_optimal_transport(scores.log_softmax(dim=-2), self.bin_score, iters=5)[:, :-1, :-1]
        
        score_min = scores.min() - scores.max()
        mask_bool = ~(mask.bool())
        mask_int = mask_bool.int()
        scores_agent = scores[:,0] + (score_min - 40) * mask_int.reshape(batch, -1)
        return scores_agent * 15

class Batch_Perception_Graph(torch.nn.Module):
    def __init__(self, args, graph_linear):
        super(Batch_Perception_Graph, self).__init__()       
        self.num_agents = args.num_agents
        self.use_frontier_nodes = args.use_frontier_nodes
        feature_dim = 32
        layers = [32, 64, 128, 256]
        gnn_layers = 3 * ['self', 'cross']
        self.node_init = MLP([4] + layers + [feature_dim])
        nn.init.constant_(self.node_init[-1].bias, 0.0)
        self.dis_init = MLP([1, feature_dim, feature_dim])
        self.gnn = AttentionalGNN(feature_dim, gnn_layers, self.num_agents)
        
    def forward(self, observations, masks, frontier_graph_data_origin, agent_graph_data_origin): 
        frontier_graph_data = copy.deepcopy(frontier_graph_data_origin)
        last_frontier_data = copy.deepcopy(frontier_graph_data_origin)
        agent_graph_data = copy.deepcopy(agent_graph_data_origin)
        last_agent_data = copy.deepcopy(agent_graph_data_origin)   
        batch = observations['graph_agent_dis'].shape[0]
        global_step = int(observations['graph_last_pos_mask'][0].sum())
        if self.use_frontier_nodes:
            ghost_mask = observations['graph_merge_frontier_mask']
        else:
            ghost_mask = observations['graph_merge_ghost_mask']
        graph_agent_dis = self.dis_init(observations['graph_agent_dis'].view(batch,-1,1).transpose(1,2)).transpose(1,2).view(batch,self.num_agents,-1,32)
        ori_ghost_node_position = observations['graph_ghost_node_position'].view(batch,-1,4)
        ori_ghost_mask = ghost_mask.view(batch,-1).unsqueeze(-1).repeat(1,1,4)
        ori_ghost_node_position = ori_ghost_node_position*ori_ghost_mask
        #ori_ghost_node_position[ori_ghost_node_position==0] = -1
        ghost_node_position = self.node_init(ori_ghost_node_position.transpose(1,2))
        agent_node_position = self.node_init(observations['agent_world_pos'].transpose(1,2))
        last_ghost_position = self.node_init(observations['graph_last_ghost_node_position'][: , :global_step].transpose(1,2)) 
        last_agent_position = self.node_init(observations['graph_last_agent_world_pos'][:, :global_step].transpose(1,2)) 
        
        all_edge = self.gnn(ghost_node_position, agent_node_position, last_agent_position, last_ghost_position, graph_agent_dis, ghost_mask)
        
        return all_edge

class LinearAssignment(nn.Module):
    def __init__(self, args, device):
        super(LinearAssignment, self).__init__()
        self.num_agents = args.num_agents
        self.device = device
    
    def forward(self, x, available_actions=None, deterministic=False):
        #batch_size = len(x)
        actions = []#torch.zeros(batch_size, 1, 1, device=self.device)
        action_log_probs = []#torch.zeros(actions.shape, device=self.device)
        for i in range(len(x)):
            action_out = Categorical(x[i].shape[-1], x[i].shape[-1])
            #action_feature = self.optimal_transport(x[i])
            action_logits = action_out(x[i].unsqueeze(0), available_actions, trans= False)
            action = action_logits.mode() if deterministic else action_logits.sample()
            action_log_prob = action_logits.log_probs(action)
            actions.append(action)
            action_log_probs.append(action_log_prob)
        
        return torch.cat(actions,0), torch.cat(action_log_probs,0)
    

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None):
        action_log_probs = []
        dist_entropy = []
        for i in range(len(x)):
            action_out = Categorical(x[i].shape[-1], x[i].shape[-1])
            #action_feature = self.optimal_transport(x[i])
            action_logits = action_out(x[i].unsqueeze(0), available_actions, trans= False)
            action_log_probs.append(action_logits.log_probs(action[i].unsqueeze(0)))
            dist_entropy.append(action_logits.entropy().mean())

        return torch.cat(action_log_probs, 0),  torch.stack(dist_entropy, 0).mean()
    
    def optimal_transport(self, P, eps=1e-7):
        u = torch.zeros(P.shape[1], device=self.device)
        while torch.max(torch.abs(u-P.sum(0))) > eps:
            u = P.sum(0)
            P = P/(u.unsqueeze(0))
            P = P/(P.sum(1).unsqueeze(1))
        return P