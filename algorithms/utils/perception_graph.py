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

class Attblock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.nhead = nhead
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, trg):
        #q = k = self.with_pos_embed(src, pos)
        
        q = src.permute(1,0,2)
        k = trg.permute(1,0,2)
        #src_mask = ~src_mask.bool(), key_padding_mask=src_mask src_mask
        
        src2, attention = self.attn(q, k, value=k)
        src2 = src2.permute(1,0,2)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attention

class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, dropout =0.1, hidden_dim=512, init='xavier'):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(input_dim, hidden_dim, init=init)
        self.gc2 = GraphConvolution(hidden_dim, hidden_dim, init=init)
        self.gc3 = GraphConvolution(hidden_dim, output_dim, init=init)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_graph, adj):
        # make the batch into one graph
        big_graph = torch.cat([graph for graph in batch_graph],0)
        B, N = batch_graph.shape[0], batch_graph.shape[1]
        big_adj = torch.zeros(B*N, B*N).to(batch_graph.device)
    
        for b in range(B):
            big_adj[b*N:(b+1)*N,b*N:(b+1)*N] = adj[b]
        
        x = self.dropout(F.relu(self.gc1(big_graph,big_adj)))
        x = self.dropout(F.relu(self.gc2(x,big_adj)))
        big_output = self.gc3(x, big_adj)

        batch_output = torch.stack(big_output.split(N))
        return batch_output

class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Net, self).__init__()
        self.conv1 = GCNConv(in_channels, in_channels//2)
        self.conv2 = GCNConv(in_channels//2, in_channels//2)
        self.conv3 = GCNConv(in_channels//2, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training = True)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training = True)
        x = self.conv3(x, edge_index)
        return x

def init_(m):
    init_method = nn.init.orthogonal_
    gain = nn.init.calculate_gain('relu')
    return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

    


class Perception_Graph(torch.nn.Module):
    def __init__(self, args, graph_linear):
        super(Perception_Graph, self).__init__()
        self.hidden_size = 512
        self.use_each_node = args.use_each_node
        if self.use_each_node:
            self.agent_graph = GCN(args.feature_dim+32, self.hidden_size)
            self.frontier_graph = GCN(args.feature_dim+32, self.hidden_size)
        else:
            self.agent_graph = Net(args.feature_dim+32, self.hidden_size)
            self.frontier_graph = Net(args.feature_dim+32, self.hidden_size)
        self.visual_encoder = self.load_visual_encoder(args.feature_dim)
        self.adap_pool = torch.nn.AdaptiveAvgPool1d(args.feature_dim)
        self.num_agents = args.num_agents
        #self.output_size = 2
        self.feature_init = init_(nn.Linear(self.hidden_size, self.hidden_size))
        self.node_init = nn.Sequential(
            init_(nn.Linear(3, 32)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            init_(nn.Linear(32, 64)),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            init_(nn.Linear(64, 128)),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            init_(nn.Linear(128,256)),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            init_(nn.Linear(256,32)))
        self.dis_init = nn.Sequential(
            init_(nn.Linear(1, 32)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            init_(nn.Linear(32, 64)),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            init_(nn.Linear(64, 128)),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            init_(nn.Linear(128,256)),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            init_(nn.Linear(256,32)))
        self.agent_init = nn.Sequential(
            init_(nn.Linear(3, 32)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            init_(nn.Linear(32, 64)),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            init_(nn.Linear(64, 128)),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            init_(nn.Linear(128,256)),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            init_(nn.Linear(256,32)))
        
        self.query = nn.Sequential(init_(nn.Linear(32, 32)))
        self.key = nn.Sequential(init_(nn.Linear(32, 32)))
        self.value = nn.Sequential(init_(nn.Linear(32, 32)))
        self.edge_mlp = nn.Sequential(
            init_(nn.Linear(96, 32)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            init_(nn.Linear(32, 1)))
        self.node_mlp = nn.Sequential(
            init_(nn.Linear(64, 64)),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            init_(nn.Linear(64, 32)))
        self.graph_linear = graph_linear
        #if graph_linear or self.use_each_node:
        self.graph_Decoder = Attblock(self.hidden_size, 4, self.hidden_size, 0.1)
            #self.agent_curr_Decoder = Attblock(512, 4, 1024, 0.1)
        # self.attention_layer = AttentionModule(dimensions=self.hidden_size, 
        #                                         attention_type='general')
        
    def forward(self, observations, masks, frontier_graph_data_origin, agent_graph_data_origin): 
        #curr_embedding = observations['curr_embedding']
        
        if self.use_each_node:
            batch_shape = observations['graph_ghost_node_position'].shape
            agent_graph_data = self.embed_agent_obs(observations)
            ghost_node_position = self.node_init(observations['graph_ghost_node_position'].reshape(batch_shape[0],-1,batch_shape[-1]))
            ghost_feature = torch.cat((observations['graph_merge_ghost_feature'].reshape(batch_shape[0],batch_shape[1]*batch_shape[2],-1), ghost_node_position),dim = 2) 
            edge = observations['graph_ghost_valid_mask']
            ghost_feature = self.frontier_graph(ghost_feature, edge)
            agent_node_position = self.node_init(observations['agent_world_pos'])
            agent_node_info = torch.cat((agent_graph_data, agent_node_position),dim = 2)
            agent_edge = torch.ones((batch_shape[0],self.num_agents,self.num_agents)).to("cuda:0")
            agent_node_info = self.agent_graph(agent_node_info, agent_edge)
            prediction, _ = self.graph_Decoder(agent_node_info[:,0:1], ghost_feature)
            return prediction.squeeze(1)

        else:
            frontier_graph_data = copy.deepcopy(frontier_graph_data_origin)
            agent_graph_data = copy.deepcopy(agent_graph_data_origin)
            agent_graph_data = self.embed_obs(observations, agent_graph_data)
            dis = []
            
            for i in range(len(observations['graph_ghost_node_position'])):
                origin_dis = observations['graph_agent_dis'][i][observations['graph_merge_ghost_mask'][i]!=0]
                dis.append(self.dis_init(origin_dis))
                ghost_node_position = observations['graph_ghost_node_position'][i][observations['graph_merge_ghost_mask'][i]!=0]
                agent_node_position = observations['agent_world_pos'][i]
                
                ghost_node_position = self.node_init(ghost_node_position)
                #frontier_graph_data[i].x = self.feature_init(frontier_graph_data[i].x)
                
                frontier_node_info = torch.cat((frontier_graph_data[i].x, ghost_node_position),dim = 1)
                #torch.cat((frontier_graph_data[i].x, ghost_node_position),dim = 1)
                frontier_graph_data[i].x = frontier_node_info
                
                # all_edge.append(frontier_node_info[:,0])
                
                # frontier_node_info = self.node_init(torch.cat((frontier_graph_data[i].x, ghost_node_position),dim = 1))
                # frontier_graph_data[i].x = frontier_node_info

                agent_node_position = self.node_init(agent_node_position)
                #agent_graph_data[i].x = self.feature_init(agent_graph_data[i].x)
                agent_node_info = torch.cat((agent_graph_data[i].x, agent_node_position),dim = 1)
                agent_graph_data[i].x = agent_node_info
                
                # agent_node_info = self.node_init(torch.cat((agent_graph_data[i].x, agent_node_position),dim = 1))
                # agent_graph_data[i].x = agent_node_info
                #torch.cat((agent_graph_data[i].x,agent_node_info),dim=1)
            #agent_loader = DataLoader(agent_graph_data, batch_size=len(agent_graph_data))
            #for agent_load in agent_loader:
            #agent_load = Batch.from_data_list(agent_graph_data)
                agent_out = self.agent_graph(agent_graph_data[i])
                agent_graph_data[i].x = agent_out
            #agent_graph_data = Batch.to_data_list(agent_load)
        
        
            #frontier_load = Batch.from_data_list(frontier_graph_data)
            #frontier_loader = DataLoader(frontier_graph_data, batch_size=len(frontier_graph_data))
            #for frontier_load in frontier_loader:
                frontier_out = self.frontier_graph(frontier_graph_data[i])
                frontier_graph_data[i].x = frontier_out
            #frontier_graph_data = Batch.to_data_list(frontier_load)
            # all_graph_data = []
            # all_agent_data = []
            all_edge = []
            for i in range(len(agent_graph_data)):
                
                if self.graph_linear:
                    edge_prediction = self.graph_Decoder(agent_graph_data[i].x.unsqueeze(0), frontier_graph_data[i].x.unsqueeze(0))
                else:
                    self.intra_graph_operator(ghost_node_position)
                    edge_prediction = self.intra_graph_operator(agent_graph_data[i].x.unsqueeze(0), frontier_graph_data[i].x.unsqueeze(0))
                if self.graph_linear:
                    all_edge.append(edge_prediction[0].squeeze(0))
                else:
                    all_edge.append(edge_prediction[1][:,0].squeeze(0))
                # graph_data, _ = self.graph_curr_Decoder(curr_embedding[i:i+1].unsqueeze(1), frontier_graph_data[i].x.unsqueeze(1))
                # agent_data, _ = self.agent_curr_Decoder(curr_embedding[i:i+1].unsqueeze(1), agent_graph_data[i].x.unsqueeze(1))
                # all_graph_data.append(graph_data.squeeze(1))
                # all_agent_data.append(agent_data.squeeze(1))'''
            
            return all_edge
        #torch.stack(all_edge, dim=0) #torch.cat(all_graph_data, dim = 0), torch.cat(all_agent_data, dim = 0)
    
    def load_visual_encoder(self, feature_dim):
        visual_encoder = resnet18(num_classes=feature_dim)
        dim_mlp = visual_encoder.fc.weight.shape[1]
        visual_encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), visual_encoder.fc)
        ckpt_pth = onpolicy.__path__[0]+ "/envs/habitat/model/PCL/PCL_encoder.pth"
        ckpt = torch.load(ckpt_pth, map_location=torch.device("cuda:0"))
        visual_encoder.load_state_dict(ckpt)
        visual_encoder.eval()
        return visual_encoder

    def embed_obs(self, obs_batch, agent_graph_data):
        
        patch_width = 2*obs_batch['graph_panoramic_rgb'].shape[3]//12
        with torch.no_grad():
            for idx in range(obs_batch['graph_panoramic_rgb'].shape[0]):
                vis_embed = []
                for i in range(12):
                    if i == 0:
                        img_tensor_before = torch.cat((torch.tensor(obs_batch['graph_panoramic_rgb'][idx,:,:,obs_batch['graph_panoramic_rgb'].shape[3]-patch_width//2:]).clone().detach()/255.0, torch.tensor(obs_batch['graph_panoramic_depth'][idx,:,:,obs_batch['graph_panoramic_rgb'].shape[3]-patch_width//2:]).clone().detach()),3).permute(0,3,1,2)
                        img_tensor_after =  torch.cat((torch.tensor(obs_batch['graph_panoramic_rgb'][idx,:,:,:patch_width//2]).clone().detach()/255.0, torch.tensor(obs_batch['graph_panoramic_depth'][idx,:,:,:patch_width//2]).clone().detach()),3).permute(0,3,1,2)   
                        img_tensor = torch.cat((img_tensor_before, img_tensor_after),dim=3)
                    else:
                        img_tensor = torch.cat((torch.tensor(obs_batch['graph_panoramic_rgb'][idx,:,:,i*patch_width//2:patch_width+i*patch_width//2]).clone().detach()/255.0, torch.tensor(obs_batch['graph_panoramic_depth'][idx,:,:,i*patch_width//2:patch_width+i*patch_width//2]).clone().detach()),3).permute(0,3,1,2)
                    
                    vis_embedding = self.visual_encoder(img_tensor)
                    vis_embed.append(vis_embedding)
                vis_embed = torch.cat(vis_embed,dim=1)
                vis_embed = self.adap_pool(vis_embed.unsqueeze(1))
                
                agent_graph_data[idx].x= nn.functional.normalize(vis_embed.squeeze(1),dim=1)
        return agent_graph_data
    
    def embed_agent_obs(self, obs_batch):
        agent_graph_data = []
        patch_width = 2*obs_batch['graph_panoramic_rgb'].shape[3]//12
        with torch.no_grad():
            for idx in range(obs_batch['graph_panoramic_rgb'].shape[0]):
                vis_embed = []
                for i in range(12):
                    if i == 0:
                        img_tensor_before = torch.cat((torch.tensor(obs_batch['graph_panoramic_rgb'][idx,:,:,obs_batch['graph_panoramic_rgb'].shape[3]-patch_width//2:]).clone().detach()/255.0, torch.tensor(obs_batch['graph_panoramic_depth'][idx,:,:,obs_batch['graph_panoramic_rgb'].shape[3]-patch_width//2:]).clone().detach()),3).permute(0,3,1,2)
                        img_tensor_after =  torch.cat((torch.tensor(obs_batch['graph_panoramic_rgb'][idx,:,:,:patch_width//2]).clone().detach()/255.0, torch.tensor(obs_batch['graph_panoramic_depth'][idx,:,:,:patch_width//2]).clone().detach()),3).permute(0,3,1,2)   
                        img_tensor = torch.cat((img_tensor_before, img_tensor_after),dim=3)
                    else:
                        img_tensor = torch.cat((torch.tensor(obs_batch['graph_panoramic_rgb'][idx,:,:,i*patch_width//2:patch_width+i*patch_width//2]).clone().detach()/255.0, torch.tensor(obs_batch['graph_panoramic_depth'][idx,:,:,i*patch_width//2:patch_width+i*patch_width//2]).clone().detach()),3).permute(0,3,1,2)
                    
                    vis_embedding = self.visual_encoder(img_tensor)
                    vis_embed.append(vis_embedding)
                vis_embed = torch.cat(vis_embed,dim=1)
                vis_embed = self.adap_pool(vis_embed.unsqueeze(1))
                agent_graph_data.append(nn.functional.normalize(vis_embed.squeeze(1),dim=1))
            agent_graph_data = torch.stack(agent_graph_data, dim = 0)
        return agent_graph_data
    
    def intra_graph_operator(self, x):
        q = self.query(x)
        k = self.key(x).transpose(2,1)
        v = self.value(x)
        score = torch.matmul(q,k)
        e = F.softmax(score, dim=-1)
        node_inp = torch.cat((x, torch.matmul(e,v)),dim=-1)
        x = x + self.node_mlp(node_inp)
        return x
    
    def inter_graph_operator(self, x, y, dis):
        x_q = self.query(x)
        y_k = self.key(y[0:1])
        y_v = self.value(y[0:1])
        #r_i = x_q.unsqueeze(0)
        #r_i = r_i.repeat(y.shape[0],1,1)
        #f_j = y_k.unsqueeze(1)
        f_j = y_k.repeat(x.shape[0],1)
        edge_input = torch.cat((x_q, f_j, dis), dim=-1)
        score = self.edge_mlp(edge_input).transpose(0,1)
        e = F.softmax(score, dim=0)
        # node_inp = torch.cat((x, torch.matmul(e,y_v)),dim=-1)
        # x = x + self.node_mlp(node_inp)
        
        return e

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
            # print('x_is:_{}'.format(x[i]))
            # print('x_max_is:_{}'.format(x[i].max()))
            # print('x_min_is:_{}'.format(x[i].min()))
            # print('x_avg_is:_{}'.format(x[i].mean()))
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
    
    def optimal_transport(self, P, eps=1e-2):
        u = torch.zeros(P.shape[1], device=self.device)
        while torch.max(torch.abs(u-P.sum(0))) > eps:
            u = P.sum(0)
            P = P/(u.unsqueeze(0))
            P = P/(P.sum(1).unsqueeze(1))
        return P
