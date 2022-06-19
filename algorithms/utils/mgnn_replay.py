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

class Perception_Graph(torch.nn.Module):
    def __init__(self, args, graph_linear):
        super(Perception_Graph, self).__init__()       
        self.num_agents = args.num_agents
        self.use_frontier_nodes = args.use_frontier_nodes
        self.node_init = nn.Sequential(
            init_(nn.Linear(4, 32)),
            #nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.LayerNorm(32),
            init_(nn.Linear(32, 64)),
            #nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.LayerNorm(64),
            init_(nn.Linear(64, 128)),
            #nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.LayerNorm(128),
            init_(nn.Linear(128,256)),
            #nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.LayerNorm(256),
            init_(nn.Linear(256,32)))
            #nn.ReLU(),
            #nn.BatchNorm1d(32),
            # nn.ReLU(),
            #nn.LayerNorm(32))

        self.dis_init = nn.Sequential(
            init_(nn.Linear(1, 32)),
            #nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.LayerNorm(32),
            init_(nn.Linear(32, 64)),
            #nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.LayerNorm(64),
            init_(nn.Linear(64, 128)),
            #nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.LayerNorm(128),
            init_(nn.Linear(128,256)),
            #nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.LayerNorm(256),
            init_(nn.Linear(256,32)))
            #nn.ReLU(),
            #nn.BatchNorm1d(32),
            # nn.ReLU(),
            #nn.LayerNorm(32))
        
        self.query = nn.Sequential(init_(nn.Linear(32, 32)))
            #nn.ReLU(),
            #nn.BatchNorm1d(32),
            #nn.ReLU(),
            #nn.LayerNorm(32))
        self.key = nn.Sequential(init_(nn.Linear(32, 32)))
            #nn.ReLU(),
            #nn.BatchNorm1d(32),
            # nn.ReLU(),
            #nn.LayerNorm(32))
        self.value = nn.Sequential(init_(nn.Linear(32, 32)))
            #nn.ReLU(),
            #nn.BatchNorm1d(32),
            # nn.ReLU(),
            #nn.LayerNorm(32))
        self.edge_mlp = nn.Sequential(
            init_(nn.Linear(96, 32)),
            #nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.LayerNorm(32),
            init_(nn.Linear(32, 1)))
            #nn.ReLU(),
            #nn.BatchNorm1d(1),
            # nn.ReLU(),
            #nn.LayerNorm(1))
        self.node_mlp = nn.Sequential(
            init_(nn.Linear(64, 64)),
            #nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.LayerNorm(64),
            init_(nn.Linear(64, 32)))
            #nn.ReLU(),
            #nn.BatchNorm1d(32),
            # nn.ReLU(),
            #nn.LayerNorm(32))
            
    def forward(self, observations, masks, frontier_graph_data_origin, agent_graph_data_origin): 
        # curr_embedding = observations['curr_embedding']
    
        frontier_graph_data = copy.deepcopy(frontier_graph_data_origin)
        last_frontier_data = copy.deepcopy(frontier_graph_data_origin)
        agent_graph_data = copy.deepcopy(agent_graph_data_origin)
        last_agent_data = copy.deepcopy(agent_graph_data_origin)
        # agent_graph_data = self.embed_obs(observations, agent_graph_data)
        ori_graph_agent_dis = []
        graph_agent_dis = []
        #last_agent_dis = []
        #ori_last_node_dis = []
        last_node_dis = []
        ghost_node_position_list = []
        batch = len(observations['graph_ghost_node_position'])
        global_step = int(observations['graph_last_pos_mask'][0].sum())
        for i in range(batch):
            dis = []
            for a in range(self.num_agents):
                if self.use_frontier_nodes:
                    origin_dis = observations['graph_agent_dis'][i][a, :int(torch.sum(observations['graph_merge_frontier_mask'][i]))] 
                else:
                    origin_dis = observations['graph_agent_dis'][i][a][observations['graph_merge_ghost_mask'][i].reshape(-1)!=0]
                dis.append(origin_dis)
            
            ori_graph_agent_dis.append(torch.cat(dis,dim=0))
            if self.use_frontier_nodes:
                # origin_last_node_dis = observations['last_graph_dis'][i, :global_step, :int(torch.sum(observations['graph_merge_frontier_mask'][i]))]
                # ori_last_node_dis.append(origin_last_node_dis.reshape(-1,1))
                ori_ghost_node_position = observations['graph_ghost_node_position'][i][:int(torch.sum(observations['graph_merge_frontier_mask'][i]))]
                ghost_node_position_list.append(ori_ghost_node_position)
            else:
                # origin_last_node_dis = observations['last_graph_dis'][i, :global_step, torch.where(observations['graph_merge_ghost_mask'][i].reshape(-1)!=0)[0]]
                # ori_last_node_dis.append(origin_last_node_dis.reshape(-1,1))
                ori_ghost_node_position = observations['graph_ghost_node_position'][i][observations['graph_merge_ghost_mask'][i]!=0]
                ghost_node_position_list.append(ori_ghost_node_position)
        
        ghost_node_position = self.node_init(torch.cat(ghost_node_position_list,dim = 0))  
        ori_graph_agent_dis = self.dis_init(torch.cat(ori_graph_agent_dis,dim = 0))
        #ori_last_node_dis = self.dis_init(torch.cat(ori_last_node_dis,dim = 0))
        agent_node_position = observations['agent_world_pos'].reshape(-1,4)
        agent_node_position = self.node_init(agent_node_position)
        last_ghost_position = observations['graph_last_ghost_node_position'][:, :global_step].reshape(-1,4)
        last_agent_position = observations['graph_last_agent_world_pos'][:, :global_step].reshape(-1,4)
        last_ghost_position = self.node_init(last_ghost_position)  
        last_agent_position = self.node_init(last_agent_position)
        
        last_idx = 0
        for i in range(batch):
            if self.use_frontier_nodes:
                tmp_a = ori_graph_agent_dis[self.num_agents*last_idx:self.num_agents*last_idx+self.num_agents*int(torch.sum(observations['graph_merge_frontier_mask'][i]))].reshape(self.num_agents,-1,32)
                #tmp_b = ori_last_node_dis[global_step*last_idx:global_step*last_idx+global_step*int(torch.sum(observations['graph_merge_frontier_mask'][i]))].reshape(global_step,-1,32)
                frontier_graph_data[i].x = ghost_node_position[last_idx:last_idx+int(torch.sum(observations['graph_merge_frontier_mask'][i]))].reshape(-1,32)
            else:
                tmp_a = ori_graph_agent_dis[self.num_agents*last_idx:self.num_agents*last_idx+self.num_agents*int(torch.sum(observations['graph_merge_ghost_mask'][i]))].reshape(self.num_agents,-1,32)
                #tmp_b = ori_last_node_dis[global_step*last_idx:global_step*last_idx+global_step*int(torch.sum(observations['graph_merge_ghost_mask'][i]))].reshape(global_step,-1,32)
                frontier_graph_data[i].x = ghost_node_position[last_idx:last_idx+int(torch.sum(observations['graph_merge_ghost_mask'][i]))].reshape(-1,32)
            
            graph_agent_dis.append(tmp_a)
            #last_node_dis.append(tmp_b)
            
            agent_graph_data[i].x = agent_node_position[i*self.num_agents:(i+1)*self.num_agents].reshape(-1,32)
            last_frontier_data[i].x = last_ghost_position[i*global_step:(i+1)*global_step].reshape(-1,32)
            last_agent_data[i].x = last_agent_position[i*global_step:(i+1)*global_step].reshape(-1,32)
            if self.use_frontier_nodes:
                last_idx += int(torch.sum(observations['graph_merge_frontier_mask'][i]))
            else:
                last_idx += int(torch.sum(observations['graph_merge_ghost_mask'][i]))
            
        #origin_last_agent_dis = observations['graph_last_agent_dis'][:, :global_step]
        #last_agent_dis = self.dis_init((origin_last_agent_dis.reshape(-1,1))).reshape(batch, global_step, self.num_agents, -1)
        # all_graph_data = []
        # all_agent_data = []
        for _ in range(3):
            frontier_graph_data = self.intra_graph_operator(frontier_graph_data)
            agent_graph_data = self.intra_graph_operator(agent_graph_data)
            last_frontier_data = self.intra_graph_operator(last_frontier_data)
            last_agent_data = self.intra_graph_operator(last_agent_data)
            _, last_frontier_data, frontier_graph_data = self.inter_graph_operator(last_frontier_data, frontier_graph_data, None)
            _, last_agent_data, agent_graph_data = self.inter_graph_operator(last_agent_data, agent_graph_data, None)
            e_all,  agent_graph_data, frontier_graph_data = self.inter_graph_operator(agent_graph_data, frontier_graph_data, graph_agent_dis)
        # all_edge = []
        # for i in range(batch):
        #     all_edge.append(e[0])
        return e_all
    
    def intra_graph_operator(self, xx):
        xx_all = []
        node_all = []
        for i in range(len(xx)):
            xx_all.append(xx[i].x)
        xx_all = torch.cat(xx_all,dim = 0)
        q_all = self.query(xx_all)
        k_all = self.key(xx_all)
        v_all = self.value(xx_all)
        idx = 0
        for i in range(len(xx)):
            q = q_all[idx:idx+xx[i].x.shape[0]]
            k = k_all[idx:idx+xx[i].x.shape[0]].transpose(0,1)
            v = v_all[idx:idx+xx[i].x.shape[0]]
            score = torch.matmul(q,k)
            e = F.softmax(score, dim=-1)
            xx[i].index = e
            node_inp = torch.cat((xx[i].x, torch.matmul(e,v)),dim=-1)
            idx += xx[i].x.shape[0]
            node_all.append(node_inp)
        node_all = torch.cat(node_all,dim = 0)
        node = self.node_mlp(node_all)
        idx = 0
        for i in range(len(xx)):
            xx[i].x = xx[i].x + node[idx:idx+xx[i].x.shape[0]]
            idx += xx[i].x.shape[0]
        return xx
    
    def inter_graph_operator(self, xx, yy, dis):
        xx_all = []
        yy_all = []
        for i in range(len(xx)):
            xx_all.append(xx[i].x)
            yy_all.append(yy[i].x)
        xx_all = torch.cat(xx_all, dim = 0)
        yy_all = torch.cat(yy_all, dim = 0)
        x_q_all = self.query(xx_all)
        y_k_all = self.key(yy_all)
        y_v_all = self.value(yy_all)
        y_q_all = self.query(yy_all)
        x_k_all = self.key(xx_all)
        x_v_all = self.value(xx_all)
        idx = 0
        idy = 0
        ori_edge_input_all = []
        ori_edge_copy_input_all = []
        for i in range(len(xx)):
            r_i = x_q_all[idx:idx+xx[i].x.shape[0]].unsqueeze(1)
            r_i = r_i.repeat(1,yy[i].x.shape[0],1)
            f_j = y_k_all[idy:idy+yy[i].x.shape[0]].unsqueeze(0)
            f_j = f_j.repeat(xx[i].x.shape[0],1,1)
            if dis is None:
                edge_input = torch.cat((r_i, f_j, torch.ones(r_i.shape[0], r_i.shape[1], 32).to(r_i.device)), dim=-1)
            else:
                edge_input = torch.cat((r_i, f_j, dis[i]), dim=-1)
            ori_edge_input_all.append(edge_input.reshape(-1, edge_input.shape[-1]))

            f_j = y_q_all[idy:idy+yy[i].x.shape[0]].unsqueeze(1)
            f_j = f_j.repeat(1,xx[i].x.shape[0],1)
            r_i = x_k_all[idx:idx+xx[i].x.shape[0]].unsqueeze(0)
            r_i = r_i.repeat(yy[i].x.shape[0],1,1)
            if dis is None:
                edge_copy_input = torch.cat((f_j, r_i, torch.ones(f_j.shape[0], f_j.shape[1], 32).to(r_i.device)), dim=-1)
            else:
                edge_copy_input = torch.cat((f_j, r_i, dis[i].transpose(0,1)), dim=-1)
            ori_edge_copy_input_all.append(edge_copy_input.reshape(-1,edge_copy_input.shape[-1]))
            idx += xx[i].x.shape[0]
            idy += yy[i].x.shape[0]
        
        edge_input_all = torch.cat(ori_edge_input_all, dim = 0)
        edge_copy_input_all = torch.cat(ori_edge_copy_input_all, dim = 0)
        score_all = self.edge_mlp(edge_input_all)
        score_copy_all = self.edge_mlp(edge_copy_input_all)
        idx = 0
        idy = 0
        idxy = 0
        node_inp_all = []
        node_copy_inp_all = []
        e_all = []
        for i in range(len(xx)):
            score = score_all[idxy:idxy+(xx[i].x.shape[0]*yy[i].x.shape[0])].reshape(xx[i].x.shape[0],yy[i].x.shape[0])
            e = F.softmax(score, dim=1)
            log_e = F.log_softmax(score, dim=0)*15
            e_all.append(log_e[0])
            y_v = y_v_all[idy:idy+yy[i].x.shape[0]]
            node_inp = torch.cat((xx[i].x, torch.matmul(e,y_v)),dim=-1)
            node_inp_all.append(node_inp)

            score_copy = score_copy_all[idxy:idxy+(xx[i].x.shape[0]*yy[i].x.shape[0])].reshape(yy[i].x.shape[0],xx[i].x.shape[0])
            e_copy = F.softmax(score_copy, dim=1)
            x_v = x_v_all[idx:idx+xx[i].x.shape[0]]
            node_copy_inp = torch.cat((yy[i].x, torch.matmul(e_copy,x_v)),dim=-1)
            node_copy_inp_all.append(node_copy_inp)

            idx += xx[i].x.shape[0]
            idy += yy[i].x.shape[0]
            idxy += xx[i].x.shape[0]*yy[i].x.shape[0]

        node_inp_all = torch.cat(node_inp_all, dim = 0)
        node_copy_inp_all = torch.cat(node_copy_inp_all, dim = 0)
        node_inp_all = self.node_mlp(node_inp_all)
        node_copy_inp_all = self.node_mlp(node_copy_inp_all)
        idx = 0
        idy = 0
        for i in range(len(xx)):
            xx[i].x = xx[i].x + node_inp_all[idx:idx+xx[i].x.shape[0]]
            yy[i].x = yy[i].x + node_copy_inp_all[idy:idy+yy[i].x.shape[0]]
            idx += xx[i].x.shape[0]
            idy += yy[i].x.shape[0]
        return e_all, xx, yy

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