import torch
import torch.nn as nn
from icecream import ic
from .util import init
from .distributions import Categorical
import numpy as np
from onpolicy.envs.habitat.utils.fmm_planner import FMMPlanner
import torch.nn.functional as F

def init_(m):
    init_method = nn.init.orthogonal_
    gain = nn.init.calculate_gain('relu')
    return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

class mGNNBase(nn.Module):
    def __init__(self, args, device):
        super(mGNNBase, self).__init__()
        self.max_frontier = args.max_frontier
        self.num_agents = args.num_agents
        self.output_size = args.max_frontier
        self.mgnn_blocks = args.mgnn_blocks
        self.map_resolution = args.map_resolution
        self.map_size_cm = args.map_size_cm
        self.hidden_size = 256
        
        self.node_init = nn.Sequential(
            init_(nn.Linear(3, self.hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(self.hidden_size,32))
        )#TODO did they use parameter sharing?
        self.blocks = [mGNNBlock(args, device) for _ in range(self.mgnn_blocks)]
        for b in self.blocks:
            b.to(device)
            
    def forward(self, inp, mask):
        self.batch_size = inp['obstacle_map'].shape[0]
        obstacle_map = inp['obstacle_map'].detach().cpu().numpy()
        robot_frontier_dis = [[] for _ in range(self.batch_size)]
        frontier_past_dis = [[] for _ in range(self.batch_size)]
        
        frontier_count = (torch.sum(inp['curr_frontier_mask'], dim=-1)).detach().cpu().numpy().astype(np.int32)
        past_step_count = int(torch.sum(inp['past_agent_pos_mask'], dim=-1)[0].item())
        agent_self_graph = self.node_init(inp['agent_world_pos'])
        frontier_self_graph = []
        robot_past_dis = torch.zeros(self.batch_size, self.num_agents, self.num_agents*past_step_count)
        ic(past_step_count)
        ic(inp['curr_frontier'].shape)
        for e in range(self.batch_size):
            frontier_self_graph.append(self.node_init((inp['curr_frontier'][e,:frontier_count[e]])).unsqueeze(0))
        
        agent_history = self.node_init(inp['past_agent_pos'][:,:past_step_count].view(self.batch_size, -1, 3))
        goal_history = self.node_init(inp['past_global_goal'][:,:past_step_count].view(self.batch_size,-1, 3))
        
        for e in range(self.batch_size):
            planner = FMMPlanner(1-obstacle_map[e], 360//10, 1, use_distance=True)
            for agent in range(self.num_agents):
                y, x = inp['agent_world_pos'][e, agent ,0], inp['agent_world_pos'][e, agent ,1]
                dis = planner.set_goal((x, y))
                fx = inp['curr_frontier'][e,:frontier_count[e],1].detach().cpu().numpy()
                fy = inp['curr_frontier'][e,:frontier_count[e],0].detach().cpu().numpy()
                robot_frontier_dis[e].append(dis[fx.astype(int), fy.astype(int)])
                fx = inp['past_agent_pos'][e,:past_step_count,:,1].view(-1).detach().cpu().numpy()
                fy = inp['past_agent_pos'][e,:past_step_count,:,0].view(-1).detach().cpu().numpy()
                robot_past_dis[e,agent] = torch.tensor(dis[fx.astype(int), fy.astype(int)])
            past_frontier = inp['past_global_goal'][e,:past_step_count, :].view(-1,3)
            for i in range(self.num_agents*past_step_count):
                y, x = past_frontier[i,0], past_frontier[i,1]
                dis = planner.set_goal((x, y))
                fx = inp['curr_frontier'][e,:frontier_count[e],1].detach().cpu().numpy()
                fy = inp['curr_frontier'][e,:frontier_count[e],0].detach().cpu().numpy()
                frontier_past_dis[e].append(dis[fx.astype(int), fy.astype(int)])
            robot_frontier_dis[e] = torch.tensor(robot_frontier_dis[e])
            frontier_past_dis[e] = torch.tensor(frontier_past_dis[e]).transpose(1,0)
        #with torch.autograd.set_detect_anomaly(True)
        for block in self.blocks:
            agent_self_graph, frontier_self_graph, agent_history, goal_history, e = block(agent_self_graph, frontier_self_graph, agent_history, goal_history, robot_frontier_dis, robot_past_dis, frontier_past_dis)
        #e[0].mean().backward()
        return e

class mGNNBlock(nn.Module):
    def __init__(self, args, device):
        super(mGNNBlock, self).__init__()
        self.hidden_size = 256
        self.device=device
        self.query = init_(nn.Linear(32, 32))
        self.key = init_(nn.Linear(32, 32))
        self.value = init_(nn.Linear(32, 32))
        self.node_mlp = nn.Sequential(
            init_(nn.Linear(64, self.hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(self.hidden_size, 32))
        )
        self.edge_mlp = nn.Sequential(
            init_(nn.Linear(65, self.hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(self.hidden_size, 1))
        )
        
    
    def forward(self, robot, frontier, robot_history, frontier_history, robot_frontier, robot_past, frontier_past):
        batch_size = len(frontier)
        robot = self.intra_graph_operator(robot)
        split_frontier = []
        new_frontier = []
        output_edge = []
        for e in range(batch_size):
            split_frontier.append(self.intra_graph_operator(frontier[e]))
        robot_history = self.intra_graph_operator(robot_history)
        frontier_history = self.intra_graph_operator(frontier_history)
        _,robot = self.inter_graph_operator(robot, robot_history, robot_past)
        for i in range(batch_size):
            _,temp_frontier = self.inter_graph_operator(split_frontier[i], frontier_history[i].unsqueeze(0), frontier_past[i])
            temp_edge, robot[i] = self.inter_graph_operator(robot[i].unsqueeze(0), temp_frontier, robot_frontier[i])
            output_edge.append(temp_edge)
            new_frontier.append(temp_frontier)
        frontier = split_frontier
        #output_edge[0].mean().backward()
        return robot, frontier, robot_history, frontier_history, output_edge

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
        y_k = self.key(y)
        y_v = self.value(y)
        r_i = torch.unsqueeze(x_q, dim=2)
        r_i = r_i.repeat([1,1,y.shape[1],1])
        f_j = torch.unsqueeze(y_k, dim=1)
        f_j = f_j.repeat([1,x.shape[1],1,1])
        if len(f_j.shape) != len(dis.shape):
            dis = dis.unsqueeze(-1)
            if len(f_j.shape) != len(dis.shape):
                dis = dis.unsqueeze(0)
        dis = dis.to(self.device)
        edge_input = torch.cat((r_i,f_j,dis),dim=-1).to(torch.float32)
        score = torch.squeeze(self.edge_mlp(edge_input))
        e = F.softmax(score, dim=-1)
        node_inp = torch.cat((x, torch.matmul(e,y_v)),dim=-1)
        x = x + self.node_mlp(node_inp)
        return e, x

class LinearAssignment(nn.Module):
    def __init__(self, args, device):
        super(LinearAssignment, self).__init__()
        self.num_agents = args.num_agents
        self.device = device
    
    def forward(self, x, available_actions=None, deterministic=False):
        batch_size = len(x)
        actions = torch.zeros(batch_size, self.num_agents, 1, device=self.device)
        action_log_probs = torch.zeros(actions.shape, device=self.device)
        for i in range(len(x)):
            action_out = Categorical(x[i].shape[-1], 1)
            action_out.to(self.device)
            action_feature = self.optimal_transport(x[i])
            action_logits = action_out(action_feature, available_actions)
            actions[i] = action_logits.mode() if deterministic else action_logits.sample()
            action_log_probs[i] = action_logits.log_probs(actions[i])
        return torch.flatten(actions).unsqueeze(1), torch.flatten(action_log_probs).unsqueeze(1)
    

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None):
        action_log_probs = torch.zeros(len(x), self.num_agents, 1)
        dist_entropy = torch.zeros(len(x))
        for i in range(len(x)):
            action_out = Categorical(x[i].shape[-1], 1)
            action_out = action_out.to(self.device)
            action_logits = action_out(x[i], available_actions)
            action_log_probs[i] = action_logits.log_probs(action[i])
            dist_entropy[i] = action_logits.entropy().mean()
        #dist_entropy.mean().backward()
        return action_log_probs, dist_entropy.mean()
    
    def optimal_transport(self, P, eps=1e-8):
        u = torch.zeros(P.shape[1], device=self.device)
        while torch.max(torch.abs(u-P.sum(0))) > eps:
            u = P.sum(0)
            P = P/(u.unsqueeze(0))
            P = P/(P.sum(1).unsqueeze(1))
        return P