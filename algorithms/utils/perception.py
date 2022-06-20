#from _typeshed import NoneType
import sys
import torch
import torch.nn.functional as F
from .graph_layer import GraphConvolution
import torch.nn as nn
import sys
sys.path.append("../..")

from mants_sim_to_real.algorithms.utils.vit import ViT, Attention, PreNorm, Transformer, CrossAttention, FeedForward

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

    def forward(self, src, trg, src_mask):
        #q = k = self.with_pos_embed(src, pos)
        q = src.permute(1,0,2)
        k = trg.permute(1,0,2)
        src_mask = ~src_mask.bool()
        
        src2, attention = self.attn(q, k, value=k, key_padding_mask=src_mask)
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

import math
class PositionEncoding(nn.Module):
    def __init__(self, n_filters=512, max_len=2000):
        """
        :param n_filters: same with input hidden size
        :param max_len: maximum sequence length
        """
        super(PositionEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_filters)  # (L, D)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_filters, 2).float() * - (math.log(10000.0) / n_filters))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # buffer is a tensor, not a variable, (L, D)

    def forward(self, x, times):
        """
        :Input: (*, L, D)
        :Output: (*, L, D) the same size as input
        """
        pe = []
        for b in range(x.shape[0]):
            pe.append(self.pe.data[times[b].long()]) # (#x.size(-2), n_filters)
        pe_tensor = torch.stack(pe)
        x = x + pe_tensor
        return x

class Perception(nn.Module):
    def __init__(self, args, graph_linear, embedding_network):
        super(Perception, self).__init__()

        self.args = args
        self.pe_method = 'pe' # or exp(-t)
        self.time_embedd_size = args.time_dim
        self.max_time_steps = args.max_episode_length
        # self.goal_time_embedd_index = args.max_episode_length
        self.num_agents = args.num_agents
        memory_dim = args.hidden_size
        self.hidden_size = args.hidden_size
        self.use_merge = args.use_merge
        self.use_tail_agent_info = args.use_tail_agent_info
        self.use_single = args.use_single
        self.use_other_agent = args.use_other_agent
        self.use_intra_attn = args.use_intra_attn
        self.add_ghost = args.add_ghost
        self.use_agent_node = args.use_agent_node
        self.use_distance = args.use_distance
        self.depth = 2
        
        self.graph_linear = graph_linear
        if self.pe_method == 'embedding':
            self.time_embedding = nn.Embedding(self.max_time_steps+2, self.time_embedd_size)
        elif self.pe_method == 'pe':
            self.time_embedding_1 = PositionEncoding(args.feature_dim, self.max_time_steps+10)
            self.time_embedding_2 = PositionEncoding(memory_dim, self.max_time_steps+10)
            if self.add_ghost:
                self.time_embedding_3 = PositionEncoding(args.feature_dim, self.max_time_steps+10)
        else:
            self.time_embedding = lambda t: torch.exp(-t.unsqueeze(-1)/5)

        feature_dim = args.feature_dim# + self.time_embedd_size
        #self.feature_embedding = nn.Linear(feature_dim, memory_dim)
        # self.feature_embedding = nn.Sequential(nn.Linear(feature_dim +  args.feature_dim , memory_dim),
        #                                        nn.ReLU(),
        #                                        nn.Linear(memory_dim, memory_dim))     
        self.feature_embedding = nn.Sequential(nn.Linear(args.feature_dim, memory_dim),
                                                nn.ReLU(),
                                                nn.Linear(memory_dim, memory_dim))
        # if self.add_ghost:
        #     self.feature_embedding_1 = nn.Sequential(nn.Linear(args.feature_dim, memory_dim),
        #                                             nn.ReLU(),
        #                                             nn.Linear(memory_dim, memory_dim))

        self.graph_global_GCN = GCN(input_dim=memory_dim, output_dim=memory_dim)
        self.graph_curr_Decoder = Attblock(args.hidden_size,
                                     4,
                                     1024,
                                     0.1)
        if self.use_agent_node:
            self.agent_global_GCN = GCN(input_dim=memory_dim, output_dim=memory_dim)
            self.agent_curr_Decoder = Attblock(args.hidden_size,
                                     4,
                                     1024,
                                     0.1)
        
        self.graph_encode_actor_net = nn.Linear(args.hidden_size*2, args.hidden_size)
        if self.use_agent_node:
            self.agent_encode_actor_net = nn.Linear(args.hidden_size*2, args.hidden_size)
        self.graph_encode_other_net = nn.Linear(args.hidden_size*2, args.hidden_size)
        
        if self.use_other_agent:
            self.agent_attn_layers = nn.ModuleList([])
            for _ in range(self.depth):
                self.agent_attn_layers.append(nn.ModuleList([
                        PreNorm(args.hidden_size, Attention(args.hidden_size, heads = 4, dim_head = 1024, dropout = 0.)),
                        PreNorm(args.hidden_size, FeedForward(args.hidden_size, args.hidden_size, dropout = 0.))
                    ]))
            self.last_cross_attn = nn.ModuleList([
                    nn.LayerNorm(args.hidden_size),
                    nn.Linear(args.hidden_size, 2 * 4 * 1024, bias = False),
                    CrossAttention(args.hidden_size, heads = 4, dim_head = 1024, dropout =  0.),
                    PreNorm(args.hidden_size, FeedForward(args.hidden_size, args.hidden_size, dropout =  0.))
                ])
        
        # if args.add_ghost:
        #     self.total_GCN = GCN(input_dim=memory_dim, output_dim=memory_dim)
        #     self.GCN_mixer = Attblock(args.hidden_size,
        #                                 4,
        #                                 1024,
        #                                 0.1)
        self.output_size = feature_dim
        if self.args.use_edge_info == 'learned':
            self.edge_linear = nn.Sequential(nn.Linear(6,48), nn.ReLU(), nn.Linear(48,1))


    def normalize_sparse_adj(self, adj):
        """Laplacian Normalization"""
        rowsum = adj.sum(1) # adj B * M * M
        if self.args.use_edge_info is None:
            r_inv_sqrt = torch.pow(rowsum, -0.5)
        else:
            r_inv_sqrt = torch.pow(rowsum, -1)
        r_inv_sqrt[torch.where(torch.isinf(r_inv_sqrt))] = 0.
        r_mat_inv_sqrt = torch.stack([torch.diag(k) for k in r_inv_sqrt])
        if self.args.use_edge_info is None:
            return torch.matmul(torch.matmul(adj, r_mat_inv_sqrt).transpose(1,2),r_mat_inv_sqrt)
        else:
            return torch.matmul(adj, r_mat_inv_sqrt)

    def forward(self, observations, mode='train', return_features=False): # without memory
        
        if self.use_merge:
            curr_context = []
            curr_attn = []
            B = observations['graph_merge_global_mask'].shape[0]
            max_node_num = observations['graph_merge_global_mask'].sum(dim=1).max().long()
            relative_time = observations['graph_time'] - observations['graph_merge_global_time'][:, :max_node_num]
            global_memory = self.time_embedding_1(observations['graph_merge_global_memory'][:,:max_node_num], relative_time)

            I = torch.eye(max_node_num).unsqueeze(0).repeat(B,1,1).to(global_memory.device)
            global_mask = observations['graph_merge_global_mask'][:,:max_node_num]
            
            if self.args.use_edge_info == 'learned':
                for e in range(self.args.n_rollout_threads):
                    e = e*self.num_agents
                    for a in range(max_node_num):
                        for b in range(max_node_num):
                            if observations['graph_merge_global_A'][e][a][b] != 0:
                                a_pos = observations['merge_node_pos'][e][a]
                                b_pos = observations['merge_node_pos'][e][b]
                                agent0_pos = observations['agent_world_pos'][e][0]
                                agent1_pos = observations['agent_world_pos'][e][1]
                                for agent_id in range(self.num_agents):
                                    input_tensor = torch.tensor([a_pos, b_pos, agent0_pos, agent1_pos, \
                                    observations['agent_world_pos'][e][agent_id]-a_pos, observations['agent_world_pos'][e][agent_id]-b_pos])
                                    if self.args.cuda:
                                        input_tensor = input_tensor.cuda()
                                    in_angle = self.edge_linear(input_tensor)
                                    observations['graph_merge_global_A'][e+agent_id][a][b] = in_angle

            # if self.args.add_ghost:
            #     max_ghost_num = observations['graph_merge_ghost_mask'].sum(dim=1).max().long()
            #     ghost_link = observations['graph_merge_ghost_link'].long()
                
            #     relative_ghost_time = torch.zeros([relative_time.shape[0], max_ghost_num]).to(relative_time.device)
            #     for i in range(B):
            #         relative_ghost_time[i] = relative_time[i,ghost_link[i, :max_ghost_num]]
            #     relative_total_time = torch.cat([relative_time, relative_ghost_time], dim=1)
            #     total_feature = torch.cat([observations['graph_merge_global_memory'][:,:max_node_num], observations['graph_merge_ghost_feature'][:, :max_ghost_num]], dim=1)
            
            #     total_memory = self.time_embedding_3(total_feature, relative_total_time)
            #     total_mask = torch.cat([observations['graph_merge_global_mask'][:,:max_node_num],observations['graph_merge_ghost_mask'][:,:max_ghost_num]], dim=1)
                
            #     II = torch.eye(max_node_num+max_ghost_num).unsqueeze(0).repeat(B,1,1).to(total_memory.device)
            #     total_A = torch.zeros([B, max_ghost_num+max_node_num, max_ghost_num+max_node_num]).to(II.device)
            #     for i in range(B):
            #         total_A[i, :max_node_num, :max_node_num] = observations['graph_merge_global_A'][i,:max_node_num, :max_node_num]
            #         for j in range(max_ghost_num):
            #             if observations['graph_merge_ghost_mask'][i, j] == 0:
            #                 break
            #             else:
            #                 total_A[i, ghost_link[i,j], max_node_num+j] = 1
            #                 total_A[i, max_node_num+j, ghost_link[i,j]] = 1
            #     total_A = self.normalize_sparse_adj(total_A + II)

            #goal_embedding = observations['goal_embedding']
            #global_memory_with_goal= self.feature_embedding(torch.cat((global_memory[:,:max_node_num], goal_embedding.unsqueeze(1).repeat(1,max_node_num,1)),-1))
            global_memory_with_goal= self.feature_embedding(global_memory[:,:max_node_num])
            
            if self.use_distance:
                global_D = self.normalize_sparse_adj(observations['graph_merge_global_D'][:,:max_node_num, :max_node_num] + I)
                graph_global_context = self.graph_global_GCN(global_memory_with_goal, global_D)
            else:
                global_A = self.normalize_sparse_adj(observations['graph_merge_global_A'][:,:max_node_num, :max_node_num] + I)
                graph_global_context = self.graph_global_GCN(global_memory_with_goal, global_A)

            if self.use_agent_node:
                W_I = torch.eye(self.num_agents).unsqueeze(0).repeat(B,1,1).to(global_memory.device)
                global_W = self.normalize_sparse_adj(observations['graph_agents_weights'] + W_I)
                agent_global_context = self.agent_global_GCN(observations['graph_curr_vis_embedding'], global_W)

            # if self.args.add_ghost:
            #     total_memory_with_goal = self.feature_embedding_1(total_memory)
            #     total_context = self.total_GCN(total_memory_with_goal, total_A)
            #     final_context = torch.cat([graph_global_context, total_context], dim=1)
            #     final_mask = torch.cat([global_mask, total_mask], dim=1)
            #     final_context, _ = self.GCN_mixer(final_context, final_context, final_mask)
            #     global_context = final_context
            #     global_mask = final_mask
            #     relative_time = torch.cat([relative_time, relative_total_time], dim=1)

            #curr_embedding, goal_embedding = observations['curr_embedding'], observations['merge_goal_embedding']
            curr_embedding = observations['curr_embedding']
            graph_global_context = self.time_embedding_2(graph_global_context, relative_time)
            #goal_context, goal_attn = self.goal_Decoder(goal_embedding.unsqueeze(1), global_context, global_mask)
            graph_curr_context, graph_curr_attn = self.graph_curr_Decoder(curr_embedding.unsqueeze(1), graph_global_context,  global_mask)
            graph_curr_context = graph_curr_context.squeeze(1)
            if self.use_agent_node:
                agent_curr_context, agent_curr_attn = self.agent_curr_Decoder(curr_embedding.unsqueeze(1), agent_global_context, observations['graph_agent_mask'])
                agent_curr_context = agent_curr_context.squeeze(1)
            if self.use_tail_agent_info:
                graph_curr_context = torch.cat((graph_curr_context,observations['graph_agent_id'][:,0:1].repeat(1,self.hidden_size)),dim=1)
                graph_curr_context = self.graph_encode_actor_net(graph_curr_context)
                if self.use_agent_node:
                    agent_curr_context = torch.cat((agent_curr_context,observations['graph_agent_id'][:,0:1].repeat(1,self.hidden_size)),dim=1)
                    agent_curr_context = self.agent_encode_actor_net(agent_curr_context)
            
            all_graph_curr_context = None
            all_agent_curr_context = None
            if self.use_other_agent:
                all_graph_curr_context = [graph_curr_context,]
                for agent_id in range(0, self.num_agents-1):
                    other_embedding = observations['other_curr_embedding'][agent_id]
                    other_graph_curr_context, other_curr_attn = self.graph_curr_Decoder(other_embedding.unsqueeze(1), graph_global_context,  global_mask)
                    other_graph_curr_context = other_graph_curr_context.squeeze(1)
                    if self.use_tail_agent_info:
                        other_graph_curr_context = torch.cat((other_graph_curr_context,observations['graph_agent_id'][:,agent_id+1:agent_id+2].repeat(1,self.hidden_size)),dim=1)
                        other_graph_curr_context = self.graph_encode_other_net(other_graph_curr_context)
                    all_graph_curr_context.append(other_graph_curr_context)    
            
        if self.use_other_agent:
            all_graph_curr_context = torch.stack(all_graph_curr_context, dim = 1)
            #intra_attn
            if self.use_intra_attn:
                for i in range(self.depth):
                    attn, fc = self.agent_attn_layers[i]
                    all_graph_curr_context = attn(all_graph_curr_context) + all_graph_curr_context
                    all_graph_curr_context = fc(all_graph_curr_context) + all_graph_curr_context
            
            #cross_attn
            norm, to_kv, cross_attn, ff= self.last_cross_attn
            all_graph_curr_context = norm(all_graph_curr_context)
            x = all_graph_curr_context[:, :1, :] # 64B x 1 x D
            others = all_graph_curr_context[:, 1:, :] # 64B x (n-1) x D
            if self.num_agents > 1:
                k, v = to_kv(others).chunk(2, dim=-1)
                all_graph_curr_context = cross_attn(x, k, v) + x # # 64B x 1 x D
            else:
                all_graph_curr_context = x
            all_graph_curr_context = ff(all_graph_curr_context) + all_graph_curr_context
            all_graph_curr_context = all_graph_curr_context.squeeze(1)


        if self.use_single:
            B = observations['graph_global_mask'].shape[0]
            max_node_num = observations['graph_global_mask'].sum(dim=1).max().long()
            relative_time = observations['graph_time'] - observations['graph_global_time'][:, :max_node_num]
            global_memory = self.time_embedding_1(observations['graph_global_memory'][:,:max_node_num], relative_time)
            global_mask = observations['graph_global_mask'][:,:max_node_num]         
            I = torch.eye(max_node_num).unsqueeze(0).repeat(B,1,1).to(global_memory.device)
            global_A = self.normalize_sparse_adj(observations['graph_global_A'][:,:max_node_num, :max_node_num] + I)
            #goal_embedding = observations['goal_embedding']
            #global_memory_with_goal= self.feature_embedding(torch.cat((global_memory[:,:max_node_num], goal_embedding.unsqueeze(1).repeat(1,max_node_num,1)),-1))
            global_memory_with_goal= self.feature_embedding(global_memory[:,:max_node_num])
            
            graph_global_context = self.graph_global_GCN(global_memory_with_goal, global_A)

            #curr_embedding, goal_embedding = observations['curr_embedding'], observations['merge_goal_embedding']
            curr_embedding = observations['curr_embedding']
            graph_global_context = self.time_embedding_2(graph_global_context, relative_time)
            #goal_context, goal_attn = self.goal_Decoder(goal_embedding.unsqueeze(1), global_context, global_mask)
            
            graph_curr_context, graph_curr_attn = self.graph_curr_Decoder(curr_embedding.unsqueeze(1), graph_global_context, global_mask)
            graph_curr_context = graph_curr_context.squeeze(1)
            
        if not self.use_other_agent:
            all_graph_curr_context = graph_curr_context
            if self.use_agent_node:
                all_agent_curr_context = agent_curr_context
        
        if return_features:
            #return_f = {'goal_attn': goal_attn, 'curr_attn': curr_attn}
            return_f = {'curr_attn': curr_attn}
            #return curr_context.squeeze(1), goal_context.squeeze(1), return_f
            return (all_graph_curr_context, all_agent_curr_context, return_f)
        
        return (all_graph_curr_context, all_agent_curr_context, None)
