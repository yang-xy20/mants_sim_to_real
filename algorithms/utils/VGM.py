import torch
import torch.nn as nn
#from habitat_baselines.common.utils import CategoricalNet
import sys
sys.path.append("../..")

from mants_sim_to_real.algorithms.utils.resnet_file import resnet
from mants_sim_to_real.algorithms.utils.resnet_file.resnet import ResNetEncoder
from .perception import Perception
from .perception_graph import Perception_Graph
from mants_sim_to_real.envs.habitat.utils.extractor import VisualEncoder
from .agent_attention import AttentionModule


class VGMNet(nn.Module):
    def __init__(
            self,
            observation_space,
            action_space,
            args,
            rnn_type,
            backbone,
            resnet_baseplanes,
            normalize_visual_inputs,
            graph_linear):  
        
        super(VGMNet, self).__init__()
        self.prev_action_embedding = nn.Embedding(3 + 1, 32)
        # self.orient_embedding = nn.Embedding(72, 8)
        
        self.id_embedding = nn.Embedding(args.num_agents+ 1, 32)
        self.idx_embedding = nn.Embedding(args.graph_memory_size+ 1, 32)
        self._n_prev_action = 32

        self.num_category = 50
        self._n_input_goal = 0

        self._hidden_size = args.hidden_size
        self._feature_dim = args.feature_dim
        self._num_recurrent_layers = args.num_recurrent_layers
        self._use_id_embedding = args.use_id_embedding
        # self._use_orient_embedding = args.use_orient_embedding
        self._use_idx_embedding = args.use_idx_embedding
        self._use_num_embedding = args.use_num_embedding
        self._use_action_embedding = args.use_action_embedding
        self._use_goal_embedding = args.use_goal_embedding
        self.use_merge = args.use_merge
        self.use_single = args.use_single
        self.num_agents = args.num_agents
        self.graph_memory_size = args.graph_memory_size
        self.use_other_agent = args.use_other_agent
        self.num_local_steps = args.num_local_steps
        self.use_tail_agent_info = args.use_tail_agent_info
        self.use_agent_node = args.use_agent_node
        self.graph_linear = graph_linear
        if self._use_num_embedding:
            self.num_embedding = nn.Embedding(args.graph_memory_size+ 1, 32)
        if self._use_goal_embedding:
            self.goal_embedding = nn.Embedding(args.graph_memory_size+ 1, 32)
        self.count = 2
        if self.use_agent_node:
            self.count += 1
        if self.use_single:
            self.count += self.num_agents-1

        #rnn_input_size = self._n_input_goal + self._n_prev_action
        # print('backbone')
        # print(backbone)
        # print(resnet_baseplanes)
        self.args = args
        if args.use_retrieval:
            self.visual_encoder = VisualEncoder(args)
        else:
            self.visual_encoder = ResNetEncoder(
                observation_space,
                baseplanes=resnet_baseplanes,
                ngroups=resnet_baseplanes//2,
                make_backbone=getattr(resnet, backbone),
                normalize_visual_inputs=normalize_visual_inputs,
                output_size = self._hidden_size
            )
        if self.use_agent_node:
            self.perception_unit = Perception_Graph(args.feature_dim)
        else:
            if self.use_single:
                for i in range(self.num_agents):
                    setattr(self, 'perception_unit_' + str(i), Perception(args, None))
            else:
                self.perception_unit = Perception(args, graph_linear, None)
        
        if not self.visual_encoder.is_blind:
            self.visual_fc = nn.Sequential(
                nn.Linear(
                    self._hidden_size * self.count, self._hidden_size * 2
                ),
                nn.ReLU(True),
                nn.Linear(
                    self._hidden_size * 2, self._hidden_size
                ),
                nn.ReLU(True),
            )
            if self.use_tail_agent_info:
                self.visual_embedding = nn.Linear(args.hidden_size*2, args.hidden_size)
        
        if self.use_agent_node:
            self.attention_layer = AttentionModule(dimensions=args.hidden_size, 
                                                attention_type='general')
        # self.pred_aux1 = nn.Sequential(nn.Linear(self._feature_dim, self._feature_dim),
        #                                nn.ReLU(),
        #                                nn.Linear(self._feature_dim, 1))
        # self.pred_aux2 = nn.Sequential(nn.Linear(self._feature_dim * 2, self._feature_dim),
        #                                nn.ReLU(),
        #                                nn.Linear(self._feature_dim, 1))
        # self.state_encoder = RNNStateEncoder(
        #     (0 if self.is_blind else self._hidden_size) + rnn_input_size,
        #     self._hidden_size,
        #     rnn_type = rnn_type,
        #     num_layers = self._num_recurrent_layers,
        # )

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, masks, frontier_graph_data, agent_graph_data, return_features=False):

        if self._use_action_embedding:
            periodic_actions = []
            for i in range(self.num_local_steps):
                prev_actions = self.prev_action_embedding(
                        ((observations['graph_prev_actions'][:,i:i+1].float() + 1) * masks).long().squeeze(-1)
                    )
                periodic_actions+=[prev_actions]
            periodic_actions = torch.cat(periodic_actions,dim=1)
        if self._use_id_embedding:
            agent_id_embed = self.id_embedding(
                ((observations['graph_id_embed'].float()+ 1) * masks).long().squeeze(-1)
            )
        if self._use_goal_embedding:
                prev_goal = self.goal_embedding(
                ((observations['graph_prev_goal'].float() + 1) * masks).long().squeeze(-1)
            )
        # if self._use_orient_embedding:
        #         current_orient = self.orient_embedding(
        #         ((observations['graph_current_orient'].float()) * masks).long().squeeze(-1)
        #     )
        if self.use_merge:
            if self._use_idx_embedding:
                prev_idx = self.idx_embedding(
                ((observations['graph_merge_localized_idx'].float() + 1) * masks).long().squeeze(-1)
            )
                
            if self._use_num_embedding:
                num_idx = self.num_embedding(
                ((observations['merge_graph_num_node'].float() + 1) * masks).long().squeeze(-1)
            )
        if self.use_single:
            total_context = []
        
            if self._use_idx_embedding:
                prev_idx = self.idx_embedding(
                ((observations['graph_localized_idx'].float() + 1) * masks).long().squeeze(-1)
            )
                
            if self._use_num_embedding:
                num_idx = self.num_embedding(
                ((observations['graph_num_node'].float() + 1) * masks).long().squeeze(-1)
            )
        
        input_list = [observations['graph_panoramic_rgb'][:,0].permute(0, 3, 1, 2) / 255.0,
                    observations['graph_panoramic_depth'][:,0].permute(0, 3, 1, 2)]
    
        curr_tensor = torch.cat(input_list, 1)

        if self.args.use_retrieval:
            observations['curr_visual_embedding'] = self.visual_encoder.extract(curr_tensor).view(curr_tensor.shape[0], -1)
        else:
            observations['curr_visual_embedding'] = self.visual_encoder(curr_tensor).view(curr_tensor.shape[0], -1)
        if self.use_tail_agent_info:
            observations['curr_embedding'] = observations['curr_visual_embedding']
            observations['new_curr_embedding'] = self.visual_embedding(torch.cat((observations['curr_visual_embedding'],observations['graph_agent_id'][:,0:1].repeat(1,512)), dim=1))#,observations['graph_merge_localized_idx']/self.graph_memory_size),dim=1))
        else:
            observations['curr_embedding'] = observations['curr_visual_embedding']
            observations['new_curr_embedding'] = observations['curr_visual_embedding']

        if self.use_other_agent:
            observations['other_curr_embedding'] = []
            for agent_id in range(1,self.num_agents):
                other_input_list = [observations['graph_panoramic_rgb'][:,agent_id].permute(0, 3, 1, 2) / 255.0,
                        observations['graph_panoramic_depth'][:,agent_id].permute(0, 3, 1, 2)]
                other_curr_tensor = torch.cat(other_input_list, 1)
                if self.args.use_retrieval:
                    observations['other_curr_visual_embedding'] = self.visual_encoder.extract(other_curr_tensor).view(other_curr_tensor.shape[0], -1)
                else:
                    observations['other_curr_visual_embedding'] = self.visual_encoder(other_curr_tensor).view(other_curr_tensor.shape[0], -1)
                observations['other_curr_embedding'].append(observations['other_curr_visual_embedding'])
            observations['other_curr_embedding'] = torch.stack(observations['other_curr_embedding'], dim = 0)
        # observations['curr_other_embedding'] = self.visual_encoder(curr_other_tensor).view(curr_other_tensor.shape[0], -1)
        #goal_tensor = observations['target_goal'].permute(0, 3, 1, 2)
        #observations['goal_embedding'] = self.visual_encoder(goal_tensor).view(goal_tensor.shape[0], -1)
        # if self.use_agent_node:
        #         observations['graph_agents_weights'] = self.attention_layer(observations['graph_curr_vis_embedding'])
        (curr_graph_context, curr_agent_context, _ )= self.perception_unit(observations, return_features=return_features)
            #contexts = torch.cat((curr_context, goal_context), -1)
        if self.use_merge:  
            if self.use_agent_node:
                feats = self.visual_fc(torch.cat((curr_graph_context, curr_agent_context, observations['curr_embedding']), 1))
            else:
                feats = self.visual_fc(torch.cat((curr_graph_context, observations['curr_embedding']), 1))
        
        if self.use_single:
            total_context = []          
            for i in range(self.num_agents):
                new_observations = {}
                for key in observations.keys():
                    if key in ['graph_global_memory', 'graph_global_mask', 'graph_global_A', 'graph_global_time', 'graph_localized_idx', 'graph_id_trace']:
                        split_shape = observations[key].shape[1]//self.num_agents
                        new_observations[key] = observations[key][:,split_shape*i:split_shape*i+split_shape]
                    else:
                        new_observations[key] = observations[key]
                exec("total_context.append(self.perception_unit_{}(new_observations, return_features=return_features))".format(i))
            
            curr_context = [total_context[i][0] for i in range(len(total_context))]
            curr_context = torch.cat(curr_context, dim=1)
            
            feats = self.visual_fc(torch.cat((curr_context, observations['new_curr_embedding']), 1))
            
        x = [feats]
        
        if self._use_action_embedding:
            x += [periodic_actions]
        if self._use_id_embedding:
            x += [agent_id_embed]
        # if self._use_orient_embedding:
        #     x += [current_orient]
        if self._use_idx_embedding:
            x += [prev_idx]
        if self._use_num_embedding:
            x += [num_idx]
        if self._use_goal_embedding:
            x += [prev_goal]

        x = torch.cat(x, dim=1)
        #x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)
        #pred1 = self.pred_aux1(curr_context)
        #pred2 = self.pred_aux2(contexts)
        return x#, (pred1, pred2)
