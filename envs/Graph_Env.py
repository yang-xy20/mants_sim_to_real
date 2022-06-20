from copy import deepcopy
from joblib import register_parallel_backend
import numpy as np
import math
import gym
from .graph import Graph
import torch.nn as nn
import torch
import os
from .Env import MultiHabitatEnv
from torchvision import transforms
from PIL import Image
import sys
sys.path.append("../..")

from mants_sim_to_real.utils import pose as pu
import random
import time
from collections import deque
from mants_sim_to_real.utils.fmm_planner import FMMPlanner
class GraphHabitatEnv(MultiHabitatEnv):
    def __init__(self, args, run_dir):
        self.num_agents = args.num_agents 
        self.graph_memory_size  = args.graph_memory_size
        for agent_id in range(self.num_agents):
            setattr(self, 'graph_agent_' + str(agent_id), Graph(args,self.graph_memory_size, self.num_agents))
        self.add_ghost = args.add_ghost
        self.map_resolution = args.map_resolution
        self.use_merge = args.use_merge
        self.num_local_steps = args.num_local_steps
        self.ghost_node_size = args.ghost_node_size
        self.reset_all_memory(self.num_agents)        
        self.use_mgnn = args.use_mgnn
        self.episode_length = args.episode_length
        self.use_double_matching = args.use_double_matching
        self.dis_gap = args.dis_gap
        self.build_graph = args.build_graph
        
        
        self.dt = 10
        super().__init__(args, True, run_dir)
   

    def update_merge_graph(self, pos):
        ghost_pos = self.compute_ghost_pos(pos)
     
        self.localize( pos, ghost_pos, [False for _ in range(self.num_agents)], init_pos = True)
        
        self.agent_last_pos = pos
        self.add_node_flag = np.array([True for _ in range(self.num_agents)])
        global_memory_dict = self.get_global_memory()
        infos = self.update_infos(global_memory_dict)
        return infos

    def update_merge_step_graph(self, infos, pos, max_size, left_corner = None, ratio=None, obstacle_map=None,explored_map_no_obs=None, explored_map=None):
        ghost_pos = self.compute_ghost_pos(pos)
        self.localize(pos, ghost_pos,[False for _ in range(self.num_agents)])
        
        if self.all_args.cut_ghost:
           
            if np.any(infos['add_node']):
                add_node = infos['add_node']                
                self.localize(pos,ghost_pos,[False for _ in range(self.num_agents)], add_node=add_node)
            else:
                add_node= [False for _ in range(self.num_agents)]
            
            if infos['update'] is True:
                self.agent_last_pos = pos
               
                for agent_id in range(self.num_agents):
                    exec('self.graph_agent_{}.check_ghost_outside(max_size,left_corner[agent_id], obstacle_map[agent_id], explored_map_no_obs[agent_id], explored_map[agent_id], ratio, self.map_resolution)'.format(agent_id))
                    exec('self.graph_agent_{}.ghost_check()'.format(agent_id))
                
                self.add_node_flag = np.array([True for _ in range(self.num_agents)])
               
                
                for agent_id in range(self.num_agents):
                    if not add_node[agent_id]:
                        if self.graph_agent_0.check_around(pos[agent_id]):
                            add_node[agent_id] = True
                if np.any(add_node):
                    self.localize( pos,  ghost_pos,[False for _ in range(self.num_agents)], add_node=add_node)
                    for agent_id in range(self.num_agents):
                        exec('self.graph_agent_{}.check_ghost_outside(max_size,left_corner[agent_id], obstacle_map[agent_id], explored_map_no_obs[agent_id], explored_map[agent_id], ratio, self.map_resolution)'.format(agent_id))
                        exec('self.graph_agent_{}.ghost_check()'.format(agent_id))
                
                for agent_id in range(self.num_agents):
                    for b in range(self.num_agents):
                        if b!=agent_id:
                            exec('self.graph_agent_{}.merge_ghost_nodes(self.graph_agent_{}.ghost_mask)'.format(agent_id,b))
              
                while self.graph_agent_0.ghost_mask.sum() < 2:
                    add_node = np.array([True for _ in range(self.num_agents)])
                    self.localize( pos, ghost_pos,[False for _ in range(self.num_agents)], add_node=add_node)
                
        global_memory_dict = self.get_global_memory()
        infos = self.update_infos(global_memory_dict)
    
        return infos, 0
    
    def compute_ghost_pos(self, pos):
        world_ghost_loc = [[] for _ in range(self.num_agents)]
        angles = [-180+(360/self.ghost_node_size)*i for i in range(self.ghost_node_size)]
        for agent_id in range(self.num_agents):
            for i in range(len(angles)):
                world_ghost_loc[agent_id].append(pu.get_new_pose_from_dis(pos[agent_id], self.dis_gap, angles[i]))
        return world_ghost_loc

    def reset_all_memory(self, num_agents=None):
        
        for agent_id in range(self.num_agents):
            exec('self.graph_agent_{}.reset()'.format(agent_id))
        
    # assume memory index == node index
    def localize(self, world_position, world_ghost_position, done_list, add_node=False, init_pos = False):
        # The position is only used for visualizations.
        

        if np.any(add_node): 
            for agent_id in range(self.num_agents):
                for b in range(self.num_agents):
                    if add_node[b]:
                        new_node_idx = []
                        exec('new_node_idx.append(self.graph_agent_{}.num_node())'.format(agent_id))
                        exec('self.graph_agent_{}.add_node(new_node_idx[-1], b, world_position[b])'.format(agent_id))                   
                        exec('self.graph_agent_{}.add_edge(new_node_idx[-1], self.graph_agent_{}.last_localized_node_idx[b])'.format(agent_id,agent_id))
                        exec('self.graph_agent_{}.record_localized_state(new_node_idx[-1], b)'.format(agent_id))
                        if self.add_ghost:
                            exec('self.graph_agent_{}.add_ghost_node(b, self.ghost_node_size, world_ghost_position[b])'.format(agent_id))
        elif init_pos:
            for agent_id in range(self.num_agents):
                exec('self.graph_agent_{}.reset_at()'.format(agent_id))
                for b in range(self.num_agents):
                    if b != 0:
                        new_node_idx = []
                        exec('new_node_idx.append(self.graph_agent_{}.num_node())'.format(agent_id))
                        exec('self.graph_agent_{}.add_node(new_node_idx[-1], b, world_position[b])'.format(agent_id))
                        #exec('self.graph_agent_{}.add_edge(new_node_idx[-1], self.graph_agent_{}.last_localized_node_idx[b])'.format(agent_id,agent_id))
                        exec('self.graph_agent_{}.record_localized_state(new_node_idx[-1], b)'.format(agent_id))
                    else:
                        exec('self.graph_agent_{}.initialize_graph(b, world_position[b])'.format(agent_id))                  
                    if self.add_ghost:
                        exec('self.graph_agent_{}.add_ghost_node(b, self.ghost_node_size, world_ghost_position[b])'.format(agent_id))
            exec('self.graph_agent_{}.num_init_nodes = len(self.graph_agent_{}.node_position_list)'.format(agent_id,agent_id))

    def update_infos(self, global_memory_dict):
        # add memory to obs
        infos = {}
        for key in self.g_obs_space.keys():
            if key in ['agent_graph_node_dis','graph_agent_dis', 'graph_ghost_valid_mask',\
            'graph_prev_goal','graph_agent_id','agent_world_pos','graph_agent_mask','global_merge_goal','global_merge_obs',\
            'graph_last_ghost_node_position','graph_last_agent_world_pos','last_graph_dis','graph_last_agent_dis',\
            'graph_last_node_position','graph_last_pos_mask']:
                pass
            elif key == 'merge_node_pos':
                infos[key] = np.expand_dims(global_memory_dict[key], axis=0).repeat(self.num_agents, 0)
            
            else:
                infos[key] = global_memory_dict[key]
        return infos
    
    def get_global_memory(self, mode='feature'):
        global_memory_dict = { }
        global_memory_dict['graph_ghost_node_position'] = self.graph_agent_0.ghost_node_position
        if self.use_merge:
            global_memory_dict['graph_merge_ghost_mask'] = self.graph_agent_0.ghost_mask
            if self.use_double_matching:
                global_memory_dict['graph_node_pos'] = self.graph_agent_0.node_position_list
        return global_memory_dict
    
    
    def build_graph_global_obs(self):
        self.g_obs_space = {}
        if self.use_merge:
            if self.use_double_matching:
                self.g_obs_space['graph_last_node_position'] = gym.spaces.Box( low=-np.Inf, high=np.Inf, shape=(self.episode_length*self.num_agents, 4), dtype = np.float32)
                self.g_obs_space['agent_graph_node_dis'] = gym.spaces.Box(
                                low=-np.Inf, high=np.Inf, shape=(self.num_agents, self.graph_memory_size, 1), dtype=np.float32)
                self.g_obs_space['graph_node_pos'] = gym.spaces.Box(
                                low=-np.Inf, high=np.Inf, shape=(self.graph_memory_size, 4), dtype=np.float32)
          
            self.g_obs_space['graph_ghost_node_position'] = gym.spaces.Box(
                            low=-np.Inf, high=np.Inf, shape=(self.graph_memory_size, self.ghost_node_size, 4), dtype=np.float32)
            self.g_obs_space['agent_world_pos'] = gym.spaces.Box(
                                low=-np.Inf, high=np.Inf, shape=(self.num_agents,4), dtype=np.int32)
            self.g_obs_space['graph_last_ghost_node_position'] = gym.spaces.Box( low=-np.Inf, high=np.Inf, shape=(self.episode_length*self.num_agents, 4), dtype = np.float32)
            self.g_obs_space['graph_last_agent_world_pos'] = gym.spaces.Box( low=-np.Inf, high=np.Inf, shape=(self.episode_length*self.num_agents, 4), dtype = np.float32)
            if self.use_mgnn:
                
                self.g_obs_space['graph_last_pos_mask'] = gym.spaces.Box(
                            low=0, high=np.Inf, shape=(self.episode_length*self.num_agents, 1), dtype=np.int32)
                
                self.g_obs_space['graph_agent_dis'] = gym.spaces.Box(
                                low=-np.Inf, high=np.Inf, shape=(self.num_agents, self.graph_memory_size*self.ghost_node_size, 1), dtype=np.float32)
          
            if self.add_ghost:
                
                self.g_obs_space['graph_merge_ghost_mask'] = gym.spaces.Box(
                            low=0, high=np.Inf, shape=(self.graph_memory_size,self.ghost_node_size), dtype=np.int32)
            
        return self.g_obs_space
    
    def node_max_num(self):
        num_node = self.graph_agent_0.num_node()
        return num_node
    
    def get_valid_num(self, inp):
        if inp is None:
            return len(np.unique(self.graph.ghost_node_link))
        else:
            return [np.unique(self.graph.ghost_node_link)[math.floor(inp[i])] for i in range(len(inp))]

    def get_valid_index(self):
        return self.graph.ghost_mask


    def get_goal_position(self, global_goal):
        self.valid_ghost_position = np.zeros((1000,2), np.float)
    
        goal = np.zeros((self.num_agents, 1, 2), np.int32)
        for agent_id in range(self.num_agents):
            for i in range(goal.shape[1]):
                (goal_x, goal_y) =self.graph_agent_0.get_ghost_positions(int(global_goal[agent_id][i]))     
                goal[agent_id, i] = [int(goal_x * 100.0/self.map_resolution),
                        int(goal_y * 100.0/self.map_resolution)]
        valid_ghost_position = self.graph_agent_0.get_all_ghost_positions()
        self.valid_ghost_position[:valid_ghost_position.shape[0]] = valid_ghost_position
        
        return goal, self.valid_ghost_position

    def get_runner_fmm_distance(self, positions, max_size):
        obstacle_map = positions['obstacle_map']
        merge_obstacle_map= np.zeros((max_size[0],max_size[1]))
        start_origin = positions['x'].reshape(-1,2)
        final_origin = positions['y'].reshape(-1,2)
        corner = positions['corner']
        final_mask = False
        obstacle_map_shape = obstacle_map.shape
        merge_obstacle_map[corner[0]:corner[0]+obstacle_map_shape[0],corner[1]:corner[1]+obstacle_map_shape[1]] = obstacle_map
        
        merge_ghost_mask = np.zeros_like(self.graph_agent_0.ghost_mask.reshape(-1))
        if final_origin.shape[0] != self.num_agents:
            final_mask = True
            if final_origin.shape[0] == self.graph_memory_size*self.ghost_node_size:
                merge_ghost_mask = self.graph_agent_0.ghost_mask.reshape(-1)
            if final_origin.shape[0] == self.graph_memory_size:
                merge_ghost_mask = self.graph_agent_0.ghost_mask.sum(axis = 1)
        distance = np.zeros((start_origin.shape[0], final_origin.shape[0], 1))
        for i in range(start_origin.shape[0]):
            r, c = start_origin[i][0], start_origin[i][1]
            start = [int(r * 100.0/self.map_resolution),
                        int(c * 100.0/self.map_resolution)]
        
           
            traversible = np.rint(merge_obstacle_map)!= True 
       
            all_goal = []
            all_index = []
            for j in range(final_origin.shape[0]): 
                if final_mask and merge_ghost_mask[j] == 0:
                    continue
                else:
                    r, c = final_origin[j][0], final_origin[j][1]
                    goal = [int(r * 100.0/self.map_resolution),
                                int(c * 100.0/self.map_resolution)]
                    all_goal.append(goal)
                    all_index.append(j)
            planner = FMMPlanner(traversible, 360//self.dt, use_distance=True)
            reachable = planner.set_goal([start[1], start[0]])
         
            
            for h in range(len(all_goal)):
                r = np.rint(merge_obstacle_map).sum(axis =0).max() 
                l = np.rint(merge_obstacle_map).sum(axis =1).max() 
                rl = r if r > l else l
                if reachable.max() < rl:
                    distance[i, all_index[h]] = -1
                else:
                
                    if reachable[all_goal[h][0], all_goal[h][1]] == reachable.max():
                        distance[i, all_index[h]] = -1
                    else:
                        distance[i, all_index[h]] = reachable[all_goal[h][0], all_goal[h][1]]/300 #reachable.max()
        return distance
        