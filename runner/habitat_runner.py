import time
import os
import gym
import numpy as np
import imageio
import math
from collections import defaultdict, deque
from itertools import chain
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
from torch.nn import functional as F
from mants_sim_to_real.utils import pose as pu
import copy
import json

from .base_runner import Runner
#from mants_sim_to_real.envs.habitat.utils.frontier import get_closest_frontier, get_frontier, nearest_frontier, max_utility_frontier, bfs_distance, rrt_global_plan, l2distance, voronoi_based_planning
from mants_sim_to_real.algorithms.utils.util import init, check
import joblib
from mants_sim_to_real.utils import visualization as vu

def _t2n(x):
    return x.detach().cpu().numpy()

def get_folders(dir, folders):
    get_dir = os.listdir(dir)
    for i in get_dir:          
        sub_dir = os.path.join(dir, i)
        if os.path.isdir(sub_dir): 
            folders.append(sub_dir) 
            get_folders(sub_dir, folders)

class HabitatRunner(Runner):
    def __init__(self, config):
        super(HabitatRunner, self).__init__(config)
        # init parameters
        self.init_hyper_parameters()
        # init map variables
        self.init_map_variables()
        # global policy
        self.init_global_policy() 


    def init_hyper_parameters(self):
        
        self.map_resolution = self.all_args.map_resolution
        self.proj_frontier = self.all_args.proj_frontier

        self.num_local_steps = self.all_args.num_local_steps
        self.grid_pos = self.all_args.grid_pos
        self.agent_invariant = self.all_args.agent_invariant
        self.grid_goal = self.all_args.grid_goal
        self.use_goal = self.all_args.use_goal
        self.use_local_single_map = self.all_args.use_local_single_map
        self.grid_last_goal = self.all_args.grid_last_goal
        self.grid_size = self.all_args.grid_size
        self.use_single = self.all_args.use_single
        self.figure_m, self.ax_m = plt.subplots(1, 1, figsize=(6,6),facecolor="white",num="Scene {} Merge Map".format(0))
        self.time_step = 0
        
    def init_map_variables(self):
        # each agent rotation
        self.full_w, self.full_h = 240,240

        # ft
        self.ft_merge_map = np.zeros((self.n_rollout_threads, 2, self.full_w, self.full_h), dtype = np.float32) # only explored and obstacle
        self.ft_goals = np.zeros((self.n_rollout_threads, self.num_agents, 2), dtype = np.int32)
        self.ft_pre_goals = np.zeros((self.n_rollout_threads, self.num_agents, 2), dtype = np.int32)
        self.ft_last_merge_explored_ratio = np.zeros((self.n_rollout_threads, 1), dtype= np.float32)
        self.ft_mask = np.ones((self.full_w, self.full_h), dtype=np.int32)
        self.ft_go_steps = np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype= np.int32)
        self.ft_map = [None for _ in range(self.n_rollout_threads)]
        self.ft_lx = [None for _ in range(self.n_rollout_threads)]
        self.ft_ly = [None for _ in range(self.n_rollout_threads)]
               
    
    
    def init_global_policy(self):
        self.global_input = {}
        
        if self.grid_pos:
            self.global_input['grid_pos'] = np.zeros((self.n_rollout_threads, self.num_agents, self.num_agents*2, self.grid_size, self.grid_size), dtype=np.int32)
        
        if self.grid_last_goal:
            self.global_input['grid_goal'] = np.zeros((self.n_rollout_threads, self.num_agents, self.num_agents*2, self.grid_size, self.grid_size), dtype=np.int32)

        
        if self.use_single:
            self.global_input['global_obs'] = np.zeros((self.n_rollout_threads, self.num_agents, 4, self.full_w, self.full_h), dtype=np.float32)
            if self.use_goal:
                self.global_input['global_goal'] = np.zeros((self.n_rollout_threads, self.num_agents, 2, self.full_w, self.full_h), dtype=np.float32)
       

        self.share_global_input = self.global_input.copy()
        
       
        self.global_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32) 
        self.global_goal = np.zeros((self.n_rollout_threads, self.num_agents, 2), dtype=np.float32)
        self.revise_global_goal = np.zeros((self.n_rollout_threads, self.num_agents, 2), dtype=np.int32)
        
        self.first_compute = True
       
    
    def first_compute_global_input(self):
        global_goal_map = np.zeros((self.num_agents, self.full_w, self.full_h), dtype=np.float32)
        global_pos_map = np.zeros((self.num_agents, self.full_w, self.full_h), dtype=np.float32)
        self.all_global_goal_map = np.zeros((self.num_agents, self.full_w, self.full_h), dtype=np.float32)
        grid_pos = np.zeros((self.n_rollout_threads, self.num_agents*2, self.grid_size), dtype=np.int32)
        
        grid_goal = np.zeros((self.n_rollout_threads, self.num_agents*2, self.grid_size), dtype=np.int32)
        grid_locs = np.zeros((self.n_rollout_threads, self.num_agents,2))
        grid_goal_pos = np.zeros((self.n_rollout_threads, self.num_agents,2))
        for a in range(self.num_agents): 
            global_goal_map[ a, int(self.global_goal[0,a, 0]-2):int(self.global_goal[0,a, 0]+3) ,int(self.global_goal[0,a, 1]-2):int(self.global_goal[0,a, 1]+3) ] =  1
            global_pos_map[ a,int(self.pos[a, 0]-2):int(self.pos[a, 0]+3),int(self.pos[a,1]-2):int(self.pos[a,1]+3)] =  1
            grid_locs[0, a, 0] = int(self.pos[a,0]/(self.map_size[0]/self.grid_size))
            grid_locs[0, a, 1] = int(self.pos[a,1]/(self.map_size[1]/self.grid_size))
            grid_goal_pos[0, a, 0] = int(self.global_goal[0,a,0]/(self.map_size[0]/self.grid_size))
            grid_goal_pos[0, a, 1] = int(self.global_goal[0,a,1]/(self.map_size[1]/self.grid_size))

        if self.grid_pos: 
            for i in range(self.grid_size):
                grid_pos[:, 0:self.num_agents, i] = i - grid_locs[:, :, 0]
                grid_pos[:, self.num_agents:2*self.num_agents+1, i] = i - grid_locs[:, :,1]
            x_pos = np.expand_dims(grid_pos[:, 0:self.num_agents], 1).repeat(self.num_agents, axis=1)
            
            self.global_input['grid_pos'][ :, :, 0:2*self.num_agents:2] = np.expand_dims(x_pos, -1).repeat(self.grid_size, axis=-1)
                
            y_pos = np.expand_dims(grid_pos[:, self.num_agents:2*self.num_agents+1], 1).repeat(self.num_agents, axis=1)
             
            self.global_input['grid_pos'][:, :, 1:2*self.num_agents:2] = np.expand_dims(y_pos, -2).repeat(self.grid_size, axis=-2)
            self.global_input['grid_pos'] += self.grid_size - 1 #(0-14)
            if self.agent_invariant:
                for a in range(1, self.num_agents):
                    self.global_input['grid_pos'][:, a] = np.roll(self.global_input['grid_pos'][:, a], -2*a, axis=1)
        
        if self.grid_last_goal: 
            for i in range(self.grid_size):
                grid_goal[:, 0:self.num_agents, i] = i - grid_goal_pos[:, :, 0]
                grid_goal[:, self.num_agents:2*self.num_agents+1, i] = i - grid_goal_pos[:, :,1]
            x_goal = np.expand_dims(grid_goal[:, 0:self.num_agents], 1).repeat(self.num_agents, axis=1)
           
            self.global_input['grid_goal'][ :, :, 0:2*self.num_agents:2] = np.expand_dims(x_goal, -1).repeat(self.grid_size, axis=-1)
                
            y_goal = np.expand_dims(grid_goal[:, self.num_agents:2*self.num_agents+1], 1).repeat(self.num_agents, axis=1)
              
            self.global_input['grid_goal'][:, :, 1:2*self.num_agents:2] = np.expand_dims(y_goal, -2).repeat(self.grid_size, axis=-2)
            self.global_input['grid_goal'] += self.grid_size - 1 #(0-14)
            if self.agent_invariant:
                for a in range(1, self.num_agents):
                    self.global_input['grid_goal'][:, a] = np.roll(self.global_input['grid_goal'][:, a], -2*a, axis=1)
        
        for a in range(self.num_agents):# TODO @CHAO  
            for e in range(self.n_rollout_threads):
                self.global_input['global_obs'][e, a, 0] = cv2.resize(self.all_obstacle_map[a], (self.full_h, self.full_w))
                self.global_input['global_obs'][e, a, 1] = cv2.resize(self.all_obstacle_map[a], (self.full_h, self.full_w))
                self.global_input['global_obs'][e, a, 2] = cv2.resize(global_pos_map[a], (self.full_h, self.full_w))
                self.global_input['global_obs'][e, a, 3] = cv2.resize(self.global_trace_map[a], (self.full_h, self.full_w))                   
                self.global_input['global_goal'][e, a, 0] = cv2.resize(global_goal_map[a], (self.full_h, self.full_w))       
                self.global_input['global_goal'][e, a, 1] = cv2.resize(self.all_global_goal_map[a], (self.full_h, self.full_w))   
              
        all_global_cnn_input = [[] for _ in range(self.num_agents)]
        for agent_id in range(self.num_agents):
            for key in self.global_input.keys():
                if key not in ['stack_obs','grid_pos','grid_goal']:
                    all_global_cnn_input[agent_id].append(self.global_input[key][:, agent_id])
            all_global_cnn_input[agent_id] = np.concatenate(all_global_cnn_input[agent_id], axis=1) #[e,n,...]
        
        all_global_cnn_input = np.stack(all_global_cnn_input, axis=1)
        
        self.global_input['stack_obs'] = np.zeros((self.n_rollout_threads, self.num_agents, all_global_cnn_input.shape[2] * self.num_agents, self.full_w, self.full_h), dtype=np.float32)
        self.share_global_input['stack_obs'] = np.zeros((self.n_rollout_threads, self.num_agents, all_global_cnn_input.shape[2] * self.num_agents, self.full_w, self.full_h), dtype=np.float32)
        for agent_id in range(self.num_agents):
            self.global_input['stack_obs'][:, agent_id] = all_global_cnn_input.reshape(self.n_rollout_threads, -1, *all_global_cnn_input.shape[3:]).copy()
        
        for a in range(1, self.num_agents):
            self.global_input['stack_obs'][:, a] = np.roll(self.global_input['stack_obs'][:, a], -all_global_cnn_input.shape[2]*a, axis=1)

        for key in self.global_input.keys():
            self.share_global_input[key] = self.global_input[key].copy()
        
        self.first_compute = False
    
    def compute_global_input(self):
        
        global_goal_map = np.zeros((self.num_agents, self.full_w, self.full_h), dtype=np.float32)
        global_pos_map = np.zeros((self.num_agents, self.full_w, self.full_h), dtype=np.float32)
        grid_pos = np.zeros((self.n_rollout_threads, self.num_agents*2, self.grid_size), dtype=np.int32)
        
        grid_goal = np.zeros((self.n_rollout_threads, self.num_agents*2, self.grid_size), dtype=np.int32)
        grid_locs = np.zeros((self.n_rollout_threads, self.num_agents,2))
        grid_goal_pos = np.zeros((self.n_rollout_threads, self.num_agents,2))
        for a in range(self.num_agents): 
            global_goal_map[ a, int(self.global_goal[0,a, 0]-2):int(self.global_goal[0,a, 0]+3) ,int(self.global_goal[0,a, 1]-2):int(self.global_goal[0,a, 1]+3 )] =  1
            global_pos_map[ a,int(self.pos[a, 0]-2):int(self.pos[a, 0]+3),int(self.pos[a,1]-2):int(self.pos[a,1]+3)] =  1
            grid_locs[0, a, 0] = int(self.pos[a,0]/(self.map_size[0]/self.grid_size))
            grid_locs[0, a, 1] = int(self.pos[a,1]/(self.map_size[1]/self.grid_size))
            grid_goal_pos[0, a, 0] = int(self.global_goal[0,a,0]/(self.map_size[0]/self.grid_size))
            grid_goal_pos[0, a, 1] = int(self.global_goal[0,a,1]/(self.map_size[1]/self.grid_size))

        if self.grid_pos: 
            for i in range(self.grid_size):
                grid_pos[:, 0:self.num_agents, i] = i - grid_locs[:, :, 0]
                grid_pos[:, self.num_agents:2*self.num_agents+1, i] = i - grid_locs[:, :,1]
            x_pos = np.expand_dims(grid_pos[:, 0:self.num_agents], 1).repeat(self.num_agents, axis=1)
            
            self.global_input['grid_pos'][ :, :, 0:2*self.num_agents:2] = np.expand_dims(x_pos, -1).repeat(self.grid_size, axis=-1)
                
            y_pos = np.expand_dims(grid_pos[:, self.num_agents:2*self.num_agents+1], 1).repeat(self.num_agents, axis=1)
             
            self.global_input['grid_pos'][:, :, 1:2*self.num_agents:2] = np.expand_dims(y_pos, -2).repeat(self.grid_size, axis=-2)
            self.global_input['grid_pos'] += self.grid_size - 1 #(0-14)
            if self.agent_invariant:
                for a in range(1, self.num_agents):
                    self.global_input['grid_pos'][:, a] = np.roll(self.global_input['grid_pos'][:, a], -2*a, axis=1)
        
        if self.grid_last_goal: 
            for i in range(self.grid_size):
                grid_goal[:, 0:self.num_agents, i] = i - grid_goal_pos[:, :, 0]
                grid_goal[:, self.num_agents:2*self.num_agents+1, i] = i - grid_goal_pos[:, :,1]
            x_goal = np.expand_dims(grid_goal[:, 0:self.num_agents], 1).repeat(self.num_agents, axis=1)
           
            self.global_input['grid_goal'][ :, :, 0:2*self.num_agents:2] = np.expand_dims(x_goal, -1).repeat(self.grid_size, axis=-1)
                
            y_goal = np.expand_dims(grid_goal[:, self.num_agents:2*self.num_agents+1], 1).repeat(self.num_agents, axis=1)
              
            self.global_input['grid_goal'][:, :, 1:2*self.num_agents:2] = np.expand_dims(y_goal, -2).repeat(self.grid_size, axis=-2)
            self.global_input['grid_goal'] += self.grid_size - 1 #(0-14)
            if self.agent_invariant:
                for a in range(1, self.num_agents):
                    self.global_input['grid_goal'][:, a] = np.roll(self.global_input['grid_goal'][:, a], -2*a, axis=1)
        
        for a in range(self.num_agents):# TODO @CHAO  
            for e in range(self.n_rollout_threads):
                self.global_input['global_obs'][e, a, 0] = cv2.resize(self.all_obstacle_map[a], (self.full_h, self.full_w))
                self.global_input['global_obs'][e, a, 1] = cv2.resize(self.all_obstacle_map[a], (self.full_h, self.full_w))
                self.global_input['global_obs'][e, a, 2] = cv2.resize(global_pos_map[a], (self.full_h, self.full_w))
                self.global_input['global_obs'][e, a, 3] = cv2.resize(self.global_trace_map[a], (self.full_h, self.full_w))                   
                self.global_input['global_goal'][e, a, 0] = cv2.resize(global_goal_map[a], (self.full_h, self.full_w))       
                self.global_input['global_goal'][e, a, 1] = cv2.resize(self.all_global_goal_map[a], (self.full_h, self.full_w))   
              
        all_global_cnn_input = [[] for _ in range(self.num_agents)]
        for agent_id in range(self.num_agents):
            for key in self.global_input.keys():
                if key not in ['stack_obs','grid_pos','grid_goal']:
                    all_global_cnn_input[agent_id].append(self.global_input[key][:, agent_id])
            all_global_cnn_input[agent_id] = np.concatenate(all_global_cnn_input[agent_id], axis=1) #[e,n,...]
        
        all_global_cnn_input = np.stack(all_global_cnn_input, axis=1)
        
        self.global_input['stack_obs'] = np.zeros((self.n_rollout_threads, self.num_agents, all_global_cnn_input.shape[2] * self.num_agents, self.full_w, self.full_h), dtype=np.float32)
        self.share_global_input['stack_obs'] = np.zeros((self.n_rollout_threads, self.num_agents, all_global_cnn_input.shape[2] * self.num_agents, self.full_w, self.full_h), dtype=np.float32)
        for agent_id in range(self.num_agents):
            self.global_input['stack_obs'][:, agent_id] = all_global_cnn_input.reshape(self.n_rollout_threads, -1, *all_global_cnn_input.shape[3:]).copy()
        
        for a in range(1, self.num_agents):
            self.global_input['stack_obs'][:, a] = np.roll(self.global_input['stack_obs'][:, a], -all_global_cnn_input.shape[2]*a, axis=1)

        for key in self.global_input.keys():
            self.share_global_input[key] = self.global_input[key].copy()
    
    def goal_to_frontier(self):
        if self.use_gt_map:
            merge_map =  self.transform(self.gt_full_map, self.trans, self.rotation, self.agent_trans, self.agent_rotation, -1)
        else:
            merge_map =  self.transform(self.full_map, self.trans, self.rotation, self.agent_trans, self.agent_rotation, -1)
        for e in range(self.n_rollout_threads):
            for agent_id in range(self.num_agents):
                goals = np.zeros((2), dtype = np.int32)
                if self.use_local:
                    goals[0] = self.global_goal[e, agent_id, 0]*(self.local_map_w)+self.local_lmb[e, agent_id, 0]+self.center_w-math.ceil(self.sim_map_size[e][agent_id][0]/2)-5  
                    goals[1] = self.global_goal[e, agent_id, 1]*(self.local_map_h)+self.local_lmb[e, agent_id, 2]+self.center_h-math.ceil(self.sim_map_size[e][agent_id][1]/2)-5
                else:  
                    goals[0] = self.global_goal[e, agent_id, 0]*(self.sim_map_size[e][agent_id][0]+10)+self.center_w-math.ceil(self.sim_map_size[e][agent_id][0]/2)-5  
                    goals[1] = self.global_goal[e, agent_id, 1]*(self.sim_map_size[e][agent_id][1]+10)+self.center_h-math.ceil(self.sim_map_size[e][agent_id][1]/2)-5
                if self.direction_goal:
                    goals[0] = int(self.global_goal[e, agent_id, 0] * self.full_w)
                    goals[1] = int(self.global_goal[e, agent_id, 1] * self.full_h)
                
                goals = get_closest_frontier(merge_map[e], self.world_locs[e, agent_id], goals)

                if self.use_local:
                    self.global_goal[e, agent_id, 0] = (goals[0] - (self.local_lmb[e, agent_id, 0]+self.center_w-math.ceil(self.sim_map_size[e][agent_id][0]/2)-5)) / self.local_map_w
                    self.global_goal[e, agent_id, 1] = (goals[1] - (self.local_lmb[e, agent_id, 2]+self.center_h-math.ceil(self.sim_map_size[e][agent_id][1]/2)-5)) / self.local_map_h
                else:
                    self.global_goal[e, agent_id, 0] = (goals[0] - (self.center_w-math.ceil(self.sim_map_size[e][agent_id][0]/2)-5)) / (self.sim_map_size[e][agent_id][0]+10)
                    self.global_goal[e, agent_id, 1] = (goals[1] - (self.center_h-math.ceil(self.sim_map_size[e][agent_id][1]/2)-5)) / (self.sim_map_size[e][agent_id][1]+10)
                if self.direction_goal:
                    self.global_goal[e, agent_id, 0] = goals[0] / self.full_w
                    self.global_goal[e, agent_id, 1] = goals[1] / self.full_h
                
                self.global_goal[e, agent_id, 0] = max(0, min(1, self.global_goal[e, agent_id, 0]))
                self.global_goal[e, agent_id, 1] = max(0, min(1, self.global_goal[e, agent_id, 1]))

    
    
    def eval_compute_global_goal(self, rnn_states):      
        self.trainer.prep_rollout()

        concat_obs = {}
        for key in self.global_input.keys():
            concat_obs[key] = np.concatenate(self.global_input[key])
        
        # print(self.all_args.direction_greedy)
        actions, rnn_states = self.trainer.policy.act(concat_obs,
                                    np.concatenate(rnn_states),
                                    np.concatenate(self.global_masks),
                                    deterministic = True)

        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

        # Compute planner inputs
        if self.grid_goal:
            actions = actions.detach().clone()
            actions[:, 1:] = nn.Sigmoid()(actions[:, 1:])
            action = np.array(np.split(_t2n(actions), self.n_rollout_threads))
            r, c = action[:, :, 0].astype(np.int32) // self.grid_size, action[:, :, 0].astype(np.int32) % self.grid_size
            self.global_goal[:, :, 0] = ((action[:, :, 1] + r) / self.grid_size)*self.map_size[0]
            self.global_goal[:, :, 1] = ((action[:, :, 2] + c) / self.grid_size)*self.map_size[1]
        
        if self.proj_frontier:
            self.goal_to_frontier()
        for agent_id in range(self.num_agents): 
            self.all_global_goal_map[agent_id, int(self.global_goal[0, agent_id, 0]-2):int(self.global_goal[0, agent_id, 0]+3),int(self.global_goal[0, agent_id, 1]-2):int(self.global_goal[0, agent_id, 1]+3)]=1
        return rnn_states
    
    def ft_compute_global_goal(self, e):
        locations = self.update_ft_merge_map(e)
        
        inputs = {
            'map_pred' : self.ft_merge_map[e,0],
            'exp_pred' : self.ft_merge_map[e,1],
            'locations' : locations
        }
        goal_mask = [self.ft_go_steps[e][agent_id]<15 for agent_id in range(self.num_agents)]

        num_choose = self.num_agents - sum(goal_mask)
        self.env_info['merge_global_goal_num'] += num_choose

        goals = self.ft_get_goal(inputs, goal_mask, pre_goals = self.ft_pre_goals[e], e=e)

        for agent_id in range(self.num_agents):
            if not goal_mask[agent_id] or 'utility' in self.all_args.algorithm_name:
                self.ft_pre_goals[e][agent_id] = np.array(goals[agent_id], dtype=np.int32) # goals before rotation

        self.ft_goals[e]=self.rot_ft_goals(e, goals, goal_mask)
    
    def rot_ft_goals(self, e, goals, goal_mask = None):
        if goal_mask == None:
            goal_mask = [True for _ in range(self.num_agents)]
        ft_goals = np.zeros((self.num_agents, 2), dtype = np.int32)
        for agent_id in range(self.num_agents):
            if goal_mask[agent_id]:
                ft_goals[agent_id] = self.ft_goals[e][agent_id]
                continue
            self.ft_go_steps[e][agent_id] = 0
            output = np.zeros((1, 1, self.full_w, self.full_h), dtype =np.float32)
            output[0, 0, goals[agent_id][0]-1 : goals[agent_id][0]+2, goals[agent_id][1]-1:goals[agent_id][1]+2] = 1
            agent_n_trans = F.grid_sample(torch.from_numpy(output).float(), self.agent_trans[e][agent_id].float(), align_corners = True)
            map = F.grid_sample(agent_n_trans.float(), self.agent_rotation[e][agent_id].float(), align_corners = True)[0, 0, :, :].numpy()

            (index_a, index_b) = np.unravel_index(np.argmax(map[ :, :], axis=None), map[ :, :].shape) # might be wrong!!
            ft_goals[agent_id] = np.array([index_a, index_b], dtype = np.float32)
        return ft_goals
    
    def compute_merge_map_boundary(self, e, a, ft = True):
        return 0, self.full_w, 0, self.full_h

    def ft_compute_local_input(self):
        self.local_input = []

        for e in range(self.n_rollout_threads):
            p_input = defaultdict(list)
            for a in range(self.num_agents):
                lx, rx, ly, ry = self.compute_merge_map_boundary(e, a)
                p_input['goal'].append([int(self.ft_goals[e, a][0])-lx, int(self.ft_goals[e,a][1])-ly])
                if self.use_local_single_map:
                    if self.use_gt_map:
                        p_input['map_pred'].append(self.gt_full_map[e, a, 0, :, :].copy())
                        p_input['exp_pred'].append(self.gt_full_map[e, a, 1, :, :].copy())
                    else:
                        p_input['map_pred'].append(self.full_map[e, a, 0, :, :].copy())
                        p_input['exp_pred'].append(self.full_map[e, a, 1, :, :].copy())
                else:
                    trans_map = self.rot_map(self.merge_map[e, a, 0:2], e, a, self.agent_trans, self.agent_rotation)
                    p_input['map_pred'].append(trans_map[0, :, :].copy())
                    p_input['exp_pred'].append(trans_map[1, :, :].copy())
                pose_pred = self.planner_pose_inputs[e, a].copy()
                pose_pred[3:] = np.array((lx, rx, ly, ry))
                p_input['pose_pred'].append(pose_pred)
            self.local_input.append(p_input)
            
    def update_ft_merge_map(self, e):
        if self.use_gt_map:
            full_map = self.gt_full_map[e]
        else:
            full_map = self.full_map[e]
        self.ft_merge_map[e] = np.zeros((2, self.full_w, self.full_h), dtype = np.float32) # only explored and obstacle
        locations = []
        for agent_id in range(self.num_agents):
            output = torch.from_numpy(full_map[agent_id].copy())
            n_rotated = F.grid_sample(output.unsqueeze(0).float(), self.rotation[e][agent_id].float(), align_corners=True)
            n_map = F.grid_sample(n_rotated.float(), self.trans[e][agent_id].float(), align_corners = True)
            agent_merge_map = n_map[0, :, :, :].numpy()

            (index_a, index_b) = np.unravel_index(np.argmax(agent_merge_map[2, :, :], axis=None), agent_merge_map[2, :, :].shape) # might be inaccurate !!        
            self.ft_merge_map[e] += agent_merge_map[:2]
            locations.append((index_a, index_b))

        for i in range(2):
            self.ft_merge_map[e,i][self.ft_merge_map[e,i] > 1] = 1
            self.ft_merge_map[e,i][self.ft_merge_map[e,i] < self.map_threshold] = 0
        
        return locations
    
    def ft_get_goal(self, inputs, goal_mask, pre_goals = None, e=None):
        obstacle = inputs['map_pred']
        explored = inputs['exp_pred']
        locations = inputs['locations']

        if all(goal_mask):
            goals = []
            for agent_id in range(self.num_agents):
                goals.append((self.ft_pre_goals[e,agent_id][0], self.ft_pre_goals[e, agent_id][1]))
            return goals

        obstacle = np.rint(obstacle).astype(np.int32)
        explored = np.rint(explored).astype(np.int32)
        explored[obstacle == 1] = 1

        H, W = explored.shape
        steps = [(-1,0),(1,0),(0,-1),(0,1)]
        map, (lx, ly), unexplored = get_frontier(obstacle, explored, locations)
        '''
        map: H x W
            - 0 for explored & available cell
            - 1 for obstacle
            - 2 for target (frontier)
        '''
        self.ft_map[e] = map.copy()
        self.ft_lx[e] = lx
        self.ft_ly[e] = ly
        
        goals = []
        locations = [(x-lx, y-ly) for x, y in locations]
        if self.all_args.algorithm_name in ['ft_utility', 'ft_voronoi']:
            pre_goals = pre_goals.copy()
            pre_goals[:, 0] -= lx
            pre_goals[:, 1] -= ly
            if self.all_args.algorithm_name == 'ft_utility':
                goals = max_utility_frontier(map, unexplored, locations, clear_radius = self.all_args.ft_clear_radius, cluster_radius = self.all_args.ft_cluster_radius, utility_radius = self.all_args.utility_radius, pre_goals = pre_goals, goal_mask = goal_mask, random_goal=self.all_args.ft_use_random)
            elif self.all_args.algorithm_name == 'ft_voronoi':
                goals = voronoi_based_planning(map, unexplored, locations, clear_radius = self.all_args.ft_clear_radius, cluster_radius = self.all_args.ft_cluster_radius, utility_radius = self.all_args.utility_radius, pre_goals = pre_goals, goal_mask = goal_mask, random_goal=self.all_args.ft_use_random)
            goals[:, 0] += lx
            goals[:, 1] += ly
        else:
            for agent_id in range(self.num_agents): # replan when goal is not target
                if goal_mask[agent_id]:
                    goals.append((-1,-1))
                    continue
                if self.all_args.algorithm_name == 'ft_apf':
                    apf = APF(self.all_args)
                    path = apf.schedule(map, unexplored, locations, steps, agent_id, clear_disk = True, random_goal=self.all_args.ft_use_random)
                    goal = path[-1]
                elif self.all_args.algorithm_name == 'ft_nearest':
                    goal = nearest_frontier(map, unexplored, locations, steps, agent_id, clear_radius = self.all_args.ft_clear_radius, cluster_radius = self.all_args.ft_cluster_radius, random_goal=self.all_args.ft_use_random)
                elif self.all_args.algorithm_name == 'ft_rrt':
                    goal = rrt_global_plan(map, unexplored, locations, agent_id, clear_radius = self.all_args.ft_clear_radius, cluster_radius = self.all_args.ft_cluster_radius, step = self.env_step, utility_radius = self.all_args.utility_radius, random_goal=self.all_args.ft_use_random)
                else:
                    raise NotImplementedError
                goals.append((goal[0] + lx, goal[1] + ly))
        return goals
  
    def render_gifs(self):
        gif_dir = str(self.run_dir / 'gifs')

        folders = []
        get_folders(gif_dir, folders)
        filer_folders = [folder for folder in folders if "all" in folder or "merge" in folder]

        for folder in filer_folders:
            image_names = sorted(os.listdir(folder))

            frames = []
            for image_name in image_names:
                if image_name.split('.')[-1] == "gif":
                    continue
                image_path = os.path.join(folder, image_name)
                frame = imageio.imread(image_path)
                frames.append(frame)

            imageio.mimsave(str(folder) + '/render.gif', frames, duration=self.all_args.ifi)
    
    @torch.no_grad()
    def init_reset(self, map_size, pos, left_corner, obstacle_map, explored_map):
        self.map_size = map_size
        self.pos = np.array(pos)
        self.pos[:,0] = self.pos[:,0]+map_size[0]//2
        self.pos[:,1] = self.pos[:,1]+map_size[1]//2
        self.left_corner = np.array(left_corner)
        self.left_corner[:,0] = self.left_corner[:,0]+map_size[0]//2
        self.left_corner[:,1] = self.left_corner[:,1]+map_size[1]//2
        self.obstacle_map = obstacle_map
        self.explored_map = explored_map
        self.all_obstacle_map = np.zeros((self.map_size[0],self.map_size[1]))
        self.all_explored_map = np.zeros((self.map_size[0],self.map_size[1]))
        for agent_id in range(self.num_agents):
            map_shape = self.obstacle_map[agent_id].shape
            all_obstacle_map = np.zeros((self.map_size[0],self.map_size[1]))
            all_explored_map = np.zeros((self.map_size[0],self.map_size[1]))
            all_obstacle_map[self.left_corner[agent_id,0]:self.left_corner[agent_id,0]+map_shape[0],\
            self.left_corner[agent_id,1]:self.left_corner[agent_id,1]+map_shape[1]] = self.obstacle_map[agent_id]
            all_explored_map[self.left_corner[agent_id,0]:self.left_corner[agent_id,0]+map_shape[0],\
            self.left_corner[agent_id,1]:self.left_corner[agent_id,1]+map_shape[1]] = self.explored_map[agent_id]
            self.all_obstacle_map = np.maximum(self.all_obstacle_map,all_obstacle_map)
            self.all_explored_map = np.maximum(self.all_explored_map,all_explored_map)
            

        self.rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        self.global_trace_map = np.zeros((self.num_agents, self.full_w, self.full_h), dtype=np.float32)
        self.first_compute = True
        self.first_compute_global_input()
        
        self.rnn_states = self.eval_compute_global_goal(self.rnn_states)
        self.global_goal_position = self.global_goal[0].copy()
        for a in range(self.num_agents): 
            self.global_goal_position[a,0] = self.global_goal_position[a,0]-self.left_corner[a,0]
            self.global_goal_position[a,1] = self.global_goal_position[a,1]-self.left_corner[a,1]
        return self.global_goal_position
    
    def get_pos(self,pos):
        self.pos = np.array(pos)
        self.pos[:,0] = self.pos[:,0]+self.map_size[0]//2
        self.pos[:,1] = self.pos[:,1]+self.map_size[1]//2
        for agent_id in range(self.num_agents):
            self.global_trace_map[agent_id, int(self.pos[agent_id,0]-2):int(self.pos[agent_id,0]+3),int(self.pos[agent_id,1]-2):int(self.pos[agent_id,1]+3)] = 1

    @torch.no_grad()
    def get_global_goal_position(self, pos, left_corner, obstacle_map, explored_map):
        self.pos = np.array(pos)
        self.pos[:,0] = self.pos[:,0]+self.map_size[0]//2
        self.pos[:,1] = self.pos[:,1]+self.map_size[1]//2
        self.left_corner = np.array(left_corner)
        self.left_corner[:,0] = self.left_corner[:,0]+self.map_size[0]//2
        self.left_corner[:,1] = self.left_corner[:,1]+self.map_size[1]//2
        self.obstacle_map = obstacle_map
        self.explored_map = explored_map
        self.all_obstacle_map = np.zeros((self.map_size[0],self.map_size[1]))
        self.all_explored_map = np.zeros((self.map_size[0],self.map_size[1]))
        for agent_id in range(self.num_agents):
            map_shape = self.obstacle_map[agent_id].shape
            all_obstacle_map = np.zeros((self.map_size[0],self.map_size[1]))
            all_explored_map = np.zeros((self.map_size[0],self.map_size[1]))
            all_obstacle_map[self.left_corner[agent_id,0]:self.left_corner[agent_id,0]+map_shape[0],\
            self.left_corner[agent_id,1]:self.left_corner[agent_id,1]+map_shape[1]] = self.obstacle_map[agent_id]
            all_explored_map[self.left_corner[agent_id,0]:self.left_corner[agent_id,0]+map_shape[0],\
            self.left_corner[agent_id,1]:self.left_corner[agent_id,1]+map_shape[1]] = self.explored_map[agent_id]
            self.all_obstacle_map = np.maximum(self.all_obstacle_map,all_obstacle_map)
            self.all_explored_map = np.maximum(self.all_explored_map,all_explored_map)
        self.compute_global_input()
        self.global_trace_map = np.zeros((self.num_agents, self.full_w, self.full_h), dtype=np.float32)
        self.rnn_states = self.eval_compute_global_goal(self.rnn_states)
        self.global_goal_position = self.global_goal[0].copy() 
        for a in range(self.num_agents): 
            self.global_goal_position[a,0] = self.global_goal_position[a,0]-self.left_corner[a,0]
            self.global_goal_position[a,1] = self.global_goal_position[a,1]-self.left_corner[a,1]
        return self.global_goal_position  
        
    
    @torch.no_grad()
    def eval_ft(self):
        self.eval_infos = defaultdict(list)
        #self.panoramic_infos = defaultdict(list)
        
        for episode in range(self.all_args.eval_episodes):
            start = time.time()
            # store each episode ratio or reward
            self.init_env_info()
            self.env_step = 0
            if not self.use_delta_reward:
                self.rewards = np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32) 

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            # init map and pose 
            self.init_map_and_pose() 
            reset_choose = np.ones(self.n_eval_rollout_threads) == 1.0
            # reset env
            self.obs, infos = self.envs.reset(reset_choose)
            self.trans = [infos[e]['trans'] for e in range(self.n_rollout_threads)]
            self.rotation = [infos[e]['rotation'] for e in range(self.n_rollout_threads)]
            self.scene_id = [infos[e]['scene_id'] for e in range(self.n_rollout_threads)]
            self.agent_trans = [infos[e]['agent_trans'] for e in range(self.n_rollout_threads)]
            self.agent_rotation = [infos[e]['agent_rotation'] for e in range(self.n_rollout_threads)]
            self.explorable_map = np.array([infos[e]['explorable_map'] for e in range(self.n_rollout_threads)])
            self.sim_map_size = np.array([infos[e]['sim_map_size'] for e in range(self.n_rollout_threads)])
            self.empty_map = np.array([infos[e]['empty_map'] for e in range(self.n_rollout_threads)])
            self.gt_full_map = np.array([infos[e]['gt_full_map'] for e in range(self.n_rollout_threads)])
            if self.action_mask:
                self.global_input['action_mask_obs'] = np.ones((self.n_rollout_threads, self.num_agents, 1, self.grid_size, self.grid_size), dtype=np.int32)
                self.global_input['action_mask'] = np.ones((self.n_rollout_threads, self.num_agents, 1, self.grid_size, self.grid_size), dtype=np.int32)

            # Predict map from frame 1:
            self.run_slam_module(self.obs, self.obs, infos)

            # Compute Global policy input
            self.first_compute = True
            self.first_compute_global_input()

            # Compute Global goal
            for e in range(self.n_rollout_threads):
                self.ft_compute_global_goal(e)

            # compute local input

            self.ft_compute_local_input()

            # Output stores local goals as well as the the ground-truth action
            self.global_output = self.envs.get_short_term_goal(self.local_input)
            self.global_output = np.array(self.global_output, dtype = np.long)
            auc_area = np.zeros((self.n_rollout_threads, self.max_episode_length), dtype=np.float32)
            auc_single_area = np.zeros((self.n_rollout_threads, self.num_agents, self.max_episode_length), dtype=np.float32)
            for step in range(self.max_episode_length):
                self.env_step = step
                if step % 15 == 0:
                    print(step)
                local_step = step % self.num_local_steps
                global_step = (step // self.num_local_steps) % self.episode_length
                eval_global_step = step // self.num_local_steps + 1

                self.last_obs = copy.deepcopy(self.obs)

                # Sample actions
                actions_env = self.compute_local_action()

                # Obser reward and next obs
                self.obs, reward, dones, infos = self.envs.step(actions_env)
                #break
                self.gt_full_map = np.array([infos[e]['gt_full_map'] for e in range(self.n_rollout_threads)])
                for e in range(self.n_rollout_threads):
                    # for agent_id in range(self.num_agents):
                    #     self.panoramic_infos['graph_panoramic_rgb'].append(infos[e]['graph_panoramic_rgb'][agent_id])
                    #     self.panoramic_infos['graph_panoramic_depth'].append(infos[e]['graph_panoramic_depth'][agent_id])
                    for key in self.sum_env_info_keys:
                        if key in infos[e].keys():
                            self.env_info['sum_{}'.format(key)][e] += np.array(infos[e][key])
                            if key == 'merge_explored_ratio' and self.use_eval:
                                self.auc_infos['merge_auc'][episode, e, step] = self.auc_infos['merge_auc'][episode, e, step-1] + np.array(infos[e][key])
                                auc_area[e, step] = auc_area[e, step-1] + np.array(infos[e][key])
                            if key == 'explored_ratio' and self.use_eval:
                                self.auc_infos['agent_auc'][episode, e, :, step] = self.auc_infos['agent_auc'][episode, e, :, step-1] + np.array(infos[e][key])
                                auc_single_area[e, :, step] = auc_single_area[e, :, step-1] + np.array(infos[e][key])
                    for key in self.equal_env_info_keys:
                        if key == 'explored_ratio_step':
                            for agent_id in range(self.num_agents):
                                agent_k = "agent{}_{}".format(agent_id, key)
                                if agent_k in infos[e].keys():
                                    self.env_info[key][e][agent_id] = infos[e][agent_k]
                        elif key == 'balanced_ratio':
                            for agent_id in range(self.num_agents):
                                for a in range(self.num_agents):
                                    agent_k = "agent{}/{}_{}".format(agent_id, a, key)
                                    if agent_k in infos[e].keys():
                                        self.env_info[key][e][agent_id] = infos[e][agent_k] 
                        else:
                            if key in infos[e].keys():
                                self.env_info[key][e] = infos[e][key]
                    if self.env_info['sum_merge_explored_ratio'][e] <= self.all_args.explored_ratio_down_threshold:
                        self.env_info['merge_global_goal_num_%.2f'%self.all_args.explored_ratio_down_threshold][e] = self.env_info['merge_global_goal_num'][e]
                    
                    if self.num_agents==1:
                        if step in [49, 99, 149, 199, 249, 299, 349, 399, 449]:
                            self.env_info[str(step+1)+'step_merge_auc'][e] = auc_area[e, :step+1].sum().copy()
                            self.env_info[str(step+1)+'step_auc'][e] = auc_single_area[e, :, :step+1].sum(axis = 1)
                            self.env_info[str(step+1)+'step_merge_overlap_ratio'][e] = infos[e]['overlap_ratio']
                    else:
                        if step in [49, 99, 119, 149, 179, 199, 249, 299]:
                            self.env_info[str(step+1)+'step_merge_auc'][e] = auc_area[e, :step+1].sum().copy()
                            self.env_info[str(step+1)+'step_auc'][e] = auc_single_area[e, :, :step+1].sum(axis = 1)
                            self.env_info[str(step+1)+'step_merge_overlap_ratio'][e] = infos[e]['overlap_ratio'] 
                
                                
                self.local_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                self.local_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                self.global_masks *= self.local_masks

                self.run_slam_module(self.last_obs, self.obs, infos)
                self.update_local_map()
                self.update_map_and_pose()

                for a in range(self.num_agents):
                    if self.use_gt_map:
                        self.merge_map[:, a] = self.transform(self.gt_full_map, self.trans, self.rotation, self.agent_trans, self.agent_rotation, a)
                    else:
                        self.merge_map[:, a] = self.transform(self.full_map, self.trans, self.rotation, self.agent_trans, self.agent_rotation, a)

                # Global Policy
                self.ft_go_steps += 1

                for e in range (self.n_rollout_threads):
                    self.ft_last_merge_explored_ratio[e] = self.env_info['sum_merge_explored_ratio'][e]
                    self.ft_compute_global_goal(e) 
                        
                # Local Policy
                self.ft_compute_local_input()

                # Output stores local goals as well as the the ground-truth action
                self.global_output = self.envs.get_short_term_goal(self.local_input)
                self.global_output = np.array(self.global_output, dtype = np.long)
            
            self.convert_info()
            total_num_steps = (episode + 1) * self.max_episode_length * self.n_rollout_threads
            if not self.use_render :
                end = time.time()
                self.env_infos['merge_runtime'].append((end-start)/self.max_episode_length)
                self.log_env(self.env_infos, total_num_steps)
                self.log_single_env(self.env_infos, total_num_steps)
                self.log_agent(self.env_infos, total_num_steps)
            
        for k, v in self.env_infos.items():
            print("eval average {}: {}".format(k, np.nanmean(v) if k == 'merge_explored_ratio_step' or k == "merge_explored_ratio_step_0.95"else np.mean(v)))

        if self.all_args.save_gifs:
            print("generating gifs....")
            self.render_gifs()
            print("done!")
        #joblib.dump(self.panoramic_infos,'./dataset/22') 

    def render(self, obstacle_map, explored_map, pos, gif_dir):
        self.merge_obstacle_map = np.zeros((self.map_size[0],self.map_size[1]))
        self.merge_explored_map = np.zeros((self.map_size[0],self.map_size[1]))
        merge_zero_map = np.zeros((self.map_size[0],self.map_size[1]))
        for agent_id in range(self.num_agents):
            shape = obstacle_map[agent_id].shape
            corner = self.left_corner[agent_id]
            merge_obstacle_map = np.zeros((self.map_size[0],self.map_size[1]))
            merge_explored_map = np.zeros((self.map_size[0],self.map_size[1]))
            merge_obstacle_map[corner[0]:corner[0]+shape[0],corner[1]:corner[1]+shape[1]]=obstacle_map[agent_id]
            merge_explored_map[corner[0]:corner[0]+shape[0],corner[1]:corner[1]+shape[1]]=explored_map[agent_id]
            self.merge_obstacle_map = np.maximum(self.merge_obstacle_map,merge_obstacle_map)
            self.merge_explored_map = np.maximum(self.merge_explored_map,merge_explored_map)
        
        self.merge_explored_map[self.merge_explored_map!=1]=0
        
        vis_grid_gt = vu.get_colored_map(self.merge_obstacle_map,
                                    merge_zero_map,
                                    merge_zero_map,
                                    merge_zero_map,
                                    self.global_goal[0],
                                    self.merge_explored_map,
                                    merge_zero_map,
                                    merge_zero_map)
        
        vis_grid_pred = vu.get_colored_map(self.merge_obstacle_map,
                                    merge_zero_map,
                                    merge_zero_map,
                                    merge_zero_map,
                                    self.global_goal[0],
                                    self.merge_explored_map,
                                    merge_zero_map,
                                    merge_zero_map,)

        # vis_grid_gt = np.flipud(vis_grid_gt)
        # vis_grid_pred = np.flipud(vis_grid_pred)
       
        vu.visualize_map(self.figure_m, self.ax_m, vis_grid_gt[:, :, ::-1], vis_grid_pred[:, :, ::-1],
                        self.pos*5/100, self.pos*5/100, 
                        self.global_goal[0],
                        None,
                        None,
                        None,
                        None,
                        None,
                        #np.array(self.frontier_loc),
                        gif_dir,
                        self.time_step, 
                        True,
                        True)
        self.time_step += 1
            
   