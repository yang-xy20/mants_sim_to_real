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
import copy
import json
from .base_runner import Runner
import sys
sys.path.append("../..")
import joblib

# from torch_geometric.data import Data
from mants_sim_to_real.utils import pose as pu
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

class GraphHabitatRunner(Runner):
    def __init__(self, config):
        super(GraphHabitatRunner, self).__init__(config)
        # init parameters
        self.init_hyper_parameters()
        # global policy
        self.init_global_policy() 

    def init_hyper_parameters(self):
        self.map_resolution = self.all_args.map_resolution
        self.render_merge = self.all_args.render_merge
        self.visualize_input = self.all_args.visualize_input
        #build graph
        self.learn_to_build_graph = self.all_args.learn_to_build_graph
        self.graph_memory_size = self.all_args.graph_memory_size
        self.add_ghost = self.all_args.add_ghost
        self.use_merge = self.all_args.use_merge
        self.use_mgnn = self.all_args.use_mgnn
        self.use_all_ghost_add = self.all_args.use_all_ghost_add
        self.use_render = self.all_args.use_render
        self.ghost_node_size = self.all_args.ghost_node_size
        self.use_double_matching = self.all_args.use_double_matching
        self.figure_m, self.ax_m = plt.subplots(1, 1, figsize=(6,6),facecolor="white",num="Scene {} Merge Map".format(0))
        
        self.time_step = 0
    def init_global_policy(self):
        self.global_input = {}
        self.last_agent_world_pos = np.zeros((self.n_rollout_threads, (1000)*self.num_agents, 2), dtype = np.float32)
        if self.use_double_matching:
            self.global_input['agent_graph_node_dis'] = np.zeros((self.n_rollout_threads, self.num_agents, self.num_agents, self.graph_memory_size, 1), dtype = np.float32)
            self.global_input['graph_node_pos'] = np.zeros((self.n_rollout_threads, self.num_agents, self.graph_memory_size, 4), dtype = np.float32)
            self.global_input['graph_last_node_position'] = np.zeros((self.n_rollout_threads, self.num_agents, 1000*self.num_agents, 4), dtype = np.float32)
        self.global_input['graph_ghost_node_position'] = np.zeros((self.n_rollout_threads, self.num_agents, self.graph_memory_size, self.ghost_node_size, 4), dtype = np.float32)
        self.global_input['agent_world_pos'] = np.zeros((self.n_rollout_threads, self.num_agents, self.num_agents, 4), dtype = np.float32)
        self.global_input['graph_last_ghost_node_position'] = np.zeros((self.n_rollout_threads, self.num_agents, 1000*self.num_agents, 4), dtype = np.float32)
        self.global_input['graph_last_agent_world_pos'] = np.zeros((self.n_rollout_threads, self.num_agents, 1000*self.num_agents, 4), dtype = np.float32)
        if self.use_mgnn:
            self.global_input['graph_last_pos_mask'] = np.zeros((self.n_rollout_threads, self.num_agents, 1000*self.num_agents, 1), dtype = np.int32)
            self.global_input['graph_agent_dis'] = np.zeros((self.n_rollout_threads, self.num_agents, self.num_agents, self.graph_memory_size*self.ghost_node_size, 1), dtype = np.float32)
        if self.use_merge:
            if self.add_ghost:
                self.global_input['graph_merge_ghost_mask'] = np.zeros((self.n_rollout_threads, self.num_agents, self.graph_memory_size, self.ghost_node_size), dtype = np.int32)
        self.share_global_input = self.global_input.copy()
        self.global_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32) 
        self.global_goal = np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        if self.use_double_matching:
            self.node_goal = np.zeros((self.n_rollout_threads, self.num_agents, 2), dtype=np.float32)
        self.revise_global_goal = np.zeros((self.n_rollout_threads, self.num_agents, 2), dtype=np.int32)
        #self.global_goal_position  = np.zeros((self.n_rollout_threads, self.num_agents, 2), dtype=np.int32)
        if self.visualize_input:
            plt.ion()
            self.fig, self.ax = plt.subplots(self.num_agents*2, 8, figsize=(10, 2.5), facecolor="whitesmoke")
            self.fig_d, self.ax_d = plt.subplots(self.num_agents, 3 * self.all_args.direction_k + 1, figsize=(10, 2.5), facecolor = "whitesmoke")

    def compute_graph_input(self, infos, global_step):
      
        for key in self.global_input.keys():
            if key == 'graph_ghost_valid_mask':
                for e in range(self.n_rollout_threads):
                    edge = np.ones((self.graph_memory_size*self.ghost_node_size, self.graph_memory_size*self.ghost_node_size))
                    edge[np.where(infos['graph_merge_ghost_mask'].reshape(-1)==0), :] = 0
                    edge[:,np.where(infos['graph_merge_ghost_mask'].reshape(-1)==0)] = 0
                    self.global_input[key][e] = np.expand_dims(edge, 0).repeat(self.num_agents, axis=0)
            elif key == 'agent_world_pos':
                for e in range(self.n_rollout_threads):
                    agent_world_pos = self.pos.copy()
                    agent_world_pos[:,0] = agent_world_pos[:,0]/(self.max_size[0]*(self.map_resolution/100))
                    agent_world_pos[:,1] = agent_world_pos[:,1]/(self.max_size[1]*(self.map_resolution/100))
                    
                    agent_world_pos = np.concatenate((agent_world_pos, np.zeros((self.num_agents,1)),np.ones((self.num_agents,1))), axis = -1)
                    for a in range(self.num_agents):
                        self.global_input[key][e, a] = np.roll(agent_world_pos, a, axis=0)
            elif key == 'graph_prev_goal':
                self.global_input[key] = self.goal.copy()
            elif 'merge' in key:
                for e in range(self.n_rollout_threads):
                    self.global_input[key][e] = np.expand_dims(np.array(infos[key]),0).repeat(self.num_agents, axis=0)
            elif key == 'graph_agent_dis':
                for iter_ in range(self.num_agents):
                    position={}
                    for e in range(self.n_rollout_threads):
                        position['x'] = self.pos.copy()
                        position['y'] = infos['graph_ghost_node_position'].copy()
                        position['obstacle_map'] = self.obstacle_map[iter_]
                        position['corner'] = self.corner[iter_]
                    fmm_dis = self.compute_fmm_distance(position)
                    self.global_input[key][:,iter_] = np.roll(fmm_dis, iter_, axis=1)
            elif key == 'graph_agent_mask':
                pass
            elif key == 'graph_node_pos':
                self.global_input[key][:,:,:,:] = 0
                for e in range(self.n_rollout_threads):
                    world_node_pos = np.array(infos['graph_node_pos'])
                    world_node_pos[:,0] = world_node_pos[:,0]/(self.max_size[0]*(self.map_resolution/100))
                    world_node_pos[:,1] = world_node_pos[:,1]/(self.max_size[1]*(self.map_resolution/100))
                    world_node_pos[:,2] = 0
                    world_node_pos = np.concatenate((world_node_pos, np.ones((world_node_pos.shape[0],1))), axis = -1)
                    for agent_id in range(self.num_agents):
                        self.global_input[key][e, agent_id, :len(infos['graph_node_pos'])] = world_node_pos.copy()
            elif key == 'agent_graph_node_dis':              
                for agent_id in range(self.num_agents):
                    position={} 
                    for e in range(self.n_rollout_threads):
                        position['x'] = self.pos.copy()
                        count = len(infos['graph_node_pos'])
                        position['y'] = np.concatenate((np.array(infos['graph_node_pos']).copy(),np.zeros((self.graph_memory_size-count,3))),axis=0)
                        position['obstacle_map']  = self.obstacle_map[agent_id]
                        position['corner']  = self.corner[agent_id]
                    fmm_dis = self.compute_fmm_distance(position)
                    self.global_input[key][:,agent_id] = np.roll(fmm_dis, agent_id, axis=1)
            elif key == 'graph_last_node_position':
                if self.first_compute:
                    self.global_input[key][:, :, global_step*self.num_agents:(global_step+1)*self.num_agents,2] = 1
                else:
                    for e in range(self.n_rollout_threads):
                        node_world_pos = self.node_goal[e].copy()
                        node_world_pos[:,0] = node_world_pos[:,0]/(self.max_size[0]*(self.map_resolution/100))
                        node_world_pos[:,1] = node_world_pos[:,1]/(self.max_size[1]*(self.map_resolution/100))
                        node_world_pos = np.concatenate((node_world_pos,\
                        np.ones((self.num_agents,1)), np.zeros((self.num_agents,1))), axis=-1)
                        for agent_id in range(self.num_agents):
                            self.global_input[key][e, agent_id, (global_step-1)*self.num_agents:global_step*self.num_agents] = np.roll(node_world_pos, agent_id, axis = 0)
            elif key == 'graph_ghost_node_position':
                self.global_input[key][:,:,:,:] = 0
                for e in range(self.n_rollout_threads):
                    for iter_ in range(self.num_agents):
                        ghost_world_pos = infos[key].copy()
                        ghost_world_pos[:,:,0] = ghost_world_pos[:,:,0]/(self.max_size[0]*(self.map_resolution/100))
                        ghost_world_pos[:,:,1] = ghost_world_pos[:,:,1]/(self.max_size[1]*(self.map_resolution/100))
                        ghost_world_pos = np.concatenate((ghost_world_pos, np.zeros((self.graph_memory_size,self.ghost_node_size,1)), np.ones((self.graph_memory_size,self.ghost_node_size,1))), axis = -1)
                        self.global_input[key][e, iter_] = ghost_world_pos.copy()
            elif key == 'graph_last_ghost_node_position':
                if self.first_compute:
                    self.global_input[key][:, :, global_step*self.num_agents:(global_step+1)*self.num_agents,2] = 1
                else:
                    for e in range(self.n_rollout_threads):
                        ghost_world_pos = self.global_goal_position.copy()
                        ghost_world_pos[:,0] = ghost_world_pos[:,0]/self.max_size[0]
                        ghost_world_pos[:,1] = ghost_world_pos[:,1]/self.max_size[1]
                        ghost_world_pos = np.concatenate((ghost_world_pos,\
                        np.ones((self.num_agents,1)), np.zeros((self.num_agents,1))), axis=-1)
                        for a in range(self.num_agents):
                            self.global_input[key][e, a, (global_step-1)*self.num_agents:global_step*self.num_agents] = np.roll(ghost_world_pos, a, axis = 0)
            elif key == 'graph_last_agent_world_pos':
                if self.first_compute:
                    for e in range(self.n_rollout_threads):
                        agent_world_pos = self.pos.copy()
                        agent_world_pos[:,0] = agent_world_pos[:,0]/(self.max_size[0]*(self.map_resolution/100))
                        agent_world_pos[:,1] = agent_world_pos[:,1] /(self.max_size[1]*(self.map_resolution/100))
                   
                        agent_world_pos = np.concatenate((agent_world_pos,np.ones((self.num_agents,1)),np.zeros((self.num_agents,1))), axis = -1)
                        for a in range(self.num_agents):
                            self.global_input[key][e, a, global_step*self.num_agents:(global_step+1)*self.num_agents] = np.roll(agent_world_pos, a, axis=0)
                else:
                    last_agent_world_pos = self.last_agent_world_pos[:, (global_step-1)*self.num_agents:global_step*self.num_agents].copy()
                    last_agent_world_pos[:,:,0] = last_agent_world_pos[:,:,0]//(self.max_size[0]*(self.map_resolution/100))
                    last_agent_world_pos[:,:,1] = last_agent_world_pos[:,:,1]//(self.max_size[1]*(self.map_resolution/100))
                   
                    last_agent_world_pos = np.concatenate((last_agent_world_pos,np.ones((self.n_rollout_threads,self.num_agents,1)),np.zeros((self.n_rollout_threads,self.num_agents,1))),axis = -1)
                    for a in range(self.num_agents):
                        self.global_input[key][:, a, (global_step-1)*self.num_agents:global_step*self.num_agents] = np.roll(last_agent_world_pos, a, axis=1)
            elif key == 'graph_last_pos_mask':
                if self.first_compute:
                    self.global_input[key][:,:, :(global_step+1)*self.num_agents] = 1
                else:
                    self.global_input[key][:,:, :global_step*self.num_agents] = 1
        for e in range(self.n_rollout_threads):
            self.last_agent_world_pos[e, global_step*self.num_agents:(global_step+1)*self.num_agents] = self.pos.copy()
                 
        for key in self.global_input.keys():
            self.share_global_input[key] = self.global_input[key].copy()
         
        if self.visualize_input:
            self.visualize_obs(self.fig, self.ax, self.share_global_input)
    
    def eval_compute_global_goal(self, rnn_states):      
        self.trainer.prep_rollout()
        concat_obs = {}
        for key in self.global_input.keys():
            concat_obs[key] = np.concatenate(self.global_input[key])
        

        actions, rnn_states = self.trainer.policy.act(concat_obs,
                                    np.concatenate(rnn_states),
                                    np.concatenate(self.global_masks),
                                    available_actions = None,
                                    available_actions_first = None, #np.concatenate(self.available_actions_first) if self.use_each_node else None,
                                    available_actions_second= None, #np.concatenate(self.available_actions_second) if self.use_each_node else None,
                                    deterministic=True)
        
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

        # Compute planner inputs
        self.global_goal = np.array(np.split(_t2n(actions), self.n_rollout_threads))
        if self.use_double_matching:
            for e in range(self.n_rollout_threads):
                mask = self.global_input['graph_merge_ghost_mask'][e,0]
                node_counts = np.sum(mask,axis=-1)
                for a in range(self.num_agents):
                    temp_idx = 0
                    for idx in range(node_counts.shape[0]):
                        if temp_idx > self.global_goal[e,a]:
                            temp_idx = idx - 1
                            break
                        else:
                            temp_idx += node_counts[idx]
                    self.node_goal[e,a] = self.global_input['graph_node_pos'][e,a,temp_idx,:2]
        return rnn_states, actions
    
    def compute_fmm_distance(self, positions):
        fmm_dis = self.envs.get_runner_fmm_distance(positions,self.max_size)
        return fmm_dis

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
    
    def visualize_obs(self, fig, ax, obs):
        # individual
        for agent_id in range(self.num_agents * 2):
            sub_ax = ax[agent_id]
            for i in range(8):
                sub_ax[i].clear()
                sub_ax[i].set_yticks([])
                sub_ax[i].set_xticks([])
                sub_ax[i].set_yticklabels([])
                sub_ax[i].set_xticklabels([])
                if agent_id < self.num_agents:
                    sub_ax[i].imshow(obs["global_merge_obs"][0, agent_id,i])
                elif agent_id >= self.num_agents and i<4:
                    sub_ax[i].imshow(obs["global_obs"][0, agent_id-self.num_agents,i])
        plt.gcf().canvas.flush_events()
        # plt.pause(0.1)
        fig.canvas.start_event_loop(0.001)
        plt.gcf().canvas.flush_events()
    
    @torch.no_grad()
    def init_reset(self, pos, max_size, obstacle_map,corner):
        self.rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        self.add_node = np.ones((self.n_rollout_threads,self.num_agents))*False
        self.add_node_flag = np.ones((self.n_rollout_threads,self.num_agents))*False
        self.max_size = max_size
        
        self.obstacle_map = obstacle_map
        corner = np.array(corner) 
        self.corner = np.array(corner) 
        self.corner[:,0] = corner[:,0]+ max_size[0]//2
        self.corner[:,1] = corner[:,1]+ max_size[1]//2
        
        pos = np.array(pos) 
        self.pos = np.array(pos,np.float)
        self.pos[:,0] = (pos[:,0] + max_size[0]//2) * self.map_resolution/100.0
        self.pos[:,1] = (pos[:,1] + max_size[1]//2) * self.map_resolution/100.0
        
        infos = self.envs.update_merge_graph(self.pos)
        self.goal = np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype=np.int32)
        #self.global_goal_position = np.zeros((self.n_rollout_threads, self.num_agents, 2), dtype=np.int32)

        self.first_compute = True
        self.compute_graph_input(infos,0)
        self.rnn_states, _ = self.eval_compute_global_goal(self.rnn_states)

        # compute local input
        self.goal = np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype=np.int32)
        self.first_compute = False
        
        for e in range(self.n_rollout_threads):
            self.goal[e] = self.global_goal[e]
            
        self.global_goal_position, self.valid_ghost_position = self.envs.get_goal_position(self.goal[0])
        self.global_goal_position = self.global_goal_position[:,0]
        
        self.add_ghost_flag = np.ones((1,self.valid_ghost_position.shape[0]))*False
        global_goal_position = self.global_goal_position.copy()
        return global_goal_position

    @torch.no_grad()
    def build_graph(self, pos, left_corner=None, ratio=None, explored_map=None, explored_map_no_obs=None, obstacle_map=None, update=False): 
        pos = np.array(pos) 
        self.pos = np.array(pos, np.float)
        self.pos[:,0] = (pos[:,0] + self.max_size[0]//2) * self.map_resolution/100.0
        self.pos[:,1] = (pos[:,1] + self.max_size[1]//2) * self.map_resolution/100.0
        for e in range(self.n_rollout_threads):
            for agent_id in range(self.num_agents):
                if self.use_all_ghost_add:
                    for ppos in range(self.valid_ghost_position.shape[0]):
                        if self.valid_ghost_position[pos].sum() == 0:
                            pass
                        else:
                            if pu.get_l2_distance(self.pos[agent_id,0] ,self.valid_ghost_position[ppos,0],\
                            self.pos[agent_id,1], self.valid_ghost_position[ppos,1]) < 0.5 and \
                            self.add_ghost_flag[e, ppos] == False:
                                self.add_node[e][agent_id] = True
                                self.add_ghost_flag[e, ppos] = True
        env_infos={}
        for e in range(self.n_rollout_threads):
            if update:
                env_infos['update'] = True
                env_infos['add_node'] = self.add_node[0]
                left_corner = np.array(left_corner)
                self.corner = left_corner
                self.corner[:,0] = left_corner[:,0]+self.max_size[0]//2
                self.corner[:,1] = left_corner[:,1]+self.max_size[1]//2
                infos, reward = self.envs.update_merge_step_graph(env_infos,self.pos, self.max_size, self.corner, ratio, obstacle_map,explored_map_no_obs, explored_map)
            else:
                env_infos['update'] = False
                env_infos['add_node'] = self.add_node[0]
                infos, reward = self.envs.update_merge_step_graph(env_infos, self.pos, self.max_size)

        self.add_node = np.ones((self.n_rollout_threads,self.num_agents))*False
        self.local_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        #self.local_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        self.global_masks *= self.local_masks
        return infos

    @torch.no_grad()
    def get_global_goal(self, obstacle_map, global_step, infos):                
        # For every global step, update the full and local maps
        self.add_node_flag = np.ones((self.n_rollout_threads,self.num_agents))*False
        self.global_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        
        #define the updated map and pose
        self.obstacle_map = obstacle_map
        self.compute_graph_input(infos, global_step+1)
        # Compute Global goal
        self.rnn_states, _ = self.eval_compute_global_goal(self.rnn_states)                        
    
        for e in range(self.n_rollout_threads):
            self.goal[e] = self.global_goal[e]
            
        self.global_goal_position, self.valid_ghost_position = self.envs.get_goal_position(self.goal[0])
        self.global_goal_position = self.global_goal_position[:,0]
        
        self.add_ghost_flag = np.ones((1,self.valid_ghost_position.shape[0]))*False
        global_goal_position = self.global_goal_position.copy()
        return  global_goal_position

    def render(self, obstacle_map, explored_map, pos, gif_dir):
        self.merge_obstacle_map = np.zeros((self.max_size[0],self.max_size[1]))
        self.merge_explored_map = np.zeros((self.max_size[0],self.max_size[1]))
        merge_zero_map = np.zeros((self.max_size[0],self.max_size[1]))
        for agent_id in range(self.num_agents):
            shape = obstacle_map[agent_id].shape
            corner = self.corner[agent_id]
            merge_obstacle_map = np.zeros((self.max_size[0],self.max_size[1]))
            merge_explored_map = np.zeros((self.max_size[0],self.max_size[1]))
            merge_obstacle_map[corner[0]:corner[0]+shape[0],corner[1]:corner[1]+shape[1]]=obstacle_map[agent_id]
            merge_explored_map[corner[0]:corner[0]+shape[0],corner[1]:corner[1]+shape[1]]=explored_map[agent_id]
            self.merge_obstacle_map = np.maximum(self.merge_obstacle_map,merge_obstacle_map)
            self.merge_explored_map = np.maximum(self.merge_explored_map,merge_explored_map)
        
        self.merge_explored_map[self.merge_explored_map!=1]=0
        
        vis_grid_gt = vu.get_colored_map(self.merge_obstacle_map,
                                    merge_zero_map,
                                    merge_zero_map,
                                    merge_zero_map,
                                    self.global_goal_position,
                                    self.merge_explored_map,
                                    merge_zero_map,
                                    merge_zero_map)
        
        vis_grid_pred = vu.get_colored_map(self.merge_obstacle_map,
                                    merge_zero_map,
                                    merge_zero_map,
                                    merge_zero_map,
                                    self.global_goal_position,
                                    self.merge_explored_map,
                                    merge_zero_map,
                                    merge_zero_map,)

        # vis_grid_gt = np.flipud(vis_grid_gt)
        # vis_grid_pred = np.flipud(vis_grid_pred)
       
        vu.visualize_map(self.figure_m, self.ax_m, vis_grid_gt[:, :, ::-1], vis_grid_pred[:, :, ::-1],
                        self.pos, self.pos, 
                        self.envs.graph_agent_0.node_position_list ,
                        self.envs.graph_agent_0.A,
                        self.envs.graph_agent_0.last_localized_node_idx,
                        self.envs.graph_agent_0.ghost_node_position,
                        self.envs.graph_agent_0.ghost_mask,
                        #np.array(self.frontier_loc),
                        gif_dir,
                        self.time_step, 
                        True,
                        True)
        self.time_step += 1



# for k, v in self.env_infos.items():
#     print("eval average {}: {}".format(k, np.nanmean(v) if k == 'merge_explored_ratio_step' or k == "merge_explored_ratio_step_0.95"else np.mean(v)))

# if self.all_args.save_gifs:
#     print("generating gifs....")
#     self.render_gifs()
#     print("done!")
