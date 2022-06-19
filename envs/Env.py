import numpy as np
import gym
import onpolicy
from onpolicy.utils.multi_discrete import MultiDiscrete
from .graph import Graph
import torch.nn as nn
import torch
import os


class MultiHabitatEnv(object):
    def __init__(self, args, run_dir):

        self.all_args = args
        
        self.run_dir = run_dir
        self.num_agents = args.num_agents
        self.build_graph = args.build_graph
        self.add_ghost = args.add_ghost
        self.num_local_steps = args.num_local_steps
  
        self.use_mgnn = args.use_mgnn

        global_observation_space = self.build_graph_global_obs()
       
        
        share_global_observation_space = global_observation_space.copy()
        
        
        global_observation_space = gym.spaces.Dict(global_observation_space)
        share_global_observation_space = gym.spaces.Dict(share_global_observation_space)

        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        for _ in range(self.num_agents):
            self.observation_space.append(global_observation_space)
            self.share_observation_space.append(share_global_observation_space)
            self.action_space.append(gym.spaces.Discrete(self.graph_memory_size)) 

   
  
   