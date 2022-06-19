    
import time
import os
import numpy as np
from itertools import chain
import torch
from tensorboardX import SummaryWriter
from onpolicy.utils.shared_buffer import SharedReplayBuffer
from onpolicy.utils.util import update_linear_schedule
import gym
import copy

def _t2n(x):
    return x.detach().cpu().numpy()

class Runner(object):
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']   

        # dir
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.recurrent_N = self.all_args.recurrent_N
        self.model_dir = self.all_args.model_dir
        self.hidden_size = self.all_args.hidden_size
        
        from onpolicy.sim_to_real.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
        from onpolicy.sim_to_real.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy

        
        share_observation_space = self.envs.observation_space[0]

       
        # policy network
        self.policy = Policy(self.all_args,
                            self.envs.observation_space[0],
                            share_observation_space,
                            self.envs.action_space[0],
                            device = self.device)

        if self.model_dir is not None:
            self.restore()

        # algorithm
        self.trainer = TrainAlgo(self.all_args, self.policy, device = self.device)
        
    def restore(self):
        policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor.pt', map_location=self.device)
        self.policy.actor.load_state_dict(policy_actor_state_dict)
            
 
  