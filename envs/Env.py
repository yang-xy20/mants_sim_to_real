import gym
import numpy as np

class MultiHabitatEnv(object):
    def __init__(self, args, build_graph, run_dir):

        self.all_args = args
        
        self.run_dir = run_dir
        self.num_agents = args.num_agents
        
        self.num_local_steps = args.num_local_steps
        local_w, local_h = 240,240
       
        if build_graph:
            global_observation_space = self.build_graph_global_obs()
        else:
            global_observation_space = {}
            if args.grid_pos:
                global_observation_space['grid_pos'] = gym.spaces.Box(
                    low=0, high=args.grid_size, shape=(self.num_agents*2, args.grid_size, args.grid_size), dtype='int')
            if args.grid_last_goal:
               
                global_observation_space['grid_goal'] = gym.spaces.Box(
                    low=0, high=args.grid_size, shape=(self.num_agents*2, args.grid_size, args.grid_size), dtype='int')
            
            c_single_stack = 0
            
            if args.use_single:
                global_observation_space['global_obs'] = gym.spaces.Box(
                        low=0, high=1, shape=(4, local_w, local_h), dtype='uint8')
                c_single_stack += 4
                
                if args.use_goal:
                    global_observation_space['global_goal'] = gym.spaces.Box(
                        low=0, high=1, shape=(2, local_w, local_h), dtype='uint8')
                    c_single_stack += 2
            
            
            global_observation_space['stack_obs'] = gym.spaces.Box(
                low=0, high=1, shape=(c_single_stack * self.num_agents, local_w, local_h), dtype='uint8')
        
        share_global_observation_space = global_observation_space.copy()
        
        
        global_observation_space = gym.spaces.Dict(global_observation_space)
        share_global_observation_space = gym.spaces.Dict(share_global_observation_space)

        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        for _ in range(self.num_agents):
            self.observation_space.append(global_observation_space)
            self.share_observation_space.append(share_global_observation_space)
            if build_graph:
                self.action_space.append(gym.spaces.Discrete(self.graph_memory_size)) 
            else:
                self.action_space.append([gym.spaces.Discrete(args.grid_size ** 2), gym.spaces.Box(
                    low=0.0, high=1.0, shape=(2,), dtype=np.float32)])

   
  
   