#!/usr/bin/env python
import sys
import os
import socket
import numpy as np
from pathlib import Path
from collections import deque
import torch
from onpolicy.sim_to_real.envs.Graph_Env import GraphHabitatEnv
from onpolicy.sim_to_real.utils.config import get_config
from fakesim import fakesim

def make_eval_env(all_args, run_dir):
    env = GraphHabitatEnv(args=all_args,run_dir=run_dir)
    return env

def parse_args(args, parser):
    parser.add_argument('--num_agents', type=int,
                        default=1, help="number of players")
    # visual params
    parser.add_argument("--render_merge", action='store_false', default=True,
                        help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.")
    parser.add_argument("--visualize_input", action='store_true', default=False,
                        help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.")
    # graph
    parser.add_argument('--build_graph', default=False, action='store_true')
    parser.add_argument('--add_ghost', default=False, action='store_true')
    parser.add_argument('--use_merge', default=False, action='store_true')
    parser.add_argument('--use_global_goal', default=False, action='store_true')
    parser.add_argument('--cut_ghost', default=False, action='store_true')
    parser.add_argument('--learn_to_build_graph', default=False, action='store_true')
    parser.add_argument('--use_mgnn', default=False, action='store_true')
    parser.add_argument('--dis_gap', default=2, type=int)
    parser.add_argument('--use_all_ghost_add', default=False, action='store_true')
    parser.add_argument('--ghost_node_size', default=12, type=int)
    parser.add_argument('--use_double_matching', default=False, action='store_true')
    parser.add_argument('--matching_type', type=str)
    parser.add_argument('--map_resolution', type=int, default=5)
    parser.add_argument('--graph_memory_size', default=100, type=int)
    
    parser.add_argument('--num_local_steps', type=int, default=25,
                    help="""Number of steps the local can
                        perform between each global instruction""")

    # image retrieval
    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # cuda 
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    # run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
    #                0] + "/results") / all_args.env_name / all_args.algorithm_name / all_args.experiment_name
    # if not run_dir.exists():
    #     os.makedirs(str(run_dir))

    # # wandb
    # if not run_dir.exists():
    #     curr_run = 'run1'
    # else:
    #     exst_run_nums = [int(str(folder.name).split('run')[
    #                             1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
    #     if len(exst_run_nums) == 0:
    #         curr_run = 'run1'
    #     else:
    #         curr_run = 'run%i' % (max(exst_run_nums) + 1)
    #     run_dir = run_dir / curr_run
    #     if not run_dir.exists():
    #         os.makedirs(str(run_dir))

    # setproctitle.setproctitle(str(all_args.algorithm_name) + "-" +
    #                           str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))
    run_dir = './test'
    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_eval_env(all_args, run_dir)
    eval_envs = None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    
    from onpolicy.sim_to_real.runner.graph_habitat_runner import GraphHabitatRunner as Runner

    runner = Runner(config)
    
    return runner, all_args.num_local_steps


if __name__ == "__main__":
    runner, num_local_steps = main(sys.argv[1:])#init_graph_runner
    fake_sim = fakesim(2)
    pos, _, max_size, _, _, obstacle_map, left_corner = fake_sim.reset()#zzl 
    max_size = np.array((420,560))
    
    step = 0
    global_goal_position = runner.init_reset(pos, max_size, obstacle_map,left_corner)#init_graph_runner
    while step < 100:
        global_step = step // num_local_steps
        pos, ratio, explored_map, explored_map_no_obs, obstacle_map, left_corner= fake_sim.step(global_goal_position)#zzl 
        #import pdb;pdb.set_trace()
        if step % num_local_steps == num_local_steps-1:
            update = True
            infos = runner.build_graph(pos, left_corner, ratio, explored_map, explored_map_no_obs, obstacle_map, update)#build_graph
            
        else: 
            update = False
            infos = runner.build_graph(pos, update = update)#build_graph
        if step % num_local_steps == num_local_steps-1:
            global_goal_position = runner.get_global_goal(obstacle_map, global_step, infos)#global_goal
        runner.render(obstacle_map, explored_map, pos, './test/figures')
        step += 1
    
