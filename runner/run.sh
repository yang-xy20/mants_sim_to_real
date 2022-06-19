CUDA_VISIBLE_DEVICES=0,5 python main.py --n_rollout_threads 1 --ghost_node_size 24 --use_all_ghost_add \
--learn_to_build_graph --dis_gap 3 --graph_memory_size 100 --build_graph --use_merge --add_ghost \
--feature_dim 512 --hidden_size 256 --use_mgnn --use_global_goal --cut_ghost --num_local_steps 15 \
--use_recurrent_policy --num_agents 2 --model_dir "/home/yangxy/yangxy/onpolicy/onpolicy/sim_to_real/data/model"