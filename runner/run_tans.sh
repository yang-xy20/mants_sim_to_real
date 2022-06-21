CUDA_VISIBLE_DEVICES=0,5 python main_tans.py --n_rollout_threads 1 --hidden_size 256 --num_local_steps 15 \
--use_recurrent_policy --num_agents 2 --use_single --use_goal --grid_goal --use_grid_simple \
--grid_pos --grid_last_goal --cnn_use_transformer --use_share_cnn_model --agent_invariant \
--invariant_type alter_attn --use_pos_embedding --use_id_embedding --multi_layer_cross_attn \
--add_grid_pos --use_self_attn --use_intra_attn --use_maxpool2d \
--cnn_layers_params "32,3,1,1 64,3,1,1 128,3,1,1 64,3,1,1 32,3,2,1" \
--use_tans --model_dir /home/yangxy/yangxy/onpolicy/onpolicy/mants_sim_to_real/data/tans_model/
#--ft_use_random