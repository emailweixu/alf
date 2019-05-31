CUDA_VISIBLE_DEVICES=1 python train_ppo_pr2_img.py \
                    --root_dir=$1 \
                    --env_name=SocialBot-Pr2Gripper-v0 \
                    --gin_param='train_eval.learning_rate=2e-4' \
                    --gin_param='PPOAgent.use_gae=True' \
                    --gin_param='PPOAgent.use_td_lambda_return=True' \
                    --num_parallel_environments=4 \
                    --num_environment_steps=30000000 \
                    --collect_episodes_per_iteration=4 \
                    --gin_param='train_eval.entropy_regularization=0.0' \
                    --gin_param='train_eval.num_epochs=10' \
                    --gin_param='train_eval.use_tf_functions=False' \
                    --gin_param='PPOAgent.importance_ratio_clipping=0.2' \
                    --gin_param='PPOAgent.check_numerics=True' \
                    --gin_param='PPOAgent.initial_adaptive_kl_beta=0' \
                    --gin_param='PPOAgent.normalize_rewards=False' \
                    --gin_param='PPOAgent.kl_cutoff_factor=0' \
                    --gin_param='PPOAgent.normalize_observations=False' \
                    --alsologtostderr

# if we need multiple frames of observation, could add a single wrapper.
#--gin_param='suite_socialbot/load.gym_env_wrappers=(@suite_socialbot/FrameStackWrapper,)'