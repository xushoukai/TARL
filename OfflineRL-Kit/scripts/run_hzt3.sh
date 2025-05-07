# CUDA_VISIBLE_DEVICES=2 python /lichenghao/lch/transfer_rl-main/OfflineRL-Kit/run_example/run_eval_iql_ln_before_activation.py \
#     --task walker2d-medium-v2 --algo-name eval_iql_ln_before_activation_best_lr0_buffer10sample1 \
#     --checkpoints "/lichenghao/lch/transfer_rl-main/OfflineRL-Kit/scripts/log/walker2d-medium-v2/iql_ln_before_activation_best_and_last10episode/seed_0&timestamp_25-0330-113638/checkpoint/policy_1000.pth"

# CUDA_VISIBLE_DEVICES=1 python /lichenghao/lch/transfer_rl-main/OfflineRL-Kit/run_example/run_cql_ln_before_activation_lzh.py \
#     --task walker2d-full-replay --algo-name cql_ln_before_activation_best_and_last10episode \

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
export CUDA_VISIBLE_DEVICES=5
python /lichenghao/lch/transfer_rl-main/OfflineRL-Kit/run_example/run_eval_iql_ln_before_activation_lzh.py \
    --task walker2d-expert-v2 --algo-name eval_iql_walker2d-expert-v2_mean_nofilter \
    --checkpoints "/lichenghao/lch/transfer_rl-main/OfflineRL-Kit/scripts/log/walker2d-expert-v2/iql_ln_before_activation_best_and_last10episode_walker2d-expert-v2/seed_0&timestamp_25-0330-115415/checkpoint/best_policy.pth" \
    --val_epoch 10 \
    --loss_agg_type mean

