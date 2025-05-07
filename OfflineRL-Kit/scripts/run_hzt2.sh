# CUDA_VISIBLE_DEVICES=1 python /lichenghao/lch/transfer_rl-main/OfflineRL-Kit/run_example/run_iql_ln_before_activation.py \
#     --task walker2d-expert-v2 --algo-name iql_ln_before_activation_best_and_last10episode_walker2d-expert-v2 \


# CUDA_VISIBLE_DEVICES=1 python /lichenghao/lch/transfer_rl-main/OfflineRL-Kit/run_example/run_eval_iql_ln_before_activation.py \
#     --task walker2d-expert-v2 --algo-name eval_cql_ln_before_activation_best_lr1e-6_buffer10sample1 \
#     --tta-actor-lr 1e-6  --eval_episodes 10 --only-eval-tta False \
#     --checkpoints "/lichenghao/lch/transfer_rl-main/OfflineRL-Kit/scripts/log/walker2d-medium-v2/iql_ln_before_activation_best_and_last10episode/seed_0&timestamp_25-0330-113638/checkpoint/best_policy.pth"

CUDA_VISIBLE_DEVICES=1 python /lichenghao/lch/transfer_rl-main/OfflineRL-Kit/run_example/run_cql_ln_before_activation_lzh.py \
    --task walker2d-full-replay --algo-name cql_ln_before_activation_best_and_last10episode \

# #用最后一个policy
# CUDA_VISIBLE_DEVICES=0 python /lichenghao/lch/transfer_rl-main/OfflineRL-Kit/run_example/run_eval_iql_ln_before_activation.py \
#     --task walker2d-medium-v2 --algo-name eval_iql_ln_before_activation_best_lr0_buffer10sample1 \
#     --checkpoints "/lichenghao/lch/transfer_rl-main/OfflineRL-Kit/scripts/log/walker2d-medium-v2/iql_ln_before_activation_best_and_last10episode/seed_0&timestamp_25-0330-113638/checkpoint/policy_1000.pth"