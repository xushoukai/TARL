export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
export CUDA_VISIBLE_DEVICES=0
python /lichenghao/lch/transfer_rl-main/OfflineRL-Kit/run_example/run_eval_cql_ln_before_activation_lch.py \
    --task walker2d-medium-v2 --algo-name eval_debug \
    --checkpoints "/lichenghao/lch/transfer_rl-main/OfflineRL-Kit/run_example/log/walker2d-medium-v2/cql_ln_before_activation_best/seed_0&timestamp_25-0330-123004/checkpoint/policy.pth" \
    --loss_agg_type abs --is_entropy_filter Ture  --entropy_threshold 0.9