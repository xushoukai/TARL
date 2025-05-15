

CUDA_VISIBLE_DEVICES=0 python /OfflineRL-Kit/run_example/run_cql_ln_before_activation.py \
    --task walker2d-full-replay --algo-name cql_ln_before_activation_best_and_last10episode \

# #用最后一个policy
CUDA_VISIBLE_DEVICES=0 python /OfflineRL-Kit/run_example/run_eval_cql_ln_before_activation.py \
    --task walker2d-full-replay --algo-name eval_iql_ln_before_activation_best_lr0_buffer10sample1 \
    --checkpoints "xxxxxxx"