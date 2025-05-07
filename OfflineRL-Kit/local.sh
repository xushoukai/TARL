#python run_example/run_mcq_ln_before_activation.py \
#    --task umaze-v2 
python run_example/run_eval_mcq_ln_before_activation.py \
    --task antmaze-large-diverse-v2 --algo-name tta_mcql_lr1e-6_action_consistent_episode200_bs64_max_q_test \
    --tta-actor-lr 1e-6 --klcoef 0.0 --eval_episodes 100 --only-eval-tta False \
    --checkpoints "/apdcephfs/share_1594716/zihaolian/code/transfer_rl/log_OfflineRL-kit/walker2d-random-v2/mcq_ln_before_activation_best/seed_0&timestamp_23-0925-220658/checkpoint/best_policy.pth"
