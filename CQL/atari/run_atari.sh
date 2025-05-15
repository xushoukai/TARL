
CUDA_VISIBLE_DEVICES=6,7 python -um batch_rl.fixed_replay.train \
 --base_dir=/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
 --replay_dir=/CQL/atari/datasets/Asterix/1 \
 --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
 --gin_bindings='FixedReplayRunner.num_iterations=10001' \
 --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5_augData"' \
 --gin_bindings='FixedReplayRunner.only_eval=True' \
 --gin_bindings='FixedReplayRunner.seed=0' \
 --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
 --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
 --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
 --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
 --gin_bindings='DQNAgent.action_class_dim=9' \
 --gin_bindings='DQNAgent.entropy=0.1' \
 --gin_bindings='DQNAgent.kl_coef=1.5'
