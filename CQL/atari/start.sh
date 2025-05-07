# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/REM_200it \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --agent_name=multi_head_dqn \
#  --gin_files='batch_rl/fixed_replay/configs/rem.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=200' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' 


# CUDA_VISIBLE_DEVICES=0,1,2 python -um batch_rl.fixed_replay.train \
#   --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/REM_200it \
#   --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#   --agent_name=multi_head_dqn \
#   --gin_files='batch_rl/fixed_replay/configs/rem.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=200' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' 


# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/REM_200it \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --agent_name=multi_head_dqn \
#  --gin_files='batch_rl/fixed_replay/configs/rem.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=200' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' 


# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/REM_200it \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --agent_name=multi_head_dqn \
#  --gin_files='batch_rl/fixed_replay/configs/rem.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=200' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' 


# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_200it \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --agent_name=quantile \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=200' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' 


# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_200it \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --agent_name=quantile \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=200' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' 


# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_200it \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --agent_name=quantile \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=200' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' 


# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_200it \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --agent_name=quantile \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=200' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' 


# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_200it \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --agent_name=quantile \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=200' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' 


#CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
# --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/QR_DQN_200it \
# --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
# --agent_name=quantile \
# --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
# --gin_bindings='FixedReplayRunner.num_iterations=200' \
# --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
# --gin_bindings='FixedReplayQuantileAgent.minq_weight=0'


# sleep 1h

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/QR_DQN_200it \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --agent_name=quantile \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=200' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=0'


# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/QR_DQN_200it \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --agent_name=quantile \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=200' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=0'


# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/QR_DQN_200it \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --agent_name=quantile \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=200' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=0'


# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/QR_DQN_200it \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --agent_name=quantile \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=200' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=0'


# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_200it \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --agent_name=quantile \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=200' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' 


# # # 2023/08/16
# # # gpu017 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Qbert action dim: 18 
# CUDA_VISIBLE_DEVICES=0,1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/QR_DQN_200it \
#    --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#    --agent_name=quantile \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=0.0' 

# wait

# sleep 1d

# # # 2023/08/16
# # # gpu017 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Qbert action dim: 18 
# CUDA_VISIBLE_DEVICES=0,1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/QR_DQN_200it \
#    --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#    --agent_name=quantile \
#    --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=0.0' \


# CUDA_VISIBLE_DEVICES=0,1,2 python -um batch_rl.fixed_replay.train \
#   --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/REM_200it \
#   --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#   --agent_name=multi_head_dqn \
#   --gin_files='batch_rl/fixed_replay/configs/rem.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=200' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' 

# wait

# # # 2023/08/16
# # # gpu017 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Qbert action dim: 18 
# CUDA_VISIBLE_DEVICES=0,1,2 python -um batch_rl.fixed_replay.train \
#    --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/REM_200it \
#    --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#    --agent_name=multi_head_dqn \
#    --gin_files='batch_rl/fixed_replay/configs/rem.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' 


# # # 2023/09/08
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# instance: AILab_MLC_QY3DB5994C16E74F5090BE8F307ADE
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#   --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"'


# # # 2023/09/08
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# instance: AILab_MLC_QY8DCBBC221F744682A1E3287BA9F6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#   --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"'


# # # 2023/09/08
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# instance: 	AILab_MLC_QYB6E7811834834997836188CEB55F
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#   --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"'


# # # 2023/09/08
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# instance: AILab_MLC_QY6C5AC745FA594CE48D4141A63F94
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#   --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"'


# # # 2023/09/08
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# # instance: AILab_MLC_QY4FC3412E403F429385151EF504F5
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#   --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"'


# # # 2023/09/08
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# # instance: AILab_MLC_QYA7F78B77EEEA439EA2DBC74D782F
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/QR_DQN_1000it_correct \
#   --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"'


# # # 2023/09/08
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# # instance: AILab_MLC_QY49428858270842229F07EBA21947
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/QR_DQN_1000it_correct \
#   --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"'


# # # 2023/09/08
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# instance: AILab_MLC_QYDD54FAD9D13B41709FBE414558DA
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/QR_DQN_1000it_correct \
#   --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"'


# # # 2023/09/08
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# instance: AILab_MLC_QYFE4C244EDEAE49EA8E1941042C99
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/QR_DQN_1000it_correct \
#   --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"'


# # # 2023/09/08
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# # instance: AILab_MLC_QY633272D3FB4844768D034EEC5078
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/QR_DQN_1000it_correct \
#   --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"'


# # # # 2023/09/10
# # # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # # action dim: 6
# # # instance: AILab_MLC_QY633272D3FB4844768D034EEC5078
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/REM_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --agent_name=multi_head_dqn \
#  --gin_files='batch_rl/fixed_replay/configs/rem.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.network="atari_helpers.MultiHeadQConFcLayerNormNetwork"' 


# # # # 2023/09/10
# # # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # # action dim: 6
# # # instance: AILab_MLC_QY633272D3FB4844768D034EEC5078
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/REM_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --agent_name=multi_head_dqn \
#  --gin_files='batch_rl/fixed_replay/configs/rem.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.network="atari_helpers.MultiHeadQConFcLayerNormNetwork"' 


# # # # 2023/09/10
# # # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # # action dim: 6
# # # instance: AILab_MLC_QY633272D3FB4844768D034EEC5078
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/REM_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --agent_name=multi_head_dqn \
#  --gin_files='batch_rl/fixed_replay/configs/rem.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.network="atari_helpers.MultiHeadQConFcLayerNormNetwork"' 


# # # # 2023/09/10
# # # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # # action dim: 6
# # # instance: AILab_MLC_QY633272D3FB4844768D034EEC5078
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/REM_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --agent_name=multi_head_dqn \
#  --gin_files='batch_rl/fixed_replay/configs/rem.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.network="atari_helpers.MultiHeadQConFcLayerNormNetwork"' 


# # # # 2023/09/10
# # # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # # action dim: 6
# # # instance: AILab_MLC_QY633272D3FB4844768D034EEC5078
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/REM_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --agent_name=multi_head_dqn \
#  --gin_files='batch_rl/fixed_replay/configs/rem.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.network="atari_helpers.MultiHeadQConFcLayerNormNetwork"' 


# # # 2023/09/13
# # # baseline: CQL, QuantileConFcLayerNormDropoutNetwork
# # # action dim: 6
# # instance: AILab_MLC_QY039FA461C26F462A839DFCFCD57B
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLnDropout_1000it_correct \
#   --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormDropoutNetwork"'


# # # 2023/09/13
# # # baseline: CQL, QuantileConFcLayerNormDropoutNetwork
# # # action dim: 6
# instance: AILab_MLC_QYC6AF1C9AF49043289E6EA7B6A26B
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLnDropout_1000it_correct \
#   --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormDropoutNetwork"'


# # # 2023/09/13
# # # baseline: CQL, QuantileConFcLayerNormDropoutNetwork
# # # action dim: 6
# # instance: AILab_MLC_QY4E8E9906885146489C872319483F
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLnDropout_1000it_correct \
#   --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormDropoutNetwork"'


# # # 2023/09/13
# # # baseline: CQL, QuantileConFcLayerNormDropoutNetwork
# # # action dim: 6
# # instance: AILab_MLC_QY606FFF013AED4CE598330808BBF7
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLnDropout_1000it_correct \
#   --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormDropoutNetwork"'


# # # 2023/09/13
# # # baseline: CQL, QuantileConFcLayerNormDropoutNetwork
# # # action dim: 6
# # instance: AILab_MLC_QYAD9BE103FA964C1E93BC228D8FA4
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLnDropout_1000it_correct \
#   --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormDropoutNetwork"'


# # 2023/08/03
# # baseline: CQL, QuantileLayerNormNetwork
# # Asterix action dim: 9
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=9' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait

# # 2023/08/03
# # baseline: CQL, QuantileLayerNormNetwork
# # Asterix action dim: 9
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=9' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # 2023/08/03
# # baseline: CQL, QuantileLayerNormNetwork
# # Asterix action dim: 9
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl3.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=9' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=3.0'

# wait

# # 2023/08/03
# # baseline: CQL, QuantileLayerNormNetwork
# # Asterix action dim: 9
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl4.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=9' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=4.0'

# wait

# # 2023/08/03
# # baseline: CQL, QuantileLayerNormNetwork
# # Asterix action dim: 9
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl5.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=9' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=5.0'

# wait

# # 2023/08/03
# # baseline: CQL, QuantileLayerNormNetwork
# # Asterix action dim: 9
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl1.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=9' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait

# # 2023/08/03
# # baseline: CQL, QuantileLayerNormNetwork
# # Asterix action dim: 9
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl1.5"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=9' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # 2023/08/03
# # baseline: CQL, QuantileLayerNormNetwork
# # Asterix action dim: 9
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl2.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=9' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # 2023/08/03
# # baseline: CQL, QuantileLayerNormNetwork
# # Breakout action dim: 4
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=4' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait

# # 2023/08/03
# # baseline: CQL, QuantileLayerNormNetwork
# # Breakout action dim: 4
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=4' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # 2023/08/03
# # baseline: CQL, QuantileLayerNormNetwork
# # Breakout action dim: 4
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl3.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=4' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=3.0'

# wait

# # 2023/08/03
# # baseline: CQL, QuantileLayerNormNetwork
# # Breakout action dim: 4
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl4.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=4' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=4.0'

# wait

# # 2023/08/03
# # baseline: CQL, QuantileLayerNormNetwork
# # Breakout action dim: 4
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl5.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=4' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=5.0'

# wait

# # 2023/08/03
# # baseline: CQL, QuantileLayerNormNetwork
# # Breakout action dim: 4
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl1.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=4' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait

# # 2023/08/03
# # baseline: CQL, QuantileLayerNormNetwork
# # Breakout action dim: 4
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl1.5"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=4' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # 2023/08/03
# # baseline: CQL, QuantileLayerNormNetwork
# # Breakout action dim: 4
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl2.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=4' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # 2023/08/03
# # baseline: CQL, QuantileLayerNormNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait

# # 2023/08/03
# # baseline: CQL, QuantileLayerNormNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # 2023/08/03
# # baseline: CQL, QuantileLayerNormNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl3.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=3.0'

# wait

# # 2023/08/03
# # baseline: CQL, QuantileLayerNormNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl4.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=4.0'

# wait

# # 2023/08/03
# # baseline: CQL, QuantileLayerNormNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl5.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=5.0'

# wait

# # 2023/08/03
# # baseline: CQL, QuantileLayerNormNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl1.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait

# # 2023/08/03
# # baseline: CQL, QuantileLayerNormNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl1.5"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # 2023/08/03
# # baseline: CQL, QuantileLayerNormNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl2.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # 2023/08/03
# # baseline: CQL, QuantileLayerNormNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait

# # 2023/08/03
# # baseline: CQL, QuantileLayerNormNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # 2023/08/03
# # baseline: CQL, QuantileLayerNormNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl3.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=3.0'

# wait

# # 2023/08/03
# # baseline: CQL, QuantileLayerNormNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl4.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=4.0'

# wait

# # 2023/08/03
# # baseline: CQL, QuantileLayerNormNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl5.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=5.0'

# wait

# # 2023/08/03
# # baseline: CQL, QuantileLayerNormNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl1.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait

# # 2023/08/03
# # baseline: CQL, QuantileLayerNormNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl1.5"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # 2023/08/03
# # baseline: CQL, QuantileLayerNormNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl2.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # # 2023/08/03
# # # baseline: CQL, QuantileLayerNormNetwork
# # # Seaquest action dim: 18
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl2.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=18' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# # # 2023/08/03
# # # baseline: CQL, QuantileLayerNormNetwork
# # # Seaquest action dim: 18
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=18' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # # 2023/08/03
# # # baseline: CQL, QuantileLayerNormNetwork
# # # Seaquest action dim: 18
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl5.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=18' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=5.0'

# wait 

# # # 2023/08/03
# # # baseline: CQL, QuantileLayerNormNetwork
# # # Seaquest action dim: 18
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl3.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=18' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=3.0'

# wait 

# # # 2023/08/03
# # # baseline: CQL, QuantileLayerNormNetwork
# # # Seaquest action dim: 18
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl4.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=18' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=4.0'

# wait 

# # # 2023/08/03
# # # baseline: CQL, QuantileLayerNormNetwork
# # # Seaquest action dim: 18
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=18' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait 

# # # 2023/08/03
# # # baseline: CQL, QuantileLayerNormNetwork
# # # Seaquest action dim: 18
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl1.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=18' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait

# # # 2023/08/03
# # # baseline: CQL, QuantileLayerNormNetwork
# # # Seaquest action dim: 18
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl1.5"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=18' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# # # 2023/08/03
# # # baseline: CQL, QuantileLayerNormNetwork
# # # Seaquest action dim: 18
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/QR_DQN_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=18' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'


# # 2023/08/03
# # baseline: CQL, QuantileLayerNormNetwork
# # Qbert action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/QR_DQN_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'


# # 2023/08/03
# # baseline: CQL, QuantileLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/QR_DQN_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# # 2023/08/03
# # baseline: CQL, QuantileLayerNormNetwork
# # Breakout action dim: 4
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/QR_DQN_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=4' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# # 2023/08/03
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Asterix action dim: 9
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/QR_DQN_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=9' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'


# # 2023/08/03
# # baseline: CQL, QuantileLayerNormNetwork
# # Qbert action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/QR_DQN_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'


# # 2023/09/24
# # baseline: CQL, QuantileLayerNormNetwork
# # Qbert action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/QR_DQN_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl1.5"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'


# # 2023/09/24
# # baseline: CQL, QuantileLayerNormNetwork
# # Qbert action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/QR_DQN_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl5.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=5.0'


# # # # 2023/09/24
# # # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # # Qbert action dim: 6
# # # instance: AILab_MLC_QY633272D3FB4844768D034EEC5078
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/REM_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --agent_name=multi_head_dqn \
#  --gin_files='batch_rl/fixed_replay/configs/rem.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.network="atari_helpers.MultiHeadQConFcLayerNormNetwork"'  \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # # # 2023/09/24
# # # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # # Qbert action dim: 6
# # # instance: AILab_MLC_QY633272D3FB4844768D034EEC5078
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/REM_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --agent_name=multi_head_dqn \
#  --gin_files='batch_rl/fixed_replay/configs/rem.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.network="atari_helpers.MultiHeadQConFcLayerNormNetwork"'  \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait

# # # # 2023/09/24
# # # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # # Qbert action dim: 6
# # # instance: AILab_MLC_QY633272D3FB4844768D034EEC5078
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/REM_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --agent_name=multi_head_dqn \
#  --gin_files='batch_rl/fixed_replay/configs/rem.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.network="atari_helpers.MultiHeadQConFcLayerNormNetwork"'  \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # # # 2023/09/24
# # # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # # Qbert action dim: 6
# # # instance: AILab_MLC_QY633272D3FB4844768D034EEC5078
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/REM_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --agent_name=multi_head_dqn \
#  --gin_files='batch_rl/fixed_replay/configs/rem.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl2.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.network="atari_helpers.MultiHeadQConFcLayerNormNetwork"'  \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # # # 2023/09/24
# # # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # # Qbert action dim: 6
# # # instance: AILab_MLC_QY633272D3FB4844768D034EEC5078
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/REM_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --agent_name=multi_head_dqn \
#  --gin_files='batch_rl/fixed_replay/configs/rem.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl1.5"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.network="atari_helpers.MultiHeadQConFcLayerNormNetwork"'  \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # # # 2023/09/24
# # # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # # Qbert action dim: 6
# # # instance: AILab_MLC_QY633272D3FB4844768D034EEC5078
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/REM_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --agent_name=multi_head_dqn \
#  --gin_files='batch_rl/fixed_replay/configs/rem.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl1.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.network="atari_helpers.MultiHeadQConFcLayerNormNetwork"'  \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait

# # # # 2023/09/24
# # # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # # Pong action dim: 6
# # # instance: AILab_MLC_QY633272D3FB4844768D034EEC5078
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/REM_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --agent_name=multi_head_dqn \
#  --gin_files='batch_rl/fixed_replay/configs/rem.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.network="atari_helpers.MultiHeadQConFcLayerNormNetwork"'  \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # # # 2023/09/24
# # # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # # Pong action dim: 6
# # # instance: AILab_MLC_QY633272D3FB4844768D034EEC5078
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/REM_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --agent_name=multi_head_dqn \
#  --gin_files='batch_rl/fixed_replay/configs/rem.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.network="atari_helpers.MultiHeadQConFcLayerNormNetwork"'  \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait

# # # # 2023/09/24
# # # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # # Pong action dim: 6
# # # instance: AILab_MLC_QY633272D3FB4844768D034EEC5078
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/REM_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --agent_name=multi_head_dqn \
#  --gin_files='batch_rl/fixed_replay/configs/rem.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.network="atari_helpers.MultiHeadQConFcLayerNormNetwork"'  \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # # # 2023/09/24
# # # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # # Pong action dim: 6
# # # instance: AILab_MLC_QY633272D3FB4844768D034EEC5078
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/REM_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --agent_name=multi_head_dqn \
#  --gin_files='batch_rl/fixed_replay/configs/rem.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl1.5"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.network="atari_helpers.MultiHeadQConFcLayerNormNetwork"'  \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # # # 2023/09/24
# # # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # # Pong action dim: 6
# # # instance: AILab_MLC_QY633272D3FB4844768D034EEC5078
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/REM_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --agent_name=multi_head_dqn \
#  --gin_files='batch_rl/fixed_replay/configs/rem.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl1.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.network="atari_helpers.MultiHeadQConFcLayerNormNetwork"'  \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait

# # # # 2023/09/24
# # # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # # Pong action dim: 6
# # # instance: AILab_MLC_QY633272D3FB4844768D034EEC5078
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/REM_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --agent_name=multi_head_dqn \
#  --gin_files='batch_rl/fixed_replay/configs/rem.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl2.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.network="atari_helpers.MultiHeadQConFcLayerNormNetwork"'  \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # # # 2023/09/24
# # # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # # Breakout action dim: 4
# # # instance: AILab_MLC_QY633272D3FB4844768D034EEC5078
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/REM_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --agent_name=multi_head_dqn \
#  --gin_files='batch_rl/fixed_replay/configs/rem.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.network="atari_helpers.MultiHeadQConFcLayerNormNetwork"'  \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=4' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait

# # # # 2023/09/24
# # # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # # Breakout action dim: 4
# # # instance: AILab_MLC_QY633272D3FB4844768D034EEC5078
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/REM_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --agent_name=multi_head_dqn \
#  --gin_files='batch_rl/fixed_replay/configs/rem.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.network="atari_helpers.MultiHeadQConFcLayerNormNetwork"'  \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=4' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # # # 2023/09/24
# # # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # # Breakout action dim: 4
# # # instance: AILab_MLC_QY633272D3FB4844768D034EEC5078
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/REM_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --agent_name=multi_head_dqn \
#  --gin_files='batch_rl/fixed_replay/configs/rem.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.network="atari_helpers.MultiHeadQConFcLayerNormNetwork"'  \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=4' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # # # 2023/09/24
# # # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # # Breakout action dim: 4
# # # instance: AILab_MLC_QY633272D3FB4844768D034EEC5078
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/REM_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --agent_name=multi_head_dqn \
#  --gin_files='batch_rl/fixed_replay/configs/rem.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl1.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.network="atari_helpers.MultiHeadQConFcLayerNormNetwork"'  \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=4' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait

# # # # 2023/09/24
# # # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # # Breakout action dim: 4
# # # instance: AILab_MLC_QY633272D3FB4844768D034EEC5078
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/REM_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --agent_name=multi_head_dqn \
#  --gin_files='batch_rl/fixed_replay/configs/rem.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl1.5"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.network="atari_helpers.MultiHeadQConFcLayerNormNetwork"'  \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=4' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # # # 2023/09/24
# # # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # # Breakout action dim: 4
# # # instance: AILab_MLC_QY633272D3FB4844768D034EEC5078
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/REM_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --agent_name=multi_head_dqn \
#  --gin_files='batch_rl/fixed_replay/configs/rem.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl2.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.network="atari_helpers.MultiHeadQConFcLayerNormNetwork"'  \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=4' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # # # 2023/09/24
# # # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # # Asterix action dim: 9
# # # instance: AILab_MLC_QY633272D3FB4844768D034EEC5078
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/REM_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --agent_name=multi_head_dqn \
#  --gin_files='batch_rl/fixed_replay/configs/rem.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.network="atari_helpers.MultiHeadQConFcLayerNormNetwork"'  \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=9' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait

# # # # 2023/09/24
# # # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # # Asterix action dim: 9
# # # instance: AILab_MLC_QY633272D3FB4844768D034EEC5078
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/REM_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --agent_name=multi_head_dqn \
#  --gin_files='batch_rl/fixed_replay/configs/rem.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.network="atari_helpers.MultiHeadQConFcLayerNormNetwork"'  \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=9' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # # # 2023/09/24
# # # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # # Asterix action dim: 9
# # # instance: AILab_MLC_QY633272D3FB4844768D034EEC5078
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/REM_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --agent_name=multi_head_dqn \
#  --gin_files='batch_rl/fixed_replay/configs/rem.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.network="atari_helpers.MultiHeadQConFcLayerNormNetwork"'  \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=9' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef2.0'

# wait

# # # # 2023/09/24
# # # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # # Asterix action dim: 9
# # # instance: AILab_MLC_QY633272D3FB4844768D034EEC5078
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/REM_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --agent_name=multi_head_dqn \
#  --gin_files='batch_rl/fixed_replay/configs/rem.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl1.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.network="atari_helpers.MultiHeadQConFcLayerNormNetwork"'  \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=9' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait

# # # # 2023/09/24
# # # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # # Asterix action dim: 9
# # # instance: AILab_MLC_QY633272D3FB4844768D034EEC5078
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/REM_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --agent_name=multi_head_dqn \
#  --gin_files='batch_rl/fixed_replay/configs/rem.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl1.5"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.network="atari_helpers.MultiHeadQConFcLayerNormNetwork"'  \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=9' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # # # 2023/09/24
# # # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # # Asterix action dim: 9
# # # instance: AILab_MLC_QY633272D3FB4844768D034EEC5078
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/REM_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --agent_name=multi_head_dqn \
#  --gin_files='batch_rl/fixed_replay/configs/rem.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl2.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.network="atari_helpers.MultiHeadQConFcLayerNormNetwork"'  \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=9' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef2.0'

# wait

# # # # 2023/09/24
# # # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # # Seaquest action dim: 18
# # # instance: AILab_MLC_QY633272D3FB4844768D034EEC5078
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/REM_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --agent_name=multi_head_dqn \
#  --gin_files='batch_rl/fixed_replay/configs/rem.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.network="atari_helpers.MultiHeadQConFcLayerNormNetwork"'  \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=18' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait

# # # # 2023/09/24
# # # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # # Seaquest action dim: 18
# # # instance: AILab_MLC_QY633272D3FB4844768D034EEC5078
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/REM_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --agent_name=multi_head_dqn \
#  --gin_files='batch_rl/fixed_replay/configs/rem.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.network="atari_helpers.MultiHeadQConFcLayerNormNetwork"'  \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=18' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # # # 2023/09/24
# # # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # # Seaquest action dim: 18
# # # instance: AILab_MLC_QY633272D3FB4844768D034EEC5078
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/REM_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --agent_name=multi_head_dqn \
#  --gin_files='batch_rl/fixed_replay/configs/rem.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.network="atari_helpers.MultiHeadQConFcLayerNormNetwork"'  \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=18' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait
 
# # # # 2023/09/24
# # # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # # Seaquest action dim: 18
# # # instance: AILab_MLC_QY633272D3FB4844768D034EEC5078
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/REM_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --agent_name=multi_head_dqn \
#  --gin_files='batch_rl/fixed_replay/configs/rem.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl1.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.network="atari_helpers.MultiHeadQConFcLayerNormNetwork"'  \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=18' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait

# # # # 2023/09/24
# # # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # # Seaquest action dim: 18
# # # instance: AILab_MLC_QY633272D3FB4844768D034EEC5078
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/REM_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --agent_name=multi_head_dqn \
#  --gin_files='batch_rl/fixed_replay/configs/rem.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl1.5"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.network="atari_helpers.MultiHeadQConFcLayerNormNetwork"'  \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=18' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # # # 2023/09/24
# # # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # # Seaquest action dim: 18
# # # instance: AILab_MLC_QY633272D3FB4844768D034EEC5078
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/REM_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --agent_name=multi_head_dqn \
#  --gin_files='batch_rl/fixed_replay/configs/rem.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl2.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.network="atari_helpers.MultiHeadQConFcLayerNormNetwork"'  \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=18' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait


# # # 2023/08/03
# # # baseline: CQL, QuantileLayerNormNetwork
# # # Seaquest action dim: 18
# # # # instance: AILab_MLC_QY58346FD9498644D99027FBF2C0AC
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5_lr1e-6"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=18' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # # 2023/08/03
# # # baseline: CQL, QuantileLayerNormNetwork
# # # Asterix action dim: 9
# # # # instance: AILab_MLC_QY58346FD9498644D99027FBF2C0AC
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5_lr1e-6"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=9' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # # 2023/08/03
# # # baseline: CQL, QuantileLayerNormNetwork
# # # Pong action dim: 6
# # # # instance: AILab_MLC_QY58346FD9498644D99027FBF2C0AC
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5_lr1e-6"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # # 2023/08/03
# # # baseline: CQL, QuantileLayerNormNetwork
# # # Qbert action dim: 6
# # # # instance: AILab_MLC_QY58346FD9498644D99027FBF2C0AC
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5_lr1e-6"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # # 2023/08/03
# # # baseline: CQL, QuantileLayerNormNetwork
# # # Breakout action dim: 4
# # # # instance: AILab_MLC_QY58346FD9498644D99027FBF2C0AC
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5_lr1e-6"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=4' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# ############################################################################################################################
# # instance: AILab_MLC_QYDC9994AFE44E4E3E83E721B563EB 
# # time: 2023-09-25 16:59:28
# ############################################################################################################################
# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Breakout action dim: 4
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.0_lr1e-6"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=4' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Breakout action dim: 4
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl0.5_lr1e-6"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=4' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.5'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Qbert action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl0.5_lr1e-6"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.5'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Qbert action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.0_lr1e-6"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl0.5_lr1e-6"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.5'

# wait


# ############################################################################################################################
# # # instance: AILab_MLC_QYEA891B36319847E1B330671EB550 
# # # time: 	2023-09-25 17:02:10
# ############################################################################################################################
# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.0_lr1e-6"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Seaquest action dim: 18
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl0.5_lr1e-6"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=18' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.5'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Seaquest action dim: 18
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.0_lr1e-6"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=18' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Asterix action dim: 9
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl0.5_lr1e-6"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=9' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.5'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Asterix action dim: 9
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.0_lr1e-6"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=9' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait


# ############################################################################################################################
# # # instance: 	AILab_MLC_QY17FB4AE947314113B62B7DAE63DA 
# # # time: 2023-09-25 17:09:08
# ############################################################################################################################
# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.0_lr5e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl0.5_lr5e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.5'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5_lr5e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.0_lr5e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.5_lr5e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.5'

# wait

# ############################################################################################################################
# # # instance: AILab_MLC_QY388D89647DEB4CCFB1C4E8CE9476 
# # # time: 2023-09-25 17:11:44
# ###########################################################################################################################
# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Asterix action dim: 9
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.0_lr5e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=9' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Asterix action dim: 9
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.5_lr5e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=9' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.5'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Asterix action dim: 9
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl0.5_lr5e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=9' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.5'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Asterix action dim: 9
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5_lr5e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=9' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Asterix action dim: 9
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.0_lr5e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=9' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait

# ############################################################################################################################
# # # instance: 	AILab_MLC_QY5B9197BD455142E98DA1303747B2 
# # # time: 2023-09-25 17:15:06
# ###########################################################################################################################
# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Seaquest action dim: 18
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.0_lr5e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=18' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Seaquest action dim: 18
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.5_lr5e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=18' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.5'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Seaquest action dim: 18
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5_lr5e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=18' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Seaquest action dim: 18
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.0_lr5e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=18' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Seaquest action dim: 18
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl0.5_lr5e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=18' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.5'

# wait

# ############################################################################################################################
# # # instance: 	AILab_MLC_QY0B6B28312D64417287C1A60CB2D2 
# # # time: 2023-09-25 17:17:54
# ###########################################################################################################################
# # # Qbert action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.0_lr5e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # # Qbert action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.5_lr5e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.5'

# wait

# # # Qbert action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5_lr5e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # # Qbert action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl0.5_lr5e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.5'

# wait

# # # Qbert action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.0_lr5e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait

# # ############################################################################################################################
# # # # instance: 	AILab_MLC_QYF4ED7B95F1D84141AD98F31724DC 
# # # # time: 2023-09-25 17:20:34
# # ###########################################################################################################################
# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Breakout action dim: 4
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl0.5_lr5e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=4' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.5'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Breakout action dim: 4
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5_lr5e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=4' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Breakout action dim: 4
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.5_lr5e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=4' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.5'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Breakout action dim: 4
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.0_lr5e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=4' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Breakout action dim: 4
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.0_lr5e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=4' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait

# # ############################################################################################################################
# # # # instance: 	AILab_MLC_QY0216B42E944D4D11910089EE8F28 
# # # # time: 2023-09-25 18:54:12
# # ###########################################################################################################################
# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Breakout action dim: 4
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl0.5_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=4' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.5'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Breakout action dim: 4
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=4' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Breakout action dim: 4
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.5_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=4' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.5'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Breakout action dim: 4
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.0_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=4' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Breakout action dim: 4
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.0_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=4' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait

# ############################################################################################################################
# # # instance: 	AILab_MLC_QYD83B0E85856C4255B98B9944BADF 
# # # time: 	2023-09-25 18:57:39
# ###########################################################################################################################
# # # Qbert action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.0_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # # Qbert action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.5_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.5'

# wait

# # # Qbert action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # # Qbert action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl0.5_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.5'

# wait

# # # Qbert action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.0_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait

# ############################################################################################################################
# # # instance: 	AILab_MLC_QYF4481FA6B69F42398760F3B22F93 
# # # time: 2023-09-25 19:00:19
# ###########################################################################################################################
# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Seaquest action dim: 18
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.0_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=18' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Seaquest action dim: 18
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.5_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=18' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.5'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Seaquest action dim: 18
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=18' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Seaquest action dim: 18
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.0_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=18' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Seaquest action dim: 18
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl0.5_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=18' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.5'

# wait

# ############################################################################################################################
# # # instance: AILab_MLC_QY4781341D0C3D44708E41002F79AF 
# # # time: 2023-09-25 19:06:48
# ###########################################################################################################################
# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Asterix action dim: 9
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.0_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=9' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Asterix action dim: 9
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.5_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=9' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.5'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Asterix action dim: 9
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl0.5_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=9' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.5'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Asterix action dim: 9
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=9' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Asterix action dim: 9
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.0_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=9' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait


# ############################################################################################################################
# # # instance: 	AILab_MLC_QY51D3AF15C9F0405BB79AFBCA06EC 
# # # time: 	2023-09-25 19:08:02
# ############################################################################################################################
# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.0_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl0.5_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.5'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.0_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.5_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.5'

# wait

# ############################################################################################################################
# # # instance: 	AILab_MLC_QY6C81D6909BF64F3E82E7C9CBB9C8 
# # # time: 	2023-09-26 00:56:49
# ############################################################################################################################
# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl1.0_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl0.5_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=0.5'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl1.5_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl2.0_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl2.5_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=2.5'

# wait

# ############################################################################################################################
# # # instance: 	AILab_MLC_QY9755F45E09EA4FDD9AE4D702E0EC 
# # # time: 	2023-09-26 00:59:01
# ###########################################################################################################################
# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Asterix action dim: 9
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl2.0_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=9' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Asterix action dim: 9
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl2.5_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=9' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=2.5'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Asterix action dim: 9
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl0.5_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=9' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=0.5'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Asterix action dim: 9
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl1.5_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=9' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Asterix action dim: 9
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl1.0_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=9' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait


# ############################################################################################################################
# # # instance: 	AILab_MLC_QYE520063BF79244BEBF5EBAD67AB9 
# # # time: 2023-09-26 01:01:19
# ###########################################################################################################################
# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Seaquest action dim: 18
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl2.0_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=18' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Seaquest action dim: 18
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl2.5_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=18' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=2.5'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Seaquest action dim: 18
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl1.5_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=18' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Seaquest action dim: 18
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl1.0_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=18' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Seaquest action dim: 18
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl0.5_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=18' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=0.5'

# wait

# ############################################################################################################################
# # # instance: 	AILab_MLC_QYE520063BF79244BEBF5EBAD67AB9 
# # # time: 	2023-09-26 01:01:19
# ###########################################################################################################################
# # # Qbert action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl2.0_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # # Qbert action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl2.5_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=2.5'

# wait

# # # Qbert action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl1.5_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # # Qbert action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl0.5_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=0.5'

# wait

# # # Qbert action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl1.0_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait

# # ############################################################################################################################
# # # # instance: 		AILab_MLC_QY6E5C6398C5B141A98E0DA3D1C818 
# # # # time: 2023-09-26 01:03:19
# # ###########################################################################################################################
# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Breakout action dim: 4
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl0.5_lr5e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=4' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.5'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Breakout action dim: 4
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5_lr5e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=4' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Breakout action dim: 4
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.5_lr5e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=4' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.5'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Breakout action dim: 4
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.0_lr5e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=4' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Breakout action dim: 4
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.0_lr5e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=4' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait

# # ############################################################################################################################
# # # # instance: 	AILab_MLC_QY391CC6C8FB0F46C1AE35E44E9C72 
# # # # time: 2023-09-26 01:05:12
# # ###########################################################################################################################
# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Breakout action dim: 4
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl0.5_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=4' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=0.5'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Breakout action dim: 4
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl1.5_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=4' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Breakout action dim: 4
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl2.5_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=4' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=2.5'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Breakout action dim: 4
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl2.0_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=4' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Breakout action dim: 4
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl1.0_lr1e-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=4' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait


# ############################################################################################################################
# # # instance: 	AILab_MLC_QYE520063BF79244BEBF5EBAD67AB9 
# # # time: 	2023-09-26 01:01:19
# ###########################################################################################################################
# # # Qbert action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl2.0_lr5e-8"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # # Qbert action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl2.5_lr5e-8"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=2.5'

# wait

# # # Qbert action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl1.5_lr5e-8"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # # Qbert action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl0.5_lr5e-8"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=0.5'

# wait

# # # Qbert action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl1.0_lr5e-8"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait

# ############################################################################################################################
# # # instance: 	AILab_MLC_QYE520063BF79244BEBF5EBAD67AB9 
# # # time: 	2023-09-26 01:01:19
# ###########################################################################################################################
# # # Qbert action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl2.0_lr1e-8"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # # Qbert action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl2.5_lr1e-8"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=2.5'

# wait

# # # Qbert action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl1.5_lr1e-8"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # # Qbert action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl0.5_lr1e-8"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=0.5'

# wait

# # # Qbert action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl1.0_lr1e-8"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait

# ##########################################################################################
# # instance: AILab_MLC_QYCD1158EE18C84CAA85058547D639 date: 2023-09-28 02:46:34
# ##########################################################################################
# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl0.0_le-5"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0'

# wait


# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.3_kl0.0_le-5"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.3' \
#  --gin_bindings='DQNAgent.kl_coef=0.0'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.5_kl0.0_le-5"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.5' \
#  --gin_bindings='DQNAgent.kl_coef=0.0'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.7_kl0.0_le-5"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.7' \
#  --gin_bindings='DQNAgent.kl_coef=0.0'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.9_kl0.0_le-5"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.9' \
#  --gin_bindings='DQNAgent.kl_coef=0.0'

# wait

# ##########################################################################################
# # instance: AILab_MLC_QYA5E3A30DBDBF44C3A462BFE5414E date: 	2023-09-28 02:47:30
# ##########################################################################################
# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl0.0_le-5"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=0.0'

# wait


# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.03_kl0.0_le-5"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.03' \
#  --gin_bindings='DQNAgent.kl_coef=0.0'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.05_kl0.0_le-5"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.05' \
#  --gin_bindings='DQNAgent.kl_coef=0.0'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.07_kl0.0_le-5"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.07' \
#  --gin_bindings='DQNAgent.kl_coef=0.0'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.09_kl0.0_le-5"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.09' \
#  --gin_bindings='DQNAgent.kl_coef=0.0'

# wait

# ##########################################################################################
# # instance: AILab_MLC_QYC244308DBAF04E81B4652954CF04 date: 2023-09-28 02:48:36
# ##########################################################################################
# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl0.0_le-6"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0'

# wait


# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.3_kl0.0_le-6"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.3' \
#  --gin_bindings='DQNAgent.kl_coef=0.0'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.5_kl0.0_le-6"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.5' \
#  --gin_bindings='DQNAgent.kl_coef=0.0'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.7_kl0.0_le-6"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.7' \
#  --gin_bindings='DQNAgent.kl_coef=0.0'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.9_kl0.0_le-6"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.9' \
#  --gin_bindings='DQNAgent.kl_coef=0.0'

# wait

# ##########################################################################################
# # instance: AILab_MLC_QYF01229271B484038922593046D88 date: 2023-09-28 02:49:24
# ##########################################################################################
# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl0.0_le-6"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=0.0'

# wait


# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.03_kl0.0_le-6"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.03' \
#  --gin_bindings='DQNAgent.kl_coef=0.0'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.05_kl0.0_le-6"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.05' \
#  --gin_bindings='DQNAgent.kl_coef=0.0'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.07_kl0.0_le-6"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.07' \
#  --gin_bindings='DQNAgent.kl_coef=0.0'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.09_kl0.0_le-6"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.09' \
#  --gin_bindings='DQNAgent.kl_coef=0.0'

# wait

# ##########################################################################################
# # instance: AILab_MLC_QYD62FAEF7169C4678992BAAEAF11C date: 2023-09-28 02:51:31
# ##########################################################################################
# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01_kl0.0_le-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.01' \
#  --gin_bindings='DQNAgent.kl_coef=0.0'

# wait


# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.03_kl0.0_le-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.03' \
#  --gin_bindings='DQNAgent.kl_coef=0.0'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.05_kl0.0_le-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.05' \
#  --gin_bindings='DQNAgent.kl_coef=0.0'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.07_kl0.0_le-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.07' \
#  --gin_bindings='DQNAgent.kl_coef=0.0'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.09_kl0.0_le-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.09' \
#  --gin_bindings='DQNAgent.kl_coef=0.0'

# wait

# ##########################################################################################
# # instance: AILab_MLC_QY2FD0D409172D417BADB29C1F1563 date: 2023-09-28 02:53:39
# ##########################################################################################
# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl0.0_le-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0'

# wait


# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.3_kl0.0_le-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.3' \
#  --gin_bindings='DQNAgent.kl_coef=0.0'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.5_kl0.0_le-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.5' \
#  --gin_bindings='DQNAgent.kl_coef=0.0'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.7_kl0.0_le-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.7' \
#  --gin_bindings='DQNAgent.kl_coef=0.0'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.9_kl0.0_le-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.9' \
#  --gin_bindings='DQNAgent.kl_coef=0.0'

# wait


# # ##########################################################################################
# # # instance: AILab_MLC_QY80BDB294A5A347DE870A8FAD8841 date: 2023-09-28 20:58:41
# # ##########################################################################################
# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy1.0_kl0.0_le-7"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=1.0' \
#  --gin_bindings='DQNAgent.kl_coef=0.0'

# wait

# # ##########################################################################################
# # # instance: AILab_MLC_QY80BDB294A5A347DE870A8FAD8841 date: 2023-09-28 20:58:41
# # ##########################################################################################
# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy1.0_kl0.0_3e-4"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=1.0' \
#  --gin_bindings='DQNAgent.kl_coef=0.0'

# wait

# # ##########################################################################################
# # # instance: AILab_MLC_QY29802580604E47B298F49BCC9DD7 date: 2023-09-28 21:13:33
# # ##########################################################################################
# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl0.5_1e-5_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.5'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.0_1e-5_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5_1e-5_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.0_1e-5_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.5_1e-5_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.5'

# wait

# # ##########################################################################################
# # # instance: AILab_MLC_QY5D116B803C7A4099A5F49FDBEA6F date: 2023-09-28 21:15:08
# # ##########################################################################################
# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl0.5_1e-6_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.5'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.0_1e-6_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5_1e-6_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.0_1e-6_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.5_1e-6_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.5'

# wait

# # ##########################################################################################
# # # instance: AILab_MLC_QYFDD77D55C9144812B13AD1A92654 date: 2023-09-28 21:16:34
# # ##########################################################################################
# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl0.5_1e-7_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.5'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.0_1e-7_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5_1e-7_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.0_1e-7_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.5_1e-7_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.5'

# wait

# # ##########################################################################################
# # # instance: AILab_MLC_QYC164E08C245E4F62880DDB9E0D44 date: 2023-09-28 21:17:44
# # ##########################################################################################
# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl0.5_1e-8_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.5'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.0_1e-8_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5_1e-8_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.0_1e-8_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.5_1e-8_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.5'

# wait

# # ##########################################################################################
# # # instance: AILab_MLC_QYC164E08C245E4F62880DDB9E0D44 date: 2023-09-28 21:17:44
# # ##########################################################################################
# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl0.5_1e-8_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.5'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.0_1e-8_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5_1e-8_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.0_1e-8_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.5_1e-8_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.5'

# wait

# # ##########################################################################################
# # # instance: AILab_MLC_QY611BBED50F34429DB15F1063B843 date: 	2023-09-28 21:21:02
# # ##########################################################################################
# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl0.5_5e-8_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.5'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.0_5e-8_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5_5e-8_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.0_5e-8_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.5_5e-8_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.5'

# wait

# # ##########################################################################################
# # # instance:  AILab_MLC_QY99181C3E635D4B22B314F1D88ACA  date: 	2023-09-28 21:21:56
# # ##########################################################################################
# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl0.5_5e-7_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.5'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.0_5e-7_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5_5e-7_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.0_5e-7_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.5_5e-7_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.5'

# wait

# # ##########################################################################################
# # # instance:  AILab_MLC_QY4CBC7181172743A3A4A1BDF6DBC1  date: 	2023-09-28 21:23:19
# # ##########################################################################################
# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl0.5_5e-6_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.5'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.0_5e-6_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5_5e-6_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.0_5e-6_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.5_5e-6_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.5'

# wait

# # ##########################################################################################
# # # instance:  AILab_MLC_QYD1DCE8788D9146718F58EB2C214E  date: 		2023-09-28 21:24:11
# # ##########################################################################################
# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl0.5_5e-5_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.5'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.0_5e-5_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5_5e-5_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.0_5e-5_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.5_5e-5_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.5'

# wait

# instance :AILab_MLC_QY86F20D10CA9E431A9E19DFE25262
# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy1.0_kl0.0_5e-6"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=1.0' \
#  --gin_bindings='DQNAgent.kl_coef=0.0'

# wait

# # ##########################################################################################
# # # instance:  AILab_MLC_QYBB93176369E54794925FBDE15801  date: 		2023-09-28 21:24:11
# # ##########################################################################################
# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl0.5_1e-4_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.5'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.0_1e-4_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5_1e-4_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.0_1e-4_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.5_1e-4_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.5'

# wait

# # ##########################################################################################
# # # instance:  AILab_MLC_QYDA8CC0A5512641E48BC5A869B169  date: 		2023-09-28 21:24:11
# # ##########################################################################################
# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl0.5_3e-4_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.5'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.0_3e-4_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.0'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5_3e-4_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.0_3e-4_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # 2023/09/28
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Pong action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.5_3e-4_fixed"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.5'

# wait

# ############################################################################################################################
# # # instance: AILab_MLC_QY4781341D0C3D44708E41002F79AF 
# # # time: 2023-09-25 19:06:48
# ###########################################################################################################################
# # # 2023/09/25
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Asterix action dim: 9
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.0_lr1e-7_test"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=9' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.0'

# wait

# # # 2023/10/22
# # # baseline: CQL, 	AILab_MLC_QY3257E36B2AF54454900D47B8FD7E
# # # action dim: 18
# # instance: AILab_MLC_QY5151697300E9451DA6CFB2BB8514
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/QR_DQN_ConvAndLnDropout_1000it_correct \
#   --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=0.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormDropoutNetwork"'


# # # 2023/10/22
# # # baseline: CQL, QuantileConFcLayerNormDropoutNetwork
# # # action dim: 6
# # instance: AILab_MLC_QYAE14ECBE16DD43FB8D8257608637
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/QR_DQN_ConvAndLnDropout_1000it_correct \
#   --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=0.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormDropoutNetwork"'


# # 2023/10/22
# # # baseline: CQL, QuantileConFcLayerNormDropoutNetwork
# # # action dim: 6
# # instance: AILab_MLC_QYA26387EE181A47B194E3475E7077
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/QR_DQN_ConvAndLnDropout_1000it_correct \
#   --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=0.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormDropoutNetwork"'


# # 2023/10/22
# # # baseline: CQL, QuantileConFcLayerNormDropoutNetwork
# # # action dim: 9
# # instance: AILab_MLC_QY03481D33583C49AA9B1761323D94
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/QR_DQN_ConvAndLnDropout_1000it_correct \
#   --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=0.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormDropoutNetwork"'


# # 2023/10/22
# # # baseline: CQL, QuantileConFcLayerNormDropoutNetwork
# # # action dim: 4
# # instance: AILab_MLC_QY14AF3D9B7F434628A6E9FB62A8AF
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/QR_DQN_ConvAndLnDropout_1000it_correct \
#   --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=0.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormDropoutNetwork"'


# # 2023/10/22
# # # # baseline: REM, MultiHeadQConFcLayerNormDropoutNetwork
# # # action dim: 18
# # # instance: AILab_MLC_QYB3487BFE88C84DEFA1FC5F453C17
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/REM_ConvAndLnDropout_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --agent_name=multi_head_dqn \
#  --gin_files='batch_rl/fixed_replay/configs/rem.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.network="atari_helpers.MultiHeadQConFcLayerNormDropoutNetwork"' 


# # 2023/10/22
# # # # baseline: REM, MultiHeadQConFcLayerNormDropoutNetwork
# # # # action dim: 6
# # # instance: AILab_MLC_QY1A0D1A26B8214DB39B52461049AE
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/REM_ConvAndLnDropout_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --agent_name=multi_head_dqn \
#  --gin_files='batch_rl/fixed_replay/configs/rem.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.network="atari_helpers.MultiHeadQConFcLayerNormDropoutNetwork"' 


# # 2023/10/22
# # # # baseline: REM, MultiHeadQConFcLayerNormDropoutNetwork
# # # # action dim: 4
# # # instance: 	AILab_MLC_QYE30948B387554DC183E5764363B2
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/REM_ConvAndLnDropout_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --agent_name=multi_head_dqn \
#  --gin_files='batch_rl/fixed_replay/configs/rem.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.network="atari_helpers.MultiHeadQConFcLayerNormDropoutNetwork"' 


# # # 2023/10/22
# # # # baseline: REM, MultiHeadQConFcLayerNormDropoutNetwork
# # # # action dim: 6
# # # instance: 	AILab_MLC_QY89E62830A4884C2DAAC8A2A3F136
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/REM_ConvAndLnDropout_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --agent_name=multi_head_dqn \
#  --gin_files='batch_rl/fixed_replay/configs/rem.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.network="atari_helpers.MultiHeadQConFcLayerNormDropoutNetwork"' 


# # # # 2023/10/22
# # # # baseline: REM, MultiHeadQConFcLayerNormDropoutNetwork
# # # # action dim: 9
# # # instance: AILab_MLC_QYD307C3993ABC428E8378E01F7A49
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/REM_ConvAndLnDropout_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --agent_name=multi_head_dqn \
#  --gin_files='batch_rl/fixed_replay/configs/rem.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayMultiHeadDQNAgent.network="atari_helpers.MultiHeadQConFcLayerNormDropoutNetwork"' 

<<<<<<< HEAD
############################################################################################################################
# # instance: AILab_MLC_QY4781341D0C3D44708E41002F79AF 
# # time: 2023-10-23 19:06:48
###########################################################################################################################
# # augmentation test-time data
# # Asterix action dim: 9
=======
# ############################################################################################################################
# # # instance: 	AILab_MLC_QYA63947674A66426E9F6A4A0CFE79	 
# # # time: 2023-10-28 01:23:27
# ###########################################################################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy1.0_tent_lr1e-6_decayEntropy"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.entropy=1.0' \
#  --gin_bindings='DQNAgent.kl_coef=1.5' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' 

# # sleep 1d

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy1.0_tent_lr1e-5_decayEntropy"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.entropy=1.0' \
#  --gin_bindings='DQNAgent.kl_coef=1.5' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-5' 


# ############################################################################################################################
# # # instance: AILab_MLC_QYF12F789CAD8C4C28BB4DDD8063E1 
# # # time: 	2023-10-28 01:24:18
# ###########################################################################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy1.0_tent_lr1e-4_decayEntropy"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.entropy=1.0' \
#  --gin_bindings='DQNAgent.kl_coef=1.5' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-4' 

# # sleep 1d

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy1.0_tent_lr3e-4_decayEntropy"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.entropy=1.0' \
#  --gin_bindings='DQNAgent.kl_coef=1.5' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.test_learning_lr=3e-4' 

# ############################################################################################################################
# # # instance: 		AILab_MLC_QY730BB5EDCCE84BA09608D150B63E	 
# # # time: 	2023-10-28 01:30:54
# ###########################################################################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy1.0_kl1.5_lr1e-6_decayEntropy_augmentation"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.entropy=1.0' \
#  --gin_bindings='DQNAgent.kl_coef=1.5' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' 

# # sleep 1d

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy1.0_kl1.5_lr1e-5_decayEntropy_augmentation"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.entropy=1.0' \
#  --gin_bindings='DQNAgent.kl_coef=1.5' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-5' 


# ############################################################################################################################
# # # instance: 	AILab_MLC_QY79AE4F9E93AF453B8262C9B93FBC 
# # # time: 			2023-10-28 01:29:36
# ###########################################################################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy1.0_kl1.5_lr1e-4_decayEntropy_augmentation"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.entropy=1.0' \
#  --gin_bindings='DQNAgent.kl_coef=1.5' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-4' 

# # sleep 1d

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy1.0_kl1.5_lr3e-4_decayEntropy_augmentation"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.entropy=1.0' \
#  --gin_bindings='DQNAgent.kl_coef=1.5' \
#  --gin_bindings='DQNAgent.test_learning_lr=3e-4' 


# ############################################################################################################################
# # # instance: 	AILab_MLC_QY9A5C9C59E7664394AF3E8E9BE6FB 
# # # time: 				2023-10-28 01:32:32
# ###########################################################################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy1.0_kl0.0_decayEntropy_augmentation"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.entropy=1.0' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-4' 

# # sleep 1d

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy1.0_kl0.0_lr3e-4_decayEntropy_augmentation"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.entropy=1.0' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=3e-4' 

# ############################################################################################################################
# # # instance: 			AILab_MLC_QYA47DB7815607406BAABCE1C4D9C1	 
# # # time: 		2023-10-28 01:33:56
# ###########################################################################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy1.0_kl0.0_lr1e-6_decayEntropy_augmentation"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.entropy=1.0' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' 

# # sleep 1d

# wait

# ##############################################################################
# # instance: AILab_MLC_QYF4D15B3441324DD0943053822933 	2023-11-02 01:19:07
# ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' 

# wait 

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' 

# wait 

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' 

# wait 

# # ##############################################################################
# # # instance: AILab_MLC_QYE646E1328CF545FA95EC436FBFC7 	2023-11-02 01:22:58
# # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' 

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' 

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_is_decayLr"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.is_decayLr=True' 


# # ##############################################################################
# # # instance: AILab_MLC_QYFF2C771E17B74857BF3A288D7A57 	2023-11-02 01:27:19
# # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_is_decayLr"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.is_decayLr=True' 

# wait 

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_is_decayLr"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.is_decayLr=True' 

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_is_decayLr"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.is_decayLr=True'  

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_is_decayLr"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.is_decayLr=True'   


# # ##############################################################################
# # # instance: AILab_MLC_QYFF2C771E17B74857BF3A288D7A57 	2023-11-02 01:27:19
# # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_is_augmentation"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.is_augmentation=True' 

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_is_augmentation"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.is_augmentation=True' 

# wait 

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_is_augmentation"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.is_augmentation=True' 

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_is_augmentation"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.is_augmentation=True'  

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_is_augmentation"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.is_augmentation=True'   

# # ##############################################################################
# # # instance: 	AILab_MLC_QYEDEA3645F3CC4251BF42A0199603 	2023-11-08 17:20:45
# # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_updateBatch_buffer_correct"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True'
# #  --gin_bindings='DQNAgent.is_multi_step=True' \

# # sleep 1d

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_updateBatch_buffer_correct"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True'

# # sleep 1d


# # ##############################################################################
# # # instance: AILab_MLC_QY2F2E6A4FD7BE4BED980F4881F77B  2023-11-08 17:19:30
# # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_updateBatch_buffer_correct"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True'
# #  --gin_bindings='DQNAgent.is_multi_step=True' \
# #  --gin_bindings='DQNAgent.update_with_batch=True'

# # sleep 1d

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_updateBatch_buffer_correct"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True'

# sleep 1d

# # ##############################################################################
# # # instance: AILab_MLC_QY0427A97EDCA44E768579FD3CEF01 		2023-11-08 20:04:07
# # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_updateBatch_buffer_correct"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True'

# sleep 1d

# # wait


# # # ##############################################################################
# # # # instance: AILab_MLC_QY1397A523D24043FB8C0D78E78595 	2023-11-08 20:08:27
# # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_is_augmentation"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.is_augmentation=True'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_is_augmentation"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.is_augmentation=True'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_is_augmentation"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.is_augmentation=True'


# # ##############################################################################
# # # instance: 	AILab_MLC_QY7D0127B9201D477C990BE856FE01 	2023-11-10 01:04:43
# # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_final_max_q_episode"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' 

# wait 

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_final_max_q_episode"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' 

#  wait 

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_final_max_q_episode"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' 


# # ##############################################################################
# # # instance: 	AILab_MLC_QY32ED59023AA8400A8D6D0128393A 	2023-11-10 01:05:51
# # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_final_max_q_episode"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' 

# wait 

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_final_max_q_episode"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' 


# # ##############################################################################
# # # instance: 	AILab_MLC_QY897A66C7A44846B78503FF38CFA2 2023-11-10 01:08:55
# # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_discount_max_q_episode"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

# wait 

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_discount_max_q_episode"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

# wait 

# # ##############################################################################
# # # instance: AILab_MLC_QY1A5D8BD6C0084B14A4DBEC77BA91 	2023-11-10 01:09:31
# # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_discount_max_q_episode"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

# wait 

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_discount_max_q_episode"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

# wait 

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_discount_max_q_episode"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

# # ##############################################################################
# # # instance: 	AILab_MLC_QYCC25D0D48302451C9B03838E1234 	2023-11-10 11:03:36
# # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_final_max_q_episode2000_seeds4"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' 

# wait 

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_final_max_q_episode2000_seeds4"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' 

# # ##############################################################################
# # # instance: 	AILab_MLC_QYF03F34B194004EBFAFDB70BD3C14 	2023-11-10 11:05:06
# # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_final_max_q_episode2000_seeds4"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' 

# wait 

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_final_max_q_episode2000_seeds4"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' 

#  wait 

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_final_max_q_episode2000_seeds4"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' 

# # ##############################################################################
# # # instance: 	AILab_MLC_QYBAA54DAB98D649BC8E4B095A7AF3 	2023-11-10 11:06:34
# # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_discount_max_q_episode2000_seeds4"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

# wait 

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_discount_max_q_episode2000_seeds4"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

# wait 

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_discount_max_q_episode2000_seeds4"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

# # ##############################################################################
# # # instance: 	AILab_MLC_QYA4163435CB5F4116A2E0E13E043F 2023-11-10 11:08:37
# # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_discount_max_q_episode2000_seeds4"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

# wait 

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_discount_max_q_episode2000_seeds4"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'


# # # ##############################################################################
# # # # instance: AILab_MLC_QY99A4AF949C074BFDAD14898AE5C2  2023-11-13 22:16:29
# # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_discount_max_q_episode"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_final_max_q_episode"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' 

# # # ##############################################################################
# # # # instance: AILab_MLC_QY3FBD62A2E13643259D4F4FF8C03A  2023-11-14 00:58:17
# # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-7_upate_with_discount_max_q_esilonAction"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-7' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-7_upate_with_final_max_q_esilonAction"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-7' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' 

# # # ##############################################################################
# # # # instance: AILab_MLC_QYF880693EF8674F46884A44101F4B  2023-11-14 01:00:28
# # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-7_upate_with_discount_max_q_esilonAction"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-7' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-7_upate_with_final_max_q_esilonAction"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-7' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' 

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-7_upate_with_discount_max_q_esilonAction"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-7' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-7_upate_with_final_max_q_esilonAction"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-7' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' 


# # # ##############################################################################
# # # # instance: AILab_MLC_QY8F591ADBFE584D7FA6D228162BAB  2023-11-15 01:51:43
# # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_discount_max_q_esilonAction_updateWholeEpisode"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_discount_max_q_esilonAction_updateWholeEpisode"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_discount_max_q_esilonAction_updateWholeEpisode"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_discount_max_q_esilonAction_updateWholeEpisode"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'

# wait

# # # # ##############################################################################
# # # # # instance: AILab_MLC_QY898AE11DFFF14EB693A61521F161  2023-11-15 03:02:10
# # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_10step"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_10step"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'


# # # # ##############################################################################
# # # # # instance: AILab_MLC_QY028F42B81038441CA06DC1EF5845  2023-11-15 03:03:28
# # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_10step"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_10step"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_10step"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'


# # # # ##############################################################################
# # # # # instance: AILab_MLC_QY3AE782C30179405DA70BA0AB6090  2023-11-21 15:25:00
# # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_min_total_entropy"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_min_total_entropy"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200'


# # # # ##############################################################################
# # # # # instance: AILab_MLC_QYE6D1921BAA73483C9EFDF6F25F21  2023-11-21 16:04:11
# # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_min_total_entropy"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_min_total_entropy"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200'


# # # # ##############################################################################
# # # # # instance: AILab_MLC_QYA100C12F14D44EFE956144E2794D  2023-11-21 17:33:33
# # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_min_total_entropy_updateWholeEpisode"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'


# # # # ##############################################################################
# # # # # instance: 	AILab_MLC_QY2121121832314352AF2C83B511E6   2023-11-21 17:32:31
# # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_min_total_entropy_updateWholeEpisode"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_min_total_entropy_updateWholeEpisode"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'


# # # # ##############################################################################
# # # # # instance: 	AILab_MLC_QY54D573AF2E944E019C8DF51FE1C5  	2023-11-21 17:31:48
# # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_min_total_entropy_updateWholeEpisode"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_min_total_entropy_updateWholeEpisode"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'

# # # # ##############################################################################
# # # # # instance: 	AILab_MLC_QY2A4F7A1C70884B0588BD6B134EE8  2023-11-22 11:35:01
# # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_zero_return"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_with_zero_return=True'

# # # # ##############################################################################
# # # # # instance: AILab_MLC_QY8880FE6F48F5461891D2CD2CB8C2  	2023-11-22 11:35:40
# # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_zero_return"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_with_zero_return=True'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_zero_return"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_with_zero_return=True'

# # # # ##############################################################################
# # # # # instance: AILab_MLC_QYEF91217682CA436186570F9F3E44  	2023-11-22 11:36:11
# # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_zero_return"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_with_zero_return=True'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_zero_return"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_with_zero_return=True'


# # # # ##############################################################################
# # # # # instance: AILab_MLC_QY3337A44C54854313BD7BD0783F7E  	2023-11-22 16:46:07
# # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_zero_return_discountNextQ"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_with_zero_return=True'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_zero_return_discountNextQ"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_with_zero_return=True'

# # # # ##############################################################################
# # # # # instance: AILab_MLC_QY089CC4E5EA49461DB4AA15F5FE17  	2023-11-22 16:47:32
# # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_zero_return_discountNextQ"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_with_zero_return=True'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_zero_return_discountNextQ"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_with_zero_return=True'

# # # # ##############################################################################
# # # # # instance: 	AILab_MLC_QY4092D2BB18414095AC0D2197AF06   2023-11-22 17:29:37
# # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_zero_return_discountNextQ"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_with_zero_return=True'

# # # # ##############################################################################
# # # # # instance: AILab_MLC_QYBDCDD21766B7413FBFC94AD867EA  2023-11-22 17:29:22
# # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_zero_return_discountNextQ_targetNextQ"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_with_zero_return=True'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_zero_return_discountNextQ_targetNextQ"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_with_zero_return=True'

# # # # ##############################################################################
# # # # # instance: AILab_MLC_QYF1C7BF1BB6C04FBE8A6905CB5A8A  	2023-11-22 17:29:08
# # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_zero_return_discountNextQ_targetNextQ"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_with_zero_return=True'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_zero_return_discountNextQ_targetNextQ"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_with_zero_return=True'

# # # # ##############################################################################
# # # # # instance: 	AILab_MLC_QYF32F5E90B4384C5690868C501863   2023-11-22 22:40:18
# # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_zero_return_discountNextQ_selectEpisode"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_with_zero_return=True'

# # # # ##############################################################################
# # # # # instance: 	AILab_MLC_QYC1F04093EA3B443D95A5431BFD3A   2023-11-22 22:40:04
# # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_zero_return_discountNextQ_selectEpisode"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_with_zero_return=True'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_zero_return_discountNextQ_selectEpisode"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_with_zero_return=True'

# # # # ##############################################################################
# # # # # instance: AILab_MLC_QY1A759E2B34AB44EE99D9A5AF6EF3   2023-11-22 22:39:42
# # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_zero_return_discountNextQ_selectEpisode"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_with_zero_return=True'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_zero_return_discountNextQ_selectEpisode"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_with_zero_return=True'

# # # # ##############################################################################
# # # # # instance: AILab_MLC_QY2A44F4157A6144098F4D165D2DD4   2023-11-23 11:30:07
# # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_zero_return_discountNextQ_selectEpisode_batchUpdate"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_with_zero_return=True'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_zero_return_discountNextQ_selectEpisode_batchUpdate"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_with_zero_return=True'

# # # # ##############################################################################
# # # # # instance: 	AILab_MLC_QY27E94180B3AA4877A1A4644DAF15   2023-11-23 11:30:59
# # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_zero_return_discountNextQ_selectEpisode_batchUpdate"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_with_zero_return=True'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_zero_return_discountNextQ_selectEpisode_batchUpdate"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_with_zero_return=True'

# # # # ##############################################################################
# # # # # instance: 	AILab_MLC_QYD14ABE3795C54797AD25AD0714AD   2023-11-23 11:31:26
# # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_zero_return_discountNextQ_selectEpisode_batchUpdate"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_with_zero_return=True'

# # # # ##############################################################################
# # # # # instance: 	AILab_MLC_QYF788355BF7D14EED99443644A8DE   2023-11-23 13:51:50
# # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_zero_return_discountNextQ_batchUpdate"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_with_zero_return=True'


# # # # ##############################################################################
# # # # # instance: AILab_MLC_QY480764D753654558A00EDA2E4811   	2023-11-23 13:52:47
# # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_zero_return_discountNextQ_batchUpdate"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_with_zero_return=True'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_zero_return_discountNextQ_batchUpdate"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_with_zero_return=True'

# # # # ##############################################################################
# # # # # instance: 	AILab_MLC_QY43B5E9FF6F9649DE842DC06E2DB4   2023-11-23 13:53:31
# # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_zero_return_discountNextQ_batchUpdate"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_with_zero_return=True'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_zero_return_discountNextQ_batchUpdate"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_with_zero_return=True'

# # # # ##############################################################################
# # # # # instance: AILab_MLC_QY5D105C2649C3401C85801F9D39C5   2023-11-23 17:59:43
# # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_bellman_equation"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_with_zero_return=True' \
#  --gin_bindings='DQNAgent.update_with_bellman_equation=True'

# # # # ##############################################################################
# # # # # instance: 	AILab_MLC_QY708FFE29BD124FD18001F52A058C   2023-11-23 18:00:03
# # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_bellman_equation"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_with_zero_return=True' \
#  --gin_bindings='DQNAgent.update_with_bellman_equation=True'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_bellman_equation"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_with_zero_return=True' \
#  --gin_bindings='DQNAgent.update_with_bellman_equation=True'

# # # # ##############################################################################
# # # # # instance: 	AILab_MLC_QYB934B054708E4B22B46949FEEF16   2023-11-23 18:00:47
# # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_bellman_equation"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_with_zero_return=True' \
#  --gin_bindings='DQNAgent.update_with_bellman_equation=True'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_bellman_equation"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_with_zero_return=True' \
#  --gin_bindings='DQNAgent.update_with_bellman_equation=True'


# # # # ##############################################################################
# # # # # instance: AILab_MLC_QY8D63E8C0CB454F14B54B708BF6FF   2023-11-23 22:17:25
# # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_bellman_equation_woSelection"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_with_zero_return=True' \
#  --gin_bindings='DQNAgent.update_with_bellman_equation=True'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_bellman_equation_woSelection"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_with_zero_return=True' \
#  --gin_bindings='DQNAgent.update_with_bellman_equation=True'


# # # # ##############################################################################
# # # # # instance: AILab_MLC_QYA382F62A24304DD899B7A7B46886   2023-11-23 22:18:12
# # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_bellman_equation_woSelection"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_with_zero_return=True' \
#  --gin_bindings='DQNAgent.update_with_bellman_equation=True'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_bellman_equation_woSelection"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_with_zero_return=True' \
#  --gin_bindings='DQNAgent.update_with_bellman_equation=True'

# # # # ##############################################################################
# # # # # instance: AILab_MLC_QYF02DFBD7F6114B2EB54E865C84F5   2023-11-23 22:19:06
# # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_bellman_equation_woSelection"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='DQNAgent.update_with_zero_return=True' \
#  --gin_bindings='DQNAgent.update_with_bellman_equation=True'

# # # # ##############################################################################
# # # # # instance: AILab_MLC_QYD076C19567454FF1B37DB8E2C3D4   2023-11-23 22:46:35
# # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_discount_alpha1_max_q_beta1_entropy"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

# # # # ##############################################################################
# # # # # instance: AILab_MLC_QY07C74E99A8D54B4786359449E72D   2023-11-23 22:47:37
# # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_discount_alpha1_max_q_beta1_entropy"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

# wait 

# # # # # ##############################################################################
# # # # # # instance: AILab_MLC_QY6CF373D04C92491A86C73B259166   2023-11-24 01:10:39
# # # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_discount_alpha1_max_q_beta1_entropy"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

# # # # ##############################################################################
# # # # # instance: AILab_MLC_QYC029208871DE4986BB9EDB41E914   2023-11-23 22:48:16
# # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_discount_alpha1_max_q_beta1_entropy"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

# wait 

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_discount_alpha1_max_q_beta1_entropy"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

# # # # # ##############################################################################
# # # # # # instance: AILab_MLC_QY35C9D742DC4547528D4262330D32   2023-11-24 01:10:39
# # # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_discount_alpha1_max_q_beta0.5_entropy"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_discount_alpha1_max_q_beta0.5_entropy"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_discount_alpha1_max_q_beta0.5_entropy"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_discount_alpha1_max_q_beta0.5_entropy"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

# # # # # # ##############################################################################
# # # # # # # instance: AILab_MLC_QY293D2892F856452DA32B1F261BDA   2023-11-24 01:17:25
# # # # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_discount_alpha1_max_q_beta0.5_entropy"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

# # # # # # ##############################################################################
# # # # # # # instance: AILab_MLC_QY01B522F048034966B5278ADE0AE3   2023-11-24 01:21:14
# # # # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_discount_alpha1_max_q_beta2.0_entropy"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

# # # # # ##############################################################################
# # # # # # instance: AILab_MLC_QYA125BC122B5D4AC389143A717691	   2023-11-24 01:26:40
# # # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_discount_max_q_and_divided_entropy"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_discount_max_q_and_divided_entropy"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_discount_max_q_and_divided_entropy"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_discount_max_q_and_divided_entropy"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

# # # # # ##############################################################################
# # # # # # instance: AILab_MLC_QY6AA082F4AC914049A945CB42639F	   2023-11-24 19:42:04
# # # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_discount_max_q_and_divided_entropy"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

# # # # # ##############################################################################
# # # # # # instance: AILab_MLC_QY357C66A5135D4B13B635DB70AE8F	   2023-11-24 01:28:53
# # # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_bellman_equation_batch32"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='DQNAgent.update_with_zero_return=True' \
#  --gin_bindings='DQNAgent.update_with_bellman_equation=True' 

# # # # # ##############################################################################
# # # # # # instance: 	AILab_MLC_QY4C89B701C94545338B53FF93E536	2023-11-24 19:42:23
# # # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_bellman_equation_batch32"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='DQNAgent.update_with_zero_return=True' \
#  --gin_bindings='DQNAgent.update_with_bellman_equation=True' 

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_bellman_equation_batch32"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='DQNAgent.update_with_zero_return=True' \
#  --gin_bindings='DQNAgent.update_with_bellman_equation=True' 

# # # # # ##############################################################################
# # # # # # instance: AILab_MLC_QYC32203DABCB34059AC38D543C347	   2023-11-24 19:42:43
# # # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_bellman_equation_batch32"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='DQNAgent.update_with_zero_return=True' \
#  --gin_bindings='DQNAgent.update_with_bellman_equation=True' 

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_update_with_bellman_equation_batch32"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='DQNAgent.update_with_zero_return=True' \
#  --gin_bindings='DQNAgent.update_with_bellman_equation=True' 

# # # # # # ##############################################################################
# # # # # # # instance: AILab_MLC_QY4085F427D38B46EA87505A92F832   2023-11-24 19:57:22
# # # # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_discount_alpha1_max_q_beta0.1_entropy"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

# # # # # # ##############################################################################
# # # # # # # instance: AILab_MLC_QY3B9ABD6803D44FB0B8D9E21CD065   2023-11-24 19:58:04
# # # # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_discount_alpha1_max_q_beta0.1_entropy"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_discount_alpha1_max_q_beta0.1_entropy"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

# # # # # # ##############################################################################
# # # # # # # instance: AILab_MLC_QY53A24616A2B04C828B13990EFA7E   2023-11-24 19:58:53
# # # # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_discount_alpha1_max_q_beta0.1_entropy"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_discount_alpha1_max_q_beta0.1_entropy"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'


# # # # # # ##############################################################################
# # # # # # # instance: AILab_MLC_QY81AE5245F8D8412499DF16297B80   2023-11-26 22:00:57
# # # # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_discount_alpha1_max_q_beta1_entropy_updateWholeEpisode"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'

# wait 

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_discount_alpha1_max_q_beta1_entropy_updateWholeEpisode"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'

# # # # # # ##############################################################################
# # # # # # # instance: AILab_MLC_QYD7EF851D4F574B1394BF0E0FD056   2023-11-26 22:01:15
# # # # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_discount_alpha1_max_q_beta1_entropy_updateWholeEpisode"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'

# wait 

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_discount_alpha1_max_q_beta1_entropy_updateWholeEpisode"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'


# # # # # # ##############################################################################
# # # # # # # instance: AILab_MLC_QY18D168C9B84C4FE1B02BAE99370E   2023-11-26 22:01:31
# # # # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_discount_alpha1_max_q_beta1_entropy_updateWholeEpisode"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'

# # # # # # ##############################################################################
# # # # # # # instance: AILab_MLC_QYE7A25072251C4FF79F31D464AB6E   2023-11-26 22:03:35
# # # # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_discount_alpha1_max_q_beta1_entropy_updateWholeEpisode_bs32"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'

# # # # # # ##############################################################################
# # # # # # # instance: AILab_MLC_QY94A3FEA83AF1427D96E588F2DF0B   2023-11-26 22:03:51
# # # # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_discount_alpha1_max_q_beta1_entropy_updateWholeEpisode_bs32"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'

# wait 

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_discount_alpha1_max_q_beta1_entropy_updateWholeEpisode_bs32"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'


# # # # # # ##############################################################################
# # # # # # # instance: AILab_MLC_QYE8CC9C334E2148FC8F4F8BA8A248   2023-11-26 22:04:09
# # # # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_discount_alpha1_max_q_beta1_entropy_updateWholeEpisode_bs32"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'

# wait 

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_discount_alpha1_max_q_beta1_entropy_updateWholeEpisode_bs32"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'

# # # # # # ##############################################################################
# # # # # # # instance: AILab_MLC_QY40050555EFFA4F808BE2781BADBC   2023-11-26 22:05:33
# # # # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-7_upate_with_discount_alpha1_max_q_beta1_entropy_updateWholeEpisode_bs32"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-7' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'

# wait 

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-7_upate_with_discount_alpha1_max_q_beta1_entropy_updateWholeEpisode_bs32"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-7' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'

# # # # # # ##############################################################################
# # # # # # # instance: AILab_MLC_QY2F38DCA1B4D549B88657645B6286   	2023-11-26 22:05:50
# # # # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-7_upate_with_discount_alpha1_max_q_beta1_entropy_updateWholeEpisode_bs32"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-7' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'

# # # # # # ##############################################################################
# # # # # # # instance: AILab_MLC_QYCD682DD839AB42D781185FF4BB24   	2023-11-26 22:06:17
# # # # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-7_upate_with_discount_alpha1_max_q_beta1_entropy_updateWholeEpisode_bs32"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-7' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'

# wait 

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-7_upate_with_discount_alpha1_max_q_beta1_entropy_updateWholeEpisode_bs32"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-7' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'

# # # # # # ##############################################################################
# # # # # # # instance: AILab_MLC_QY26E3A64085B64098837163BF4EDF   2023-11-26 22:08:40
# # # # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-7_upate_with_discount_alpha1_max_q_beta1_entropy_updateWholeEpisode"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-7' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'

# wait 

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-7_upate_with_discount_alpha1_max_q_beta1_entropy_updateWholeEpisode"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-7' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'


# # # # # # ##############################################################################
# # # # # # # instance: AILab_MLC_QYA7DE1A4A55AD4E7DA827728B0298   2023-11-26 22:09:00
# # # # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-7_upate_with_discount_alpha1_max_q_beta1_entropy_updateWholeEpisode"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-7' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'

# # # # # # ##############################################################################
# # # # # # # instance: AILab_MLC_QY7B7CDE9828EB46F295C6A5514440   2023-11-26 22:09:48
# # # # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-7_upate_with_discount_alpha1_max_q_beta1_entropy_updateWholeEpisode"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-7' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'

# wait 

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-7_upate_with_discount_alpha1_max_q_beta1_entropy_updateWholeEpisode"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-7' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'

# # # # # # ##############################################################################
# # # # # # # instance: AILab_MLC_QY99B5BE88CC4E47E4BD285135B277   	2023-11-26 22:16:43
# # # # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-7_upate_with_discount_alpha1_max_q_beta1_entropy"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-7' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

# wait 

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-7_upate_with_discount_alpha1_max_q_beta1_entropy"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-7' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'


# # # # # # ##############################################################################
# # # # # # # instance: AILab_MLC_QYA0ECF8C1B77F4F95A03331F6F1C4   2023-11-26 22:16:10
# # # # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-7_upate_with_discount_alpha1_max_q_beta1_entropy"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-7' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

# # # # # # ##############################################################################
# # # # # # # instance: AILab_MLC_QYA03649C688754A119D20FED3AFAF   2023-11-26 22:15:40
# # # # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-7_upate_with_discount_alpha1_max_q_beta1_entropy"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-7' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

# wait 

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-7_upate_with_discount_alpha1_max_q_beta1_entropy"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-7' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'


# # # # # # ##############################################################################
# # # # # # # instance: AILab_MLC_QY3ABD2F13673B48F2995310C9DC6A   	2023-11-29 01:05:08
# # # # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-7_upate_with_first_discount_max_q_second_entropy"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-7' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-7_upate_with_first_discount_max_q_second_entropy"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-7' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

# # # # # # ##############################################################################
# # # # # # # instance: AILab_MLC_QY8107F7F48FFA45F497879666A946   	2023-11-29 01:05:27
# # # # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-7_upate_with_first_discount_max_q_second_entropy"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-7' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-7_upate_with_first_discount_max_q_second_entropy"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-7' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

# # # # # # ##############################################################################
# # # # # # # instance: AILab_MLC_QY0EA3E8F4C58545468756F5F8EC86   	2023-11-29 01:06:03
# # # # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-7_upate_with_first_discount_max_q_second_entropy"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-7' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

# # # # # # ##############################################################################
# # # # # # # instance: AILab_MLC_QY0E6AC0A2F6214517B618CD458962  2023-12-04 10:25:34
# # # # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_first_discount_max_q_second_entropy_bs32"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

# # # # # # ##############################################################################
# # # # # # # instance: AILab_MLC_QYECAAEC160F324F3F83D491E3925B   	2023-11-29 01:07:19
# # # # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_first_discount_max_q_second_entropy"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_first_discount_max_q_second_entropy"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

# # # # # # ##############################################################################
# # # # # # # instance: AILab_MLC_QY3D092B3599E845A7893589906E42   	2023-11-29 01:08:18
# # # # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_first_discount_max_q_second_entropy"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_first_discount_max_q_second_entropy"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

# # # # # # ##############################################################################
# # # # # # # instance: AILab_MLC_QY6FC38F52FDDF4E2F9B6979CDF168   	2023-11-29 01:15:01
# # # # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-7_upate_with_first_discount_max_q_second_entropy_updateWholeEpisode"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-7' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-7_upate_with_first_discount_max_q_second_entropy_updateWholeEpisode"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-7' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'

# # # # # # ##############################################################################
# # # # # # # instance: AILab_MLC_QYE2D07C12AB7643BBA11931E7D804   	2023-11-29 01:14:16
# # # # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-7_upate_with_first_discount_max_q_second_entropy_updateWholeEpisode"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-7' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-7_upate_with_first_discount_max_q_second_entropy_updateWholeEpisode"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-7' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'

# # # # # # ##############################################################################
# # # # # # # instance: AILab_MLC_QY29C0F507F622465FBD627568253B   	2023-12-04 10:23:35
# # # # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-7_upate_with_first_discount_max_q_second_entropy_updateWholeEpisode_bs32"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-7' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'

# # # # # # ##############################################################################
# # # # # # # instance: AILab_MLC_QY8B8300C858E74904AD7F7A2A090C   	2023-12-05 17:48:30
# # # # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_first_discount_max_q_second_entropy_updateWholeEpisode_kl1.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Pong/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_first_discount_max_q_second_entropy_updateWholeEpisode_kl1.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'

# # # # # # ##############################################################################
# # # # # # # instance: AILab_MLC_QY6F92D94F8A18425A8656509E988D   	2023-12-05 17:49:23
# # # # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Qbert/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_first_discount_max_q_second_entropy_updateWholeEpisode_kl1.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Seaquest/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_first_discount_max_q_second_entropy_updateWholeEpisode_kl1.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'


# # # # # # ##############################################################################
# # # # # # # instance: AILab_MLC_QYFA556D1FD43B42A2AD75D8B8BC37  	2023-12-06 10:24:49
# # # # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_first_discount_max_q_second_entropy_updateWholeEpisode_kl0.1"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.1' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_first_discount_max_q_second_entropy_updateWholeEpisode_kl0.5"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.5' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Breakout/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.0_lr1e-6_upate_with_first_discount_max_q_second_entropy_updateWholeEpisode_kl2.0"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'

# # # # # # ##############################################################################
# # # # # # # instance: train_AILab_MLC_QY_Atari  	2023-12-06 10:24:49
# # # # # # ##############################################################################
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.1_lr1e-6_upate_with_first_discount_max_q_second_entropy_updateWholeEpisode"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.1' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.5_lr1e-6_upate_with_first_discount_max_q_second_entropy_updateWholeEpisode"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=0.5' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'

# wait

# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl2.0_lr1e-6_upate_with_first_discount_max_q_second_entropy_updateWholeEpisode"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayRunner.seed=0' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.adapted_tent=True' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=2.0' \
#  --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
#  --gin_bindings='DQNAgent.update_with_batch=True' \
#  --gin_bindings='DQNAgent.is_multi_step=True' \
#  --gin_bindings='FixedReplayRunner.eval_episode=200' \
#  --gin_bindings='FixedReplayRunner.max_q_discount=0.9' \
#  --gin_bindings='DQNAgent.update_whole_episode=True'

# # # # # ##############################################################################
# # # # # # task id: train_AILab_MLC_QY_Atari_CB092C5475F4A  	2023-12-07 10:59:03
# # # # # ##############################################################################
CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
 --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
 --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
 --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
 --gin_bindings='FixedReplayRunner.num_iterations=10001' \
 --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.1_lr1e-6_upate_with_first_discount_max_q_second_entropy"' \
 --gin_bindings='FixedReplayRunner.only_eval=True' \
 --gin_bindings='FixedReplayRunner.seed=0' \
 --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
 --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
 --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
 --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
 --gin_bindings='DQNAgent.adapted_tent=True' \
 --gin_bindings='DQNAgent.entropy=0.1' \
 --gin_bindings='DQNAgent.kl_coef=0.1' \
 --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
 --gin_bindings='DQNAgent.update_with_batch=True' \
 --gin_bindings='DQNAgent.is_multi_step=True' \
 --gin_bindings='FixedReplayRunner.eval_episode=200' \
 --gin_bindings='FixedReplayRunner.max_q_discount=0.9' 

wait

>>>>>>> ed0d7b9c1ae446c6cffed41684f9178952a22685
CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
 --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
 --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
 --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
 --gin_bindings='FixedReplayRunner.num_iterations=10001' \
<<<<<<< HEAD
 --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5_augData"' \
=======
 --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl0.5_lr1e-6_upate_with_first_discount_max_q_second_entropy"' \
>>>>>>> ed0d7b9c1ae446c6cffed41684f9178952a22685
 --gin_bindings='FixedReplayRunner.only_eval=True' \
 --gin_bindings='FixedReplayRunner.seed=0' \
 --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
 --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
 --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
 --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
<<<<<<< HEAD
 --gin_bindings='DQNAgent.action_class_dim=9' \
 --gin_bindings='DQNAgent.entropy=0.1' \
 --gin_bindings='DQNAgent.kl_coef=1.5'

sleep 1d
=======
 --gin_bindings='DQNAgent.adapted_tent=True' \
 --gin_bindings='DQNAgent.entropy=0.1' \
 --gin_bindings='DQNAgent.kl_coef=0.5' \
 --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
 --gin_bindings='DQNAgent.update_with_batch=True' \
 --gin_bindings='DQNAgent.is_multi_step=True' \
 --gin_bindings='FixedReplayRunner.eval_episode=200' \
 --gin_bindings='FixedReplayRunner.max_q_discount=0.9'

wait

CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
 --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
 --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
 --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
 --gin_bindings='FixedReplayRunner.num_iterations=10001' \
 --gin_bindings='FixedReplayRunner.eval_dir="tta_entropy0.1_kl2.0_lr1e-6_upate_with_first_discount_max_q_second_entropy"' \
 --gin_bindings='FixedReplayRunner.only_eval=True' \
 --gin_bindings='FixedReplayRunner.seed=0' \
 --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
 --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
 --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
 --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
 --gin_bindings='DQNAgent.adapted_tent=True' \
 --gin_bindings='DQNAgent.entropy=0.1' \
 --gin_bindings='DQNAgent.kl_coef=2.0' \
 --gin_bindings='DQNAgent.test_learning_lr=1e-6' \
 --gin_bindings='DQNAgent.update_with_batch=True' \
 --gin_bindings='DQNAgent.is_multi_step=True' \
 --gin_bindings='FixedReplayRunner.eval_episode=200' \
 --gin_bindings='FixedReplayRunner.max_q_discount=0.9' 
>>>>>>> ed0d7b9c1ae446c6cffed41684f9178952a22685
