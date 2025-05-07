<<<<<<< HEAD
# # # 2023/06/11
# # # gpu016 tmux: rl
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=5 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=100' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \


# # # 2023/06/11
# # # gpu016 tmux: rl2
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=4 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=100' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"'


# # 2023/06/11
# # gpu017 tmux: rl (entropy0.9) gpu017 tmux: rl2 (entropy0.8) 
# # gpu018 tmux: rl (entropy0.7) gpu018 tmux: rl2 (entropy0.6) 
# # gpu019 tmux: rl (entropy0.5) gpu019 tmux: rl2 (entropy0.4) 
# # gpu021 tmux: rl (entropy0.3) gpu021 tmux: rl2 (entropy0.2) 
# # gpu022 tmux: rl (entropy0.1) gpu022 tmux: rl2 (entropy1) 
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_entropy0.1 \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=100' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \


# # 2023/06/11
# # gpu017 tmux: rl (entropy0.9) gpu017 tmux: rl2 (entropy0.8) 
# # gpu018 tmux: rl (entropy0.7) gpu018 tmux: rl2 (entropy0.6) 
# # gpu019 tmux: rl (entropy0.5) gpu019 tmux: rl2 (entropy0.4) 
# # gpu021 tmux: rl (entropy0.3) gpu021 tmux: rl2 (entropy0.2) 
# # gpu022 tmux: rl (entropy0.1) gpu022 tmux: rl2 (entropy1) 
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_entropy1 \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=100' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.entropy=1' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \


# # # 2023/06/11
# # # gpu023 tmux: rl
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=100' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \


# # # 2023/06/11
# # # gpu023 tmux: rl2
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Breakout/1%/CQL \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Breakout/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=100' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \


# # 2023/06/11
# # gpu017 tmux: rl (entropy0.9) gpu017 tmux: rl2 (entropy0.8) 
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Asterix/1%/CQL_ln\
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Asterix/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=100' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=False' \
#   --gin_bindings='DQNAgent.entropy=1' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \


# # # 2023/06/11
# # # gpu016 tmux: rl2
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=4 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_copy \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=27' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.entropy=0.3' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"'


# # # 2023/06/12
# # # gpu026 tmux: rl2
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Breakout/1%/CQL_ln \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Breakout/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=100' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"'


# # # 2023/06/12
# # # gpu026 tmux: rl
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ln \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=100' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"'


# # # 2023/06/11
# # # gpu016 tmux: rl2
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_copy \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.entropy=0.3' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"'


# # # 2023/06/12
# # # gpu023 tmux: rl
# # # baseline: CQL, QuantileNetwork; test lr: 1e-6; finetune full layer; entropy: 0.1
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=4 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ln_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.entropy=0.2' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"'


# # 2023/06/12
# # gpu016 tmux: rl2
# # baseline: CQL, QuantileNetwork; test lr: 1e-6; finetune full layer; entropy: 0.1
# # action dim: 6
#CUDA_VISIBLE_DEVICES=4 python -um batch_rl.fixed_replay.train \
#  --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ln_tta \
#  --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#  --agent_name=quantile \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=101' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"'


# # 2023/06/12
# # gpu017 tmux: rl2
# # baseline: CQL, QuantileNetwork; test lr: 1e-6; finetune full layer; entropy: 0.1
# # action dim: 6
#CUDA_VISIBLE_DEVICES=4 python -um batch_rl.fixed_replay.train \
#	  --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta \
#	    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#	      --agent_name=quantile \
#	        --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#		  --gin_bindings='FixedReplayRunner.num_iterations=101' \
#		    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#		      --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#		        --gin_bindings='FixedReplayRunner.only_eval=True' \
#			  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#			    --gin_bindings='DQNAgent.action_class_dim=6' \
#			      --gin_bindings='DQNAgent.entropy=0.1' \
#			        --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"'


# # 2023/06/12
# # gpu016 tmux: rl2
# # baseline: CQL, QuantileNetwork; test lr: 1e-6; finetune full layer; entropy: 0.1
# # action dim: 6
#CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Breakout/1%/CQL_ln_tta \
#  --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Breakout/1 \
#  --agent_name=quantile \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=101' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=4' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"'


# # # 2023/06/12s
# # # gpu018 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=100' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"'


# # # 2023/06/15
# # # gpu021 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=5 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn_1000it \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"'


# # # 2023/06/15
# # # gpu020 tmux: rl1
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ConvAndFcLn \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=100' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"'


# # 2023/06/15
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=4 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_tent"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.adapted_tent=True' 


# # 2023/06/15
# # # # gpu016 tmux: rl2
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1'


# # 2023/06/15
# # # # gpu026 tmux: rl
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_epsilon0.01"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='DQNAgent.epsilon_adapted_eval=0.01' \
#    --gin_bindings='DQNAgent.is_epsilon_adapted=True'

# wait

# # 2023/06/15
# # # # gpu026 tmux: rl
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.2"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.2'

# wait

# # 2023/06/15
# # # # gpu026 tmux: rl
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.3"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.3'

# wait

# # 2023/06/15
# # # # gpu026 tmux: rl
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.4"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.4'

# wait

# # 2023/06/15
# # # # gpu026 tmux: rl
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.5"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.5'

# wait

# # 2023/06/15
# # # # gpu026 tmux: rl
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.6"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.6'

# wait

# # 2023/06/15
# # # # gpu026 tmux: rl2
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.7"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.7'

# wait

# # 2023/06/15
# # # # gpu026 tmux: rl2
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.8"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.8'

wait

# # 2023/06/15
# # # # gpu026 tmux: rl2
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.9"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.9'

# wait

# # 2023/06/15
# # # # gpu026 tmux: rl2
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy1.0"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=1.0'


# # 2023/06/15
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=4 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_tent_entropy0.1"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.adapted_tent=True' \
#   --gin_bindings='DQNAgent.entropy=0.1'

# wait

# # 2023/06/15
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=4 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_tent_entropy0.2"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.adapted_tent=True' \
#   --gin_bindings='DQNAgent.entropy=0.2'


# # 2023/06/15
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_tent_entropy0.01"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.adapted_tent=True' \
#   --gin_bindings='DQNAgent.entropy=0.01'

# wait

# # 2023/06/15
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_tent_entropy0.02"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.adapted_tent=True' \
#   --gin_bindings='DQNAgent.entropy=0.02'


# # 2023/06/15
# # # # gpu026 tmux: rl2
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy1.0_test"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1'


# # # 2023/06/16
# # # gpu016 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Breakout/1%/CQL_ConvAndFcLn \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Breakout/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=100' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"'


# # # 2023/06/16
# # # gpu019 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_1000it \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \


# # # 2023/06/17
# # # gpu016 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.1'


# # # 2023/06/17
# # # gpu016 tmux: rl2
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_tent_entropy0.1"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.adapted_tent=True' \
#    --gin_bindings='DQNAgent.entropy=0.1'


# # # 2023/06/17
# # # gpu018 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_tent_entropy0.01"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.adapted_tent=True' \
#    --gin_bindings='DQNAgent.entropy=0.01'


# # # 2023/06/17
# # # gpu018 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.001"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.001'


# # 2023/06/17
# # # # gpu017 tmux: rl
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.01'


# # # 2023/06/17
# # # gpu017 tmux: rl
# # # baseline: CQL, 
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Seaquest/1%/CQL \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Seaquest/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=100' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \


# # # 2023/06/17
# # # gpu017 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.01'


# # # 2023/06/17
# # # gpu017 tmux: rl2
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.1'

  
# # # 2023/06/17
# # # gpu018 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_tent_entropy0.1"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.adapted_tent=True' \
#    --gin_bindings='DQNAgent.entropy=0.1'


# # # 2023/06/17
# # # gpu018 tmux: rl2
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_tent_entropy0.01"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.adapted_tent=True' \
#    --gin_bindings='DQNAgent.entropy=0.01'

# wait

# # # 2023/06/17
# # # gpu017 tmux: rl2
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.01'


# # # 2023/06/12s
# # # gpu019 tmux: rl2
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Breakout/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Breakout/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=100' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"'


# # # 2023/06/12s
# # # gpu017 tmux: rl2
# # # baseline: CQL, 
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Breakout/1%/CQL \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Breakout/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=100' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \



# # # 2023/06/17
# # # gpu017 tmux: rl2
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.2"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.2'

# wait

# # # 2023/06/17
# # # gpu017 tmux: rl2
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.3"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.3'


# # # 2023/06/17
# # # gpu017 tmux: rl2
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.2"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.2'

# wait

# # # 2023/06/18
# # # gpu016 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_finallayerAdapted_entropy0.1"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.1' 
#    # --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \

# wait

# # # 2023/06/18
# # # gpu016 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_finallayerAdapted_entropy0.2"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.2' 
#    # --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \


# # 2023/06/15
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=4 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_tent"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.adapted_tent=True' \
#   --gin_bindings='DQNAgent.entropy=0.1'


# # # 2023/06/18
# # # gpu018 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_finallayerAdapted_entropy0.1"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.1' 
#    # --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \

# wait

# # # 2023/06/18
# # # gpu018 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_finallayerAdapted_entropy0.2"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.2' 
#    # --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \


# # # 2023/06/18
# # # gpu016 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.2"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.2' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' 

# wait


# # # 2023/06/18
# # # gpu016 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.3"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.3' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' 


# # # 2023/06/19
# # # gpu016 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.4"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.4' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' 

# wait


# # # 2023/06/19
# # # gpu016 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.5"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.5' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' 

# # # 2023/06/19
# # # gpu016 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.6"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.6' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' 

# wait


# # # 2023/06/19
# # # gpu016 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.7"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.7' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' 


# # # 2023/06/19
# # # gpu016 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.8"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.8' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' 

# wait


# # # 2023/06/19
# # # gpu016 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.9"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.9' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' 


# # # 2023/06/20
# # # gpu017 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Breakout/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Breakout/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=4' \
#    --gin_bindings='DQNAgent.entropy=0.1' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"'

# wait


# # # 2023/06/20
# # # gpu017 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Breakout/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Breakout/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=4' \
#    --gin_bindings='DQNAgent.entropy=0.01' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"'

# wait

# # # 2023/06/20
# # # gpu017 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Breakout/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Breakout/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.2"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=4' \
#    --gin_bindings='DQNAgent.entropy=0.2' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"'


# # # 2023/06/20
# # # gpu022 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Breakout/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Breakout/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_tent_entropy0.1"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=4' \
#    --gin_bindings='DQNAgent.entropy=0.1' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#    --gin_bindings='DQNAgent.adapted_tent=True'

# wait


# # # 2023/06/20
# # # gpu022 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Breakout/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Breakout/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_tent_entropy0.01"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=4' \
#    --gin_bindings='DQNAgent.entropy=0.01' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#    --gin_bindings='DQNAgent.adapted_tent=True'


# # # 2023/06/20
# # # gpu017 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_finallayerAdapted_entropy0.1"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.1' \
#    --gin_bindings='DQNAgent.adapted_final=True'
#    # --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \

# wait

# # # 2023/06/20
# # # gpu017 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_finallayerAdapted_entropy0.2"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.2' \
#    --gin_bindings='DQNAgent.adapted_final=True'
#    # --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \

# wait

# # # 2023/06/20
# # # gpu017 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_finallayerAdapted_entropy0.3"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.3' \
#    --gin_bindings='DQNAgent.adapted_final=True'
#    # --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \


# # # 2023/06/20
# # # gpu016 tmux: rl2
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_finallayerAdapted_entropy0.1"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.1' \
#    --gin_bindings='DQNAgent.adapted_final=True'
#    # --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \

# wait

# # # 2023/06/20
# # # gpu016 tmux: rl2
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_finallayerAdapted_entropy0.2"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.2' \
#    --gin_bindings='DQNAgent.adapted_final=True'
#    # --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \

# wait

# # # 2023/06/20
# # # gpu016 tmux: rl2
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_finallayerAdapted_entropy0.3"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.3' \
#    --gin_bindings='DQNAgent.adapted_final=True'
#    # --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwo



# # # 2023/06/20
# # # gpu019 tmux: rl2
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_finallayerAdapted_entropy0.1"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.1' \
#    --gin_bindings='DQNAgent.adapted_final_ln=True' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' 


# wait


# # # 2023/06/20
# # # gpu019 tmux: rl2
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_finallayerAdapted_entropy0.2"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.2' \
#    --gin_bindings='DQNAgent.adapted_final_ln=True' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' 


# # # 2023/06/20
# # # gpu016 tmux: rl2
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_finallayerAdapted_entropy0.3"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.3' \
#    --gin_bindings='DQNAgent.adapted_final_ln=True' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' 


# wait


# # # 2023/06/20
# # # gpu016 tmux: rl2
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_finallayerAdapted_entropy0.4"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.4' \
#    --gin_bindings='DQNAgent.adapted_final_ln=True' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' 


# wait

# # # 2023/06/20
# # # gpu016 tmux: rl2
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_finallayerAdapted_entropy0.01"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.01' \
#    --gin_bindings='DQNAgent.adapted_final_ln=True' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"'


# # # 2023/06/11
# # # gpu016 tmux: rl
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=5 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_test \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \


# # # 2023/06/20
# # # gpu016 tmux: rl2
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_test \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_finallayerAdapted_entropy0.01"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.01' \
#    --gin_bindings='DQNAgent.adapted_final_ln=True' \


# # 2023/06/15
# # # # gpu026 tmux: rl2
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ln_test \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.7"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.7'


# # # 2023/06/11
# # # gpu016 tmux: rl2
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"'


# # # 2023/06/11
# # # gpu016 tmux: rl2
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Breakout/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"'


# # # 2023/06/11
# # # gpu017 tmux: rl2
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ConvAndLn_1000it \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"'


# # 2023/06/25
# # gpu016 tmux: rl
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=4 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_tent_visual"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.adapted_tent=True' 


# # 2023/06/25
# # gpu016 tmux: rl
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=3 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_visual"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1'


# # # 2023/06/11
# # # gpu017 tmux: rl2
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ConvAndLn_1000it \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"'


# # 2023/06/11
# # gpu018 tmux: rl2 跑错了，这里的数据量应该是 10%
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Asterix/1%/CQL\
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Asterix/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \


# # 2023/06/11
# # gpu019 tmux: rl2 
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=6 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Seaquest/1%/CQL\
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Seaquest/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \


# # 2023/06/25
# # gpu016 tmux: rl
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=4 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=100' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_tent_visual"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.adapted_tent=True' 


# # # 2023/06/17
# # # gpu018 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_tent_entropy0.01"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.adapted_tent=True' \
#    --gin_bindings='DQNAgent.entropy=0.01'


# # 2023/06/27
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=100' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_visual"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1'


# # 2023/06/27
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=100' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_tent_entropy0.1_visual"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.adapted_tent=True' \
#   --gin_bindings='DQNAgent.entropy=0.1'


# # # 2023/06/11
# # # gpu016 tmux: rl2
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_dataset2 \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/2 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # # 2023/06/11
# # # gpu016 tmux: rl2
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/10%/CQL \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=1.0'


# # 2023/06/30
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=3 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn_1000it_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=604' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1'


# # 2023/06/30
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=3 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_test"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1' \


# # 2023/06/27
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=7 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=100' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_tent_entropy0.1_visual2"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.adapted_tent=True' \
#   --gin_bindings='DQNAgent.entropy=0.1' \


# # 2023/07/01
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl0.5"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='DQNAgent.kl_coef=0.5' 

# wait

# # 2023/07/01
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.0"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='DQNAgent.kl_coef=2.0' 

# wait

# # 2023/07/01
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='DQNAgent.kl_coef=1.5'

# wait 

# # 2023/07/01
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl0.1"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='DQNAgent.kl_coef=0.1'


# # 2023/07/01
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn_1000it_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=604' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl0.5"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='DQNAgent.kl_coef=0.5' 

# wait

# # 2023/07/01
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn_1000it_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=604' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.0"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='DQNAgent.kl_coef=2.0' 

# wait

# # 2023/07/01
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn_1000it_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=604' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='DQNAgent.kl_coef=1.5'

# wait 

# # 2023/07/01
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn_1000it_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=604' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl0.1"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='DQNAgent.kl_coef=0.1'


# # # 2023/07/03
# # # gpu020 tmux: rl
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/10%/CQL \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=1.0'


# # # 2023/07/03
# # # gpu022 tmux: rl
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/10%/CQL_ConvAndLn_1000it \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=1.0'


# # # 2023/07/03
# # # gpu022 tmux: rl2
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/10%/CQL_ConvAndLn_1000it \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=1.0'


# # # 2023/07/03
# # # gpu019 tmux: rl
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=7 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Asterix/10%/CQL_ConvAndLn_1000it \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Asterix/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=1.0'


# # # 2023/07/03
# # # gpu021 tmux: rl
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=5 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Asterix/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # # 2023/07/03
# # # gpu027 tmux: rl 跑错了
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=7 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Seaquest/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # # 2023/07/03
# # # gpu027 tmux: rl2
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=6 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Seaquest/10%/CQL_ConvAndLn_1000it \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Seaquest/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=1.0'


# # # 2023/07/03
# # # gpu025 tmux: rl
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Breakout/10%/CQL_ConvAndLn_1000it \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Breakout/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=1.0'


# # 2023/07/04
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAnd_1000it_tta2 \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl0.1"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='DQNAgent.kl_coef=1.0'


# wait


# # 2023/07/04
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAnd_1000it_tta2 \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_tent"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='DQNAgent.adapted_tent=True'


# # # 2023/07/05
# # # gpu026 tmux: rl
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/10%/CQL_ConvAndLn_1000it \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=1.0'


# # 2023/07/06
# # # # gpu026 tmux: rl2
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=5 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl0.5"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='DQNAgent.kl_coef=0.5' 

# wait 

# # 2023/07/06
# # # # gpu026 tmux: rl2
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=5 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.0"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='DQNAgent.kl_coef=1.0' 

# wait 

# # 2023/07/06
# # # # gpu026 tmux: rl2
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=5 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='DQNAgent.kl_coef=1.5' 

# wait 


# # # 2023/07/06
# # # # gpu026 tmux: rl2
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=5 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl0.1_test"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='DQNAgent.kl_coef=0.1' 

# # wait 


# # # 2023/07/06
# # # gpu023 tmux: rl
# # # baseline: DQN, NatureDQNLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/DQN/Qbert/10%/CQL_ConvAndFcLn_1000it \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --agent_name=dqn \
#    --gin_files='batch_rl/fixed_replay/configs/dqn.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='DQNAgent.network="atari_lib.NatureDQNLayerNormNetwork"' 


# # # 2023/07/06
# # # # gpu026 tmux: rl2
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=7 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_tent_moment"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='DQNAgent.adapted_tent=True' \
#   --gin_bindings='DQNAgent.apapted_moment=True'


# # # 2023/07/03
# # # gpu027 tmux: rl2
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=6 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Seaquest/10%/CQL_ConvAndLn_1000it \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Seaquest/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=1.0'


# # # 2023/07/06
# # # # gpu026 tmux: rl2
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=7 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAnd_1000it_tta2 \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_tent_moment"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='DQNAgent.adapted_tent=True' \
#   --gin_bindings='DQNAgent.apapted_moment=True'


# # # 2023/07/06
# # # # gpu026 tmux: rl2
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=7 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAnd_1000it_tta2 \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_tent"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='DQNAgent.adapted_tent=True'


# # # 2023/07/03
# # # gpu021 tmux: rl
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=5 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Asterix/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # # 2023/07/08
# # # gpu027 tmux: rl 正确 Seaquest 1%
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=7 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Seaquest/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # # 2023/07/08
# # # gpu017 tmux: rl 正确 Qbert 1%
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Breakout/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # # 2023/07/08
# # # gpu017 tmux: rl2 正确 Qbert 1%
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # # 2023/07/09
# # # # gpu016 tmux: rl
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=7 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ConvAndLn_1000it \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_tent"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='DQNAgent.adapted_tent=True'


# # # 2023/07/09
# # # # gpu016 tmux: rl2
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=4 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_error_10% \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Breakout/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_tent_kl1.5"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='DQNAgent.kl_coef=1.5'
#   # --gin_bindings='DQNAgent.adapted_tent=True' \
  

# # # 2023/07/09
# # # # gpu016 tmux: rl2
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=7 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_tent_kl1.5"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='DQNAgent.kl_coef=1.5'


# wait 


# # # 2023/07/09
# # # # gpu016 tmux: rl
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=7 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ConvAndLn_1000it \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_tent_kl1.5"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='DQNAgent.kl_coef=1.5'


# # # 2023/07/11
# # # gpu018 tmux: rl 正确 Qbert 1%
# # # baseline: CQL
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_1000it_correct \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' 
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # # 2023/07/11
# # # gpu018 tmux: rl2 正确 Qbert 1%
# # # baseline: CQL
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' 
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # # 2023/07/14
# # # gpu026 tmux: rl 正确 Asterix 1%
# # # baseline: QuantileConFcLayerNormNetwork
# CUDA_VISIBLE_DEVICES=2 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Asterix/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' 
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # # 2023/07/14
# # # gpu022 tmux: rl2 正确 Asterix 1%
# # # baseline: QuantileConFcLayerNormNetwork
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Breakout/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' 
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # # 2023/07/14
# # # gpu018 tmux: rl
# # # baseline: DQN, NatureDQNLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/DQN/Qbert/10%/CQL_ConvAndFcLn_1000it \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --agent_name=dqn \
#    --gin_files='batch_rl/fixed_replay/configs/dqn.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='DQNAgent.network="atari_lib.NatureDQNLayerNormNetwork"' 


# # # 2023/07/14
# # # gpu018 tmux: rl2 正确 Qbert 1%
# # # baseline: CQL
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' 
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # # 2023/07/14
# # # gpu018 tmux: rl 正确 Pong 1%
# # # baseline: CQL
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' 
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


#   # # 2023/07/14
# # # gpu016 tmux: rl 正确 Asterix 1%
# # # baseline: QuantileConFcLayerNormNetwork
# CUDA_VISIBLE_DEVICES=2 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Asterix/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' 
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # # 2023/07/14
# # # gpu016 tmux: rl2 正确 Breakout 1%
# # # baseline: QuantileConFcLayerNormNetwork
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Breakout/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' 
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # # 2023/07/14
# # # gpu018 tmux: rl 正确 Pong 1%
# # # baseline: QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' 
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # # 2023/07/14
# # # gpu017 tmux: rl 正确 Qbert 1%
# # # baseline: QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' 
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'
 

#  # # 2023/07/14
# # # gpu024 tmux: rl 正确 Seaquest 1%
# # # baseline: QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Seaquest/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' 
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # # 2023/07/14
# # # gpu022 tmux: rl
# # # baseline: DQN, NatureDQNLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/DQN/Qbert/1%/CQL_ConvAndFcLn_1000it_correct \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --agent_name=dqn \
#    --gin_files='batch_rl/fixed_replay/configs/dqn.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='DQNAgent.network="atari_lib.NatureDQNLayerNormNetwork"' 

  
# # # 2023/07/14
# # # gpu022 tmux: rl3
# # # baseline: DQN, NatureDQNLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/DQN/Pong/1%/CQL_ConvAndFcLn_1000it_correct \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#    --agent_name=dqn \
#    --gin_files='batch_rl/fixed_replay/configs/dqn.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#    --gin_bindings='DQNAgent.network="atari_lib.NatureDQNLayerNormNetwork"' 


# # # 2023/07/14
# # # gpu019 tmux: rl
# # # baseline: DQN, NatureDQNLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=7 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/DQN/Seaquest/1%/CQL_ConvAndFcLn_1000it_correct \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Seaquest/1 \
#    --agent_name=dqn \
#    --gin_files='batch_rl/fixed_replay/configs/dqn.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#    --gin_bindings='DQNAgent.network="atari_lib.NatureDQNLayerNormNetwork"'  


# # # 2023/07/14
# # # gpu026 tmux: rl
# # # baseline: DQN, NatureDQNLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=2 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/DQN/Breakout/1%/CQL_ConvAndFcLn_1000it_correct \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Breakout/1 \
#    --agent_name=dqn \
#    --gin_files='batch_rl/fixed_replay/configs/dqn.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#    --gin_bindings='DQNAgent.network="atari_lib.NatureDQNLayerNormNetwork"'

  
# # # 2023/07/14
# # # gpu027 tmux: rl
# # # baseline: DQN, NatureDQNLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=7 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/DQN/Asterix/1%/CQL_ConvAndFcLn_1000it_correct \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Asterix/1 \
#    --agent_name=dqn \
#    --gin_files='batch_rl/fixed_replay/configs/dqn.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#    --gin_bindings='DQNAgent.network="atari_lib.NatureDQNLayerNormNetwork"' 


# # # 2023/07/18
# # # gpu018 tmux: rl2
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Breakout action dim: 6 
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_error_10% \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=100' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_finallayerAdapted_entropy0.1_correct"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.1' \
#    --gin_bindings='DQNAgent.adapted_final=True'

# wait 

# # # 2023/07/18
# # # gpu018 tmux: rl2
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Breakout action dim: 6 
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_error_10% \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=100' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_finallayerAdapted_entropy0.2_correct"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.2' \
#    --gin_bindings='DQNAgent.adapted_final=True'

# wait 

# # # 2023/06/20
# # # gpu018 tmux: rl2
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Breakout action dim: 4 
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Breakout/1%/CQL \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Breakout/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=100' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_finallayerAdapted_entropy0.3"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=4' \
#    --gin_bindings='DQNAgent.entropy=0.3' \
#    --gin_bindings='DQNAgent.adapted_final=True'


# wait 

# # # 2023/06/20
# # # gpu018 tmux: rl2
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Breakout action dim: 4 
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_error_10% \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=100' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_finallayerAdapted_entropy0.3_correct"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.3' \
#    --gin_bindings='DQNAgent.adapted_final=True'
   

# # # 2023/07/20
# # # # gpu017 tmux: rl2
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Breakout action dim: 4
# CUDA_VISIBLE_DEVICES=2 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/DQN/Breakout/1%/CQL_ConvAndFcLn_1000it_correct \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Breakout/1 \
#   --agent_name=dqn \
#   --gin_files='batch_rl/fixed_replay/configs/dqn.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#   --gin_bindings='DQNAgent.network="atari_lib.NatureDQNLayerNormNetwork"' \
#   --gin_bindings='DQNAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=4' \
#   --gin_bindings='DQNAgent.entropy=0.1'

# wait 

# # # 2023/07/20
# # # # gpu017 tmux: rl2
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=2 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/DQN/Qbert/1%/CQL_ConvAndFcLn_1000it_correct \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --agent_name=dqn \
#   --gin_files='batch_rl/fixed_replay/configs/dqn.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='DQNAgent.network="atari_lib.NatureDQNLayerNormNetwork"' \
#   --gin_bindings='DQNAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1'

# wait

# # # 2023/07/20
# # # # gpu017 tmux: rl2
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=2 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/DQN/Seaquest/1%/CQL_ConvAndFcLn_1000it_correct \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Seaquest/1 \
#   --agent_name=dqn \
#   --gin_files='batch_rl/fixed_replay/configs/dqn.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#   --gin_bindings='DQNAgent.network="atari_lib.NatureDQNLayerNormNetwork"' \
#   --gin_bindings='DQNAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1'

# wait

# # # 2023/07/20
# # # # gpu017 tmux: rl2
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=2 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Asterix/1%/CQL_ConvAndFcLn_1000it_correct \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Asterix/1 \
#   --agent_name=dqn \
#   --gin_files='batch_rl/fixed_replay/configs/dqn.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_tent"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#   --gin_bindings='DQNAgent.network="atari_lib.NatureDQNLayerNormNetwork"' \
#   --gin_bindings='DQNAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1'

#  # # 2023/07/23
# # # gpu025 tmux: rl 正确 Seaquest 1%
# # # baseline: QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Seaquest/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' 
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # # 2023/07/23
# # # gpu027 tmux: rl2 正确 Qbert 1%
# # # baseline: DQN, NatureDQNLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/DQN/Qbert/1%/CQL_ConvAndFcLn_1000it_correct \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --agent_name=dqn \
#    --gin_files='batch_rl/fixed_replay/configs/dqn.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='DQNAgent.network="atari_lib.NatureDQNLayerNormNetwork"' 


# # # 2023/07/23
# # # gpu027 tmux: rl3 正确 Pong 1%
# # # baseline: DQN, NatureDQNLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/DQN/Pong/1%/CQL_ConvAndFcLn_1000it_correct \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#    --agent_name=dqn \
#    --gin_files='batch_rl/fixed_replay/configs/dqn.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#    --gin_bindings='DQNAgent.network="atari_lib.NatureDQNLayerNormNetwork"' 


# # # 2023/07/14
# # # gpu019 tmux: rl2 正确 Qbert 1%
# # # baseline: QuantileConFcLayerNormDropoutNetwork
# CUDA_VISIBLE_DEVICES=6 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndLn_dropout_1000it_correct \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormDropoutNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # # 2023/07/20
# # # # gpu017 tmux: rl2 错误了错误了
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=2 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Asterix/1%/CQL_ConvAndFcLn_1000it_correct \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Asterix/1 \
#   --agent_name=dqn \
#   --gin_files='batch_rl/fixed_replay/configs/dqn.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_tent"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#   --gin_bindings='DQNAgent.network="atari_lib.NatureDQNLayerNormNetwork"' \
#   --gin_bindings='DQNAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1'


# # # 2023/08/03
# # # gpu017 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Qbert action dim: 6 
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Asterix/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.1' \
#    --gin_bindings='DQNAgent.kl_coef=1.5'


# # # 2023/08/03
# # # gpu017 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Qbert action dim: 6 
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Asterix/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.1' \
#    --gin_bindings='DQNAgent.kl_coef=1.5'


# 2023/08/03
# baseline: CQL, QuantileNetwork
# action dim: 6
#CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'


# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/root/code/transfer_rl/CQL/logs_atari/Pong/1%/ \
#   --replay_dir=/root/datasets/Pong/1 \
#   --agent_name=quantile \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=1.0'


# # # 2023/09/08
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=4 python -um batch_rl.fixed_replay.train \
#   --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/atari/logs_atari/Asterix/1%/CQL_ln \
#   --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"'


# # # 2023/10/25
CUDA_VISIBLE_DEVICES=6,7 python -um batch_rl.fixed_replay.train \
 --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
 --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Asterix/1 \
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
=======
# # # 2023/06/11
# # # gpu016 tmux: rl
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=5 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=100' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \


# # # 2023/06/11
# # # gpu016 tmux: rl2
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=4 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=100' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"'


# # 2023/06/11
# # gpu017 tmux: rl (entropy0.9) gpu017 tmux: rl2 (entropy0.8) 
# # gpu018 tmux: rl (entropy0.7) gpu018 tmux: rl2 (entropy0.6) 
# # gpu019 tmux: rl (entropy0.5) gpu019 tmux: rl2 (entropy0.4) 
# # gpu021 tmux: rl (entropy0.3) gpu021 tmux: rl2 (entropy0.2) 
# # gpu022 tmux: rl (entropy0.1) gpu022 tmux: rl2 (entropy1) 
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_entropy0.1 \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=100' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \


# # 2023/06/11
# # gpu017 tmux: rl (entropy0.9) gpu017 tmux: rl2 (entropy0.8) 
# # gpu018 tmux: rl (entropy0.7) gpu018 tmux: rl2 (entropy0.6) 
# # gpu019 tmux: rl (entropy0.5) gpu019 tmux: rl2 (entropy0.4) 
# # gpu021 tmux: rl (entropy0.3) gpu021 tmux: rl2 (entropy0.2) 
# # gpu022 tmux: rl (entropy0.1) gpu022 tmux: rl2 (entropy1) 
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_entropy1 \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=100' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.entropy=1' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \


# # # 2023/06/11
# # # gpu023 tmux: rl
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=100' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \


# # # 2023/06/11
# # # gpu023 tmux: rl2
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Breakout/1%/CQL \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Breakout/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=100' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \


# # 2023/06/11
# # gpu017 tmux: rl (entropy0.9) gpu017 tmux: rl2 (entropy0.8) 
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Asterix/1%/CQL_ln\
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Asterix/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=100' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=False' \
#   --gin_bindings='DQNAgent.entropy=1' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \


# # # 2023/06/11
# # # gpu016 tmux: rl2
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=4 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_copy \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=27' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.entropy=0.3' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"'


# # # 2023/06/12
# # # gpu026 tmux: rl2
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Breakout/1%/CQL_ln \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Breakout/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=100' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"'


# # # 2023/06/12
# # # gpu026 tmux: rl
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ln \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=100' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"'


# # # 2023/06/11
# # # gpu016 tmux: rl2
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_copy \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.entropy=0.3' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"'


# # # 2023/06/12
# # # gpu023 tmux: rl
# # # baseline: CQL, QuantileNetwork; test lr: 1e-6; finetune full layer; entropy: 0.1
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=4 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ln_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.entropy=0.2' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"'


# # 2023/06/12
# # gpu016 tmux: rl2
# # baseline: CQL, QuantileNetwork; test lr: 1e-6; finetune full layer; entropy: 0.1
# # action dim: 6
#CUDA_VISIBLE_DEVICES=4 python -um batch_rl.fixed_replay.train \
#  --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ln_tta \
#  --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#  --agent_name=quantile \
#  --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=101' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"'


# # 2023/06/12
# # gpu017 tmux: rl2
# # baseline: CQL, QuantileNetwork; test lr: 1e-6; finetune full layer; entropy: 0.1
# # action dim: 6
#CUDA_VISIBLE_DEVICES=4 python -um batch_rl.fixed_replay.train \
#	  --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta \
#	    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#	      --agent_name=quantile \
#	        --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#		  --gin_bindings='FixedReplayRunner.num_iterations=101' \
#		    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#		      --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#		        --gin_bindings='FixedReplayRunner.only_eval=True' \
#			  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#			    --gin_bindings='DQNAgent.action_class_dim=6' \
#			      --gin_bindings='DQNAgent.entropy=0.1' \
#			        --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"'


# # 2023/06/12
# # gpu016 tmux: rl2
# # baseline: CQL, QuantileNetwork; test lr: 1e-6; finetune full layer; entropy: 0.1
# # action dim: 6
#CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Breakout/1%/CQL_ln_tta \
#  --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Breakout/1 \
#  --agent_name=quantile \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=101' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=4' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"'


# # # 2023/06/12s
# # # gpu018 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=100' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"'


# # # 2023/06/15
# # # gpu021 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=5 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn_1000it \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"'


# # # 2023/06/15
# # # gpu020 tmux: rl1
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ConvAndFcLn \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=100' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"'


# # 2023/06/15
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=4 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_tent"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.adapted_tent=True' 


# # 2023/06/15
# # # # gpu016 tmux: rl2
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1'


# # 2023/06/15
# # # # gpu026 tmux: rl
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_epsilon0.01"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='DQNAgent.epsilon_adapted_eval=0.01' \
#    --gin_bindings='DQNAgent.is_epsilon_adapted=True'

# wait

# # 2023/06/15
# # # # gpu026 tmux: rl
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.2"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.2'

# wait

# # 2023/06/15
# # # # gpu026 tmux: rl
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.3"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.3'

# wait

# # 2023/06/15
# # # # gpu026 tmux: rl
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.4"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.4'

# wait

# # 2023/06/15
# # # # gpu026 tmux: rl
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.5"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.5'

# wait

# # 2023/06/15
# # # # gpu026 tmux: rl
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.6"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.6'

# wait

# # 2023/06/15
# # # # gpu026 tmux: rl2
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.7"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.7'

# wait

# # 2023/06/15
# # # # gpu026 tmux: rl2
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.8"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.8'

wait

# # 2023/06/15
# # # # gpu026 tmux: rl2
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.9"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.9'

# wait

# # 2023/06/15
# # # # gpu026 tmux: rl2
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy1.0"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=1.0'


# # 2023/06/15
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=4 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_tent_entropy0.1"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.adapted_tent=True' \
#   --gin_bindings='DQNAgent.entropy=0.1'

# wait

# # 2023/06/15
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=4 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_tent_entropy0.2"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.adapted_tent=True' \
#   --gin_bindings='DQNAgent.entropy=0.2'


# # 2023/06/15
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_tent_entropy0.01"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.adapted_tent=True' \
#   --gin_bindings='DQNAgent.entropy=0.01'

# wait

# # 2023/06/15
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_tent_entropy0.02"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.adapted_tent=True' \
#   --gin_bindings='DQNAgent.entropy=0.02'


# # 2023/06/15
# # # # gpu026 tmux: rl2
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy1.0_test"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1'


# # # 2023/06/16
# # # gpu016 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Breakout/1%/CQL_ConvAndFcLn \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Breakout/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=100' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"'


# # # 2023/06/16
# # # gpu019 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_1000it \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \


# # # 2023/06/17
# # # gpu016 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.1'


# # # 2023/06/17
# # # gpu016 tmux: rl2
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_tent_entropy0.1"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.adapted_tent=True' \
#    --gin_bindings='DQNAgent.entropy=0.1'


# # # 2023/06/17
# # # gpu018 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_tent_entropy0.01"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.adapted_tent=True' \
#    --gin_bindings='DQNAgent.entropy=0.01'


# # # 2023/06/17
# # # gpu018 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.001"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.001'


# # 2023/06/17
# # # # gpu017 tmux: rl
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.01'


# # # 2023/06/17
# # # gpu017 tmux: rl
# # # baseline: CQL, 
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Seaquest/1%/CQL \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Seaquest/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=100' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \


# # # 2023/06/17
# # # gpu017 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.01'


# # # 2023/06/17
# # # gpu017 tmux: rl2
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.1'

  
# # # 2023/06/17
# # # gpu018 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_tent_entropy0.1"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.adapted_tent=True' \
#    --gin_bindings='DQNAgent.entropy=0.1'


# # # 2023/06/17
# # # gpu018 tmux: rl2
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_tent_entropy0.01"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.adapted_tent=True' \
#    --gin_bindings='DQNAgent.entropy=0.01'

# wait

# # # 2023/06/17
# # # gpu017 tmux: rl2
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.01'


# # # 2023/06/12s
# # # gpu019 tmux: rl2
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Breakout/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Breakout/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=100' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"'


# # # 2023/06/12s
# # # gpu017 tmux: rl2
# # # baseline: CQL, 
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Breakout/1%/CQL \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Breakout/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=100' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \



# # # 2023/06/17
# # # gpu017 tmux: rl2
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.2"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.2'

# wait

# # # 2023/06/17
# # # gpu017 tmux: rl2
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.3"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.3'


# # # 2023/06/17
# # # gpu017 tmux: rl2
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.2"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.2'

# wait

# # # 2023/06/18
# # # gpu016 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_finallayerAdapted_entropy0.1"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.1' 
#    # --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \

# wait

# # # 2023/06/18
# # # gpu016 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_finallayerAdapted_entropy0.2"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.2' 
#    # --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \


# # 2023/06/15
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=4 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_tent"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.adapted_tent=True' \
#   --gin_bindings='DQNAgent.entropy=0.1'


# # # 2023/06/18
# # # gpu018 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_finallayerAdapted_entropy0.1"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.1' 
#    # --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \

# wait

# # # 2023/06/18
# # # gpu018 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_finallayerAdapted_entropy0.2"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.2' 
#    # --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \


# # # 2023/06/18
# # # gpu016 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.2"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.2' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' 

# wait


# # # 2023/06/18
# # # gpu016 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.3"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.3' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' 


# # # 2023/06/19
# # # gpu016 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.4"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.4' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' 

# wait


# # # 2023/06/19
# # # gpu016 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.5"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.5' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' 

# # # 2023/06/19
# # # gpu016 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.6"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.6' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' 

# wait


# # # 2023/06/19
# # # gpu016 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.7"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.7' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' 


# # # 2023/06/19
# # # gpu016 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.8"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.8' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' 

# wait


# # # 2023/06/19
# # # gpu016 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.9"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.9' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' 


# # # 2023/06/20
# # # gpu017 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Breakout/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Breakout/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=4' \
#    --gin_bindings='DQNAgent.entropy=0.1' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"'

# wait


# # # 2023/06/20
# # # gpu017 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Breakout/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Breakout/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.01"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=4' \
#    --gin_bindings='DQNAgent.entropy=0.01' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"'

# wait

# # # 2023/06/20
# # # gpu017 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Breakout/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Breakout/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.2"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=4' \
#    --gin_bindings='DQNAgent.entropy=0.2' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"'


# # # 2023/06/20
# # # gpu022 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Breakout/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Breakout/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_tent_entropy0.1"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=4' \
#    --gin_bindings='DQNAgent.entropy=0.1' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#    --gin_bindings='DQNAgent.adapted_tent=True'

# wait


# # # 2023/06/20
# # # gpu022 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Breakout/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Breakout/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_tent_entropy0.01"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=4' \
#    --gin_bindings='DQNAgent.entropy=0.01' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#    --gin_bindings='DQNAgent.adapted_tent=True'


# # # 2023/06/20
# # # gpu017 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_finallayerAdapted_entropy0.1"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.1' \
#    --gin_bindings='DQNAgent.adapted_final=True'
#    # --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \

# wait

# # # 2023/06/20
# # # gpu017 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_finallayerAdapted_entropy0.2"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.2' \
#    --gin_bindings='DQNAgent.adapted_final=True'
#    # --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \

# wait

# # # 2023/06/20
# # # gpu017 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_finallayerAdapted_entropy0.3"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.3' \
#    --gin_bindings='DQNAgent.adapted_final=True'
#    # --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \


# # # 2023/06/20
# # # gpu016 tmux: rl2
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_finallayerAdapted_entropy0.1"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.1' \
#    --gin_bindings='DQNAgent.adapted_final=True'
#    # --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \

# wait

# # # 2023/06/20
# # # gpu016 tmux: rl2
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_finallayerAdapted_entropy0.2"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.2' \
#    --gin_bindings='DQNAgent.adapted_final=True'
#    # --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \

# wait

# # # 2023/06/20
# # # gpu016 tmux: rl2
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_finallayerAdapted_entropy0.3"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.3' \
#    --gin_bindings='DQNAgent.adapted_final=True'
#    # --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwo



# # # 2023/06/20
# # # gpu019 tmux: rl2
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_finallayerAdapted_entropy0.1"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.1' \
#    --gin_bindings='DQNAgent.adapted_final_ln=True' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' 


# wait


# # # 2023/06/20
# # # gpu019 tmux: rl2
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_finallayerAdapted_entropy0.2"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.2' \
#    --gin_bindings='DQNAgent.adapted_final_ln=True' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' 


# # # 2023/06/20
# # # gpu016 tmux: rl2
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_finallayerAdapted_entropy0.3"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.3' \
#    --gin_bindings='DQNAgent.adapted_final_ln=True' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' 


# wait


# # # 2023/06/20
# # # gpu016 tmux: rl2
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_finallayerAdapted_entropy0.4"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.4' \
#    --gin_bindings='DQNAgent.adapted_final_ln=True' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' 


# wait

# # # 2023/06/20
# # # gpu016 tmux: rl2
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_finallayerAdapted_entropy0.01"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.01' \
#    --gin_bindings='DQNAgent.adapted_final_ln=True' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"'


# # # 2023/06/11
# # # gpu016 tmux: rl
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=5 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_test \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \


# # # 2023/06/20
# # # gpu016 tmux: rl2
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_test \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_finallayerAdapted_entropy0.01"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.01' \
#    --gin_bindings='DQNAgent.adapted_final_ln=True' \


# # 2023/06/15
# # # # gpu026 tmux: rl2
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ln_test \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.7"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.7'


# # # 2023/06/11
# # # gpu016 tmux: rl2
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"'


# # # 2023/06/11
# # # gpu016 tmux: rl2
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Breakout/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"'


# # # 2023/06/11
# # # gpu017 tmux: rl2
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ConvAndLn_1000it \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"'


# # 2023/06/25
# # gpu016 tmux: rl
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=4 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_tent_visual"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.adapted_tent=True' 


# # 2023/06/25
# # gpu016 tmux: rl
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=3 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_visual"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1'


# # # 2023/06/11
# # # gpu017 tmux: rl2
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ConvAndLn_1000it \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"'


# # 2023/06/11
# # gpu018 tmux: rl2 跑错了，这里的数据量应该是 10%
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Asterix/1%/CQL\
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Asterix/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \


# # 2023/06/11
# # gpu019 tmux: rl2 
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=6 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Seaquest/1%/CQL\
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Seaquest/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \


# # 2023/06/25
# # gpu016 tmux: rl
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=4 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=100' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_tent_visual"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.adapted_tent=True' 


# # # 2023/06/17
# # # gpu018 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ConvAndFcLn \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=101' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_tent_entropy0.01"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.adapted_tent=True' \
#    --gin_bindings='DQNAgent.entropy=0.01'


# # 2023/06/27
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=100' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_visual"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1'


# # 2023/06/27
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=100' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_tent_entropy0.1_visual"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.adapted_tent=True' \
#   --gin_bindings='DQNAgent.entropy=0.1'


# # # 2023/06/11
# # # gpu016 tmux: rl2
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_dataset2 \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/2 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # # 2023/06/11
# # # gpu016 tmux: rl2
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/10%/CQL \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=1.0'


# # 2023/06/30
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=3 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn_1000it_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=604' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1'


# # 2023/06/30
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=3 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_test"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1' \


# # 2023/06/27
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=7 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=100' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_tent_entropy0.1_visual2"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.adapted_tent=True' \
#   --gin_bindings='DQNAgent.entropy=0.1' \


# # 2023/07/01
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl0.5"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='DQNAgent.kl_coef=0.5' 

# wait

# # 2023/07/01
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.0"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='DQNAgent.kl_coef=2.0' 

# wait

# # 2023/07/01
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='DQNAgent.kl_coef=1.5'

# wait 

# # 2023/07/01
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl0.1"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='DQNAgent.kl_coef=0.1'


# # 2023/07/01
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn_1000it_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=604' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl0.5"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='DQNAgent.kl_coef=0.5' 

# wait

# # 2023/07/01
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn_1000it_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=604' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl2.0"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='DQNAgent.kl_coef=2.0' 

# wait

# # 2023/07/01
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn_1000it_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=604' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='DQNAgent.kl_coef=1.5'

# wait 

# # 2023/07/01
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndFcLn_1000it_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=604' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl0.1"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='DQNAgent.kl_coef=0.1'


# # # 2023/07/03
# # # gpu020 tmux: rl
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/10%/CQL \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=1.0'


# # # 2023/07/03
# # # gpu022 tmux: rl
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/10%/CQL_ConvAndLn_1000it \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=1.0'


# # # 2023/07/03
# # # gpu022 tmux: rl2
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/10%/CQL_ConvAndLn_1000it \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=1.0'


# # # 2023/07/03
# # # gpu019 tmux: rl
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=7 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Asterix/10%/CQL_ConvAndLn_1000it \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Asterix/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=1.0'


# # # 2023/07/03
# # # gpu021 tmux: rl
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=5 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Asterix/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # # 2023/07/03
# # # gpu027 tmux: rl 跑错了
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=7 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Seaquest/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # # 2023/07/03
# # # gpu027 tmux: rl2
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=6 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Seaquest/10%/CQL_ConvAndLn_1000it \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Seaquest/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=1.0'


# # # 2023/07/03
# # # gpu025 tmux: rl
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Breakout/10%/CQL_ConvAndLn_1000it \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Breakout/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=1.0'


# # 2023/07/04
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAnd_1000it_tta2 \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl0.1"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='DQNAgent.kl_coef=1.0'


# wait


# # 2023/07/04
# # baseline: CQL, QuantileNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAnd_1000it_tta2 \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_tent"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='DQNAgent.adapted_tent=True'


# # # 2023/07/05
# # # gpu026 tmux: rl
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/10%/CQL_ConvAndLn_1000it \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=1.0'


# # 2023/07/06
# # # # gpu026 tmux: rl2
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=5 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl0.5"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='DQNAgent.kl_coef=0.5' 

# wait 

# # 2023/07/06
# # # # gpu026 tmux: rl2
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=5 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.0"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='DQNAgent.kl_coef=1.0' 

# wait 

# # 2023/07/06
# # # # gpu026 tmux: rl2
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=5 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='DQNAgent.kl_coef=1.5' 

# wait 


# # # 2023/07/06
# # # # gpu026 tmux: rl2
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=5 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl0.1_test"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='DQNAgent.kl_coef=0.1' 

# # wait 


# # # 2023/07/06
# # # gpu023 tmux: rl
# # # baseline: DQN, NatureDQNLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/DQN/Qbert/10%/CQL_ConvAndFcLn_1000it \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --agent_name=dqn \
#    --gin_files='batch_rl/fixed_replay/configs/dqn.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='DQNAgent.network="atari_lib.NatureDQNLayerNormNetwork"' 


# # # 2023/07/06
# # # # gpu026 tmux: rl2
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=7 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_tent_moment"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='DQNAgent.adapted_tent=True' \
#   --gin_bindings='DQNAgent.apapted_moment=True'


# # # 2023/07/03
# # # gpu027 tmux: rl2
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=6 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Seaquest/10%/CQL_ConvAndLn_1000it \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Seaquest/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=1.0'


# # # 2023/07/06
# # # # gpu026 tmux: rl2
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=7 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAnd_1000it_tta2 \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_tent_moment"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='DQNAgent.adapted_tent=True' \
#   --gin_bindings='DQNAgent.apapted_moment=True'


# # # 2023/07/06
# # # # gpu026 tmux: rl2
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=7 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAnd_1000it_tta2 \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_tent"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='DQNAgent.adapted_tent=True'


# # # 2023/07/03
# # # gpu021 tmux: rl
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=5 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Asterix/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # # 2023/07/08
# # # gpu027 tmux: rl 正确 Seaquest 1%
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=7 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Seaquest/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # # 2023/07/08
# # # gpu017 tmux: rl 正确 Qbert 1%
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Breakout/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # # 2023/07/08
# # # gpu017 tmux: rl2 正确 Qbert 1%
# # # baseline: CQL, QuantileNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # # 2023/07/09
# # # # gpu016 tmux: rl
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=7 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ConvAndLn_1000it \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_tent"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='DQNAgent.adapted_tent=True'


# # # 2023/07/09
# # # # gpu016 tmux: rl2
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=4 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_error_10% \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Breakout/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_tent_kl1.5"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='DQNAgent.kl_coef=1.5'
#   # --gin_bindings='DQNAgent.adapted_tent=True' \
  

# # # 2023/07/09
# # # # gpu016 tmux: rl2
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=7 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_tent_kl1.5"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='DQNAgent.kl_coef=1.5'


# wait 


# # # 2023/07/09
# # # # gpu016 tmux: rl
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=7 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ConvAndLn_1000it \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_tent_kl1.5"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='DQNAgent.kl_coef=1.5'


# # # 2023/07/11
# # # gpu018 tmux: rl 正确 Qbert 1%
# # # baseline: CQL
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_1000it_correct \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' 
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # # 2023/07/11
# # # gpu018 tmux: rl2 正确 Qbert 1%
# # # baseline: CQL
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' 
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # # 2023/07/14
# # # gpu026 tmux: rl 正确 Asterix 1%
# # # baseline: QuantileConFcLayerNormNetwork
# CUDA_VISIBLE_DEVICES=2 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Asterix/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' 
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # # 2023/07/14
# # # gpu022 tmux: rl2 正确 Asterix 1%
# # # baseline: QuantileConFcLayerNormNetwork
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Breakout/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' 
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # # 2023/07/14
# # # gpu018 tmux: rl
# # # baseline: DQN, NatureDQNLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/DQN/Qbert/10%/CQL_ConvAndFcLn_1000it \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --agent_name=dqn \
#    --gin_files='batch_rl/fixed_replay/configs/dqn.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='DQNAgent.network="atari_lib.NatureDQNLayerNormNetwork"' 


# # # 2023/07/14
# # # gpu018 tmux: rl2 正确 Qbert 1%
# # # baseline: CQL
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' 
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # # 2023/07/14
# # # gpu018 tmux: rl 正确 Pong 1%
# # # baseline: CQL
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' 
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


#   # # 2023/07/14
# # # gpu016 tmux: rl 正确 Asterix 1%
# # # baseline: QuantileConFcLayerNormNetwork
# CUDA_VISIBLE_DEVICES=2 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Asterix/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' 
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # # 2023/07/14
# # # gpu016 tmux: rl2 正确 Breakout 1%
# # # baseline: QuantileConFcLayerNormNetwork
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Breakout/1%/CQL_ConvAndLn_1000it_correct \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Breakout/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' 
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # # 2023/07/14
# # # gpu018 tmux: rl 正确 Pong 1%
# # # baseline: QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ConvAndLn_1000it_correct \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' 
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # # 2023/07/14
# # # gpu017 tmux: rl 正确 Qbert 1%
# # # baseline: QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndLn_1000it_correct \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' 
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'
 

#  # # 2023/07/14
# # # gpu024 tmux: rl 正确 Seaquest 1%
# # # baseline: QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Seaquest/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' 
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # # 2023/07/14
# # # gpu022 tmux: rl
# # # baseline: DQN, NatureDQNLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/DQN/Qbert/1%/CQL_ConvAndFcLn_1000it_correct \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --agent_name=dqn \
#    --gin_files='batch_rl/fixed_replay/configs/dqn.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='DQNAgent.network="atari_lib.NatureDQNLayerNormNetwork"' 

  
# # # 2023/07/14
# # # gpu022 tmux: rl3
# # # baseline: DQN, NatureDQNLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/DQN/Pong/1%/CQL_ConvAndFcLn_1000it_correct \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#    --agent_name=dqn \
#    --gin_files='batch_rl/fixed_replay/configs/dqn.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#    --gin_bindings='DQNAgent.network="atari_lib.NatureDQNLayerNormNetwork"' 


# # # 2023/07/14
# # # gpu019 tmux: rl
# # # baseline: DQN, NatureDQNLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=7 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/DQN/Seaquest/1%/CQL_ConvAndFcLn_1000it_correct \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Seaquest/1 \
#    --agent_name=dqn \
#    --gin_files='batch_rl/fixed_replay/configs/dqn.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#    --gin_bindings='DQNAgent.network="atari_lib.NatureDQNLayerNormNetwork"'  


# # # 2023/07/14
# # # gpu026 tmux: rl
# # # baseline: DQN, NatureDQNLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=2 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/DQN/Breakout/1%/CQL_ConvAndFcLn_1000it_correct \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Breakout/1 \
#    --agent_name=dqn \
#    --gin_files='batch_rl/fixed_replay/configs/dqn.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#    --gin_bindings='DQNAgent.network="atari_lib.NatureDQNLayerNormNetwork"'

  
# # # 2023/07/14
# # # gpu027 tmux: rl
# # # baseline: DQN, NatureDQNLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=7 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/DQN/Asterix/1%/CQL_ConvAndFcLn_1000it_correct \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Asterix/1 \
#    --agent_name=dqn \
#    --gin_files='batch_rl/fixed_replay/configs/dqn.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#    --gin_bindings='DQNAgent.network="atari_lib.NatureDQNLayerNormNetwork"' 


# # # 2023/07/18
# # # gpu018 tmux: rl2
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Breakout action dim: 6 
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_error_10% \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=100' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_finallayerAdapted_entropy0.1_correct"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.1' \
#    --gin_bindings='DQNAgent.adapted_final=True'

# wait 

# # # 2023/07/18
# # # gpu018 tmux: rl2
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Breakout action dim: 6 
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_error_10% \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=100' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_finallayerAdapted_entropy0.2_correct"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.2' \
#    --gin_bindings='DQNAgent.adapted_final=True'

# wait 

# # # 2023/06/20
# # # gpu018 tmux: rl2
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Breakout action dim: 4 
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Breakout/1%/CQL \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Breakout/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=100' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_finallayerAdapted_entropy0.3"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=4' \
#    --gin_bindings='DQNAgent.entropy=0.3' \
#    --gin_bindings='DQNAgent.adapted_final=True'


# wait 

# # # 2023/06/20
# # # gpu018 tmux: rl2
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Breakout action dim: 4 
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_error_10% \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=100' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_finallayerAdapted_entropy0.3_correct"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.3' \
#    --gin_bindings='DQNAgent.adapted_final=True'
   

# # # 2023/07/20
# # # # gpu017 tmux: rl2
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # Breakout action dim: 4
# CUDA_VISIBLE_DEVICES=2 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/DQN/Breakout/1%/CQL_ConvAndFcLn_1000it_correct \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Breakout/1 \
#   --agent_name=dqn \
#   --gin_files='batch_rl/fixed_replay/configs/dqn.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#   --gin_bindings='DQNAgent.network="atari_lib.NatureDQNLayerNormNetwork"' \
#   --gin_bindings='DQNAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=4' \
#   --gin_bindings='DQNAgent.entropy=0.1'

# wait 

# # # 2023/07/20
# # # # gpu017 tmux: rl2
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=2 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/DQN/Qbert/1%/CQL_ConvAndFcLn_1000it_correct \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --agent_name=dqn \
#   --gin_files='batch_rl/fixed_replay/configs/dqn.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='DQNAgent.network="atari_lib.NatureDQNLayerNormNetwork"' \
#   --gin_bindings='DQNAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1'

# wait

# # # 2023/07/20
# # # # gpu017 tmux: rl2
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=2 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/DQN/Seaquest/1%/CQL_ConvAndFcLn_1000it_correct \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Seaquest/1 \
#   --agent_name=dqn \
#   --gin_files='batch_rl/fixed_replay/configs/dqn.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#   --gin_bindings='DQNAgent.network="atari_lib.NatureDQNLayerNormNetwork"' \
#   --gin_bindings='DQNAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1'

# wait

# # # 2023/07/20
# # # # gpu017 tmux: rl2
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=2 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Asterix/1%/CQL_ConvAndFcLn_1000it_correct \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Asterix/1 \
#   --agent_name=dqn \
#   --gin_files='batch_rl/fixed_replay/configs/dqn.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_tent"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#   --gin_bindings='DQNAgent.network="atari_lib.NatureDQNLayerNormNetwork"' \
#   --gin_bindings='DQNAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1'

#  # # 2023/07/23
# # # gpu025 tmux: rl 正确 Seaquest 1%
# # # baseline: QuantileConFcLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Seaquest/1%/CQL_ConvAndLn_1000it_correct \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Seaquest/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' 
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # # 2023/07/23
# # # gpu027 tmux: rl2 正确 Qbert 1%
# # # baseline: DQN, NatureDQNLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/DQN/Qbert/1%/CQL_ConvAndFcLn_1000it_correct \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#    --agent_name=dqn \
#    --gin_files='batch_rl/fixed_replay/configs/dqn.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#    --gin_bindings='DQNAgent.network="atari_lib.NatureDQNLayerNormNetwork"' 


# # # 2023/07/23
# # # gpu027 tmux: rl3 正确 Pong 1%
# # # baseline: DQN, NatureDQNLayerNormNetwork
# # # action dim: 6
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/DQN/Pong/1%/CQL_ConvAndFcLn_1000it_correct \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#    --agent_name=dqn \
#    --gin_files='batch_rl/fixed_replay/configs/dqn.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#    --gin_bindings='DQNAgent.network="atari_lib.NatureDQNLayerNormNetwork"' 


# # # 2023/07/14
# # # gpu019 tmux: rl2 正确 Qbert 1%
# # # baseline: QuantileConFcLayerNormDropoutNetwork
# CUDA_VISIBLE_DEVICES=6 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAndLn_dropout_1000it_correct \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormDropoutNetwork"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # # 2023/07/20
# # # # gpu017 tmux: rl2 错误了错误了
# # baseline: CQL, QuantileConFcLayerNormNetwork
# # action dim: 6
# CUDA_VISIBLE_DEVICES=2 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Asterix/1%/CQL_ConvAndFcLn_1000it_correct \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Asterix/1 \
#   --agent_name=dqn \
#   --gin_files='batch_rl/fixed_replay/configs/dqn.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#   --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_tent"' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#   --gin_bindings='DQNAgent.network="atari_lib.NatureDQNLayerNormNetwork"' \
#   --gin_bindings='DQNAgent.adapted=True' \
#   --gin_bindings='DQNAgent.action_class_dim=6' \
#   --gin_bindings='DQNAgent.entropy=0.1'


# # # 2023/08/03
# # # gpu017 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Qbert action dim: 6 
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Asterix/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.1' \
#    --gin_bindings='DQNAgent.kl_coef=1.5'


# # # 2023/08/03
# # # gpu017 tmux: rl
# # # baseline: CQL, QuantileConFcLayerNormNetwork
# # # Qbert action dim: 6 
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#    --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#    --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Asterix/1 \
#    --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#    --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#    --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5"' \
#    --gin_bindings='FixedReplayRunner.only_eval=True' \
#    --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#    --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#    --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#    --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#    --gin_bindings='DQNAgent.action_class_dim=6' \
#    --gin_bindings='DQNAgent.entropy=0.1' \
#    --gin_bindings='DQNAgent.kl_coef=1.5'


# 2023/08/03
# baseline: CQL, QuantileNetwork
# action dim: 6
#CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#  --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct \
#  --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Asterix/1 \
#  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#  --gin_bindings='FixedReplayRunner.num_iterations=10001' \
#  --gin_bindings='FixedReplayRunner.eval_dir="eval_entropy0.1_kl1.5"' \
#  --gin_bindings='FixedReplayRunner.only_eval=True' \
#  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#  --gin_bindings='DQNAgent.action_class_dim=6' \
#  --gin_bindings='DQNAgent.entropy=0.1' \
#  --gin_bindings='DQNAgent.kl_coef=1.5'


# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/root/code/transfer_rl/CQL/logs_atari/Pong/1%/ \
#   --replay_dir=/root/datasets/Pong/1 \
#   --agent_name=quantile \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=1.0'


# # 2023/09/08
# # baseline: CQL, QuantileNetwork
# # action dim: 6
CUDA_VISIBLE_DEVICES=4 python -um batch_rl.fixed_replay.train \
  --base_dir=/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/atari/logs_atari/Asterix/1%/CQL_ln \
  --replay_dir=/apdcephfs/share_1594716/zihaolian/datasets/Asterix/1 \
  --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
  --gin_bindings='FixedReplayRunner.num_iterations=1000' \
  --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"'
>>>>>>> ed0d7b9c1ae446c6cffed41684f9178952a22685
