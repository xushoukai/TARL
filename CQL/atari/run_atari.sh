# gpu021
# CUDA_VISIBLE_DEVICES=2 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --agent_name=quantile \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=1.0'


# # gpu019 tmux: rl2
# # # train 1000 iteration, 10% data, use_staging=False
# CUDA_VISIBLE_DEVICES=2 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs/Quantile_lnAfterDense1_Pong1 \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --agent_name=quantile \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=1.0'


# # gpu020
# CUDA_VISIBLE_DEVICES=2 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs/Quantile_lnAfterConvAndDense1_Pong1 \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --agent_name=quantile \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileConFcLayerNormNetwork"' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=1.0'


# # gpu018
# CUDA_VISIBLE_DEVICES=2 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs/Quantile_bnAfterDense1_Pong1 \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --agent_name=quantile \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileBatchNormNetwork"' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=1.0'


# # gpu022
# CUDA_VISIBLE_DEVICES=6 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs/Quantile_tta_lnAterDense1_Pong1 \
#   --replay_dir=/mnt/ssd/datasets/atari/Pong/1 \
#   --agent_name=quantile \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=1.0'


# # gpu024
# CUDA_VISIBLE_DEVICES=2 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs/Quantile_bnAfterDense1_correctUpdate_Pong1 \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --agent_name=quantile \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileBatchNormNetwork"' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=1.0'


# # gpu020
# CUDA_VISIBLE_DEVICES=7 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs/Quantile_lnAterDense1_test2_Pong1 \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --agent_name=quantile \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # 2023/05/30 && 2023/05/31: start_iteration:15
# # gpu018 tmux: rl
# # train 100 iteration, 1% data
# CUDA_VISIBLE_DEVICES=4 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs/Qbert/1%/CQL \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --agent_name=quantile \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=100' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # 2023/05/31
# # gpu024 tmux: rl
# # # train 100 iteration, 1% data, use_staging=True
# CUDA_VISIBLE_DEVICES=2 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs/Qbert/1%/CQL_ln \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --agent_name=quantile \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=100' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # # 2023/05/31
# # # gpu019 tmux: rl
# # # train 1000 iteration, 10% data, use_staging=False
# CUDA_VISIBLE_DEVICES=6 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs/Quantile_tta_lnAterDense1_Pong1_correct \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --agent_name=quantile \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=1.0'


# #########################################################################################################
# # # 2023/05/31
# # # gpu019 tmux: rl
# # # train 1000 iteration, 10% data, use_staging=False
# #########################################################################################################
# CUDA_VISIBLE_DEVICES=7 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs/Quantile_lnAterDense1_entropy0.05_Pong1 \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --agent_name=quantile \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=100' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # # 2023/06/01
# # # gpu016 tmux: rl
# # # train 1000 iteration, 10% data, use_staging=False
# CUDA_VISIBLE_DEVICES=6 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs/Quantile_tta_lnAterDense1_Pong1_correct_decayTest \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --agent_name=quantile \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=1.0'



# # # 2023/06/01
# # # gpu016 tmux: rl
# # # train 1000 iteration, 10% data, use_staging=False
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs/Quantile_tta_lnAterDense1_Pong1_correct_decayTest_noEnptropyConstrain_test2 \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --agent_name=quantile \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=1.0'


# # # 2023/06/02
# # # gpu026 tmux: rl
# # # train 1000 iteration, 10% data, use_staging=False
# # # QuantileLayerNormNetwork：只修改一层网络；tta: 只用那些 entropy 小于 0.05 的值去更新 Q 网络
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs/Pong/Quantile_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --agent_name=quantile \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=1.0'


# # # 2023/06/02
# # # gpu026 tmux: rl
# # # train 1000 iteration, 10% data, use_staging=False
# # # QuantileLayerNormNetwork：只修改一层网络；tta: 只用那些 entropy 小于 0.05 的值去更新 Q 网络; 
# # # 使用 update_online_tta_op 函数将 online-TTA 的参数复制成 online 网络的参数
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs/Pong/Quantile_tta_online_copy_tta \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --agent_name=quantile \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=1.0'


# # # 2023/06/03
# # # gpu026 tmux: rl
# # # train 1000 iteration, 10% data, use_staging=False
# # # QuantileLayerNormNetwork：只修改一层网络；tta: 只用那些 entropy 小于 0.05 的值去更新 Q 网络; 
# # # 使用 update_online_tta_op 函数将 online-TTA 的参数复制成 online 网络的参数； 
# # # 把 self.state 替换成 self.adapted_state
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs/Pong/Quantile_tta_online_copy_tta_adapted_state_ \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --agent_name=quantile \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=False' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=1.0'


#  # 2023/06/03
# # # gpu026 tmux: rl
# # # train 1000 iteration, 10% data, use_staging=False
# # # 测试看下不 adapted 有什么效果
# CUDA_VISIBLE_DEVICES=2 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs/Pong/Quantile \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --agent_name=quantile \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=False' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=1.0'


####################################################################################
# 效果看起来是正确的
####################################################################################
# # # 2023/06/04
# # # gpu026 tmux: rl
# # # train 1000 iteration, 10% data, use_staging=False
# # # QuantileLayerNormNetwork：只修改一层网络；tta: 只用那些 entropy 小于 0.05 的值去更新 Q 网络; 
# # # 使用 update_online_tta_op 函数将 online-TTA 的参数复制成 online 网络的参数； 
# # # 把 self.state 替换成 self.adapted_state
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs/Pong/Quantile_tta_correct \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --agent_name=quantile \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=1.0'


# # # 2023/06/04
# # # gpu026 tmux: rl
# # # train 1000 iteration, 10% data, use_staging=False
# # # QuantileLayerNormNetwork：只修改一层网络；tta: 只用那些 entropy 小于 0.05 的值去更新 Q 网络; 
# # # 使用 update_online_tta_op 函数将 online-TTA 的参数复制成 online 网络的参数； 
# # # 把 self.state 替换成 self.adapted_state
# # # # 在这个实验里把 TTA online netword 的  eval_mode 设置为 True; 将测试网络设置成可学习
# # CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
# #   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs/Pong/Quantile_tta_compare \
# #   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
# #   --agent_name=quantile \
# #   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
# #   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
# #   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
# #   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
# #   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
# #   --gin_bindings='FixedReplayQuantileAgent.minq_weight=1.0'


# # # # # 2023/06/06
# # # # # gpu017 tmux: rl
# # # # # train 1000 iteration, 10% data, use_staging=False
# # # # # QuantileLayerNormNetwork：只修改一层网络；tta: 只用那些 entropy 小于 0.05 的值去更新 Q 网络; 
# # # # # 使用 update_online_tta_op 函数将 online-TTA 的参数复制成 online 网络的参数； 
# # # # # 把 self.state 替换成 self.adapted_state
# # # # # 在这个实验里把 TTA online netword 的  eval_mode 设置为 True; 将测试网络设置成可学习
# # # CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
# # #   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs/Pong/Quantile_tta_lr0.00005 \
# # #   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
# # #   --agent_name=quantile \
# # #   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
# # #   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
# # #   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
# # #   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
# # #   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
# # #   --gin_bindings='FixedReplayQuantileAgent.minq_weight=1.0'


# # # 2023/06/07
# # # gpu019 tmux: rl
# # # train 1000 iteration, 10% data, use_staging=False
# # # QuantileLayerNormNetwork：只修改一层网络；tta: 只用那些 entropy 小于 0.05 的值去更新 Q 网络; 
# # # 使用 update_online_tta_op 函数将 online-TTA 的参数复制成 online 网络的参数； 
# # # 把 self.state 替换成 self.adapted_state
# # # 在这个实验里把 TTA online netword 的  eval_mode 设置为 True; 将测试网络设置成可学习
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs/Pong/Quantile_tta_lr0.00005_entropy1e-5 \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --agent_name=quantile \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=1.0'


# # # 2023/06/07
# # # gpu016 tmux: rl
# # # train 1000 iteration, 10% data, use_staging=False
# # # QuantileLayerNormNetwork：只修改一层网络；tta: 只用那些 entropy 小于 0.05 的值去更新 Q 网络; 
# # # 使用 update_online_tta_op 函数将 online-TTA 的参数复制成 online 网络的参数； 
# # # 把 self.state 替换成 self.adapted_state
# # # 在这个实验里把 TTA online netword 的  eval_mode 设置为 True; 将测试网络设置成可学习
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs/Pong/Quantile_tta_lr0.00005_entropy1e-9 \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --agent_name=quantile \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=1000' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=1.0'


# # # 2023/06/07
# # # gpu018 tmux: rl
# # # eval 1 iteration
# # # QuantileLayerNormNetwork：只修改一层网络；tta: 只用那些 entropy 小于 0.05 的值去更新 Q 网络; 
# # # 使用 update_online_tta_op 函数将 online-TTA 的参数复制成 online 网络的参数； 
# # # 把 self.state 替换成 self.adapted_state
# # # 在这个实验里把 TTA online netword 的  eval_mode 设置为 True; 将测试网络设置成可学习
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs/Pong/Quantile_tta_lr0.00005_copy \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --agent_name=quantile \
#   --gin_files='batch_rl/fixed_replay/configs/quantile.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=45' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.entropy=1e-2' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=1.0'


# # # 2023/06/08
# # # gpu023 tmux: rl
# # # eval 1 iteration, 阈值设置为 0.5 * ln (1 / 6)
# # # QuantileLayerNormNetwork：只修改一层网络；tta: 只用那些 entropy 小于 0.05 的值去更新 Q 网络; 
# # # 使用 update_online_tta_op 函数将 online-TTA 的参数复制成 online 网络的参数； 
# # # 把 self.state 替换成 self.adapted_state
# # # 在这个实验里把 TTA online netword 的  eval_mode 设置为 True; 将测试网络设置成可学习
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs/Qbert/1%/CQL_ln_entropy0.5 \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --agent_name=quantile \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=100' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.entropy=0.5' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # # 2023/06/08
# # # gpu023 tmux: rl2
# # # eval 1 iteration, 阈值设置为 0.4 * ln (1 / 6)
# # # QuantileLayerNormNetwork：只修改一层网络；tta: 只用那些 entropy 小于 0.05 的值去更新 Q 网络; 
# # # 使用 update_online_tta_op 函数将 online-TTA 的参数复制成 online 网络的参数； 
# # # 把 self.state 替换成 self.adapted_state
# # # 在这个实验里把 TTA online netword 的  eval_mode 设置为 True; 将测试网络设置成可学习
# CUDA_VISIBLE_DEVICES=2 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs/Qbert/1%/CQL_ln_entropy0.4 \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --agent_name=quantile \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=100' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.entropy=0.4' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # # 2023/06/08
# # # gpu018 tmux: rl (entropy0.6) gpu019 tmux: rl2(entropy0.7)
# # # gpu016 tmux: rl2 (entropy0.3) gpu017 tmux: rl2(entropy0.8)
# # # eval 1 iteration, 阈值设置为 0.4 * ln (1 / 6)
# # # QuantileLayerNormNetwork：只修改一层网络；tta: 只用那些 entropy 小于 0.05 的值去更新 Q 网络; 
# # # 使用 update_online_tta_op 函数将 online-TTA 的参数复制成 online 网络的参数； 
# # # 把 self.state 替换成 self.adapted_state
# # # 在这个实验里把 TTA online netword 的  eval_mode 设置为 True; 将测试网络设置成可学习
# CUDA_VISIBLE_DEVICES=5 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs/Qbert/1%/CQL_ln_entropy0.8 \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --agent_name=quantile \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=100' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.entropy=0.8' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # # 2023/06/08
# # # gpu018 tmux: rl
# # # eval 1 iteration
# # # QuantileLayerNormNetwork：只修改一层网络；tta: 只用那些 entropy 小于 0.05 的值去更新 Q 网络; 
# # # 使用 update_online_tta_op 函数将 online-TTA 的参数复制成 online 网络的参数； 
# # # 把 self.state 替换成 self.adapted_state
# # # 在这个实验里把 TTA online netword 的  eval_mode 设置为 True; 将测试网络设置成可学习
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs/Qbert/1%/CQL_ln_entropy0.8_copy \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --agent_name=quantile \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=17' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.entropy=1e-2' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # # 2023/06/08
# # # gpu018 tmux: rl (entropy0.6) gpu019 tmux: rl2(entropy0.7)
# # # gpu016 tmux: rl2 (entropy0.3) gpu017 tmux: rl2(entropy0.8)
# # # eval 1 iteration, 阈值设置为 0.4 * ln (1 / 6)
# # # QuantileLayerNormNetwork：只修改一层网络；tta: 只用那些 entropy 小于 0.05 的值去更新 Q 网络; 
# # # 使用 update_online_tta_op 函数将 online-TTA 的参数复制成 online 网络的参数； 
# # # 把 self.state 替换成 self.adapted_state
# # # 在这个实验里把 TTA online netword 的  eval_mode 设置为 True; 将测试网络设置成可学习
# CUDA_VISIBLE_DEVICES=5 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs/Qbert/1%/CQL_ln_entropy0.8_test \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --agent_name=quantile \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=100' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.entropy=0.8' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # # 2023/06/09
# # # gpu018 tmux: rl (entropy0.6) 
# # # eval 1 iteration, 阈值设置为 0.4 * ln (1 / 6)
# # # QuantileLayerNormNetwork：只修改一层网络；tta: 只用那些 entropy 小于 0.05 的值去更新 Q 网络; 
# # # 使用 update_online_tta_op 函数将 online-TTA 的参数复制成 online 网络的参数； 
# # # 把 self.state 替换成 self.adapted_state
# # # 在这个实验里把 TTA online netword 的  eval_mode 设置为 True; 将测试网络设置成可学习
# CUDA_VISIBLE_DEVICES=5 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs/Qbert/1%/CQL_ln_entropy0.8_test2 \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --agent_name=quantile \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=100' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.entropy=0.8' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # # 2023/06/09
# # # gpu022 tmux: rl (entropy0.8) 
# # # eval 1 iteration, 阈值设置为 0.8 * ln (1 / 4)
# # # QuantileLayerNormNetwork：只修改一层网络；tta: 只用那些 entropy 小于 0.05 的值去更新 Q 网络; 
# # # 使用 update_online_tta_op 函数将 online-TTA 的参数复制成 online 网络的参数； 
# # # 把 self.state 替换成 self.adapted_state
# # # 在这个实验里把 TTA online netword 的  eval_mode 设置为 True; 将测试网络设置成可学习
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs/Breakout/1%/CQL_ln_entropy0.8 \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Breakout/1 \
#   --agent_name=quantile \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_breakout.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=100' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.entropy=0.8' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Breakout"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # 2023/06/09
# # gpu022 tmux: rl2 
# # eval 1 iteration
# # QuantileLayerNormNetwork：只修改一层网络；tta: 只用那些 entropy 小于 0.05 的值去更新 Q 网络; 
# # 使用 update_online_tta_op 函数将 online-TTA 的参数复制成 online 网络的参数； 
# # 把 self.state 替换成 self.adapted_state
# # # 在这个实验里把 TTA online netword 的  eval_mode 设置为 True; 将测试网络设置成可学习
# CUDA_VISIBLE_DEVICES=1 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs/Qbert/1%/CQL_ln_copy \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --agent_name=quantile \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=102' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='FixedReplayRunner.only_eval=True' \
#   --gin_bindings='DQNAgent.entropy=0.8' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # # 2023/06/10
# # # gpu022 tmux: rl2 
# # # baseline: CQL, QuantileNetwork
# # # action dim: 9
# CUDA_VISIBLE_DEVICES=2 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs/Asterix/1%/CQL_ln \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Asterix/1 \
#   --agent_name=quantile \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=100' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Asterix"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'
#   # --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   # --gin_bindings='DQNAgent.entropy=0.8' \


# # # 2023/06/10
# # # gpu023 tmux: rl 
# # # baseline: CQL, QuantileNetwork
# CUDA_VISIBLE_DEVICES=0 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs/Seaquest/1%/CQL_ln \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Seaquest/1 \
#   --agent_name=quantile \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_seaquest.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=100' \
#   # --gin_bindings='FixedReplayQuantileAgent.adapted=True'
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"'
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Seaquest"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'


# # # 2023/06/10
# # # gpu022 tmux: rl2 
# # # baseline: CQL, QuantileNetwork
# # # action dim: 9
# CUDA_VISIBLE_DEVICES=2 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs/Asterix/1%/CQL_ln \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Pong/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_asterix.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=100' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Pong"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0'
#   # --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   # --gin_bindings='DQNAgent.entropy=0.8' \


# # # 2023/06/10
# # # gpu022 tmux: rl2 
# # # baseline: CQL, QuantileNetwork
# # # action dim: 9
# CUDA_VISIBLE_DEVICES=5 python -um batch_rl.fixed_replay.train \
#   --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs/Qbert/1%/CQL_ln_entropy0.4_copy \
#   --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
#   --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
#   --gin_bindings='FixedReplayRunner.num_iterations=101' \
#   --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
#   --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
#   --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
#   --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
#   --gin_bindings='DQNAgent.entropy=0.1' \
#   --gin_bindings='FixedReplayRunner.only_eval=True'


# # 2023/06/10
# # gpu026 tmux: rl2 
# # baseline: CQL, QuantileNetwork
# # action dim: 9
CUDA_VISIBLE_DEVICES=5 python -um batch_rl.fixed_replay.train \
  --base_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs/Qbert/1%/CQL_ln_entropy0.4_copy \
  --replay_dir=/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/datasets/Qbert/1 \
  --gin_files='batch_rl/fixed_replay/configs/quantile_qbert.gin' \
  --gin_bindings='FixedReplayRunner.num_iterations=101' \
  --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"' \
  --gin_bindings='atari_lib.create_atari_environment.game_name="Qbert"' \
  --gin_bindings='FixedReplayQuantileAgent.minq_weight=4.0' \
  --gin_bindings='FixedReplayQuantileAgent.adapted=True' \
  --gin_bindings='DQNAgent.entropy=0.1' \
  --gin_bindings='FixedReplayRunner.only_eval=True'