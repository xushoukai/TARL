#!/bin/bash
image_full_name=mirrors.tencent.com/zihaolian/offlinerl:latest
if [ -z ${image_full_name} ]
then
   echo "please input a image name. eg: mirrors.tencent.com/star_library/g-tlinux2.2-python3.6-cuda9.0-cudnn7.6-tf1.12:latest    \n"
   exit 0
fi

# cmd="/usr/bin/python -um /apdcephfs/private_zihaolian/zihaolian/code/CQL/batch_rl/fixed_replay/train.py --base_dir=/apdcephfs/private_zihaolian/zihaolian/CQL/logs/Pong/Quantitle --replay_dir /apdcephfs/private_zihaolian/zihaolian/datasets/Pong/1  --agent_name=quantile --gin_files='/apdcephfs/private_zihaolian/zihaolian/code/CQL/batch_rl/fixed_replay/configs/quantile.gin' --gin_bindings='FixedReplayRunner.num_iterations=1000' --gin_bindings='FixedReplayQuantileAgent.adapted=False' --gin_bindings='FixedReplayQuantileAgent.network="atari_helpers.QuantileLayerNormNetwork"'  "
cmd="cd /"

if [ -z "${cmd}" ]
then
   echo "please input your train command, eg: python3.6 /apdcephfs/private_YOURRTX/train/train.py  --dataset_dir=/apdcephfs/private_YOURRTX/data     \n"
   exit 0
fi

echo ${image_full_name}
echo ${cmd}

# source activate apsnet
#nvidia-docker run -it -e NVIDIA_VISIBLE_DEVICES=all --network=host -v /apdcephfs/:/apdcephfs/  ${image_full_name}  ${cmd}
docker run -it --gpus all --network=host -v /apdcephfs/:/apdcephfs/  ${image_full_name}  ${cmd}
