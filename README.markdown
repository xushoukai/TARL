# Test-Time Adapted Reinforcement Learning with Action Entropy Regularization

## Abstract

Offline reinforcement learning is widely applied in multiple fields due to its efficiency and risk‐control advantages. However, it suffers from distribution shift between offline datasets and online environments, producing out‐of‐distribution (OOD) state–action pairs beyond the training data. As a result, conservative training policies may fail when the test environment deviates substantially from the offline dataset.

We propose **Test‐time Adapted Reinforcement Learning (TARL)** to address this issue. TARL constructs unsupervised test‐time objectives for both discrete and continuous control tasks using only test data, without relying on environmental rewards. For discrete tasks, it minimizes the entropy of predicted action probabilities to reduce uncertainty and avoid OOD actions; for continuous tasks, it models action uncertainty via the policy network’s output distribution and minimizes its variance.

To mitigate bias from overfitting and error accumulation during test‐time updates, TARL imposes a KL‐divergence constraint between the fine‐tuned and original policies. For efficiency, only layer‐normalization parameters are updated at test time.

Extensive experiments on Atari benchmarks and the D4RL dataset demonstrate TARL’s effectiveness, yielding a 13.6% relative improvement in episode return over CQL on the hopper-expert-v2 task.

![image-20250515201733637](images/image-20250515201733637.png)

## Installation

- **D4RL Environment**  
  This project builds on the OfflineRL-Kit framework. Please follow the instructions in the [OfflineRL-Kit repository](https://github.com/yihaosun1124/OfflineRL-Kit/tree/main) to install the D4RL and MuJoCo environments. 

- **Atari Environment**  
  Environment setup and dataset preparation follow the guidelines in the [google-research/batch_rl](https://github.com/google-research/batch_rl) project. :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}

## Usage

### TARL on Discrete Control Tasks
- Navigate to `TARL/CQL/atari/` and run:  
  ```bash
  ./run_atari.sh
  ```

  Modify the dataset path within the script to train on different games.
- Parameter settings are based on the [CQL implementation](https://github.com/aviralkumar2907/CQL) and can be adjusted via Gin bindings.
- Example command:

  ```bash
  python -um batch_rl.fixed_replay.train \
    --base_dir=/tmp/batch_rl \
    --replay_dir=$DATA_DIR/Pong/1 \
    --agent_name=quantile \
    --gin_files='batch_rl/fixed_replay/configs/quantile_pong.gin' \
    --gin_bindings='FixedReplayRunner.num_iterations=1000' \
    --gin_bindings='atari_lib.create_atari_environment.game_name = "Pong"'
    --gin_bindings='FixedReplayQuantileAgent.minq_weight=1.0'
  ```

**Results:**

![image-20250515191658748](images/image-20250515191658748.png)

### TARL on Continuous Control Tasks

- Training 
  
  Train a CQL model with pre-activation layer normalization on the Walker2d task:
  
  ```bash
  CUDA_VISIBLE_DEVICES=0 python TARL/OfflineRL-Kit/run_example/run_cql_ln_before_activation.py \
      --task walker2d-medium-v2 \
      --algo-name cql_ln_before_activation_best_and_last10episode
  ```
  
- Evaluation 

  Evaluate the trained model with test-time adaptation using a specified checkpoint:
  ```bash
  CUDA_VISIBLE_DEVICES=0 python TARL/OfflineRL-Kit/run_example/run_eval_cql_ln_before_activation.py \
      --task walker2d-medium-v2 \
      --algo-name eval_cql_ln_before_activation_best_lr1e-6_buffer10sample1 \
      --tta-actor-lr 1e-6 \
      --eval_episodes 10 \
      --only-eval-tta False \
      --checkpoints "${CHECKPOINT_DIR}"
  ```

- Sample Training Log

```bash
----------------------------------------------------------------------------------
| eval/episode_length                | 963      |
| eval/episode_length_std            | 110      |
| eval/normalized_episode_reward     | 83       |
| eval/normalized_episode_reward_std | 9.85     |
| loss/actor                         | -3.56    |
| loss/q1                            | 0.587    |
| loss/q2                            | 0.608    |
| loss/v                             | 0.16     |
| timestep                           | 1000000  |
----------------------------------------------------------------------------------
total time: 14475.79s
best_reward_mean: 88.23, best_reward_std: 1.86
last_10_performance: 81.98
```
**Results:**

<img src="images/image-20250515202427217.png" alt="image-20250515202427217" style="zoom:50%;" />

## Directory Structure
```
├── TRANSFER_RL-MAIN/
│   ├── CQL/
│   │   ├── atari/
│   │   ├── dopamine/
│   │   ├── online/
│   │   ├── CONTRIBUTING.md
│   │   ├── LICENSE
│   │   ├── process_result.py
│   │   ├── run_atari.sh							      # Discrete-environment testing script
│   │   └── run_result.py
├── OfflineRL-Kit/
│   ├── run_example/								      # our project modifications
│   │   ├── plotter.py
│   │   ├── run_cql.py
│   │   ├── run_cql_ln_before_activation.py  		      # Implementation of  CQL modification
│   │   └── run_eval_cql_ln_before_activation.py	      # Evaluation code 
│   ├── scripts/
│   │   └── run_walker2d-full-replay.sh				      # Experiment script 
│   ├── tune_example/
│   ├── LICENSE
│   ├── offlinerl_config.json
│   ├── README.md
│   └── setup.py
├── .gitignore
└── requirements.txt

```

