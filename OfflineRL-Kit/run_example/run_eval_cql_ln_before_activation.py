import argparse
import random

import gym
import d4rl

import numpy as np
import torch

import sys 
sys.path.append("/lichenghao/lch/transfer_rl-main/OfflineRL-Kit") 
from offlinerlkit.nets import MLP_LN_Before_Activation, MLP_LN_Before_Activation_No_Dropout
from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian
from offlinerlkit.utils.load_dataset import qlearning_dataset
# import d4rl
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.policy_trainer import TTAMFPolicyTrainer
from offlinerlkit.policy import CQLTTAPolicy


"""
suggested hypers
cql-weight=5.0, temperature=1.0 for all D4RL-Gym tasks
"""
# from loguru import logger
# logger.remove()
# logger.add(sys.stderr, level="WARNING")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="eval_cql_ln_before_activation_best_debug")
    parser.add_argument("--task", type=str, default="antmaze-medium-diverse-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoints", type=str, default="/lichenghao/lch/transfer_rl-main/OfflineRL-Kit/log/antmaze-medium-diverse-v0/cql_ln_before_activation_best/seed_0&timestamp_25-0330-120959/checkpoint/best_policy.pth")
    parser.add_argument("--hidden-dims", type=int, nargs='*', default=[256, 256, 256])
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--tta-actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--klcoef", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--target-entropy", type=int, default=None)
    parser.add_argument("--auto-alpha", default=True)
    parser.add_argument("--alpha-lr", type=float, default=1e-4)

    parser.add_argument("--cql-weight", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-q-backup", type=bool, default=False)
    parser.add_argument("--deterministic-backup", type=bool, default=True)
    parser.add_argument("--with-lagrange", type=bool, default=False)
    parser.add_argument("--lagrange-threshold", type=float, default=10.0)
    parser.add_argument("--cql-alpha-lr", type=float, default=3e-4)
    parser.add_argument("--num-repeat-actions", type=int, default=10)
    
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--only-eval-tta", type=bool, default=False)
    parser.add_argument("--weight-sample", type=bool, default=False)
    parser.add_argument("--is_entropy_filter", type=bool, default=False)
    parser.add_argument("--entropy_threshold", type=float, default=0.1)

    return parser.parse_args()


def evaluate(args=get_args()):
    # create env and dataset
    env = gym.make(args.task)
    dataset = d4rl.qlearning_dataset(env)
    print(len(dataset["observations"]))
    # print((dataset["actions"]).shape())
    # print((dataset["next_observations"]).shape())
    # print((dataset["rewards"]).shape())
    # print((dataset["terminals"]).shape())
    # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
    if 'antmaze' in args.task:
        dataset["rewards"] = (dataset["rewards"] - 0.5) * 4.0
    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)
    args.max_action = env.action_space.high[0]

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    env.seed(args.seed)

    # create policy model
    tta_actor_backbone = MLP_LN_Before_Activation_No_Dropout(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims, dropout_rate=0.1)
    actor_backbone = MLP_LN_Before_Activation(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims)
    critic1_backbone = MLP_LN_Before_Activation(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
    critic2_backbone = MLP_LN_Before_Activation(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True
    )
    tta_dist = TanhDiagGaussian(
        latent_dim=getattr(tta_actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True
    )
    actor = ActorProb(actor_backbone, dist, args.device)
    tta_actor = ActorProb(tta_actor_backbone, tta_dist, args.device)
    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    tta_actor_optim = torch.optim.Adam(tta_actor.parameters(), lr=args.tta_actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = args.target_entropy if args.target_entropy \
            else -np.prod(env.action_space.shape)

        args.target_entropy = target_entropy

        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = args.alpha

    # create policy
    policy = CQLTTAPolicy(
        actor,
        tta_actor,
        critic1,
        critic2,
        actor_optim,
        tta_actor_optim,
        critic1_optim,
        critic2_optim,
        action_space=env.action_space,
        tau=args.tau,
        gamma=args.gamma,
        alpha=alpha,
        klcoef=args.klcoef,
        cql_weight=args.cql_weight,
        temperature=args.temperature,
        max_q_backup=args.max_q_backup,
        deterministic_backup=args.deterministic_backup,
        with_lagrange=args.with_lagrange,
        lagrange_threshold=args.lagrange_threshold,
        cql_alpha_lr=args.cql_alpha_lr,
        num_repeart_actions=args.num_repeat_actions
    )

    # create buffer
    buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    buffer.load_dataset(dataset)

    # log
    log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args))
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))

    # reload best model checkpoints
    policy.load_state_dict(torch.load(args.checkpoints), strict=False)

    # create policy trainer
    policy_trainer = TTAMFPolicyTrainer(
        policy=policy,
        eval_env=env,
        buffer=buffer,
        logger=logger,
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        batch_size=args.batch_size,
        eval_episodes=args.eval_episodes,
        checkpoints=args.checkpoints
    )

    # train
    policy_trainer.evaluate(args.seed, args.only_eval_tta, args.weight_sample, val_epoch=10, entropy_threshold=args.entropy_threshold, is_entropy_filter=args.is_entropy_filter)


if __name__ == "__main__":
    evaluate()