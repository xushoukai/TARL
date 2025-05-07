import numpy as np
import torch
import torch.nn as nn

from torch.nn import functional as F
from typing import Dict, Union, Tuple
from offlinerlkit.buffer import TestReplayBuffer
from offlinerlkit.policy import SACTTAPolicy


class MCQTTABUFFERPolicy(SACTTAPolicy):
    """
    Mildly Conservative Q-Learning <Ref: https://arxiv.org/abs/2206.04745>
    """

    def __init__(
        self,
        actor: nn.Module,
        tta_actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        behavior_policy: nn.Module,
        test_buffer: TestReplayBuffer,
        actor_optim: torch.optim.Optimizer,
        tta_actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        behavior_policy_optim: torch.optim.Optimizer,
        tau: float = 0.005,
        gamma: float  = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        lmbda: float = 0.7,
        num_sampled_actions: int = 10,
        klcoef: float = 0.1, 
        moment_tau: float = 0.005
    ) -> None:
        super().__init__(
            actor,
            tta_actor,
            critic1,
            critic2,
            actor_optim,
            tta_actor_optim,
            critic1_optim,
            critic2_optim,
            tau=tau,
            gamma=gamma,
            alpha=alpha,
            klcoef=klcoef,
            moment_tau=moment_tau
        )

        self.behavior_policy = behavior_policy
        self.behavior_policy_optim = behavior_policy_optim
        self.test_buffer = test_buffer
        self._lmbda = lmbda
        self._num_sampled_actions = num_sampled_actions
        self._klcoef = klcoef

    def learn(self, batch: Dict) -> Dict[str, float]:
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]

        # print(actions[0])
        # exit()
        
        # update behavior policy
        recon, mean, std = self.behavior_policy(obss, actions)
        recon_loss = F.mse_loss(recon, actions)
        KL_loss	= -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + KL_loss

        self.behavior_policy_optim.zero_grad()
        vae_loss.backward()
        self.behavior_policy_optim.step()

        # update critic
        with torch.no_grad():
            next_actions, next_log_probs = self.actforward(next_obss)
            next_q = torch.min(
                self.critic1_old(next_obss, next_actions), self.critic2_old(next_obss, next_actions)
            ) - self._alpha * next_log_probs
            target_q_for_in_actions = rewards + self._gamma * (1 - terminals) * next_q
        q1_in, q2_in = self.critic1(obss, actions), self.critic2(obss, actions)
        critic1_loss_for_in_actions = ((q1_in - target_q_for_in_actions).pow(2)).mean()
        critic2_loss_for_in_actions = ((q2_in - target_q_for_in_actions).pow(2)).mean()

        s_in = torch.cat([obss, next_obss], dim=0)
        with torch.no_grad():
            s_in_repeat = torch.repeat_interleave(s_in, self._num_sampled_actions, 0)
            sampled_actions = self.behavior_policy.decode(s_in_repeat)
            target_q1_for_ood_actions = self.critic1_old(s_in_repeat, sampled_actions).reshape(s_in.shape[0], -1).max(1)[0].reshape(-1, 1)
            target_q2_for_ood_actions = self.critic2_old(s_in_repeat, sampled_actions).reshape(s_in.shape[0], -1).max(1)[0].reshape(-1, 1)
            target_q_for_ood_actions = torch.min(target_q1_for_ood_actions, target_q2_for_ood_actions)
            ood_actions, _ = self.actforward(s_in)
        
        q1_ood, q2_ood = self.critic1(s_in, ood_actions), self.critic2(s_in, ood_actions)
        critic1_loss_for_ood_actions = ((q1_ood - target_q_for_ood_actions).pow(2)).mean()
        critic2_loss_for_ood_actions = ((q2_ood - target_q_for_ood_actions).pow(2)).mean()

        critic1_loss = self._lmbda * critic1_loss_for_in_actions + (1 - self._lmbda) * critic1_loss_for_ood_actions
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        critic2_loss = self._lmbda * critic2_loss_for_in_actions + (1 - self._lmbda) * critic2_loss_for_ood_actions
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        # update actor
        a, log_probs = self.actforward(obss)
        q1a, q2a = self.critic1(obss, a), self.critic2(obss, a)

        actor_loss = - torch.min(q1a, q2a).mean() + self._alpha * log_probs.mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_probs = log_probs.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_probs).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self._alpha = torch.clamp(self._log_alpha.detach().exp(), 0.0, 1.0)

        self._sync_weight()

        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
            "loss/behavior_policy": vae_loss.item()
        }

        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()

        return result
    
    def log_likelihood(self, obs, action):
        recon_action = self.behavior_policy(obs, action)
        log_likelihood = -torch.sum(F.binary_cross_entropy(recon_action, action, reduction='none'), dim=1)
        return log_likelihood.mean()

    # def adapetd_mcd_actforward(
    #     self,
    #     obs: np.ndarray,
    #     deterministic: bool = False
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     # ####################################################################################################################
    #     # # 0. add a replay buffer for action
    #     # ####################################################################################################################
    #     tta_dist = self.tta_actor(obs)
    #     if deterministic:
    #         tta_squashed_action, tta_raw_action = tta_dist.mode()
    #     else:
    #         tta_squashed_action, tta_raw_action = tta_dist.rsample()
    #     tta_log_prob = tta_dist.log_prob(tta_squashed_action, tta_raw_action)
    #     if self.test_buffer.size() >= 256:
    #         batch_obs, _, _, _, _ = self.test_buffer.sample(batch_size=64)
    #         batch_tta_dist = self.tta_actor(batch_obs)
    #         if deterministic:
    #             batch_squashed_action, batch_raw_action = batch_tta_dist.mode()
    #         else:
    #             batch_squashed_action, batch_raw_action = batch_tta_dist.rsample()

    #         batch_dist = self.actor(batch_obs)

    #         kl_loss = self.kl_divergence(batch_dist.mean, batch_dist.stddev, batch_tta_dist.mean, batch_tta_dist.stddev)
    #         q1a, q2a = self.critic1(batch_obs, batch_squashed_action), self.critic2(batch_obs, batch_squashed_action)
    #         # test_loss = batch_tta_dist.entropy()
    #         test_loss = (batch_tta_dist.variance).mean()
    #         self.tta_actor_optim.zero_grad()
    #         # (test_loss.mean() + self._klcoef * kl_loss - (torch.min(q1a, q2a)).mean()).backward()
    #         (test_loss.mean() - (torch.min(q1a, q2a)).mean()).backward()
    #         self.tta_actor_optim.step()

    #         # with torch.no_grad():
    #         #     # i = 0
    #         #     for ema_param, param in zip(self.ema_params, self.tta_actor.parameters()):
    #         #         # ema_param.data.copy_(ema_param.data * 0.99 + param.data * 0.01)
    #         #         ema_param.data.copy_(ema_param.data * 0.01 + param.data * 0.99)
    #         #         # if i == 0:
    #         #         #     print((ema_param[0]).data, (param[0]).data)
    #         #         # i += 1
    #         #         # ema_param.mul_(self._moment_tau).add_((1 - self._moment_tau) * param)
    #         #         # ema_param.mul_(self._moment_tau).add_((1 - self._moment_tau) * param)
    #         #         # ema_param.mul_(1- self._moment_tau).add_(self._moment_tau * param)

    #         #     for ema_param, param in zip(self.ema_params, self.tta_actor.parameters()):
    #         #         param.data.copy_(ema_param.data)

    #     log_likelihood_action = self.log_likelihood(obs, tta_squashed_action)
    #     # ####################################################################################################################
    #     # # 1. MC dropout; Q Network Uncertainty; eval_mcq_ln_before_activation_best_MCDropout_Qvalue_Uncertainty_lr1e-6
    #     # ####################################################################################################################
    #     # repeat_obs = np.repeat(obs, 10, axis=0)
    #     # repeat_tta_dist = self.tta_actor(repeat_obs)
    #     # # dist = self.actor(obs)
    #     # if deterministic:
    #     #     repeat_tta_squashed_action, repeat_tta_raw_action = repeat_tta_dist.mode()
    #     # else:
    #     #     repeat_tta_squashed_action, repeat_tta_raw_action = repeat_tta_dist.rsample()
    #     # repeat_tta_log_prob = repeat_tta_dist.log_prob(repeat_tta_squashed_action, repeat_tta_raw_action)
    #     # tta_squashed_action = torch.mean(repeat_tta_squashed_action, dim=0)

    #     # q1a, q2a = self.critic1(repeat_obs, repeat_tta_squashed_action), self.critic2(repeat_obs, repeat_tta_squashed_action)
    #     # q_var = torch.var(q1a + q2a, dim=0)

    #     # self.tta_actor_optim.zero_grad()
    #     # (-(torch.min(q1a, q2a) / q_var).mean()).backward()
    #     # self.tta_actor_optim.step()

    #     # ###################################################################################################################
    #     # # 2. MC dropout; Action Uncertainty; eval_mcq_ln_before_activation_best_MCDropout_Action_Uncertainty_lr1e-6
    #     # ###################################################################################################################
    #     # repeat_obs = np.repeat(obs, 10, axis=0)
    #     # repeat_tta_dist = self.tta_actor(repeat_obs)
    #     # dist = self.actor(obs)
    #     # # if deterministic:
    #     # #     repeat_tta_squashed_action, repeat_tta_raw_action = repeat_tta_dist.mode()
    #     # # else:
    #     # action, _raw_action = dist.rsample()
    #     # # print("origin: ", action)
    #     # repeat_tta_squashed_action, repeat_tta_raw_action = repeat_tta_dist.rsample()
    #     # # print("tta: ", repeat_tta_squashed_action[0])
    #     # # print(repeat_tta_squashed_action)
    #     # repeat_tta_log_prob = repeat_tta_dist.log_prob(repeat_tta_squashed_action, repeat_tta_raw_action)
    #     # action_var = torch.var(repeat_tta_squashed_action, dim=0)
    #     # tta_squashed_action = torch.mean(repeat_tta_squashed_action, dim=0)

    #     # # # 4. Uncertainty loss minimize
    #     # # # tta_entropy_loss = (tta_dist.entropy()).abs()
    #     # # # kl_loss = self.kl_divergence(dist.mean, dist.stddev, repeat_tta_dist.mean, repeat_tta_dist.stddev)
    #     # q1a, q2a = self.critic1(repeat_obs, repeat_tta_squashed_action), self.critic2(repeat_obs, repeat_tta_squashed_action)

    #     # q_value = (torch.min(q1a, q2a)).mean()
    #     # # print("tta q value: ", (torch.min(q1a, q2a)))
    #     # # i = 0 
    #     # # for param in self.tta_actor.parameters():
    #     # #     if i == 0:
    #     # #         print("before: ", (param[0]).data)
    #     # #     i += 1

    #     # self.tta_actor_optim.zero_grad()
    #     # # (action_var.mean() + self._klcoef * kl_loss.mean()).backward()
    #     # # (action_var.mean() - torch.min(q1a, q2a).mean()).backward()
    #     # # (action_var.mean() - self._klcoef * (torch.min(q1a, q2a) / action_var).mean()).backward()
    #     # (- (torch.min(q1a, q2a) / (action_var + 1e-8)).mean() + self._klcoef * (action - tta_squashed_action).mean()).backward()
    #     # self.tta_actor_optim.step()

    #     # i = 0 
    #     # for param in self.tta_actor.parameters():
    #     #     if i == 0:
    #     #         print("after: ", (param[0]).data)
    #     #     i += 1

    #     #  # menent update tta_actor
    #     # with torch.no_grad():
    #     #     i = 0
    #     #     for ema_param, param in zip(self.ema_params, self.tta_actor.parameters()):
    #     #         ema_param.data.copy_(ema_param.data * 0.001 + param.data * 0.999)
    #     #         if i == 0:
    #     #             print((ema_param[0]).data, (param[0]).data)
    #     #         i += 1
    #     #         # ema_param.mul_(self._moment_tau).add_((1 - self._moment_tau) * param)
    #     #         # ema_param.mul_(self._moment_tau).add_((1 - self._moment_tau) * param)
    #     #         # ema_param.mul_(1- self._moment_tau).add_(self._moment_tau * param)

    #     #     for ema_param, param in zip(self.ema_params, self.tta_actor.parameters()):
    #     #         param.data.copy_(ema_param.data)

    #     # ###################################################################################################################
    #     # # 2. Minimize variance
    #     # ####################################################################################################################
    #     # tta_dist = self.tta_actor(obs)
    #     # if deterministic:
    #     #     tta_squashed_action, tta_raw_action = tta_dist.mode()
    #     # else:
    #     #     tta_squashed_action, tta_raw_action = tta_dist.rsample()
    #     # tta_log_prob = tta_dist.log_prob(tta_squashed_action, tta_raw_action)

    #     # loss = (tta_dist.variance).mean()
    #     # self.tta_actor_optim.zero_grad()
    #     # loss.backward()
    #     # self.tta_actor_optim.step()


    #     # # 0. replay_buffer
    #     # self.replay_buffer.append(obs)
    #     # if len(self.replay_buffer) == 32:
    #     #     replay_obs = np.squeeze(np.array(self.replay_buffer))
    #     #     replay_dist = self.tta_actor(replay_obs)
    #     #     replay_tta_squashed_action, replay_tta_raw_action = replay_dist.mode()

    #     #     # update behavior policy
    #     #     dataset_sampled_actions = self.behavior_policy.decode(torch.tensor(replay_obs, device='cuda:0', dtype=torch.float32))

    #     #     replay_obs = np.squeeze(np.array(self.replay_buffer))
    #     #     replay_dist = self.tta_actor(replay_obs)
    #     #     replay_tta_squashed_action, replay_tta_raw_action = replay_dist.mode()
    #     #     q1a, q2a = self.critic1(replay_obs, replay_tta_squashed_action), self.critic2(replay_obs, replay_tta_squashed_action)
    #     #     q1a_in, q2a_in = self.critic1(replay_obs, dataset_sampled_actions), self.critic2(replay_obs, dataset_sampled_actions)

    #     #     critic1_loss_for_ood_actions = ((q1a - q1a_in).pow(2)).mean()
    #     #     critic2_loss_for_ood_actions = ((q2a - q2a_in).pow(2)).mean()

    #     #     self.tta_actor_optim.zero_grad()
    #     #     (torch.min(critic1_loss_for_ood_actions, critic2_loss_for_ood_actions).mean() - torch.min(q1a, q2a).mean()).backward()
    #     #     self.tta_actor_optim.step()

    #     #     self.replay_buffer = []


    #     # # 2. augmentation action
    #     # new_obs = np.add(obs, 1 * np.random.normal(0, 1, obs.shape))
    #     # tta_dist = self.tta_actor(obs)
    #     # augmentatoin_tta_dist = self.tta_actor(new_obs)
    #     # if deterministic:
    #     #     tta_squashed_action, tta_raw_action = tta_dist.mode()
    #     #     augmentatoin_tta_squashed_action,  augmentatoin_tta_raw_action = augmentatoin_tta_dist.mode()
    #     # else:
    #     #     tta_squashed_action, tta_raw_action = tta_dist.rsample()
    #     #     augmentatoin_tta_squashed_action,  augmentatoin_tta_raw_action = augmentatoin_tta_dist.rsample()
    #     # tta_log_prob = tta_dist.log_prob(tta_squashed_action, tta_raw_action)

    #     # action_var = torch.mean((tta_squashed_action - augmentatoin_tta_squashed_action)**2)
        
    #     # self.tta_actor_optim.zero_grad()
    #     # # (action_var.mean() + self._klcoef * kl_loss.mean()).backward()
    #     # (action_var.mean()).backward()
    #     # self.tta_actor_optim.step()

    #     # # menent update tta_actor
    #     # with torch.no_grad():
    #     #     for ema_param, param in zip(self.ema_params, self.tta_actor.parameters()):
    #     #         ema_param.mul_(self._moment_tau).add_((1 - self._moment_tau) * param)

    #     #     for ema_param, param in zip(self.ema_params, self.tta_actor.parameters()):
    #     #         param.copy_(ema_param)

    #     # return tta_squashed_action, tta_log_prob, tta_entropy_loss.sum()
    #     return tta_squashed_action, tta_log_prob, log_likelihood_action
    #     # return tta_squashed_action, tta_log_prob, tta_log_prob