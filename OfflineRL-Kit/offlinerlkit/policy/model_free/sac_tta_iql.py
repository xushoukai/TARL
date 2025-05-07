import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from copy import deepcopy
from collections import deque
from typing import Dict, Union, Tuple
from offlinerlkit.policy import BasePolicy


class SACTTAPolicy(BasePolicy):
    """
    Soft Actor Critic <Ref: https://arxiv.org/abs/1801.01290>
    """

    def __init__(
        self,
        actor: nn.Module,
        tta_actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        actor_optim: torch.optim.Optimizer,
        tta_actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        tau: float = 0.005,
        gamma: float  = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        klcoef: float = 0.1, 
        moment_tau: float = 0.005
    ) -> None:
        super().__init__()

        self.actor = actor
        self.tta_actor = tta_actor
        self.critic1, self.critic1_old = critic1, deepcopy(critic1)
        self.critic1_old.eval()
        self.critic2, self.critic2_old = critic2, deepcopy(critic2)
        self.critic2_old.eval()

        self.actor_optim = actor_optim
        self.tta_actor_optim = tta_actor_optim
        self.critic1_optim = critic1_optim
        self.critic2_optim = critic2_optim

        self._tau = tau 
        self._gamma = gamma
        self._klcoef = klcoef
        self._moment_tau = moment_tau
        self.replay_buffer = deque(maxlen=32)
        self.learnable_var = nn.Parameter(torch.Tensor(1, device='cpu'))
        # self.tta_buffer = deque(maxlen=256)
        self.tta_buffer = []

        self.actions_list = []
        self.states_list = []

        self._is_auto_alpha = False
        if isinstance(alpha, tuple):
            self._is_auto_alpha = True
            self._target_entropy, self._log_alpha, self.alpha_optim = alpha
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = alpha

        # copy parameters from tta_actor
        self.ema_params = [param.clone().detach() for param in self.tta_actor.parameters()]

    def train(self) -> None:
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

    def eval(self) -> None:
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()

    def _sync_weight(self) -> None:
        for o, n in zip(self.critic1_old.parameters(), self.critic1.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic2_old.parameters(), self.critic2.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
    
    def _copy_weight(self) -> None:
        for o, n in zip(self.tta_actor.parameters(), self.actor.parameters()):
            o.data.copy_(n.data)

    def _moment_update_actor(self) -> None:
        for o, n in zip(self.tta_actor.parameters(), self.actor.parameters()):
            o.data.copy_(n.data)

    def actforward_dropout(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.actor(obs)
        if deterministic:
            squashed_action, raw_action = dist.mode()
        else:
            squashed_action, raw_action = dist.rsample()
        log_prob = dist.log_prob(squashed_action, raw_action)

        return squashed_action, log_prob, dist

    def actforward(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.actor(obs)
        if deterministic:
            squashed_action, raw_action = dist.mode()
        else:
            squashed_action, raw_action = dist.rsample()
        log_prob = dist.log_prob(squashed_action, raw_action)

        # print("norm:", dist.entropy())

        # actor_loss = ((dist.entropy()).abs()).sum()
        # actor_loss = (dist.entropy()).sum()

        # q1a, q2a = self.critic1(obs, squashed_action), self.critic2(obs, squashed_action)
        # q_value = (torch.min(q1a, q2a)).mean()

        # recon, mean, std = self.behavior_policy(torch.tensor(obs, device=squashed_action.device, dtype=torch.float32), squashed_action)
        # recon_loss = F.mse_loss(recon, squashed_action)
        # KL_loss	= -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        # vae_loss = recon_loss + KL_loss

        # return squashed_action, log_prob, vae_loss
        return squashed_action, log_prob, log_prob

    def kl_divergence(self, mu1, sigma1, mu2, sigma2):
        # compute KL divergence
        var1 = sigma1**2
        var2 = sigma2**2
        kl = torch.log(var2 / var1) + (var1 + (mu1 - mu2)**2) / (2 * var2) - 0.5
        return kl.mean()

    def adapetd_actforward(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tta_dist = self.tta_actor(obs)
        # dist = self.actor(obs)
        if deterministic:
            tta_squashed_action, tta_raw_action = tta_dist.mode()
            # squashed_action, raw_action = dist.mode()
        else:
            tta_squashed_action, tta_raw_action = tta_dist.rsample()
            # squashed_action, raw_action = dist.rsample()
        tta_log_prob = tta_dist.log_prob(tta_squashed_action, tta_raw_action)

        # 4. entropy maximize
        tta_entropy_loss = (tta_dist.entropy()).abs()
        self.tta_actor_optim.zero_grad()
        (tta_entropy_loss.mean()).backward()
        self.tta_actor_optim.step()

        # # menent update tta_actor
        # with torch.no_grad():
        #     for ema_param, param in zip(self.ema_params, self.tta_actor.parameters()):
        #         ema_param.mul_(self._moment_tau).add_((1 - self._moment_tau) * param)

        #     for ema_param, param in zip(self.ema_params, self.tta_actor.parameters()):
        #         param.copy_(ema_param)

         # 5. entropy maximize + Q value maximize
        # tta_entropy_loss = (tta_dist.entropy()).abs()
        # q1a, q2a = self.critic1(obs, tta_squashed_action), self.critic2(obs, tta_squashed_action)
        # self.tta_actor_optim.zero_grad()
        # (- tta_entropy_loss.mean() - torch.min(q1a, q2a).mean() * 10.0).backward()
        # self.tta_actor_optim.step()

        # # 1. entropy minize
        # # tta_entropy_loss = (tta_dist.entropy()).sum()
        # tta_entropy_loss = (tta_dist.entropy()).abs()
        # kl_loss = self.kl_divergence(dist.mean, dist.stddev, tta_dist.mean, tta_dist.stddev)
        # # print(tta_entropy_loss)

        # flag = tta_entropy_loss.lt(0.25).all()
        # if flag:
        #     print(tta_entropy_loss)
        #     self.tta_actor_optim.zero_grad()
        #     (tta_entropy_loss.mean() + self._klcoef * kl_loss).backward()
        #     self.tta_actor_optim.step()

        # # 2. entropy minize and kl divergence
        # tta_entropy_loss = (tta_dist.entropy()).sum()
        # kl_loss = self.kl_divergence(dist.mean, dist.stddev, tta_dist.mean, tta_dist.stddev)
        # self.tta_actor_optim.zero_grad()
        # (tta_entropy_loss + self._klcoef * kl_loss).backward()
        # self.tta_actor_optim.step()

        # # 3. maximize q value 
        # q1a, q2a = self.critic1(obs, tta_squashed_action), self.critic2(obs, tta_squashed_action)
        # kl_loss = self.kl_divergence(dist.mean, dist.stddev, tta_dist.mean, tta_dist.stddev)
        # # tta_entropy_loss = (tta_dist.entropy()).sum()
        # actor_loss = - torch.min(q1a, q2a).mean() + self._klcoef * kl_loss
        # self.tta_actor_optim.zero_grad()
        # actor_loss.backward()
        # self.tta_actor_optim.step()
        # # if tta_entropy_loss < 1.0:
        # #     self.tta_actor_optim.zero_grad()
        # #     actor_loss.backward()
        # #     self.tta_actor_optim.step()

        return tta_squashed_action, tta_log_prob, tta_entropy_loss.sum()

    def adapetd_mcd_actforward(
        self,
        obs: np.ndarray,
        weight_sample: bool = False,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ####################################################################################################################
        # 0. importance sampling
        ####################################################################################################################
        tta_dist = self.tta_actor(obs)
        if deterministic:
            tta_squashed_action, tta_raw_action = tta_dist.mode()
        else:
            tta_squashed_action, tta_raw_action = tta_dist.rsample()
        tta_log_prob = tta_dist.log_prob(tta_squashed_action, tta_raw_action)
        old_action_dists = self.actor(obs)

        # new_action_dists = torch.distributions.Normal(tta_dist.mean, tta_dist.stddev)
        # old_action_dists = torch.distributions.Normal(old_action_dists.mean, old_action_dists.stddev)

        kl_div = torch.mean(
            torch.distributions.kl.kl_divergence(old_action_dists, tta_dist))
        
        entropy = (tta_dist.entropy()).mean()

        # if entropy < 0.1:
        self.tta_actor_optim.zero_grad()
        (self._klcoef * kl_div.mean() + entropy).backward()
        self.tta_actor_optim.step()

        # ####################################################################################################################
        # recon, mean, std = self.behavior_policy(torch.tensor(obs, device=tta_squashed_action.device, dtype=torch.float32), tta_squashed_action)
        # recon_loss = F.mse_loss(recon, tta_squashed_action)
        # KL_loss	= -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        # vae_loss = recon_loss + KL_loss

        # ####################################################################################################################
        # # 1. MC dropout; Q Network Uncertainty; eval_mcq_ln_before_activation_best_MCDropout_Qvalue_Uncertainty_lr1e-6
        # ####################################################################################################################
        # print(tta_dist.stddev)
        # if (tta_dist.stddev <= 0.01).all():
        #     print("-" * 10)
        # repeat_obs = np.repeat(obs, 10, axis=0)
        # repeat_tta_dist = self.tta_actor(repeat_obs)
        # # dist = self.actor(obs)
        # if deterministic:
        #     repeat_tta_squashed_action, repeat_tta_raw_action = repeat_tta_dist.mode()
        #     # print(repeat_tta_squashed_action)
        # else:
        #     repeat_tta_squashed_action, repeat_tta_raw_action = repeat_tta_dist.rsample()
        # repeat_tta_log_prob = repeat_tta_dist.log_prob(repeat_tta_squashed_action, repeat_tta_raw_action)
        # # tta_squashed_action = torch.mean(repeat_tta_squashed_action, dim=0)

        # q1a, q2a = self.critic1(repeat_obs, repeat_tta_squashed_action), self.critic2(repeat_obs, repeat_tta_squashed_action)
        # q_var = torch.var(q1a + q2a, dim=0)

        # self.tta_actor_optim.zero_grad()
        # (-(torch.min(q1a, q2a) / q_var).mean()).backward()
        # self.tta_actor_optim.step()

        # ###################################################################################################################
        # # 2. MC dropout; Action Uncertainty; eval_mcq_ln_before_activation_best_MCDropout_Action_Uncertainty_lr1e-6
        # ###################################################################################################################
        # repeat_obs = np.repeat(obs, 10, axis=0)
        # repeat_tta_dist = self.tta_actor(repeat_obs)
        # old_action_dists = self.actor(repeat_obs)
        # if deterministic:
        #     repeat_tta_squashed_action, repeat_tta_raw_action = repeat_tta_dist.mode()
        # else:
        #     repeat_tta_squashed_action, repeat_tta_raw_action = repeat_tta_dist.rsample()
        # repeat_tta_log_prob = repeat_tta_dist.log_prob(repeat_tta_squashed_action, repeat_tta_raw_action)
        # action_var = torch.var(repeat_tta_squashed_action, dim=0)
        # tta_squashed_action = torch.mean(repeat_tta_squashed_action, dim=0)

        # tta_action_dists = torch.distributions.Normal(repeat_tta_dist.mean, repeat_tta_dist.stddev)
        # action_dists = torch.distributions.Normal(old_action_dists.mean, old_action_dists.stddev)

        # kl_div = torch.mean(
        #     torch.distributions.kl.kl_divergence(tta_action_dists, action_dists))

        # # # 4. Uncertainty loss minimize
        # # # tta_entropy_loss = (tta_dist.entropy()).abs()
        # # # kl_loss = self.kl_divergence(dist.mean, dist.stddev, repeat_tta_dist.mean, repeat_tta_dist.stddev)
        # q1a, q2a = self.critic1(repeat_obs, repeat_tta_squashed_action), self.critic2(repeat_obs, repeat_tta_squashed_action)
        # self.tta_actor_optim.zero_grad()
        # # (action_var.mean() + self._klcoef * kl_loss.mean()).backward()
        # # (action_var.mean() - torch.min(q1a, q2a).mean()).backward()
        # # (action_var.mean() - self._klcoef * (torch.min(q1a, q2a) / action_var).mean()).backward()
        # (- (torch.min(q1a, q2a) / (action_var + 1e-8)).mean() + self._klcoef * kl_div.mean()).backward()
        # self.tta_actor_optim.step()

        # behavior_action_dists = self.actor(obs)
        # if deterministic:
        #     behavior_squashed_action, behavior_raw_action = behavior_action_dists.mode()
        # else:
        #     behavior_squashed_action, behavior_raw_action = repeat_tta_dist.rsample()
        # recon, mean, std = self.behavior_policy(torch.tensor(obs, device=behavior_squashed_action.device, dtype=torch.float32), behavior_squashed_action)
        # recon_loss = F.mse_loss(recon, behavior_squashed_action)
        # KL_loss	= -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        # vae_loss = recon_loss + KL_loss
        
        # print("tta q value: ", (torch.min(q1a, q2a)))
        # i = 0 
        # for param in self.tta_actor.parameters():
        #     if i == 0:
        #         print("before: ", (param[0]).data)
        #     i += 1

        # self.tta_actor_optim.zero_grad()
        # # (action_var.mean() + self._klcoef * kl_loss.mean()).backward()
        # # (action_var.mean() - torch.min(q1a, q2a).mean()).backward()
        # # (action_var.mean() - self._klcoef * (torch.min(q1a, q2a) / action_var).mean()).backward()
        # (- (torch.min(q1a, q2a) / (action_var + 1e-8)).mean() + self._klcoef * (action - tta_squashed_action).mean()).backward()
        # self.tta_actor_optim.step()

        # i = 0 
        # for param in self.tta_actor.parameters():
        #     if i == 0:
        #         print("after: ", (param[0]).data)
        #     i += 1

        #  # menent update tta_actor
        # with torch.no_grad():
        #     i = 0
        #     for ema_param, param in zip(self.ema_params, self.tta_actor.parameters()):
        #         ema_param.data.copy_(ema_param.data * 0.001 + param.data * 0.999)
        #         if i == 0:
        #             print((ema_param[0]).data, (param[0]).data)
        #         i += 1
        #         # ema_param.mul_(self._moment_tau).add_((1 - self._moment_tau) * param)
        #         # ema_param.mul_(self._moment_tau).add_((1 - self._moment_tau) * param)
        #         # ema_param.mul_(1- self._moment_tau).add_(self._moment_tau * param)

        #     for ema_param, param in zip(self.ema_params, self.tta_actor.parameters()):
        #         param.data.copy_(ema_param.data)

        # ###################################################################################################################
        # # 2. Minimize variance
        # ####################################################################################################################
        # tta_dist = self.tta_actor(obs)
        # action_dist = self.actor(obs)
        # if deterministic:
        #     tta_squashed_action, tta_raw_action = tta_dist.mode()
        # else:
        #     tta_squashed_action, tta_raw_action = tta_dist.rsample()
        # tta_log_prob = tta_dist.log_prob(tta_squashed_action, tta_raw_action)

        # # variance = (tta_dist.variance).mean()
        # entropy = (tta_dist.entropy()).sum()
        # if entropy < 1:
        #     tta_action_dists = torch.distributions.Normal(tta_dist.mean, tta_dist.stddev)
        #     action_dists = torch.distributions.Normal(action_dist.mean, action_dist.stddev)
        #     kl_div = torch.mean(
        #         torch.distributions.kl.kl_divergence(tta_action_dists, action_dists))

        #     # loss = (tta_dist.variance).mean() + self._klcoef * kl_div 
        #     loss = entropy + self._klcoef * kl_div 
        #     # loss = entropy
        #     # loss = (tta_dist.variance).mean()
        #     self.tta_actor_optim.zero_grad()
        #     loss.backward()
        #     self.tta_actor_optim.step()
    

        # ###################################################################################################################
        # # 2. Minimize variance, variance with resample
        # ####################################################################################################################
        # tta_dist = self.tta_actor(obs)
        # action_dist = self.actor(obs)
        # tta_squashed_action, tta_raw_action = tta_dist.rsample()
        # tta_log_prob = tta_dist.log_prob(tta_squashed_action, tta_raw_action)

        # # variance = (tta_dist.variance).mean()
        # # entropy = (tta_dist.entropy()).sum()
        # variance_loss = (tta_dist.stddev).mean()
        # print(tta_dist.stddev)
        # # if entropy < 0.1:
        # if variance_loss < 0.2:
        #     print("-" * 50)
        #     tta_action_dists = torch.distributions.Normal(tta_dist.mean, tta_dist.stddev)
        #     action_dists = torch.distributions.Normal(action_dist.mean, action_dist.stddev)
        #     kl_div = torch.mean(
        #         torch.distributions.kl.kl_divergence(tta_action_dists, action_dists))

        #     # loss = (tta_dist.variance).mean() + self._klcoef * kl_div 
        #     # loss = entropy + self._klcoef * kl_div 
        #     # loss = variance_loss
        #     loss = variance_loss + self._klcoef * kl_div
        #     # loss = entropy + self._klcoef * kl_div
        #     # loss = (tta_dist.variance).mean()
        #     # loss = (tta_dist.variance).mean()
        #     self.tta_actor_optim.zero_grad()
        #     loss.backward()
        #     self.tta_actor_optim.step()

        # ###################################################################################################################
        # # 2. RDropout
        # ####################################################################################################################
        # repeat_obs = np.repeat(obs, 2, axis=0)
        # tta_dist = self.tta_actor(repeat_obs)
        # # action_dist = self.actor(obs)
        # if deterministic:
        #     tta_squashed_action, tta_raw_action = tta_dist.mode()
        # else:
        #     tta_squashed_action, tta_raw_action = tta_dist.rsample()

        # top_tta_action_dists = torch.distributions.Normal((tta_dist.mean)[0], (tta_dist.stddev)[0])
        # top_action_dists = torch.distributions.Normal((tta_dist.mean)[1], (tta_dist.stddev)[1])
        # p_q_kl_div = torch.mean(
        #     torch.distributions.kl.kl_divergence(top_tta_action_dists, top_action_dists))
        # q_p_kl_div = torch.mean(
        #     torch.distributions.kl.kl_divergence(top_action_dists, top_tta_action_dists))
        
        # # RDropout 更新
        # loss_fn2 = torch.nn.MSELoss(reduction='sum')
        # mse_loss =loss_fn2(tta_squashed_action[0], tta_squashed_action[1])
        # self.tta_actor_optim.zero_grad()
        # (mse_loss + 0.5 * p_q_kl_div + 0.5 * q_p_kl_div).backward()
        # self.tta_actor_optim.step()
        
        # tta_squashed_action = torch.mean(tta_squashed_action, dim=0)
        # tta_log_prob = tta_dist.log_prob(tta_squashed_action, tta_raw_action)
    
        # ###################################################################################################################
        # # 2. 增大均值的概率，通过让均值不变，方差变小
        # ####################################################################################################################
        # tta_dist = self.tta_actor(obs)
        # action_dist = self.actor(obs)
        # if deterministic:
        #     squashed_action, raw_action = action_dist.mode()
        # else:
        #     squashed_action, raw_action = action_dist.rsample()
        # tta_squashed_action, tta_raw_action = tta_dist.rsample()
        # tta_log_prob = tta_dist.log_prob(tta_squashed_action, tta_raw_action)

        # # top_tta_action_dists = torch.distributions.Normal((tta_dist.mean)[0], (tta_dist.stddev)[0])
        # # top_action_dists = torch.distributions.Normal((tta_dist.mean)[1], (tta_dist.stddev)[1])
        # # p_q_kl_div = torch.mean(
        # #     torch.distributions.kl.kl_divergence(top_tta_action_dists, top_action_dists))
        # # q_p_kl_div = torch.mean(
        # #     torch.distributions.kl.kl_divergence(top_action_dists, top_tta_action_dists))
        
        # # RDropout 更新
        # # variance = (tta_dist.stddev).mean()
        # tanh_stddev, raw_stddev = tta_dist.get_stddev()
        # variance = tanh_stddev.mean()
        # loss_fn2 = torch.nn.MSELoss(reduction='sum')
        # mse_loss =loss_fn2(tta_squashed_action, squashed_action)
        # self.tta_actor_optim.zero_grad()
        # # (mse_loss + 0.5 * p_q_kl_div + 0.5 * q_p_kl_div).backward()
        # (self._klcoef * mse_loss + variance).backward()
        # self.tta_actor_optim.step()

        # ###################################################################################################################
        
        # self.tta_buffer.append(obs)
        # if len(self.tta_buffer) == 256:
        #     torch_tta_buffer = np.array(self.tta_buffer)
        #     torch_tta_buffer = np.squeeze(torch_tta_buffer)
        #     torch_action_dist = self.actor(torch_tta_buffer)
        #     torch_stddev = torch.mean(torch_action_dist.stddev, dim=1)
        #     _, top_indices = torch.topk(-torch_stddev, k = 10)

        #     top_buffer_obs = torch_tta_buffer[(np.array(top_indices.cpu()))]
        #     top_tta_dist = self.tta_actor(top_buffer_obs)
        #     top_action_dist = self.actor(top_buffer_obs)

        #     # 用 tok 个样本来更新模型
        #     top_tta_action_dists = torch.distributions.Normal(top_tta_dist.mean, top_tta_dist.stddev)
        #     top_action_dists = torch.distributions.Normal(top_action_dist.mean, top_action_dist.stddev)
        #     kl_div = torch.mean(
        #         torch.distributions.kl.kl_divergence(top_tta_action_dists, top_action_dists))

        #     entropy = (top_tta_dist.entropy()).mean()
        #     # loss = (tta_dist.variance).mean() + self._klcoef * kl_div 
        #     # loss = entropy + self._klcoef * kl_div 
        #     # loss = variance_loss
        #     # loss = entropy + self._klcoef * kl_div 
        #     loss = entropy
        #     # loss = (tta_dist.variance).mean()
        #     # loss = (tta_dist.variance).mean()
        #     self.tta_actor_optim.zero_grad()
        #     loss.backward()
        #     self.tta_actor_optim.step()

        #     # 去掉更新过的样本
        #     top_indices_array = np.array(top_indices.cpu())
        #     mask = np.ones(torch_tta_buffer.shape[0], dtype=bool)
        #     mask[top_indices_array] = False
        #     new_array = torch_tta_buffer[mask, :]
        #     new_array = np.expand_dims(new_array, axis=1)
        #     self.tta_buffer = new_array.tolist()


        # # 0. replay_buffer
        # self.replay_buffer.append(obs)
        # if len(self.replay_buffer) == 32:
        #     replay_obs = np.squeeze(np.array(self.replay_buffer))
        #     replay_dist = self.tta_actor(replay_obs)
        #     replay_tta_squashed_action, replay_tta_raw_action = replay_dist.mode()

        #     # update behavior policy
        #     dataset_sampled_actions = self.behavior_policy.decode(torch.tensor(replay_obs, device='cuda:0', dtype=torch.float32))

        #     replay_obs = np.squeeze(np.array(self.replay_buffer))
        #     replay_dist = self.tta_actor(replay_obs)
        #     replay_tta_squashed_action, replay_tta_raw_action = replay_dist.mode()
        #     q1a, q2a = self.critic1(replay_obs, replay_tta_squashed_action), self.critic2(replay_obs, replay_tta_squashed_action)
        #     q1a_in, q2a_in = self.critic1(replay_obs, dataset_sampled_actions), self.critic2(replay_obs, dataset_sampled_actions)

        #     critic1_loss_for_ood_actions = ((q1a - q1a_in).pow(2)).mean()
        #     critic2_loss_for_ood_actions = ((q2a - q2a_in).pow(2)).mean()

        #     self.tta_actor_optim.zero_grad()
        #     (torch.min(critic1_loss_for_ood_actions, critic2_loss_for_ood_actions).mean() - torch.min(q1a, q2a).mean()).backward()
        #     self.tta_actor_optim.step()

        #     self.replay_buffer = []

        # ###################################################################################################################
        # # 1. Minimize Entropy
        # ####################################################################################################################
        # tta_dist = self.tta_actor(obs)
        # if deterministic:
        #     tta_squashed_action, tta_raw_action = tta_dist.mode()
        # else:
        #     tta_squashed_action, tta_raw_action = tta_dist.rsample()
        # tta_log_prob = tta_dist.log_prob(tta_squashed_action, tta_raw_action)
        # entropy = (tta_dist.entropy()).mean()
        # loss = entropy
        # self.tta_actor_optim.zero_grad()
        # loss.backward()
        # self.tta_actor_optim.step()

        # ########################################################
        # # 2023/10/2
        # ########################################################
        # # ##################################################################################################################
        # # # 2. Minimize Entropy, kl divergence
        # # ###################################################################################################################
        # tta_dist = self.tta_actor(obs)
        # dist = self.actor(obs)
        # if deterministic:
        #     tta_squashed_action, tta_raw_action = tta_dist.mode()
        # else:
        #     tta_squashed_action, tta_raw_action = tta_dist.rsample()
        # tta_log_prob = tta_dist.log_prob(tta_squashed_action, tta_raw_action)

        # # print("before tta entorpy: ", tta_dist.entropy())
        # entropy = (tta_dist.entropy()).mean()
        # tta_action_dists = torch.distributions.Normal(tta_dist.mean, tta_dist.stddev)
        # action_dists = torch.distributions.Normal(dist.mean, dist.stddev)
        # kl_div = torch.mean(
        #     torch.distributions.kl.kl_divergence(tta_action_dists, action_dists))
        
        # loss = entropy + self._klcoef * kl_div 
        # # loss = entropy
        # self.tta_actor_optim.zero_grad()
        # loss.backward()
        # self.tta_actor_optim.step()

        # after_tta_dist = self.tta_actor(obs)
        # print("afert tta entorpy: ", after_tta_dist.entropy())

        # ###################################################################################################################
        # # 3. Minimize Entropy, buffer, top 10%
        # ####################################################################################################################
        # tta_dist = self.tta_actor(obs)
        # if deterministic:
        #     tta_squashed_action, tta_raw_action = tta_dist.mode()
        # else:
        #     tta_squashed_action, tta_raw_action = tta_dist.rsample()
        # tta_log_prob = tta_dist.log_prob(tta_squashed_action, tta_raw_action)

        # self.tta_buffer.append(obs)
        # if len(self.tta_buffer) == 1000:
        #     torch_tta_buffer = np.array(self.tta_buffer)
        #     torch_tta_buffer = np.squeeze(torch_tta_buffer)
        #     torch_action_dist = self.tta_actor(torch_tta_buffer)
        #     torch_entropy = torch.mean(torch_action_dist.entropy(), dim=1)
        #     top_entropy, top_indices = torch.topk(torch_entropy, k = 10, largest=False)
        #     # torch_stddev = torch.mean(torch_action_dist.stddev, dim=1)
        #     # _, top_indices = torch.topk(-torch_stddev, k = 1)
        #     top_buffer_obs = torch_tta_buffer[(np.array(top_indices.cpu()))]
        #     top_tta_dist = self.tta_actor(top_buffer_obs)

        #     entropy = (top_tta_dist.entropy()).mean()
        #     loss = entropy

        #     self.tta_actor_optim.zero_grad()
        #     loss.backward()
        #     self.tta_actor_optim.step()

        #     # 去掉更新过的样本
        #     self.tta_buffer = []

        ########################################################
        # 2023/10/2
        ########################################################
        #  ###################################################################################################################
        # # 4. Minimize Entropy, buffer, top 10%, kl divergence
        # ####################################################################################################################
        # tta_dist = self.tta_actor(obs)
        # if deterministic:
        #     tta_squashed_action, tta_raw_action = tta_dist.mode()
        # else:
        #     tta_squashed_action, tta_raw_action = tta_dist.rsample()
        # tta_log_prob = tta_dist.log_prob(tta_squashed_action, tta_raw_action)

        # self.tta_buffer.append(obs)
        # if len(self.tta_buffer) == 1000:
        #     torch_tta_buffer = np.array(self.tta_buffer)
        #     torch_tta_buffer = np.squeeze(torch_tta_buffer)
        #     torch_action_dist = self.tta_actor(torch_tta_buffer)
        #     torch_entropy = torch.mean(torch_action_dist.entropy(), dim=1)
        #     top_entropy, top_indices = torch.topk(torch_entropy, k = 10, largest=False)
        #     # torch_stddev = torch.mean(torch_action_dist.stddev, dim=1)
        #     # _, top_indices = torch.topk(-torch_stddev, k = 1)
        #     top_buffer_obs = torch_tta_buffer[(np.array(top_indices.cpu()))]
        #     top_tta_dist = self.tta_actor(top_buffer_obs)
        #     top_action_dist = self.actor(top_buffer_obs)
        #     tta_action_dists = torch.distributions.Normal(top_tta_dist.mean, top_tta_dist.stddev)
        #     action_dists = torch.distributions.Normal(top_action_dist.mean, top_action_dist.stddev)
        #     kl_div = torch.mean(
        #         torch.distributions.kl.kl_divergence(tta_action_dists, action_dists))

        #     entropy = (top_tta_dist.entropy()).mean()
        #     loss = entropy + self._klcoef * kl_div

        #     self.tta_actor_optim.zero_grad()
        #     loss.backward()
        #     self.tta_actor_optim.step()

        #     # 去掉更新过的样本
        #     self.tta_buffer = []

        
        # ###################################################################################################################
        # # 3. Minimize Entropy, buffer, top 10%, weight entropy
        # ####################################################################################################################
        # tta_dist = self.tta_actor(obs)
        # if deterministic:
        #     tta_squashed_action, tta_raw_action = tta_dist.mode()
        # else:
        #     tta_squashed_action, tta_raw_action = tta_dist.rsample()
        # tta_log_prob = tta_dist.log_prob(tta_squashed_action, tta_raw_action)

        # self.tta_buffer.append(obs)
        # if len(self.tta_buffer) == 10:
        #     torch_tta_buffer = np.array(self.tta_buffer)
        #     torch_tta_buffer = np.squeeze(torch_tta_buffer)
        #     torch_action_dist = self.tta_actor(torch_tta_buffer)
        #     torch_entropy = torch.mean(torch_action_dist.entropy(), dim=1)
        #     top_entropy, top_indices = torch.topk(torch_entropy, k = 10, largest=False)

        #     if weight_sample:
        #         top_weight = torch.unsqueeze(torch.softmax((1 - top_entropy), dim=0), 1)
        #         top_buffer_obs = torch_tta_buffer[(np.array(top_indices.cpu()))]
        #         top_tta_dist = self.tta_actor(top_buffer_obs)
        #         entropy = (top_weight * top_tta_dist.entropy()).mean()
        #     else:
        #         top_buffer_obs = torch_tta_buffer[(np.array(top_indices.cpu()))]
        #         top_tta_dist = self.tta_actor(top_buffer_obs)
        #         entropy = (top_tta_dist.entropy()).mean()

        #     # if deterministic:
        #     #     top_tta_squashed_action, tta_raw_action = top_tta_dist.mode()
        #     # else:
        #     #     top_tta_squashed_action, tta_raw_action = top_tta_dist.rsample()
        #     # q1a, q2a = self.critic1(top_buffer_obs, top_tta_squashed_action), self.critic2(top_buffer_obs, top_tta_squashed_action)
        #     # q_value = (top_weight * torch.min(q1a, q2a)).mean()
        #     # loss = entropy - q_value
        #     loss = entropy 
        #     self.tta_actor_optim.zero_grad()
        #     loss.backward()
        #     self.tta_actor_optim.step()

        #     # 去掉更新过的样本
        #     self.tta_buffer = []

        ###################################################################################################################
        # # 3. Minimize Entropy, buffer, top 10%, kl divergence
        # ####################################################################################################################
        # tta_dist = self.tta_actor(obs)
        # if deterministic:
        #     tta_squashed_action, tta_raw_action = tta_dist.mode()
        # else:
        #     tta_squashed_action, tta_raw_action = tta_dist.rsample()
        # tta_log_prob = tta_dist.log_prob(tta_squashed_action, tta_raw_action)

        # self.tta_buffer.append(obs)
        # if len(self.tta_buffer) == 10:
        #     torch_tta_buffer = np.array(self.tta_buffer)
        #     torch_tta_buffer = np.squeeze(torch_tta_buffer)
        #     torch_action_dist = self.tta_actor(torch_tta_buffer)
        #     torch_entropy = torch.mean(torch_action_dist.entropy(), dim=1)
        #     print(torch_entropy)
        #     _, top_indices = torch.topk(torch_entropy, k = 1, largest=True)
        #     # top_indices = torch.randint(0, 99, (10, ), device=torch_entropy.device)
        #     print("topk: ", top_indices)
        #     top_buffer_obs = torch_tta_buffer[(np.array(top_indices.cpu()))]
        #     top_tta_dist = self.tta_actor(top_buffer_obs)
        #     top_action_dist = self.actor(top_buffer_obs)
        #     # print(top_action_dist.loc)
        #     # 用 tok 个样本来更新模型
        #     top_tta_action_dists = torch.distributions.Normal(top_tta_dist.mean, top_tta_dist.stddev)
        #     top_action_dists = torch.distributions.Normal(top_action_dist.mean, top_action_dist.stddev)
        #     kl_div = torch.mean(
        #         torch.distributions.kl.kl_divergence(top_tta_action_dists, top_action_dists))

        #     # top_tta_dist = self.tta_actor(self.tta_buffer)
        #     entropy = (top_tta_dist.entropy()).mean()
        #     loss = entropy
        #     # loss = entropy + self._klcoef * kl_div

        #     self.tta_actor_optim.zero_grad()
        #     loss.backward()
        #     self.tta_actor_optim.step()

        #     # 去掉更新过的样本
        #     self.tta_buffer = []

        # ###################################################################################################################
        # # 3. Minimize Entropy, buffer, top 10%
        # ####################################################################################################################
        # tta_dist = self.tta_actor(obs)
        # # dist = self.actor(obs)
        # if deterministic:
        #     tta_squashed_action, tta_raw_action = tta_dist.mode()
        # else:
        #     tta_squashed_action, tta_raw_action = tta_dist.rsample()
        # tta_log_prob = tta_dist.log_prob(tta_squashed_action, tta_raw_action)

        # ##############################################################################
        # # batch update 10/04:16:15
        # ##############################################################################
        # # 1. online learning
        # dist = self.tta_actor(obs)
        # if deterministic:
        #     tta_squashed_action, tta_raw_action = dist.mode()
        # else:
        #     tta_squashed_action, tta_raw_action = dist.rsample()
        # tta_log_prob = dist.log_prob(tta_squashed_action, tta_raw_action)
        # # 2. online update
        # self.tta_buffer.append(obs)
        # if len(self.tta_buffer) == 8:
        #     torch_tta_buffer = np.array(self.tta_buffer)
        #     torch_tta_buffer = np.squeeze(torch_tta_buffer)
        #     normal_action_dist = self.actor(torch_tta_buffer)
        #     torch_action_dist = self.tta_actor(torch_tta_buffer)
        #     tta_action_dists = torch.distributions.Normal(torch_action_dist.mean, torch_action_dist.stddev)
        #     normal_action_dists = torch.distributions.Normal(normal_action_dist.mean, normal_action_dist.stddev)
        #     kl_div = torch.mean(
        #         torch.distributions.kl.kl_divergence(tta_action_dists, normal_action_dists))

        #     entropy = (torch_action_dist.entropy()).mean()
        #     loss = entropy + self._klcoef * kl_div 
        #     # loss = entropy
        #     self.tta_actor_optim.zero_grad()
        #     loss.backward()
        #     self.tta_actor_optim.step()

        #     # 去掉更新过的样本
        #     self.tta_buffer = []


        # self.tta_buffer.append(obs)
        # if len(self.tta_buffer) == 9000:
        #     torch_tta_buffer = np.array(self.tta_buffer)
        #     torch_tta_buffer = np.squeeze(torch_tta_buffer)
        #     # torch_action_dist = self.actor(torch_tta_buffer)
        #     torch_action_dist = self.tta_actor(torch_tta_buffer)
        #     torch_stddev = torch.mean(torch_action_dist.stddev, dim=1)
        #     _, top_indices = torch.topk(-torch_stddev, k = 32)

        #     top_buffer_obs = torch_tta_buffer[(np.array(top_indices.cpu()))]
        #     top_tta_dist = self.tta_actor(top_buffer_obs)
        #     # top_action_dist = self.actor(top_buffer_obs)

        #     # 用 tok 个样本来更新模型
        #     # top_tta_action_dists = torch.distributions.Normal(top_tta_dist.mean, top_tta_dist.stddev)
        #     # top_action_dists = torch.distributions.Normal(top_action_dist.mean, top_action_dist.stddev)
        #     # kl_div = torch.mean(
        #     #     torch.distributions.kl.kl_divergence(top_tta_action_dists, top_action_dists))

        #     entropy = (top_tta_dist.entropy()).mean()
        #     # loss = (tta_dist.variance).mean() + self._klcoef * kl_div 
        #     # loss = entropy + self._klcoef * kl_div 
        #     # loss = variance_loss
        #     # loss = entropy + self._klcoef * kl_div 
        #     loss = entropy
        #     # loss = (tta_dist.variance).mean()
        #     # loss = (tta_dist.variance).mean()
        #     self.tta_actor_optim.zero_grad()
        #     loss.backward()
        #     self.tta_actor_optim.step()

        #     # 去掉更新过的样本
        #     top_indices_array = np.array(top_indices.cpu())
        #     mask = np.ones(torch_tta_buffer.shape[0], dtype=bool)
        #     mask[top_indices_array] = False
        #     new_array = torch_tta_buffer[mask, :]
        #     new_array = np.expand_dims(new_array, axis=1)
        #     self.tta_buffer = new_array.tolist()

        ###################################################################################################################


        # # 0. replay_buffer
        # self.replay_buffer.append(obs)
        # if len(self.replay_buffer) == 32:
        #     replay_obs = np.squeeze(np.array(self.replay_buffer))
        #     replay_dist = self.tta_actor(replay_obs)
        #     replay_tta_squashed_action, replay_tta_raw_action = replay_dist.mode()

        #     # update behavior policy
        #     dataset_sampled_actions = self.behavior_policy.decode(torch.tensor(replay_obs, device='cuda:0', dtype=torch.float32))

        #     replay_obs = np.squeeze(np.array(self.replay_buffer))
        #     replay_dist = self.tta_actor(replay_obs)
        #     replay_tta_squashed_action, replay_tta_raw_action = replay_dist.mode()
        #     q1a, q2a = self.critic1(replay_obs, replay_tta_squashed_action), self.critic2(replay_obs, replay_tta_squashed_action)
        #     q1a_in, q2a_in = self.critic1(replay_obs, dataset_sampled_actions), self.critic2(replay_obs, dataset_sampled_actions)

        #     critic1_loss_for_ood_actions = ((q1a - q1a_in).pow(2)).mean()
        #     critic2_loss_for_ood_actions = ((q2a - q2a_in).pow(2)).mean()

        #     self.tta_actor_optim.zero_grad()
        #     (torch.min(critic1_loss_for_ood_actions, critic2_loss_for_ood_actions).mean() - torch.min(q1a, q2a).mean()).backward()
        #     self.tta_actor_optim.step()

        #     self.replay_buffer = []


        # # 2. augmentation action
        # new_obs = np.add(obs, 1 * np.random.normal(0, 1, obs.shape))
        # tta_dist = self.tta_actor(obs)
        # augmentatoin_tta_dist = self.tta_actor(new_obs)
        # if deterministic:
        #     tta_squashed_action, tta_raw_action = tta_dist.mode()
        #     augmentatoin_tta_squashed_action,  augmentatoin_tta_raw_action = augmentatoin_tta_dist.mode()
        # else:
        #     tta_squashed_action, tta_raw_action = tta_dist.rsample()
        #     augmentatoin_tta_squashed_action,  augmentatoin_tta_raw_action = augmentatoin_tta_dist.rsample()
        # tta_log_prob = tta_dist.log_prob(tta_squashed_action, tta_raw_action)

        # action_var = torch.mean((tta_squashed_action - augmentatoin_tta_squashed_action)**2)
        
        # self.tta_actor_optim.zero_grad()
        # # (action_var.mean() + self._klcoef * kl_loss.mean()).backward()
        # (action_var.mean()).backward()
        # self.tta_actor_optim.step()

        # # menent update tta_actor
        # with torch.no_grad():
        #     for ema_param, param in zip(self.ema_params, self.tta_actor.parameters()):
        #         ema_param.mul_(self._moment_tau).add_((1 - self._moment_tau) * param)

        #     for ema_param, param in zip(self.ema_params, self.tta_actor.parameters()):
        #         param.copy_(ema_param)

        # return tta_squashed_action, tta_log_prob, tta_entropy_loss.sum()
        return tta_squashed_action, tta_log_prob, tta_log_prob
        # return tta_squashed_action, tta_log_prob, tta_log_prob
    
    def update_with_aug_data(self, deterministic: bool = False):
        index = torch.randperm(32)
        states = np.squeeze((np.array(self.states_list)))
        actions = torch.stack(self.actions_list, 0)
        # actions = torch.Tensor(np.squeeze((np.array(self.actions_list, dtype=np.float64))))
        mix_states = 0.6 * states + 0.4 * states[index, :]
        mix_actions = 0.6 * actions + 0.4 * actions[index, :]
        tta_dist = self.tta_actor(mix_states)
        if deterministic:
            tta_squashed_action, tta_raw_action = tta_dist.mode()
        else:
            tta_squashed_action, tta_raw_action = tta_dist.rsample()

        action_loss = (mix_actions - tta_squashed_action) ** 2
        self.tta_actor_optim.zero_grad()
        action_loss.mean().backward()
        self.tta_actor_optim.step()
    
    def adapted_multi_episode_actforward(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        ####################################################################################################################
        # 0. 从多条 episode 中挑选出 Q 值最高的 episode 来更新模型
        ####################################################################################################################
        tta_dist = self.tta_actor(obs)
        if deterministic:
            tta_squashed_action, tta_raw_action = tta_dist.mode()
        else:
            tta_squashed_action, tta_raw_action = tta_dist.rsample()
        tta_log_prob = tta_dist.log_prob(tta_squashed_action, tta_raw_action)
        
        # compute the q value of the current (s, a)
        q1a, q2a = self.critic1(obs, tta_squashed_action), self.critic2(obs, tta_squashed_action)
        q = torch.min(q1a, q2a)
        
        # compute the entropy of the current (s, a)
        entropy = (tta_dist.entropy()).mean()

        return tta_squashed_action, entropy, obs, q
    
    def select_adapted_multi_episode_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        ####################################################################################################################
        # 2023/11/29/15:51 从多条 episode 中挑选出 Q 值最高的 episode 来更新模型
        ####################################################################################################################
        action, entropy, obs, q = self.adapted_multi_episode_actforward(obs, deterministic)
        return action, entropy, obs, q
    
    def select_adapted_consistent_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        ####################################################################################################################
        # 2023/12/07/14:29 增强输入，让增强后的输入和没增强的输入保持一致性
        ####################################################################################################################
        action, entropy, obs, q = self.adapted_consistent_action_actforward(obs, deterministic)
        return action, entropy, obs, q
    
    def adapted_consistent_action_actforward(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        ####################################################################################################################
        # 0. 对 obs 进行数据增强，然后约束增强后的数据和原始数据的动作输出一致
        ####################################################################################################################
        obs_copy = obs.copy()
        rand_idxs = np.random.randint(0, obs.shape[1], size=obs.shape[0])
        # 将每行随机索引对应的元素 mask 掉
        for i, idx in enumerate(rand_idxs):
            obs_copy[i, idx] = 0

        tta_dist = self.tta_actor(obs)
        tta_copy_dist = self.tta_actor(obs_copy)
        if deterministic:
            tta_squashed_action, tta_raw_action = tta_dist.mode()
        else:
            tta_squashed_action, tta_raw_action = tta_dist.rsample()

        kl_div = torch.mean(
            torch.distributions.kl.kl_divergence(tta_dist, tta_copy_dist))
        
        # compute the q value of the current (s, a)
        q1a, q2a = self.critic1(obs, tta_squashed_action), self.critic2(obs, tta_squashed_action)
        q = torch.min(q1a, q2a)
        
        # compute the entropy of the current (s, a)
        entropy = (tta_dist.entropy()).mean()

        # if entropy < 0.1:
        self.tta_actor_optim.zero_grad()
        (kl_div.mean()).backward()
        self.tta_actor_optim.step()

        return tta_squashed_action, entropy, obs, q
    
    def update_with_batch_obs(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        tta_dist = self.tta_actor(obs)
        old_action_dists = self.actor(obs)
        kl_div = torch.mean(
            torch.distributions.kl.kl_divergence(old_action_dists, tta_dist))
        
        entropy = (tta_dist.entropy()).mean()

        # if entropy < 0.1:
        self.tta_actor_optim.zero_grad()
        (self._klcoef * kl_div.mean() + entropy).backward()
        self.tta_actor_optim.step()

    # select adapted mcd action
    def select_adapted_mcd_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        action, _, entropy = self.adapetd_mcd_actforward(obs, deterministic)
        self.actions_list.append(action.detach())
        self.states_list.append(obs)

        if len(self.actions_list) == 32:
            self.update_with_aug_data(deterministic)
            self.actions_list = []
            self.states_list = []

        return action, entropy

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        with torch.no_grad():
            action, _, entropy = self.actforward(obs, deterministic)
        return action.cpu().numpy(), entropy

    # select adapted action
    def select_adapted_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        action, _, entropy = self.adapetd_actforward(obs, deterministic)
        # action, _, entropy = self.adapetd_actforward(obs)
        # return action.cpu().numpy()
        return action, entropy

    def learn(self, batch: Dict) -> Dict[str, float]:
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]

        # update critic
        q1, q2 = self.critic1(obss, actions), self.critic2(obss, actions)
        with torch.no_grad():
            next_actions, next_log_probs = self.actforward(next_obss)
            next_q = torch.min(
                self.critic1_old(next_obss, next_actions), self.critic2_old(next_obss, next_actions)
            ) - self._alpha * next_log_probs
            target_q = rewards + self._gamma * (1 - terminals) * next_q

        critic1_loss = ((q1 - target_q).pow(2)).mean()
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        critic2_loss = ((q2 - target_q).pow(2)).mean()
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
        }

        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()

        return result

