import sys
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from copy import deepcopy
from collections import deque
from typing import Dict, Union, Tuple
from offlinerlkit.policy import BasePolicy


class SACTTAMULTULPolicy(BasePolicy):
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

    def actforward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist, logits = self.actor(obs)
        if deterministic:
            squashed_action, raw_action = dist.mode()
        else:
            squashed_action, raw_action = dist.rsample()
        log_prob = dist.log_prob(squashed_action, raw_action)
        # actor_loss = ((dist.entropy()).abs()).sum()
        actor_loss = (dist.entropy()).sum()
        return squashed_action, log_prob, actor_loss

    def kl_divergence(self, mu1, sigma1, mu2, sigma2):
        # compute KL divergence
        var1 = sigma1**2
        var2 = sigma2**2
        kl = torch.log(var2 / var1) + (var1 + (mu1 - mu2)**2) / (2 * var2) - 0.5
        return kl.mean()
    
    def compute_joint(self, x_out, x_tf_out):
        # produces variable that requires grad (since args require grad)
        bn, k = x_out.size()
        assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)

        p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
        p_i_j = p_i_j.sum(dim=0)  # k, k
        p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
        p_i_j = p_i_j / p_i_j.sum()  # normalise

        return p_i_j
    
    def instance_contrastive_Loss(self, x_out, x_tf_out, lamb=1.0, EPS=sys.float_info.epsilon):
        """compute the multul information of p and q"""
        _, k = x_out.size()
        x_out = F.softmax(x_out,dim =1)
        x_tf_out = F.softmax(x_tf_out,dim =1)
        # print("x_out: ", x_out)
        # print("x_tf_out: ", x_tf_out)
        p_i_j = self.compute_joint(x_out, x_tf_out)
        assert (p_i_j.size() == (k, k))

        # p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
        # p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)

        p_i_j = torch.where(p_i_j < EPS, torch.tensor([EPS], device=p_i_j.device), p_i_j)
        # p_j = torch.where(p_j < EPS, torch.tensor([EPS], device=p_j.device), p_j)
        # p_i = torch.where(p_i < EPS, torch.tensor([EPS], device=p_i.device), p_i)

        # loss = - p_i_j * (torch.log(p_i_j) \
        #                 - lamb * torch.log(p_j) \
        #                 - lamb * torch.log(p_i))
        loss = p_i_j * (torch.log(p_i_j))

        loss = loss.sum()
        return loss

    def adapetd_actforward(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tta_dist, tta_logits = self.tta_actor(obs)
        # dist, logits = self.actor(obs)
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
        (- tta_entropy_loss.mean()).backward()
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

    def adapted_multul_actforward(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    )-> Tuple[torch.Tensor, torch.Tensor]:
        tta_dist, tta_logits = self.tta_actor(obs)
        dist, logits = self.actor(obs)
        logits = logits.detach()
        if deterministic:
            tta_squashed_action, tta_raw_action = tta_dist.mode()
        else:
            tta_squashed_action, tta_raw_action = tta_dist.rsample()
        tta_log_prob = tta_dist.log_prob(tta_squashed_action, tta_raw_action)
        loss = self.instance_contrastive_Loss(logits, tta_logits)
        self.tta_actor_optim.zero_grad()
        loss.backward()
        self.tta_actor_optim.step()

        return tta_squashed_action, tta_squashed_action, self.learnable_var


    def adapetd_mcd_actforward(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        repeat_obs = np.repeat(obs, 10, axis=0)
        repeat_tta_dist, logits = self.tta_actor(repeat_obs)
        if deterministic:
            repeat_tta_squashed_action, repeat_tta_raw_action = repeat_tta_dist.mode()
        else:
            repeat_tta_squashed_action, repeat_tta_raw_action = repeat_tta_dist.rsample()
        repeat_tta_log_prob = repeat_tta_dist.log_prob(repeat_tta_squashed_action, repeat_tta_raw_action)
        action_var = torch.var(repeat_tta_squashed_action, dim=0)
        tta_squashed_action = torch.mean(repeat_tta_squashed_action, dim=0)
        q1a, q2a = self.critic1(repeat_obs, repeat_tta_squashed_action), self.critic2(repeat_obs, repeat_tta_squashed_action)

        self.tta_actor_optim.zero_grad()
        # (action_var.mean() - self._klcoef * (torch.min(q1a, q2a) / action_var).mean()).backward()
        (action_var.mean() - self._klcoef * (torch.min(q1a, q2a)).mean()).backward()
        self.tta_actor_optim.step()

        return tta_squashed_action, tta_squashed_action, self.learnable_var

    # select adapted mcd action
    def select_adapted_mcd_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        # action, _, entropy = self.adapetd_mcd_actforward(obs, deterministic)
        action, _, entropy = self.adapted_multul_actforward(obs, deterministic)
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

