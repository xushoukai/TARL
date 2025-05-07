import numpy as np
import torch
import torch.nn as nn

from copy import deepcopy
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
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2
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

        self._is_auto_alpha = False
        if isinstance(alpha, tuple):
            self._is_auto_alpha = True
            self._target_entropy, self._log_alpha, self.alpha_optim = alpha
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = alpha

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
        dist = self.actor(obs)
        if deterministic:
            squashed_action, raw_action = dist.mode()
        else:
            squashed_action, raw_action = dist.rsample()
        log_prob = dist.log_prob(squashed_action, raw_action)
        actor_loss = ((dist.entropy()).abs()).sum()
        return squashed_action, log_prob, actor_loss

    def kl_divergence(self, mu1, sigma1, mu2, sigma2):
        # compute KL divergence
        var1 = sigma1**2
        var2 = sigma2**2
        kl = torch.log(var2 / var1) + (var1 + (mu1 - mu2)**2) / (2 * var2) - 0.5
        return kl.mean()

    def adapetd_actforward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tta_dist = self.tta_actor(obs)
        dist = self.actor(obs)
        if deterministic:
            tta_squashed_action, tta_raw_action = tta_dist.mode()
            squashed_action, raw_action = dist.mode()
            print("*" * 200)
            print("action: ", tta_squashed_action)
            print("*" * 200)
        else:
            tta_squashed_action, tta_raw_action = tta_dist.rsample()
            squashed_action, raw_action = dist.rsample()
        tta_log_prob = dist.log_prob(tta_squashed_action, tta_raw_action)

        # update tta actor
        print("-" * 200)
        print("entropy: ", tta_dist.entropy())
        tta_actor_loss = ((tta_dist.entropy()).abs()).sum()
        # mse_loss = torch.nn.functional.mse_loss(tta_squashed_action, squashed_action)
        # kl_loss = self.kl_divergence(dist.mean, dist.stddev, tta_dist.mean, tta_dist.stddev)
        print("entropy loss: ", tta_actor_loss)
        print("-" * 200)
        # self.tta_actor_optim.zero_grad()
        # self.tta_actor_optim.first_step(zero_grad=True)
        # (tta_actor_loss + 2 * mse_loss).backward()
        # (tta_actor_loss + kl_loss).backward()
        tta_actor_loss.backward()
        # self.tta_actor_optim.second_step(zero_grad=True)
        # self.tta_actor_optim.step()

        return tta_squashed_action, tta_log_prob, tta_actor_loss

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
        self.tta_actor_optim.zero_grad()
        self.tta_actor_optim.first_step(zero_grad=True)
        action, _, entropy = self.adapetd_actforward(obs, deterministic)
        self.tta_actor_optim.second_step(zero_grad=True)
        
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

