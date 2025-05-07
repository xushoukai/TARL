import numpy as np
import torch
import torch.nn as nn
import gym

from copy import deepcopy
from typing import Dict, Union, Tuple
from offlinerlkit.policy import BasePolicy
from offlinerlkit.policy import SACTTAPolicy

class IQLTTAPolicy(BasePolicy):
    """
    Implicit Q-Learning <Ref: https://arxiv.org/abs/2110.06169>
    """

    def __init__(
        self,
        actor: nn.Module,
        tta_actor: nn.Module,
        critic_q1: nn.Module,
        critic_q2: nn.Module,
        critic_v: nn.Module,
        actor_optim: torch.optim.Optimizer,
        tta_actor_optim: torch.optim.Optimizer,
        critic_q1_optim: torch.optim.Optimizer,
        critic_q2_optim: torch.optim.Optimizer,
        critic_v_optim: torch.optim.Optimizer,
        action_space: gym.spaces.Space,
        tau: float = 0.005,
        gamma: float  = 0.99,
        expectile: float = 0.8,
        klcoef: float = 1.5, 
        temperature: float = 0.1,
        moment_tau: float = 0.005
    ) -> None:
        super().__init__()

        self.actor = actor
        self.tta_actor = tta_actor
        self.critic_q1, self.critic_q1_old = critic_q1, deepcopy(critic_q1)
        self.critic_q1_old.eval()
        self.critic_q2, self.critic_q2_old = critic_q2, deepcopy(critic_q2)
        self.critic_q2_old.eval()
        self.critic_v = critic_v

        self.actor_optim = actor_optim
        self.tta_actor_optim = tta_actor_optim
        self.critic_q1_optim = critic_q1_optim
        self.critic_q2_optim = critic_q2_optim
        self.critic_v_optim = critic_v_optim

        self.action_space = action_space
        self._tau = tau
        self._gamma = gamma
        self._klcoef = klcoef
        self._expectile = expectile
        self._temperature = temperature
        self.tta_buffer = []

        self.actions_list = []
        self.states_list = []

    def train(self) -> None:
        self.actor.train()
        self.tta_actor.train()
        self.critic_q1.train()
        self.critic_q2.train()
        self.critic_v.train()

    def eval(self) -> None:
        self.actor.eval()
        self.tta_actor.eval()
        self.critic_q1.eval()
        self.critic_q2.eval()
        self.critic_v.eval()

    def _sync_weight(self) -> None:
        for o, n in zip(self.critic_q1_old.parameters(), self.critic_q1.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic_q2_old.parameters(), self.critic_q2.parameters()):
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
    
    
    
    
    
    
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if len(obs.shape) == 1:
            obs = obs.reshape(1, -1)
        with torch.no_grad():
            dist = self.actor(obs)
            if deterministic:
                action = dist.mode().cpu().numpy()
            else:
                action = dist.sample().cpu().numpy()
        action = np.clip(action, self.action_space.low[0], self.action_space.high[0])
        return action
    
    def _expectile_regression(self, diff: torch.Tensor) -> torch.Tensor:
        weight = torch.where(diff > 0, self._expectile, (1 - self._expectile))
        return weight * (diff**2)
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
        # print(f"这是打印的tta-dist{tta_dist}")
        if deterministic:
            tta_squashed_action, tta_raw_action = tta_dist.mode()
        else:
            # print(f"这是tta rsample内容{tta_dist.rsample()}")
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
        return tta_squashed_action, tta_log_prob, tta_log_prob

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

    # # select adapted mcd action
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
        
        # update value net
        with torch.no_grad():
            q1, q2 = self.critic_q1_old(obss, actions), self.critic_q2_old(obss, actions)
            q = torch.min(q1, q2)
        v = self.critic_v(obss)
        critic_v_loss = self._expectile_regression(q-v).mean()
        self.critic_v_optim.zero_grad()
        critic_v_loss.backward()
        self.critic_v_optim.step()

        # update critic
        q1, q2 = self.critic_q1(obss, actions), self.critic_q2(obss, actions)
        with torch.no_grad():
            next_v = self.critic_v(next_obss)
            target_q = rewards + self._gamma * (1 - terminals) * next_v
        
        critic_q1_loss = ((q1 - target_q).pow(2)).mean()
        critic_q2_loss = ((q2 - target_q).pow(2)).mean()

        self.critic_q1_optim.zero_grad()
        critic_q1_loss.backward()
        self.critic_q1_optim.step()

        self.critic_q2_optim.zero_grad()
        critic_q2_loss.backward()
        self.critic_q2_optim.step()

        # update actor
        with torch.no_grad():
            q1, q2 = self.critic_q1_old(obss, actions), self.critic_q2_old(obss, actions)
            q = torch.min(q1, q2)
            v = self.critic_v(obss)
            exp_a = torch.exp((q - v) * self._temperature)
            exp_a = torch.clip(exp_a, None, 100.0)
        dist = self.actor(obss)
        log_probs = dist.log_prob(actions)
        actor_loss = -(exp_a * log_probs).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self._sync_weight()

        return {
            "loss/actor": actor_loss.item(),
            "loss/q1": critic_q1_loss.item(),
            "loss/q2": critic_q2_loss.item(),
            "loss/v": critic_v_loss.item()
        }