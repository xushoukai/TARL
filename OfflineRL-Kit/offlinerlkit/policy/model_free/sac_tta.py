import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from loguru import logger
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
        deterministic: bool = False #! 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logger.debug("Using adapetd_actforward")
        # 0. importance sampling
        tta_dist = self.tta_actor(obs)
        if deterministic:
            tta_squashed_action, tta_raw_action = tta_dist.mode()
        else:
            tta_squashed_action, tta_raw_action = tta_dist.rsample()
        tta_log_prob = tta_dist.log_prob(tta_squashed_action, tta_raw_action)

        tta_entropy_loss = (tta_dist.entropy()).abs() # ! ABS difference
        self.tta_actor_optim.zero_grad()
        (tta_entropy_loss.mean()).backward()
        self.tta_actor_optim.step()
        logger.debug(f"Update tta-actor via tta_entropy_loss:{(tta_dist.entropy()).abs()}")
        
        return tta_squashed_action, tta_log_prob, tta_entropy_loss.sum()

    def adapetd_mcd_actforward(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
        entropy_threshold: float = 0.1,
        is_entropy_filter: bool = False,
        loss_agg_type: str = "abs", 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logger.debug("Using adapetd_mcd_actforward")
        # 0. importance sampling
        tta_dist = self.tta_actor(obs)
        if deterministic:
            tta_squashed_action, tta_raw_action = tta_dist.mode()
        else:
            tta_squashed_action, tta_raw_action = tta_dist.rsample()
        tta_log_prob = tta_dist.log_prob(tta_squashed_action, tta_raw_action)
        
        old_action_dists = self.actor(obs)

        kl_div = torch.mean(
            torch.distributions.kl.kl_divergence(old_action_dists, tta_dist))
        
        # if loss_agg_type == "mean":
            # logger.warning("Using mean")
            # entropy = (tta_dist.entropy()).mean()
        # elif loss_agg_type == "abs":
            # logger.warning("Using abs")
        entropy = (tta_dist.entropy()).abs().mean()
        logger.info(f"tta_dist's entropy: {entropy}")
        logger.debug(f"Entropy in adapted mcd actforward is {entropy}")
        logger.debug(f"KL divergence is {kl_div}")
        logger.debug(f"KL coef is {self._klcoef}")

        if is_entropy_filter and entropy > entropy_threshold:
            logger.debug(f"Update tta-actor with entropy filtering {entropy} > {entropy_threshold}")
            self.tta_actor_optim.zero_grad()
            (self._klcoef * kl_div.mean() + entropy).backward()
            self.tta_actor_optim.step()
        elif is_entropy_filter is False:
            logger.debug(f"Update tta-actor with no entropy filtering")
            self.tta_actor_optim.zero_grad()
            (self._klcoef * kl_div.mean() + entropy).backward()
            self.tta_actor_optim.step()
        else:
            logger.warning(f"Not update to tta-actor due to high entropy {entropy} < {entropy_threshold}")
            
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
        logger.debug(f"Using update_with_aug_data to update with aug data: {action_loss}")
        self.tta_actor_optim.zero_grad()
        action_loss.mean().backward()
        self.tta_actor_optim.step()
        logger.debug("Update tta actor via action_loss")
    
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

    #! 0331 Above Code Come From LCH
    # max_attempts = 10  # 最大重采样次数，避免死循环
    # attempts = 0

    # while True:
    #     # 调用原始逻辑计算动作和熵
    #     action, _, entropy = self.adapetd_mcd_actforward(obs, deterministic)
        
    #     # 如果熵值小于等于 0.1，则接受该动作
    #     if entropy <= 0.1:
    #         break
        
    #     # 如果熵值超过 0.1 且未达到最大尝试次数，则重新采样
    #     if attempts < max_attempts:
    #         attempts += 1
    #         continue
    #     else:
    #         # 如果达到最大尝试次数仍未找到符合条件的动作，则强制返回当前动作
    #         print(f"Warning: Max attempts reached for entropy threshold 0.1")
    #         break
        
    # select adapted mcd action
    def select_adapted_mcd_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
        loss_agg_type: str = "abs",
        entropy_threshold:float = 0.1,
        is_entropy_filter: bool = False,
    ) -> np.ndarray:
        logger.debug("Using select_adapted_mcd_action")
        action, _, entropy = self.adapetd_mcd_actforward(obs, deterministic, loss_agg_type=loss_agg_type,entropy_threshold=entropy_threshold,is_entropy_filter=is_entropy_filter)

        # action-loss 
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
        logger.debug("Using select_adapted_action")
        action, _, entropy = self.adapetd_actforward(obs, deterministic)
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

