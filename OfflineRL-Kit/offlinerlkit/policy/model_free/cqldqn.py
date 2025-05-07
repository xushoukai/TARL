import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

import random
from copy import deepcopy
from typing import Callable, Dict, Union, Tuple
from offlinerlkit.policy import BasePolicy


class CQLDQNPolicy(BasePolicy):
    """
    <Ref: https://github.com/BY571/CQL/tree/main>
    """

    def __init__(
        self, 
        network: nn.Module,
        target_net: nn.Module,
        optimizer: torch.optim.Optimizer,
        action_size: float, 
        tau: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1e-4, 
        device: str = "cpu")-> None:
        super().__init__()

        self.network = network
        self.target_net = target_net
        self.optimizer = optimizer
        self.action_size = action_size
        self.tau = tau
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = device
    
    def train(self) -> None:
        self.network.train()

    def eval(self) -> None:
        self.network.eval()
    
    # def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
    #     with torch.no_grad():
    #         action = self.actor(obs).cpu().numpy()
    #     if not deterministic:
    #         action = action + self.exploration_noise(action.shape)
    #         action = np.clip(action, -self._max_action, self._max_action)
    #     return action
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if deterministic:
        # if random.random() > self.epsilon:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.network.eval()
            with torch.no_grad():
                action_values = self.network(state)
            self.network.train()
            action = np.argmax(action_values.cpu().data.numpy(), axis=1)
        else:
            action = random.choices(np.arange(self.action_size), k=1)
        return action
    
    def learn(self, batch):
        self.optimizer.zero_grad()
        states, actions, next_states, rewards, dones = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]

        with torch.no_grad():
            Q_targets_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        Q_a_s = self.network(states)
        print("states: ", states.shape)
        print("actions: ", actions.shape)
        print("next_states: ", next_states.shape)
        print("rewards: ", rewards.shape)
        print("dones: ", dones.shape)
        print("Q_a_s: ", Q_a_s.shape)
        # print("states: ", states)
        print(actions)
        Q_expected = Q_a_s.gather(1, actions)
        
        cql1_loss = torch.logsumexp(Q_a_s, dim=1).mean() - Q_a_s.mean()

        bellmann_error = F.mse_loss(Q_expected, Q_targets)
        
        q1_loss = cql1_loss + 0.5 * bellmann_error
        
        q1_loss.backward()
        clip_grad_norm_(self.network.parameters(), 1)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.network, self.target_net)

        result = {
            "loss/q1_loss": q1_loss.detach().item(),
            "loss/cql1_loss": cql1_loss.detach().item(),
            "loss/bellmann_error": bellmann_error.detach().item()
        }

        return result
        
        
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)