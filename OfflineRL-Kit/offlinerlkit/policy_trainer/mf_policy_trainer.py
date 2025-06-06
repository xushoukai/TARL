import time
import os

import numpy as np
import torch
from torch.nn import functional as F
import gym

from typing import Optional, Dict, List
from tqdm import tqdm
from collections import deque
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger
from offlinerlkit.policy import BasePolicy
from offlinerlkit.policy_trainer.tent import configure_model, configure_final_layer


# model-free policy trainer
class MFPolicyTrainer:
    def __init__(
        self,
        policy: BasePolicy,
        eval_env: gym.Env,
        buffer: ReplayBuffer,
        logger: Logger,
        epoch: int = 1000,
        step_per_epoch: int = 1000,
        batch_size: int = 256,
        eval_episodes: int = 10,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> None:
        self.policy = policy
        self.eval_env = eval_env
        self.buffer = buffer
        self.logger = logger

        self._epoch = epoch
        self._step_per_epoch = step_per_epoch
        self._batch_size = batch_size
        self._eval_episodes = eval_episodes
        self.lr_scheduler = lr_scheduler

    def train(self) -> Dict[str, float]:
        start_time = time.time()

        num_timesteps = 0
        last_10_performance = deque(maxlen=10)
        # record reward
        best_reward_mean = -10000
        best_reward_std = 0
        # train loop
        for e in tqdm(range(1, self._epoch + 1)):

            self.policy.train()

            pbar = tqdm(range(self._step_per_epoch), desc=f"Epoch #{e}/{self._epoch}")
            for it in pbar:
                batch = self.buffer.sample(self._batch_size)
                loss = self.policy.learn(batch)
                pbar.set_postfix(**loss)

                for k, v in loss.items():
                    self.logger.logkv_mean(k, v)
                
                num_timesteps += 1

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # evaluate current policy
            eval_info = self._evaluate()
            ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
            ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
            norm_ep_rew_mean = self.eval_env.get_normalized_score(ep_reward_mean) * 100
            norm_ep_rew_std = self.eval_env.get_normalized_score(ep_reward_std) * 100
            last_10_performance.append(norm_ep_rew_mean)
            self.logger.logkv("eval/normalized_episode_reward", norm_ep_rew_mean)
            self.logger.logkv("eval/normalized_episode_reward_std", norm_ep_rew_std)
            self.logger.logkv("eval/episode_length", ep_length_mean)
            self.logger.logkv("eval/episode_length_std", ep_length_std)
            self.logger.set_timestep(num_timesteps)
            self.logger.dumpkvs()
            # save checkpoint
            torch.save(self.policy.state_dict(), os.path.join(self.logger.checkpoint_dir, "policy.pth"))
            if e >= 990:
                torch.save(self.policy.state_dict(), os.path.join(self.logger.checkpoint_dir, "policy" + "_" + str(e) + ".pth"))
            if best_reward_mean < norm_ep_rew_mean:
                best_reward_mean = norm_ep_rew_mean
                best_reward_std = norm_ep_rew_std
                torch.save(self.policy.state_dict(), os.path.join(self.logger.checkpoint_dir, "best_policy.pth"))
                self.logger.log("best_reward_mean: {:.2f}, best_reward_std: {:.2f}".format(best_reward_mean, best_reward_std))

        self.logger.log("total time: {:.2f}s".format(time.time() - start_time))
        self.logger.log("best_reward_mean: {:.2f}, best_reward_std: {:.2f}".format(best_reward_mean, best_reward_std))
        self.logger.log("last_10_performance: {:.2f}".format(np.mean(last_10_performance)))
        torch.save(self.policy.state_dict(), os.path.join(self.logger.model_dir, "policy.pth"))
        self.logger.close()

        return {"last_10_performance": np.mean(last_10_performance)}

    def _evaluate(self) -> Dict[str, List[float]]:
        self.policy.eval()
        obs = self.eval_env.reset()
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0
        # eval_step = 0
        # j = 0

        while num_episodes < self._eval_episodes:
            action = self.policy.select_action(obs.reshape(1,-1), deterministic=True)
            next_obs, reward, terminal, _ = self.eval_env.step(action.flatten())
            episode_reward += reward
            episode_length += 1

            obs = next_obs

            # if j <= 2:
            #     # record eval data statistic 
            #     self.logger.logkv("eval/observation", obs[0])
            #     self.logger.logkv("eval/actions", action[0][0])
            #     self.logger.logkv("eval/next_observation", next_obs[0])
            #     self.logger.logkv("eval/terminal", terminal)
            #     self.logger.logkv("eval/reward", reward)
            #     self.logger.set_timestep(eval_step)
            #     self.logger.dumptensorboardsclarkvs()
            #     eval_step += 1

            if terminal:
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length}
                )
                num_episodes +=1
                episode_reward, episode_length = 0, 0
                obs = self.eval_env.reset()
                # j += 1
            
            # if j == 3:
            #     exit()
        
        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer],
            "actor": self.policy.actor
        }
    

    # tta update action
    def _adapted_evaluate(self) -> Dict[str, List[float]]:
        configure_model(self.policy.actor)
        obs = self.eval_env.reset()
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0

        while num_episodes < self._eval_episodes:
            action = self.policy.select_adapted_action(obs.reshape(1,-1), deterministic=True)
            next_obs, reward, terminal, _ = self.eval_env.step(action.flatten())
            episode_reward += reward
            episode_length += 1

            obs = next_obs

            if terminal:
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length}
                )
                num_episodes +=1
                episode_reward, episode_length = 0, 0
                obs = self.eval_env.reset()
        
        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer],
            "actor": self.policy.actor
        }

    
    def evaluate(self) -> Dict[str, float]:
        eval_start_time = time.time()
        # evaluate loop
        for e in tqdm(range(10)):
            # TODO: build a tta network
            self.policy._copy_weight()
            # TODO: copy weight from online to tta 
            # TODO: evaluate 10 times 
            # evaluate current policy
            eval_info = self._evaluate()
            ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
            ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
            norm_ep_rew_mean = self.eval_env.get_normalized_score(ep_reward_mean) * 100
            norm_ep_rew_std = self.eval_env.get_normalized_score(ep_reward_std) * 100

            # # evaluate tta policy
            # eval_info = self._adapted_evaluate()
            # adapted_ep_reward_mean, adapted_ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
            # adapted_ep_length_mean, adapted_ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
            # adapted_norm_ep_rew_mean = self.eval_env.get_normalized_score(adapted_ep_reward_mean) * 100
            # adapted_norm_ep_rew_std = self.eval_env.get_normalized_score(adapted_ep_reward_std) * 100

            # TODO: save the evaluate results
            self.logger.log("norm_ep_rew_mean: {:.2f}, norm_ep_rew_std: {:.2f}".format(norm_ep_rew_mean,  norm_ep_rew_std))

        self.logger.log("total time: {:.2f}s".format(time.time() - eval_start_time))
        self.logger.close()

    
    def compute_vae_likelihood(self, dataset):
        self.evaluate_normal()
        self.evaluate_vae(dataset)
    
    def evaluate_vae(self, dataset) -> Dict[str, float]:
        observations = np.array(dataset["observations"], dtype=np.float32)
        # next_observations = np.array(dataset["next_observations"], dtype=np.float32)
        actions = np.array(dataset["actions"], dtype=np.float32)
        # rewards = np.array(dataset["rewards"], dtype=np.float32).reshape(-1, 1)
        # terminals = np.array(dataset["terminals"], dtype=np.float32).reshape(-1, 1)

        nums = len(observations)

        num_timesteps = 0

        for i in range(nums):
            obs = torch.tensor(observations[i]).to("cuda:0").reshape(1, -1)
            action = torch.tensor(actions[i]).to("cuda:0").reshape(1, -1)

            # update behavior policy
            recon, mean, std = self.policy.behavior_policy(obs, action)
            recon_loss = F.mse_loss(recon, action)
            KL_loss	= -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            vae_loss = recon_loss + KL_loss

            self.logger.logkv("eval/vae_loss", vae_loss.item())
            self.logger.set_timestep(num_timesteps)
            self.logger.dumpkvs()
            num_timesteps += 1

    
    def evaluate_normal(self) -> Dict[str, List[float]]:
        self.policy.eval()
        obs = self.eval_env.reset()
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0
        eval_step = 0

        while num_episodes < self._eval_episodes:
            action, _ = self.policy.select_action(obs.reshape(1,-1), deterministic=True)

            obs = np.array(obs, dtype=np.float32)
            obs = torch.tensor(obs).to("cuda:0").reshape(1, -1)
            action = torch.tensor(action).to("cuda:0").reshape(1, -1)
            # update behavior policy
            recon, mean, std = self.policy.behavior_policy(obs, action)
            recon_loss = F.mse_loss(recon, action)
            KL_loss	= -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            vae_loss = recon_loss + KL_loss
            action = np.array(action.cpu(), dtype=np.float32)

            next_obs, reward, terminal, _ = self.eval_env.step(action.flatten())
            episode_reward += reward
            episode_length += 1

            obs = next_obs

            # record eval data statistic 
            self.logger.logkv("eval/vae_loss", vae_loss.item())
            self.logger.set_timestep(eval_step)
            self.logger.dumpkvs()
            self.logger.dumptensorboardsclarkvs()
            eval_step += 1

            if terminal:
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length}
                )
                num_episodes +=1
                episode_reward, episode_length = 0, 0
                obs = self.eval_env.reset()
        
        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer],
            "actor": self.policy.actor
        }