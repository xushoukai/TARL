import time
import os

import numpy as np
import torch
import torch.nn as nn
import gym
import random
from loguru import logger

from typing import Optional, Dict, List
from tqdm import tqdm
from collections import deque
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger
from offlinerlkit.policy import BasePolicy, SACTTAPolicy
from offlinerlkit.policy_trainer.tent import configure_model, configure_final_layer, configure_model_noDropout


# model-free policy trainer
class TTAMFPolicyTrainer:
    def __init__(
        self,
        policy: SACTTAPolicy,
        eval_env: gym.Env,
        buffer: ReplayBuffer,
        logger: Logger,
        epoch: int = 1000,
        step_per_epoch: int = 1000,
        batch_size: int = 256,
        eval_episodes: int = 10,
        checkpoints: str = None,
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
        self._checkpoints = checkpoints
        self.lr_scheduler = lr_scheduler

    def train(self) -> Dict[str, float]:
        start_time = time.time()

        num_timesteps = 0
        last_10_performance = deque(maxlen=10)
        # record reward
        best_reward_mean = -10000
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
            if best_reward_mean < norm_ep_rew_mean:
                best_reward_mean = norm_ep_rew_mean
                torch.save(self.policy.state_dict(), os.path.join(self.logger.checkpoint_dir, "best_policy.pth"))

        self.logger.log("total time: {:.2f}s".format(time.time() - start_time))
        torch.save(self.policy.state_dict(), os.path.join(self.logger.model_dir, "policy.pth"))
        self.logger.close()

        return {"last_10_performance": np.mean(last_10_performance)}

    def _evaluate(self, evaluate_num) -> Dict[str, List[float]]:
        self.policy.eval()
        obs = self.eval_env.reset()
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0

        while num_episodes < self._eval_episodes:
            action, vae_loss = self.policy.select_action(obs.reshape(1,-1), deterministic=True)
            next_obs, reward, terminal, _ = self.eval_env.step(action.flatten())
            # print("origin reward: ", reward)
            episode_reward += reward
            episode_length += 1

            obs = next_obs

            # self.logger.logkv("eval/vae_loss", vae_loss)
            # self.logger.set_timestep(evaluate_num)
            # self.logger.dumpkvs()
            # evaluate_num += 1

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
            "actor": self.policy.actor,
            "evaluate_num": evaluate_num
        }
    
    # tta update action
    def _adapted_evaluate(self, tta_evaluate_num) -> Dict[str, List[float]]:
        logger.debug("Using _adapted_evaluate")
        self.policy.tta_actor = configure_model_noDropout(self.policy.tta_actor)
        obs = self.eval_env.reset()
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0

        while num_episodes < self._eval_episodes:
            action, _ = self.policy.select_adapted_action(obs.reshape(1,-1), deterministic=True)
            action = action.detach().cpu().numpy()
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
            "tta/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "tta/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer],
            "tta_actor": self.policy.tta_actor, 
            "tta_evaluate_num": tta_evaluate_num
        }
    
    # tta update action for minimizing uncertainty
    def _adapted_mcd_evaluate(self, tta_evaluate_num, weight_sample, loss_agg_type="abs", 
                            entropy_threshold:float = 0.1,
                            is_entropy_filter: bool = False,) -> Dict[str, List[float]]:
        logger.debug("Using _adapted_mcd_evaluate")
        self.policy.tta_actor = configure_model_noDropout(self.policy.tta_actor)
        obs = self.eval_env.reset()
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0

        while num_episodes < self._eval_episodes:
            action, _ = self.policy.select_adapted_mcd_action(obs.reshape(1,-1), deterministic=True, loss_agg_type=loss_agg_type, entropy_threshold=entropy_threshold, is_entropy_filter=is_entropy_filter)
            action = action.detach().cpu().numpy()
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
            "tta/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "tta/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer],
            "tta_actor": self.policy.tta_actor, 
            "tta_evaluate_num": tta_evaluate_num
        }

    def _evaluate_norm(self, seed)-> Dict[str, float]:
        # evaluate current policy
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        self.eval_env.seed(seed)

        evaluate_num = 0
        ep_reward_mean_list = []
        for e in tqdm(range(10)):
            eval_info = self._evaluate(evaluate_num)
            ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
            ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
            norm_ep_rew_mean = self.eval_env.get_normalized_score(ep_reward_mean) * 100
            norm_ep_rew_std = self.eval_env.get_normalized_score(ep_reward_std) * 100
            evaluate_num = eval_info["evaluate_num"]
            ep_reward_mean_list.append(norm_ep_rew_mean)

            # save the evaluate results
            self.logger.log("norm_ep_rew_mean: {:.2f}, norm_ep_rew_std: {:.2f}".format(norm_ep_rew_mean,  norm_ep_rew_std))
            # save the evaluate results

        self.logger.log("10 evaluate result is as follows: average_norm_ep_rew_mean: {:.2f}, std_norm_ep_rew: {:.2f}".format(np.mean(ep_reward_mean_list), np.std(ep_reward_mean_list)))
    
    def _evaluate_tta(self, seed, weight_sample, use_mcd, val_epoch, loss_agg_type="abs",
                        entropy_threshold:float = 0.1,
                        is_entropy_filter: bool = False,)-> Dict[str, float]:
        # evaluate current policy
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        self.eval_env.seed(seed)

        tta_evaluate_num = 0
        adapted_ep_reward_mean_list = []
        for e in tqdm(range(val_epoch)):
            self.policy._copy_weight()
            # evaluate tta policy
            if use_mcd:
                # 1. KL term
                # 2. action-loss
                # 3. entropy.mean()
                eval_info = self._adapted_mcd_evaluate(tta_evaluate_num, weight_sample, loss_agg_type=loss_agg_type, entropy_threshold=entropy_threshold, is_entropy_filter=is_entropy_filter)
            else:
                # 1. enrtopy.abs()
                eval_info = self._adapted_evaluate(tta_evaluate_num)
            adapted_ep_reward_mean, adapted_ep_reward_std = np.mean(eval_info["tta/episode_reward"]), np.std(eval_info["tta/episode_reward"])
            adapted_ep_length_mean, adapted_ep_length_std = np.mean(eval_info["tta/episode_length"]), np.std(eval_info["tta/episode_length"])
            adapted_norm_ep_rew_mean = self.eval_env.get_normalized_score(adapted_ep_reward_mean) * 100
            adapted_norm_ep_rew_std = self.eval_env.get_normalized_score(adapted_ep_reward_std) * 100
            tta_evaluate_num = eval_info["tta_evaluate_num"]
            adapted_ep_reward_mean_list.append(adapted_norm_ep_rew_mean)
            self.logger.log("adapted_norm_ep_rew_mean: {:.2f}, adapted_norm_ep_rew_std: {:.2f}".format(adapted_norm_ep_rew_mean,  adapted_norm_ep_rew_std))
        
        self.logger.log("10 evaluate result is as follows: average_norm_ep_rew_mean: {:.2f}, std_norm_ep_rew: {:.2f}".format(np.mean(adapted_ep_reward_mean_list), np.std(adapted_ep_reward_mean_list)))

    def evaluate(self, seed, only_eval_tta, weight_sample, use_mcd=True, val_epoch=1, loss_agg_type="mean",entropy_threshold:float = 0.1, is_entropy_filter: bool = False,) -> Dict[str, float]:
        eval_start_time = time.time()    

        if not only_eval_tta:
            print("-" * 500)
        self._evaluate_norm(seed)
        self._evaluate_tta(seed, weight_sample, use_mcd, val_epoch, loss_agg_type, entropy_threshold=entropy_threshold, is_entropy_filter=is_entropy_filter)
        
        # self._evaluate_tta(seed)
        # self._evaluate_norm(seed)
    
        self.logger.log("total time: {:.2f}s".format(time.time() - eval_start_time))
        self.logger.close()
