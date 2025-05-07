import time
import os

import numpy as np
import torch
import gym
import random

from typing import Optional, Dict, List, Tuple
from tqdm import tqdm
from collections import deque
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger
from offlinerlkit.policy import BasePolicy


# model-based policy trainer
class TTAMBPolicyTrainer:
    def __init__(
        self,
        policy: BasePolicy,
        eval_env: gym.Env,
        real_buffer: ReplayBuffer,
        fake_buffer: ReplayBuffer,
        logger: Logger,
        rollout_setting: Tuple[int, int, int],
        epoch: int = 1000,
        step_per_epoch: int = 1000,
        batch_size: int = 256,
        real_ratio: float = 0.05,
        eval_episodes: int = 10,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        dynamics_update_freq: int = 0
    ) -> None:
        self.policy = policy
        self.eval_env = eval_env
        self.real_buffer = real_buffer
        self.fake_buffer = fake_buffer
        self.logger = logger

        self._rollout_freq, self._rollout_batch_size, \
            self._rollout_length = rollout_setting
        self._dynamics_update_freq = dynamics_update_freq

        self._epoch = epoch
        self._step_per_epoch = step_per_epoch
        self._batch_size = batch_size
        self._real_ratio = real_ratio
        self._eval_episodes = eval_episodes
        self.lr_scheduler = lr_scheduler

    def train(self) -> Dict[str, float]:
        start_time = time.time()

        num_timesteps = 0
        last_10_performance = deque(maxlen=10)
        best_reward_mean = -10000
        best_reward_std = 0
        # train loop
        for e in range(1, self._epoch + 1):

            self.policy.train()

            pbar = tqdm(range(self._step_per_epoch), desc=f"Epoch #{e}/{self._epoch}")
            for it in pbar:
                if num_timesteps % self._rollout_freq == 0:
                    init_obss = self.real_buffer.sample(self._rollout_batch_size)["observations"].cpu().numpy()
                    rollout_transitions, rollout_info = self.policy.rollout(init_obss, self._rollout_length)
                    self.fake_buffer.add_batch(**rollout_transitions)
                    self.logger.log(
                        "num rollout transitions: {}, reward mean: {:.4f}".\
                            format(rollout_info["num_transitions"], rollout_info["reward_mean"])
                    )
                    for _key, _value in rollout_info.items():
                        self.logger.logkv_mean("rollout_info/"+_key, _value)

                real_sample_size = int(self._batch_size * self._real_ratio)
                fake_sample_size = self._batch_size - real_sample_size
                real_batch = self.real_buffer.sample(batch_size=real_sample_size)
                fake_batch = self.fake_buffer.sample(batch_size=fake_sample_size)
                batch = {"real": real_batch, "fake": fake_batch}
                loss = self.policy.learn(batch)
                pbar.set_postfix(**loss)

                for k, v in loss.items():
                    self.logger.logkv_mean(k, v)
                
                # update the dynamics if necessary
                if 0 < self._dynamics_update_freq and (num_timesteps+1)%self._dynamics_update_freq == 0:
                    dynamics_update_info = self.policy.update_dynamics(self.real_buffer)
                    for k, v in dynamics_update_info.items():
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
            self.logger.dumpkvs(exclude=["dynamics_training_progress"])

            # save checkpoint
            torch.save(self.policy.state_dict(), os.path.join(self.logger.checkpoint_dir, "policy.pth"))
            if e >= 2990:
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

        while num_episodes < self._eval_episodes:
            action = self.policy.select_action(obs.reshape(1, -1), deterministic=True)
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
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }

    def _evaluate_norm(self, seed)-> Dict[str, float]:
        # evaluate current policy
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        self.eval_env.seed(seed) 

        # evaluate policy
        ep_reward_mean_list = []
        for e in tqdm(range(10)):
            eval_info = self._evaluate()
            ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
            ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
            norm_ep_rew_mean = self.eval_env.get_normalized_score(ep_reward_mean) * 100
            norm_ep_rew_std = self.eval_env.get_normalized_score(ep_reward_std) * 100
            ep_reward_mean_list.append(norm_ep_rew_mean)

            # save the evaluate results
            self.logger.log("norm_ep_rew_mean: {:.2f}, norm_ep_rew_std: {:.2f}".format(norm_ep_rew_mean,  norm_ep_rew_std))
            # save the evaluate results

        self.logger.log("10 evaluate result is as follows: average_norm_ep_rew_mean: {:.2f}, std_norm_ep_rew: {:.2f}".format(np.mean(ep_reward_mean_list), np.std(ep_reward_mean_list)))


    def _evaluate_tta(self, seed)-> Dict[str, float]:
        # evaluate current policy
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        self.eval_env.seed(seed)

        tta_evaluate_num = 0
        adapted_ep_reward_mean_list = []
        for e in tqdm(range(10)):
            self.policy._copy_weight()
            # evaluate tta policy
            eval_info = self._adapted_mcd_evaluate(tta_evaluate_num)
            # eval_info = self._adapted_evaluate(tta_evaluate_num)
            adapted_ep_reward_mean, adapted_ep_reward_std = np.mean(eval_info["tta/episode_reward"]), np.std(eval_info["tta/episode_reward"])
            adapted_ep_length_mean, adapted_ep_length_std = np.mean(eval_info["tta/episode_length"]), np.std(eval_info["tta/episode_length"])
            adapted_norm_ep_rew_mean = self.eval_env.get_normalized_score(adapted_ep_reward_mean) * 100
            adapted_norm_ep_rew_std = self.eval_env.get_normalized_score(adapted_ep_reward_std) * 100
            tta_evaluate_num = eval_info["tta_evaluate_num"]
            adapted_ep_reward_mean_list.append(adapted_norm_ep_rew_mean)
            self.logger.log("adapted_norm_ep_rew_mean: {:.2f}, adapted_norm_ep_rew_std: {:.2f}".format(adapted_norm_ep_rew_mean,  adapted_norm_ep_rew_std))
        
        self.logger.log("10 evaluate result is as follows: average_norm_ep_rew_mean: {:.2f}, std_norm_ep_rew: {:.2f}".format(np.mean(adapted_ep_reward_mean_list), np.std(adapted_ep_reward_mean_list)))


    def evaluate(self, seed, only_eval_tta) -> Dict[str, float]:
        eval_start_time = time.time()    

        # if not only_eval_tta:
        #     print("-" * 500)
        #     self._evaluate_norm(seed)
        # self._evaluate_tta(seed)
        
        self._evaluate_tta(seed)
        self._evaluate_norm(seed)
    
        self.logger.log("total time: {:.2f}s".format(time.time() - eval_start_time))
        self.logger.close()