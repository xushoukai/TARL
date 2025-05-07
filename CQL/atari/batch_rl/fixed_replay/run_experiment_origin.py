<<<<<<< HEAD
# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runner for experiments with a fixed replay buffer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import gin
import time
import random
import numpy as np
from tqdm import tqdm
from absl import logging
import tensorflow.compat.v1 as tf

import sys
sys.path.append("/apdcephfs/private_zihaolian/zihaolian/code/transfer_rl/CQL/atari/dopamine")

# from torch.utils.tensorboard import SummaryWriter
from dopamine.discrete_domains import checkpointer
from dopamine.discrete_domains import iteration_statistics
from dopamine.discrete_domains import run_experiment


@gin.configurable
class FixedReplayRunner(run_experiment.Runner):
  """Object that handles running Dopamine experiments with fixed replay buffer."""

  def _initialize_checkpointer_and_maybe_resume(self, checkpoint_file_prefix):
    super(FixedReplayRunner, self)._initialize_checkpointer_and_maybe_resume(
        checkpoint_file_prefix)

    # Code for the loading a checkpoint at initialization
    init_checkpoint_dir = self._agent._init_checkpoint_dir  # pylint: disable=protected-access
    if (self._start_iteration == 1) and (init_checkpoint_dir is not None):
      if checkpointer.get_latest_checkpoint_number(self._checkpoint_dir) < 0:
        # No checkpoint loaded yet, read init_checkpoint_dir
        init_checkpointer = checkpointer.Checkpointer(
            init_checkpoint_dir, checkpoint_file_prefix)
        latest_init_checkpoint = checkpointer.get_latest_checkpoint_number(
            init_checkpoint_dir)
        # latest_init_checkpoint = 210
        if latest_init_checkpoint >= 0:
          experiment_data = init_checkpointer.load_checkpoint(
              latest_init_checkpoint)
          if self._agent.unbundle(
              init_checkpoint_dir, latest_init_checkpoint, experiment_data):
            if experiment_data is not None:
              assert 'logs' in experiment_data
              assert 'current_iteration' in experiment_data
              self._logger.data = experiment_data['logs']
              self._start_iteration = experiment_data['current_iteration'] + 1
            tf.logging.info(
                'Reloaded checkpoint from %s and will start from iteration %d',
                init_checkpoint_dir, self._start_iteration)

  def _run_train_phase(self):
    """Run training phase."""
    self._agent.eval_mode = False
    start_time = time.time()
    # for _ in range(self._training_steps):
    for i in tqdm(range(self._training_steps)):
      # tf.logging.info('train step: %d', i)
      self._agent._train_step()  # pylint: disable=protected-access
    time_delta = time.time() - start_time
    tf.logging.info('Average training steps per second: %.2f',
                    self._training_steps / time_delta)
    
  def _run_eval_iteration(self, iteration):
    """Runs one iteration of agent/environment interaction."""
    statistics = iteration_statistics.IterationStatistics()
    tf.logging.info('Starting iteration %d', iteration)
    num_episodes_eval, average_reward_eval = self._run_eval_phase(statistics)
    self._save_tensorboard_summaries(
        iteration, num_episodes_eval, average_reward_eval)
    
    if self._agent.adapted:
      adapted_num_episodes_eval, adapted_average_reward_eval = self._run_adapted_eval_phase(statistics)
      self._save_tta_tensorboard_summaries(
        iteration, num_episodes_eval, average_reward_eval, adapted_num_episodes_eval, adapted_average_reward_eval)
  
  def setup_seed(self, seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)

  def _run_one_iteration(self, iteration):
    """Runs one iteration of agent/environment interaction."""
    statistics = iteration_statistics.IterationStatistics()
    tf.logging.info('Starting iteration %d', iteration)
    if not self.only_eval:
      # pylint: disable=protected-access
      if not self._agent._replay_suffix:
        # Reload the replay buffer
        self._agent._replay.memory.reload_buffer(num_buffers=5)
      # pylint: enable=protected-access
      self._run_train_phase()

    eval_num = 1
    result_path = self._base_dir + "/result.txt"
    if self.only_eval:
      eval_num = 10
      result_path = self._base_dir + "/" + self._eval_dir + "/result.txt"
      reward_list = []
      adapted_list = []

    if self.only_eval:
      self.setup_seed(0)
    for i in range(eval_num):
      num_episodes_eval, average_reward_eval = self._run_eval_phase(statistics)
      # num_episodes_eval, average_reward_eval = 0, 0
      with open(result_path, "a+") as f:
        f.write("iteration: {}, entropy: {}, num_episode_val: {}, average_reward_eval: {}\n".format(
          iteration, self._agent.entropy, num_episodes_eval, average_reward_eval))

      if self.only_eval:
        reward_list.append(average_reward_eval)
          
        if i == 9:
          mean_reward_eval = np.mean(reward_list)
          std_reward_eval = np.std(reward_list)
          with open(result_path, "a+") as f:
            f.write("mean_reward_eval: {}, std_reward_eval: {}\n".format(mean_reward_eval, std_reward_eval))
      
      if not self.only_eval:
        self._save_tensorboard_summaries(
            iteration, num_episodes_eval, average_reward_eval)

    if self.only_eval:
      self.setup_seed(0)

    if self._agent.adapted:
      for i in range(eval_num):
        # copy parameter of online network to online-tta network
        self._agent.update_online_tta_op()
        if self._agent.apapted_moment:
          self._agent.copy_tta_to_monent()
        #   # print("online: ", (self._agent.online_convnet.layers[2].get_weights()[0])[0][2][45][12])
        #   # print("adapted: ", (self._agent.test_online_convnet.layers[2].get_weights()[0])[0][2][45][12])

        adapted_num_episodes_eval, adapted_average_reward_eval = self._run_adapted_eval_phase(statistics)
        with open(result_path, "a+") as f:
          f.write("iteration: {}, entropy: {}, adapted_num_episodes_eval: {}, adapted_average_reward_eval: {}\n".format(
            iteration, self._agent.entropy, adapted_num_episodes_eval, adapted_average_reward_eval))
        
        self._save_tta_tensorboard_summaries(
          iteration, adapted_num_episodes_eval, adapted_average_reward_eval)
  
        adapted_list.append(adapted_average_reward_eval)

        # # record entropy change
        # self._agent._compute_entropy()
          
        if i == 9:
          adapted_mean_reward_eval = np.mean(adapted_list)
          adapted_std_reward_eval = np.std(adapted_list)
          with open(result_path, "a+") as f:
            f.write("adapted_mean_reward_eval: {}, adapted_std_reward_eval: {}\n".format(adapted_mean_reward_eval, adapted_std_reward_eval))

    # tta_layer = self._agent.test_online_convnet.layers[2].get_weights()[0]
    # target_layer = self._agent.online_convnet.layers[2].get_weights()[0]
    # print("adapted parameter: {}, online parameter: {}".format(tta_layer[0][0][34][45], target_layer[0][0][34][45]))
    # print("adapted parameter: {}, online parameter: {}".format(tta_layer[0][1][23][34], target_layer[0][1][23][34]))
    # print("adapted parameter: {}, online parameter: {}".format(tta_layer[1][0][12][54], target_layer[1][0][12][54]))

    return statistics.data_lists
  
  def _save_tensorboard_summaries(self, iteration,
                                  num_episodes_eval,
                                  average_reward_eval):
    """Save statistics as tensorboard summaries.

    Args:
      iteration: int, The current iteration number.
      num_episodes_eval: int, number of evaluation episodes run.
      average_reward_eval: float, The average evaluation reward.
    """
    summary = tf.Summary(value=[
        tf.Summary.Value(tag='Eval/NumEpisodes',
                         simple_value=num_episodes_eval),
        tf.Summary.Value(tag='Eval/AverageReturns',
                         simple_value=average_reward_eval)
    ])
    self._summary_writer.add_summary(summary, iteration)

  def _save_tta_tensorboard_summaries(self, iteration,
                                  adapted_num_episodes_eval,
                                  adapted_average_reward_eval):
    """Save statistics as tensorboard summaries.

    Args:
      iteration: int, The current iteration number.
      num_episodes_eval: int, number of evaluation episodes run.
      average_reward_eval: float, The average evaluation reward.
    """
    summary = tf.Summary(value=[
        tf.Summary.Value(tag='Eval/AdaptedNumEpisodes',
                         simple_value=adapted_num_episodes_eval),
        tf.Summary.Value(tag='Eval/AdaptedAverageReturns',
                         simple_value=adapted_average_reward_eval)
    ])
    self._summary_writer.add_summary(summary, iteration)
  
  # def _save_tta_tensorboard_summaries(self, iteration,
  #                                 num_episodes_eval,
  #                                 average_reward_eval,
  #                                 adapted_num_episodes_eval,
  #                                 adapted_average_reward_eval):
  #   """Save statistics as tensorboard summaries.

  #   Args:
  #     iteration: int, The current iteration number.
  #     num_episodes_eval: int, number of evaluation episodes run.
  #     average_reward_eval: float, The average evaluation reward.
  #   """
  #   summary = tf.Summary(value=[
  #       tf.Summary.Value(tag='Eval/NumEpisodes',
  #                        simple_value=num_episodes_eval),
  #       tf.Summary.Value(tag='Eval/AverageReturns',
  #                        simple_value=average_reward_eval),
  #       tf.Summary.Value(tag='Eval/AdaptedNumEpisodes',
  #                        simple_value=adapted_num_episodes_eval),
  #       tf.Summary.Value(tag='Eval/AdaptedAverageReturns',
  #                        simple_value=adapted_average_reward_eval)
  #   ])
  #   self._summary_writer.add_summary(summary, iteration)

=======
# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runner for experiments with a fixed replay buffer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import gin
import time
import random
import numpy as np
from tqdm import tqdm
from absl import logging
import tensorflow.compat.v1 as tf

import sys
sys.path.append("/apdcephfs/private_zihaolian/zihaolian/code/transfer_rl/CQL/atari/dopamine")

# from torch.utils.tensorboard import SummaryWriter
from dopamine.discrete_domains import checkpointer
from dopamine.discrete_domains import iteration_statistics
from dopamine.discrete_domains import run_experiment


@gin.configurable
class FixedReplayRunner(run_experiment.Runner):
  """Object that handles running Dopamine experiments with fixed replay buffer."""

  def _initialize_checkpointer_and_maybe_resume(self, checkpoint_file_prefix):
    super(FixedReplayRunner, self)._initialize_checkpointer_and_maybe_resume(
        checkpoint_file_prefix)

    # Code for the loading a checkpoint at initialization
    init_checkpoint_dir = self._agent._init_checkpoint_dir  # pylint: disable=protected-access
    if (self._start_iteration == 1) and (init_checkpoint_dir is not None):
      if checkpointer.get_latest_checkpoint_number(self._checkpoint_dir) < 0:
        # No checkpoint loaded yet, read init_checkpoint_dir
        init_checkpointer = checkpointer.Checkpointer(
            init_checkpoint_dir, checkpoint_file_prefix)
        latest_init_checkpoint = checkpointer.get_latest_checkpoint_number(
            init_checkpoint_dir)
        # latest_init_checkpoint = 210
        if latest_init_checkpoint >= 0:
          experiment_data = init_checkpointer.load_checkpoint(
              latest_init_checkpoint)
          if self._agent.unbundle(
              init_checkpoint_dir, latest_init_checkpoint, experiment_data):
            if experiment_data is not None:
              assert 'logs' in experiment_data
              assert 'current_iteration' in experiment_data
              self._logger.data = experiment_data['logs']
              self._start_iteration = experiment_data['current_iteration'] + 1
            tf.logging.info(
                'Reloaded checkpoint from %s and will start from iteration %d',
                init_checkpoint_dir, self._start_iteration)

  def _run_train_phase(self):
    """Run training phase."""
    self._agent.eval_mode = False
    start_time = time.time()
    # for _ in range(self._training_steps):
    for i in tqdm(range(self._training_steps)):
      # tf.logging.info('train step: %d', i)
      self._agent._train_step()  # pylint: disable=protected-access
    time_delta = time.time() - start_time
    tf.logging.info('Average training steps per second: %.2f',
                    self._training_steps / time_delta)
    
  def _run_eval_iteration(self, iteration):
    """Runs one iteration of agent/environment interaction."""
    statistics = iteration_statistics.IterationStatistics()
    tf.logging.info('Starting iteration %d', iteration)
    num_episodes_eval, average_reward_eval = self._run_eval_phase(statistics)
    self._save_tensorboard_summaries(
        iteration, num_episodes_eval, average_reward_eval)
    
    if self._agent.adapted:
      adapted_num_episodes_eval, adapted_average_reward_eval = self._run_adapted_eval_phase(statistics)
      self._save_tta_tensorboard_summaries(
        iteration, num_episodes_eval, average_reward_eval, adapted_num_episodes_eval, adapted_average_reward_eval)
  
  def setup_seed(self, seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)

  def _run_one_iteration(self, iteration):
    """Runs one iteration of agent/environment interaction."""
    statistics = iteration_statistics.IterationStatistics()
    tf.logging.info('Starting iteration %d', iteration)
    if not self.only_eval:
      # pylint: disable=protected-access
      if not self._agent._replay_suffix:
        # Reload the replay buffer
        self._agent._replay.memory.reload_buffer(num_buffers=5)
      # pylint: enable=protected-access
      self._run_train_phase()

    eval_num = 1
    result_path = self._base_dir + "/result.txt"
    if self.only_eval:
      eval_num = 10
      result_path = self._base_dir + "/" + self._eval_dir + "/result.txt"
      reward_list = []
      adapted_list = []

    if self.only_eval:
      self.setup_seed(0)
    for i in range(eval_num):
      num_episodes_eval, average_reward_eval = self._run_eval_phase(statistics)
      # num_episodes_eval, average_reward_eval = 0, 0
      with open(result_path, "a+") as f:
        f.write("iteration: {}, entropy: {}, num_episode_val: {}, average_reward_eval: {}\n".format(
          iteration, self._agent.entropy, num_episodes_eval, average_reward_eval))

      if self.only_eval:
        reward_list.append(average_reward_eval)
          
        if i == 9:
          mean_reward_eval = np.mean(reward_list)
          std_reward_eval = np.std(reward_list)
          with open(result_path, "a+") as f:
            f.write("mean_reward_eval: {}, std_reward_eval: {}\n".format(mean_reward_eval, std_reward_eval))
      
      if not self.only_eval:
        self._save_tensorboard_summaries(
            iteration, num_episodes_eval, average_reward_eval)

    if self.only_eval:
      self.setup_seed(0)

    if self._agent.adapted:
      for i in range(eval_num):
        # copy parameter of online network to online-tta network
        self._agent.update_online_tta_op()
        if self._agent.apapted_moment:
          self._agent.copy_tta_to_monent()
        #   # print("online: ", (self._agent.online_convnet.layers[2].get_weights()[0])[0][2][45][12])
        #   # print("adapted: ", (self._agent.test_online_convnet.layers[2].get_weights()[0])[0][2][45][12])

        adapted_num_episodes_eval, adapted_average_reward_eval = self._run_adapted_eval_phase(statistics)
        with open(result_path, "a+") as f:
          f.write("iteration: {}, entropy: {}, adapted_num_episodes_eval: {}, adapted_average_reward_eval: {}\n".format(
            iteration, self._agent.entropy, adapted_num_episodes_eval, adapted_average_reward_eval))
        
        self._save_tta_tensorboard_summaries(
          iteration, adapted_num_episodes_eval, adapted_average_reward_eval)
  
        adapted_list.append(adapted_average_reward_eval)

        # # record entropy change
        # self._agent._compute_entropy()
          
        if i == 9:
          adapted_mean_reward_eval = np.mean(adapted_list)
          adapted_std_reward_eval = np.std(adapted_list)
          with open(result_path, "a+") as f:
            f.write("adapted_mean_reward_eval: {}, adapted_std_reward_eval: {}\n".format(adapted_mean_reward_eval, adapted_std_reward_eval))

    # tta_layer = self._agent.test_online_convnet.layers[2].get_weights()[0]
    # target_layer = self._agent.online_convnet.layers[2].get_weights()[0]
    # print("adapted parameter: {}, online parameter: {}".format(tta_layer[0][0][34][45], target_layer[0][0][34][45]))
    # print("adapted parameter: {}, online parameter: {}".format(tta_layer[0][1][23][34], target_layer[0][1][23][34]))
    # print("adapted parameter: {}, online parameter: {}".format(tta_layer[1][0][12][54], target_layer[1][0][12][54]))

    return statistics.data_lists
  
  def _save_tensorboard_summaries(self, iteration,
                                  num_episodes_eval,
                                  average_reward_eval):
    """Save statistics as tensorboard summaries.

    Args:
      iteration: int, The current iteration number.
      num_episodes_eval: int, number of evaluation episodes run.
      average_reward_eval: float, The average evaluation reward.
    """
    summary = tf.Summary(value=[
        tf.Summary.Value(tag='Eval/NumEpisodes',
                         simple_value=num_episodes_eval),
        tf.Summary.Value(tag='Eval/AverageReturns',
                         simple_value=average_reward_eval)
    ])
    self._summary_writer.add_summary(summary, iteration)

  def _save_tta_tensorboard_summaries(self, iteration,
                                  adapted_num_episodes_eval,
                                  adapted_average_reward_eval):
    """Save statistics as tensorboard summaries.

    Args:
      iteration: int, The current iteration number.
      num_episodes_eval: int, number of evaluation episodes run.
      average_reward_eval: float, The average evaluation reward.
    """
    summary = tf.Summary(value=[
        tf.Summary.Value(tag='Eval/AdaptedNumEpisodes',
                         simple_value=adapted_num_episodes_eval),
        tf.Summary.Value(tag='Eval/AdaptedAverageReturns',
                         simple_value=adapted_average_reward_eval)
    ])
    self._summary_writer.add_summary(summary, iteration)
  
  # def _save_tta_tensorboard_summaries(self, iteration,
  #                                 num_episodes_eval,
  #                                 average_reward_eval,
  #                                 adapted_num_episodes_eval,
  #                                 adapted_average_reward_eval):
  #   """Save statistics as tensorboard summaries.

  #   Args:
  #     iteration: int, The current iteration number.
  #     num_episodes_eval: int, number of evaluation episodes run.
  #     average_reward_eval: float, The average evaluation reward.
  #   """
  #   summary = tf.Summary(value=[
  #       tf.Summary.Value(tag='Eval/NumEpisodes',
  #                        simple_value=num_episodes_eval),
  #       tf.Summary.Value(tag='Eval/AverageReturns',
  #                        simple_value=average_reward_eval),
  #       tf.Summary.Value(tag='Eval/AdaptedNumEpisodes',
  #                        simple_value=adapted_num_episodes_eval),
  #       tf.Summary.Value(tag='Eval/AdaptedAverageReturns',
  #                        simple_value=adapted_average_reward_eval)
  #   ])
  #   self._summary_writer.add_summary(summary, iteration)

>>>>>>> ed0d7b9c1ae446c6cffed41684f9178952a22685
