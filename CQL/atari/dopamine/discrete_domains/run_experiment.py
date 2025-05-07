<<<<<<< HEAD
# coding=utf-8
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module defining classes and helper methods for general agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
from shutil import copy

from absl import logging

from dopamine.agents.dqn import dqn_agent
from dopamine.agents.implicit_quantile import implicit_quantile_agent
from dopamine.agents.rainbow import rainbow_agent
from dopamine.discrete_domains import atari_lib
from dopamine.discrete_domains import checkpointer
from dopamine.discrete_domains import iteration_statistics
from dopamine.discrete_domains import logger
from dopamine.jax.agents.dqn import dqn_agent as jax_dqn_agent
from dopamine.jax.agents.full_rainbow import full_rainbow_agent
from dopamine.jax.agents.implicit_quantile import implicit_quantile_agent as jax_implicit_quantile_agent
from dopamine.jax.agents.quantile import quantile_agent as jax_quantile_agent
from dopamine.jax.agents.rainbow import rainbow_agent as jax_rainbow_agent
from dopamine.metrics import collector_dispatcher
from dopamine.metrics import statistics_instance
import gin.tf
import numpy as np
import tensorflow as tf


def load_gin_configs(gin_files, gin_bindings):
  """Loads gin configuration files.

  Args:
    gin_files: list, of paths to the gin configuration files for this
      experiment.
    gin_bindings: list, of gin parameter bindings to override the values in
      the config files.
  """
  gin.parse_config_files_and_bindings(gin_files,
                                      bindings=gin_bindings,
                                      skip_unknown=False)


@gin.configurable
def create_agent(sess, environment, agent_name=None, summary_writer=None,
                 debug_mode=False):
  """Creates an agent.

  Args:
    sess: A `tf.compat.v1.Session` object for running associated ops.
    environment: A gym environment (e.g. Atari 2600).
    agent_name: str, name of the agent to create.
    summary_writer: A Tensorflow summary writer to pass to the agent
      for in-agent training statistics in Tensorboard.
    debug_mode: bool, whether to output Tensorboard summaries. If set to true,
      the agent will output in-episode statistics to Tensorboard. Disabled by
      default as this results in slower training.

  Returns:
    agent: An RL agent.

  Raises:
    ValueError: If `agent_name` is not in supported list.
  """
  assert agent_name is not None
  if not debug_mode:
    summary_writer = None
  if agent_name.startswith('dqn'):
    return dqn_agent.DQNAgent(sess, num_actions=environment.action_space.n,
                              summary_writer=summary_writer)
  elif agent_name == 'rainbow':
    return rainbow_agent.RainbowAgent(
        sess, num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  elif agent_name == 'implicit_quantile':
    return implicit_quantile_agent.ImplicitQuantileAgent(
        sess, num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  elif agent_name == 'jax_dqn':
    return jax_dqn_agent.JaxDQNAgent(num_actions=environment.action_space.n,
                                     summary_writer=summary_writer)
  elif agent_name == 'jax_quantile':
    return jax_quantile_agent.JaxQuantileAgent(
        num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  elif agent_name == 'jax_rainbow':
    return jax_rainbow_agent.JaxRainbowAgent(
        num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  elif agent_name == 'full_rainbow':
    return full_rainbow_agent.JaxFullRainbowAgent(
        num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  elif agent_name == 'jax_implicit_quantile':
    return jax_implicit_quantile_agent.JaxImplicitQuantileAgent(
        num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  else:
    raise ValueError('Unknown agent: {}'.format(agent_name))


@gin.configurable
def create_runner(base_dir, schedule='continuous_train_and_eval'):
  """Creates an experiment Runner.

  Args:
    base_dir: str, base directory for hosting all subdirectories.
    schedule: string, which type of Runner to use.

  Returns:
    runner: A `Runner` like object.

  Raises:
    ValueError: When an unknown schedule is encountered.
  """
  assert base_dir is not None
  # Continuously runs training and evaluation until max num_iterations is hit.
  if schedule == 'continuous_train_and_eval':
    return Runner(base_dir, create_agent)
  # Continuously runs training until max num_iterations is hit.
  elif schedule == 'continuous_train':
    return TrainRunner(base_dir, create_agent)
  else:
    raise ValueError('Unknown schedule: {}'.format(schedule))


@gin.configurable
class Runner(object):
  """Object that handles running Dopamine experiments.

  Here we use the term 'experiment' to mean simulating interactions between the
  agent and the environment and reporting some statistics pertaining to these
  interactions.

  A simple scenario to train a DQN agent is as follows:

  ```python
  import dopamine.discrete_domains.atari_lib
  base_dir = '/tmp/simple_example'
  def create_agent(sess, environment):
    return dqn_agent.DQNAgent(sess, num_actions=environment.action_space.n)
  runner = Runner(base_dir, create_agent, atari_lib.create_atari_environment)
  runner.run()
  ```
  """

  def __init__(self,
               base_dir,
               create_agent_fn,
               only_eval=False,
               eval_dir='tent',
               create_environment_fn=atari_lib.create_atari_environment,
               checkpoint_file_prefix='ckpt',
               logging_file_prefix='log',
               log_every_n=1,
               num_iterations=200,
               training_steps=250000,
               evaluation_steps=125000,
               max_steps_per_episode=27000,
               clip_rewards=True,
               use_legacy_logger=True,
               fine_grained_print_to_console=True,
               seed=0):
    """Initialize the Runner object in charge of running a full experiment.

    Args:
      base_dir: str, the base directory to host all required sub-directories.
      create_agent_fn: A function that takes as args a Tensorflow session and an
        environment, and returns an agent.
      create_environment_fn: A function which receives a problem name and
        creates a Gym environment for that problem (e.g. an Atari 2600 game).
      checkpoint_file_prefix: str, the prefix to use for checkpoint files.
      logging_file_prefix: str, prefix to use for the log files.
      log_every_n: int, the frequency for writing logs.
      num_iterations: int, the iteration number threshold (must be greater than
        start_iteration).
      training_steps: int, the number of training steps to perform.
      evaluation_steps: int, the number of evaluation steps to perform.
      max_steps_per_episode: int, maximum number of steps after which an episode
        terminates.
      clip_rewards: bool, whether to clip rewards in [-1, 1].
      use_legacy_logger: bool, whether to use the legacy Logger. This will be
        deprecated soon, replaced with the new CollectorDispatcher setup.
      fine_grained_print_to_console: bool, whether to print fine-grained
        progress to console (useful for debugging).

    This constructor will take the following actions:
    - Initialize an environment.
    - Initialize a `tf.compat.v1.Session`.
    - Initialize a logger.
    - Initialize an agent.
    - Reload from the latest checkpoint, if available, and initialize the
      Checkpointer object.
    """
    assert base_dir is not None

    self.only_eval = only_eval
    self._eval_dir = eval_dir
    self._legacy_logger_enabled = use_legacy_logger
    self._fine_grained_print_to_console_enabled = fine_grained_print_to_console
    self._logging_file_prefix = logging_file_prefix
    self._log_every_n = log_every_n
    self._num_iterations = num_iterations
    self._training_steps = training_steps
    self._evaluation_steps = evaluation_steps
    self._max_steps_per_episode = max_steps_per_episode
    self._base_dir = base_dir
    self._clip_rewards = clip_rewards
    self._seed = seed
    self.record_test_reward_step = 0
    self.record_tta_reward_step = 0
    self._create_directories()

    self._environment, self._originEnv = create_environment_fn()
    # The agent is now in charge of setting up the session.
    self._sess = None
    # We're using a bit of a hack in that we pass in _base_dir instead of an
    # actually SummaryWriter. This is because the agent is now in charge of the
    # session, but needs to create the SummaryWriter before creating the ops,
    # and in order to do so, it requires the base directory.
    # create eval tensorflow
    if self.only_eval:
      summary_writer = self._base_dir + "/" + self._eval_dir
    else:
      summary_writer = self._base_dir

    self._agent = create_agent_fn(self._sess, self._environment,
                                  summary_writer=summary_writer)
    if hasattr(self._agent, '_sess'):
      self._sess = self._agent._sess
    self._summary_writer = self._agent.summary_writer

    self._initialize_checkpointer_and_maybe_resume(checkpoint_file_prefix)

    # Create a collector dispatcher for metrics reporting.
    self._collector_dispatcher = collector_dispatcher.CollectorDispatcher(
        self._base_dir)
    set_collector_dispatcher_fn = getattr(
        self._agent, 'set_collector_dispatcher', None)
    if callable(set_collector_dispatcher_fn):
      set_collector_dispatcher_fn(self._collector_dispatcher)

  @property
  def _use_legacy_logger(self):
    if not hasattr(self, '_legacy_logger_enabled'):
      return True
    return self._legacy_logger_enabled

  @property
  def _has_collector_dispatcher(self):
    if not hasattr(self, '_collector_dispatcher'):
      return False
    return True

  @property
  def _fine_grained_print_to_console(self):
    if not hasattr(self, '_fine_grained_print_to_console_enabled'):
      return True
    return self._fine_grained_print_to_console_enabled

  def _create_directories(self):
    """Create necessary sub-directories."""
    self._checkpoint_dir = os.path.join(self._base_dir, 'checkpoints')
    if self._use_legacy_logger:
      logging.warning(
          'DEPRECATION WARNING: Logger is being deprecated. '
          'Please switch to CollectorDispatcher!')
      self._logger = logger.Logger(os.path.join(self._base_dir, 'logs'))
    
    if self.only_eval:
      eval_path = self._base_dir + "/" + self._eval_dir
      if not os.path.exists(eval_path):
        os.makedirs(eval_path)

  def _initialize_checkpointer_and_maybe_resume(self, checkpoint_file_prefix):
    """Reloads the latest checkpoint if it exists.

    This method will first create a `Checkpointer` object and then call
    `checkpointer.get_latest_checkpoint_number` to determine if there is a valid
    checkpoint in self._checkpoint_dir, and what the largest file number is.
    If a valid checkpoint file is found, it will load the bundled data from this
    file and will pass it to the agent for it to reload its data.
    If the agent is able to successfully unbundle, this method will verify that
    the unbundled data contains the keys,'logs' and 'current_iteration'. It will
    then load the `Logger`'s data from the bundle, and will return the iteration
    number keyed by 'current_iteration' as one of the return values (along with
    the `Checkpointer` object).

    Args:
      checkpoint_file_prefix: str, the checkpoint file prefix.

    Returns:
      start_iteration: int, the iteration number to start the experiment from.
      experiment_checkpointer: `Checkpointer` object for the experiment.
    """
    self._checkpointer = checkpointer.Checkpointer(self._checkpoint_dir,
                                                   checkpoint_file_prefix)
    self._start_iteration = 1
    # Check if checkpoint exists. Note that the existence of checkpoint 0 means
    # that we have finished iteration 0 (so we will start from iteration 1).
    latest_checkpoint_version = checkpointer.get_latest_checkpoint_number(
        self._checkpoint_dir)
    # latest_checkpoint_version = 210
    if latest_checkpoint_version >= 0:
      experiment_data = self._checkpointer.load_checkpoint(
          latest_checkpoint_version)
      if self._agent.unbundle(
          self._checkpoint_dir, latest_checkpoint_version, experiment_data):
        if experiment_data is not None:
          assert 'logs' in experiment_data
          assert 'current_iteration' in experiment_data
          if self._use_legacy_logger:
            self._logger.data = experiment_data['logs']
          self._start_iteration = experiment_data['current_iteration'] + 1
        logging.info('Reloaded checkpoint and will start from iteration %d',
                     self._start_iteration)

  def _initialize_episode(self):
    """Initialization for a new episode.

    Returns:
      action: int, the initial action chosen by the agent.
    """
    initial_observation = self._environment.reset()
    # print("initial_observation'shape: ", initial_observation.shape)
    # exit()
    return self._agent.begin_episode(initial_observation)
  
  def _adapted_initialize_episode(self):
    """Initialization for a new episode.

    Returns:
      action: int, the initial action chosen by the agent.
    """
    initial_observation = self._environment.reset()

    # print("adapted initial_observation: ", initial_observation)
    # print("adapted initial_observation'shape: ", initial_observation.shape)
    # exit()

    # for (w_online_layer, w_tta_layer) in zip(self._agent.test_online_convnet.layers, self._agent.online_convnet.layers):
    #   # Assign weights from online to target network.
    #   print(((w_online_layer.get_weights()))[0].all() == ((w_tta_layer.get_weights())[0]).all())

    return self._agent.adapted_begin_episode(initial_observation)

  def _run_one_step(self, action):
    """Executes a single step in the environment.

    Args:
      action: int, the action to perform in the environment.

    Returns:
      The observation, reward, and is_terminal values returned from the
        environment.
    """
    observation, reward, is_terminal, _ = self._environment.step(action)
    return observation, reward, is_terminal

  def _end_episode(self, reward, terminal=True):
    """Finalizes an episode run.

    Args:
      reward: float, the last reward from the environment.
      terminal: bool, whether the last state-action led to a terminal state.
    """
    if isinstance(self._agent, jax_dqn_agent.JaxDQNAgent):
      self._agent.end_episode(reward, terminal)
    else:
      # TODO(joshgreaves): Add terminal signal to TF dopamine agents
      self._agent.end_episode(reward)

  def _run_adapted_one_episode(self):
    """Executes a full trajectory of the agent interacting with the environment.

    Returns:
      The number of steps taken and the total reward.
    """
    step_number = 0
    total_reward = 0.

    # TODO: adapted in test-time
    action = self._adapted_initialize_episode()
    is_terminal = False

    # Keep interacting until we reach a terminal state.
    while True:
      observation, reward, is_terminal = self._run_one_step(action)

      total_reward += reward
      step_number += 1

      if self._clip_rewards:
        # Perform reward clipping.
        reward = np.clip(reward, -1, 1)

      if (self._environment.game_over or
          step_number == self._max_steps_per_episode):
        # Stop the run loop once we reach the true end of episode.
        break
      elif is_terminal:
        # If we lose a life but the episode is not over, signal an artificial
        # end of episode to the agent.
        self._end_episode(reward, is_terminal)
        # TODO: adapted in test-time
        action = self._agent.adapted_begin_episode(observation)
      else:
        # TODO: modify to test-time adaptation
        action = self._agent.adapted_step(reward, observation)

      # record reward tensorboard
      if self._summary_writer is not None:
        self._save_reward_summaries(self.record_tta_reward_step, reward, "tta")
      self.record_tta_reward_step += 1

    self._end_episode(reward, is_terminal)

    return step_number, total_reward

  def _run_one_episode(self):
    """Executes a full trajectory of the agent interacting with the environment.

    Returns:
      The number of steps taken and the total reward.
    """
    step_number = 0
    total_reward = 0.

    # TODO: adapted in test-time
    action = self._initialize_episode()
    is_terminal = False

    # Keep interacting until we reach a terminal state.
    while True:
      observation, reward, is_terminal = self._run_one_step(action)

      total_reward += reward
      step_number += 1

      if self._clip_rewards:
        # Perform reward clipping.
        reward = np.clip(reward, -1, 1)

      if (self._environment.game_over or
          step_number == self._max_steps_per_episode):
        # Stop the run loop once we reach the true end of episode.
        break
      elif is_terminal:
        # If we lose a life but the episode is not over, signal an artificial
        # end of episode to the agent.
        self._end_episode(reward, is_terminal)
        # TODO: adapted in test-time
        action = self._agent.begin_episode(observation)
      else:
        # TODO: modify to test-time adaptation
        action = self._agent.step(reward, observation)

        # sys.stdout.write('action: {} '.format(action))
        # sys.stdout.flush()

      # record reward tensorboard
      if self._summary_writer is not None:
        self._save_reward_summaries(self.record_test_reward_step, reward, "test")
      self.record_test_reward_step += 1

    self._end_episode(reward, is_terminal)

    return step_number, total_reward

  def _run_one_phase(self, min_steps, statistics, run_mode_str):
    """Runs the agent/environment loop until a desired number of steps.

    We follow the Machado et al., 2017 convention of running full episodes,
    and terminating once we've run a minimum number of steps.

    Args:
      min_steps: int, minimum number of steps to generate in this phase.
      statistics: `IterationStatistics` object which records the experimental
        results.
      run_mode_str: str, describes the run mode for this agent.

    Returns:
      Tuple containing the number of steps taken in this phase (int), the sum of
        returns (float), and the number of episodes performed (int).
    """
    step_count = 0
    num_episodes = 0
    sum_returns = 0.

    while step_count < min_steps:
      episode_length, episode_return = self._run_one_episode()
      statistics.append({
          '{}_episode_lengths'.format(run_mode_str): episode_length,
          '{}_episode_returns'.format(run_mode_str): episode_return
      })
      step_count += episode_length
      sum_returns += episode_return
      num_episodes += 1
      if self._fine_grained_print_to_console:
        # We use sys.stdout.write instead of logging so as to flush frequently
        # without generating a line break.
        sys.stdout.write('Steps executed: {} '.format(step_count) +
                          'Episode length: {} '.format(episode_length) +
                          'Return: {}\r'.format(episode_return))
        sys.stdout.flush()
    return step_count, sum_returns, num_episodes

  def _run_adapted_one_phase(self, min_steps, statistics, run_mode_str):
    """Runs the agent/environment loop until a desired number of steps.

    We follow the Machado et al., 2017 convention of running full episodes,
    and terminating once we've run a minimum number of steps.

    Args:
      min_steps: int, minimum number of steps to generate in this phase.
      statistics: `IterationStatistics` object which records the experimental
        results.
      run_mode_str: str, describes the run mode for this agent.

    Returns:
      Tuple containing the number of steps taken in this phase (int), the sum of
        returns (float), and the number of episodes performed (int).
    """
    step_count = 0
    num_episodes = 0
    sum_returns = 0.

    while step_count < min_steps:
      episode_length, episode_return = self._run_adapted_one_episode()
      statistics.append({
          '{}_adapted_episode_lengths'.format(run_mode_str): episode_length,
          '{}_adapted_episode_returns'.format(run_mode_str): episode_return
      })
      step_count += episode_length
      sum_returns += episode_return
      num_episodes += 1
      if self._fine_grained_print_to_console:
        # We use sys.stdout.write instead of logging so as to flush frequently
        # without generating a line break.
        sys.stdout.write('Adapted Steps executed: {} '.format(step_count) +
                          'Adpated Episode length: {} '.format(episode_length) +
                          'Adpated Return: {}\r'.format(episode_return))
        sys.stdout.flush()
    return step_count, sum_returns, num_episodes

  def _run_train_phase(self, statistics):
    """Run training phase.

    Args:
      statistics: `IterationStatistics` object which records the experimental
        results. Note - This object is modified by this method.

    Returns:
      num_episodes: int, The number of episodes run in this phase.
      average_reward: float, The average reward generated in this phase.
      average_steps_per_second: float, The average number of steps per second.
    """
    # Perform the training phase, during which the agent learns.
    self._agent.eval_mode = False
    start_time = time.time()
    number_steps, sum_returns, num_episodes = self._run_one_phase(
        self._training_steps, statistics, 'train')
    average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
    statistics.append({'train_average_return': average_return})
    time_delta = time.time() - start_time
    average_steps_per_second = number_steps / time_delta
    statistics.append(
        {'train_average_steps_per_second': average_steps_per_second})
    logging.info('Average undiscounted return per training episode: %.2f',
                 average_return)
    logging.info('Average training steps per second: %.2f',
                 average_steps_per_second)
    return num_episodes, average_return, average_steps_per_second
  
  def _run_adapted_eval_phase(self, statistics):
    """Run evaluation phase.

    Args:
      statistics: `IterationStatistics` object which records the experimental
        results. Note - This object is modified by this method.

    Returns:
      num_episodes: int, The number of episodes run in this phase.
      average_reward: float, The average reward generated in this phase.
    """
    # Perform the evaluation phase -- no learning.
    self._agent.eval_mode = True
    _, sum_returns, num_episodes = self._run_adapted_one_phase(
        self._evaluation_steps, statistics, 'tta')
    average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
    logging.info('Adapted Average undiscounted return per evaluation episode: %.2f',
                 average_return)
    statistics.append({'eval_adapted_average_return': average_return})
    return num_episodes, average_return

  def _run_eval_phase(self, statistics):
    """Run evaluation phase.

    Args:
      statistics: `IterationStatistics` object which records the experimental
        results. Note - This object is modified by this method.

    Returns:
      num_episodes: int, The number of episodes run in this phase.
      average_reward: float, The average reward generated in this phase.
    """
    # Perform the evaluation phase -- no learning.
    self._agent.eval_mode = True
    _, sum_returns, num_episodes = self._run_one_phase(
        self._evaluation_steps, statistics, 'eval')
    average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
    logging.info('Average undiscounted return per evaluation episode: %.2f',
                 average_return)
    statistics.append({'eval_average_return': average_return})
    return num_episodes, average_return
  
  def _run_one_iteration(self, iteration):
    """Runs one iteration of agent/environment interaction.

    An iteration involves running several episodes until a certain number of
    steps are obtained. The interleaving of train/eval phases implemented here
    are to match the implementation of (Mnih et al., 2015).

    Args:
      iteration: int, current iteration number, used as a global_step for saving
        Tensorboard summaries.

    Returns:
      A dict containing summary statistics for this iteration.
    """
    statistics = iteration_statistics.IterationStatistics()
    logging.info('Starting iteration %d', iteration)
    num_episodes_train, average_reward_train, average_steps_per_second = (
        self._run_train_phase(statistics))
    num_episodes_eval, average_reward_eval = self._run_eval_phase(
        statistics)

    if self._has_collector_dispatcher:
      self._collector_dispatcher.write([
          statistics_instance.StatisticsInstance('Train/NumEpisodes',
                                                 num_episodes_train,
                                                 iteration),
          statistics_instance.StatisticsInstance('Train/AverageReturns',
                                                 average_reward_train,
                                                 iteration),
          statistics_instance.StatisticsInstance('Train/AverageStepsPerSecond',
                                                 average_steps_per_second,
                                                 iteration),
          statistics_instance.StatisticsInstance('Eval/NumEpisodes',
                                                 num_episodes_eval,
                                                 iteration),
          statistics_instance.StatisticsInstance('Eval/AverageReturns',
                                                 average_reward_eval,
                                                 iteration),
      ])
    if self._summary_writer is not None:
      self._save_tensorboard_summaries(iteration, num_episodes_train,
                                       average_reward_train, num_episodes_eval,
                                       average_reward_eval,
                                       average_steps_per_second)
    return statistics.data_lists
  
  def _save_reward_summaries(self, step, reward, eval_str):
    if self._summary_writer is None:
      return

    if self._sess is None:
      with self._summary_writer.as_default():
        tf.summary.scalar(eval_str + '/reward', reward,
                          step=step)
      self._summary_writer.flush()
    else:
      summary = tf.compat.v1.Summary(value=[
          tf.compat.v1.Summary.Value(
              tag=eval_str + '/reward', simple_value=reward),
      ])
      self._summary_writer.add_summary(summary, step)

  def _save_tensorboard_summaries(self, iteration,
                                  num_episodes_train,
                                  average_reward_train,
                                  num_episodes_eval,
                                  average_reward_eval,
                                  average_steps_per_second):
    """Save statistics as tensorboard summaries.

    Args:
      iteration: int, The current iteration number.
      num_episodes_train: int, number of training episodes run.
      average_reward_train: float, The average training reward.
      num_episodes_eval: int, number of evaluation episodes run.
      average_reward_eval: float, The average evaluation reward.
      average_steps_per_second: float, The average number of steps per second.
    """
    if self._summary_writer is None:
      return

    if self._sess is None:
      with self._summary_writer.as_default():
        tf.summary.scalar('Train/NumEpisodes', num_episodes_train,
                          step=iteration)
        tf.summary.scalar('Train/AverageReturns', average_reward_train,
                          step=iteration)
        tf.summary.scalar('Train/AverageStepsPerSecond',
                          average_steps_per_second, step=iteration)
        tf.summary.scalar('Eval/NumEpisodes', num_episodes_eval, step=iteration)
        tf.summary.scalar('Eval/AverageReturns', average_reward_eval,
                          step=iteration)
      self._summary_writer.flush()
    else:
      summary = tf.compat.v1.Summary(value=[
          tf.compat.v1.Summary.Value(
              tag='Train/NumEpisodes', simple_value=num_episodes_train),
          tf.compat.v1.Summary.Value(
              tag='Train/AverageReturns', simple_value=average_reward_train),
          tf.compat.v1.Summary.Value(
              tag='Train/AverageStepsPerSecond',
              simple_value=average_steps_per_second),
          tf.compat.v1.Summary.Value(
              tag='Eval/NumEpisodes', simple_value=num_episodes_eval),
          tf.compat.v1.Summary.Value(
              tag='Eval/AverageReturns', simple_value=average_reward_eval)
      ])
      self._summary_writer.add_summary(summary, iteration)

  def _log_experiment(self, iteration, statistics):
    """Records the results of the current iteration.

    Args:
      iteration: int, iteration number.
      statistics: `IterationStatistics` object containing statistics to log.
    """
    if not hasattr(self, '_logger'):
      return

    self._logger['iteration_{:d}'.format(iteration)] = statistics
    if iteration % self._log_every_n == 0:
      self._logger.log_to_file(self._logging_file_prefix, iteration, is_delete=False)

  def _checkpoint_experiment(self, iteration):
    """Checkpoint experiment data.

    Args:
      iteration: int, iteration number for checkpointing.
    """
    experiment_data = self._agent.bundle_and_checkpoint(self._checkpoint_dir,
                                                        iteration)
    if experiment_data:
      experiment_data['current_iteration'] = iteration
      if self._use_legacy_logger:
        experiment_data['logs'] = self._logger.data
      self._checkpointer.save_checkpoint(iteration, experiment_data, is_delete=False)

  def save_sub_best_checkpoints(self, iteration):
    ckpt_file = os.path.join(self._checkpoint_dir, "ckpt.10000")
    sentinel_file = os.path.join(self._checkpoint_dir, "sentinel_checkpoint_complete.10000")
    tf_ckpt_data_file = os.path.join(self._checkpoint_dir, "tf_ckpt-10000.data-00000-of-00001")
    tf_ckpt_index_file = os.path.join(self._checkpoint_dir, "tf_ckpt-10000.index")
    tf_ckpt_meta_file = os.path.join(self._checkpoint_dir, "tf_ckpt-10000.meta")
    logs_dir = os.path.join(self._base_dir, 'logs', "log_10000")

    new_ckpt_file = os.path.join(self._checkpoint_dir, "ckpt." + str(iteration + 10))
    new_sentinel_file = os.path.join(self._checkpoint_dir, "sentinel_checkpoint_complete." + str(iteration + 10))
    new_tf_ckpt_data_file = os.path.join(self._checkpoint_dir, "tf_ckpt-" + str(iteration + 10) + ".data-00000-of-00001")
    new_tf_ckpt_index_file = os.path.join(self._checkpoint_dir, "tf_ckpt-" + str(iteration + 10) + ".index")
    new_tf_ckpt_meta_file = os.path.join(self._checkpoint_dir, "tf_ckpt-" + str(iteration + 10) + ".meta")
    new_logs_dir = os.path.join(self._base_dir, 'logs', "log_" + str(iteration + 10))    

    copy(ckpt_file, new_ckpt_file)
    copy(sentinel_file, new_sentinel_file)
    copy(tf_ckpt_data_file, new_tf_ckpt_data_file)
    copy(tf_ckpt_index_file, new_tf_ckpt_index_file)
    copy(tf_ckpt_meta_file, new_tf_ckpt_meta_file)
    copy(logs_dir, new_logs_dir)

  def run_experiment(self):
    """Runs a full experiment, spread over multiple iterations."""
    logging.info('Beginning training...')
    if (self._num_iterations + 1) <= self._start_iteration:
    # if (self._num_iterations + 1) <= 212:
      logging.warning('num_iterations (%d) < start_iteration(%d)',
                      self._num_iterations, self._start_iteration)
      return

    best_average_returns = -float('inf')
    for iteration in range(self._start_iteration, self._num_iterations + 1):
      statistics = self._run_one_iteration(iteration)
      eval_average_return_list = statistics["eval_average_return"]
      average_return = eval_average_return_list[len(eval_average_return_list) - 1]

      if not self.only_eval:
        # return best checkpoints
        if best_average_returns < average_return:
          best_average_returns = average_return
          if self._use_legacy_logger:
            self._log_experiment(10000, statistics)
        
          self._checkpoint_experiment(10000)

        # record every 100 iteration
        if iteration % 100 == 0 and iteration != 0:
          if self._use_legacy_logger:
            self._log_experiment(iteration, statistics)
        
          self._checkpoint_experiment(iteration)

          # record the best checkpoints in every 100 iterations
          self.save_sub_best_checkpoints(iteration)

          # record best result
          result_path = self._base_dir + "/result.txt"
          with open(result_path, "a+") as f:
              f.write("iteration: {}, best_average_returns: {}\n".format(
                iteration, best_average_returns))

          if self._has_collector_dispatcher:
            self._collector_dispatcher.flush()
          
    if self._summary_writer is not None:
      self._summary_writer.flush()
    
    if not self.only_eval:
      if self._has_collector_dispatcher:
        self._collector_dispatcher.close()


@gin.configurable
class TrainRunner(Runner):
  """Object that handles running experiments.

  The `TrainRunner` differs from the base `Runner` class in that it does not
  the evaluation phase. Checkpointing and logging for the train phase are
  preserved as before.
  """

  def __init__(self, base_dir, create_agent_fn,
               create_environment_fn=atari_lib.create_atari_environment):
    """Initialize the TrainRunner object in charge of running a full experiment.

    Args:
      base_dir: str, the base directory to host all required sub-directories.
      create_agent_fn: A function that takes as args a Tensorflow session and an
        environment, and returns an agent.
      create_environment_fn: A function which receives a problem name and
        creates a Gym environment for that problem (e.g. an Atari 2600 game).
    """
    logging.info('Creating TrainRunner ...')
    super(TrainRunner, self).__init__(base_dir, create_agent_fn,
                                      create_environment_fn)
    self._agent.eval_mode = False

  def _run_one_iteration(self, iteration):
    """Runs one iteration of agent/environment interaction.

    An iteration involves running several episodes until a certain number of
    steps are obtained. This method differs from the `_run_one_iteration` method
    in the base `Runner` class in that it only runs the train phase.

    Args:
      iteration: int, current iteration number, used as a global_step for saving
        Tensorboard summaries.

    Returns:
      A dict containing summary statistics for this iteration.
    """
    statistics = iteration_statistics.IterationStatistics()
    num_episodes_train, average_reward_train, average_steps_per_second = (
        self._run_train_phase(statistics))

    self._collector_dispatcher.write([
        statistics_instance.StatisticsInstance('Train/NumEpisodes',
                                               num_episodes_train,
                                               iteration),
        statistics_instance.StatisticsInstance('Train/AverageReturns',
                                               average_reward_train,
                                               iteration),
        statistics_instance.StatisticsInstance('Train/AverageStepsPerSecond',
                                               average_steps_per_second,
                                               iteration),
    ])
    if self._summary_writer is not None:
      self._save_tensorboard_summaries(iteration, num_episodes_train,
                                       average_reward_train,
                                       average_steps_per_second)
    return statistics.data_lists

  def _save_tensorboard_summaries(self, iteration, num_episodes,
                                  average_reward, average_steps_per_second):
    """Save statistics as tensorboard summaries."""
    if self._summary_writer is None:
      return

    if self._sess is None:
      with self._summary_writer.as_default():
        tf.summary.scalar('Train/NumEpisodes', num_episodes, step=iteration)
        tf.summary.scalar('Train/AverageReturns', average_reward,
                          step=iteration)
        tf.summary.scalar('Train/AverageStepsPerSecond',
                          average_steps_per_second, step=iteration)
      self._summary_writer.flush()
    else:
      summary = tf.compat.v1.Summary(value=[
          tf.compat.v1.Summary.Value(
              tag='Train/NumEpisodes', simple_value=num_episodes),
          tf.compat.v1.Summary.Value(
              tag='Train/AverageReturns', simple_value=average_reward),
          tf.compat.v1.Summary.Value(
              tag='Train/AverageStepsPerSecond',
              simple_value=average_steps_per_second),
      ])
      self._summary_writer.add_summary(summary, iteration)
=======
# coding=utf-8
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module defining classes and helper methods for general agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import heapq
from shutil import copy

from absl import logging

from dopamine.agents.dqn import dqn_agent
from dopamine.agents.implicit_quantile import implicit_quantile_agent
from dopamine.agents.rainbow import rainbow_agent
from dopamine.discrete_domains import atari_lib
from dopamine.discrete_domains import checkpointer
from dopamine.discrete_domains import iteration_statistics
from dopamine.discrete_domains import logger
from dopamine.jax.agents.dqn import dqn_agent as jax_dqn_agent
from dopamine.jax.agents.full_rainbow import full_rainbow_agent
from dopamine.jax.agents.implicit_quantile import implicit_quantile_agent as jax_implicit_quantile_agent
from dopamine.jax.agents.quantile import quantile_agent as jax_quantile_agent
from dopamine.jax.agents.rainbow import rainbow_agent as jax_rainbow_agent
from dopamine.metrics import collector_dispatcher
from dopamine.metrics import statistics_instance
import gin.tf
import numpy as np
import tensorflow as tf


def load_gin_configs(gin_files, gin_bindings):
  """Loads gin configuration files.

  Args:
    gin_files: list, of paths to the gin configuration files for this
      experiment.
    gin_bindings: list, of gin parameter bindings to override the values in
      the config files.
  """
  gin.parse_config_files_and_bindings(gin_files,
                                      bindings=gin_bindings,
                                      skip_unknown=False)


@gin.configurable
def create_agent(sess, environment, agent_name=None, summary_writer=None,
                 debug_mode=False):
  """Creates an agent.

  Args:
    sess: A `tf.compat.v1.Session` object for running associated ops.
    environment: A gym environment (e.g. Atari 2600).
    agent_name: str, name of the agent to create.
    summary_writer: A Tensorflow summary writer to pass to the agent
      for in-agent training statistics in Tensorboard.
    debug_mode: bool, whether to output Tensorboard summaries. If set to true,
      the agent will output in-episode statistics to Tensorboard. Disabled by
      default as this results in slower training.

  Returns:
    agent: An RL agent.

  Raises:
    ValueError: If `agent_name` is not in supported list.
  """
  assert agent_name is not None
  if not debug_mode:
    summary_writer = None
  if agent_name.startswith('dqn'):
    return dqn_agent.DQNAgent(sess, num_actions=environment.action_space.n,
                              summary_writer=summary_writer)
  elif agent_name == 'rainbow':
    return rainbow_agent.RainbowAgent(
        sess, num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  elif agent_name == 'implicit_quantile':
    return implicit_quantile_agent.ImplicitQuantileAgent(
        sess, num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  elif agent_name == 'jax_dqn':
    return jax_dqn_agent.JaxDQNAgent(num_actions=environment.action_space.n,
                                     summary_writer=summary_writer)
  elif agent_name == 'jax_quantile':
    return jax_quantile_agent.JaxQuantileAgent(
        num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  elif agent_name == 'jax_rainbow':
    return jax_rainbow_agent.JaxRainbowAgent(
        num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  elif agent_name == 'full_rainbow':
    return full_rainbow_agent.JaxFullRainbowAgent(
        num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  elif agent_name == 'jax_implicit_quantile':
    return jax_implicit_quantile_agent.JaxImplicitQuantileAgent(
        num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  else:
    raise ValueError('Unknown agent: {}'.format(agent_name))


@gin.configurable
def create_runner(base_dir, schedule='continuous_train_and_eval'):
  """Creates an experiment Runner.

  Args:
    base_dir: str, base directory for hosting all subdirectories.
    schedule: string, which type of Runner to use.

  Returns:
    runner: A `Runner` like object.

  Raises:
    ValueError: When an unknown schedule is encountered.
  """
  assert base_dir is not None
  # Continuously runs training and evaluation until max num_iterations is hit.
  if schedule == 'continuous_train_and_eval':
    return Runner(base_dir, create_agent)
  # Continuously runs training until max num_iterations is hit.
  elif schedule == 'continuous_train':
    return TrainRunner(base_dir, create_agent)
  else:
    raise ValueError('Unknown schedule: {}'.format(schedule))


@gin.configurable
class Runner(object):
  """Object that handles running Dopamine experiments.

  Here we use the term 'experiment' to mean simulating interactions between the
  agent and the environment and reporting some statistics pertaining to these
  interactions.

  A simple scenario to train a DQN agent is as follows:

  ```python
  import dopamine.discrete_domains.atari_lib
  base_dir = '/tmp/simple_example'
  def create_agent(sess, environment):
    return dqn_agent.DQNAgent(sess, num_actions=environment.action_space.n)
  runner = Runner(base_dir, create_agent, atari_lib.create_atari_environment)
  runner.run()
  ```
  """

  def __init__(self,
               base_dir,
               create_agent_fn,
               only_eval=False,
               eval_dir='tent',
               create_environment_fn=atari_lib.create_atari_environment,
               checkpoint_file_prefix='ckpt',
               logging_file_prefix='log',
               log_every_n=1,
               num_iterations=200,
               training_steps=250000,
               evaluation_steps=125000,
               max_steps_per_episode=27000,
               clip_rewards=True,
               use_legacy_logger=True,
               fine_grained_print_to_console=True,
               seed=0,
               eval_episode=200,
               max_q_discount=0.):
    """Initialize the Runner object in charge of running a full experiment.

    Args:
      base_dir: str, the base directory to host all required sub-directories.
      create_agent_fn: A function that takes as args a Tensorflow session and an
        environment, and returns an agent.
      create_environment_fn: A function which receives a problem name and
        creates a Gym environment for that problem (e.g. an Atari 2600 game).
      checkpoint_file_prefix: str, the prefix to use for checkpoint files.
      logging_file_prefix: str, prefix to use for the log files.
      log_every_n: int, the frequency for writing logs.
      num_iterations: int, the iteration number threshold (must be greater than
        start_iteration).
      training_steps: int, the number of training steps to perform.
      evaluation_steps: int, the number of evaluation steps to perform.
      max_steps_per_episode: int, maximum number of steps after which an episode
        terminates.
      clip_rewards: bool, whether to clip rewards in [-1, 1].
      use_legacy_logger: bool, whether to use the legacy Logger. This will be
        deprecated soon, replaced with the new CollectorDispatcher setup.
      fine_grained_print_to_console: bool, whether to print fine-grained
        progress to console (useful for debugging).

    This constructor will take the following actions:
    - Initialize an environment.
    - Initialize a `tf.compat.v1.Session`.
    - Initialize a logger.
    - Initialize an agent.
    - Reload from the latest checkpoint, if available, and initialize the
      Checkpointer object.
    """
    assert base_dir is not None

    self.only_eval = only_eval
    self._eval_dir = eval_dir
    self._legacy_logger_enabled = use_legacy_logger
    self._fine_grained_print_to_console_enabled = fine_grained_print_to_console
    self._logging_file_prefix = logging_file_prefix
    self._log_every_n = log_every_n
    self._num_iterations = num_iterations
    self._training_steps = training_steps
    self._evaluation_steps = evaluation_steps
    self._max_steps_per_episode = max_steps_per_episode
    self._base_dir = base_dir
    self._clip_rewards = clip_rewards
    self._seed = seed
    self._eval_episode = eval_episode
    self.record_test_reward_step = 0
    self.record_tta_reward_step = 0
    self.max_q_discount = max_q_discount
    self._create_directories()

    self._environment, self._originEnv = create_environment_fn()
    # The agent is now in charge of setting up the session.
    self._sess = None
    # We're using a bit of a hack in that we pass in _base_dir instead of an
    # actually SummaryWriter. This is because the agent is now in charge of the
    # session, but needs to create the SummaryWriter before creating the ops,
    # and in order to do so, it requires the base directory.
    # create eval tensorflow
    if self.only_eval:
      summary_writer = self._base_dir + "/" + self._eval_dir
    else:
      summary_writer = self._base_dir

    self._agent = create_agent_fn(self._sess, self._environment,
                                  summary_writer=summary_writer)
    if hasattr(self._agent, '_sess'):
      self._sess = self._agent._sess
    self._summary_writer = self._agent.summary_writer

    self._initialize_checkpointer_and_maybe_resume(checkpoint_file_prefix)

    # Create a collector dispatcher for metrics reporting.
    self._collector_dispatcher = collector_dispatcher.CollectorDispatcher(
        self._base_dir)
    set_collector_dispatcher_fn = getattr(
        self._agent, 'set_collector_dispatcher', None)
    if callable(set_collector_dispatcher_fn):
      set_collector_dispatcher_fn(self._collector_dispatcher)

  @property
  def _use_legacy_logger(self):
    if not hasattr(self, '_legacy_logger_enabled'):
      return True
    return self._legacy_logger_enabled

  @property
  def _has_collector_dispatcher(self):
    if not hasattr(self, '_collector_dispatcher'):
      return False
    return True

  @property
  def _fine_grained_print_to_console(self):
    if not hasattr(self, '_fine_grained_print_to_console_enabled'):
      return True
    return self._fine_grained_print_to_console_enabled

  def _create_directories(self):
    """Create necessary sub-directories."""
    self._checkpoint_dir = os.path.join(self._base_dir, 'checkpoints')
    if self._use_legacy_logger:
      logging.warning(
          'DEPRECATION WARNING: Logger is being deprecated. '
          'Please switch to CollectorDispatcher!')
      self._logger = logger.Logger(os.path.join(self._base_dir, 'logs'))
    
    if self.only_eval:
      eval_path = self._base_dir + "/" + self._eval_dir
      if not os.path.exists(eval_path):
        os.makedirs(eval_path)

  def _initialize_checkpointer_and_maybe_resume(self, checkpoint_file_prefix):
    """Reloads the latest checkpoint if it exists.

    This method will first create a `Checkpointer` object and then call
    `checkpointer.get_latest_checkpoint_number` to determine if there is a valid
    checkpoint in self._checkpoint_dir, and what the largest file number is.
    If a valid checkpoint file is found, it will load the bundled data from this
    file and will pass it to the agent for it to reload its data.
    If the agent is able to successfully unbundle, this method will verify that
    the unbundled data contains the keys,'logs' and 'current_iteration'. It will
    then load the `Logger`'s data from the bundle, and will return the iteration
    number keyed by 'current_iteration' as one of the return values (along with
    the `Checkpointer` object).

    Args:
      checkpoint_file_prefix: str, the checkpoint file prefix.

    Returns:
      start_iteration: int, the iteration number to start the experiment from.
      experiment_checkpointer: `Checkpointer` object for the experiment.
    """
    self._checkpointer = checkpointer.Checkpointer(self._checkpoint_dir,
                                                   checkpoint_file_prefix)
    self._start_iteration = 1
    # Check if checkpoint exists. Note that the existence of checkpoint 0 means
    # that we have finished iteration 0 (so we will start from iteration 1).
    latest_checkpoint_version = checkpointer.get_latest_checkpoint_number(
        self._checkpoint_dir)
    # latest_checkpoint_version = 210
    if latest_checkpoint_version >= 0:
      experiment_data = self._checkpointer.load_checkpoint(
          latest_checkpoint_version)
      if self._agent.unbundle(
          self._checkpoint_dir, latest_checkpoint_version, experiment_data):
        if experiment_data is not None:
          assert 'logs' in experiment_data
          assert 'current_iteration' in experiment_data
          if self._use_legacy_logger:
            self._logger.data = experiment_data['logs']
          self._start_iteration = experiment_data['current_iteration'] + 1
        logging.info('Reloaded checkpoint and will start from iteration %d',
                     self._start_iteration)

  def _initialize_episode(self):
    """Initialization for a new episode.

    Returns:
      action: int, the initial action chosen by the agent.
    """
    initial_observation = self._environment.reset()
    # print("initial_observation'shape: ", initial_observation.shape)
    # exit()
    return self._agent.begin_episode(initial_observation)
  
  def _initialize_reference_episode(self):
    """Initialization for a new episode.

    Returns:
      action: int, the initial action chosen by the agent.
    """
    initial_observation = self._environment.reset()
    # print("initial_observation'shape: ", initial_observation.shape)
    # exit()
    return self._agent.begin_reference_episode(initial_observation)
  
  def _initialize_single_step_episode(self, i):
    """Initialization for a new episode.

    Returns:
      action: int, the initial action chosen by the agent.
    """
    initial_observation = self._environment.reset()
    action, max_q_value = self._agent.begin_single_step_episode(initial_observation, i)
    return initial_observation, action, max_q_value
  
  def _initialize_single_step_episode_by_entropy(self, i):
    """Initialization for a new episode.

    Returns:
      action: int, the initial action chosen by the agent.
    """
    initial_observation = self._environment.reset()
    action, max_q_value, entropy = self._agent.begin_single_step_episode_by_entropy(initial_observation, i)
    return initial_observation, action, max_q_value, entropy
  
  def _adapted_initialize_episode(self):
    """Initialization for a new episode.

    Returns:
      action: int, the initial action chosen by the agent.
    """
    initial_observation = self._environment.reset()

    # print("adapted initial_observation: ", initial_observation)
    # print("adapted initial_observation'shape: ", initial_observation.shape)
    # exit()

    # for (w_online_layer, w_tta_layer) in zip(self._agent.test_online_convnet.layers, self._agent.online_convnet.layers):
    #   # Assign weights from online to target network.
    #   print(((w_online_layer.get_weights()))[0].all() == ((w_tta_layer.get_weights())[0]).all())

    return self._agent.adapted_begin_episode(initial_observation)

  def _adapted_initialize_episode_multi_step(self, i):
    """Initialization for a new episode.

    Returns:
      action: int, the initial action chosen by the agent.
    """
    initial_observation = self._environment.reset()

    action, max_q_value = self._agent.adapted_begin_episode_multi_step(initial_observation, i)

    return initial_observation, action, max_q_value
  
  def _adapted_initialize_episode_with_bellman(self):
    """Initialization for a new episode.

    Returns:
      action: int, the initial action chosen by the agent.
    """
    initial_observation = self._environment.reset()

    return initial_observation, self._agent.adapted_begin_episode(initial_observation)

  def _run_one_step(self, action):
    """Executes a single step in the environment.

    Args:
      action: int, the action to perform in the environment.

    Returns:
      The observation, reward, and is_terminal values returned from the
        environment.
    """
    observation, reward, is_terminal, info = self._environment.step(action)
    # print("info: ", info)
    return observation, reward, is_terminal

  def _run_one_step_multi_episode(self, action):
    """Executes a single step in the environment.

    Args:
      action: int, the action to perform in the environment.

    Returns:
      The observation, reward, and is_terminal values returned from the
        environment.
    """
    observation, reward, is_terminal, info = self._environment.step(action)
    return observation, reward, is_terminal, info

  def _end_episode(self, reward, terminal=True):
    """Finalizes an episode run.

    Args:
      reward: float, the last reward from the environment.
      terminal: bool, whether the last state-action led to a terminal state.
    """
    if isinstance(self._agent, jax_dqn_agent.JaxDQNAgent):
      self._agent.end_episode(reward, terminal)
    else:
      # TODO(joshgreaves): Add terminal signal to TF dopamine agents
      self._agent.end_episode(reward)

  def _run_reference_one_episode(self):
    """Executes a full trajectory of the agent interacting with the environment.

    Returns:
      The number of steps taken and the total reward.
    """
    step_number = 0
    total_reward = 0.

    # TODO: adapted in test-time
    action = self._initialize_reference_episode()
    is_terminal = False

    # Keep interacting until we reach a terminal state.
    while True:
      observation, reward, is_terminal = self._run_one_step(action)

      total_reward += reward
      step_number += 1

      if self._clip_rewards:
        # Perform reward clipping.
        reward = np.clip(reward, -1, 1)

      if (self._environment.game_over or
          step_number == self._max_steps_per_episode):
        # Stop the run loop once we reach the true end of episode.
        break
      elif is_terminal:
        # If we lose a life but the episode is not over, signal an artificial
        # end of episode to the agent.
        self._end_episode(reward, is_terminal)
        # TODO: adapted in test-time
        action = self._agent.begin_reference_episode(observation)
      else:
        # TODO: modify to test-time adaptation
        action = self._agent.reference_step(reward, observation)

    self._end_episode(reward, is_terminal)

    return step_number, total_reward

  #######################################################################################
  # 2023/11/7: update with k-step state-action
  #######################################################################################
  def _run_adapted_single_step_one_episode(self):
    """Executes a full trajectory of the agent interacting with the environment.

    Returns:
      The number of steps taken and the total reward.
    """
    step_number = 0
    total_reward = 0.

    # TODO: adapted in test-time
    episode_num = 10
    step_times = 31

    init_obs, action, max_q_value = self._initialize_single_step_episode(0)
    is_terminal = False

    observation_list = [[init_obs] for j in range(episode_num)]
    reward_list = [[0] for j in range(episode_num)]
    is_terminal_list = [[is_terminal] for j in range(episode_num)]
    action_list = [[action] for j in range(episode_num)]
    max_q_value_list = [[max_q_value] for j in range(episode_num)]
    # state_list = [[self._agent.adapted_state] for j in range(10)]
    multi_step_state = getattr(self._agent, "multi_step_state_%d" % 0) 
    for i in range(1, episode_num):
      setattr(self._agent, "multi_step_state_%d" % i, multi_step_state)
    # TODO: 这里可能没有设置其他举证的初始变量
    state_list = [[getattr(self._agent, "multi_step_state_%d" % j)] for j in range(episode_num)]

    max_q_value_action = 0
    index_of_max_q = 0
    # Keep interacting until we reach a terminal state.
    while True:
      for i in range(episode_num):
        # 这里要重新对所有 multi_step_state 进行赋值
        observation, reward, is_terminal = self._run_one_step(action_list[i][-1])
        observation_list[i].append(observation)
        is_terminal_list[i].append(is_terminal)
        reward_list[i].append(reward)

        if step_number == 0:
          total_reward += reward

        if self._clip_rewards:
          # Perform reward clipping.
          reward = np.clip(reward, -1, 1)
        
      step_number += 1

      if (self._environment.game_over or
          step_number == self._max_steps_per_episode):
        # Stop the run loop once we reach the true end of episode.
        break
      elif is_terminal:
        # If we lose a life but the episode is not over, signal an artificial
        # end of episode to the agent.
        self._end_episode(reward, is_terminal)
        # TODO: adapted in test-time
        for i in range(len(observation_list)):
          action, max_q_value = self._agent.begin_single_step_episode(observation_list[i][-1], i)
          action_list[i].append(action)
          max_q_value_list[i].append(max_q_value)
          state_list[i].append(getattr(self._agent, "multi_step_state_%d" % i))

        max_q_list = []
        for i in range(len(max_q_value_list)):
          max_q_list.append(max_q_value_list[i][-1])

        index_of_max_q = max_q_list.index(max(max_q_list))

        # compute the total reward
        for r in reward_list[index_of_max_q][1:]:
          total_reward += r

        # reset the temp variables
        for i in range(episode_num):
          observation_list[i] = []
          state_list[i] = []
          reward_list[i] = []
          is_terminal_list[i] = []
          action_list[i] = []
          max_q_value_list[i] = []

        multi_step_state = getattr(self._agent, "multi_step_state_%d" % index_of_max_q) 
        for i in range(0, episode_num):
          setattr(self._agent, "multi_step_state_%d" % i, multi_step_state)

      else:
        # TODO: modify to support multi-step update test-time adaptation
        for i in range(len(observation_list)):
          action, max_q_value = self._agent.adapted_single_step(reward_list[i][-1], observation_list[i][-1], i)
          action_list[i].append(action)
          max_q_value_list[i].append(max_q_value)
          state_list[i].append(getattr(self._agent, "multi_step_state_%d" % i))

        if step_number % step_times == 0 and step_number != 0:
          max_q_list = []
          for i in range(len(max_q_value_list)):
            max_q_list.append(max_q_value_list[i][-1])

          index_of_max_q = max_q_list.index(max(max_q_list))

          # decide the next action
          max_q_value_action = action_list[index_of_max_q][-1]
          max_q_value = max_q_value_list[index_of_max_q][-1]
          max_q_is_terminal = is_terminal_list[index_of_max_q][-1]
          max_q_reward = reward_list[index_of_max_q][-1]
          max_q_state = state_list[index_of_max_q][-1]
          max_q_observation = observation_list[index_of_max_q][-1]

          # compute the total reward
          for r in reward_list[index_of_max_q][1:]:
            total_reward += r

          # use the max Q value list to update the policy
          state_update_list = state_list[index_of_max_q]
          self._agent._update_model_with_max_q_value_step(state_update_list)

          # reset the temp variables
          for i in range(episode_num):
            observation_list[i] = [max_q_observation]
            state_list[i] = [max_q_state]
            reward_list[i] = [max_q_reward]
            is_terminal_list[i] = [max_q_is_terminal]
            action_list[i] = [max_q_value_action]
            max_q_value_list[i] = [max_q_value]
          
          multi_step_state = getattr(self._agent, "multi_step_state_%d" % index_of_max_q) 
          for i in range(0, episode_num):
            setattr(self._agent, "multi_step_state_%d" % i, multi_step_state)

    self._end_episode(reward, is_terminal)

    return step_number, total_reward
  
  #######################################################################################
  # 2023/11/9: update with weight q value episode
  #######################################################################################
  def _run_adapted_one_episode_with_discount_max_q_zero_return(self):
    """Executes a full trajectory of the agent interacting with the environment.

    Returns:
      The number of steps taken and the total reward.
    """
    step_number = 0
    total_reward = 0.
    q_value_discount = self.max_q_discount

    state_list = []
    q_value_list = []
    reward_list = []
    total_max_q_value = 0.
    init_obs, action, max_q_value = self._initialize_single_step_episode(0)
    total_max_q_value += max_q_value * (q_value_discount ** step_number)
    is_terminal = False
    state_list.append(getattr(self._agent, "multi_step_state_%d" % 0))
    q_value_list.append(max_q_value)

    # Keep interacting until we reach a terminal state.
    while True:
      observation, reward, is_terminal = self._run_one_step(action)

      total_reward += reward
      reward_list.append(reward)

      if self._clip_rewards:
        # Perform reward clipping.
        reward = np.clip(reward, -1, 1)
        
      step_number += 1

      if (self._environment.game_over or
          step_number == self._max_steps_per_episode):
        # Stop the run loop once we reach the true end of episode.
        break
      elif is_terminal:
        # If we lose a life but the episode is not over, signal an artificial
        # end of episode to the agent.
        self._end_episode(reward, is_terminal)
        # TODO: adapted in test-time
        action, max_q_value = self._agent.begin_single_step_episode(observation, 0)
        state_list.append(getattr(self._agent, "multi_step_state_%d" % 0))
        q_value_list.append(max_q_value)
        total_max_q_value += max_q_value * (q_value_discount ** step_number)

      else:
        # TODO: modify to support multi-step update test-time adaptation
        action, max_q_value = self._agent.adapted_single_step(reward, observation, 0)
        state_list.append(getattr(self._agent, "multi_step_state_%d" % 0))
        q_value_list.append(max_q_value)
        total_max_q_value += max_q_value * (q_value_discount ** step_number)

    self._end_episode(reward, is_terminal)

    if self.max_q_discount > 0.:
      return step_number, total_reward, total_max_q_value, state_list, q_value_list, reward_list
    else:
      return step_number, total_reward, max_q_value, state_list, q_value_list, reward_list

  #######################################################################################
  # 2023/11/9: update with weight q value episode
  #######################################################################################
  def _run_adapted_one_episode_with_discount_max_q(self):
    """Executes a full trajectory of the agent interacting with the environment.

    Returns:
      The number of steps taken and the total reward.
    """
    step_number = 0
    total_reward = 0.
    q_value_discount = self.max_q_discount

    state_list = []
    total_max_q_value = 0.
    init_obs, action, max_q_value = self._initialize_single_step_episode(0)
    total_max_q_value += max_q_value * (q_value_discount ** step_number)
    is_terminal = False
    state_list.append(getattr(self._agent, "multi_step_state_%d" % 0))

    # Keep interacting until we reach a terminal state.
    while True:
      observation, reward, is_terminal = self._run_one_step(action)

      total_reward += reward

      if self._clip_rewards:
        # Perform reward clipping.
        reward = np.clip(reward, -1, 1)
        
      step_number += 1

      if (self._environment.game_over or
          step_number == self._max_steps_per_episode):
        # Stop the run loop once we reach the true end of episode.
        break
      elif is_terminal:
        # If we lose a life but the episode is not over, signal an artificial
        # end of episode to the agent.
        self._end_episode(reward, is_terminal)
        # TODO: adapted in test-time
        action, max_q_value = self._agent.begin_single_step_episode(observation, 0)

      else:
        # TODO: modify to support multi-step update test-time adaptation
        action, max_q_value = self._agent.adapted_single_step(reward, observation, 0)
        state_list.append(getattr(self._agent, "multi_step_state_%d" % 0))
        total_max_q_value += max_q_value * (q_value_discount ** step_number)

    self._end_episode(reward, is_terminal)

    if self.max_q_discount > 0.:
      return step_number, total_reward, total_max_q_value, state_list
    else:
      return step_number, total_reward, max_q_value, state_list
  
  #######################################################################################
  # 2023/11/9: update with weight q value episode
  #######################################################################################
  def _run_adapted_one_episode_with_discount_max_q_and_min_entropy(self):
    """Executes a full trajectory of the agent interacting with the environment.

    Returns:
      The number of steps taken and the total reward.
    """
    step_number = 0
    total_reward = 0.
    q_value_discount = self.max_q_discount
    entropy_value_discount = 1

    state_list = []
    total_max_q_value = 0.
    total_entropy_value = 0.
    init_obs, action, max_q_value, entropy = self._initialize_single_step_episode_by_entropy(0)
    total_max_q_value += max_q_value * (q_value_discount ** step_number)
    total_entropy_value += entropy * (entropy_value_discount ** step_number)
    is_terminal = False
    state_list.append(getattr(self._agent, "multi_step_state_%d" % 0))

    # Keep interacting until we reach a terminal state.
    while True:
      observation, reward, is_terminal = self._run_one_step(action)

      total_reward += reward

      if self._clip_rewards:
        # Perform reward clipping.
        reward = np.clip(reward, -1, 1)
        
      step_number += 1

      if (self._environment.game_over or
          step_number == self._max_steps_per_episode):
        # Stop the run loop once we reach the true end of episode.
        break
      elif is_terminal:
        # If we lose a life but the episode is not over, signal an artificial
        # end of episode to the agent.
        self._end_episode(reward, is_terminal)
        # TODO: adapted in test-time
        action, max_q_value, entropy = self._agent.begin_single_step_episode_by_entropy(observation, 0)
        state_list.append(getattr(self._agent, "multi_step_state_%d" % 0))
        total_max_q_value += max_q_value * (q_value_discount ** step_number)
        total_entropy_value += entropy * (entropy_value_discount ** step_number)
      else:
        # TODO: modify to support multi-step update test-time adaptation
        action, max_q_value, entropy = self._agent.adapted_single_step_by_entropy(reward, observation, 0)
        state_list.append(getattr(self._agent, "multi_step_state_%d" % 0))
        total_max_q_value += max_q_value * (q_value_discount ** step_number)
        total_entropy_value += entropy * (entropy_value_discount ** step_number)

    self._end_episode(reward, is_terminal)

    if self.max_q_discount > 0.:
      return step_number, total_reward, total_max_q_value, total_entropy_value, state_list
    else:
      return step_number, total_reward, max_q_value, total_entropy_value, state_list

  #######################################################################################
  # 2023/11/21: update with weighted entropy
  #######################################################################################
  def _run_adapted_one_episode_with_discount_entropy(self):
    """Executes a full trajectory of the agent interacting with the environment.

    Returns:
      The number of steps taken and the total reward.
    """
    step_number = 0
    total_reward = 0.
    entropy_value_discount = 1

    state_list = []
    total_entropy_value = 0.
    init_obs, action, max_q_value, entropy = self._initialize_single_step_episode_by_entropy(0)
    total_entropy_value += entropy * (entropy_value_discount ** step_number)
    is_terminal = False
    state_list.append(getattr(self._agent, "multi_step_state_%d" % 0))

    # Keep interacting until we reach a terminal state.
    while True:
      observation, reward, is_terminal = self._run_one_step(action)

      total_reward += reward

      if self._clip_rewards:
        # Perform reward clipping.
        reward = np.clip(reward, -1, 1)
        
      step_number += 1

      if (self._environment.game_over or
          step_number == self._max_steps_per_episode):
        # Stop the run loop once we reach the true end of episode.
        break
      elif is_terminal:
        # If we lose a life but the episode is not over, signal an artificial
        # end of episode to the agent.
        self._end_episode(reward, is_terminal)
        # TODO: adapted in test-time
        action, max_q_value, entropy = self._agent.begin_single_step_episode_by_entropy(observation, 0)

      else:
        # TODO: modify to support multi-step update test-time adaptation
        action, max_q_value, entropy = self._agent.adapted_single_step_by_entropy(reward, observation, 0)
        state_list.append(getattr(self._agent, "multi_step_state_%d" % 0))
        total_entropy_value += entropy * (entropy_value_discount ** step_number)

    self._end_episode(reward, is_terminal)

    return step_number, total_reward, total_entropy_value, state_list

  #######################################################################################
  # 2023/11/9: update with max q value episode
  #######################################################################################
  def _run_adapted_one_episode_with_max_q(self):
    """Executes a full trajectory of the agent interacting with the environment.

    Returns:
      The number of steps taken and the total reward.
    """
    step_number = 0
    total_reward = 0.

    state_list = []

    init_obs, action, max_q_value = self._initialize_single_step_episode(0)
    is_terminal = False
    state_list.append(getattr(self._agent, "multi_step_state_%d" % 0))

    # Keep interacting until we reach a terminal state.
    while True:
      observation, reward, is_terminal = self._run_one_step(action)

      total_reward += reward

      if self._clip_rewards:
        # Perform reward clipping.
        reward = np.clip(reward, -1, 1)
        
      step_number += 1

      if (self._environment.game_over or
          step_number == self._max_steps_per_episode):
        # Stop the run loop once we reach the true end of episode.
        break
      elif is_terminal:
        # If we lose a life but the episode is not over, signal an artificial
        # end of episode to the agent.
        self._end_episode(reward, is_terminal)
        # TODO: adapted in test-time
        action, max_q_value = self._agent.begin_single_step_episode(observation, 0)

      else:
        # TODO: modify to support multi-step update test-time adaptation
        action, max_q_value = self._agent.adapted_single_step(reward, observation, 0)
        state_list.append(getattr(self._agent, "multi_step_state_%d" % 0))

    self._end_episode(reward, is_terminal)

    return step_number, total_reward, max_q_value, state_list
  
  def _run_adapted_one_episode_multi_step(self):
    """Executes a full trajectory of the agent interacting with the environment.

    Returns:
      The number of steps taken and the total reward.
    """
    step_number = 0
    total_reward = 0.

    # TODO: adapted in test-time
    episode_num = 10
    step_times = 16

    init_obs, action, max_q_value = self._adapted_initialize_episode_multi_step(0)
    is_terminal = False

    observation_list = [[init_obs] for j in range(episode_num)]
    reward_list = [[0] for j in range(episode_num)]
    is_terminal_list = [[is_terminal] for j in range(episode_num)]
    action_list = [[action] for j in range(episode_num)]
    max_q_value_list = [[max_q_value] for j in range(episode_num)]
    lives_list = [[3] for j in range(episode_num)]
    multi_step_state = getattr(self._agent, "multi_step_state_%d" % 0) 
    for i in range(episode_num):
      setattr(self._agent, "multi_step_state_%d" % i , multi_step_state)
    state_list = [[getattr(self._agent, "multi_step_state_%d" % j)] for j in range(episode_num)]
    screen_buffer_list = [[self._environment.screen_buffer] for j in range(episode_num)]
    
    update_state_list = [multi_step_state]

    index_of_max_q = 0
    # Keep interacting until we reach a terminal state.
    while True:
      for i in range(episode_num):
        # reset the observation back to the environment variable
        self._environment.screen_buffer = screen_buffer_list[i][-1]
        self._environment.game_over = is_terminal_list[i][-1]
        self._environment.lives = lives_list[i][-1]
        observation, reward, is_terminal, info = self._run_one_step_multi_episode(action_list[i][-1])
        lives_list[i].append(info["lives"])
        screen_buffer_list[i].append(self._environment.screen_buffer)
        observation_list[i].append(observation)
        is_terminal_list[i].append(is_terminal)
        reward_list[i].append(reward)

        if self._clip_rewards:
          # Perform reward clipping.
          reward = np.clip(reward, -1, 1)

      if (self._environment.game_over or
          step_number == self._max_steps_per_episode):
        # Stop the run loop once we reach the true end of episode.
        break
      elif is_terminal:
        # If we lose a life but the episode is not over, signal an artificial
        # end of episode to the agent.
        self._end_episode(reward, is_terminal)

        # TODO: adapted in test-time
        for i in range(len(observation_list)):
          action, max_q_value = self._agent.adapted_begin_episode_multi_step(observation_list[i][-1], i)
          action_list[i].append(action)
          max_q_value_list[i].append(max_q_value)
          # state_list[i].append(self._agent.adapted_state)
          state_list[i].append(getattr(self._agent, "multi_step_state_%d" % i))
      else:
        # TODO: modify to support multi-step update test-time adaptation
        for i in range(len(observation_list)):
          action, max_q_value = self._agent.adapted_multi_step(reward_list[i][-1], observation_list[i][-1], i)
          action_list[i].append(action)
          max_q_value_list[i].append(max_q_value)
          state_list[i].append(getattr(self._agent, "multi_step_state_%d" % i))
      
      # get the index of the largest Q value among several episodes
      if step_number % step_times == 0 and step_number != 0:
        max_q_list = []
        for i in range(len(max_q_value_list)):
          max_q_list.append(max_q_value_list[i][-1])
        index_of_max_q = max_q_list.index(max(max_q_list))

        # compute the total reward
        for r in reward_list[index_of_max_q][:-1]:
          total_reward += r

        for state in state_list[index_of_max_q][:-1]:
          update_state_list.append(state)

        # # use the max Q value list to update the policy
        # state_update_list = state_list[index_of_max_q]
        # self._agent._update_model_with_max_q_value_step(state_update_list)

        # reset the temp variables
        for i in range(episode_num):
          observation_list[i] = [observation_list[index_of_max_q][-1]]
          state_list[i] = [state_list[index_of_max_q][-1]]
          reward_list[i] = [reward_list[index_of_max_q][-1]]
          is_terminal_list[i] = [is_terminal_list[index_of_max_q][-1]]
          action_list[i] = [action_list[index_of_max_q][-1]]
          max_q_value_list[i] = [max_q_value_list[index_of_max_q][-1]]
          screen_buffer_list[i] = [screen_buffer_list[index_of_max_q][-1]]
          lives_list[i] = [lives_list[index_of_max_q][-1]]
          
      step_number += 1

    self._end_episode(reward, is_terminal)

    return step_number, total_reward, update_state_list
  
  #######################################################################################
  # 2023/11/22: update with zero return bellman equation
  #######################################################################################
  def _run_adapted_one_episode_zero_return(self, step_count):
    """Executes a full trajectory of the agent interacting with the environment.

    Returns:
      The number of steps taken and the total reward.
    """
    step_number = 0
    total_reward = 0.

    obs, action, max_q_value = self._initialize_single_step_episode(0)
    is_terminal = False
    state = getattr(self._agent, "multi_step_state_%d" % 0)

    # Keep interacting until we reach a terminal state.
    while True:
      obs, reward, is_terminal = self._run_one_step(action)

      total_reward += reward
      if self._summary_writer is not None:
        summary = tf.compat.v1.Summary(value=[
            tf.compat.v1.Summary.Value(
                tag='tta/reward', simple_value=reward)
        ])
        self._summary_writer.add_summary(summary, step_count + step_number)

      if self._clip_rewards:
        # Perform reward clipping.
        reward = np.clip(reward, -1, 1)
        
      step_number += 1

      if (self._environment.game_over or
          step_number == self._max_steps_per_episode):
        # Stop the run loop once we reach the true end of episode.
        break
      elif is_terminal:
        # If we lose a life but the episode is not over, signal an artificial
        # end of episode to the agent.
        self._end_episode(reward, is_terminal)
        # TODO: adapted in test-time
        action, next_max_q_value = self._agent.begin_single_step_episode(obs, 0)
        next_state = getattr(self._agent, "multi_step_state_%d" % 0)

      else:
        # TODO: modify to support multi-step update test-time adaptation
        action, next_max_q_value = self._agent.adapted_single_step(reward, obs, 0)
        next_state = getattr(self._agent, "multi_step_state_%d" % 0)
        # update with zero return 
        self._agent.update_network_with_zero_return(state, max_q_value, next_state, next_max_q_value)

        state = next_state
        max_q_value = next_max_q_value

        if self._summary_writer is not None:
          summary = tf.compat.v1.Summary(value=[
              tf.compat.v1.Summary.Value(
                  tag='tta/max_q_value', simple_value=max_q_value),
          ])
          self._summary_writer.add_summary(summary, step_count + step_number)
        
        # copy tta parameters to the target parameters
        if (step_count + step_number) % 200 == 0:
          self._agent.copy_tta_to_target()

    self._end_episode(reward, is_terminal)

    return step_number, total_reward

  def _run_adapted_one_episode(self):
    """Executes a full trajectory of the agent interacting with the environment.

    Returns:
      The number of steps taken and the total reward.
    """
    step_number = 0
    total_reward = 0.

    # TODO: adapted in test-time
    action = self._adapted_initialize_episode()
    is_terminal = False

    # Keep interacting until we reach a terminal state.
    while True:
      observation, reward, is_terminal = self._run_one_step(action)
      # print(tf.reduce_all(tf.equal(observation, observation2)))

      total_reward += reward
      step_number += 1

      if self._clip_rewards:
        # Perform reward clipping.
        reward = np.clip(reward, -1, 1)

      if (self._environment.game_over or
          step_number == self._max_steps_per_episode):
        # Stop the run loop once we reach the true end of episode.
        break
      elif is_terminal:
        # If we lose a life but the episode is not over, signal an artificial
        # end of episode to the agent.
        self._end_episode(reward, is_terminal)
        # TODO: adapted in test-time
        action = self._agent.adapted_begin_episode(observation)
      else:
        # TODO: modify to test-time adaptation
        action = self._agent.adapted_step(reward, observation)

      # record reward tensorboard
      if self._summary_writer is not None:
        self._save_reward_summaries(self.record_tta_reward_step, reward, "tta")
      self.record_tta_reward_step += 1

    self._end_episode(reward, is_terminal)

    return step_number, total_reward

  def _run_adapted_one_episode_with_bellman(self):
    """Executes a full trajectory of the agent interacting with the environment.

    Returns:
      The number of steps taken and the total reward.
    """
    step_number = 0
    total_reward = 0.

    # TODO: adapted in test-time
    observation, action = self._adapted_initialize_episode_with_bellman()
    is_terminal = False

    # Keep interacting until we reach a terminal state.
    while True:
      next_observation, reward, is_terminal = self._run_one_step(action)

      # update with the bellman equation
      self._agent.update_with_bellman(observation, action, reward, next_observation)

      observation = next_observation
      total_reward += reward
      step_number += 1

      if self._clip_rewards:
        # Perform reward clipping.
        reward = np.clip(reward, -1, 1)

      if (self._environment.game_over or
          step_number == self._max_steps_per_episode):
        # Stop the run loop once we reach the true end of episode.
        break
      elif is_terminal:
        # If we lose a life but the episode is not over, signal an artificial
        # end of episode to the agent.
        self._end_episode(reward, is_terminal)
        # TODO: adapted in test-time
        action = self._agent.begin_episode(observation)
      else:
        # TODO: modify to test-time adaptation
        action = self._agent.step(reward, observation)

      # record reward tensorboard
      if self._summary_writer is not None:
        self._save_reward_summaries(self.record_tta_reward_step, reward, "tta")
      self.record_tta_reward_step += 1

    self._end_episode(reward, is_terminal)

    return step_number, total_reward
  
  def _run_one_episode(self):
    """Executes a full trajectory of the agent interacting with the environment.

    Returns:
      The number of steps taken and the total reward.
    """
    step_number = 0
    total_reward = 0.

    # TODO: adapted in test-time
    action = self._initialize_episode()
    is_terminal = False

    # Keep interacting until we reach a terminal state.
    while True:
      observation, reward, is_terminal = self._run_one_step(action)

      total_reward += reward
      step_number += 1

      if self._clip_rewards:
        # Perform reward clipping.
        reward = np.clip(reward, -1, 1)

      if (self._environment.game_over or
          step_number == self._max_steps_per_episode):
        # Stop the run loop once we reach the true end of episode.
        break
      elif is_terminal:
        # If we lose a life but the episode is not over, signal an artificial
        # end of episode to the agent.
        self._end_episode(reward, is_terminal)
        # TODO: adapted in test-time
        action = self._agent.begin_episode(observation)
      else:
        # TODO: modify to test-time adaptation
        action = self._agent.step(reward, observation)

        # sys.stdout.write('action: {} '.format(action))
        # sys.stdout.flush()

      # record reward tensorboard
      if self._summary_writer is not None:
        self._save_reward_summaries(self.record_test_reward_step, reward, "eval")
      self.record_test_reward_step += 1

    self._end_episode(reward, is_terminal)

    return step_number, total_reward
  
  # def _run_eval(self, episode_num, statistics):


  def _run_one_phase(self, min_steps, statistics, run_mode_str):
    """Runs the agent/environment loop until a desired number of steps.

    We follow the Machado et al., 2017 convention of running full episodes,
    and terminating once we've run a minimum number of steps.

    Args:
      min_steps: int, minimum number of steps to generate in this phase.
      statistics: `IterationStatistics` object which records the experimental
        results.
      run_mode_str: str, describes the run mode for this agent.

    Returns:
      Tuple containing the number of steps taken in this phase (int), the sum of
        returns (float), and the number of episodes performed (int).
    """
    step_count = 0
    num_episodes = 0
    sum_returns = 0.

    ########################################################################################
    # 2023/11/9 use 200 episode to evaluate the performance of the policy
    ########################################################################################
    # while step_count < min_steps:
    while num_episodes < self._eval_episode:
      episode_length, episode_return = self._run_one_episode()
      statistics.append({
          '{}_episode_lengths'.format(run_mode_str): episode_length,
          '{}_episode_returns'.format(run_mode_str): episode_return
      })
      step_count += episode_length
      sum_returns += episode_return
      num_episodes += 1
      if self._fine_grained_print_to_console:
        # We use sys.stdout.write instead of logging so as to flush frequently
        # without generating a line break.
        sys.stdout.write('Steps executed: {} '.format(step_count) +
                          'Episode length: {} '.format(episode_length) +
                          'Return: {}\r'.format(episode_return))
        sys.stdout.flush()

    return step_count, sum_returns, num_episodes

  def select_top_k_indices(self, arr, k):
    # 选择一个 list 中的 topk 个元素的索引
    return heapq.nlargest(k, range(len(arr)), key=arr.__getitem__)

  def _run_adapted_one_phase(self, min_steps, statistics, run_mode_str):
    """Runs the agent/environment loop until a desired number of steps.

    We follow the Machado et al., 2017 convention of running full episodes,
    and terminating once we've run a minimum number of steps.

    Args:
      min_steps: int, minimum number of steps to generate in this phase.
      statistics: `IterationStatistics` object which records the experimental
        results.
      run_mode_str: str, describes the run mode for this agent.

    Returns:
      Tuple containing the number of steps taken in this phase (int), the sum of
        returns (float), and the number of episodes performed (int).
    """
    step_count = 0
    num_episodes = 0
    sum_returns = 0.
    update_step_count = 0
    update_episode_count = 0
    alpha = 1
    beta = 1
    ########################################################################################
    # 2023/11/9 use 200 episode to evaluate the performance of the policy
    ########################################################################################
    max_q_list = []
    discount_total_q_list = []
    total_entropy_list = []
    episode_state_list = []
    episode_q_value_list = []

    # while step_count < min_steps:
    while num_episodes < self._eval_episode:
      # ##################################################################################################################
      # # 2023/11/28: 先从 10 个 episode 中选择出 topk 大 Q 值的 episode, 再从 topk 个 episode 中选择出最小 entropy 的 episode
      # ##################################################################################################################
      episode_length, episode_return, total_q_value, total_entropy, state_list = self._run_adapted_one_episode_with_discount_max_q_and_min_entropy()
      discount_total_q_list.append(total_q_value)
      total_entropy_list.append(total_entropy)
      episode_state_list.append(state_list)

      if num_episodes % 10 == 9 and num_episodes != 0:
        topk_index_list = self.select_top_k_indices(discount_total_q_list, 4)
        top_k_q_list = [total_entropy_list[i] for i in topk_index_list]
        index_of_min_evaluate = total_entropy_list.index(min(top_k_q_list))
        update_state_list = episode_state_list[index_of_min_evaluate]
        ############################################################################################
        # 2023/11/26 update with batch size of 32
        ############################################################################################
        # self._agent._update_model_with_episode(update_state_list)

        ############################################################################################
        # 2023/11/26 update with batch size of 32
        ############################################################################################
        self._agent._update_model_with_kl_loss(update_state_list)
        # clear trajectory buffer
        discount_total_q_list = []
        total_entropy_list = []
        episode_state_list = []
      ##########################################################################################

      # # ########################################################################################
      # # # 2023/11/23 use the max alpha * Q - beta * entropy to update the Q network
      # # ########################################################################################
      # episode_length, episode_return, total_q_value, total_entropy, state_list = self._run_adapted_one_episode_with_discount_max_q_and_min_entropy()
      # discount_total_q_list.append(total_q_value)
      # total_entropy_list.append(total_entropy)
      # episode_state_list.append(state_list)

      # if num_episodes % 10 == 9 and num_episodes != 0:
      #   evaluate_list = [alpha * discount_total_q_list[i] - beta * total_entropy_list[i] for i in range(len(discount_total_q_list))]
      #   # evaluate_list = [discount_total_q_list[i] / total_entropy_list[i] for i in range(len(discount_total_q_list))]
      #   index_of_max_evaluate = evaluate_list.index(max(evaluate_list))
      #   update_state_list = episode_state_list[index_of_max_evaluate]

      #   ############################################################################################
      #   # 2023/11/26 update with batch size of 32
      #   ############################################################################################
      #   # for i in range(0, len(update_state_list), 32):
      #   #   if i + 32 < len(update_state_list):
      #   #     self._agent._update_model_with_episode(update_state_list[i:i+32])
      #   #   else:
      #   #     self._agent._update_model_with_episode(update_state_list[i:])

      #   self._agent._update_model_with_episode(update_state_list)
      #   # clear trajectory buffer
      #   discount_total_q_list = []
      #   total_entropy_list = []
      #   episode_state_list = []
      # ##########################################################################################

      # ############################################################################################
      # # 2023/11/24 use a batch sample to update the Q Network with the bellman equation
      # ############################################################################################
      # episode_length, episode_return, max_q_value, state_list, q_value_list, reward_list = self._run_adapted_one_episode_with_discount_max_q_zero_return()
      # for i in range(0, len(state_list) - 1, 32):
      #   batch_state = np.squeeze(np.stack(state_list[i:i+32]), axis=1)
      #   next_batch_state = np.squeeze(np.stack(state_list[i+1:i+33]), axis=1)
      #   reward = np.squeeze(np.stack(reward_list[i:i+32]))

      #   self._agent.update_network_with_zero_return(batch_state, next_batch_state, reward)
      #   update_step_count += 1

      # if update_step_count % 200 == 0: 
      #   self._agent.copy_tta_to_target()
      # ############################################################################################


      # episode_length, episode_return = self._run_adapted_one_episode_zero_return(step_count)
      # episode_length, episode_return, total_entropy, state_list = self._run_adapted_one_episode_with_discount_entropy()
      # episode_length, episode_return, update_state_list = self._run_adapted_one_episode_multi_step()
      # episode_length, episode_return = self._run_reference_one_episode()
      # episode_length, episode_return = self._run_adapted_single_step_one_episode()
      # episode_length, episode_return = self._run_adapted_one_episode()
      # if self.max_q_discount > 0.:
      #   episode_length, episode_return, max_q_value, state_list = self._run_adapted_one_episode_with_discount_max_q()
      # else:
      #   episode_length, episode_return, max_q_value, state_list = self._run_adapted_one_episode_with_max_q()

      # ########################################################################################
      # # 2023/11/22 use the max_q_value episode to update the Q network
      # ########################################################################################
      # episode_length, episode_return, max_q_value, state_list, q_value_list, reward_list = self._run_adapted_one_episode_with_discount_max_q_zero_return()

      # max_q_value = 0
      # state_list = []

      # max_q_list.append(max_q_value)
      # # total_entropy_list.append(total_entropy)
      # episode_state_list.append(state_list)
      # episode_q_value_list.append(q_value_list)

      # ####################################################################################################
      # # 2023/11/23 update with zero return and a batch sample
      # ####################################################################################################
      # update_state_list = []
      # update_next_state_list = []
      # for i in range(len(q_value_list) - 1):
      #   if abs(q_value_list[i] - q_value_list[i + 1]) < 50:
      #     update_state_list.append(state_list[i])
      #     update_next_state_list.append(state_list[i + 1])
      #     update_step_count += 1

      # self._agent.update_network_with_zero_return_update_with_batch(update_state_list, update_next_state_list)
      
      # if update_step_count % 100 == 0: 
      #   self._agent.copy_tta_to_target()


      # ####################################################################################################
      # # 2023/11/23 update with bellman equation
      # ####################################################################################################
      # for i in range(len(q_value_list) - 1):
      #   self._agent.update_network_with_zero_return(state_list[i], state_list[i + 1], reward_list[i])
      #   update_step_count += 1

      # if update_step_count % 100 == 0: 
      #   self._agent.copy_tta_to_target()

      # # update with the max q value episode
      # if num_episodes % 10 == 9 and num_episodes != 0:
      #   # index_of_min_entropy = total_entropy_list.index(min(total_entropy_list))
      #   # update_state_list = episode_state_list[index_of_min_entropy]

      #   index_of_max_q = max_q_list.index(max(max_q_list))
      #   update_state_list = episode_state_list[index_of_max_q]
      #   update_q_value_list = episode_q_value_list[index_of_max_q]

      #   ####################################################################################################
      #   # 2023/11/23 update with zero return and a batch sample in the whole episode
      #   ####################################################################################################
      #   state_list = []
      #   next_state_list = []
      #   for i in range(len(update_q_value_list) - 1):
      #     if abs(update_q_value_list[i] - update_q_value_list[i + 1]) < 50:
      #       state_list.append(update_state_list[i])
      #       next_state_list.append(update_state_list[i + 1])
      #   self._agent.update_network_with_zero_return_update_with_batch(state_list, next_state_list)
      #   self._agent.copy_tta_to_target()

      #   # self._agent._update_model_with_episode(update_state_list)
      #   # clear trajectory buffer
      #   max_q_list = [] 
      #   # total_entropy_list = []
      #   episode_state_list = []
      #   episode_q_value_list = []

      statistics.append({
          '{}_adapted_episode_lengths'.format(run_mode_str): episode_length,
          '{}_adapted_episode_returns'.format(run_mode_str): episode_return
      })
      step_count += episode_length
      sum_returns += episode_return
      num_episodes += 1

      if self._fine_grained_print_to_console:
        # We use sys.stdout.write instead of logging so as to flush frequently
        # without generating a line break.
        sys.stdout.write('Adapted Steps executed: {} '.format(step_count) +
                          'Adpated Episode length: {} '.format(episode_length) +
                          'Adpated Return: {}\r'.format(episode_return))
        sys.stdout.flush()

      if self._summary_writer is not None:
        summary = tf.compat.v1.Summary(value=[
            tf.compat.v1.Summary.Value(
                tag='tta/total_entropy', simple_value=total_entropy),
            tf.compat.v1.Summary.Value(
                tag='tta/total_q_value', simple_value=total_q_value),
            tf.compat.v1.Summary.Value(
                tag='tta/return', simple_value=episode_return),
        ])
        self._summary_writer.add_summary(summary, num_episodes)

    return step_count, sum_returns, num_episodes

  def _run_train_phase(self, statistics):
    """Run training phase.

    Args:
      statistics: `IterationStatistics` object which records the experimental
        results. Note - This object is modified by this method.

    Returns:
      num_episodes: int, The number of episodes run in this phase.
      average_reward: float, The average reward generated in this phase.
      average_steps_per_second: float, The average number of steps per second.
    """
    # Perform the training phase, during which the agent learns.
    self._agent.eval_mode = False
    start_time = time.time()
    number_steps, sum_returns, num_episodes = self._run_one_phase(
        self._training_steps, statistics, 'train')
    average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
    statistics.append({'train_average_return': average_return})
    time_delta = time.time() - start_time
    average_steps_per_second = number_steps / time_delta
    statistics.append(
        {'train_average_steps_per_second': average_steps_per_second})
    logging.info('Average undiscounted return per training episode: %.2f',
                 average_return)
    logging.info('Average training steps per second: %.2f',
                 average_steps_per_second)
    return num_episodes, average_return, average_steps_per_second
  
  def _run_adapted_eval_phase(self, statistics):
    """Run evaluation phase.

    Args:
      statistics: `IterationStatistics` object which records the experimental
        results. Note - This object is modified by this method.

    Returns:
      num_episodes: int, The number of episodes run in this phase.
      average_reward: float, The average reward generated in this phase.
    """
    # Perform the evaluation phase -- no learning.
    self._agent.eval_mode = True
    step_count, sum_returns, num_episodes = self._run_adapted_one_phase(
        self._evaluation_steps, statistics, 'tta')
    average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
    average_step_count = step_count / num_episodes if num_episodes > 0 else 0.0
    logging.info('Adapted Average undiscounted return per evaluation episode: %.2f',
                 average_return)
    logging.info('Adapted Average step return per evaluation episode: %.2f',
                 average_step_count)
    statistics.append({'eval_adapted_average_return': average_return})
    statistics.append({'eval_adapted_average_step_count': average_step_count})
    return num_episodes, average_return, average_step_count

  def _run_eval_phase(self, statistics):
    """Run evaluation phase.

    Args:
      statistics: `IterationStatistics` object which records the experimental
        results. Note - This object is modified by this method.

    Returns:
      num_episodes: int, The number of episodes run in this phase.
      average_reward: float, The average reward generated in this phase.
    """
    # Perform the evaluation phase -- no learning.
    self._agent.eval_mode = True
    step_count, sum_returns, num_episodes = self._run_one_phase(
        self._evaluation_steps, statistics, 'eval')
    average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
    average_step_count = step_count / num_episodes if num_episodes > 0 else 0.0
    logging.info('Average undiscounted return per evaluation episode: %.2f',
                 average_return)
    statistics.append({'eval_average_return': average_return})
    statistics.append({'eval_average_step_count': average_step_count})
    return num_episodes, average_return, average_step_count
  
  def _run_one_iteration(self, iteration):
    """Runs one iteration of agent/environment interaction.

    An iteration involves running several episodes until a certain number of
    steps are obtained. The interleaving of train/eval phases implemented here
    are to match the implementation of (Mnih et al., 2015).

    Args:
      iteration: int, current iteration number, used as a global_step for saving
        Tensorboard summaries.

    Returns:
      A dict containing summary statistics for this iteration.
    """
    statistics = iteration_statistics.IterationStatistics()
    logging.info('Starting iteration %d', iteration)
    num_episodes_train, average_reward_train, average_steps_per_second = (
        self._run_train_phase(statistics))
    num_episodes_eval, average_reward_eval = self._run_eval_phase(
        statistics)

    if self._has_collector_dispatcher:
      self._collector_dispatcher.write([
          statistics_instance.StatisticsInstance('Train/NumEpisodes',
                                                 num_episodes_train,
                                                 iteration),
          statistics_instance.StatisticsInstance('Train/AverageReturns',
                                                 average_reward_train,
                                                 iteration),
          statistics_instance.StatisticsInstance('Train/AverageStepsPerSecond',
                                                 average_steps_per_second,
                                                 iteration),
          statistics_instance.StatisticsInstance('Eval/NumEpisodes',
                                                 num_episodes_eval,
                                                 iteration),
          statistics_instance.StatisticsInstance('Eval/AverageReturns',
                                                 average_reward_eval,
                                                 iteration),
      ])
    if self._summary_writer is not None:
      self._save_tensorboard_summaries(iteration, num_episodes_train,
                                       average_reward_train, num_episodes_eval,
                                       average_reward_eval,
                                       average_steps_per_second)
    return statistics.data_lists
  
  def _save_reward_summaries(self, step, reward, eval_str):
    if self._summary_writer is None:
      return

    if self._sess is None:
      with self._summary_writer.as_default():
        tf.summary.scalar(eval_str + '/reward', reward,
                          step=step)
      self._summary_writer.flush()
    else:
      summary = tf.compat.v1.Summary(value=[
          tf.compat.v1.Summary.Value(
              tag=eval_str + '/reward', simple_value=reward),
      ])
      self._summary_writer.add_summary(summary, step)

  def _save_tensorboard_summaries(self, iteration,
                                  num_episodes_train,
                                  average_reward_train,
                                  num_episodes_eval,
                                  average_reward_eval,
                                  average_steps_per_second):
    """Save statistics as tensorboard summaries.

    Args:
      iteration: int, The current iteration number.
      num_episodes_train: int, number of training episodes run.
      average_reward_train: float, The average training reward.
      num_episodes_eval: int, number of evaluation episodes run.
      average_reward_eval: float, The average evaluation reward.
      average_steps_per_second: float, The average number of steps per second.
    """
    if self._summary_writer is None:
      return

    if self._sess is None:
      with self._summary_writer.as_default():
        tf.summary.scalar('Train/NumEpisodes', num_episodes_train,
                          step=iteration)
        tf.summary.scalar('Train/AverageReturns', average_reward_train,
                          step=iteration)
        tf.summary.scalar('Train/AverageStepsPerSecond',
                          average_steps_per_second, step=iteration)
        tf.summary.scalar('Eval/NumEpisodes', num_episodes_eval, step=iteration)
        tf.summary.scalar('Eval/AverageReturns', average_reward_eval,
                          step=iteration)
      self._summary_writer.flush()
    else:
      summary = tf.compat.v1.Summary(value=[
          tf.compat.v1.Summary.Value(
              tag='Train/NumEpisodes', simple_value=num_episodes_train),
          tf.compat.v1.Summary.Value(
              tag='Train/AverageReturns', simple_value=average_reward_train),
          tf.compat.v1.Summary.Value(
              tag='Train/AverageStepsPerSecond',
              simple_value=average_steps_per_second),
          tf.compat.v1.Summary.Value(
              tag='Eval/NumEpisodes', simple_value=num_episodes_eval),
          tf.compat.v1.Summary.Value(
              tag='Eval/AverageReturns', simple_value=average_reward_eval)
      ])
      self._summary_writer.add_summary(summary, iteration)

  def _log_experiment(self, iteration, statistics):
    """Records the results of the current iteration.

    Args:
      iteration: int, iteration number.
      statistics: `IterationStatistics` object containing statistics to log.
    """
    if not hasattr(self, '_logger'):
      return

    self._logger['iteration_{:d}'.format(iteration)] = statistics
    if iteration % self._log_every_n == 0:
      self._logger.log_to_file(self._logging_file_prefix, iteration, is_delete=False)

  def _checkpoint_experiment(self, iteration):
    """Checkpoint experiment data.

    Args:
      iteration: int, iteration number for checkpointing.
    """
    experiment_data = self._agent.bundle_and_checkpoint(self._checkpoint_dir,
                                                        iteration)
    if experiment_data:
      experiment_data['current_iteration'] = iteration
      if self._use_legacy_logger:
        experiment_data['logs'] = self._logger.data
      self._checkpointer.save_checkpoint(iteration, experiment_data, is_delete=False)

  def save_sub_best_checkpoints(self, iteration):
    ckpt_file = os.path.join(self._checkpoint_dir, "ckpt.10000")
    sentinel_file = os.path.join(self._checkpoint_dir, "sentinel_checkpoint_complete.10000")
    tf_ckpt_data_file = os.path.join(self._checkpoint_dir, "tf_ckpt-10000.data-00000-of-00001")
    tf_ckpt_index_file = os.path.join(self._checkpoint_dir, "tf_ckpt-10000.index")
    tf_ckpt_meta_file = os.path.join(self._checkpoint_dir, "tf_ckpt-10000.meta")
    logs_dir = os.path.join(self._base_dir, 'logs', "log_10000")

    new_ckpt_file = os.path.join(self._checkpoint_dir, "ckpt." + str(iteration + 10))
    new_sentinel_file = os.path.join(self._checkpoint_dir, "sentinel_checkpoint_complete." + str(iteration + 10))
    new_tf_ckpt_data_file = os.path.join(self._checkpoint_dir, "tf_ckpt-" + str(iteration + 10) + ".data-00000-of-00001")
    new_tf_ckpt_index_file = os.path.join(self._checkpoint_dir, "tf_ckpt-" + str(iteration + 10) + ".index")
    new_tf_ckpt_meta_file = os.path.join(self._checkpoint_dir, "tf_ckpt-" + str(iteration + 10) + ".meta")
    new_logs_dir = os.path.join(self._base_dir, 'logs', "log_" + str(iteration + 10))    

    copy(ckpt_file, new_ckpt_file)
    copy(sentinel_file, new_sentinel_file)
    copy(tf_ckpt_data_file, new_tf_ckpt_data_file)
    copy(tf_ckpt_index_file, new_tf_ckpt_index_file)
    copy(tf_ckpt_meta_file, new_tf_ckpt_meta_file)
    copy(logs_dir, new_logs_dir)

  def run_experiment(self):
    """Runs a full experiment, spread over multiple iterations."""
    logging.info('Beginning training...')
    if (self._num_iterations + 1) <= self._start_iteration:
    # if (self._num_iterations + 1) <= 212:
      logging.warning('num_iterations (%d) < start_iteration(%d)',
                      self._num_iterations, self._start_iteration)
      return

    best_average_returns = -float('inf')
    for iteration in range(self._start_iteration, self._num_iterations + 1):
      statistics = self._run_one_iteration(iteration)
      eval_average_return_list = statistics["eval_average_return"]
      average_return = eval_average_return_list[len(eval_average_return_list) - 1]

      if not self.only_eval:
        # return best checkpoints
        if best_average_returns < average_return:
          best_average_returns = average_return
          if self._use_legacy_logger:
            self._log_experiment(10000, statistics)
        
          self._checkpoint_experiment(10000)

        # record every 100 iteration
        if iteration % 100 == 0 and iteration != 0:
          if self._use_legacy_logger:
            self._log_experiment(iteration, statistics)
        
          self._checkpoint_experiment(iteration)

          # record the best checkpoints in every 100 iterations
          self.save_sub_best_checkpoints(iteration)

          # record best result
          result_path = self._base_dir + "/result.txt"
          with open(result_path, "a+") as f:
              f.write("iteration: {}, best_average_returns: {}\n".format(
                iteration, best_average_returns))

          if self._has_collector_dispatcher:
            self._collector_dispatcher.flush()
          
    if self._summary_writer is not None:
      self._summary_writer.flush()
    
    if not self.only_eval:
      if self._has_collector_dispatcher:
        self._collector_dispatcher.close()


@gin.configurable
class TrainRunner(Runner):
  """Object that handles running experiments.

  The `TrainRunner` differs from the base `Runner` class in that it does not
  the evaluation phase. Checkpointing and logging for the train phase are
  preserved as before.
  """

  def __init__(self, base_dir, create_agent_fn,
               create_environment_fn=atari_lib.create_atari_environment):
    """Initialize the TrainRunner object in charge of running a full experiment.

    Args:
      base_dir: str, the base directory to host all required sub-directories.
      create_agent_fn: A function that takes as args a Tensorflow session and an
        environment, and returns an agent.
      create_environment_fn: A function which receives a problem name and
        creates a Gym environment for that problem (e.g. an Atari 2600 game).
    """
    logging.info('Creating TrainRunner ...')
    super(TrainRunner, self).__init__(base_dir, create_agent_fn,
                                      create_environment_fn)
    self._agent.eval_mode = False

  def _run_one_iteration(self, iteration):
    """Runs one iteration of agent/environment interaction.

    An iteration involves running several episodes until a certain number of
    steps are obtained. This method differs from the `_run_one_iteration` method
    in the base `Runner` class in that it only runs the train phase.

    Args:
      iteration: int, current iteration number, used as a global_step for saving
        Tensorboard summaries.

    Returns:
      A dict containing summary statistics for this iteration.
    """
    statistics = iteration_statistics.IterationStatistics()
    num_episodes_train, average_reward_train, average_steps_per_second = (
        self._run_train_phase(statistics))

    self._collector_dispatcher.write([
        statistics_instance.StatisticsInstance('Train/NumEpisodes',
                                               num_episodes_train,
                                               iteration),
        statistics_instance.StatisticsInstance('Train/AverageReturns',
                                               average_reward_train,
                                               iteration),
        statistics_instance.StatisticsInstance('Train/AverageStepsPerSecond',
                                               average_steps_per_second,
                                               iteration),
    ])
    if self._summary_writer is not None:
      self._save_tensorboard_summaries(iteration, num_episodes_train,
                                       average_reward_train,
                                       average_steps_per_second)
    return statistics.data_lists

  def _save_tensorboard_summaries(self, iteration, num_episodes,
                                  average_reward, average_steps_per_second):
    """Save statistics as tensorboard summaries."""
    if self._summary_writer is None:
      return

    if self._sess is None:
      with self._summary_writer.as_default():
        tf.summary.scalar('Train/NumEpisodes', num_episodes, step=iteration)
        tf.summary.scalar('Train/AverageReturns', average_reward,
                          step=iteration)
        tf.summary.scalar('Train/AverageStepsPerSecond',
                          average_steps_per_second, step=iteration)
      self._summary_writer.flush()
    else:
      summary = tf.compat.v1.Summary(value=[
          tf.compat.v1.Summary.Value(
              tag='Train/NumEpisodes', simple_value=num_episodes),
          tf.compat.v1.Summary.Value(
              tag='Train/AverageReturns', simple_value=average_reward),
          tf.compat.v1.Summary.Value(
              tag='Train/AverageStepsPerSecond',
              simple_value=average_steps_per_second),
      ])
      self._summary_writer.add_summary(summary, iteration)
>>>>>>> ed0d7b9c1ae446c6cffed41684f9178952a22685
