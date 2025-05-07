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
"""Compact implementation of a DQN agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys
import copy
import random

from absl import logging

from dopamine.discrete_domains import atari_lib
from dopamine.replay_memory import circular_replay_buffer
from dopamine.adapt import tent
import gin.tf
import numpy as np
import tensorflow as tf


# These are aliases which are used by other classes.
NATURE_DQN_OBSERVATION_SHAPE = atari_lib.NATURE_DQN_OBSERVATION_SHAPE
NATURE_DQN_DTYPE = atari_lib.NATURE_DQN_DTYPE
NATURE_DQN_STACK_SIZE = atari_lib.NATURE_DQN_STACK_SIZE
nature_dqn_network = atari_lib.NatureDQNNetwork


@gin.configurable
def linearly_decaying_epsilon(decay_period, step, warmup_steps, epsilon):
  """Returns the current epsilon for the agent's epsilon-greedy policy.

  This follows the Nature DQN schedule of a linearly decaying epsilon (Mnih et
  al., 2015). The schedule is as follows:
    Begin at 1. until warmup_steps steps have been taken; then
    Linearly decay epsilon from 1. to epsilon in decay_period steps; and then
    Use epsilon from there on.

  Args:
    decay_period: float, the period over which epsilon is decayed.
    step: int, the number of training steps completed so far.
    warmup_steps: int, the number of steps taken before epsilon is decayed.
    epsilon: float, the final value to which to decay the epsilon parameter.

  Returns:
    A float, the current epsilon value computed according to the schedule.
  """
  steps_left = decay_period + warmup_steps - step
  bonus = (1.0 - epsilon) * steps_left / decay_period
  bonus = np.clip(bonus, 0., 1. - epsilon)
  return epsilon + bonus


@gin.configurable
def identity_epsilon(unused_decay_period, unused_step, unused_warmup_steps,
                     epsilon):
  return epsilon


def get_entropy(entropy, action_class_dim):
  return -(entropy * math.log(1 / action_class_dim))


@gin.configurable
class DQNAgent(object):
  """An implementation of the DQN agent."""

  def __init__(self,
               sess,
               num_actions,
               adapted=False,
               adapted_tent=False,
               adapted_final=False,
               adapted_final_ln=False,
               entropy=0.8,
               kl_coef=1.0,
               apapted_moment=False,
               moment_update_tta=0.99,
               action_class_dim=6,
               observation_shape=atari_lib.NATURE_DQN_OBSERVATION_SHAPE,
               observation_dtype=atari_lib.NATURE_DQN_DTYPE,
               stack_size=atari_lib.NATURE_DQN_STACK_SIZE,
               network=atari_lib.NatureDQNNetwork,
               gamma=0.99,
               update_horizon=1,
               min_replay_history=20000,
               update_period=4,
               target_update_period=8000,
               epsilon_fn=linearly_decaying_epsilon,
               epsilon_train=0.01,
               epsilon_eval=0.001,
               is_epsilon_adapted=False,
               epsilon_adapted_eval=0.01,
               epsilon_decay_period=250000,
               tf_device='/cpu:*',
               eval_mode=False,
               use_staging=False,
               max_tf_checkpoints_to_keep=100,
               optimizer=tf.compat.v1.train.RMSPropOptimizer(
                   learning_rate=0.00025,
                   decay=0.95,
                   momentum=0.0,
                   epsilon=0.00001,
                   centered=True),
               test_optimizer=tf.compat.v1.train.AdamOptimizer(
                  #  learning_rate=0.00005,
                   learning_rate=3e-4,
                   epsilon=0.0003125),
               summary_writer=None,
               summary_writing_frequency=500,
               allow_partial_reload=False):
    """Initializes the agent and constructs the components of its graph.

    Args:
      sess: `tf.compat.v1.Session`, for executing ops.
      num_actions: int, number of actions the agent can take at any state.
      observation_shape: tuple of ints describing the observation shape.
      observation_dtype: tf.DType, specifies the type of the observations. Note
        that if your inputs are continuous, you should set this to tf.float32.
      stack_size: int, number of frames to use in state stack.
      network: tf.Keras.Model, expecting 2 parameters: num_actions,
        network_type. A call to this object will return an instantiation of the
        network provided. The network returned can be run with different inputs
        to create different outputs. See
        dopamine.discrete_domains.atari_lib.NatureDQNNetwork as an example.
      gamma: float, discount factor with the usual RL meaning.
      update_horizon: int, horizon at which updates are performed, the 'n' in
        n-step update.
      min_replay_history: int, number of transitions that should be experienced
        before the agent begins training its value function.
      update_period: int, period between DQN updates.
      target_update_period: int, update period for the target network.
      epsilon_fn: function expecting 4 parameters:
        (decay_period, step, warmup_steps, epsilon). This function should return
        the epsilon value used for exploration during training.
      epsilon_train: float, the value to which the agent's epsilon is eventually
        decayed during training.
      epsilon_eval: float, epsilon used when evaluating the agent.
      epsilon_decay_period: int, length of the epsilon decay schedule.
      tf_device: str, Tensorflow device on which the agent's graph is executed.
      eval_mode: bool, True for evaluation and False for training.
      use_staging: bool, when True use a staging area to prefetch the next
        training batch, speeding training up by about 30%.
      max_tf_checkpoints_to_keep: int, the number of TensorFlow checkpoints to
        keep.
      optimizer: `tf.compat.v1.train.Optimizer`, for training the value
        function.
      summary_writer: SummaryWriter object for outputting training statistics.
        Summary writing disabled if set to None.
        May also be a str specifying the base directory, in which case the
        SummaryWriter will be created by the agent.
      summary_writing_frequency: int, frequency with which summaries will be
        written. Lower values will result in slower training.
      allow_partial_reload: bool, whether we allow reloading a partial agent
        (for instance, only the network parameters).
    """
    assert isinstance(observation_shape, tuple)
    logging.info('Creating %s agent with the following parameters:',
                 self.__class__.__name__)
    logging.info('\t gamma: %f', gamma)
    logging.info('\t update_horizon: %f', update_horizon)
    logging.info('\t min_replay_history: %d', min_replay_history)
    logging.info('\t update_period: %d', update_period)
    logging.info('\t target_update_period: %d', target_update_period)
    logging.info('\t epsilon_train: %f', epsilon_train)
    logging.info('\t epsilon_eval: %f', epsilon_eval)
    logging.info('\t epsilon_decay_period: %d', epsilon_decay_period)
    logging.info('\t tf_device: %s', tf_device)
    logging.info('\t use_staging: %s', use_staging)
    logging.info('\t optimizer: %s', optimizer)
    logging.info('\t max_tf_checkpoints_to_keep: %d',
                 max_tf_checkpoints_to_keep)

    self.num_actions = num_actions
    self.adapted = adapted
    self.moment_update_tta = moment_update_tta
    self.apapted_moment = apapted_moment
    self.adapted_tent = adapted_tent
    self.adapted_final = adapted_final
    self.adapted_final_ln = adapted_final_ln
    self.action_class_dim = action_class_dim
    self.entropy = get_entropy(entropy=entropy, action_class_dim=self.action_class_dim)
    self.kl_coef = kl_coef
    self.observation_shape = tuple(observation_shape)
    self.observation_dtype = observation_dtype
    self.stack_size = stack_size
    self.network = network
    self.gamma = gamma
    self.update_horizon = update_horizon
    self.cumulative_gamma = math.pow(gamma, update_horizon)
    self.min_replay_history = min_replay_history
    self.target_update_period = target_update_period
    self.epsilon_fn = epsilon_fn
    self.epsilon_train = epsilon_train
    self.epsilon_eval = epsilon_eval
    self.epsilon_adapted_eval = epsilon_adapted_eval
    self.is_epsilon_adapted = is_epsilon_adapted
    self.epsilon_decay_period = epsilon_decay_period
    self.update_period = update_period
    self.eval_mode = eval_mode
    self.training_steps = 0
    self.test_step = 0
    self.adapted_test_step = 0
    self.high_entropy_step = 0
    self.optimizer = optimizer
    self.test_optimizer = test_optimizer
    self.replay_states = []
    self.states_list = []
    self.actions_list = []
    self.replay_size = 32
    tf.compat.v1.disable_v2_behavior()
    if isinstance(summary_writer, str):  # If we're passing in directory name.
      self.summary_writer = tf.compat.v1.summary.FileWriter(summary_writer)
    else:
      self.summary_writer = summary_writer
    self.summary_writing_frequency = summary_writing_frequency
    self.allow_partial_reload = allow_partial_reload

    if sess is None:
      config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
      # Allocate only subset of the GPU memory as needed which allows for
      # running multiple agents/workers on the same GPU.
      config.gpu_options.allow_growth = True
      self._sess = tf.compat.v1.Session('', config=config)
    else:
      self._sess = sess

    # # store the state become a batch
    # self.batch_state = []

    with tf.device(tf_device):
      # Create a placeholder for the state input to the DQN network.
      # The last axis indicates the number of consecutive frames stacked.
      state_shape = (1,) + self.observation_shape + (stack_size,)
      self.state = np.zeros(state_shape)
      self.state_ph = tf.compat.v1.placeholder(
          self.observation_dtype, state_shape, name='state_ph')
      self.adapted_state = np.zeros(state_shape)
      self.adapted_state_ph = tf.compat.v1.placeholder(
          self.observation_dtype, state_shape, name='adapted_state_ph')
      self._replay = self._build_replay_buffer(use_staging)

      ############################################################
      # 2023/10/23: mixup augmenation data
      ############################################################
      aug_state_shape = (self.replay_size,) + self.observation_shape + (stack_size,)
      aug_action_shape = (self.replay_size,) + (self.action_class_dim, )
      self.aug_state_ph = tf.compat.v1.placeholder(
          self.observation_dtype, aug_state_shape, name='aug_state_ph')
      self.aug_action_ph = tf.compat.v1.placeholder(
          self.observation_dtype, aug_action_shape, name='aug_action_ph')
      ############################################################

      self._build_networks()

      self._train_op = self._build_train_op()

      if self.adapted:
        self._sync_online_to_offline_ops = self._build_online_to_offline_op()
      
      self._sync_qt_ops = self._build_sync_op()

    if self.summary_writer is not None:
      # All tf.summaries should have been defined prior to running this.
      self._merged_summaries = tf.compat.v1.summary.merge_all()

    if self.adapted:
      vars_too_save = [v for v in tf.compat.v1.global_variables() 
                       if not any(keyword in v.name for keyword in ['Moment', 'TTA', 'beta1_power:1', 'beta2_power:1', 'beta1_power_1', "beta2_power_1", "Online/Conv/bias/RMSProp"])]
      # vars_too_save = [v for v in tf.compat.v1.global_variables() 
      #                  if not any(keyword in v.name for keyword in ['TTA', 'beta1_power:1', 'beta2_power:1', 'beta1_power_1', "beta2_power_1"])]
      var_map = vars_too_save
    else:
      var_map = atari_lib.maybe_transform_variable_names(
          tf.compat.v1.global_variables())
    
    # var_map = atari_lib.transform_variable_names(var_map)
    # print(tf.compat.v1.global_variables())
    # print(var_map)
    # exit()
    self._saver = tf.compat.v1.train.Saver(
        var_list=var_map, max_to_keep=max_tf_checkpoints_to_keep)

    # Variables to be initialized by the agent once it interacts with the
    # environment.
    self._observation = None
    self._last_observation = None

    if self.summary_writer is not None:
      self.summary_writer.add_graph(graph=tf.compat.v1.get_default_graph())
    self._sess.run(tf.compat.v1.global_variables_initializer())

  def _create_network(self, name):
    """Builds the convolutional network used to compute the agent's Q-values.

    Args:
      name: str, this name is passed to the tf.keras.Model and used to create
        variable scope under the hood by the tf.keras.Model.
    Returns:
      network: tf.keras.Model, the network instantiated by the Keras model.
    """
    if callable(self.network):
      network = self.network(self.num_actions, name=name)
    else:
      network_class = eval(self.network)
      network = network_class(self.num_actions, name=name)

    return network

  def _build_networks(self):
    """Builds the Q-value network computations needed for acting and training.

    These are:
      self.online_convnet: For computing the current state's Q-values.
      self.target_convnet: For computing the next state's target Q-values.
      self._net_outputs: The actual Q-values.
      self._q_argmax: The action maximizing the current state's Q-values.
      self._replay_net_outputs: The replayed states' Q-values.
      self._replay_next_target_net_outputs: The replayed next states' target
        Q-values (see Mnih et al., 2015 for details).
    """

    # _network_template instantiates the model and returns the network object.
    # The network object can be used to generate different outputs in the graph.
    # At each call to the network, the parameters will be reused.

    self.online_convnet = self._create_network(name='Online')
    self.target_convnet = self._create_network(name='Target')
    
    # add normalization update
    if self.network == "atari_helpers.QuantileBatchNormNetwork":
      if self.eval_mode == True:
        self._net_outputs = self.online_convnet(self.state_ph, training=False)
        # TODO(bellemare): Ties should be broken. They are unlikely to happen when
        # using a deep network, but may affect performance with a linear
        # approximation scheme.
        self._q_entropy = tf.reduce_mean(tent.softmax_entropy(self._net_outputs.q_values), axis=0)
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
          self.test_op = self.test_optimizer.minimize(self._q_entropy)

        self._q_argmax = tf.argmax(self._net_outputs.q_values, axis=1)[0]
        self._replay_net_outputs = self.online_convnet(self._replay.states, training=False)
        self._replay_next_target_net_outputs = self.target_convnet(
            self._replay.next_states, training=False)
      else:
        self._net_outputs = self.online_convnet(self.state_ph, training=True)
        # TODO(bellemare): Ties should be broken. They are unlikely to happen when
        # using a deep network, but may affect performance with a linear
        # approximation scheme.
        self._q_argmax = tf.argmax(self._net_outputs.q_values, axis=1)[0]
        self._replay_net_outputs = self.online_convnet(self._replay.states, training=True)
        self._replay_next_target_net_outputs = self.target_convnet(
            self._replay.next_states, training=True)
    else:
      self._net_outputs = self.online_convnet(self.state_ph)
      # TODO(bellemare): Ties should be broken. They are unlikely to happen when
      # using a deep network, but may affect performance with a linear
      # approximation scheme.
      self._q_argmax = tf.argmax(self._net_outputs.q_values, axis=1)[0]

      self._replay_net_outputs = self.online_convnet(self._replay.states)
      self._replay_next_target_net_outputs = self.target_convnet(
          self._replay.next_states)

      if self.adapted:
        self.online_norm_q_values, _ = tf.linalg.normalize(self._net_outputs.q_values, ord=2, axis=-1)
        self._online_test_prob = tf.nn.softmax(self.online_norm_q_values, axis=-1)
        self._q_entropy = -tf.reduce_sum(self._online_test_prob * tf.math.log(self._online_test_prob))
        # self._q_entropy = tf.reduce_mean(tent.softmax_entropy(self._net_outputs.q_values), axis=0)
        self._q_value = (self._net_outputs.q_values)[0][self._q_argmax]
      
        # recore eval summary
        if self.summary_writer is not None:
          with tf.compat.v1.variable_scope('eval'):
            # eval_q_value = tf.compat.v1.summary.histogram('eval_q_values_distribution', self._net_outputs.q_values)
            eval_q_value = tf.compat.v1.summary.scalar('eval_q_value', self._q_value) 
            eval_q_entropy = tf.compat.v1.summary.scalar('entorpy', self._q_entropy)
            eval_q_action = tf.compat.v1.summary.scalar('action', self._q_argmax)
            self._merged_eval_summaries = tf.compat.v1.summary.merge([eval_q_value, eval_q_entropy, eval_q_action])

    # add test-time adaptation
    if self.adapted:
      self.test_online_convnet = self._create_network(name='TTA')
      if self.adapted_final:
        self.test_online_convnet = tent.finetune_final_layer(self.test_online_convnet)
      elif self.adapted_final_ln:
        self.test_online_convnet = tent.configure_final_ln_layer(self.test_online_convnet)
      else:
        self.test_online_convnet = tent.configure_model(self.test_online_convnet)
      self._test_net_outputs = self.test_online_convnet(self.adapted_state_ph)

      self.norm_q_values, _ = tf.linalg.normalize(self._test_net_outputs.q_values, ord=2, axis=-1)
      self._test_prob = tf.nn.softmax(self.norm_q_values, axis=-1)
      self._test_q_entropy = -tf.reduce_sum(self._test_prob * tf.math.log(self._test_prob))

      # self._test_q_entropy = tf.reduce_mean(tent.softmax_entropy(self._test_net_outputs.q_values), axis=0)
      self._test_q_argmax = tf.argmax(self._test_net_outputs.q_values, axis=1)[0]
      if self.adapted_tent:
        self._test_op = self.test_optimizer.minimize(self._test_q_entropy)
      else:
        self._test_before_action_distribution = tf.stop_gradient(tf.nn.softmax(self._net_outputs.q_values, axis=-1))
        self._test_after_action_distribution = tf.nn.softmax(self._test_net_outputs.q_values, axis=-1)
        self._test_kl_divergence = tf.reduce_sum(self._test_before_action_distribution * ((tf.math.log(self._test_before_action_distribution + 1e-8)) - (tf.math.log(self._test_after_action_distribution + 1e-8))))
        self._test_loss = self.kl_coef * self._test_kl_divergence + self._test_q_entropy
        self._test_op = self.test_optimizer.minimize(self._test_loss)
        # self._test_op = self.test_optimizer.minimize(self._test_q_entropy)

      ############################################################
      # 2023/10/24: mixup augmenation data
      ############################################################
      # get pesudo label 
      self._pesudo_norm_q_values, _ = tf.linalg.normalize(self._test_net_outputs.q_values, ord=2, axis=-1)
      self._pesudo_aug_test_prob = tf.nn.softmax(self._pesudo_norm_q_values, axis=-1)
      shape = tf.shape(self._pesudo_aug_test_prob)
      max_value = tf.reduce_max(self._pesudo_aug_test_prob, axis=1)
      self._action_label = tf.one_hot(tf.cast(max_value, tf.int64), shape[1], axis=-1) 
      # update with the augmentation data
      self._aug_test_net_outputs = self.test_online_convnet(self.aug_state_ph)
      self._aug_norm_q_values, _ = tf.linalg.normalize(self._aug_test_net_outputs.q_values, ord=2, axis=-1)
      self.softmax_aug_data = tf.nn.softmax(self._aug_norm_q_values, axis=-1)
      # build the cross entropy loss
      self.neg_soft_log = tf.cast(self.aug_action_ph, tf.float32) * tf.math.log(self.softmax_aug_data)
      self._aug_cross_entropy_loss = - tf.reduce_sum(self.neg_soft_log)
      self._aug_test_op = self.test_optimizer.minimize(tf.reduce_sum(self._aug_cross_entropy_loss))
      ############################################################

      # recore eval summary
      if self.summary_writer is not None:
        with tf.compat.v1.variable_scope('tta'):
          # tta_q_value = tf.compat.v1.summary.histogram('tta_q_values_distribution', self._test_net_outputs.q_values)
          tta_q_value = tf.compat.v1.summary.scalar('tta_q_value', (self._test_net_outputs.q_values)[0][self._test_q_argmax])
          tta_entorpy = tf.compat.v1.summary.scalar('entorpy', self._test_q_entropy)
          tta_action = tf.compat.v1.summary.scalar('action', self._test_q_argmax)
          # tta_state = tf.compat.v1.summary.tensor_summary("tta_state", self.adapted_state_ph)
          # tta_entropy_before = tf.compat.v1.summary.scalar('before_entropy', self._test_q_entropy)
          # tta_entropy_after = tf.compat.v1.summary.scalar('after_entropy', self._test_q_entropy)
          # self._merged_tta_tensor_summaries = tf.compat.v1.summary.merge([tta_entropy_before])
          self._merged_tta_summaries = tf.compat.v1.summary.merge([tta_q_value, tta_entorpy, tta_action])
          # self._merged_tta_summaries = tf.compat.v1.summary.merge([tta_q_value, tta_entorpy])
          # self._merged_tta_entropy_before_summaries = tf.compat.v1.summary.merge([tta_entropy_before])
          # self._merged_tta_entropy_after_summaries = tf.compat.v1.summary.merge([tta_entropy_after])

          # # record extra entropy
          # eval_q_entropy = tf.compat.v1.summary.scalar('eval_entorpy', self._q_entropy)
          # tta_q_entropy = tf.compat.v1.summary.scalar('tta_entorpy', self._test_q_entropy)
          # self._merged_entropy_summaries = tf.compat.v1.summary.merge([eval_q_entropy, tta_q_entropy])

    # add moment update
    if self.apapted_moment:
      self.moment_convnet = self._create_network(name='Moment')
      self._moment_net_outputs = self.moment_convnet(self.adapted_state_ph)
      self._copy_tta_to_moment_op = self._build_tta_to_moment_op()
      self._tta_moment_update_op = self._build_moment_update_op()

  def _build_replay_buffer(self, use_staging):
    """Creates the replay buffer used by the agent.

    Args:
      use_staging: bool, if True, uses a staging area to prefetch data for
        faster training.

    Returns:
      A WrapperReplayBuffer object.
    """
    return circular_replay_buffer.WrappedReplayBuffer(
        observation_shape=self.observation_shape,
        stack_size=self.stack_size,
        use_staging=use_staging,
        update_horizon=self.update_horizon,
        gamma=self.gamma,
        observation_dtype=self.observation_dtype.as_numpy_dtype)

  def _build_target_q_op(self):
    """Build an op used as a target for the Q-value.

    Returns:
      target_q_op: An op calculating the Q-value.
    """
    # Get the maximum Q-value across the actions dimension.
    replay_next_qt_max = tf.reduce_max(
        self._replay_next_target_net_outputs.q_values, 1)
    # Calculate the Bellman target value.
    #   Q_t = R_t + \gamma^N * Q'_t+1
    # where,
    #   Q'_t+1 = \argmax_a Q(S_t+1, a)
    #          (or) 0 if S_t is a terminal state,
    # and
    #   N is the update horizon (by default, N=1).
    return self._replay.rewards + self.cumulative_gamma * replay_next_qt_max * (
        1. - tf.cast(self._replay.terminals, tf.float32))

  def _build_train_op(self):
    """Builds a training op.

    Returns:
      train_op: An op performing one step of training from replay data.
    """
    replay_action_one_hot = tf.one_hot(
        self._replay.actions, self.num_actions, 1., 0., name='action_one_hot')
    replay_chosen_q = tf.reduce_sum(
        self._replay_net_outputs.q_values * replay_action_one_hot,
        axis=1,
        name='replay_chosen_q')

    target = tf.stop_gradient(self._build_target_q_op())
    loss = tf.compat.v1.losses.huber_loss(
        target, replay_chosen_q, reduction=tf.losses.Reduction.NONE)
    if self.summary_writer is not None:
      with tf.compat.v1.variable_scope('Losses'):
        merge_train_summary = tf.compat.v1.summary.scalar('HuberLoss', tf.reduce_mean(loss))

    # add bn update operation into graph
    if self.network == "atari_helpers.QuantileBatchNormNetwork":
      update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        train_op = self.optimizer.minimize(tf.reduce_mean(loss))
    else:
      train_op = self.optimizer.minimize(tf.reduce_mean(loss))
    return train_op, loss, merge_train_summary
    # return train_op

  def _build_sync_op(self):
    """Builds ops for assigning weights from online to target network.

    Returns:
      ops: A list of ops assigning weights from online to target network.
    """
    # Get trainable variables from online and target DQNs
    sync_qt_ops = []
    scope = tf.compat.v1.get_default_graph().get_name_scope()
    trainables_online = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
        scope=os.path.join(scope, 'Online'))
    trainables_target = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
        scope=os.path.join(scope, 'Target'))

    for (w_online, w_target) in zip(trainables_online, trainables_target):
      # Assign weights from online to target network.
      sync_qt_ops.append(w_target.assign(w_online, use_locking=True))
    return sync_qt_ops
  
  def _build_moment_update_op(self):
    """Builds ops for assigning weights from online to target network.

    Returns:
      ops: A list of ops assigning weights from online to target network.
    """
    # Get trainable variables from online and target DQNs
    sync_qt_ops = []
    scope = tf.compat.v1.get_default_graph().get_name_scope()
    trainables_tta = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
        scope=os.path.join(scope, 'TTA'))
    trainables_moment = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
        scope=os.path.join(scope, 'Moment'))
    
    # moment update parameters
    for (w_tta, w_moment) in zip(trainables_tta, trainables_moment):
      # Assign weights from online to target network.
      sync_qt_ops.append(w_tta.assign(w_moment * self.moment_update_tta + (1 - self.moment_update_tta) * w_tta, use_locking=True))
    return sync_qt_ops

  def _build_tta_to_moment_op(self):
    """Builds ops for assigning weights from online to target network.

    Returns:
      ops: A list of ops assigning weights from online to target network.
    """
    # Get trainable variables from online and target DQNs
    sync_qt_ops = []
    scope = tf.compat.v1.get_default_graph().get_name_scope()
    trainables_tta = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
        scope=os.path.join(scope, 'TTA'))
    trainables_moment = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
        scope=os.path.join(scope, 'Moment'))
    
    for (w_tta, w_moment) in zip(trainables_tta, trainables_moment):
      # Assign weights from online to target network.
      sync_qt_ops.append(w_moment.assign(w_tta, use_locking=True))

    return sync_qt_ops 
  
  def _build_online_to_offline_op(self):
    """Builds ops for assigning weights from online to target network.

    Returns:
      ops: A list of ops assigning weights from online to target network.
    """
    # Get trainable variables from online and target DQNs
    # for i, layer in enumerate(self.online_convnet.layers): 
    #   self.test_online_convnet.layers[i].set_weights(copy.deepcopy(layer.get_weights()))
      # self.test_online_convnet.layers[i].set_bi
      # print(layer.get_weights())

    sync_online_to_offline_ops = []
    scope = tf.compat.v1.get_default_graph().get_name_scope()
    trainables_online = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
        scope=os.path.join(scope, 'Online'))
    trainables_tta = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
        scope=os.path.join(scope, 'TTA'))

    for (w_online, w_tta) in zip(trainables_online, trainables_tta):
      # Assign weights from online to target network.
      sync_online_to_offline_ops.append(w_tta.assign(w_online, use_locking=True))
    return sync_online_to_offline_ops

  # # copy parameters from online network to online-tta network
  # def copy_network_params(self):
  #   for i, layer in enumerate(self.online_convnet.layers): 
  #     self.test_online_convnet.layers[i].set_weights(copy.deepcopy(layer.get_weights()))
  #     print(layer.get_weights())

  def begin_episode(self, observation):
    """Returns the agent's first action for this episode.

    Args:
      observation: numpy array, the environment's initial observation.

    Returns:
      int, the selected action.
    """
    self._reset_state()
    self._record_observation(observation)

    if not self.eval_mode:
      self._train_step()

    self.action = self._select_action()
    return self.action
  
  def adapted_begin_episode(self, observation):
    """Returns the agent's first action for this episode.

    Args:
      observation: numpy array, the environment's initial observation.

    Returns:
      int, the selected action.
    """
    self._reset_adapted_state()
    self._record_adapted_observation(observation)

    if not self.eval_mode:
      self._train_step()

    self.action = self._select_adapted_action()
    return self.action

  def step(self, reward, observation):
    """Records the most recent transition and returns the agent's next action.

    We store the observation of the last time step since we want to store it
    with the reward.

    Args:
      reward: float, the reward received from the agent's most recent action.
      observation: numpy array, the most recent observation.

    Returns:
      int, the selected action.
    """
    self._last_observation = self._observation
    self._record_observation(observation)

    if not self.eval_mode:
      self._store_transition(self._last_observation, self.action, reward, False)
      self._train_step()

    self.action = self._select_action()
    return self.action

  def end_episode(self, reward):
    """Signals the end of the episode to the agent.

    We store the observation of the current time step, which is the last
    observation of the episode.

    Args:
      reward: float, the last reward from the environment.
    """
    if not self.eval_mode:
      self._store_transition(self._observation, self.action, reward, True)

  def mixup_sample(self, features1, labels1, features2, labels2, alpha):
    weight1 = tf.cast(alpha, dtype=tf.float32)
    weight2 = 1.0 - weight1
    mixed_features = weight1 * features1 + weight2 * features2
    mixed_labels = weight1 * labels1 + weight2 * labels2
    return mixed_features, mixed_labels

  def mixup_batch(self, features, labels, alpha):
    num_samples = len(features)
    indices = tf.range(num_samples, dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)
    mixed_features, mixed_labels = self.mixup_sample(
        features, labels, tf.gather(features, shuffled_indices),
        tf.gather(labels, shuffled_indices), alpha)
    return mixed_features, mixed_labels

  def _select_adapted_action(self):
    """Select an action from the set of available actions.

    Chooses an action randomly with probability self._calculate_epsilon(), and
    otherwise acts greedily according to the current Q-value estimates.

    Returns:
       int, the selected action.
    """
    if self.eval_mode:
      if self.is_epsilon_adapted:
        epsilon = self.epsilon_adapted_eval # default: 0.001      
      else:
        epsilon = self.epsilon_eval # default: 0.001         

    else:
      epsilon = self.epsilon_fn(
          self.epsilon_decay_period,  # 250000
          self.training_steps,        # 0
          self.min_replay_history,    # 20000
          self.epsilon_train)         # 0.01
    if random.random() <= epsilon:
      # Choose a random action with probability epsilon.
      return random.randint(0, self.num_actions - 1)
    else:
      # # 创建一个初始值为 0 的 1 维张量（大小为 84 x 84）
      # self.adapted_state = np.ones((1, 84, 84, 4))
      # self._sess.run(self._sync_online_to_offline_ops)
      
      # print("adapted action: ", self._sess.run(self._test_net_outputs.q_values, {self.adapted_state_ph: self.adapted_state}))
      # print("online  action: ", self._sess.run(self._net_outputs.q_values, {self.state_ph: self.adapted_state}))
      # print("adapted probabilities: ", self._sess.run(self._test_net_outputs.probabilities[0][1][0:4], {self.adapted_state_ph: self.adapted_state}))
      # print("online  probabilities: ", self._sess.run(self._net_outputs.probabilities[0][1][0:4], {self.state_ph: self.adapted_state}))
      # print("adapted logits: ", self._sess.run(self._test_net_outputs.logits[0][1][0:4], {self.adapted_state_ph: self.adapted_state}))
      # print("online  logits: ", self._sess.run(self._net_outputs.logits[0][1][0:4], {self.state_ph: self.adapted_state}))

      # # tta_layer = self.test_online_convnet.layers[2].get_weights()[0]
      # # target_layer = self.online_convnet.layers[2].get_weights()[0]
      # # print("adapted parameter: {}, online parameter: {}".format(tta_layer[0][0][34][45], target_layer[0][0][34][45]))
      # # print("adapted parameter: {}, online parameter: {}".format(tta_layer[0][1][23][34], target_layer[0][1][23][34]))
      # # print("adapted parameter: {}, online parameter: {}".format(tta_layer[1][0][12][54], target_layer[1][0][12][54]))
      # exit()

      # self.batch_state.append(self.adapted_state)
      # if len(self.batch_state) == 32:
      #   batch_state = np.array(self.batch_state)
      #   action = self._sess.run(self._test_q_argmax, {self.adapted_state_ph: batch_state})
      #   print('*' * 100)
      #   print(action)
      #   print('*' * 100)

      # Choose the action with highest Q-value at the current state.
      action = self._sess.run(self._test_q_argmax, {self.adapted_state_ph: self.adapted_state})
      entropy = self._sess.run(self._test_q_entropy, {self.adapted_state_ph: self.adapted_state})

      output = self._sess.run(self._test_net_outputs.q_values, {self.adapted_state_ph: self.adapted_state})
      normal_output = self._sess.run(self.norm_q_values, {self.adapted_state_ph: self.adapted_state})
      print("output: ", output)
      print("normal output: ", normal_output)
      print("pro: ", self._sess.run(self._test_prob, {self.adapted_state_ph: self.adapted_state}))
      if self.summary_writer is not None:
        summary = self._sess.run(self._merged_tta_summaries, {self.adapted_state_ph: self.adapted_state})
        self.summary_writer.add_summary(summary, self.adapted_test_step)
        self.adapted_test_step += 1

      # print("tta Q value: ", self._sess.run(self._test_net_outputs.q_values, {self.adapted_state_ph: self.adapted_state}))
      # test-time adaptation
      if self.adapted_tent:
        if entropy < self.entropy:
          self._sess.run(self._test_op, {self.adapted_state_ph: self.adapted_state})

          if self.apapted_moment:
            # moment update tta network parameters
            self._sess.run(self._tta_moment_update_op)
            # copy tta network parameters to moment network
            self._sess.run(self._copy_tta_to_moment_op)

        # else: 
        #   if self.summary_writer is not None:
        #     summary = self._sess.run(self._merged_tta_tensor_summaries, {self.adapted_state_ph: self.adapted_state})
        #     self.summary_writer.add_summary(summary, self.high_entropy_step)
        #     self.high_entropy_step += 1
        #     self.replay_states.append(self.adapted_state)
      else:
        if entropy < self.entropy:
          # update model
          self._sess.run(self._test_op, {self.state_ph: self.adapted_state, 
                                        self.adapted_state_ph: self.adapted_state})

          ############################################################
          # 2023/10/23: mixup augmenation data
          # build statis graph, feed the data for the operator
          ############################################################
          print("-" * 200)
          print("state type: ", type(self.adapted_state))
          print("action type: ", type(action))
          print("state shape: ", self.adapted_state.shape)
          print("state: ", self.adapted_state)
          print("action: ", action)
          print("-" * 200)
          self.states_list.append(self.adapted_state)
          self.actions_list.append(self._sess.run(self._action_label, {self.adapted_state_ph: self.adapted_state}))

          if len(self.states_list) == 32:
            # transform tensor
            aug_states = np.squeeze(np.stack(self.states_list), axis=1)
            aug_actions = tf.squeeze(tf.stack(self.actions_list, axis=0, name="stack_actions"))
            aug_actions = tf.cast(aug_actions, dtypes=tf.float32)
            print("aug_state's shape: ", aug_states.shape)
            print("aug_action's shape: ", aug_actions.shape)
            # mixup data
            mixed_states, mixed_actions = self.mixup_batch(aug_states, aug_actions, 0.9)
            # update model
            self._sess.run(self._aug_test_op, {self.aug_state_ph: mixed_states, 
                                          self.aug_action_ph: mixed_actions})
            # clear replay buffer
            self.states_list = []
            self.actions_list = []
          ############################################################

        # else:
        #   if self.summary_writer is not None:
        #     summary = self._sess.run(self._merged_tta_tensor_summaries, {self.adapted_state_ph: self.adapted_state})
        #     self.summary_writer.add_summary(summary, self.high_entropy_step)
        #     self.high_entropy_step += 1
        #     self.replay_states.append(self.adapted_state)

      return action

  def _compute_entropy(self):
    for (i, state) in enumerate(self.replay_states):
      if self.summary_writer is not None:
        summary = self._sess.run(self._merged_entropy_summaries, {self.adapted_state_ph: state,
                                                                  self.state_ph: state})
        self.summary_writer.add_summary(summary, i)

  def _select_action(self):
    """Select an action from the set of available actions.

    Chooses an action randomly with probability self._calculate_epsilon(), and
    otherwise acts greedily according to the current Q-value estimates.

    Returns:
       int, the selected action.
    """
    if self.eval_mode:
      epsilon = self.epsilon_eval
    else:
      epsilon = self.epsilon_fn(
          self.epsilon_decay_period,
          self.training_steps,
          self.min_replay_history,
          self.epsilon_train)
    if random.random() <= epsilon:
      # Choose a random action with probability epsilon.
      return random.randint(0, self.num_actions - 1)
    else:
      # # 创建一个初始值为 0 的 1 维张量（大小为 84 x 84）
      # self.state = np.ones((1, 84, 84, 4))
      # # self.state = np.arange(0, 84 * 84 * 4).reshape((1, 84, 84, 4)) 
      # # 将张量 reshape 成形状为 (84, 84, 1)
      # target_layer = self.online_convnet.layers[2].get_weights()[0]
      # print("select_action online parameter: ", target_layer[0][0][34][45])
      # print("select_action online parameter: ", target_layer[0][1][23][34])
      # print("select_action online parameter: ", target_layer[1][0][12][54])
      # print("select_action online action: ", self._sess.run(self._net_outputs.q_values, {self.state_ph: self.state}))

      # # Choose the action with highest Q-value at the current state.
      # if self.adapted and self.summary_writer is not None:
      #   summary = self._sess.run(self._merged_eval_summaries, feed_dict={self.state_ph: self.state})
      #   self.summary_writer.add_summary(summary, self.test_step)
      #   # print(summary)
      #   # print(self.summary_writer)

      #   self.test_step += 1
      # # print("Q value: : ", self._sess.run(self._net_outputs.q_values, {self.state_ph: self.state}))

      return self._sess.run(self._q_argmax, {self.state_ph: self.state})

  def copy_tta_to_monent(self):
    self._sess.run(self._copy_tta_to_moment_op)

  def update_online_tta_op(self):
    self._sess.run(self._sync_online_to_offline_ops)

  def _train_step(self):
    """Runs a single training step.

    Runs a training op if both:
      (1) A minimum number of frames have been added to the replay buff_train_steper.
      (2) `training_steps` is a multiple of `update_period`.

    Also, syncs weights from online to target network if training steps is a
    multiple of target update period.
    """
    # Run a train op at the rate of self.update_period if enough training steps
    # have been run. This matches the Nature DQN behaviour.
    if self._replay.memory.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        _, _, merged_train_summaries = self._sess.run(self._train_op)
        if (self.summary_writer is not None and
            self.training_steps > 0 and
            self.training_steps % self.summary_writing_frequency == 0):
          # summary = self._sess.run(self._merged_summaries)
          self.summary_writer.add_summary(merged_train_summaries, self.training_steps)

      if self.training_steps % self.target_update_period == 0:
        self._sess.run(self._sync_qt_ops)

    self.training_steps += 1

  def _record_observation(self, observation):
    """Records an observation and update state.

    Extracts a frame from the observation vector and overwrites the oldest
    frame in the state buffer.

    Args:
      observation: numpy array, an observation from the environment.
    """
    # Set current observation. We do the reshaping to handle environments
    # without frame stacking.
    self._observation = np.reshape(observation, self.observation_shape)
    # Swap out the oldest frame with the current frame.
    self.state = np.roll(self.state, -1, axis=-1)
    self.state[0, ..., -1] = self._observation
    # print("state: ", self.state)
  
  def _record_adapted_observation(self, observation):
    """Records an observation and update state.

    Extracts a frame from the observation vector and overwrites the oldest
    frame in the state buffer.

    Args:
      observation: numpy array, an observation from the environment.
    """
    # Set current observation. We do the reshaping to handle environments
    # without frame stacking.
    self._adpted_observation = np.reshape(observation, self.observation_shape)
    # Swap out the oldest frame with the current frame.
    self.adapted_state = np.roll(self.adapted_state, -1, axis=-1)
    self.adapted_state[0, ..., -1] = self._adpted_observation
    # print("adapted state: ", self.adapted_state)

  def _store_transition(self, last_observation, action, reward, is_terminal):
    """Stores an experienced transition.

    Executes a tf session and executes replay buffer ops in order to store the
    following tuple in the replay buffer:
      (last_observation, action, reward, is_terminal).

    Pedantically speaking, this does not actually store an entire transition
    since the next state is recorded on the following time step.

    Args:
      last_observation: numpy array, last observation.
      action: int, the action taken.
      reward: float, the reward.
      is_terminal: bool, indicating if the current state is a terminal state.
    """
    self._replay.add(last_observation, action, reward, is_terminal)

  def _reset_state(self):
    """Resets the agent state by filling it with zeros."""
    self.state.fill(0)
  
  def _reset_adapted_state(self):
    """Resets the agent state by filling it with zeros."""
    self.adapted_state.fill(0)

  def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
    """Returns a self-contained bundle of the agent's state.

    This is used for checkpointing. It will return a dictionary containing all
    non-TensorFlow objects (to be saved into a file by the caller), and it saves
    all TensorFlow objects into a checkpoint file.

    Args:
      checkpoint_dir: str, directory where TensorFlow objects will be saved.
      iteration_number: int, iteration number to use for naming the checkpoint
        file.

    Returns:
      A dict containing additional Python objects to be checkpointed by the
        experiment. If the checkpoint directory does not exist, returns None.
    """
    if not tf.io.gfile.exists(checkpoint_dir):
      return None
    # Call the Tensorflow saver to checkpoint the graph.
    self._saver.save(
        self._sess,
        os.path.join(checkpoint_dir, 'tf_ckpt'),
        global_step=iteration_number)
    # Checkpoint the out-of-graph replay buffer.
    self._replay.save(checkpoint_dir, iteration_number)
    bundle_dictionary = {}
    bundle_dictionary['state'] = self.state
    bundle_dictionary['training_steps'] = self.training_steps
    return bundle_dictionary

  def unbundle(self, checkpoint_dir, iteration_number, bundle_dictionary):
    """Restores the agent from a checkpoint.

    Restores the agent's Python objects to those specified in bundle_dictionary,
    and restores the TensorFlow objects to those specified in the
    checkpoint_dir. If the checkpoint_dir does not exist, will not reset the
      agent's state.

    Args:
      checkpoint_dir: str, path to the checkpoint saved by tf.Save.
      iteration_number: int, checkpoint version, used when restoring the replay
        buffer.
      bundle_dictionary: dict, containing additional Python objects owned by
        the agent.

    Returns:
      bool, True if unbundling was successful.
    """
    try:
      # self._replay.load() will throw a NotFoundError if it does not find all
      # the necessary files.
      self._replay.load(checkpoint_dir, iteration_number)
    except tf.errors.NotFoundError:
      if not self.allow_partial_reload:
        # If we don't allow partial reloads, we will return False.
        return False
      logging.warning('Unable to reload replay buffer!')
    if bundle_dictionary is not None:
      for key in self.__dict__:
        if key in bundle_dictionary:
          self.__dict__[key] = bundle_dictionary[key]
    elif not self.allow_partial_reload:
      return False
    else:
      logging.warning("Unable to reload the agent's parameters!")
    # Restore the agent's TensorFlow graph.
    self._saver.restore(self._sess,
                        os.path.join(checkpoint_dir,
                                     'tf_ckpt-{}'.format(iteration_number)))
    return True
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
"""Compact implementation of a DQN agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys
import copy
import random

from absl import logging

from dopamine.discrete_domains import atari_lib
from dopamine.replay_memory import circular_replay_buffer
from dopamine.adapt import tent
import gin.tf
import numpy as np
import tensorflow as tf


# These are aliases which are used by other classes.
NATURE_DQN_OBSERVATION_SHAPE = atari_lib.NATURE_DQN_OBSERVATION_SHAPE
NATURE_DQN_DTYPE = atari_lib.NATURE_DQN_DTYPE
NATURE_DQN_STACK_SIZE = atari_lib.NATURE_DQN_STACK_SIZE
nature_dqn_network = atari_lib.NatureDQNNetwork


@gin.configurable
def linearly_decaying_epsilon(decay_period, step, warmup_steps, epsilon):
  """Returns the current epsilon for the agent's epsilon-greedy policy.

  This follows the Nature DQN schedule of a linearly decaying epsilon (Mnih et
  al., 2015). The schedule is as follows:
    Begin at 1. until warmup_steps steps have been taken; then
    Linearly decay epsilon from 1. to epsilon in decay_period steps; and then
    Use epsilon from there on.

  Args:
    decay_period: float, the period over which epsilon is decayed.
    step: int, the number of training steps completed so far.
    warmup_steps: int, the number of steps taken before epsilon is decayed.
    epsilon: float, the final value to which to decay the epsilon parameter.

  Returns:
    A float, the current epsilon value computed according to the schedule.
  """
  steps_left = decay_period + warmup_steps - step
  bonus = (1.0 - epsilon) * steps_left / decay_period
  bonus = np.clip(bonus, 0., 1. - epsilon)
  return epsilon + bonus


@gin.configurable
def identity_epsilon(unused_decay_period, unused_step, unused_warmup_steps,
                     epsilon):
  return epsilon


def get_entropy(entropy, action_class_dim):
  return -(entropy * math.log(1 / action_class_dim))


@gin.configurable
class DQNAgent(object):
  """An implementation of the DQN agent."""

  def __init__(self,
               sess,
               num_actions,
               adapted=False,
               adapted_tent=False,
               adapted_final=False,
               adapted_final_ln=False,
               is_augmentation=False,
               is_cross_entropy=False,
               update_with_batch=False,
               update_with_multiclass=False,
               is_decayLr=False,
               is_multi_step=False,
               entropy=0.8,
               kl_coef=1.0,
               apapted_moment=False,
               moment_update_tta=0.99,
               observation_shape=atari_lib.NATURE_DQN_OBSERVATION_SHAPE,
               observation_dtype=atari_lib.NATURE_DQN_DTYPE,
               stack_size=atari_lib.NATURE_DQN_STACK_SIZE,
               network=atari_lib.NatureDQNNetwork,
               gamma=0.99,
               update_horizon=1,
               min_replay_history=20000,
               update_period=4,
               target_update_period=8000,
               epsilon_fn=linearly_decaying_epsilon,
               epsilon_train=0.01,
               epsilon_eval=0.001,
               is_epsilon_adapted=False,
               epsilon_adapted_eval=0.01,
               epsilon_decay_period=250000,
               tf_device='/cpu:*',
               eval_mode=False,
               use_staging=False,
               max_tf_checkpoints_to_keep=100,
               optimizer=tf.compat.v1.train.RMSPropOptimizer(
                   learning_rate=0.00025,
                   decay=0.95,
                   momentum=0.0,
                   epsilon=0.00001,
                   centered=True),
               test_learning_lr=1e-6,
               summary_writer=None,
               summary_writing_frequency=500,
               allow_partial_reload=False,
               update_whole_episode=False,
               update_with_zero_return=False,
               update_with_bellman_equation=False):
    """Initializes the agent and constructs the components of its graph.

    Args:
      sess: `tf.compat.v1.Session`, for executing ops.
      num_actions: int, number of actions the agent can take at any state.
      observation_shape: tuple of ints describing the observation shape.
      observation_dtype: tf.DType, specifies the type of the observations. Note
        that if your inputs are continuous, you should set this to tf.float32.
      stack_size: int, number of frames to use in state stack.
      network: tf.Keras.Model, expecting 2 parameters: num_actions,
        network_type. A call to this object will return an instantiation of the
        network provided. The network returned can be run with different inputs
        to create different outputs. See
        dopamine.discrete_domains.atari_lib.NatureDQNNetwork as an example.
      gamma: float, discount factor with the usual RL meaning.
      update_horizon: int, horizon at which updates are performed, the 'n' in
        n-step update.
      min_replay_history: int, number of transitions that should be experienced
        before the agent begins training its value function.
      update_period: int, period between DQN updates.
      target_update_period: int, update period for the target network.
      epsilon_fn: function expecting 4 parameters:
        (decay_period, step, warmup_steps, epsilon). This function should return
        the epsilon value used for exploration during training.
      epsilon_train: float, the value to which the agent's epsilon is eventually
        decayed during training.
      epsilon_eval: float, epsilon used when evaluating the agent.
      epsilon_decay_period: int, length of the epsilon decay schedule.
      tf_device: str, Tensorflow device on which the agent's graph is executed.
      eval_mode: bool, True for evaluation and False for training.
      use_staging: bool, when True use a staging area to prefetch the next
        training batch, speeding training up by about 30%.
      max_tf_checkpoints_to_keep: int, the number of TensorFlow checkpoints to
        keep.
      optimizer: `tf.compat.v1.train.Optimizer`, for training the value
        function.
      summary_writer: SummaryWriter object for outputting training statistics.
        Summary writing disabled if set to None.
        May also be a str specifying the base directory, in which case the
        SummaryWriter will be created by the agent.
      summary_writing_frequency: int, frequency with which summaries will be
        written. Lower values will result in slower training.
      allow_partial_reload: bool, whether we allow reloading a partial agent
        (for instance, only the network parameters).
    """
    assert isinstance(observation_shape, tuple)
    logging.info('Creating %s agent with the following parameters:',
                 self.__class__.__name__)
    logging.info('\t gamma: %f', gamma)
    logging.info('\t update_horizon: %f', update_horizon)
    logging.info('\t min_replay_history: %d', min_replay_history)
    logging.info('\t update_period: %d', update_period)
    logging.info('\t target_update_period: %d', target_update_period)
    logging.info('\t epsilon_train: %f', epsilon_train)
    logging.info('\t epsilon_eval: %f', epsilon_eval)
    logging.info('\t epsilon_decay_period: %d', epsilon_decay_period)
    logging.info('\t tf_device: %s', tf_device)
    logging.info('\t use_staging: %s', use_staging)
    logging.info('\t optimizer: %s', optimizer)
    logging.info('\t max_tf_checkpoints_to_keep: %d',
                 max_tf_checkpoints_to_keep)

    self.num_actions = num_actions
    self.adapted = adapted
    self.moment_update_tta = moment_update_tta
    self.apapted_moment = apapted_moment
    self.adapted_tent = adapted_tent
    self.adapted_final = adapted_final
    self.adapted_final_ln = adapted_final_ln
    self.update_with_multiclass = update_with_multiclass
    self.is_augmentation = is_augmentation
    self.is_decayLr = is_decayLr
    self.is_multi_step = is_multi_step
    self.entropy = get_entropy(entropy=entropy, action_class_dim=self.num_actions)
    self.kl_coef = kl_coef
    self.observation_shape = tuple(observation_shape)
    self.observation_dtype = observation_dtype
    self.stack_size = stack_size
    self.network = network
    self.gamma = gamma
    self.update_horizon = update_horizon
    self.cumulative_gamma = math.pow(gamma, update_horizon)
    self.min_replay_history = min_replay_history
    self.target_update_period = target_update_period
    self.epsilon_fn = epsilon_fn
    self.epsilon_train = epsilon_train
    self.epsilon_eval = epsilon_eval
    self.epsilon_adapted_eval = epsilon_adapted_eval
    self.is_epsilon_adapted = is_epsilon_adapted
    self.epsilon_decay_period = epsilon_decay_period
    self.update_period = update_period
    self.eval_mode = eval_mode
    self.training_steps = 0
    self.test_step = 0
    self.adapted_test_step = 0
    self.high_entropy_step = 0
    self.optimizer = optimizer
    self.is_cross_entropy = is_cross_entropy
    self.replay_states = []
    self.states_list = []
    self.actions_list = []
    self.replay_size = 32
    self.batch_size = 32
    self.update_with_batch = update_with_batch
    self.test_learning_lr = test_learning_lr
    self.update_whole_episode = update_whole_episode
    self.update_with_zero_return = update_with_zero_return
    self.update_with_bellman_equation = update_with_bellman_equation
    test_optimizer=tf.compat.v1.train.AdamOptimizer(
                   learning_rate=self.test_learning_lr,
                   epsilon=0.0003125)
    self.test_optimizer = test_optimizer
    
    tf.compat.v1.disable_v2_behavior()
    if isinstance(summary_writer, str):  # If we're passing in directory name.
      self.summary_writer = tf.compat.v1.summary.FileWriter(summary_writer)
    else:
      self.summary_writer = summary_writer
    self.summary_writing_frequency = summary_writing_frequency
    self.allow_partial_reload = allow_partial_reload

    if sess is None:
      config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
      # Allocate only subset of the GPU memory as needed which allows for
      # running multiple agents/workers on the same GPU.
      config.gpu_options.allow_growth = True
      self._sess = tf.compat.v1.Session('', config=config)
    else:
      self._sess = sess

    # # store the state become a batch
    # self.batch_state = []

    with tf.device(tf_device):
      # Create a placeholder for the state input to the DQN network.
      # The last axis indicates the number of consecutive frames stacked.
      state_shape = (1,) + self.observation_shape + (stack_size,)
      self.state = np.zeros(state_shape)
      self.state_ph = tf.compat.v1.placeholder(
          self.observation_dtype, state_shape, name='state_ph')
      self.adapted_state = np.zeros(state_shape)
      self.adapted_state_ph = tf.compat.v1.placeholder(
          self.observation_dtype, state_shape, name='adapted_state_ph')
      self._replay = self._build_replay_buffer(use_staging)

      # ############################################################
      # # 2023/11/22: update with zero return
      # ############################################################
      # if self.update_with_zero_return:
      #   self.next_state_ph = tf.compat.v1.placeholder(
      #     self.observation_dtype, state_shape, name='next_state_ph')

      ############################################################
      # 2023/11/23: update with zero return, batch update
      ############################################################
      batch_state_shape = (None,) + self.observation_shape + (stack_size,)
      if self.update_with_zero_return:
        self.next_state_ph = tf.compat.v1.placeholder(
          self.observation_dtype, batch_state_shape, name='next_state_ph')
        self.batch_state_ph = tf.compat.v1.placeholder(
          self.observation_dtype, batch_state_shape, name='batch_state_ph')
      if self.update_with_bellman_equation:
        reward_shape = (None, )
        self.reward_ph = tf.compat.v1.placeholder(
          tf.float32, reward_shape, name='next_state_ph')

      ############################################################
      # 2023/11/03: create state input
      ############################################################
      if self.is_multi_step:
        # create multi variable
        for i in range(10):
          setattr(self, "multi_step_state_%d" % i, np.zeros(state_shape))

      ############################################################
      # 2023/11/03: reference state input
      ############################################################
      self.reference_state = np.zeros(state_shape)

      ############################################################
      # 2023/10/25: record random action
      ############################################################
      self.random_ph = tf.compat.v1.placeholder(
          tf.int32, name='random_ph')

      ############################################################
      # 2023/10/23: mixup augmenation data
      ############################################################
      # aug_state_shape = (self.replay_size,) + self.observation_shape + (stack_size,)
      # aug_action_shape = (self.replay_size,) + (self.num_actions, )
      aug_state_shape = (None,) + self.observation_shape + (stack_size,)
      aug_action_shape = (None,) + (self.num_actions, )
      self.aug_state_ph = tf.compat.v1.placeholder(
          self.observation_dtype, aug_state_shape, name='aug_state_ph')
      self.aug_action_ph = tf.compat.v1.placeholder(
          self.observation_dtype, aug_action_shape, name='aug_action_ph')
      ############################################################

      ############################################################
      # 2023/10/29: record image
      ############################################################
      single_state_shape = (1,) + self.observation_shape + (1,) 
      self.single_state_ph = tf.compat.v1.placeholder(
          self.observation_dtype, single_state_shape, name='single_state_ph')
      self.single_state_ph2 = tf.compat.v1.placeholder(
          self.observation_dtype, single_state_shape, name='single_state_ph2')
      self.single_state_ph3 = tf.compat.v1.placeholder(
          self.observation_dtype, single_state_shape, name='single_state_ph3')
      self.single_state_ph4 = tf.compat.v1.placeholder(
          self.observation_dtype, single_state_shape, name='single_state_ph4')
      ############################################################

      self._build_networks()

      self._train_op = self._build_train_op()

      if self.adapted:
        self._sync_online_to_offline_ops = self._build_online_to_offline_op()
      
      self._sync_qt_ops = self._build_sync_op()

    if self.summary_writer is not None:
      # All tf.summaries should have been defined prior to running this.
      self._merged_summaries = tf.compat.v1.summary.merge_all()

    if self.adapted:
      vars_too_save = [v for v in tf.compat.v1.global_variables() 
                       if not any(keyword in v.name for keyword in ['QTarget', 'Moment', 'TTA', 'beta1_power:1', 'beta2_power:1', 'beta1_power_1', "beta2_power_1", "Online/Conv/bias/RMSProp"])]
      # vars_too_save = [v for v in tf.compat.v1.global_variables() 
      #                  if not any(keyword in v.name for keyword in ['TTA', 'beta1_power:1', 'beta2_power:1', 'beta1_power_1', "beta2_power_1"])]
      var_map = vars_too_save
    else:
      var_map = atari_lib.maybe_transform_variable_names(
          tf.compat.v1.global_variables())
    
    # var_map = atari_lib.transform_variable_names(var_map)
    # print(tf.compat.v1.global_variables())
    # print(var_map)
    # exit()
    self._saver = tf.compat.v1.train.Saver(
        var_list=var_map, max_to_keep=max_tf_checkpoints_to_keep)

    # Variables to be initialized by the agent once it interacts with the
    # environment.
    self._observation = None
    self._last_observation = None

    if self.summary_writer is not None:
      self.summary_writer.add_graph(graph=tf.compat.v1.get_default_graph())
    self._sess.run(tf.compat.v1.global_variables_initializer())

  def _create_network(self, name):
    """Builds the convolutional network used to compute the agent's Q-values.

    Args:
      name: str, this name is passed to the tf.keras.Model and used to create
        variable scope under the hood by the tf.keras.Model.
    Returns:
      network: tf.keras.Model, the network instantiated by the Keras model.
    """
    if callable(self.network):
      network = self.network(self.num_actions, name=name)
    else:
      network_class = eval(self.network)
      network = network_class(self.num_actions, name=name)

    return network

  def _build_networks(self):
    """Builds the Q-value network computations needed for acting and training.

    These are:
      self.online_convnet: For computing the current state's Q-values.
      self.target_convnet: For computing the next state's target Q-values.
      self._net_outputs: The actual Q-values.
      self._q_argmax: The action maximizing the current state's Q-values.
      self._replay_net_outputs: The replayed states' Q-values.
      self._replay_next_target_net_outputs: The replayed next states' target
        Q-values (see Mnih et al., 2015 for details).
    """

    # _network_template instantiates the model and returns the network object.
    # The network object can be used to generate different outputs in the graph.
    # At each call to the network, the parameters will be reused.

    self.online_convnet = self._create_network(name='Online')
    self.target_convnet = self._create_network(name='Target')
    
    # add normalization update
    if self.network == "atari_helpers.QuantileBatchNormNetwork":
      if self.eval_mode == True:
        self._net_outputs = self.online_convnet(self.state_ph, training=False)
        # TODO(bellemare): Ties should be broken. They are unlikely to happen when
        # using a deep network, but may affect performance with a linear
        # approximation scheme.
        self._q_entropy = tf.reduce_mean(tent.softmax_entropy(self._net_outputs.q_values), axis=0)
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
          self.test_op = self.test_optimizer.minimize(self._q_entropy)

        self._q_argmax = tf.argmax(self._net_outputs.q_values, axis=1)[0]
        self._replay_net_outputs = self.online_convnet(self._replay.states, training=False)
        self._replay_next_target_net_outputs = self.target_convnet(
            self._replay.next_states, training=False)
      else:
        self._net_outputs = self.online_convnet(self.state_ph, training=True)
        # TODO(bellemare): Ties should be broken. They are unlikely to happen when
        # using a deep network, but may affect performance with a linear
        # approximation scheme.
        self._q_argmax = tf.argmax(self._net_outputs.q_values, axis=1)[0]
        self._replay_net_outputs = self.online_convnet(self._replay.states, training=True)
        self._replay_next_target_net_outputs = self.target_convnet(
            self._replay.next_states, training=True)
    else:
      self._net_outputs = self.online_convnet(self.state_ph)
      # TODO(bellemare): Ties should be broken. They are unlikely to happen when
      # using a deep network, but may affect performance with a linear
      # approximation scheme.
      self._q_argmax = tf.argmax(self._net_outputs.q_values, axis=1)[0]

      self._replay_net_outputs = self.online_convnet(self._replay.states)
      self._replay_next_target_net_outputs = self.target_convnet(
          self._replay.next_states)

      if self.adapted:
        ############################################################
        # 2023/10/30: compute the overflow entropy
        ############################################################
        self._max_eval_q_value = (self._net_outputs.q_values)[0][self._q_argmax]
        self.eval_norm_q_values = self._net_outputs.q_values - self._max_eval_q_value
        self.eval_exp_norm_q_values = tf.exp(self.eval_norm_q_values)
        self._eval_prob = tf.nn.softmax(self.eval_norm_q_values, axis=-1)
        self._eval_log_prob = self.eval_norm_q_values - tf.math.log(tf.reduce_sum(self.eval_exp_norm_q_values, axis=-1))
        self._q_entropy = -tf.reduce_sum(self._eval_prob * self._eval_log_prob)
        ############################################################

        # the entropy will overflow
        # self.online_norm_q_values, _ = tf.linalg.normalize(self._net_outputs.q_values, ord=2, axis=-1)
        # self._online_test_prob = tf.nn.softmax(self.online_norm_q_values, axis=-1)
        # self._q_entropy = -tf.reduce_sum(self._online_test_prob * tf.math.log(self._online_test_prob))

        # the entropy will overflow
        # self._q_entropy = tf.reduce_mean(tent.softmax_entropy(self._net_outputs.q_values), axis=0)
        self._q_value = (self._net_outputs.q_values)[0][self._q_argmax]
      
        # record eval summary
        if self.summary_writer is not None:
          with tf.compat.v1.variable_scope('eval'):
            # eval_q_value = tf.compat.v1.summary.histogram('eval_q_values_distribution', self._net_outputs.q_values)
            eval_q_value = tf.compat.v1.summary.scalar('eval_q_value', self._q_value) 
            eval_q_entropy = tf.compat.v1.summary.scalar('entropy', self._q_entropy)
            eval_q_action = tf.compat.v1.summary.scalar('action', self._q_argmax)
            self._merged_eval_summaries = tf.compat.v1.summary.merge([eval_q_value, eval_q_entropy, eval_q_action])

    # add test-time adaptation
    if self.adapted:
      self.test_online_convnet = self._create_network(name='TTA')
      if self.adapted_final:
        self.test_online_convnet = tent.finetune_final_layer(self.test_online_convnet)
      elif self.adapted_final_ln:
        self.test_online_convnet = tent.configure_final_ln_layer(self.test_online_convnet)
      else:
        self.test_online_convnet = tent.configure_model(self.test_online_convnet)

      self._test_net_outputs = self.test_online_convnet(self.adapted_state_ph)

      # L2 normalize the q network output
      # self.norm_q_values, _ = tf.linalg.normalize(self._test_net_outputs.q_values, ord=2, axis=-1)
      # self._test_prob = tf.nn.softmax(self.norm_q_values, axis=-1)
      # self._test_q_entropy = -tf.reduce_sum(self._test_prob * tf.math.log(self._test_prob))
      self._q_pro = self._test_net_outputs.probabilities

      # self._test_q_entropy = tf.reduce_mean(tent.softmax_entropy(self._test_net_outputs.q_values), axis=0)
      self._test_q_argmax = tf.argmax(self._test_net_outputs.q_values, axis=1)[0]

      # ############################################################
      # # 2023/11/22: update with zero return
      # ############################################################
      # if self.update_with_zero_return:
      #   self._next_state_outputs = self.test_online_convnet(self.next_state_ph)
      #   self._next_q_argmax = tf.argmax(self._next_state_outputs.q_values, axis=1)[0]
      #   self._next_q_value = (self._next_state_outputs.q_values)[0][self._next_q_argmax]
      #   self._q_value = (self._test_net_outputs.q_values)[0][self._test_q_argmax]
      #   self._difference_between_q_and_next_q = (self._q_value - 0.99 * self._next_q_value) ** 2
      #   self._difference_between_q_and_next_q_op = self.test_optimizer.minimize(tf.reduce_sum(self._difference_between_q_and_next_q))

      
      # ############################################################
      # # 2023/11/22: create the target network for tta
      # ############################################################
      # if self.update_with_zero_return:
      #   self.tta_target_convnet = self._create_network(name='QTarget')
      #   self._tta_next_target_net_outputs = self.tta_target_convnet(
      #         self.next_state_ph, training=True)
      #   self._next_q_argmax = tf.argmax(self._tta_next_target_net_outputs.q_values, axis=1)[0]
      #   self._next_q_value = (self._tta_next_target_net_outputs.q_values)[0][self._next_q_argmax]

      #   self._q_value = (self._test_net_outputs.q_values)[0][self._test_q_argmax]
      #   self._difference_between_q_and_next_q = (self._q_value - 0.99 * tf.stop_gradient(self._next_q_value)) ** 2
      #   self._difference_between_q_and_next_q_op = self.test_optimizer.minimize(tf.reduce_sum(self._difference_between_q_and_next_q))
      #   self._tta_copy_to_target_op = self._build_sync_tta_op()

      ############################################################
      # 2023/11/23: create the target network for tta and batch update
      ############################################################
      if self.update_with_zero_return:
        self.tta_target_convnet = self._create_network(name='QTarget')
        self._tta_next_target_net_outputs = self.tta_target_convnet(
              self.next_state_ph, training=True)
        self._next_q_argmax = tf.argmax(self._tta_next_target_net_outputs.q_values, axis=1)[0]
        self._next_q_value = (self._tta_next_target_net_outputs.q_values)[0][self._next_q_argmax]

        self._tta_convnet_outputs = self.test_online_convnet(self.batch_state_ph)
        self._tta_q_argmax = tf.argmax(self._tta_convnet_outputs.q_values, axis=1)[0]
        self._q_value = (self._tta_convnet_outputs.q_values)[0][self._tta_q_argmax]

        if self.update_with_bellman_equation:
          self._difference_between_q_and_next_q = (self._q_value - self.reward_ph -  0.99 * tf.stop_gradient(self._next_q_value)) ** 2
        self._difference_between_q_and_next_q = (self._q_value - 0.99 * tf.stop_gradient(self._next_q_value)) ** 2
        self._difference_between_q_and_next_q_op = self.test_optimizer.minimize(tf.reduce_sum(self._difference_between_q_and_next_q))
        self._tta_copy_to_target_op = self._build_sync_tta_op()

      ############################################################
      # 2023/10/30: compute the overflow entropy
      ############################################################
      self._max_q_value = (self._test_net_outputs.q_values)[0][self._test_q_argmax]
      self.norm_q_values = self._test_net_outputs.q_values - self._max_q_value
      self.exp_norm_q_values = tf.exp(self.norm_q_values)
      self._test_prob = tf.nn.softmax(self.norm_q_values, axis=-1)
      self._log_prob = self.norm_q_values - tf.math.log(tf.reduce_sum(self.exp_norm_q_values, axis=-1))
      self._test_q_entropy = -tf.reduce_sum(self._test_prob * self._log_prob)
      ############################################################

      if self.adapted_tent:
        self._test_op = self.test_optimizer.minimize(self._test_q_entropy)
      else:
        self._test_before_action_distribution = tf.stop_gradient(tf.nn.softmax(self._net_outputs.q_values, axis=-1))
        self._test_after_action_distribution = tf.nn.softmax(self._test_net_outputs.q_values, axis=-1)
        self._test_kl_divergence = tf.reduce_sum(self._test_before_action_distribution * ((tf.math.log(self._test_before_action_distribution + 1e-8)) - (tf.math.log(self._test_after_action_distribution + 1e-8))))
        self._test_loss = self.kl_coef * self._test_kl_divergence + self._test_q_entropy
        self._test_op = self.test_optimizer.minimize(self._test_loss)

      ############################################################
      # 2023/10/24: mixup augmenation data
      ############################################################
      # get pesudo label 
      self._pesudo_max_q_values = tf.reduce_max(self._test_net_outputs.q_values, axis=-1, keepdims=True)
      self._pesudo_norm_q_values = self._test_net_outputs.q_values - self._pesudo_max_q_values
      self._pesudo_aug_test_prob = tf.nn.softmax(self._pesudo_norm_q_values, axis=-1)
      # self._pesudo_norm_q_values, _ = tf.linalg.normalize(self._test_net_outputs.q_values, ord=2, axis=-1)
      # self._pesudo_aug_test_prob = tf.nn.softmax(self._pesudo_norm_q_values, axis=-1)
      shape = tf.shape(self._pesudo_aug_test_prob)
      max_value = tf.reduce_max(self._pesudo_aug_test_prob, axis=1)
      self._action_label = tf.one_hot(tf.cast(max_value, tf.int64), shape[1], axis=-1) 
      # update with the augmentation data
      self._aug_test_net_outputs = self.test_online_convnet(self.aug_state_ph)

      ############################################################
      # 2023/10/31 15:06 avoid data overflow 
      ############################################################
      self.batch_max_value = tf.reduce_max(self._aug_test_net_outputs.q_values, axis=-1, keepdims=True)
      self.batch_norm_q_values = self._aug_test_net_outputs.q_values - self.batch_max_value
      self.batch_exp_norm_q_values = tf.exp(self.batch_norm_q_values)
      self._batch_test_prob = tf.nn.softmax(self.batch_norm_q_values, axis=-1)
      self._batch_log_prob = self.batch_norm_q_values - tf.math.log(tf.reduce_sum(self.batch_exp_norm_q_values, axis=-1, keepdims=True))

      ############################################################
      # 2023/10/31 20:04 add multi class constrain
      ############################################################
      if self.update_with_multiclass:
        self.batch_mean_q_values = tf.reduce_mean(self.batch_norm_q_values, axis=0, keepdims=True)
        self.batch_mean_max_value = tf.reduce_max(self.batch_mean_q_values, axis=-1, keepdims=True)
        self.mean_norm_q_values = self.batch_mean_q_values - self.batch_mean_max_value
        self.mean_exp_norm_q_values = tf.exp(self.mean_norm_q_values)
        self._mean_test_prob = tf.nn.softmax(self.mean_norm_q_values, axis=-1)
        self._mean_log_prob = self.mean_norm_q_values - tf.math.log(tf.reduce_sum(self.mean_exp_norm_q_values, axis=-1))
        self._mean_test_q_entropy = tf.reduce_sum(self._mean_test_prob * self._mean_log_prob)
      ############################################################

      # self._aug_norm_q_values, _ = tf.linalg.normalize(self._aug_test_net_outputs.q_values, ord=2, axis=-1)
      # self.softmax_aug_data = tf.nn.softmax(self._aug_norm_q_values, axis=-1)
      # build the batch entropy minimize loss
      if self.update_with_batch:
        #self._batch_entropy_minimize_loss = -tf.reduce_sum(self.softmax_aug_data * tf.math.log(self.softmax_aug_data))
        self._batch_entropy_minimize_loss = -tf.reduce_sum(self._batch_test_prob * self._batch_log_prob)
        self._batch_test_op = self.test_optimizer.minimize(tf.reduce_sum(self._batch_entropy_minimize_loss))

        ############################################################
        # 2023/12/05 16:47 add kl loss for training 
        ############################################################
        self._batch_norm_net_outputs = self.online_convnet(self.aug_state_ph)
        self.batch_max_value_before = tf.reduce_max(self._batch_norm_net_outputs.q_values, axis=-1, keepdims=True)
        self.batch_norm_q_values_before = self._batch_norm_net_outputs.q_values - self.batch_max_value_before
        self.batch_exp_norm_q_values_before = tf.exp(self.batch_norm_q_values_before)
        self._batch_before_log_prob = self.batch_norm_q_values_before - tf.math.log(tf.reduce_sum(self.batch_exp_norm_q_values_before, axis=-1, keepdims=True))
        self._batch_before_stop_gradient_log_prob = tf.stop_gradient(self._batch_before_log_prob)
        self._batch_before_stop_gradient_prob = tf.stop_gradient(tf.nn.softmax(self._batch_norm_net_outputs.q_values, axis=-1))

        self.batch_max_value_after = tf.reduce_max(self._aug_test_net_outputs.q_values, axis=-1, keepdims=True)
        self.batch_norm_q_values_after = self._aug_test_net_outputs.q_values - self.batch_max_value_after
        self.batch_exp_norm_q_values_after= tf.exp(self.batch_norm_q_values_after)
        self._batch_after_log_prob = self.batch_norm_q_values_after - tf.math.log(tf.reduce_sum(self.batch_exp_norm_q_values_after, axis=-1, keepdims=True))

        self._batch_kl_divergence = tf.reduce_sum(self._batch_before_stop_gradient_prob * (self._batch_before_stop_gradient_log_prob - self._batch_after_log_prob))
        self._batch_test_with_kl_op = self.test_optimizer.minimize(tf.reduce_sum(self.kl_coef * self._batch_kl_divergence + self._batch_entropy_minimize_loss))
        ############################################################

      # build the cross entropy loss
      # self.neg_soft_log = tf.cast(self.aug_action_ph, tf.float32) * tf.math.log(self.softmax_aug_data)
      self.neg_soft_log = tf.cast(self.aug_action_ph, tf.float32) * self._batch_log_prob
      self._aug_cross_entropy_loss = - tf.reduce_sum(self.neg_soft_log)
      self._aug_test_op = self.test_optimizer.minimize(tf.reduce_sum(self._aug_cross_entropy_loss))
      ############################################################

      ############################################################
      # 2023/10/28 15:37 
      ############################################################
      # compute the gradients of the network
      self._tta_gradient = tf.gradients(self._test_q_entropy, self.test_online_convnet.trainable_variables)
      ############################################################
      
      # recore eval summary
      if self.summary_writer is not None:
        ############################################################
        # 2023/10/28 15:37 
        ############################################################
        with tf.compat.v1.variable_scope('tta_gradient'):
          gradient_list = []
          for i in range(len(self._tta_gradient)):
            gradient_list.append(tf.compat.v1.summary.scalar('layer_' + str(i) + '_gradient', tf.norm(self._tta_gradient[i])))
          self._merged_tta_gradient_summaries = tf.compat.v1.summary.merge(gradient_list)
        ############################################################

        ############################################################
        # 2023/10/29 21:54 
        ############################################################
        with tf.compat.v1.variable_scope('image'):
          atari_image = tf.compat.v1.summary.image("image", self.single_state_ph)
          atari_image2 = tf.compat.v1.summary.image("image2", self.single_state_ph2)
          atari_image3 = tf.compat.v1.summary.image("image3", self.single_state_ph3)
          atari_image4 = tf.compat.v1.summary.image("image4", self.single_state_ph4)
          self._merged_image_summaries = tf.compat.v1.summary.merge([atari_image, atari_image2, atari_image3, atari_image4])
        ############################################################

        with tf.compat.v1.variable_scope('tta'):
          # tta_q_value = tf.compat.v1.summary.histogram('tta_q_values_distribution', self._test_net_outputs.q_values)
          tta_q_value = tf.compat.v1.summary.scalar('tta_q_value', (self._test_net_outputs.q_values)[0][self._test_q_argmax])
          tta_entorpy = tf.compat.v1.summary.scalar('entorpy', self._test_q_entropy)
          tta_action = tf.compat.v1.summary.scalar('action', self._test_q_argmax)
          # tta_state = tf.compat.v1.summary.tensor_summary("tta_state", self.adapted_state_ph)
          # tta_entropy_before = tf.compat.v1.summary.scalar('before_entropy', self._test_q_entropy)
          # tta_entropy_after = tf.compat.v1.summary.scalar('after_entropy', self._test_q_entropy)
          # self._merged_tta_tensor_summaries = tf.compat.v1.summary.merge([tta_entropy_before])
          self._merged_tta_summaries = tf.compat.v1.summary.merge([tta_q_value, tta_entorpy, tta_action])
          # self._merged_tta_summaries = tf.compat.v1.summary.merge([tta_q_value, tta_entorpy])
          # self._merged_tta_entropy_before_summaries = tf.compat.v1.summary.merge([tta_entropy_before])
          # self._merged_tta_entropy_after_summaries = tf.compat.v1.summary.merge([tta_entropy_after])

          # # record extra entropy
          # eval_q_entropy = tf.compat.v1.summary.scalar('eval_entorpy', self._q_entropy)
          # tta_q_entropy = tf.compat.v1.summary.scalar('tta_entorpy', self._test_q_entropy)
          # self._merged_entropy_summaries = tf.compat.v1.summary.merge([eval_q_entropy, tta_q_entropy])

        ############################################################
        # 2023/10/25: record random step
        ############################################################
        with tf.compat.v1.variable_scope('random'):
          tta_random_action = tf.compat.v1.summary.scalar('random_action', self.random_ph)
          self._merged_random_summaries = tf.compat.v1.summary.merge([tta_random_action])

    # add moment update
    if self.apapted_moment:
      self.moment_convnet = self._create_network(name='Moment')
      self._moment_net_outputs = self.moment_convnet(self.adapted_state_ph)
      self._copy_tta_to_moment_op = self._build_tta_to_moment_op()
      self._tta_moment_update_op = self._build_moment_update_op()

  def _build_replay_buffer(self, use_staging):
    """Creates the replay buffer used by the agent.

    Args:
      use_staging: bool, if True, uses a staging area to prefetch data for
        faster training.

    Returns:
      A WrapperReplayBuffer object.
    """
    return circular_replay_buffer.WrappedReplayBuffer(
        observation_shape=self.observation_shape,
        stack_size=self.stack_size,
        use_staging=use_staging,
        update_horizon=self.update_horizon,
        gamma=self.gamma,
        observation_dtype=self.observation_dtype.as_numpy_dtype)

  def _build_target_q_op(self):
    """Build an op used as a target for the Q-value.

    Returns:
      target_q_op: An op calculating the Q-value.
    """
    # Get the maximum Q-value across the actions dimension.
    replay_next_qt_max = tf.reduce_max(
        self._replay_next_target_net_outputs.q_values, 1)
    # Calculate the Bellman target value.
    #   Q_t = R_t + \gamma^N * Q'_t+1
    # where,
    #   Q'_t+1 = \argmax_a Q(S_t+1, a)
    #          (or) 0 if S_t is a terminal state,
    # and
    #   N is the update horizon (by default, N=1).
    return self._replay.rewards + self.cumulative_gamma * replay_next_qt_max * (
        1. - tf.cast(self._replay.terminals, tf.float32))

  def _build_train_op(self):
    """Builds a training op.

    Returns:
      train_op: An op performing one step of training from replay data.
    """
    replay_action_one_hot = tf.one_hot(
        self._replay.actions, self.num_actions, 1., 0., name='action_one_hot')
    replay_chosen_q = tf.reduce_sum(
        self._replay_net_outputs.q_values * replay_action_one_hot,
        axis=1,
        name='replay_chosen_q')

    target = tf.stop_gradient(self._build_target_q_op())
    loss = tf.compat.v1.losses.huber_loss(
        target, replay_chosen_q, reduction=tf.losses.Reduction.NONE)
    if self.summary_writer is not None:
      with tf.compat.v1.variable_scope('Losses'):
        merge_train_summary = tf.compat.v1.summary.scalar('HuberLoss', tf.reduce_mean(loss))

    # add bn update operation into graph
    if self.network == "atari_helpers.QuantileBatchNormNetwork":
      update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        train_op = self.optimizer.minimize(tf.reduce_mean(loss))
    else:
      train_op = self.optimizer.minimize(tf.reduce_mean(loss))
    return train_op, loss, merge_train_summary
    # return train_op

  def _build_sync_op(self):
    """Builds ops for assigning weights from online to target network.

    Returns:
      ops: A list of ops assigning weights from online to target network.
    """
    # Get trainable variables from online and target DQNs
    sync_qt_ops = []
    scope = tf.compat.v1.get_default_graph().get_name_scope()
    trainables_online = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
        scope=os.path.join(scope, 'Online'))
    trainables_target = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
        scope=os.path.join(scope, 'Target'))

    for (w_online, w_target) in zip(trainables_online, trainables_target):
      # Assign weights from online to target network.
      sync_qt_ops.append(w_target.assign(w_online, use_locking=True))
    return sync_qt_ops
  
  def _build_sync_tta_op(self):
    """Builds ops for assigning weights from online to target network.

    Returns:
      ops: A list of ops assigning weights from online to target network.
    """
    # Get trainable variables from online and target DQNs
    sync_qt_ops = []
    scope = tf.compat.v1.get_default_graph().get_name_scope()
    trainables_tta = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
        scope=os.path.join(scope, 'TTA'))
    trainables_target = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
        scope=os.path.join(scope, 'QTarget'))

    for (w_tta, w_target) in zip(trainables_tta, trainables_target):
      # Assign weights from online to target network.
      sync_qt_ops.append(w_target.assign(w_tta, use_locking=True))
    return sync_qt_ops
  
  def _build_moment_update_op(self):
    """Builds ops for assigning weights from online to target network.

    Returns:
      ops: A list of ops assigning weights from online to target network.
    """
    # Get trainable variables from online and target DQNs
    sync_qt_ops = []
    scope = tf.compat.v1.get_default_graph().get_name_scope()
    trainables_tta = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
        scope=os.path.join(scope, 'TTA'))
    trainables_moment = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
        scope=os.path.join(scope, 'Moment'))
    
    # moment update parameters
    for (w_tta, w_moment) in zip(trainables_tta, trainables_moment):
      # Assign weights from online to target network.
      sync_qt_ops.append(w_tta.assign(w_moment * self.moment_update_tta + (1 - self.moment_update_tta) * w_tta, use_locking=True))
    return sync_qt_ops

  def _build_tta_to_moment_op(self):
    """Builds ops for assigning weights from online to target network.

    Returns:
      ops: A list of ops assigning weights from online to target network.
    """
    # Get trainable variables from online and target DQNs
    sync_qt_ops = []
    scope = tf.compat.v1.get_default_graph().get_name_scope()
    trainables_tta = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
        scope=os.path.join(scope, 'TTA'))
    trainables_moment = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
        scope=os.path.join(scope, 'Moment'))
    
    for (w_tta, w_moment) in zip(trainables_tta, trainables_moment):
      # Assign weights from online to target network.
      sync_qt_ops.append(w_moment.assign(w_tta, use_locking=True))

    return sync_qt_ops 
  
  def _build_online_to_offline_op(self):
    """Builds ops for assigning weights from online to target network.

    Returns:
      ops: A list of ops assigning weights from online to target network.
    """
    # Get trainable variables from online and target DQNs
    # for i, layer in enumerate(self.online_convnet.layers): 
    #   self.test_online_convnet.layers[i].set_weights(copy.deepcopy(layer.get_weights()))
      # self.test_online_convnet.layers[i].set_bi
      # print(layer.get_weights())

    sync_online_to_offline_ops = []
    scope = tf.compat.v1.get_default_graph().get_name_scope()
    trainables_online = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
        scope=os.path.join(scope, 'Online'))
    trainables_tta = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
        scope=os.path.join(scope, 'TTA'))

    for (w_online, w_tta) in zip(trainables_online, trainables_tta):
      # Assign weights from online to target network.
      sync_online_to_offline_ops.append(w_tta.assign(w_online, use_locking=True))
    return sync_online_to_offline_ops

  # # copy parameters from online network to online-tta network
  # def copy_network_params(self):
  #   for i, layer in enumerate(self.online_convnet.layers): 
  #     self.test_online_convnet.layers[i].set_weights(copy.deepcopy(layer.get_weights()))
  #     print(layer.get_weights())

  def update_with_bellman(self, observation, action, reward, next_observation):
    replay_action_one_hot = tf.one_hot(
        self._replay.actions, self.num_actions, 1., 0., name='adapted_action_one_hot')
    replay_chosen_q = tf.reduce_sum(
        self._replay_net_outputs.q_values * replay_action_one_hot,
        axis=1,
        name='replay_chosen_q')

    target = tf.stop_gradient(self._build_target_q_op())
    loss = tf.compat.v1.losses.huber_loss(
        target, replay_chosen_q, reduction=tf.losses.Reduction.NONE)
    if self.summary_writer is not None:
      with tf.compat.v1.variable_scope('Losses'):
        merge_train_summary = tf.compat.v1.summary.scalar('HuberLoss', tf.reduce_mean(loss))

    # add bn update operation into graph
    if self.network == "atari_helpers.QuantileBatchNormNetwork":
      update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        train_op = self.optimizer.minimize(tf.reduce_mean(loss))
    else:
      train_op = self.optimizer.minimize(tf.reduce_mean(loss))

  def begin_episode(self, observation):
    """Returns the agent's first action for this episode.

    Args:
      observation: numpy array, the environment's initial observation.

    Returns:
      int, the selected action.
    """
    self._reset_state()
    self._record_observation(observation)

    if not self.eval_mode:
      self._train_step()

    self.action = self._select_action()
    return self.action

  def begin_reference_episode(self, observation):
    """Returns the agent's first action for this episode.

    Args:
      observation: numpy array, the environment's initial observation.

    Returns:
      int, the selected action.
    """
    self._reset_reference_state()
    self._record_reference_observation(observation)

    if not self.eval_mode:
      self._train_step()

    self.action = self._select_reference_action()
    return self.action
  
  def begin_single_step_episode(self, observation, i):
    """Returns the agent's first action for this episode.

    Args:
      observation: numpy array, the environment's initial observation.

    Returns:
      int, the selected action.
    """
    self._reset_single_step_state(i)
    self._record_single_step_observation(observation, i)

    if not self.eval_mode:
      self._train_step()

    action, max_q_value = self._select_single_step_action(i)
    return action, max_q_value
  
  def begin_single_step_episode_by_entropy(self, observation, i):
    """Returns the agent's first action for this episode.

    Args:
      observation: numpy array, the environment's initial observation.

    Returns:
      int, the selected action.
    """
    self._reset_single_step_state(i)
    self._record_single_step_observation(observation, i)

    if not self.eval_mode:
      self._train_step()

    action, max_q_value, entropy = self._select_single_step_action_by_entropy(i)
    return action, max_q_value, entropy
  
  def adapted_begin_episode_multi_step(self, observation, i):
    """Returns the agent's first action for this episode.

    Args:
      observation: numpy array, the environment's initial observation.

    Returns:
      int, the selected action.
    """
    self._reset_multi_step_adapted_state(i)
    self._record_multi_step_adapted_observation(observation, i)
    self.action, max_q_value = self._select_adapted_multi_step_action(i)
    return self.action, max_q_value
  
  def adapted_begin_episode_single_step(self, observation, i):
    """Returns the agent's first action for this episode.

    Args:
      observation: numpy array, the environment's initial observation.

    Returns:
      int, the selected action.
    """
    self._reset_single_step_state(i)
    self._record_single_step_observation(observation, i)
    self.action, max_q_value = self._select_adapted_multi_step_action(i)
    return self.action, max_q_value
  
  def adapted_begin_episode(self, observation):
    """Returns the agent's first action for this episode.

    Args:
      observation: numpy array, the environment's initial observation.

    Returns:
      int, the selected action.
    """
    self._reset_adapted_state()
    self._record_adapted_observation(observation)

    if not self.eval_mode:
      self._train_step()

    self.action = self._select_adapted_action()
    return self.action

  def step(self, reward, observation):
    """Records the most recent transition and returns the agent's next action.

    We store the observation of the last time step since we want to store it
    with the reward.

    Args:
      reward: float, the reward received from the agent's most recent action.
      observation: numpy array, the most recent observation.

    Returns:
      int, the selected action.
    """
    self._last_observation = self._observation
    self._record_observation(observation)

    if not self.eval_mode:
      self._store_transition(self._last_observation, self.action, reward, False)
      self._train_step()

    self.action = self._select_action()
    return self.action
  
  def reference_step(self, reward, observation):
    """Records the most recent transition and returns the agent's next action.

    We store the observation of the last time step since we want to store it
    with the reward.

    Args:
      reward: float, the reward received from the agent's most recent action.
      observation: numpy array, the most recent observation.

    Returns:
      int, the selected action.
    """
    self._record_reference_observation(observation)
    action = self._select_reference_action()
    return action

  def adapted_single_step(self, reward, observation, i):
    """Records the most recent transition and returns the agent's next action.

    We store the observation of the last time step since we want to store it
    with the reward.

    Args:
      reward: float, the reward received from the agent's most recent action.
      observation: numpy array, the most recent observation.

    Returns:
      int, the selected action.
    """
    self._record_single_step_observation(observation, i)
    action, max_value = self._select_single_step_action(i)
    return action, max_value
  
  def adapted_single_step_by_entropy(self, reward, observation, i):
    """Records the most recent transition and returns the agent's next action.

    We store the observation of the last time step since we want to store it
    with the reward.

    Args:
      reward: float, the reward received from the agent's most recent action.
      observation: numpy array, the most recent observation.

    Returns:
      int, the selected action.
    """
    self._record_single_step_observation(observation, i)
    action, max_value, entropy = self._select_single_step_action_by_entropy(i)
    return action, max_value, entropy
  
  def adapted_multi_step(self, reward, observation, i):
    """Records the most recent transition and returns the agent's next action.

    Args:
      reward: float, the reward received from the agent's most recent action.
      observation: numpy array, the most recent observation.

    Returns:
      int, the selected action.
    """
    self._record_multi_step_adapted_observation(observation, i)
    action, max_q_value = self._select_adapted_multi_step_action(i)
    return action, max_q_value

  def end_episode(self, reward):
    """Signals the end of the episode to the agent.

    We store the observation of the current time step, which is the last
    observation of the episode.

    Args:
      reward: float, the last reward from the environment.
    """
    if not self.eval_mode:
      self._store_transition(self._observation, self.action, reward, True)

  def mixup_sample(self, features1, labels1, features2, labels2, alpha):
    lam = np.random.beta(alpha, alpha)
    mixed_features = lam * features1 + (1 - lam) * features2
    mixed_labels = lam * labels1 + (1 - lam) * labels2
    return mixed_features, mixed_labels

  def mixup_batch(self, features, labels, alpha):
    num_samples = features.shape[0]
    indices = np.arange(num_samples, dtype=np.int32)
    shuffled_indices = np.random.permutation(indices)
    mixed_features, mixed_labels = self.mixup_sample(
        features, labels, features[shuffled_indices], labels[shuffled_indices], alpha)
    return mixed_features, mixed_labels
  
  #########################################################################
  # 2023/10/30 define cosine learning rate decay
  #########################################################################
  def cosine_decay_lr(self, step, initial_lr, total_steps):
    lr = 0.5 * initial_lr * (1 + math.cos(math.pi * step / total_steps))
    return lr
  #########################################################################

  def _update_model_with_max_q_value_step(self, state_list):
    batch_states = np.squeeze(np.stack(state_list))
    self._sess.run(self._batch_test_op, {self.aug_state_ph: batch_states})

  def _update_model_with_episode(self, state_list):
    update_list = []
    for state in state_list:
      if self.update_whole_episode:
        update_list.append(state)
      else:
        entropy = self._sess.run(self._test_q_entropy, {self.adapted_state_ph: state})
        if entropy < self.entropy:
          update_list.append(state)

    #########################################################################
    # 2023/12/04 update with the batch 32
    #########################################################################
    for i in range(0, len(update_list), 32):
      if i + 32 < len(update_list):
        batch_states = np.squeeze(np.stack(update_list[i:i+32]), axis=1)
        self._sess.run(self._batch_test_op, {self.aug_state_ph: batch_states})
      else:
        batch_states = np.squeeze(np.stack(update_list[i:]), axis=1)
        self._sess.run(self._batch_test_op, {self.aug_state_ph: batch_states})
    # batch_states = np.squeeze(np.stack(update_list), axis=1)
    # self._sess.run(self._batch_test_op, {self.aug_state_ph: batch_states})

  def _update_model_with_kl_loss(self, state_list):
    update_list = []
    for state in state_list:
      if self.update_whole_episode:
        update_list.append(state)
      else:
        entropy = self._sess.run(self._test_q_entropy, {self.adapted_state_ph: state})
        if entropy < self.entropy:
          update_list.append(state)

    #########################################################################
    # 2023/12/04 update with the batch 32
    #########################################################################
    for i in range(0, len(update_list), 32):
      if i + 32 < len(update_list):
        batch_states = np.squeeze(np.stack(update_list[i:i+32]), axis=1)
        self._sess.run(self._batch_test_with_kl_op, {self.aug_state_ph: batch_states})
      else:
        batch_states = np.squeeze(np.stack(update_list[i:]), axis=1)
        self._sess.run(self._batch_test_with_kl_op, {self.aug_state_ph: batch_states})

  def _select_adapted_multi_step_action(self, i):
    """Select an action from the set of available actions.

    Chooses an action randomly with probability self._calculate_epsilon(), and
    otherwise acts greedily according to the current Q-value estimates.

    Returns:
       int, the selected action.
    """
    # Choose the action with highest Q-value at the current state.
    if self.eval_mode:
      if self.is_epsilon_adapted:
        epsilon = self.epsilon_adapted_eval # default: 0.001      
      else:
        epsilon = self.epsilon_eval # default: 0.001         

    else:
      epsilon = self.epsilon_fn(
          self.epsilon_decay_period,  # 250000
          self.training_steps,        # 0
          self.min_replay_history,    # 20000
          self.epsilon_train)         # 0.01
    if random.random() <= epsilon:
      # Choose a random action with probability epsilon.
      action = random.randint(0, self.num_actions - 1)
      if self.summary_writer is not None:
        summary = self._sess.run(self._merged_random_summaries, {self.random_ph: action})
        self.summary_writer.add_summary(summary, self.adapted_test_step)
        
      return action, 0
    
    else:
      multi_step_state = getattr(self, "multi_step_state_%d" % i)
      action = self._sess.run(self._test_q_argmax, {self.adapted_state_ph: multi_step_state})
      entropy = self._sess.run(self._test_q_entropy, {self.adapted_state_ph: multi_step_state})
      max_q_value = self._sess.run(self._max_q_value, {self.adapted_state_ph: multi_step_state})

      return action, max_q_value
    
  def update_network_with_zero_return(self, obs, next_obs, reward):
    if self.update_with_bellman_equation:
      self._sess.run(self._difference_between_q_and_next_q_op, {self.batch_state_ph: obs, self.next_state_ph: next_obs, self.reward_ph: reward})
    else:
      self._sess.run(self._difference_between_q_and_next_q_op, {self.batch_state_ph: obs, self.next_state_ph: next_obs})

  def update_network_with_zero_return_update_with_batch(self, state_list, next_state_list):
    batch_state = np.squeeze(np.stack(state_list), axis=1)
    next_batch_state = np.squeeze(np.stack(next_state_list), axis=1)
    self._sess.run(self._difference_between_q_and_next_q_op, {self.batch_state_ph: batch_state, self.next_state_ph: next_batch_state})

  def copy_tta_to_target(self):
    self._sess.run(self._tta_copy_to_target_op)
  
  def _select_adapted_action(self):
    """Select an action from the set of available actions.

    Chooses an action randomly with probability self._calculate_epsilon(), and
    otherwise acts greedily according to the current Q-value estimates.

    Returns:
       int, the selected action.
    """
    if self.eval_mode:
      if self.is_epsilon_adapted:
        epsilon = self.epsilon_adapted_eval # default: 0.001      
      else:
        epsilon = self.epsilon_eval # default: 0.001         

    else:
      epsilon = self.epsilon_fn(
          self.epsilon_decay_period,  # 250000
          self.training_steps,        # 0
          self.min_replay_history,    # 20000
          self.epsilon_train)         # 0.01
    if random.random() <= epsilon:
      # Choose a random action with probability epsilon.
      action = random.randint(0, self.num_actions - 1)
      if self.summary_writer is not None:
        summary = self._sess.run(self._merged_random_summaries, {self.random_ph: action})
        self.summary_writer.add_summary(summary, self.adapted_test_step)

      return action
    else:
      # # 创建一个初始值为 0 的 1 维张量（大小为 84 x 84）
      # self.adapted_state = np.ones((1, 84, 84, 4))
      # self._sess.run(self._sync_online_to_offline_ops)
      
      # print("adapted action: ", self._sess.run(self._test_net_outputs.q_values, {self.adapted_state_ph: self.adapted_state}))
      # print("online  action: ", self._sess.run(self._net_outputs.q_values, {self.state_ph: self.adapted_state}))
      # print("adapted probabilities: ", self._sess.run(self._test_net_outputs.probabilities[0][1][0:4], {self.adapted_state_ph: self.adapted_state}))
      # print("online  probabilities: ", self._sess.run(self._net_outputs.probabilities[0][1][0:4], {self.state_ph: self.adapted_state}))
      # print("adapted logits: ", self._sess.run(self._test_net_outputs.logits[0][1][0:4], {self.adapted_state_ph: self.adapted_state}))
      # print("online  logits: ", self._sess.run(self._net_outputs.logits[0][1][0:4], {self.state_ph: self.adapted_state}))

      # # tta_layer = self.test_online_convnet.layers[2].get_weights()[0]
      # # target_layer = self.online_convnet.layers[2].get_weights()[0]
      # # print("adapted parameter: {}, online parameter: {}".format(tta_layer[0][0][34][45], target_layer[0][0][34][45]))
      # # print("adapted parameter: {}, online parameter: {}".format(tta_layer[0][1][23][34], target_layer[0][1][23][34]))
      # # print("adapted parameter: {}, online parameter: {}".format(tta_layer[1][0][12][54], target_layer[1][0][12][54]))
      # exit()

      # self.batch_state.append(self.adapted_state)
      # if len(self.batch_state) == 32:
      #   batch_state = np.array(self.batch_state)
      #   action = self._sess.run(self._test_q_argmax, {self.adapted_state_ph: batch_state})
      #   print('*' * 100)
      #   print(action)
      #   print('*' * 100)

      # Choose the action with highest Q-value at the current state.
      action = self._sess.run(self._test_q_argmax, {self.adapted_state_ph: self.adapted_state})
      entropy = self._sess.run(self._test_q_entropy, {self.adapted_state_ph: self.adapted_state})

      # output = self._sess.run(self._test_net_outputs.q_values, {self.adapted_state_ph: self.adapted_state})
      # log_pro = self._sess.run(self._log_prob, {self.adapted_state_ph: self.adapted_state})
      # normal_output = self._sess.run(self.norm_q_values, {self.adapted_state_ph: self.adapted_state})
      # print("output: ", output)
      # print("log_pro: ", log_pro)
      # # print("normal output: ", normal_output)
      # print("adapted pro: ", self._sess.run(self._test_prob, {self.adapted_state_ph: self.adapted_state}))
      # print("pro: ", self._sess.run(self._q_pro, {self.adapted_state_ph: self.adapted_state}))
      # print("entropy: ", entropy)
      if self.summary_writer is not None:
        summary = self._sess.run(self._merged_tta_summaries, {self.adapted_state_ph: self.adapted_state})
        self.summary_writer.add_summary(summary, self.adapted_test_step)
        summary2 = self._sess.run(self._merged_tta_gradient_summaries, {self.adapted_state_ph: self.adapted_state})
        self.summary_writer.add_summary(summary2, self.adapted_test_step)
        # summary3 = self._sess.run(self._merged_image_summaries, {self.single_state_ph: self.adapted_state[0],
        #                                                          self.single_state_ph2: self.adapted_state[1],
        #                                                          self.single_state_ph3: self.adapted_state[2],
        #                                                          self.single_state_ph4: self.adapted_state[3]})
        # self.summary_writer.add_summary(summary3, self.adapted_test_step)
        self.adapted_test_step += 1

        #########################################################################
        # 2023/10/30 decay the learning rate
        #########################################################################
        if self.is_decayLr:
          learning_rate = self.cosine_decay_lr(self.adapted_test_step, self.test_learning_lr, 125000)
          self.test_optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=learning_rate,
                    epsilon=0.0003125)
        #########################################################################

        # decay the entropy threshold
        # self.entropy = 2.19 - 0.01 * (self.adapted_test_step // 15000)
        # # decay the kl coef
        # self.kl_coef = 1.5 - 0.1 * (self.adapted_test_step // 15000)

      # gradient_values = self._sess.run(self._tta_gradient, {self.adapted_state_ph: self.adapted_state})
      # for var, grad in zip(self.test_online_convnet.trainable_variables, gradient_values):
      #   if grad is not None:
      #     # print(f"Variable {var.name} has mean {tf.reduce_mean(grad)}, std {tf.reduce_std(grad)}")
      #     print(f"Variable {var.name}")

      # print("tta Q value: ", self._sess.run(self._test_net_outputs.q_values, {self.adapted_state_ph: self.adapted_state}))
      # test-time adaptation
      if self.adapted_tent:
        if entropy < self.entropy:
          self.states_list.append(self.adapted_state)
          # action_label = self._sess.run(self._action_label, {self.adapted_state_ph: self.adapted_state})
          action_label = self._sess.run(self._pesudo_aug_test_prob, {self.adapted_state_ph: self.adapted_state})
          self.actions_list.append(action_label)
          
          if self.update_with_batch:
            # if len(self.states_list) == 32:
            if len(self.states_list) > 32:
              # transform tensor
              # state shape: （32， 84， 84， 4); action shape: (32, 9)
              batch_states = np.squeeze(np.stack(self.states_list), axis=1)
              indices = np.random.choice(batch_states.shape[0], size=32, replace=False)
              update_batch_states = batch_states[indices]
              self._sess.run(self._batch_test_op, {self.aug_state_ph: update_batch_states})

              # update with multi class
              if self.update_with_multiclass:
                self._sess.run(self._mean_test_q_entropy, {self.aug_state_ph: batch_states})
          else:
            self._sess.run(self._test_op, {self.adapted_state_ph: self.adapted_state})

          # gradient_values = self._sess.run(self._tta_gradient, {self.adapted_state_ph: self.adapted_state})

          # for i, gradient in enumerate(gradient_values):
          #   print(f"Gradient of layer {i}: {gradient}")

          if self.apapted_moment:
            # moment update tta network parameters
            self._sess.run(self._tta_moment_update_op)
            # copy tta network parameters to moment network
            self._sess.run(self._copy_tta_to_moment_op)

          ############################################################
          # 2023/10/29: use the high confidence data
          # build statis graph, feed the data for the operator
          ############################################################
          if self.is_cross_entropy:
            if len(self.states_list) == 32:
              # transform tensor
              # state shape: （32， 84， 84， 4); action shape: (32, 9)
              aug_states = np.squeeze(np.stack(self.states_list), axis=1)
              aug_actions = np.squeeze(np.stack(self.actions_list))
              # update model
              self._sess.run(self._aug_test_op, {self.aug_state_ph: aug_states, 
                                            self.aug_action_ph: aug_actions})
              # clear replay buffer
              self.states_list = []
              self.actions_list = []
          ############################################################

          ############################################################
          # 2023/10/23: mixup augmenation data
          # build statis graph, feed the data for the operator
          ############################################################
          # action_label is a one-hot vector with shape (32, 9)
          if self.is_augmentation:
            if len(self.states_list) == 32:
              # transform tensor
              # state shape: （32， 84， 84， 4); action shape: (32, 9)
              aug_states = np.squeeze(np.stack(self.states_list), axis=1)
              aug_actions = np.squeeze(np.stack(self.actions_list))
              # mixup data
              mixed_states, mixed_actions = self.mixup_batch(aug_states, aug_actions, 0.9)
              # update model
              self._sess.run(self._aug_test_op, {self.aug_state_ph: mixed_states, 
                                            self.aug_action_ph: mixed_actions})
          ############################################################

          # clear replay buffer
          if len(self.states_list) == 32:
          # if len(self.states_list) == 1000:
            self.states_list = []
            self.actions_list = []

        # else: 
        #   if self.summary_writer is not None:
        #     summary = self._sess.run(self._merged_tta_tensor_summaries, {self.adapted_state_ph: self.adapted_state})
        #     self.summary_writer.add_summary(summary, self.high_entropy_step)
        #     self.high_entropy_step += 1
        #     self.replay_states.append(self.adapted_state)
      else:
        if entropy < self.entropy:
          # update model
          self._sess.run(self._test_op, {self.state_ph: self.adapted_state, 
                                        self.adapted_state_ph: self.adapted_state})

          ############################################################
          # 2023/10/23: mixup augmenation data
          # build statis graph, feed the data for the operator
          ############################################################
          # action_label is a one-hot vector with shape (32, 9)
          if self.is_augmentation:
            self.states_list.append(self.adapted_state)
            action_label = self._sess.run(self._action_label, {self.adapted_state_ph: self.adapted_state})
            self.actions_list.append(action_label)

            if len(self.states_list) == 32:
              # transform tensor
              # state shape: （32， 84， 84， 4); action shape: (32, 9)
              aug_states = np.squeeze(np.stack(self.states_list), axis=1)
              aug_actions = np.squeeze(np.stack(self.actions_list))
              # mixup data
              mixed_states, mixed_actions = self.mixup_batch(aug_states, aug_actions, 0.9)
              # update model
              self._sess.run(self._aug_test_op, {self.aug_state_ph: mixed_states, 
                                            self.aug_action_ph: mixed_actions})
              # clear replay buffer
              self.states_list = []
              self.actions_list = []
          ############################################################

        # else:
        #   if self.summary_writer is not None:
        #     summary = self._sess.run(self._merged_tta_tensor_summaries, {self.adapted_state_ph: self.adapted_state})
        #     self.summary_writer.add_summary(summary, self.high_entropy_step)
        #     self.high_entropy_step += 1
        #     self.replay_states.append(self.adapted_state)

      return action

  def _compute_entropy(self):
    for (i, state) in enumerate(self.replay_states):
      if self.summary_writer is not None:
        summary = self._sess.run(self._merged_entropy_summaries, {self.adapted_state_ph: state,
                                                                  self.state_ph: state})
        self.summary_writer.add_summary(summary, i)

  def _select_single_step_action(self, i):
    """Select an action from the set of available actions.

    Chooses an action randomly with probability self._calculate_epsilon(), and
    otherwise acts greedily according to the current Q-value estimates.

    Returns:
      int, the selected action.
    """
    if self.eval_mode:
      epsilon = self.epsilon_eval
    else:
      epsilon = self.epsilon_fn(
          self.epsilon_decay_period,
          self.training_steps,
          self.min_replay_history,
          self.epsilon_train)
    if random.random() <= epsilon:
      # Choose a random action with probability epsilon.
      return random.randint(0, self.num_actions - 1), 0
    else:
      multi_step_state = getattr(self, "multi_step_state_%d" % i)
      action = self._sess.run(self._test_q_argmax, {self.adapted_state_ph: multi_step_state})
      entropy = self._sess.run(self._test_q_entropy, {self.adapted_state_ph: multi_step_state})
      max_q_value = self._sess.run(self._max_q_value, {self.adapted_state_ph: multi_step_state})
      return action, max_q_value
  
  def _select_single_step_action_by_entropy(self, i):
    """Select an action from the set of available actions.

    Chooses an action randomly with probability self._calculate_epsilon(), and
    otherwise acts greedily according to the current Q-value estimates.

    Returns:
      int, the selected action.
    """
    if self.eval_mode:
      epsilon = self.epsilon_eval
    else:
      epsilon = self.epsilon_fn(
          self.epsilon_decay_period,
          self.training_steps,
          self.min_replay_history,
          self.epsilon_train)
    if random.random() <= epsilon:
      # Choose a random action with probability epsilon.
      return random.randint(0, self.num_actions - 1), 0, 0
    else:
      multi_step_state = getattr(self, "multi_step_state_%d" % i)
      action = self._sess.run(self._test_q_argmax, {self.adapted_state_ph: multi_step_state})
      entropy = self._sess.run(self._test_q_entropy, {self.adapted_state_ph: multi_step_state})
      max_q_value = self._sess.run(self._max_q_value, {self.adapted_state_ph: multi_step_state})
      return action, max_q_value, entropy

  def _select_reference_action(self):
    """Select an action from the set of available actions.

    Chooses an action randomly with probability self._calculate_epsilon(), and
    otherwise acts greedily according to the current Q-value estimates.

    Returns:
       int, the selected action.
    """
    # if self.eval_mode:
    #   epsilon = self.epsilon_eval
    # else:
    #   epsilon = self.epsilon_fn(
    #       self.epsilon_decay_period,
    #       self.training_steps,
    #       self.min_replay_history,
    #       self.epsilon_train)
    # if random.random() <= epsilon:
    #   # Choose a random action with probability epsilon.
    #   return random.randint(0, self.num_actions - 1)
    # else:
    #   # return self._sess.run(self._q_argmax, {self.state_ph: self.reference_state})
    return self._sess.run(self._q_argmax, {self.state_ph: getattr(self, "multi_step_state_0")})

  def _select_action(self):
    """Select an action from the set of available actions.

    Chooses an action randomly with probability self._calculate_epsilon(), and
    otherwise acts greedily according to the current Q-value estimates.

    Returns:
       int, the selected action.
    """
    if self.eval_mode:
      epsilon = self.epsilon_eval
    else:
      epsilon = self.epsilon_fn(
          self.epsilon_decay_period,
          self.training_steps,
          self.min_replay_history,
          self.epsilon_train)
    if random.random() <= epsilon:
      # Choose a random action with probability epsilon.
      return random.randint(0, self.num_actions - 1)
    else:
      # print("entropy: ", self._sess.run(self._q_entropy, {self.state_ph: self.state}))
      # # 创建一个初始值为 0 的 1 维张量（大小为 84 x 84）
      # self.state = np.ones((1, 84, 84, 4))
      # # self.state = np.arange(0, 84 * 84 * 4).reshape((1, 84, 84, 4)) 
      # # 将张量 reshape 成形状为 (84, 84, 1)
      # target_layer = self.online_convnet.layers[2].get_weights()[0]
      # print("select_action online parameter: ", target_layer[0][0][34][45])
      # print("select_action online parameter: ", target_layer[0][1][23][34])
      # print("select_action online parameter: ", target_layer[1][0][12][54])
      # print("select_action online action: ", self._sess.run(self._net_outputs.q_values, {self.state_ph: self.state}))

      # # Choose the action with highest Q-value at the current state.
      if self.summary_writer is not None:
        summary = self._sess.run(self._merged_eval_summaries, feed_dict={self.state_ph: self.state})
        self.summary_writer.add_summary(summary, self.test_step)
        # print(summary)
        # print(self.summary_writer)

        self.test_step += 1
      # # print("Q value: : ", self._sess.run(self._net_outputs.q_values, {self.state_ph: self.state}))

      return self._sess.run(self._q_argmax, {self.state_ph: self.state})
      # return self._sess.run(self._test_q_argmax, {self.state_ph: self.state})

  def copy_tta_to_monent(self):
    self._sess.run(self._copy_tta_to_moment_op)

  def update_online_tta_op(self):
    self._sess.run(self._sync_online_to_offline_ops)

  def _train_step(self):
    """Runs a single training step.

    Runs a training op if both:
      (1) A minimum number of frames have been added to the replay buff_train_steper.
      (2) `training_steps` is a multiple of `update_period`.

    Also, syncs weights from online to target network if training steps is a
    multiple of target update period.
    """
    # Run a train op at the rate of self.update_period if enough training steps
    # have been run. This matches the Nature DQN behaviour.
    if self._replay.memory.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        _, _, merged_train_summaries = self._sess.run(self._train_op)
        if (self.summary_writer is not None and
            self.training_steps > 0 and
            self.training_steps % self.summary_writing_frequency == 0):
          # summary = self._sess.run(self._merged_summaries)
          self.summary_writer.add_summary(merged_train_summaries, self.training_steps)

      if self.training_steps % self.target_update_period == 0:
        self._sess.run(self._sync_qt_ops)

    self.training_steps += 1

  def _record_observation(self, observation):
    """Records an observation and update state.

    Extracts a frame from the observation vector and overwrites the oldest
    frame in the state buffer.

    Args:
      observation: numpy array, an observation from the environment.
    """
    # Set current observation. We do the reshaping to handle environments
    # without frame stacking.
    self._observation = np.reshape(observation, self.observation_shape)
    # Swap out the oldest frame with the current frame.
    self.state = np.roll(self.state, -1, axis=-1)
    self.state[0, ..., -1] = self._observation
    # print("state: ", self.state)
  
  def _record_reference_observation(self, observation):
    """Records an observation and update state.

    Extracts a frame from the observation vector and overwrites the oldest
    frame in the state buffer.

    Args:
      observation: numpy array, an observation from the environment.
    """
    # Set current observation. We do the reshaping to handle environments
    # without frame stacking.
    self._observation = np.reshape(observation, self.observation_shape)
    # Swap out the oldest frame with the current frame.
    multi_state = getattr(self, "multi_step_state_0")
    multi_state = np.roll(multi_state, -1, axis=-1)
    multi_state[0, ..., -1] = self._observation
    setattr(self, "multi_step_state_0", multi_state)
  
  def _record_single_step_observation(self, observation, i):
    """Records an observation and update state.

    Extracts a frame from the observation vector and overwrites the oldest
    frame in the state buffer.

    Args:
      observation: numpy array, an observation from the environment.
    """
    # Set current observation. We do the reshaping to handle environments
    # without frame stacking.
    self._observation = np.reshape(observation, self.observation_shape)
    # Swap out the oldest frame with the current frame.
    multi_state = getattr(self, "multi_step_state_%d" % i)
    multi_state = np.roll(multi_state, -1, axis=-1)
    multi_state[0, ..., -1] = self._observation
    setattr(self, "multi_step_state_%d" % i, multi_state)
  
  def _record_multi_step_adapted_observation(self, observation, i):
    """Records an observation and update state.

    Extracts a frame from the observation vector and overwrites the oldest
    frame in the state buffer.

    Args:
      observation: numpy array, an observation from the environment.
    """
    # Set current observation. We do the reshaping to handle environments
    # without frame stacking.
    multi_step_state = getattr(self, "multi_step_state_%d" % i)
    self._adpted_observation = np.reshape(observation, self.observation_shape)
    # Swap out the oldest frame with the current frame.
    multi_step_state = np.roll(multi_step_state, -1, axis=-1)
    multi_step_state[0, ..., -1] = self._adpted_observation
    setattr(self, "multi_step_state_%d" % i, multi_step_state)

  def _record_adapted_observation(self, observation):
    """Records an observation and update state.

    Extracts a frame from the observation vector and overwrites the oldest
    frame in the state buffer.

    Args:
      observation: numpy array, an observation from the environment.
    """
    # Set current observation. We do the reshaping to handle environments
    # without frame stacking.
    self._adpted_observation = np.reshape(observation, self.observation_shape)
    # Swap out the oldest frame with the current frame.
    self.adapted_state = np.roll(self.adapted_state, -1, axis=-1)
    self.adapted_state[0, ..., -1] = self._adpted_observation
    # print("adapted state: ", self.adapted_state)

  def _store_transition(self, last_observation, action, reward, is_terminal):
    """Stores an experienced transition.

    Executes a tf session and executes replay buffer ops in order to store the
    following tuple in the replay buffer:
      (last_observation, action, reward, is_terminal).

    Pedantically speaking, this does not actually store an entire transition
    since the next state is recorded on the following time step.

    Args:
      last_observation: numpy array, last observation.
      action: int, the action taken.
      reward: float, the reward.
      is_terminal: bool, indicating if the current state is a terminal state.
    """
    self._replay.add(last_observation, action, reward, is_terminal)

  def _reset_state(self):
    """Resets the agent state by filling it with zeros."""
    self.state.fill(0)
  
  def _reset_single_step_state(self, i):
    """Resets the agent state by filling it with zeros."""
    multi_state = getattr(self, "multi_step_state_%d" % i)
    multi_state.fill(0)
    setattr(self, "multi_step_state_%d" % i, multi_state)
  
  def _reset_reference_state(self):
    """Resets the agent state by filling it with zeros."""
    # self.reference_state.fill(0)
    multi_state = getattr(self, "multi_step_state_0")
    multi_state.fill(0)
    setattr(self, "multi_step_state_0", multi_state)
    # self.reference_state.fill(0)
  
  def _reset_adapted_state(self):
    """Resets the agent state by filling it with zeros."""
    self.adapted_state.fill(0)

  def _reset_multi_step_adapted_state(self, i):
    """Resets the agent state by filling it with zeros."""
    multi_step_state = getattr(self, "multi_step_state_%d" % i)
    multi_step_state.fill(0)
    setattr(self, "multi_step_state_%d" % i, multi_step_state)

  def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
    """Returns a self-contained bundle of the agent's state.

    This is used for checkpointing. It will return a dictionary containing all
    non-TensorFlow objects (to be saved into a file by the caller), and it saves
    all TensorFlow objects into a checkpoint file.

    Args:
      checkpoint_dir: str, directory where TensorFlow objects will be saved.
      iteration_number: int, iteration number to use for naming the checkpoint
        file.

    Returns:
      A dict containing additional Python objects to be checkpointed by the
        experiment. If the checkpoint directory does not exist, returns None.
    """
    if not tf.io.gfile.exists(checkpoint_dir):
      return None
    # Call the Tensorflow saver to checkpoint the graph.
    self._saver.save(
        self._sess,
        os.path.join(checkpoint_dir, 'tf_ckpt'),
        global_step=iteration_number)
    # Checkpoint the out-of-graph replay buffer.
    self._replay.save(checkpoint_dir, iteration_number)
    bundle_dictionary = {}
    bundle_dictionary['state'] = self.state
    bundle_dictionary['training_steps'] = self.training_steps
    return bundle_dictionary

  def unbundle(self, checkpoint_dir, iteration_number, bundle_dictionary):
    """Restores the agent from a checkpoint.

    Restores the agent's Python objects to those specified in bundle_dictionary,
    and restores the TensorFlow objects to those specified in the
    checkpoint_dir. If the checkpoint_dir does not exist, will not reset the
      agent's state.

    Args:
      checkpoint_dir: str, path to the checkpoint saved by tf.Save.
      iteration_number: int, checkpoint version, used when restoring the replay
        buffer.
      bundle_dictionary: dict, containing additional Python objects owned by
        the agent.

    Returns:
      bool, True if unbundling was successful.
    """
    try:
      # self._replay.load() will throw a NotFoundError if it does not find all
      # the necessary files.
      self._replay.load(checkpoint_dir, iteration_number)
    except tf.errors.NotFoundError:
      if not self.allow_partial_reload:
        # If we don't allow partial reloads, we will return False.
        return False
      logging.warning('Unable to reload replay buffer!')
    if bundle_dictionary is not None:
      for key in self.__dict__:
        if key in bundle_dictionary:
          self.__dict__[key] = bundle_dictionary[key]
    elif not self.allow_partial_reload:
      return False
    else:
      logging.warning("Unable to reload the agent's parameters!")
    # Restore the agent's TensorFlow graph.
    self._saver.restore(self._sess,
                        os.path.join(checkpoint_dir,
                                     'tf_ckpt-{}'.format(iteration_number)))
    return True
>>>>>>> ed0d7b9c1ae446c6cffed41684f9178952a22685
