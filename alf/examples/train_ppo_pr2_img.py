# coding=utf-8
# Copyright 2018 The TF-Agents Authors.
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

r"""Train PR2-Gripper Env with PPO using image observations
it is based on train_ppo_pr2.py

additional modifications include:
1. modify actor_network and value_network to support multiple observations with encoder network
2. split collect_episodes_per_iteration into minibatch to avoid out-of-memory

to train just
./ppo_pr2_img.sh ~/model_path (with an empty social_bot.log file)

also be sure to set use_internal_states_only to False in pr2.py

The current status: after 12 hours (1.6M environment steps), avg episode reward is about 1.0.
roughly could touch the object a couple of times within each episode.

"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import logging

from absl import app
from absl import flags
from absl import logging as absl_logging

import tensorflow as tf
from tf_agents.agents.ppo import ppo_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import tf_py_environment
from alf.environments import suite_socialbot
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
#from tf_agents.networks import actor_distribution_network
import actor_distribution_network
from tf_agents.networks import actor_distribution_rnn_network
#from tf_agents.networks import value_network
import value_network
from tf_agents.networks import encoding_network
from tf_agents.networks import value_rnn_network

from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

import gin.tf
import gin

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('env_name', 'HalfCheetah-v2', 'Name of an environment')
flags.DEFINE_integer('replay_buffer_capacity', 1001,
                     'Replay buffer capacity per env.')
flags.DEFINE_integer('num_parallel_environments', 30,
                     'Number of environments to run in parallel')
flags.DEFINE_integer('num_environment_steps', 10000000,
                     'Number of environment steps to run before finishing.')
flags.DEFINE_integer('num_epochs', 25,
                     'Number of epochs for computing policy updates.')
flags.DEFINE_integer(
    'collect_episodes_per_iteration', 30,
    'The number of episodes to take in the environment before '
    'each update. This is the total across all parallel '
    'environments.')
flags.DEFINE_integer('num_eval_episodes', 30,
                     'The number of episodes to run eval on.')
flags.DEFINE_boolean('use_rnns', False,
                     'If true, use RNN for policy and value function.')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding parameters.')
FLAGS = flags.FLAGS

def get_encoder(conv_layer_params, tf_env, fc_layers, name):
  layers = []
  for (filters, kernel_size, strides) in conv_layer_params:
    layers.append(
      tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        activation=tf.keras.activations.softsign,
        kernel_initializer=tf.compat.v1.keras.initializers.glorot_uniform(),
        dtype=tf.float32,
        name=name))
  layers.append(tf.keras.layers.Flatten())
  image_processing_layers = tf.keras.Sequential(layers)
  encoder = encoding_network.EncodingNetwork(
    tf_env.observation_spec(),
    preprocessing_layers=(image_processing_layers, tf.keras.layers.Lambda(lambda x: x)),
    preprocessing_combiner=tf.keras.layers.Concatenate(axis=-1),
    fc_layer_params=fc_layers,
    activation_fn=tf.keras.activations.softsign,
    kernel_initializer=tf.compat.v1.keras.initializers.glorot_uniform(),
    batch_squash=True,
    dtype=tf.float32)
  return encoder


@gin.configurable
def train_eval(
    root_dir,
    env_name='SocialBot-Pr2Gripper-v0',
    env_load_fn=suite_socialbot.load,
    random_seed=0,
    # TODO(b/127576522): rename to policy_fc_layers.
    actor_fc_layers=(100, 50, 25),
    value_fc_layers=(100, 50, 25),
    conv_layer_params= [(32,8,4),(64,4,2),(64,3,2)],
    use_rnns=False,
    # Params for collect
    num_environment_steps=10000000,
    collect_episodes_per_iteration=30,
    num_parallel_environments=30,
    replay_buffer_capacity=1001,  # Per-environment
    # Params for train
    num_epochs=25,
    learning_rate=1e-4,
    entropy_regularization=0.002,
    # Params for eval
    num_eval_episodes=30,
    eval_interval=1000,

    train_checkpoint_interval=1000,
    policy_checkpoint_interval=1000,

    # Params for summaries and absl_logging
    log_interval=50,
    summary_interval=50,
    summaries_flush_secs=1,
    use_tf_functions=True,
    debug_summaries=False,
    summarize_grads_and_vars=False,
    steps_per_episode=101,
    mini_batches = 4):
  """A simple train and eval for PPO."""
  if root_dir is None:
    raise AttributeError('train_eval requires a root_dir.')

  root_dir = os.path.expanduser(root_dir)
  train_dir = os.path.join(root_dir, 'train')
  eval_dir = os.path.join(root_dir, 'eval')

  train_summary_writer = tf.compat.v2.summary.create_file_writer(
      train_dir, flush_millis=summaries_flush_secs * 1000)
  train_summary_writer.set_as_default()

  eval_summary_writer = tf.compat.v2.summary.create_file_writer(
      eval_dir, flush_millis=summaries_flush_secs * 1000)
  eval_metrics = [
      tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
      tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes)
  ]

  global_step = tf.compat.v1.train.get_or_create_global_step()
  with tf.compat.v2.summary.record_if(
      lambda: tf.math.equal(global_step % summary_interval, 0)):
    tf.compat.v1.set_random_seed(random_seed)

    if num_parallel_environments > 1:
      tf_env = tf_py_environment.TFPyEnvironment(
        parallel_py_environment.ParallelPyEnvironment(
          [lambda: env_load_fn(env_name, wrap_with_process=False)] * num_parallel_environments))
    else:
      tf_env = tf_py_environment.TFPyEnvironment(env_load_fn(env_name))

    eval_tf_env = tf_py_environment.TFPyEnvironment(env_load_fn(env_name))


    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    if use_rnns:
      actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
        tf_env.observation_spec(),
        tf_env.action_spec(),
        input_fc_layer_params=actor_fc_layers,
        output_fc_layer_params=None,
        conv_layer_params=conv_layer_params,
        activation_fn=tf.keras.activations.softsign)
      value_net = value_rnn_network.ValueRnnNetwork(
        tf_env.observation_spec(),
        input_fc_layer_params=value_fc_layers,
        output_fc_layer_params=None,
        conv_layer_params=conv_layer_params,
        activation_fn=tf.keras.activations.softsign)
    else:
      actor_encoder = get_encoder(conv_layer_params, tf_env, actor_fc_layers, 'actor/encoder')
      critic_encoder = get_encoder(conv_layer_params, tf_env, value_fc_layers, 'value/encoder')
      #import ipdb; ipdb.set_trace()

      actor_net = actor_distribution_network.ActorDistributionNetwork(
          tf_env.observation_spec(),
          tf_env.action_spec(),
          encoder=actor_encoder)

      value_net = value_network.ValueNetwork(
          tf_env.observation_spec(),
          encoder=critic_encoder)

    tf_agent = ppo_agent.PPOAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        optimizer,
        actor_net=actor_net,
        value_net=value_net,
        entropy_regularization=entropy_regularization,
        num_epochs=num_epochs,

        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=global_step)
    tf_agent.initialize()

    environment_steps_metric = tf_metrics.EnvironmentSteps()
    step_metrics = [
        tf_metrics.NumberOfEpisodes(),
        environment_steps_metric,
    ]

    train_metrics = step_metrics + [
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric(),
    ]

    eval_policy = tf_agent.policy
    collect_policy = tf_agent.collect_policy

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        tf_agent.collect_data_spec,
        batch_size=num_parallel_environments,
        max_length=replay_buffer_capacity)

    collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        tf_env,
        collect_policy,
        observers=[replay_buffer.add_batch] + train_metrics,
        num_episodes=collect_episodes_per_iteration)

    train_checkpointer = common.Checkpointer(
        ckpt_dir=train_dir,
        agent=tf_agent,
        global_step=global_step,
        metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
    policy_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'policy'),
        policy=tf_agent.policy,
        global_step=global_step)

    train_checkpointer.initialize_or_restore()
    #policy_checkpointer.initialize_or_restore()

    if use_tf_functions:
      # TODO(b/123828980): Enable once the cause for slowdown was identified.
      collect_driver.run = common.function(collect_driver.run, autograph=False)
      tf_agent.train = common.function(tf_agent.train, autograph=False)

    collect_time = 0
    train_time = 0
    timed_at_step = global_step.numpy()

    while environment_steps_metric.result() < num_environment_steps:
      global_step_val = global_step.numpy()
      if global_step_val % eval_interval == 0 and global_step_val > 0:
        metric_utils.eager_compute(
            eval_metrics,
            eval_tf_env,
            eval_policy,
            num_episodes=num_eval_episodes,
            train_step=global_step,
            summary_writer=eval_summary_writer,
            summary_prefix='Metrics',
        )

      start_time = time.time()
      collect_driver.run()
      collect_time += time.time() - start_time

      start_time = time.time()
      #import ipdb;ipdb.set_trace()

      for i in range(2 * mini_batches):
        trajectories, _ = replay_buffer.get_next(
          sample_batch_size=collect_episodes_per_iteration // mini_batches,
          num_steps=steps_per_episode
        )
        total_loss, _ = tf_agent.train(experience=trajectories)
      #trajectories = replay_buffer.gather_all()
      #total_loss, _ = tf_agent.train(experience=trajectories)
      replay_buffer.clear()
      train_time += time.time() - start_time

      for train_metric in train_metrics:
        train_metric.tf_summaries(
            train_step=global_step, step_metrics=step_metrics)

      if global_step_val % log_interval == 0:
        absl_logging.info('step = %d, loss = %f', global_step_val, total_loss)
        steps_per_sec = (
            (global_step_val - timed_at_step) / (collect_time + train_time))
        absl_logging.info('%.3f steps/sec', steps_per_sec)
        absl_logging.info('collect_time = {}, train_time = {}'.format(
            collect_time, train_time))
        with tf.compat.v2.summary.record_if(True):
          tf.compat.v2.summary.scalar(
              name='global_steps_per_sec', data=steps_per_sec, step=global_step)

        timed_at_step = global_step_val
        collect_time = 0
        train_time = 0

      if global_step_val % train_checkpoint_interval == 0:
        train_checkpointer.save(global_step=global_step_val)

      if global_step_val % policy_checkpoint_interval == 0:
        policy_checkpointer.save(global_step=global_step_val)

    # One final eval before exiting.
    metric_utils.eager_compute(
        eval_metrics,
        eval_tf_env,
        eval_policy,
        num_episodes=num_eval_episodes,
        train_step=global_step,
        summary_writer=eval_summary_writer,
        summary_prefix='Metrics',
    )


def main(_):
  logging.basicConfig(level=logging.INFO)
  logging.getLogger().addHandler(
    logging.FileHandler(filename=FLAGS.root_dir + '/social_bot.log'))

  absl_logging.set_verbosity(absl_logging.INFO)
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
  tf.compat.v1.enable_v2_behavior()
  train_eval(
      FLAGS.root_dir,
      env_name=FLAGS.env_name,
      use_rnns=FLAGS.use_rnns,
      num_environment_steps=FLAGS.num_environment_steps,
      collect_episodes_per_iteration=FLAGS.collect_episodes_per_iteration,
      num_parallel_environments=FLAGS.num_parallel_environments,
      replay_buffer_capacity=FLAGS.replay_buffer_capacity,
      num_epochs=FLAGS.num_epochs,
      num_eval_episodes=FLAGS.num_eval_episodes)


if __name__ == '__main__':
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
  for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
  #tf.config.gpu.set_per_process_memory_growth(True)
  flags.mark_flag_as_required('root_dir')

  app.run(main)