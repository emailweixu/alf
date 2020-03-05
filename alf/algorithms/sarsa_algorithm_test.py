# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
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
"""Tests for sarsa_algorithm.py."""

from absl import logging
from absl.testing import parameterized
import functools
import gin
import torch

import alf
from alf.algorithms.actor_critic_algorithm import ActorCriticAlgorithm
from alf.algorithms.config import TrainerConfig
from alf.algorithms.ppo_algorithm_test import unroll
from alf.algorithms.sarsa_algorithm import SarsaAlgorithm
from alf.environments.suite_unittest import ActionType, PolicyUnittestEnv
from alf.networks import (
    ActorDistributionNetwork, ActorDistributionRNNNetwork,
    StableNormalProjectionNetwork, CriticNetwork, CriticRNNNetwork)
from alf.utils import common
from alf.utils.math_ops import clipped_exp

DEBUGGING = False


def _create_algorithm(env, use_rnn, on_policy, fast_critic_bias_speed,
                      learning_rate):
    observation_spec = env.observation_spec()
    action_spec = env.action_spec()
    fc_layer_params = (16, 16)
    continuous_projection_net_ctor = functools.partial(
        alf.networks.NormalProjectionNetwork,
        state_dependent_std=True,
        scale_distribution=True,
        std_transform=clipped_exp)

    # TODO: test ddpg after ActorNetwork is ready
    if use_rnn:
        actor_net = ActorDistributionRNNNetwork(
            observation_spec,
            action_spec,
            fc_layer_params=fc_layer_params,
            lstm_hidden_size=(4, ))
        critic_net = CriticRNNNetwork((observation_spec, action_spec),
                                      joint_fc_layer_params=fc_layer_params,
                                      lstm_hidden_size=(4, ))
    else:
        actor_net = ActorDistributionNetwork(
            observation_spec,
            action_spec,
            fc_layer_params=fc_layer_params,
            continuous_projection_net_ctor=continuous_projection_net_ctor)
        critic_net = CriticNetwork((observation_spec, action_spec),
                                   joint_fc_layer_params=fc_layer_params)

    config = TrainerConfig(
        root_dir="dummy",
        unroll_length=1,
        initial_collect_steps=500,
        use_rollout_state=True,
        mini_batch_length=1,
        mini_batch_size=256,
        num_updates_per_train_step=1,
        whole_replay_buffer_training=False,
        clear_replay_buffer=False,
        debug_summaries=DEBUGGING,
        summarize_grads_and_vars=DEBUGGING,
        summarize_action_distributions=DEBUGGING)

    return SarsaAlgorithm(
        observation_spec=observation_spec,
        action_spec=action_spec,
        env=env,
        config=config,
        on_policy=on_policy,
        fast_critic_bias_speed=fast_critic_bias_speed,
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=torch.optim.Adam(lr=learning_rate),
        critic_optimizer=torch.optim.Adam(lr=learning_rate),
        alpha_optimizer=torch.optim.Adam(lr=learning_rate),
        debug_summaries=DEBUGGING)


class SarsaTest(parameterized.TestCase, alf.test.TestCase):
    # TODO: on_policy=True is very unstable, try to figure out the possible
    # reason.
    @parameterized.parameters(
        dict(on_policy=False, use_rnn=False),
        dict(on_policy=False, use_rnn=True),
        dict(on_policy=False, fast_critic_bias_speed=10.),
    )
    def test_sarsa(self,
                   on_policy=False,
                   use_rnn=False,
                   fast_critic_bias_speed=0.):
        logging.info("on_policy=%s use_rnn=%s fast_critic_bias_speed=%s" %
                     (on_policy, use_rnn, fast_critic_bias_speed))
        env_class = PolicyUnittestEnv
        learning_rate = 2e-2
        iterations = 500
        num_env = 1
        if on_policy:
            num_env = 128
        steps_per_episode = 13
        env = env_class(
            num_env, steps_per_episode, action_type=ActionType.Continuous)
        eval_env = env_class(
            100, steps_per_episode, action_type=ActionType.Continuous)

        algorithm = _create_algorithm(
            env,
            on_policy=on_policy,
            use_rnn=use_rnn,
            fast_critic_bias_speed=fast_critic_bias_speed,
            learning_rate=learning_rate)

        env.reset()
        eval_env.reset()
        for i in range(iterations):
            algorithm.train_iter()

            eval_env.reset()
            eval_time_step = unroll(eval_env, algorithm, steps_per_episode - 1)
            logging.log_every_n_seconds(
                logging.INFO,
                "%d reward=%f" % (i, float(eval_time_step.reward.mean())),
                n_seconds=1)

        self.assertAlmostEqual(
            1.0, float(eval_time_step.reward.mean()), delta=0.3)


if __name__ == '__main__':
    alf.test.main()
