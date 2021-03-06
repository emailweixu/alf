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
"""Test various functions related to calculating values."""

import collections
import tensorflow as tf

from tf_agents.trajectories.time_step import TimeStep, StepType

from alf.utils import common, value_ops

# scalable_agent code copied from https://github.com/deepmind/scalable_agent/blob/master/experiment.py
# under https://www.apache.org/licenses/LICENSE-2.0
# BEGIN(scalable_agent code):

VTraceFromLogitsReturns = collections.namedtuple('VTraceFromLogitsReturns', [
    'vs', 'pg_advantages', 'log_rhos', 'behaviour_action_log_probs',
    'target_action_log_probs'
])

VTraceReturns = collections.namedtuple('VTraceReturns', 'vs pg_advantages')


def log_probs_from_logits_and_actions(policy_logits, actions):
    """Computes action log-probs from policy logits and actions.
  In the notation used throughout documentation and comments, T refers to the
  time dimension ranging from 0 to T-1. B refers to the batch size and
  NUM_ACTIONS refers to the number of actions.
  Args:
    policy_logits (tensor): A float32 tensor of shape [T, B, NUM_ACTIONS] with
      un-normalized log-probabilities parameterizing a softmax policy.
    actions (tensor): An int32 tensor of shape [T, B] with actions.
  Returns:
    A float32 tensor of shape [T, B] corresponding to the sampling log
    probability of the chosen action w.r.t. the policy.
  """
    policy_logits = tf.convert_to_tensor(policy_logits, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions, dtype=tf.int32)

    policy_logits.shape.assert_has_rank(3)
    actions.shape.assert_has_rank(2)

    return -tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=policy_logits, labels=actions)


def from_logits(behaviour_policy_logits,
                target_policy_logits,
                actions,
                discounts,
                rewards,
                values,
                bootstrap_value,
                clip_rho_threshold=1.0,
                clip_pg_rho_threshold=1.0,
                name='vtrace_from_logits'):
    r"""V-trace for softmax policies.
  Calculates V-trace actor critic targets for softmax polices as described in
  "IMPALA: Scalable Distributed Deep-RL with
  Importance Weighted Actor-Learner Architectures"
  by Espeholt, Soyer, Munos et al.
  Target policy refers to the policy we are interested in improving and
  behaviour policy refers to the policy that generated the given
  rewards and actions.
  In the notation used throughout documentation and comments, T refers to the
  time dimension ranging from 0 to T-1. B refers to the batch size and
  NUM_ACTIONS refers to the number of actions.
  Args:
    behaviour_policy_logits (tensor): A float32 tensor of shape [T, B, NUM_ACTIONS] with
      un-normalized log-probabilities parametrizing the softmax behaviour
      policy.
    target_policy_logits (tensor): A float32 tensor of shape [T, B, NUM_ACTIONS] with
      un-normalized log-probabilities parametrizing the softmax target policy.
    actions (tensor): An int32 tensor of shape [T, B] of actions sampled from the
      behaviour policy.
    discounts (tensor): A float32 tensor of shape [T, B] with the discount encountered
      when following the behaviour policy.
    rewards (tensor): A float32 tensor of shape [T, B] with the rewards generated by
      following the behaviour policy.
    values (tensor): A float32 tensor of shape [T, B] with the value function estimates
      wrt. the target policy.
    bootstrap_value (tensor): A float32 of shape [B] with the value function estimate at
      time T.
    clip_rho_threshold (tensor): A scalar float32 tensor with the clipping threshold for
      importance weights (rho) when calculating the baseline targets (vs).
      rho^bar in the paper.
    clip_pg_rho_threshold (tensor): A scalar float32 tensor with the clipping threshold
      on rho_s in \rho_s \delta log \pi(a|x) (r + \gamma v_{s+1} - V(x_s)).
    name (str): The name scope that all V-trace operations will be created in.
  Returns:
    A `VTraceFromLogitsReturns` namedtuple with the following fields:
      vs: A float32 tensor of shape [T, B]. Can be used as target to train a
          baseline (V(x_t) - vs_t)^2.
      pg_advantages: A float 32 tensor of shape [T, B]. Can be used as an
        estimate of the advantage in the calculation of policy gradients.
      log_rhos: A float32 tensor of shape [T, B] containing the log importance
        sampling weights (log rhos).
      behaviour_action_log_probs: A float32 tensor of shape [T, B] containing
        behaviour policy action log probabilities (log \mu(a_t)).
      target_action_log_probs: A float32 tensor of shape [T, B] containing
        target policy action probabilities (log \pi(a_t)).
  """
    behaviour_policy_logits = tf.convert_to_tensor(
        behaviour_policy_logits, dtype=tf.float32)
    target_policy_logits = tf.convert_to_tensor(
        target_policy_logits, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions, dtype=tf.int32)

    # Make sure tensor ranks are as expected.
    # The rest will be checked by from_action_log_probs.
    behaviour_policy_logits.shape.assert_has_rank(3)
    target_policy_logits.shape.assert_has_rank(3)
    actions.shape.assert_has_rank(2)

    with tf.name_scope(
            name,
            values=[
                behaviour_policy_logits, target_policy_logits, actions,
                discounts, rewards, values, bootstrap_value
            ]):
        target_action_log_probs = log_probs_from_logits_and_actions(
            target_policy_logits, actions)
        behaviour_action_log_probs = log_probs_from_logits_and_actions(
            behaviour_policy_logits, actions)
        log_rhos = target_action_log_probs - behaviour_action_log_probs
        vtrace_returns = from_importance_weights(
            log_rhos=log_rhos,
            discounts=discounts,
            rewards=rewards,
            values=values,
            bootstrap_value=bootstrap_value,
            clip_rho_threshold=clip_rho_threshold,
            clip_pg_rho_threshold=clip_pg_rho_threshold)
        return VTraceFromLogitsReturns(
            log_rhos=log_rhos,
            behaviour_action_log_probs=behaviour_action_log_probs,
            target_action_log_probs=target_action_log_probs,
            **vtrace_returns._asdict())


def from_importance_weights(log_rhos,
                            discounts,
                            rewards,
                            values,
                            bootstrap_value,
                            clip_rho_threshold=1.0,
                            clip_pg_rho_threshold=1.0,
                            name='vtrace_from_importance_weights'):
    r"""V-trace from log importance weights.
  Calculates V-trace actor critic targets as described in
  "IMPALA: Scalable Distributed Deep-RL with
  Importance Weighted Actor-Learner Architectures"
  by Espeholt, Soyer, Munos et al.
  In the notation used throughout documentation and comments, T refers to the
  time dimension ranging from 0 to T-1. B refers to the batch size and
  NUM_ACTIONS refers to the number of actions. This code also supports the
  case where all tensors have the same number of additional dimensions, e.g.,
  `rewards` is [T, B, C], `values` is [T, B, C], `bootstrap_value` is [B, C].
  Args:
    log_rhos (tensor): A float32 tensor of shape [T, B, NUM_ACTIONS] representing the log
      importance sampling weights, i.e.
      log(target_policy(a) / behaviour_policy(a)). V-trace performs operations
      on rhos in log-space for numerical stability.
    discounts (tensor): A float32 tensor of shape [T, B] with discounts encountered when
      following the behaviour policy.
    rewards (tensor): A float32 tensor of shape [T, B] containing rewards generated by
      following the behaviour policy.
    values (tensor): A float32 tensor of shape [T, B] with the value function estimates
      wrt. the target policy.
    bootstrap_value (tensor): A float32 of shape [B] with the value function estimate at
      time T.
    clip_rho_threshold (tensor): A scalar float32 tensor with the clipping threshold for
      importance weights (rho) when calculating the baseline targets (vs).
      rho^bar in the paper. If None, no clipping is applied.
    clip_pg_rho_threshold (tensor): A scalar float32 tensor with the clipping threshold
      on rho_s in \rho_s \delta log \pi(a|x) (r + \gamma v_{s+1} - V(x_s)). If
      None, no clipping is applied.
    name (str): The name scope that all V-trace operations will be created in.
  Returns:
    A VTraceReturns namedtuple (vs, pg_advantages) where:
      vs: A float32 tensor of shape [T, B]. Can be used as target to
        train a baseline (V(x_t) - vs_t)^2.
      pg_advantages: A float32 tensor of shape [T, B]. Can be used as the
        advantage in the calculation of policy gradients.
  """
    log_rhos = tf.convert_to_tensor(log_rhos, dtype=tf.float32)
    discounts = tf.convert_to_tensor(discounts, dtype=tf.float32)
    rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
    values = tf.convert_to_tensor(values, dtype=tf.float32)
    bootstrap_value = tf.convert_to_tensor(bootstrap_value, dtype=tf.float32)
    if clip_rho_threshold is not None:
        clip_rho_threshold = tf.convert_to_tensor(
            clip_rho_threshold, dtype=tf.float32)
    if clip_pg_rho_threshold is not None:
        clip_pg_rho_threshold = tf.convert_to_tensor(
            clip_pg_rho_threshold, dtype=tf.float32)

    # Make sure tensor ranks are consistent.
    rho_rank = log_rhos.shape.ndims  # Usually 2.
    values.shape.assert_has_rank(rho_rank)
    bootstrap_value.shape.assert_has_rank(rho_rank - 1)
    discounts.shape.assert_has_rank(rho_rank)
    rewards.shape.assert_has_rank(rho_rank)
    if clip_rho_threshold is not None:
        clip_rho_threshold.shape.assert_has_rank(0)
    if clip_pg_rho_threshold is not None:
        clip_pg_rho_threshold.shape.assert_has_rank(0)

    with tf.name_scope(
            name
    ):  # removed: , values=[log_rhos, discounts, rewards, values, bootstrap_value]
        rhos = tf.exp(log_rhos)
        if clip_rho_threshold is not None:
            clipped_rhos = tf.minimum(
                clip_rho_threshold, rhos, name='clipped_rhos')
        else:
            clipped_rhos = rhos

        cs = tf.minimum(1.0, rhos, name='cs')
        # Append bootstrapped value to get [v1, ..., v_t+1]
        values_t_plus_1 = tf.concat(
            [values[1:], tf.expand_dims(bootstrap_value, 0)], axis=0)
        deltas = clipped_rhos * (
            rewards + discounts * values_t_plus_1 - values)

        sequences = (discounts, cs, deltas)

        # V-trace vs are calculated through a scan from the back to the beginning
        # of the given trajectory.
        def scanfunc(acc, sequence_item):
            discount_t, c_t, delta_t = sequence_item
            return delta_t + discount_t * c_t * acc

        initial_values = tf.zeros_like(bootstrap_value)
        vs_minus_v_xs = tf.scan(
            fn=scanfunc,
            elems=sequences,
            initializer=initial_values,
            parallel_iterations=1,
            back_prop=False,
            reverse=True,  # Computation starts from the back.
            name='scan')

        # Add V(x_s) to get v_s.
        vs = tf.add(vs_minus_v_xs, values, name='vs')

        # Advantage for policy gradient.
        vs_t_plus_1 = tf.concat(
            [vs[1:], tf.expand_dims(bootstrap_value, 0)], axis=0)
        if clip_pg_rho_threshold is not None:
            clipped_pg_rhos = tf.minimum(
                clip_pg_rho_threshold, rhos, name='clipped_pg_rhos')
        else:
            clipped_pg_rhos = rhos
        pg_advantages = (
            clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values))

        # Make sure no gradients backpropagated through the returned values.
        return VTraceReturns(
            vs=tf.stop_gradient(vs),
            pg_advantages=tf.stop_gradient(pg_advantages))


# END(scalable_agent code)


class DiscountedReturnTest(tf.test.TestCase):
    """Tests for alf.utils.value_ops.discounted_return
    """

    def test_discounted_return(self):
        values = tf.constant([[1.] * 5], tf.float32)
        step_types = tf.constant([[StepType.MID] * 5], tf.int64)
        rewards = tf.constant([[2.] * 5], tf.float32)
        discounts = tf.constant([[0.9] * 5], tf.float32)
        expected = tf.constant(
            [[(((1 * 0.9 + 2) * 0.9 + 2) * 0.9 + 2) * 0.9 + 2,
              ((1 * 0.9 + 2) * 0.9 + 2) * 0.9 + 2,
              (1 * 0.9 + 2) * 0.9 + 2, 1 * 0.9 + 2]],
            dtype=tf.float32)
        self.assertAllClose(
            value_ops.discounted_return(
                rewards=rewards,
                values=values,
                step_types=step_types,
                discounts=discounts,
                time_major=False), expected)

        # two episodes, and exceed by time limit (discount=1)
        step_types = tf.constant([[
            StepType.MID, StepType.MID, StepType.LAST, StepType.MID,
            StepType.MID
        ]], tf.int32)
        expected = tf.constant(
            [[(1 * 0.9 + 2) * 0.9 + 2, 1 * 0.9 + 2, 1, 1 * 0.9 + 2]],
            dtype=tf.float32)
        self.assertAllClose(
            value_ops.discounted_return(
                rewards=rewards,
                values=values,
                step_types=step_types,
                discounts=discounts,
                time_major=False), expected)

        # two episodes, and end normal (discount=0)
        step_types = tf.constant([[
            StepType.MID, StepType.MID, StepType.LAST, StepType.MID,
            StepType.MID
        ]], tf.int32)
        discounts = tf.constant([[0.9, 0.9, 0.0, 0.9, 0.9]])
        expected = tf.constant([[(0 * 0.9 + 2) * 0.9 + 2, 2, 1, 1 * 0.9 + 2]],
                               dtype=tf.float32)

        self.assertAllClose(
            value_ops.discounted_return(
                rewards=rewards,
                values=values,
                step_types=step_types,
                discounts=discounts,
                time_major=False), expected)


class GeneralizedAdvantageTest(tf.test.TestCase):
    """Tests for alf.utils.value_ops.generalized_advantage_estimation
    """

    def test_generalized_advantage_estimation(self):
        values = tf.constant([[2.] * 5], tf.float32)
        step_types = tf.constant([[StepType.MID] * 5], tf.int64)
        rewards = tf.constant([[3.] * 5], tf.float32)
        discounts = tf.constant([[0.9] * 5], tf.float32)
        td_lambda = 0.6 / 0.9

        d = 2 * 0.9 + 1
        expected = tf.constant([[((d * 0.6 + d) * 0.6 + d) * 0.6 + d,
                                 (d * 0.6 + d) * 0.6 + d, d * 0.6 + d, d]],
                               dtype=tf.float32)
        self.assertAllClose(
            value_ops.generalized_advantage_estimation(
                rewards=rewards,
                values=values,
                step_types=step_types,
                discounts=discounts,
                td_lambda=td_lambda,
                time_major=False), expected)

        # two episodes, and exceed by time limit (discount=1)

        step_types = tf.constant([[
            StepType.MID, StepType.MID, StepType.LAST, StepType.MID,
            StepType.MID
        ]], tf.int32)
        expected = tf.constant([[d * 0.6 + d, d, 0, d]], dtype=tf.float32)
        self.assertAllClose(
            value_ops.generalized_advantage_estimation(
                rewards=rewards,
                values=values,
                step_types=step_types,
                discounts=discounts,
                td_lambda=td_lambda,
                time_major=False), expected)

        # two episodes, and end normal (discount=0)
        step_types = tf.constant([[
            StepType.MID, StepType.MID, StepType.LAST, StepType.MID,
            StepType.MID
        ]], tf.int32)
        discounts = tf.constant([[0.9, 0.9, 0.0, 0.9, 0.9]])
        expected = tf.constant([[1 * 0.6 + d, 1, 0, d]], dtype=tf.float32)

        self.assertAllClose(
            value_ops.generalized_advantage_estimation(
                rewards=rewards,
                values=values,
                step_types=step_types,
                discounts=discounts,
                td_lambda=td_lambda,
                time_major=False), expected)


def vtrace_scalable_agent(imp_weights, discounts, rewards, values, step_types):
    # scalable agent has a one step shifted definition of some of these values.
    # E.g. action in alf is prev_action that caused the current reward.
    log_imp_weights = tf.math.log(imp_weights)
    log_imp_weights = tf.transpose(a=log_imp_weights)[:-1]
    discounts = tf.transpose(a=discounts)[1:]
    rewards = tf.transpose(a=rewards)[1:]
    values = tf.transpose(a=values)
    final_value = values[-1]
    values = values[:-1]
    vtrace_returns = from_importance_weights(
        log_rhos=log_imp_weights,
        discounts=discounts,
        rewards=rewards,
        values=values,
        bootstrap_value=final_value,
        clip_rho_threshold=1.0,
        clip_pg_rho_threshold=1.0)
    vs = vtrace_returns.vs
    vs = common.tensor_extend(vs, final_value)
    adv = vtrace_returns.pg_advantages
    adv = common.tensor_extend_zero(adv)
    vs = tf.transpose(a=vs)
    adv = tf.transpose(a=adv)
    return vs, adv


class VTraceTest(tf.test.TestCase):
    """Tests for alf.utils.value_ops.vtrace_returns_and_advantages_impl
    """

    def test_vtrace_returns_and_advantages_impl_on_policy_no_last_step(self):
        """Test vtrace_returns_and_advantages_impl on policy no last_step
            in the middle of the trajectory.
        """
        importance_ratio_clipped = tf.constant([[1.] * 5], tf.float32)
        values = tf.constant([[2.] * 5], tf.float32)
        step_types = tf.constant([[StepType.MID] * 5], tf.int64)
        rewards = tf.constant([[3.] * 5], tf.float32)
        discounts = tf.constant([[0.9] * 5], tf.float32)
        td_lambda = 1.0

        returns, advantages = value_ops.vtrace_returns_and_advantages_impl(
            importance_ratio_clipped,
            rewards,
            values,
            step_types,
            discounts,
            time_major=False)
        sa_returns, sa_adv = vtrace_scalable_agent(
            importance_ratio_clipped, discounts, rewards, values, step_types)
        self.assertAllClose(
            sa_adv, advantages, msg='advantages differ from scalable_agent')
        self.assertAllClose(
            sa_returns, returns, msg='returns differ from scalable_agent')
        expected_advantages = value_ops.generalized_advantage_estimation(
            rewards=rewards,
            values=values,
            step_types=step_types,
            discounts=discounts,
            td_lambda=td_lambda,
            time_major=False)
        expected_advantages = tf.transpose(a=expected_advantages)
        expected_advantages = common.tensor_extend_zero(expected_advantages)
        expected_advantages = tf.transpose(a=expected_advantages)
        self.assertAllClose(
            expected_advantages, advantages, msg='advantages differ from gold')

        expected_returns = value_ops.discounted_return(
            rewards=rewards,
            values=values,
            step_types=step_types,
            discounts=discounts,
            time_major=False)
        expected_returns = tf.transpose(a=expected_returns)
        values = tf.transpose(a=values)
        expected_returns = common.tensor_extend(expected_returns, values[-1])
        expected_returns = tf.transpose(a=expected_returns)
        self.assertAllClose(
            expected_returns, returns, msg='returns differ from gold')

    def test_vtrace_returns_and_advantages_impl_on_policy_has_last_step(self):
        """Test vtrace_returns_and_advantages_impl on policy has last_step
            in the middle of the trajectory.
        """
        importance_ratio_clipped = tf.constant([[1.] * 5], tf.float32)
        values = tf.constant([[2., 2.1, 2.2, 2.3, 2.4]], tf.float32)
        step_types = tf.constant([[
            StepType.MID, StepType.MID, StepType.LAST, StepType.MID,
            StepType.MID
        ]], tf.int32)
        rewards = tf.constant([[3., 3.1, 3.2, 3.3, 3.4]], tf.float32)
        discounts = tf.constant([[0.9, 0.9, 0.0, 0.9, 0.9]])
        td_lambda = 1.0

        returns, advantages = value_ops.vtrace_returns_and_advantages_impl(
            importance_ratio_clipped,
            rewards,
            values,
            step_types,
            discounts,
            time_major=False)

        expected_advantages = value_ops.generalized_advantage_estimation(
            rewards=rewards,
            values=values,
            step_types=step_types,
            discounts=discounts,
            td_lambda=td_lambda,
            time_major=False)
        expected_advantages = tf.transpose(a=expected_advantages)
        expected_advantages = common.tensor_extend_zero(expected_advantages)
        expected_advantages = tf.transpose(a=expected_advantages)
        self.assertAllClose(
            expected_advantages, advantages, msg='advantages differ')

        expected_returns = value_ops.discounted_return(
            rewards=rewards,
            values=values,
            step_types=step_types,
            discounts=discounts,
            time_major=False)
        expected_returns = tf.transpose(a=expected_returns)
        values = tf.transpose(a=values)
        expected_returns = common.tensor_extend(expected_returns, values[-1])
        expected_returns = tf.transpose(a=expected_returns)
        self.assertAllClose(expected_returns, returns, msg='returns differ')

    def test_vtrace_returns_and_advantages_impl_off_policy_has_last_step(self):
        """Test vtrace_returns_and_advantages_impl off policy has last_step
            in the middle of the trajectory.
        """
        r = 0.999
        d = 0.9
        importance_ratio_clipped = tf.constant([[r] * 5], tf.float32)
        values = tf.constant([[2.] * 5], tf.float32)
        step_types = tf.constant([[
            StepType.MID, StepType.MID, StepType.LAST, StepType.MID,
            StepType.MID
        ]], tf.int32)
        rewards = tf.constant([[3.] * 5], tf.float32)
        discounts = tf.constant([[d, d, 0., d, d]])
        td_lambda = 1.0

        returns, advantages = value_ops.vtrace_returns_and_advantages_impl(
            importance_ratio_clipped,
            rewards,
            values,
            step_types,
            discounts,
            time_major=False)

        td3 = (3. + 2. * d - 2.) * r
        expected_returns = tf.constant(
            [[td3 + d * r * (3. - 2.) * r, r, 0, td3, 0]], tf.float32) + values
        # 5.695401, 2.999   , 2.      , 4.7972  , 2.
        self.assertAllClose(expected_returns, returns, msg='returns differ')

        is_lasts = tf.cast(
            tf.equal(tf.transpose(a=step_types), StepType.LAST), tf.float32)
        expected_advantages = (1 - is_lasts[:-1]) * r * (
            tf.transpose(a=rewards)[1:] + tf.transpose(a=discounts)[1:] *
            tf.transpose(a=expected_returns)[1:] - tf.transpose(a=values)[:-1])
        expected_advantages = common.tensor_extend_zero(expected_advantages)
        expected_advantages = tf.transpose(a=expected_advantages)
        # 3.695401, 0.999   , 0.      , 2.7972  , 0.
        self.assertAllClose(
            expected_advantages, advantages, msg='advantages differ')

        # a case where values are not uniform over time.
        values = tf.constant([[0., 1., 2., 3., 4.]], tf.float32)
        returns, advantages = value_ops.vtrace_returns_and_advantages_impl(
            importance_ratio_clipped,
            rewards,
            values,
            step_types,
            discounts,
            time_major=False)

        td3 = (3. + 4. * d - 3) * r
        td1 = 2 * r
        expected_returns = tf.constant([[(3. + 1. * d - 0) * r + d * r * td1,
                                         td1, 0, td3, 0]], tf.float32) + values
        # 5.692502, 2.998   , 2.      , 6.5964  , 4.
        self.assertAllClose(expected_returns, returns, msg='returns differ')

        is_lasts = tf.cast(
            tf.equal(tf.transpose(a=step_types), StepType.LAST), tf.float32)
        expected_advantages = (1 - is_lasts[:-1]) * r * (
            tf.transpose(a=rewards)[1:] + tf.transpose(a=discounts)[1:] *
            tf.transpose(a=expected_returns)[1:] - tf.transpose(a=values)[:-1])
        expected_advantages = common.tensor_extend_zero(expected_advantages)
        expected_advantages = tf.transpose(a=expected_advantages)
        # 5.692502, 1.998   , 0.      , 3.5964  , 0.
        self.assertAllClose(
            expected_advantages, advantages, msg='advantages differ')

    def test_vtrace_returns_and_advantages_impl_off_policy_no_last_step(self):
        """Test vtrace_returns_and_advantages_impl off policy no last_step
            in the middle of the trajectory.  Testing against scalable agent
            only, as there was no last_step in that implementation.
        """
        r = 0.999
        d = 0.9
        importance_ratio_clipped = tf.constant([[r] * 5], tf.float32)
        values = tf.constant([[2.] * 5], tf.float32)
        step_types = tf.constant([[StepType.MID] * 5], tf.int32)
        rewards = tf.constant([[3.] * 5], tf.float32)
        discounts = tf.constant([[d] * 5])

        returns, advantages = value_ops.vtrace_returns_and_advantages_impl(
            importance_ratio_clipped,
            rewards,
            values,
            step_types,
            discounts,
            time_major=False)

        sa_returns, sa_adv = vtrace_scalable_agent(
            importance_ratio_clipped, discounts, rewards, values, step_types)
        self.assertAllClose(
            sa_adv, advantages, msg='advantages differ from scalable_agent 1')
        self.assertAllClose(
            sa_returns, returns, msg='returns differ from scalable_agent 1')

        # a case where these values are not uniform over time.
        importance_ratio_clipped = tf.constant(
            [[0.999, 0.888, 0.777, 0.666, 0.555]], tf.float32)
        rewards = tf.constant([[2., 3., 4., 5., 6.]], tf.float32)
        values = tf.constant([[0., 1., 2., 3., 4.]], tf.float32)
        discounts = tf.constant([[0.9, 0.8, 0.7, 0.6, 0.5]], tf.float32)

        returns, advantages = value_ops.vtrace_returns_and_advantages_impl(
            importance_ratio_clipped,
            rewards,
            values,
            step_types,
            discounts,
            time_major=False)

        sa_returns, sa_adv = vtrace_scalable_agent(
            importance_ratio_clipped, discounts, rewards, values, step_types)
        self.assertAllClose(
            sa_adv, advantages, msg='advantages differ from scalable_agent 2')
        self.assertAllClose(
            sa_returns, returns, msg='returns differ from scalable_agent 2')

    def test_vtrace_impl_on_policy_has_last_step_with_lambda(self):
        """Test vtrace_returns_and_advantages_impl on policy has last_step
            in the middle of the trajectory, and has td_lambda = 0.95
             Hasn't passed test yet.
        """
        importance_ratio_clipped = tf.constant([[1.] * 5], tf.float32)
        values = tf.constant([[2.] * 5], tf.float32)
        step_types = tf.constant([[
            StepType.MID, StepType.MID, StepType.LAST, StepType.MID,
            StepType.MID
        ]], tf.int32)
        rewards = tf.constant([[3.] * 5], tf.float32)
        discounts = tf.constant([[0.9, 0.9, 0.0, 0.9, 0.9]])
        td_lambda = 0.95

        returns, advantages = value_ops.vtrace_returns_and_advantages_impl(
            importance_ratio_clipped,
            rewards,
            values,
            step_types,
            discounts,
            td_lambda=td_lambda,
            time_major=False)

        # advantages will differ from GAE when td_lambda is not 1.0
        self.assertAllClose([[3.7, 1., 0., 2.8, 0.]],
                            advantages,
                            msg='advantages wrong')

        # returns will differ when td_lambda is not 1.0
        self.assertAllClose([[5.655, 3., 2., 4.8, 2.]],
                            returns,
                            msg='returns wrong')


if __name__ == '__main__':
    from alf.utils.common import set_per_process_memory_growth

    set_per_process_memory_growth()
    tf.test.main()
