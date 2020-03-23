# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.#
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
"""Agent for integrating multiple algorithms."""

from typing import Callable

import gin

import torch

from alf.networks.network import Network


from alf.algorithms.algorithm import Algorithm
from alf.algorithms.agent import Agent
from alf.algorithms.actor_critic_algorithm import ActorCriticAlgorithm
from alf.algorithms.entropy_target_algorithm import EntropyTargetAlgorithm
from alf.algorithms.icm_algorithm import ICMAlgorithm
from alf.algorithms.agent_helpers import accumulate_loss_info, after_update, accumulate_algortihm_rewards
from alf.algorithms.on_policy_algorithm import OnPolicyAlgorithm, RLAlgorithm
from alf.data_structures import (TimeStep, Experience, LossInfo, namedtuple,
                                 AlgStep, StepType, TrainingInfo)
from alf.utils.common import cast_transformer
from alf.utils.math_ops import add_ignore_empty
from alf.algorithms.config import TrainerConfig

AgentState = namedtuple(
    "AgentState", ["rl", "irm", "goal_generator"], default_value=())

AgentInfo = namedtuple(
    "AgentInfo", ["rl", "irm", "goal_generator", "entropy_target"],
    default_value=())

AgentLossInfo = namedtuple(
    "AgentLossInfo", ["rl", "irm", "goal_generator", "entropy_target"],
    default_value=())

@gin.configurable
class HierarchicalAgent(Agent):
    """Agent

    Agent is a master algorithm that integrates different algorithms together.
    """

    def __init__(self,
                 name="HierarchicalAgentAlgorithm",
                 **kwargs):
        """Create a Hierarchical Agent

        Args:
            name (str): Name of this algorithm.
        """
        super(HierarchicalAgent, self).__init__(
            name=name, **kwargs
        )

    def rollout_step(self, time_step: TimeStep, state: AgentState):
        """Rollout for one step."""
        new_state = AgentState()
        info = AgentInfo()
        observation = time_step.observation

        if self._goal_generator is not None:
            goal_step = self._goal_generator.rollout_step(
                time_step._replace(observation=observation),
                state.goal_generator)
            info = info._replace(goal_generator=goal_step.info)
            new_state = new_state._replace(goal_generator=goal_step.state)

            rl_observation = [observation, goal_step.output]
            irm_observation = rl_observation
        
        rl_step = self._rl_algorithm.rollout(
            time_step._replace(observation=rl_observation), state.rl, mode)
        if self._goal_generator is not None:
            irm_observation.append(rl_step.action)
        if self._irm is not None:
            irm_step = self._irm.train_step(
                time_step._replace(observation=irm_observation),
                state=state.irm)
            info = info._replace(irm=icm_step.info)
            new_state = new_state._replace(irm=icm_step.state)


        new_state = new_state._replace(rl=rl_step.state)
        info = info._replace(rl=rl_step.info)

        if self._entropy_target_algorithm and self.is_on_policy():
            # For off-policy training, skip entropy_target_algorithm
            # during `unroll()
            assert 'action_distribution' in rl_step.info._fields, (
                "PolicyStep from rl_algorithm.rollout() does not contain "
                "`action_distribution`, which is required by "
                "`enforce_entropy_target`")
            et_step = self._entropy_target_algorithm.train_step(
                rl_step.info.action_distribution,
                step_type=time_step.step_type)
            info = info._replace(entropy_target=et_step.info)

        return AlgStep(action=rl_step.action, state=new_state, info=info)

    def train_step(self, exp: Experience, state):
        new_state = AgentState()
        info = AgentInfo()
        observation = self._encode(exp)

        if self._goal_generator is not None:
            goal_step = self._goal_generator.train_step(
                exp._replace(observation=observation),
                state.goal_generator,
                mode=RLAlgorithm.OFF_POLICY_TRAINING)

            #info = info._replace(goal_generator=goal_step.info)
            # Have to keep rollout state and training state specs consistent
            new_state = new_state._replace(goal_generator=goal_step.state)

            goal = exp.rollout_info.goal_generator.action

            rl_observation = [observation, goal]
            irm_observation = [observation,
                                #exp.rollout_info.goal_generator.observation],
                                goal]

        rl_step = self._rl_algorithm.train_step(
            exp._replace(observation=rl_observation), state.rl)
        
        if self._goal_generator is not None:
            irm_observation.append(rl_step.action)
        if self._irm is not None:
            # compute intrinsic rewards with update-to-date parameters when training
            irm_step = self._icm.train_step(
                exp._replace(observation=irm_observation),
                state=state.irm,
                calc_intrinsic_reward=False)
            info = info._replace(irm=icm_step.info)
            new_state = new_state._replace(irm=icm_step.state)

        new_state = new_state._replace(rl=rl_step.state)
        info = info._replace(rl=rl_step.info)

        if self._entropy_target_algorithm:
            assert 'action_distribution' in rl_step.info._fields, (
                "PolicyStep from rl_algorithm.train_step() does not contain "
                "`action_distribution`, which is required by "
                "`enforce_entropy_target`")
            et_step = self._entropytarget_algorithm.train_step(
                rl_step.info.action_distribution, step_type=exp.step_type)
            info = info._replace(entropy_target=et_step.info)

        return AlgStep(action=rl_step.action, state=new_state, info=info)

#    def calc_loss(self, training_info):
#        """Calculate loss."""
#        def _update_loss(loss_info, training_info, name, algorithm, ignore_name=False):
#            if algorithm is None:
#                return loss_info
#            if not ignore_name:
#                info = getattr(training_info.info, name)
#            else:
#                info = training_info
#
#            new_loss_info = algorithm.calc_loss(info)
#            return LossInfo(
#                loss=add_ignore_empty(loss_info.loss, new_loss_info.loss),
#                scalar_loss=add_ignore_empty(loss_info.scalar_loss,
#                                             new_loss_info.scalar_loss),
#                extra=loss_info.extra._replace(**{name: new_loss_info.extra}))
#
#        assert training_info.rollout_info != ()
#
#        rl_loss_info = self._rl_algorithm.calc_loss(
#            training_info._replace(info=training_info.info.rl, reward=
#            training_info.info.irm.reward)) # updated intrinsic rewards
#        loss_info = rl_loss_info._replace(
#            extra=AgentLossInfo(rl=rl_loss_info.extra))
#        loss_info = _update_loss(loss_info, training_info, 'goal_generator', self._goal_generator)
#        loss_info = _update_loss(loss_info, training_info, 'irm', self._icm)
#        loss_info = _update_loss(loss_info, training_info, 'entropy_target',
#                                    self._entropy_target_algorithm)
#        return loss_info

    def preprocess_experience(self, exp: Experience):
        #reward = self.calc_training_reward(exp.reward, exp.rollout_info)
        new_exp = self._rl_algorithm.preprocess_experience(
            exp._replace(#reward=reward,
                         rollout_info=exp.rollout_info.rl))
        return new_exp._replace(
            rollout_info=exp.rollout_info._replace(rl=new_exp.rollout_info))

