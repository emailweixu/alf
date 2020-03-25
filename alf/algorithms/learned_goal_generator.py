# Copyright (c) 2020 Horizon Robotics. All Rights Reserved.
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

import gin
import torch
import numpy as np
import functools

from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.data_structures import (TimeStep, Experience, LossInfo, namedtuple,
                                 AlgStep, StepType, TrainingInfo, make_experience)

from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.algorithms.sac_algorithm import SacAlgorithm
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.utils.summary_utils import summarize_action
from alf.utils import dist_utils

import alf.utils.common as common


GoalState = namedtuple("GoalState", ["goal", "observation", "steps", "rl_reward", "rl_discount", "rl"], default_value=())
GoalInfo = namedtuple("GoalInfo", ["goal", "observation", "switch_skills"], default_value=())

class ScalarSpec(TensorSpec):
    """Simplified spec for scalar tensor."""

    def __init__(self, dtype=torch.float32):
        super().__init__(shape=(), dtype=dtype)

@gin.configurable
def create_goal_spec(num_of_skills):
    return BoundedTensorSpec((), dtype=torch.int64, minimum=0, maximum = num_of_skills-1)


@gin.configurable
def create_time_step_spec(num_of_steps):
    # +1 because the steps can be [0, num_of_steps]
    return TensorSpec((num_of_steps + 1,))


def create_train_state_spec(num_of_goals, rl_algorithm_cls, observation_spec, name):
    import inspect
    signature = inspect.signature(rl_algorithm_cls)
    default_name = signature.parameters['name'].default
    name = name + '/' + default_name
    rl = rl_algorithm_cls(name=name, debug_summaries=True)
    train_state_spec = GoalState(
        goal=TensorSpec((num_of_goals, ), dtype=torch.float32),
        observation=observation_spec,
        steps=ScalarSpec(dtype=torch.int64),
        rl_reward=ScalarSpec(),
        rl_discount=ScalarSpec(),
        rl=rl.train_state_spec)
    return rl, train_state_spec


@gin.configurable
class LearnedCategoricalGoalGenerator(RLAlgorithm):
    """Learned Goal Generation Module

    This module generates a learned categorical goal for the agent
    in the beginning of every episode, and at the specified 
    num_steps_before_policy_switch frequency.
    """

    def __init__(self,
                 num_of_goals,
                 #train_state_spec,
                 observation_spec,
                 num_steps_before_policy_switch=5,
                 initial_collect_steps=1000,
                 rl_algorithm_cls=SacAlgorithm,
                 discount=0.99,
                 mini_batch_size=80,
                 mini_batch_length=2,
                 replay_buffer_max_length=1e7,
                 name="LearnedCategoricalGoalGenerator"):
        """
        Args:
            num_of_goals (int): total number of goals the agent can sample from
        """
        goal_spec = create_goal_spec(num_of_goals)
        self._num_of_goals = num_of_goals
        self._num_steps_before_policy_switch = num_steps_before_policy_switch
        self._action_spec = goal_spec
        self._discount = discount
        self._mini_batch_size = mini_batch_size
        self._mini_batch_length = mini_batch_length
        self._initial_collect_steps = initial_collect_steps
        #self._train_state_spec=train_state_spec
        rl, self._train_state_spec = create_train_state_spec(
            num_of_goals, 
            rl_algorithm_cls, 
            observation_spec,
            name)
        super().__init__(
            observation_spec=observation_spec,
            action_spec=goal_spec,
            train_state_spec=self._train_state_spec, 
            name=name)
        self._rl = rl
        self._rl.set_exp_replayer("uniform", common._env.batch_size, max_length=int(replay_buffer_max_length))


    def _trainable_attributes_to_ignore(self):
        return ["_rl"]

    def _generate_goal(self, time_step: TimeStep, state: GoalState, rollout):
        rl_time_step = time_step._replace(
            reward=(state.rl_reward / self._num_steps_before_policy_switch).to(torch.float32),
            discount=state.rl_discount,
            prev_action=torch.argmax(state.goal, axis=-1).to(torch.int64),
        )
        rl_step = self._rl.rollout_step(
            time_step=rl_time_step, state=state.rl)

        if rollout == True:
            rl_step = dist_utils.distributions_to_params(rl_step)
            exp = make_experience(rl_time_step, rl_step, state.rl)
            self._rl.observe(exp)
        return state._replace(
            goal=torch.nn.functional.one_hot(rl_step.output, self._num_of_goals).to(torch.float32),
            observation=time_step.observation, # record the init obs for the next K steps
            steps=torch.zeros_like(state.steps),
            rl_reward=torch.zeros_like(state.rl_reward),
            rl_discount=torch.ones_like(state.rl_discount),
            rl=rl_step.state,
        )

    def _update_goal(self, time_step, state, switch_skills, step_type, rollout):
        new_goal_mask = ((step_type == StepType.FIRST) | switch_skills) | (step_type == StepType.LAST)
        new_goal_mask = new_goal_mask.unsqueeze(dim=-1)
        new_state = self._generate_goal(time_step, state, rollout)
        new_state = new_state._replace(goal=torch.where(new_goal_mask, new_state.goal, state.goal))
        return new_state

    def _step(self, time_step: TimeStep, state, rollout):
        """Perform one step of rollout or prediction.

        Note that as RandomCategoricalGoalGenerator is a non-trainable module,
        and it will randomly generate goals for episode beginnings.

        Args:
            time_step (TimeStep): input time_step data
            state (nested Tensor): consistent with train_state_spec
        Returns:
            AlgStep:
                output (Tensor); one-hot goal vectors
                state (nested Tensor):
                info (GoalInfo): storing any info that will be put into a replay
                    buffer (if off-policy training is used.
        """
        goal = state.goal
        step_type = time_step.step_type

        environment_reward = time_step.reward

        state = state._replace(rl_reward=state.rl_reward + environment_reward,
                                rl_discount=state.rl_discount*time_step.discount)

        switch_skills = torch.fmod(state.steps,
                                self._num_steps_before_policy_switch) == 0

        new_state = self._update_goal(time_step, state, switch_skills, step_type, rollout)

        new_state = new_state._replace(steps=new_state.steps + 1)

        return AlgStep(
            output=(new_state.goal),
            state=new_state,
            info=GoalInfo(goal=new_state.goal,
                            observation=state.observation, # should be from the old state
                            )
        )

    def rollout_step(self, time_step: TimeStep, state):
        return self._step(time_step, state, rollout=True)

    def predict_step(self, time_step: TimeStep, state, epsilon_greedy):
        return self._step(time_step, state, rollout=False)

    def train_step(self, exp: Experience, state):
        """For off-policy training, the current output goal should be taken from
        the goal in `exp.rollout_info` (historical goals generated during rollout).

        Note that we cannot take the goal from `state` and pass it down because
        the first state might be a zero vector. And we also cannot resample
        the goal online because that might be inconsistent with the sampled
        experience trajectory.

        Args:
            exp (Experience): the experience data whose `rollout_info` has been
                replaced with goal generator `rollout_info`.
            state (nested Tensor):

        Returns:
            AlgStep:
                output (Tensor); one-hot goal vectors
                state (nested Tensor):
                info (GoalInfo): for training.
        """
        goal = exp.rollout_info.goal
        return AlgStep(output=goal, state=state, info=GoalInfo(goal=goal))

    def _rl_train(self):
        if self._rl._exp_replayer._buffer.total_size() > self._initial_collect_steps:
            experience = self._rl._exp_replayer.replay(
                sample_batch_size=self._mini_batch_size,
                mini_batch_length=self._mini_batch_length)
            self._rl._train_experience(
                experience, 
                num_updates=1, 
                mini_batch_size=self._mini_batch_size,
                mini_batch_length=self._mini_batch_length,
                update_counter_every_mini_batch=False)
                #mini_batch_size=self._mini_batch_size,
                #mini_batch_length=self._mini_batch_length,
                #whole_replay_buffer_training=False,
                #clear_replay_buffer=False)

    def calc_loss(self, info: TrainingInfo):
        self._rl_train()
        info = info.rollout_info
        info = info._replace(goal=torch.argmax(info.goal, dim=-1))
        summarize_action(info.goal.detach().cpu().numpy(), self._action_spec, name=self._name)
        return LossInfo()

