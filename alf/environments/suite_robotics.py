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
"""Suite for loading OpenAI Robotics environments.

**NOTE**: Mujoco requires separated installation.

(gym >= 0.10, and mujoco>=1.50)

Follow the instructions at:

https://github.com/openai/mujoco-py

"""

try:
    import mujoco_py
except ImportError:
    mujoco_py = None

import gym
from gym.wrappers import FlattenDictWrapper

import gin
from alf.environments import suite_gym, torch_wrappers, process_environment
from alf.environments.utils import UnwrappedEnvChecker

_unwrapped_env_checker_ = UnwrappedEnvChecker()



def is_available():
    return mujoco_py is not None


class SparseReward(gym.Wrapper):
    """
    Convert the original -1/0 rewards to 0/1
    """

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        if reward == 0:
            done = True
        return ob, reward + 1, done, info


@gin.configurable
def load(environment_name,
         env_id=None,
         discount=1.0,
         max_episode_steps=None,
         gym_env_wrappers=(),
         torch_env_wrappers=(),
         wrap_with_process=False):
    """Loads the selected environment and wraps it with the specified wrappers.

    Note that by default a TimeLimit wrapper is used to limit episode lengths
    to the default benchmarks defined by the registered environments.

    Args:
        environment_name: Name for the environment to load.
        discount: Discount to use for the environment.
        max_episode_steps: If None the max_episode_steps will be set to the default
            step limit defined in the environment's spec. No limit is applied if set
            to 0 or if there is no timestep_limit set in the environment's spec.
        gym_env_wrappers: Iterable with references to wrapper classes to use
            directly on the gym environment.
        env_wrappers: Iterable with references to wrapper classes to use on the
            gym_wrapped environment.

    Returns:
        A PyEnvironmentBase instance.
    """
    _unwrapped_env_checker_.check_and_update(wrap_with_process)

    gym_spec = gym.spec(environment_name)
    env = gym_spec.make()

    if max_episode_steps is None:
        if gym_spec.max_episode_steps is not None:
            max_episode_steps = gym_spec.max_episode_steps
        else:
            max_episode_steps = 0
    
    def env_ctor(env_id=None):
        return suite_gym.wrap_env(
            env,
            env_id=env_id,
            discount=discount,
            max_episode_steps=max_episode_steps,
            gym_env_wrappers=gym_env_wrappers,
            torch_env_wrappers=torch_env_wrappers)

    # concat robot's observation and the goal location
    #env = FlattenDictWrapper(env, ["observation", "desired_goal"])
    env = FlattenDictWrapper(env, ["observation"])
    env = SparseReward(env)
    
    if wrap_with_process:
        process_env = process_environment.ProcessEnvironment(
            functools.partial(env_ctor))
        process_env.start()
        torch_env = torch_wrappers.TorchEnvironmentBaseWrapper(process_env)
    else:
        torch_env = env_ctor(env_id=env_id)

    return torch_env