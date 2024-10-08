# import gymnasium as gym
# from gymnasium.core import Wrapper, ObservationWrapper
# from gymnasium import spaces
# import numpy as np
# from minigrid.wrappers import FullyObsWrapper
#
# class MaxStepsWrapper(Wrapper):
#     def __init__(self, env: gym.Env, max_steps: int):
#         super().__init__(env)
#         self.max_steps = max_steps
#         self.env.max_steps = max_steps
#         self.env.env.max_steps = max_steps
#     def reset(self, **kwargs):
#         state = self.env.reset()
#         if 'options' in kwargs and 'max_steps' in kwargs['options']:
#             self.env.max_steps = kwargs['options']['max_steps']
#         else:
#             self.env.max_steps = self.max_steps
#         return state
#
#     def step(self, action):
#         return self.env.step(action)
#
#
# class FullyCustom(FullyObsWrapper):
#     def __init__(self, env: gym.Env, max_steps: int):
#         super().__init__(env)
#         self.max_steps = max_steps
#         self.env.max_steps = max_steps
#
#     def reset(self, **kwargs):
#         # Reset the environment and update the max_steps
#         state = super().reset()
#         if 'options' in kwargs and 'max_steps' in kwargs['options']:
#             self.env.max_steps = kwargs['options']['max_steps']
#         else:
#             self.env.max_steps = self.max_steps
#         return state
#

import gymnasium as gym
from gymnasium.core import Wrapper, ObservationWrapper
from gymnasium import spaces
import numpy as np
from minigrid.wrappers import FullyObsWrapper
import cv2
import numpy as np
import torch
from gymnasium.spaces import Box


class ActionSpaceWrapper(Wrapper):
    def __init__(self, env: gym.Env,  max_steps, new_action_space: int):
        super().__init__(env)
        self.new_action_space = new_action_space
        self.action_space = spaces.Discrete(new_action_space)
        self.env.action_space = self.action_space


        self.max_steps = max_steps
        self.env.max_steps = max_steps

    def reset(self, **kwargs):
        # Reset the environment and update the max_steps
        state = super().reset()



        self.env.max_steps = self.max_steps
        return state


'''     
    def step(self, action):
        # Map the action to the original action space
        original_action = self._map_action(action)
        return self.env.step(original_action)

    def _map_action(self, action):
        # Map the action from the new action space to the original action space
        # Example mapping: action = action if action < self.new_action_space - 1 else self.new_action_space - 1
        return action
'''
class MaxStepsWrapper(Wrapper):
    def __init__(self, env: gym.Env, max_steps: int, symbolic, seed, max_episode_length, action_repeat, bit_depth, new_action_space=3):
        super().__init__(env)
        self.max_steps = max_steps
        self.env.max_steps = max_steps
        self.env.env.max_steps = max_steps

        self.symbolic = symbolic
        # self._env.seed(seed)
        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat
        self.bit_depth = bit_depth

        self.action_space = spaces.Discrete(new_action_space)
        self.env.action_space = self.action_space
        self.env.env.action_space = self.action_space
        #self.env.env.evm.action_space = self.action_space
    def reset(self, **kwargs):
        self.t = 0
        state = self.env.reset()
        if 'options' in kwargs and 'max_steps' in kwargs['options']:
            self.env.max_steps = kwargs['options']['max_steps']
            self.env.env.max_steps = kwargs['options']['max_steps']
        else:
            self.env.max_steps = self.max_steps
            self.env.env.max_steps = self.max_steps
        state = self.preprocess_state(state[0])
        #return state
        return _images_to_observation(state, self.bit_depth)

    def step(self, action):
        observation, r, terminated, truncated, info = self.env.step(action)
        obs = self.preprocess_state(observation)
        obs = _images_to_observation(obs, self.bit_depth)
        return obs, r, terminated, truncated, info


    def preprocess_state(self, observation):
        if isinstance(observation, dict):
            return observation['image'].transpose(2,0,1)
        else:
            return observation.transpose(2,0, 1)
class FullyCustom(FullyObsWrapper):
    def __init__(self, env: gym.Env, max_steps: int):
        super().__init__(env)
        self.max_steps = max_steps
        self.env.max_steps = max_steps


    def reset(self, **kwargs):
        # Reset the environment and update the max_steps
        state = super().reset()
        if 'options' in kwargs and 'max_steps' in kwargs['options']:
            self.env.max_steps = kwargs['options']['max_steps']
        else:
            self.env.max_steps = self.max_steps
        return state

# Preprocesses an observation inplace (from float32 Tensor [0, 255] to [-0.5, 0.5])
def preprocess_observation_(observation, bit_depth):
    observation.div_(2 ** (8 - bit_depth)).floor_().div_(2**bit_depth).sub_(
        0.5
    )  # Quantise to given bit depth and centre
    observation.add_(
        torch.rand_like(observation).div_(2**bit_depth)
    )  # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)


# Postprocess an observation for storage (from float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])
def postprocess_observation(observation, bit_depth):
    return np.clip(np.floor((observation + 0.5) * 2**bit_depth) * 2 ** (8 - bit_depth), 0, 2**8 - 1).astype(
        np.uint8
    )




def _images_to_observation(images, bit_depth):
    '''
    images = torch.tensor(
        cv2.resize(images, (64, 64), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1), dtype=torch.float32
    )  # Resize and put channel first
    '''
    images = torch.tensor(
        cv2.resize(images.transpose(2,1,0), (64, 64), interpolation=cv2.INTER_LINEAR).transpose(2,0,1), dtype=torch.float32
    )
    preprocess_observation_(images, bit_depth)  # Quantise, centre and dequantise inplace
    return images.unsqueeze(dim=0)  # Add batch dimension


class EnvBatcher:
    def __init__(self, env_class, env_args, env_kwargs, n):
        self.n = n
        self.envs = [env_class(*env_args, **env_kwargs) for _ in range(n)]
        self.dones = [True] * n

    # Resets every environment and returns observation
    def reset(self):
        observations = [env.reset() for env in self.envs]
        self.dones = [False] * self.n
        return torch.cat(observations)

    # Steps/resets every environment and returns (observation, reward, done)
    def step(self, actions):
        done_mask = torch.nonzero(torch.tensor(self.dones))[
            :, 0
        ]  # Done mask to blank out observations and zero rewards for previously terminated environments
        observations, rewards, dones = zip(*[env.step(action) for env, action in zip(self.envs, actions)])
        dones = [
            d or prev_d for d, prev_d in zip(dones, self.dones)
        ]  # Env should remain terminated if previously terminated
        self.dones = dones
        observations, rewards, dones = (
            torch.cat(observations),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.uint8),
        )
        observations[done_mask] = 0
        rewards[done_mask] = 0
        return observations, rewards, dones

    def close(self):
        [env.close() for env in self.envs]
