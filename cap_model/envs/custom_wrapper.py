import gymnasium as gym
from gymnasium.core import Wrapper, ObservationWrapper
from gymnasium import spaces
import numpy as np
from minigrid.wrappers import FullyObsWrapper

class MaxStepsWrapper(Wrapper):
    def __init__(self, env: gym.Env, max_steps: int):
        super().__init__(env)
        self.max_steps = max_steps
        self.env.max_steps = max_steps
        self.env.env.max_steps = max_steps
    def reset(self, **kwargs):
        state = self.env.reset()
        if 'options' in kwargs and 'max_steps' in kwargs['options']:
            self.env.max_steps = kwargs['options']['max_steps']
        else:
            self.env.max_steps = self.max_steps
        state = self.preprocess_state(state[0])
        return state

    #def step(self, action):
    #    return self.env.step(action)
    def step(self, action):
        observation, r, terminated, truncated, info = self.env.step(action)
        obs = self.preprocess_state(observation)
        return obs, r, terminated, truncated, info


    def preprocess_state(self, observation):
        if isinstance(observation, dict):
            return observation['image'].flatten()
        else:
            return observation.flatten()
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

