import gymnasium as gym
from gymnasium.core import Wrapper, ObservationWrapper
from gymnasium import spaces
import numpy as np
from minigrid.wrappers import FullyObsWrapper
import cv2


class MaxStepsWrapper(Wrapper):
    def __init__(self, env: gym.Env, max_steps: int, new_action_space = 3):
        super().__init__(env)
        self.max_steps = max_steps
        self.env.max_steps = max_steps
        self.env.env.max_steps = max_steps

        self.action_space = spaces.Discrete(new_action_space)
        self.env.action_space = self.action_space
        self.env.env.action_space = self.action_space

        self._env = env

    def sample_random_action(self):
        action = np.zeros((1, self._env.action_space.n,), dtype=float)
        idx = np.random.randint(0, self._env.action_space.n, size=(1,))[0]
        action[0, idx] = 1
        return action

    def reset(self, **kwargs):
        observation = self.env.reset()

        # if 'options' in kwargs and 'max_steps' in kwargs['options']:
        #     self.env.max_steps = kwargs['options']['max_steps']
        # else:
        #     self.env.max_steps = self.max_steps
        self.max_steps = self.max_steps
        #state = self.preprocess_state(state[0])
        # observation이 딕셔너리일 경우
        # observation이 딕셔너리일 경우 'image' 키에서 이미지 데이터 추출
        if isinstance(observation, tuple) and isinstance(observation[0], dict) and 'image' in observation[0]:
            observation = observation[0]['image']  # 딕셔너리에서 'image' 키의 값을 추출
        else:
            raise TypeError(f"Unsupported observation type or missing 'image' key: {type(observation)}")

            # 이미지가 NumPy 배열인지 확인하고, NumPy 배열로 변환
        if not isinstance(observation, np.ndarray):
            raise TypeError(f"Unsupported image type: {type(observation)}")

            # 이미지가 비어있는지 확인
        if observation.size == 0:
            raise ValueError("Observation image is empty.")

        observation = cv2.resize(observation, (64, 64), interpolation=cv2.INTER_LINEAR)
        observation = np.clip(observation, 0, 255).astype(np.uint8)
        observation = np.transpose(observation, (2, 0, 1))  # 3, 64, 64
        return observation

    #def step(self, action):
    #    return self.env.step(action)
    def step(self, action):
        observation, r, terminated, truncated, info = self.env.step(action)
        #obs = self.preprocess_state(observation)

        observation = observation['image']
        observation = cv2.resize(observation, (64, 64), interpolation=cv2.INTER_LINEAR)
        observation = np.clip(observation, 0, 255).astype(np.uint8)
        observation = np.transpose(observation, (2, 0, 1))  # 3, 64, 64
        return observation, r, terminated, truncated, info


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