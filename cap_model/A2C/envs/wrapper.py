import gymnasium as gym
from gymnasium.core import Wrapper
from gymnasium import spaces
import numpy as np
from minigrid.wrappers import FullyObsWrapper


class ActionSpaceWrapper(Wrapper):
    def __init__(self, env: gym.Env, max_steps: int, new_action_space: int):
        super().__init__(env)
        self.new_action_space = new_action_space
        self.action_space = spaces.Discrete(new_action_space)

        # 환경의 action_space를 새롭게 설정
        self.env.action_space = self.action_space

        self.max_steps = max_steps
        self.env.max_steps = max_steps

        # observation_space가 Dict인지 확인
        if isinstance(env.observation_space, spaces.Dict):
            self.observation_space = env.observation_space['image']
        else:
            self.observation_space = env.observation_space  # 기본 관찰 공간 사용

    def reset(self, **kwargs):
        # 환경을 리셋하고 max_steps 업데이트
        state = super().reset(**kwargs)
        self.env.max_steps = self.max_steps

        return state



class MaxStepsWrapper(Wrapper):
    def __init__(self, env: gym.Env, max_steps: int):
        super().__init__(env)
        self.max_steps = max_steps

        # observation_space가 Dict인지 확인 후 설정
        if isinstance(env.observation_space, spaces.Dict):
            self.observation_space = env.observation_space['image']
        else:
            self.observation_space = env.observation_space  # 기본 관찰 공간 사용

        self.env.max_steps = max_steps

    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)  # 여기를 수정하여 args를 전달
        self.env.max_steps = self.max_steps
        return state

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        #obs = self.preprocess_state(observation)
        return obs, r, terminated, truncated, info

    def preprocess_state(self, observation):
        # 관찰값에서 'image'만 반환
        if isinstance(observation, dict):
            return observation['image'].flatten()
        else:
            return observation.flatten()


class FullyCustom(Wrapper):  # FullyObsWrapper 대신 Wrapper를 상속합니다.
    def __init__(self, env: gym.Env, max_steps: int):
        super().__init__(env)
        self.max_steps = max_steps

        # 'image' 만 사용하는 observation_space를 설정
        if isinstance(env.observation_space, spaces.Dict):
            self.observation_space = env.observation_space['image']
        else:
            self.observation_space = env.observation_space  # 혹은 env가 Dict가 아닐 경우 그대로 사용

    def reset(self, **kwargs):
        # 환경을 리셋하고 max_steps 업데이트
        state = super().reset(**kwargs)
        self.env.max_steps = self.max_steps
        return state

    def observation(self, obs):
        # 관찰값에서 'image'만 반환
        return obs['image'] if isinstance(obs, dict) else obs

class GymMoreRedBallsWrapper(gym.ObservationWrapper):
    def __init__(self, env, max_steps: int):
        super().__init__(env)
        # 기존 환경의 관찰 공간에서 'image'만을 사용
        self.observation_space = env.observation_space['image']
        self.max_steps = max_steps

    def observation(self, obs):
        # 관찰값에서 'image'만 반환
        return obs['image']
