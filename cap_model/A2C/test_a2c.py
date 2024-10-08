from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
import wandb
from wandb.integration.sb3 import WandbCallback
from envs.GymMoreRedBalls import GymMoreRedBalls
from envs.wrapper import GymMoreRedBallsWrapper, FullyCustom, MaxStepsWrapper, ActionSpaceWrapper

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

# 환경 감싸기
env = make_vec_env("CartPole-v1", n_envs=1, monitor_dir="./logs")

# 모델 정의
model = A2C("MlpPolicy", env, verbose=1)

# 학습 시작
model.learn(total_timesteps=100000)
