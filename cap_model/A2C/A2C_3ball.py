from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
import wandb
from wandb.integration.sb3 import WandbCallback
from envs.GymMoreRedBalls import GymMoreRedBalls
from envs.wrapper import GymMoreRedBallsWrapper, FullyCustom, MaxStepsWrapper, ActionSpaceWrapper


# Custom callback to stop after a given number of episodes and log rewards
class EpisodeStoppingCallback(BaseCallback):
    def __init__(self, total_episodes, verbose=1):
        super().__init__(verbose)
        self.total_episodes = total_episodes
        self.episode_count = 0
        self.episode_rewards = []
        self.current_episode_reward = 0

    def _on_step(self) -> bool:
        # Step reward is found in self.locals['rewards']
        reward = self.locals['rewards'][0]  # Step reward
        self.current_episode_reward += reward  # Accumulate reward for the current episode

        # Check if the episode is done
        if self.locals['dones'][0]:
            self.episode_count += 1
            self.episode_rewards.append(self.current_episode_reward)

            # Log the reward to WandB
            wandb.log({
                "episode_reward": self.current_episode_reward,
                "episode": self.episode_count,
                "global_step": self.num_timesteps  # Log global step as well
            }, step=self.episode_count)

            # Reset current episode reward
            self.current_episode_reward = 0

            # Stop training if the target number of episodes is reached
            if self.episode_count >= self.total_episodes:
                print(f"Training stopped after {self.episode_count} episodes")
                return False
        return True



# Initialize WandB
run = wandb.init(project="3ball_CAP", entity='hails', sync_tensorboard=True, config={
    "learning_rate": 2e-4,  # Dreamer의 model_lr
    "n_steps": 50,  # Dreamer의 batch_length
    "gamma": 0.999,  # Dreamer의 discount
    "gae_lambda": 0.95,  # Dreamer의 lambda_
    "ent_coef": 0.0,  # 기본 설정
    "vf_coef": 0.5,  # 기본 설정
    "max_grad_norm": 100.0,  # Dreamer의 grad_clip
    "rms_prop_eps": 1e-5,  # Dreamer의 eps
    "normalize_advantage": False,
    "max_episodes": 1000,
    "max_steps": 2000,
})

# Environment creation function
def make_env():
    env = GymMoreRedBalls(room_size=20, render_mode="rgb_array")
    env = GymMoreRedBallsWrapper(env, max_steps=2000)
    env = ActionSpaceWrapper(env, max_steps=2000, new_action_space=3)
    env = FullyCustom(env, max_steps=2000)
    return MaxStepsWrapper(env, max_steps=2000)

# Create a vectorized environment
env = DummyVecEnv([make_env])

# Set up custom logger (for stdout, CSV, Tensorboard)
tmp_path = "/tmp/sb3_log/"
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

# Create the A2C model
model = A2C(
    "MlpPolicy",
    env,
    learning_rate=wandb.config.learning_rate,
    n_steps=wandb.config.n_steps,
    gamma=wandb.config.gamma,
    gae_lambda=wandb.config.gae_lambda,
    ent_coef=wandb.config.ent_coef,
    vf_coef=wandb.config.vf_coef,
    max_grad_norm=wandb.config.max_grad_norm,
    rms_prop_eps=wandb.config.rms_prop_eps,
    verbose=1,
    tensorboard_log=f"runs/{run.id}"
)

# Set the new logger
model.set_logger(new_logger)

# Create the callback to stop training after a number of episodes
episode_callback = EpisodeStoppingCallback(total_episodes=wandb.config.max_episodes)

# Train the model using the callback to stop after the specified number of episodes
model.learn(total_timesteps=int(1e9), callback=episode_callback)

# Finish the run
run.finish()
