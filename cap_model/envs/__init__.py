from gymnasium.envs.registration import register

register(
    id='GymMoreRedBalls-v0',
    entry_point='envs.GymMoreRedBalls:GymMoreRedBalls',
    max_episode_steps=1000,
)
