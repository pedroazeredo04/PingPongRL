from stable_baselines3 import SAC
from src.gym import PingPongEnv

env = PingPongEnv()
machine = SAC.load("models/sac_pingpong", env=env)

machine.learn(total_timesteps=5_000, reset_num_timesteps=False)
machine.save("models/sac_pingpong")
