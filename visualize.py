"""Play one ball with the trained model and show it.

    python visualize.py        -> a random ball
    python visualize.py 42     -> the ball for seed 42 (same ball every time)
"""
import sys

import numpy as np
from stable_baselines3 import SAC

from src.gym import PingPongEnv

MODEL_PATH = "models/sac_pingpong_10k"

if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else None

    env = PingPongEnv(render_mode="human")
    model = SAC.load(MODEL_PATH)

    obs, info = env.reset(seed=seed)
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    np.set_printoptions(precision=3, suppress=True)
    print(f"seed:          {seed}")
    print(f"action [-1,1]: {action}")
    print(f"outcome:       {info['outcome']}")
    print(f"target:        {info['target']}")
    print(f"landed at:     {info['landing_point']}")
    print(f"landing error: {info['landing_error']:.3f} m")
    print(f"reward:        {reward:.3f}")

    env.render()
