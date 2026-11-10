from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

from src.gym import PingPongEnv
from src.plot import plot_landing_map

N_BALLS = 300
EVAL_SEED = 0
MODEL_PATH = "models/sac_pingpong"
# alpha = 0, beta = 0, speed = 2 m/s (action[2] = +1 maps to MAX_RACKET_SPEED)
FIXED_ACTION = np.array([0.0, 0.0, 1.0], dtype=np.float32)


def evaluate(env, policy, n_balls, seed):
    """Play n_balls with policy(obs) -> action. Same seed = same balls."""
    results = {"landing_points": [], "targets": [], "outcomes": [],
               "landing_errors": [], "rewards": []}

    obs, info = env.reset(seed=seed)
    for _ in range(n_balls):
        obs, reward, terminated, truncated, info = env.step(policy(obs))

        results["landing_points"].append(info["landing_point"])
        results["targets"].append(info["target"])
        results["outcomes"].append(info["outcome"])
        results["rewards"].append(reward)
        if not info["is_fault"]:
            results["landing_errors"].append(info["landing_error"])

        obs, info = env.reset()
    return results


def print_summary(name, results):
    n = len(results["outcomes"])
    errors = results["landing_errors"]
    print(name)
    print(f"  return rate:   {len(errors) / n:.1%}")
    if errors:
        print(f"  landing error: {np.mean(errors):.3f} ± {np.std(errors):.3f} m (landed balls only)")
    print(f"  mean reward:   {np.mean(results['rewards']):.3f}")
    print(f"  outcomes:      {dict(Counter(results['outcomes']))}\n")


if __name__ == "__main__":
    env = PingPongEnv()
    check_env(env)

    model = SAC.load(MODEL_PATH)

    policies = {
        f"Fixed action {FIXED_ACTION}": lambda obs: FIXED_ACTION,
        "Learned policy (SAC)": lambda obs: model.predict(obs, deterministic=True)[0],
    }

    fig, axes = plt.subplots(len(policies), 1, figsize=(10, 5 * len(policies)))
    for ax, (name, policy) in zip(axes, policies.items()):
        results = evaluate(env, policy, N_BALLS, EVAL_SEED)
        print_summary(name, results)
        plot_landing_map(results["landing_points"], results["targets"], results["outcomes"],
                         ax=ax, title=f"{name}, {N_BALLS} balls")

    plt.show()
