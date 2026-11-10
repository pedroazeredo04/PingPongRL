from typing import Optional
import numpy as np
import gymnasium as gym

from .constants import TABLE_LENGTH, TABLE_WIDTH

from .physics import *
from .launcher import *
from .plot import *

MAX_LAUNCHER_SAMPLE_RETRIES = 1000
TARGET_MARGIN_NET  = 0.30   # m, keep targets this far behind the net
TARGET_MARGIN_EDGE = 0.15   # m, keep targets this far inside the end line and sides

MAX_RACKET_SPEED = 2.0
MAX_RACKET_TILT = np.deg2rad(20)  # 20 degrees
MAX_RACKET_TURN = np.deg2rad(30)  # 30 degrees

FALLING_SPEED_WEIGHT = 0.01
LANDING_ERROR_WEIGHT = 1.0
FAULT_WEIGHT = 2.0

def get_reward(event_state, traj_outcome, target):
    final_ball_pos_2D = event_state[0:2]
    landing_error_penalty = - LANDING_ERROR_WEIGHT * np.linalg.norm(final_ball_pos_2D - target)

    fault_penalty = 0 if traj_outcome==TrajectoryOutcome.VALID else - FAULT_WEIGHT

    # Only apply if ball landed
    vz = event_state[5]
    falling_speed_penalty = - FALLING_SPEED_WEIGHT * abs(vz) if  (traj_outcome==TrajectoryOutcome.VALID) else 0

    return landing_error_penalty + falling_speed_penalty + fault_penalty

class PingPongEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, p_serve: float = 0.0, render_mode: Optional[str] = None):
        self.p_serve = p_serve
        self.render_mode = render_mode

        # OBSERVATION SPACE:
        #  What the agent sees to make a decision.

        # Initialize to invalid
        self._ball_state_at_hitting_plane = np.zeros(9)
        self._target = np.zeros(2)

        # Vector with 11 real numbers:
        #  obs_space[0:3]  -> Ball position at hit
        #  obs_space[3:6]  -> Ball linear velocity at hit
        #  obs_space[6:9]  -> Ball angular velocity at hit
        #  obs_space[9:11] -> Target position at the table.
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)

        # Vector with 3 numbers:
        #  action[0] -> Racket tilt (up or down)
        #  action[1] -> Racket turn (left or right)
        #  action[2] -> racket speed in the angle direction
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def _action_to_racket_state(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64)

        racket_tilt_rad = action[0] * MAX_RACKET_TILT
        racket_turn_rad = action[1] * MAX_RACKET_TURN
        racket_speed_abs = (action[2] + 1) / 2 * MAX_RACKET_SPEED

        racket_normal = np.array([np.cos(racket_tilt_rad) * np.cos(racket_turn_rad), np.cos(racket_tilt_rad) * np.sin(racket_turn_rad), np.sin(racket_tilt_rad)])
        # Not (racket_speed_abs, 0, 0). 
        # Will move in the same direction as the normal
        racket_speed = racket_normal * racket_speed_abs

        return np.concatenate([racket_normal, racket_speed])

    def _get_obs(self):
        OBS_SCALE = np.array([1, 1, 1, 5, 5, 5, 200, 200, 200, 1, 1])

        obs = np.concatenate((self._ball_state_at_hitting_plane, self._target))

        return (obs / OBS_SCALE).astype(np.float32)


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        is_serve = (self.np_random.random() < self.p_serve)

        traj_valid = False
        n_retries = 0

        while not traj_valid and (n_retries < MAX_LAUNCHER_SAMPLE_RETRIES):
            n_retries += 1

            if is_serve:
                initial_state = launcher_serve(self.np_random)
            else:
                initial_state = launcher_return(self.np_random)
            
            self._traj_incoming, traj_outcome, self._ball_state_at_hitting_plane = simulate_incoming(initial_state, is_serve)
            
            if traj_outcome == TrajectoryOutcome.VALID:
                traj_valid = True

        if not traj_valid:
            raise RuntimeError("launcher produced no valid ball in N tries")

        target_x = self.np_random.uniform(TARGET_MARGIN_NET,  TABLE_LENGTH/2 - TARGET_MARGIN_EDGE)
        target_y = self.np_random.uniform(-(TABLE_WIDTH/2 - TARGET_MARGIN_EDGE),  TABLE_WIDTH/2 - TARGET_MARGIN_EDGE)
        self._target = np.array([target_x, target_y])

        return self._get_obs(), {}


    def step(self, action):
        racket_state = self._action_to_racket_state(action)

        ball_state_after_hit = hit(self._ball_state_at_hitting_plane, racket_state)
        self._traj_outgoing, traj_outcome, event_state = simulate_outgoing(ball_state_after_hit)

        reward = get_reward(event_state, traj_outcome, self._target)
        terminated = True
        truncated = False

        info = {
            "outcome": traj_outcome.name,
            "landing_point": event_state[0:2],
            "landing_error": np.linalg.norm(event_state[0:2] - self._target),
            "is_fault": traj_outcome != TrajectoryOutcome.VALID,
            "target": self._target.copy(),
        }

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return
        fig, ax = plot_trajectories(
            [self._traj_incoming, self._traj_outgoing], labels=["incoming", "outgoing"]
        )
        tx, ty = self._target
        ax.scatter(tx, ty, TABLE_HEIGHT, marker="x", s=80, color="red", label="target")
        ax.legend()
        plt.show()
