"""Random initial ball states, like a ball launcher on the human's side.

Convention (same as physics.py):
    origin at table centre on the floor, x along the table, y across, z up.
    Robot side:  x < 0.   Human / launcher side: x > 0.
    Launched balls fly toward -x.

Spin signs for a ball flying toward -x:
    topspin   -> omega_y < 0
    backspin  -> omega_y > 0
    sidespin  -> omega_z != 0 (sign = which way it curves)

These functions ONLY sample a starting state. They do not check that the
ball is playable; must reject bad samples after simulating
"""

from __future__ import annotations

import numpy as np

from . import constants as C

L = C.TABLE_LENGTH
H = C.TABLE_HEIGHT

# ---------------------------------------------------------------------------
# Ranges (low, high). Positions in m, speeds in m/s, angles in deg, spin in rad/s.
# ---------------------------------------------------------------------------

RETURN_RANGES = {
    "x_behind_end": (0.2, 0.8),   # distance behind the human's end line
    "y": (-0.6, 0.6),
    "z_above_table": (0.0, 0.4),
    "speed": (5.0, 10.0),
    # Elevation depends on speed: slow balls need a higher arc to clear the
    # net and still land on the table, fast balls a flatter one.
    # elevation = a + b * speed + uniform(-spread, spread), positive = upward.
    "elevation_line": (50.0, -4.0),  # 5 m/s -> ~30 deg, 10 m/s -> ~10 deg
    "elevation_spread": 10.0,
    "aim_y": (-0.6, 0.6),         # rough aim point across the robot's half
    "spin_main": (0.0, 200.0),    # top/backspin magnitude
    "p_topspin": 0.8,             # probability the main spin is topspin
    "sidespin": (-50.0, 50.0),
}

SERVE_RANGES = {
    "x_behind_end": (0.0, 0.3),
    "y": (-0.6, 0.6),
    "z_above_table": (0.05, 0.30),
    "speed": (4.0, 6.5),
    # Downward: the serve must bounce on the server's side first.
    "elevation_line": (-22.5, 0.0),  # -35..-10 deg, independent of speed
    "elevation_spread": 12.5,
    "aim_y": (-0.6, 0.6),
    "spin_main": (0.0, 300.0),
    "p_topspin": 0.3,             # serves are mostly backspin
    "sidespin": (-200.0, 200.0),
}

# Point the launcher roughly aims at: middle of the robot's half.
AIM_X = -L / 4


def _uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    return float(rng.uniform(bounds[0], bounds[1]))


def _velocity(speed: float, elevation_deg: float, azimuth_rad: float) -> np.ndarray:
    """Velocity toward -x. Azimuth 0 = straight down the table; it rotates
    the direction in the x-y plane. Elevation tilts it up (+) or down (-)."""
    el = np.deg2rad(elevation_deg)
    horizontal = speed * np.cos(el)
    return np.array([
        -horizontal * np.cos(azimuth_rad),
        -horizontal * np.sin(azimuth_rad),
        speed * np.sin(el),
    ])


def _sample(rng: np.random.Generator, r: dict) -> np.ndarray:
    x = L / 2 + _uniform(rng, r["x_behind_end"])
    y = _uniform(rng, r["y"])
    z = H + _uniform(rng, r["z_above_table"])

    # Aim azimuth from the launch point toward a random spot on the robot's
    # half. Purely random azimuths would send most balls off the side.
    aim_y = _uniform(rng, r["aim_y"])
    azimuth = np.arctan2(y - aim_y, x - AIM_X)
    speed = _uniform(rng, r["speed"])
    a, b = r["elevation_line"]
    spread = r["elevation_spread"]
    elevation = a + b * speed + _uniform(rng, (-spread, spread))
    v = _velocity(speed, elevation, azimuth)

    main = _uniform(rng, r["spin_main"])
    omega_y = -main if rng.random() < r["p_topspin"] else main
    omega = np.array([0.0, omega_y, _uniform(rng, r["sidespin"])])

    return np.concatenate(([x, y, z], v, omega))


def launcher_return(rng: np.random.Generator) -> np.ndarray:
    """A rally ball: hit from behind the human's end, flies over the net,
    should bounce once on the robot's side. Returns a (9,) state."""
    return _sample(rng, RETURN_RANGES)


def launcher_serve(rng: np.random.Generator) -> np.ndarray:
    """A serve: hit downward near the human's end, should bounce on the
    human's side, then on the robot's side. Returns a (9,) state."""
    return _sample(rng, SERVE_RANGES)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    np.set_printoptions(precision=2, suppress=True)
    print("return:", launcher_return(rng))
    print("serve: ", launcher_serve(rng))
