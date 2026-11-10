"""Matplotlib plots for ball trajectories.

This module only draws. It never imports physics.py and never simulates:
you hand it arrays, it hands you a figure.

Coordinate convention:
    origin  = on the floor, at the exact middle of the table
    x       = along the table, from the robot (x = -TABLE_LENGTH/2) to the opponent (x = TABLE_LENGTH/2)
    y       = across the table, centre line at y = 0
    z       = up; the table surface is at z = TABLE_HEIGHT

A trajectory is any array of shape (T, >=3) whose first three columns are
the ball position. Your (T, 9) state history works as-is.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from . import constants as C

TABLE_COLOR = "#2e6f9e"
LINE_COLOR = "white"
NET_COLOR = "0.25"


def _positions(trajectory: np.ndarray) -> np.ndarray:
    """Return the (T, 3) position columns, with a clear error for bad shapes."""
    traj = np.asarray(trajectory, dtype=float)
    if traj.ndim != 2 or traj.shape[1] < 3:
        raise ValueError(
            f"trajectory must have shape (T, >=3), got {traj.shape}. "
            "Did you forget np.array(list_of_states)?"
        )
    return traj[:, :3]


def _as_list(trajectories) -> list[np.ndarray]:
    """Accept a single (T, >=3) array or a list of them; always return a list."""
    if isinstance(trajectories, np.ndarray):
        _positions(trajectories)  # fail early with the real shape in the message
        return [trajectories]
    return list(trajectories)


def draw_table(ax) -> None:
    """Draw the table surface, its lines and the net on a 3D axis."""
    L, W, H = C.TABLE_LENGTH, C.TABLE_WIDTH, C.TABLE_HEIGHT
    half_w = W / 2

    surface = [[(-L/2, -half_w, H), (L/2, -half_w, H), (L/2, half_w, H), (-L/2, half_w, H)]]
    ax.add_collection3d(
        Poly3DCollection(surface, facecolor=TABLE_COLOR, alpha=0.35, edgecolor=LINE_COLOR)
    )
    # Centre line; the white border comes from the polygon's edgecolor.
    ax.plot([-L/2, L/2], [0, 0], [H, H], color=LINE_COLOR, lw=0.8)

    # Net: a vertical rectangle at mid-table, overhanging the sides by NET_OVERHANG_PER_SIDE.
    net_x = 0.0
    net_half = half_w + C.NET_OVERHANG_PER_SIDE
    net = [[
        (net_x, -net_half, H),
        (net_x, net_half, H),
        (net_x, net_half, H + C.NET_HEIGHT),
        (net_x, -net_half, H + C.NET_HEIGHT),
    ]]
    ax.add_collection3d(
        Poly3DCollection(net, facecolor=NET_COLOR, alpha=0.4, edgecolor=NET_COLOR)
    )


def _set_equal_aspect(ax, points: np.ndarray) -> None:
    """Make 1 m look like 1 m on every axis (matplotlib 3D doesn't by default)."""
    L, W = C.TABLE_LENGTH, C.TABLE_WIDTH
    lo = np.minimum(points.min(axis=0), [-L/2, -W / 2, 0.0])
    hi = np.maximum(points.max(axis=0), [+L/2, W / 2, C.TABLE_HEIGHT + C.NET_HEIGHT])
    pad = 0.05 * (hi - lo).max()
    lo, hi = lo - pad, hi + pad
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo, zoom=1.2)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(5))


def plot_trajectories(
    trajectories: np.ndarray | list[np.ndarray],
    labels: list[str] | None = None,
    ax=None,
    show_table: bool = True,
    title: str | None = None,
):
    """Plot one or more trajectories in 3D over the table.

    trajectories: a single (T, >=3) array, or a list of them.
    labels:       optional legend entry per trajectory.
    ax:           an existing 3D axis to draw into; a new figure is made if None.
    Returns (fig, ax) so the caller decides whether to show() or savefig().
    """
    trajectories = _as_list(trajectories)
    if labels is not None and len(labels) != len(trajectories):
        raise ValueError("need one label per trajectory")

    if ax is None:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(projection="3d")
    else:
        fig = ax.figure

    if show_table:
        draw_table(ax)

    all_points = []
    for i, traj in enumerate(trajectories):
        pos = _positions(traj)
        label = labels[i] if labels is not None else None
        (line,) = ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], lw=1.8, label=label)
        # Mark the start so direction of flight is obvious.
        ax.scatter(*pos[0], color=line.get_color(), s=20)
        # Mark end
        ax.scatter(*pos[-1], facecolors="none", edgecolors=line.get_color(), s=50)
        all_points.append(pos)

    _set_equal_aspect(ax, np.vstack(all_points))
    ax.set_xlabel("x [m] (along table)")
    ax.set_ylabel("y [m] (across)")
    ax.set_zlabel("z [m]")
    ax.view_init(elev=C.PITCH, azim=C.YAW)
    if labels is not None:
        ax.legend(loc="upper left")
    if title:
        ax.set_title(title)
    return fig, ax


def plot_side_view(
    trajectories: np.ndarray | list[np.ndarray],
    labels: list[str] | None = None,
    ax=None,
    title: str | None = None,
):
    """2D x-z view (looking from the side). Best for comparing heights,
    e.g. drag vs no drag, topspin vs backspin, sim vs analytic parabola."""
    trajectories = _as_list(trajectories)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    L, H = C.TABLE_LENGTH, C.TABLE_HEIGHT
    ax.plot([-L/2, L/2], [H, H], color=TABLE_COLOR, lw=4, solid_capstyle="butt", label="table")
    ax.plot([0, 0], [H, H + C.NET_HEIGHT], color=NET_COLOR, lw=3, label="net")

    for i, traj in enumerate(trajectories):
        pos = _positions(traj)
        label = labels[i] if labels is not None else None
        ax.plot(pos[:, 0], pos[:, 2], lw=1.8, label=label)

    ax.set_aspect("equal")
    ax.set_xlabel("x [m] (along table)")
    ax.set_ylabel("z [m]")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    if title:
        ax.set_title(title)
    return fig, ax


if __name__ == "__main__":
    # Self-test with an analytic parabola (no physics.py needed):
    # python -m src.plotting
    t = np.linspace(0.0, 0.55, 200)
    p0 = np.array([C.TABLE_LENGTH, 0.0, 1.0])
    v0 = np.array([-5.0, 0.4, 1.5])
    pos = p0 + np.outer(t, v0) + np.outer(0.5 * t**2, [0.0, 0.0, -C.GRAVITY])

    plot_trajectories(pos, labels=["analytic parabola"], title="Self-test")
    plot_side_view(pos, labels=["analytic parabola"])
    plt.show()


# Outcome colors: fixed categorical order (checked to be colorblind-safe), plus a
# distinct marker per outcome so identity never relies on color alone.
OUTCOME_STYLE = {
    "VALID":            ("#2a78d6", "o", "landed"),
    "NET":              ("#eb6834", "s", "net"),
    "OUT":              ("#1baf7a", "^", "out"),
    "SAME_SIDE_BOUNCE": ("#eda100", "D", "own side"),
}
TARGET_COLOR = "0.35"


def plot_landing_map(landing_points, targets, outcomes, ax=None, title=None):
    """Top-down view of where each returned ball came down vs. where it was aimed.

    landing_points: (N, 2) array of (x, y) where the outgoing ball stopped.
    targets:        (N, 2) array of (x, y) targets.
    outcomes:       N outcome names, e.g. "VALID", "NET" (TrajectoryOutcome.name).
    A thin grey line joins each target to its landing point: the miss.
    """
    landing_points = np.asarray(landing_points, dtype=float)
    targets = np.asarray(targets, dtype=float)
    outcomes = np.asarray(outcomes)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5.5))
    else:
        fig = ax.figure

    # Table outline, centre line and net, seen from above.
    L, W = C.TABLE_LENGTH, C.TABLE_WIDTH
    ax.add_patch(plt.Rectangle((-L / 2, -W / 2), L, W,
                               facecolor=TABLE_COLOR, alpha=0.15,
                               edgecolor=TABLE_COLOR, lw=1.5, zorder=0))
    ax.plot([-L / 2, L / 2], [0, 0], color=TABLE_COLOR, lw=0.8, alpha=0.6, zorder=0)
    net_half = W / 2 + C.NET_OVERHANG_PER_SIDE
    ax.plot([0, 0], [-net_half, net_half], color=NET_COLOR, lw=3, zorder=1)

    # Miss lines: target -> landing point.
    for t, p in zip(targets, landing_points):
        ax.plot([t[0], p[0]], [t[1], p[1]], color="0.6", lw=0.8, alpha=0.6, zorder=2)

    ax.scatter(targets[:, 0], targets[:, 1], marker="x", s=30, lw=1.2,
               color=TARGET_COLOR, zorder=3, label="target")

    n = len(outcomes)
    for name, (color, marker, label) in OUTCOME_STYLE.items():
        mask = outcomes == name
        if mask.any():
            ax.scatter(landing_points[mask, 0], landing_points[mask, 1],
                       marker=marker, s=36, color=color, edgecolor="white", lw=0.8,
                       zorder=4, label=f"{label} ({mask.sum()}/{n})")

    ax.set_aspect("equal")
    ax.set_xlabel("x [m] (robot side < 0 < human side)")
    ax.set_ylabel("y [m]")
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax