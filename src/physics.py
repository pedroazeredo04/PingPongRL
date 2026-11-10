"""
Numerical simulation and ball-contact models.

This module deliberately has no plotting, notebook, CLI, or SciPy dependency.
"""

import numpy as np
from enum import Enum

from .constants import (
    AIR_DENSITY,
    BALL_MASS,
    BALL_RADIUS,
    BALL_ROT_INERTIA,
    DRAG_COEFFICIENT,
    DRAG_CONSTANT,
    DT,
    GRAVITY,
    MAGNUS_CONSTANT,
    NET_OVERHANG_PER_SIDE,
    NET_HEIGHT,
    NET_RESTITUTION,
    TABLE_FRICTION,
    TABLE_HEIGHT,
    TABLE_LENGTH,
    TABLE_WIDTH,
    RACKET_RESTITUTION,
    RACKET_FRICTION,
    T_MAX,
    STEP_LIMIT,
    HITTING_PLANE_X,
)

class Event(Enum):
    OUT = 1
    NET = 2
    TABLE = 3
    HITTING_PLANE = 4
    NONE = -1

class TrajectoryOutcome(Enum):
    VALID = 1
    DOUBLE_BOUNCE = 2
    NET = 3
    OUT = 4
    NO_BOUNCE = 5
    SAME_SIDE_BOUNCE = 6

# Applies RK4 to the ball_state
# returns a new ball_state after a time dt

# Note: Ball state consists of:
#  ball_position = ball_state[0:3]
#  ball_lin_vel = ball_state[3:6]
#  ball_ang_vel = ball_state[6:9]
def step(ball_state: np.ndarray) -> np.ndarray:
    k1 = evaluate(ball_state)
    k2 = evaluate(ball_state + (DT/2) * k1)
    k3 = evaluate(ball_state + (DT/2) * k2)
    k4 = evaluate(ball_state + DT * k3)

    next_state = ball_state + (DT/6) * (k1 + 2*k2 + 2*k3 + k4)

    return next_state
    

def evaluate(state: np.ndarray) -> np.ndarray:
    assert state.shape == (9,)
    position = state[0:3]
    lin_vel = state[3:6]
    ang_vel = state[6:9]

    Kd = DRAG_CONSTANT
    Km = MAGNUS_CONSTANT
    speed = np.linalg.norm(lin_vel)

    lin_acc = - Kd * lin_vel * speed + Km * np.cross(ang_vel, lin_vel) - np.array([0, 0, GRAVITY])

    dstate_dt = np.concatenate((lin_vel, lin_acc, np.zeros(3)))

    return dstate_dt

# Returns the event and where it happened.
def detect_event(state: np.ndarray, next_state: np.ndarray) -> tuple[Event, np.ndarray]:
    x_pos = state[0]
    y_pos = state[1]
    z_pos = state[2]

    x_pos_next = next_state[0]
    y_pos_next = next_state[1]
    z_pos_next = next_state[2]

    # Table collision
    # This ignores collisions on the edge of the table.
    #  I'm assuming robot wouldn't catch those anyways, 
    #  so they will just be a ball out at some point
    if ((-TABLE_LENGTH/2 < x_pos < +TABLE_LENGTH/2) and
        (-TABLE_WIDTH/2 < y_pos < +TABLE_WIDTH/2) and
        (z_pos > TABLE_HEIGHT + BALL_RADIUS) and 
        (z_pos_next <= TABLE_HEIGHT + BALL_RADIUS)):

        f = (z_pos - (TABLE_HEIGHT + BALL_RADIUS))/(z_pos - z_pos_next)
        event_state = state + f * (next_state - state) # Interpolated
        
        return Event.TABLE, event_state
    
    # OUT
    # Only when ball hits the floor. 
    #  Otherwise, ball is still in game.
    if (z_pos_next < 0.0):
        # Doesn't interpolate output, whatever
        return Event.OUT, next_state

    # NET
    #  x_pos and x_pos_next have different signals
    if ((x_pos * x_pos_next < 0.0) and
        (-TABLE_WIDTH/2 - NET_OVERHANG_PER_SIDE < y_pos < +TABLE_WIDTH/2 + NET_OVERHANG_PER_SIDE) and
        (TABLE_HEIGHT - BALL_RADIUS < z_pos < TABLE_HEIGHT + BALL_RADIUS + NET_HEIGHT)):

        f = (x_pos)/(x_pos - x_pos_next)
        event_state = state + f * (next_state - state) # Interpolated

        return Event.NET, event_state

    # Hitting plane event
    if ((x_pos > HITTING_PLANE_X) and
        (x_pos_next <= HITTING_PLANE_X) and 
        (z_pos > TABLE_HEIGHT)):

        f = (x_pos - HITTING_PLANE_X)/(x_pos - x_pos_next)
        event_state = state + f * (next_state - state) # Interpolated

        return Event.HITTING_PLANE, event_state

    return Event.NONE, next_state


# Steps the simulation until the incoming ball is out of play OR reaches the HITTING_PLANE.
# Returns the state_list (to plot), the TrajectoryOutcome and the event_state.
def simulate_incoming(initial_state: np.ndarray, is_serve: bool = False) -> tuple[np.ndarray, TrajectoryOutcome, np.ndarray]:
    state = initial_state
    state_list = [initial_state]

    n_sender_side_bounces = 0
    n_receiver_side_bounces = 0

    for _ in range(STEP_LIMIT):
        next_state = step(state)
        event, event_state = detect_event(state, next_state)

        if event == Event.NONE:
            state_list.append(next_state)
            state = next_state
            continue

        # Something happened: record where, then decide what it means.
        state_list.append(event_state)

        if event == Event.TABLE:
            if event_state[0] > 0:
                n_sender_side_bounces += 1
            else:
                n_receiver_side_bounces += 1

            if n_sender_side_bounces > 1 or n_receiver_side_bounces > 1:
                return np.array(state_list), TrajectoryOutcome.DOUBLE_BOUNCE, event_state

            if n_sender_side_bounces > 0 and not is_serve:
                # A return (non-serve) must go straight to the receiver's side.
                return np.array(state_list), TrajectoryOutcome.SAME_SIDE_BOUNCE, event_state

            if n_receiver_side_bounces > 0 and is_serve and n_sender_side_bounces == 0:
                # A serve must bounce on the sender's side first.
                return np.array(state_list), TrajectoryOutcome.NO_BOUNCE, event_state

            state = bounce_table(event_state)
            continue

        if event == Event.NET:
            return np.array(state_list), TrajectoryOutcome.NET, event_state

        if event == Event.OUT:
            return np.array(state_list), TrajectoryOutcome.OUT, event_state

        # Event.HITTING_PLANE: legal only after exactly one bounce on the receiver's side.
        # (The serve rule was already checked at that bounce.)
        if n_receiver_side_bounces == 0:
            return np.array(state_list), TrajectoryOutcome.NO_BOUNCE, event_state

        return np.array(state_list), TrajectoryOutcome.VALID, event_state

    # Step limit reached without the phase ending: treat as out.
    return np.array(state_list), TrajectoryOutcome.OUT, state


# Steps the simulation until the outgoing ball lands, hits the net, or goes out.
# Returns the state_list (to plot), the TrajectoryOutcome and the event_state.
def simulate_outgoing(initial_state: np.ndarray) -> tuple[np.ndarray, TrajectoryOutcome, np.ndarray]:
    state = initial_state
    state_list = [initial_state]

    for _ in range(STEP_LIMIT):
        next_state = step(state)
        event, event_state = detect_event(state, next_state)

        if event == Event.NONE:
            state_list.append(next_state)
            state = next_state
            continue

        # Something happened: record where, then decide what it means.
        state_list.append(event_state)

        if event == Event.TABLE:
            if event_state[0] > 0:
                outcome = TrajectoryOutcome.VALID
            else:
                outcome = TrajectoryOutcome.SAME_SIDE_BOUNCE
        elif event == Event.NET:
            outcome = TrajectoryOutcome.NET
        else:
            # OUT. (HITTING_PLANE can't fire here: it only triggers for balls moving toward -x.)
            outcome = TrajectoryOutcome.OUT

        return np.array(state_list), outcome, event_state

    # Step limit reached without any event: treat as out.
    return np.array(state_list), TrajectoryOutcome.OUT, state


# Receives: 
#  1. ball_state before hit
#  2. racket_state (what the RL outputs):
#   racket_state[0:3] -> racket_normal
#   racket_state[3:6] -> racket_velocity

# Outputs: 
# New ball_state after collision with the racket
def hit(ball_state: np.ndarray, racket_state: np.ndarray) -> np.ndarray:
    ball_position = ball_state[0:3]
    ball_velocity = ball_state[3:6]
    ball_ang_vel = ball_state[6:9]

    racket_normal = racket_state[0:3]
    racket_velocity = racket_state[3:6]

    assert np.isclose(np.linalg.norm(racket_normal), 1.0)

    # Coordinate frame: RACKET ANGLE
    relative_velocity = ball_velocity - racket_velocity
    relative_normal_velocity = ( relative_velocity @ racket_normal ) * racket_normal
    relative_tangent_velocity = relative_velocity - relative_normal_velocity

    new_relative_normal_velocity = - RACKET_RESTITUTION * relative_normal_velocity
    new_relative_tangent_velocity = + RACKET_FRICTION * relative_tangent_velocity
    new_relative_velocity = new_relative_normal_velocity + new_relative_tangent_velocity

    # Back to world coordinate frame
    new_ball_velocity = new_relative_velocity + racket_velocity

    return np.concatenate((ball_position, new_ball_velocity, ball_ang_vel))


# Bounce table
def bounce_table(state: np.ndarray) -> np.ndarray:
    state_copy = state.copy()

    # Reflect linear velocity in z axis:
    state_copy[5] *= -1

    # Ensure z position is exactly at table height
    state_copy[2] = TABLE_HEIGHT + BALL_RADIUS

    return state_copy
