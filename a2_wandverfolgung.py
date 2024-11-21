from typing import Tuple

import numpy as np
from enum import Enum
import json
from collections import deque
from dataclasses import dataclass

from robosimpy.gui import *
from robosimpy.robots import Robot, Wheel
from robosimpy import worlds

try:
    import importlib.resources as pkg_resources
except:
    # importlib.resources is only available from Python 3.7,
    # but there is a fallback package
    import importlib_resources as pkg_resources

# The provided simple_world file gets loaded from the robosimpy package
with pkg_resources.open_text(worlds, "simple_world.json") as file:
    simple_world = json.load(file)

# Creating a robot with two wheels, a laser scanner with 8 beams,
# a simple, square enclosure and a starting pose.
robot = Robot(
    state=np.array([2, 3, 0]),
    initial_C_p=np.array([[0., 0, 0],
                          [0, 0., 0],
                          [0, 0, 0.]]),
    wheels={
        "vr": Wheel(+1.0 / 2.0 * np.pi, 0.23, 0, 0, steerable=False, motor=True),
        "vl": Wheel(-1.0 / 2.0 * np.pi, 0.23, np.pi, 0, steerable=False, motor=True),
    },
    lasers=np.linspace(np.pi / 2, -np.pi / 2, num=8, endpoint=True),
    enclosure=np.array(
        [
            [-0.3, +0.3, +0.3, +0.3],
            [+0.3, +0.3, +0.3, -0.3],
            [+0.3, -0.3, -0.3, -0.3],
            [-0.3, -0.3, -0.3, +0.3],
        ]
    ),
)


# CONST
TARGET_DISTANCE_TO_WALL = 0.5
MAX_ERROR = 0.1
TARGET_DISTANCE_TO_OBSTACLE = 0.7
SPEED_MULT_ON_TURN = 5
SPEED = 0.03
FOLLOW_RIGHT = False


def wall_following_control(
        laser_distances: np.ndarray,
        target_distance_to_wall: float = TARGET_DISTANCE_TO_WALL,
        max_error: float = MAX_ERROR,
        target_distance_to_obstacles: float = TARGET_DISTANCE_TO_OBSTACLE,
        speed_mult_on_turn: float = SPEED_MULT_ON_TURN,
        speed: float = SPEED,
        follow_right: bool = FOLLOW_RIGHT
) -> Tuple[float, float]:
    vl, vr = 0, 0
    wall_laser_idx = len(laser_distances) - 1 if follow_right else 0
    min_obstacle_distance = min(laser_distances[:-1]) if follow_right else min(laser_distances[1:])

    if min_obstacle_distance < target_distance_to_obstacles:
        # turn away from the obstacle
        if follow_right:
            vl, vr = -speed, speed
        else:
            vl, vr = speed, -speed
    else:
        # error from the needed distance
        error = laser_distances[wall_laser_idx] - target_distance_to_wall
        if error < max_error:
            # forward
            vl, vr = speed, speed
        else:
            # error is bigger - correct the speed
            #P - proportional scaling
            P = 1 / (abs(error) * speed_mult_on_turn)
            if follow_right:
                vl, vr = (speed, P * speed)
            else:
                vl, vr = P * speed, speed
    return vl, vr


# The main function of the simulation. Gets called in a loop at most 60 times a second
# (depending on the computation time spent in this method). Passing a RoboSimPyWidget object as
# an argument allows for the usage of drawing commands, like "draw_lasers" or "draw_error_ellipse".
# The argument "world" provides access to the currently used world, while "inputs" is a list of
# pressed keys (for example ['a', 'w'] if the appropriate keys are currently pressed at the same time).
def state_update(api: RoboSimPyWidget, world, inputs):
    # The time step provided by the api.
    # Either a fixed value or the computation time of the last iteration in seconds.
    dt = api.dt

    # Get a measurement from the simulated laser sensor and display it.
    laser_distances, hitpoints = shoot_lasers(
        robot.pos, robot.theta, robot.lasers, world
    )
    api.draw_lasers(robot.pos, hitpoints)
    print(laser_distances)

    vl, vr = wall_following_control(laser_distances)

    # Error propagation of odometry.
    robot.C_p = update_c_p(
        robot.C_p, robot.theta, vl, vr, dt, robot.wheels["vr"].l, 0.001, 0.001
    )
    api.draw_error_ellipse(robot.C_p, robot.pos)

    # Updating the robot pose using the previous pose, the time step, and motion commands for left and right wheel.
    robot.pos[0], robot.pos[1], robot.theta = update_pose(
        robot.pos[0], robot.pos[1], robot.theta, vl, vr, dt, robot.wheels["vr"].l
    )


# Path integration, see Siegwart/Nourbakhsh, p. 188
def update_pose(x, y, theta, vl, vr, dt, l):
    sl = dt * vl
    sr = dt * vr
    b = 2.0 * l
    slsr2 = (sl + sr) / 2.0
    srsl2b = (sr - sl) / (2.0 * b)
    x = x + slsr2 * np.cos(theta + srsl2b)
    y = y + slsr2 * np.sin(theta + srsl2b)
    theta = theta + 2.0 * srsl2b
    return np.array([x, y, theta])


# Motion jacobian, see Siegwart/Nourbakhsh, p. 189
def motion_jacobian(theta, vl, vr, dt, l):
    sl = dt * vl
    sr = dt * vr
    b = 2.0 * l
    ds = (sl + sr) / 2.0
    dtheta = (sr - sl) / b
    tt = theta + dtheta / 2.0
    ss = ds / b
    c = 0.5 * np.cos(tt)
    s = 0.5 * np.sin(tt)
    return np.array(
        [[c - ss * s, c + ss * s], [s + ss * c, s - ss * c], [1.0 / b, -1.0 / b]]
    )


# Pose jacobian, siehe Siegwart/Nourbakhsh, p. 189
def pose_jacobian(theta, vl, vr, dt, l):
    sl = dt * vl
    sr = dt * vr
    b = 2.0 * l
    ds = (sl + sr) / 2.0
    dtheta = (sr - sl) / b
    tt = theta + dtheta / 2.0
    c = np.cos(tt)
    s = np.sin(tt)
    return np.array([[1.0, 0.0, -ds * s], [0.0, 1.0, ds * c], [0.0, 0.0, 1.0]])


# Motion covariance, see siehe Siegwart/Nourbakhsh, p. 188
def motion_covariance(vl, vr, dt, kl, kr):
    sl = dt * vl
    sr = dt * vr
    return np.array([[kr * abs(sr), 0.0], [0.0, kl * abs(sl)]])


# Update of odometry error estimation
def update_c_p(C_p, theta, vl, vr, dt, l, kl, kr):
    f_p = pose_jacobian(theta, vl, vr, dt, l)
    f_delta = motion_jacobian(theta, vl, vr, dt, l)
    c_delta = motion_covariance(vl, vr, dt, kl, kr)
    return f_p @ C_p @ f_p.T + f_delta @ c_delta @ f_delta.T


if __name__ == "__main__":
    RoboSimPyApp(
        robot=robot,
        state_update=state_update,
        world=simple_world,
        meter2px=100,
        gui_scale=1.0,
        fixed_timestep=True,
    ).run()
