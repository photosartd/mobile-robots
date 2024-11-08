import numpy as np
import json

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
    state=np.array([2, 2, 0]),
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

# The main function of the simulation. Gets called in a loop at most 60 times a second
# (depending on the computation time spent in this method). Passing a RoboSimPyWidget object as
# an argument allows for the usage of drawing commands, like "draw_lasers" or "draw_error_ellipse".
# The argument "world" provides access to the currently used world, while "inputs" is a list of
# pressed keys (for example ['a', 'w'] if the appropriate keys are currently pressed at the same time).
def state_update(api: RoboSimPyWidget, world, inputs):
    # The time step provided by the api.
    # Either a fixed value or the computation time of the last iteration in seconds.
    dt = api.dt

    # Setting speed for left and right wheels.
    vl, vr = 0, 0

    # Get a measurement from the simulated laser sensor and display it.
    laser_distances, hitpoints = shoot_lasers(
        robot.pos, robot.theta, robot.lasers, world
    )
    api.draw_lasers(robot.pos, hitpoints)
    print(laser_distances)

    vr = (0.01 if "e" in inputs else -0.01 if "d" in inputs else 0.0) * dt
    vl = (0.01 if "q" in inputs else -0.01 if "a" in inputs else 0.0) * dt

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
        meter2px=50,
        gui_scale=1.0,
        fixed_timestep=True,
    ).run()
