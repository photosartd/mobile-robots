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


# a) Implementieren Sie die grafische Visualisierung des Dreirads. Achten Sie dabei darauf, dass sich alle
# R¨ ader innerhalb der ¨ außeren H¨ ulle des Roboters befinden, damit die Kollisionsdetektion mit den W¨ anden
# funktioniert.
def get_enclosure(w, h, center=(0,0)):
    """
    Parameters
    ----------
    w - width of the robot
    h - height of the robot
    center - center of the robot (x,y). If differs from 0, the enclosure won't be symmetric around the center of the
    robot

    Returns
    -------

    """
    x, y = center
    # half of sides
    hw = w / 2
    hh = h / 2
    # centered halves
    hw_left = hw + x
    hw_right = hw - x

    hh_up = hh - y
    hh_down = hh + y

    enclosure = np.array(
        [
            [-hw_left, +hh_up, +hw_right, +hh_up],
            [+hw_right, +hh_up, +hw_right, -hh_down],
            [+hw_right, -hh_down, -hw_left, -hh_down],
            [-hw_left, -hh_down, -hw_left, +hh_up],
        ]
    )
    return enclosure

beta_prev = 0

robot = Robot(
    state=np.array([2, 2, 0]),
    initial_C_p=np.array([[0., 0, 0],
                        [0, 0., 0],
                        [0, 0, 0.]]),
    wheels={
        "hl": Wheel(+1.0 / 2.0 * np.pi, 0.23, 0, 0, steerable=False, motor=False),
        "hr": Wheel(-1.0 / 2.0 * np.pi, 0.23, np.pi, 0, steerable=False, motor=False),
        "v": Wheel(0, 0.46, beta_prev, 0, steerable=True, motor=True)
    },
    lasers=np.linspace(np.pi / 2, -np.pi / 2, num=8, endpoint=True),
    enclosure=get_enclosure(1.0, 0.6, center=(-0.2, 0))
)

# b) Implementieren Sie die Kinematik und Pfadintegration des Dreirads. Zeigen Sie anhand einer einfa-
# chen Tastatursteuerung, dass Sie das Gef¨ ahrt im Simulationsraum steuern konnen.
# The main function of the simulation. Gets called in a loop at most 60 times a second
# (depending on the computation time spent in this method). Passing a RoboSimPyWidget object as
# an argument allows for the usage of drawing commands, like "draw_lasers" or "draw_error_ellipse".
# The argument "world" provides access to the currently used world, while "inputs" is a list of
# pressed keys (for example ['a', 'w'] if the appropriate keys are currently pressed at the same time).
def state_update(api: RoboSimPyWidget, world, inputs):
    # The time step provided by the api.
    # Either a fixed value or the computation time of the last iteration in seconds.
    global beta_prev
    dt = api.dt

    # Setting speed for the Vorderrad
    v = 0
    beta = robot.wheels["v"].beta

    # Get a measurement from the simulated laser sensor and display it.
    laser_distances, hitpoints = shoot_lasers(
        robot.pos, robot.theta, robot.lasers, world
    )
    api.draw_lasers(robot.pos, hitpoints)
    print(laser_distances)

    # speed
    v = (0.01 if "w" in inputs else -0.01 if "s" in inputs else 0.0) * dt
    # angle
    beta = (beta - 0.02 if "a" in inputs else beta + 0.02 if "d" in inputs else beta)
    # Note: for some reason the directions in the gui are different
    delta_beta = np.arccos(np.cos(beta - beta_prev))
    beta_prev = beta
    robot.wheels["v"].beta = beta

    # Error propagation of odometry.
    robot.C_p = update_c_p(
        robot.C_p, robot.theta, v, dt, robot.wheels["v"].l, beta, 0.001, 0.001, delta_beta
    )
    api.draw_error_ellipse(robot.C_p, robot.pos)

    # Updating the robot pose using the previous pose, the time step, and motion commands for left and right wheel.
    robot.pos[0], robot.pos[1], robot.theta = update_pose(
        robot.pos[0], robot.pos[1], robot.theta, v, dt, robot.wheels["v"].l, beta
    )

# Path integration, see Siegwart/Nourbakhsh, p. 188
def update_pose(x, y, theta, v, dt, l, beta):
    s = v * dt
    s2l = s / (2 * l)
    x = x + s * np.sin(beta) * np.cos(theta - s2l * np.cos(beta))
    y = y + s * np.sin(beta) * np.sin(theta - s2l * np.cos(beta))
    theta = theta + (-s * np.cos(beta) / l)
    return np.array([x, y, theta])

# Motion jacobian, see Siegwart/Nourbakhsh, p. 189
def motion_jacobian(theta, v, dt, l, beta):
    s = v * dt
    s2l = s / (2 * l)
    s_beta = np.sin(beta)
    c_beta = np.cos(beta)
    theta_beta_diff = theta - s2l * c_beta
    sin_theta_beta_diff = np.sin(theta_beta_diff)
    cos_theta_beta_diff = np.cos(theta_beta_diff)

    dfx_ds = s_beta * (cos_theta_beta_diff + s2l * c_beta * sin_theta_beta_diff)
    dfy_ds = s_beta * (sin_theta_beta_diff - s2l * c_beta * cos_theta_beta_diff)
    dftheta_ds = - c_beta / l

    dfx_dbeta = s * (c_beta * cos_theta_beta_diff - s_beta * sin_theta_beta_diff * s2l * s_beta)
    dfy_dbeta = s * (c_beta * sin_theta_beta_diff + s_beta * cos_theta_beta_diff * s2l * s_beta)
    dftheta_dbeta = s * s_beta / l
    return np.array(
        [
            [dfx_ds, dfx_dbeta],
            [dfy_ds, dfy_dbeta],
            [dftheta_ds, dftheta_dbeta]
        ]
    )

# Pose jacobian, siehe Siegwart/Nourbakhsh, p. 189
def pose_jacobian(theta, v, dt, l, beta):
    s = v * dt
    s2l = s / (2 * l)
    dfx_dtheta = -s * np.sin(beta) * np.sin(theta - s2l * np.cos(beta))
    dfy_dtheta = s * np.sin(beta) * np.cos(theta - s2l * np.cos(beta))
    return np.array(
        [
            [1.0, 0.0, dfx_dtheta],
            [0.0, 1.0, dfy_dtheta],
            [0.0, 0.0, 1.0]]
    )

# Motion covariance, see siehe Siegwart/Nourbakhsh, p. 188
def motion_covariance(v, dt, k_s, k_beta, delta_beta):
    s = v * dt
    # Note: Annahme dass der Fehler von beta nur mit der zurückgelegten Strecke korreliert
    # (beta_t1 - beta_t0)
    return np.array([[k_s * abs(s), 0.0], [0.0, k_beta * abs(delta_beta)]])

# Update of odometry error estimation
def update_c_p(C_p, theta, v, dt, l, beta, k_s, k_beta, delta_beta):
    f_p = pose_jacobian(theta, v, dt, l, beta)
    f_delta = motion_jacobian(theta, v, dt, l, beta)
    c_delta = motion_covariance(v, dt, k_s, k_beta, delta_beta)
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
