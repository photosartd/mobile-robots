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


class Beacons:
    def __init__(
            self,
            positions: np.ndarray,
            color: tuple = (1.0, 0.0, 0.0, 1.0),
            pointsize: int = 15
    ):
        """
        Parameters
        ----------
        positions: [nx2] array with positions of beacons
        """
        self.positions = positions
        self.color = color
        self.pointsize = pointsize

    def draw(self, api: RoboSimPyWidget):
        api.draw_particles(self.positions, pointsize=self.pointsize, color=self.color)

    def sample(self) -> np.ndarray:
        return self.positions[np.random.choice(len(self.positions), size=1)].copy().reshape((2,))


class KalmanFilter:
    def __init__(
            self,
            beacons: Beacons,
            state: np.ndarray,
            covariance: np.ndarray | None = None,
            sigma_k: float = 0.01,
            sigma_p: float = 0.01,
            g: float = 3.0
    ):
        """
        Parameters
        ----------
        state [x,y,theta] numpy array of size [3,]
        """
        self.beacons = beacons
        self.state = state
        if covariance is None:
            self.covariance = np.zeros((state.shape[0], state.shape[0]))
        else:
            self.covariance = covariance
        self.sigma_k = sigma_k
        self.sigma_p = sigma_p
        self.g = g

    def choose_beacon(self, beacon_pose: np.ndarray, x_k_k_1: np.ndarray, C_k_k_1: np.ndarray, threshold: float = 9.0):
        z_real = self.z(beacon_pose, v=True)
        matches = []
        best_match_dist = np.inf
        z_best = None
        for current_beacon in self.beacons.positions:
            z_cap = self.z(current_beacon, v=False)
            Hk = KalmanFilter.Hk(current_beacon, x_k_k_1[:2])
            Sk = KalmanFilter.Sk(Hk=Hk, C_k_k_1=C_k_k_1, Nk=self.Nk, Vk=self.Vk)
            nu = z_real - z_cap # Innovation
            dist = KalmanFilter.mahalanobis_distance(nu, Sk)
            if dist < threshold:
                if dist < best_match_dist:
                    best_match_dist = dist
                    z_best = z_real
                    matches.append(current_beacon)
        if len(matches) == 0:
            return False, None, z_best
        elif len(matches) == 1:
            return True, matches[0], z_best
        else:
            return False, None, z_best

    def update(self, x_k_k_1: np.ndarray, C_k_k_1: np.ndarray, beacon_pose: np.ndarray) -> None:
        # Establish a match
        match_found, beacon, z_real = self.choose_beacon(
            beacon_pose=beacon_pose,
            x_k_k_1=x_k_k_1,
            C_k_k_1=C_k_k_1,
            threshold=self.g
        )
        if not match_found:
            print("No matches found. Using predict step")
            self.state = x_k_k_1
            self.covariance = C_k_k_1
            return # No update
        else:
            print("Match found")
            beacon_pose = beacon # Update based on the match
        z = z_real#self.z(beacon_pose, v=True)
        z_cap = self.z(beacon_pose, v=False)
        print(f"Z mit Rauschen: {z[1]}")
        print(f"Z: {z_cap[1]}")

        Hk = KalmanFilter.Hk(beacon_pose, x_k_k_1[:2])
        Sk = KalmanFilter.Sk(Hk=Hk, C_k_k_1=C_k_k_1, Nk=self.Nk, Vk=self.Vk)
        K = KalmanFilter.K(Hk=Hk, C_k_k_1=C_k_k_1, Sk=Sk)
        x_k_k = x_k_k_1 + (K @ (z - z_cap)).reshape((3,))
        C_k_k = C_k_k_1 - K @ Sk @ K.T # (np.identity(n=C_k_k_1.shape[0]) - K @ Hk) @ C_k_k_1
        self.state = x_k_k
        self.covariance = C_k_k

    def draw(self, api: RoboSimPyWidget):
        api.draw_particles(self.state.reshape((1, 3)), pointsize=15, color=(0.0, 1.0, 0.0, 1.0))

    @property
    def pos(self) -> np.ndarray:
        return self.state[:2]

    @property
    def theta(self) -> float:
        return self.state[2]

    @property
    def Nk(self) -> np.ndarray:
        return np.array(
            [
                [self.sigma_k ** 2, 0],
                [0, self.sigma_p ** 2]
            ]
        )

    @property
    def Vk(self) -> np.ndarray:
        """See V_k = \frac{\partial h}{\partial v}"""
        return np.array(
            [
                [1, 0],
                [0, -1]
            ]
        )

    def z(self, beacon_pos: np.ndarray, v: bool = False):
        """
        Parameters
        ----------
        beacon_pos
        v

        Returns
        -------
        z = [2x1] array with \beta and \alpha
        """
        if v:
            v_k = np.random.normal(loc=0, scale=self.sigma_k, size=1).item()
            v_p = np.random.normal(loc=0, scale=self.sigma_p, size=1).item()
        else:
            v_k = 0.0
            v_p = 0.0
        return np.array(
            [[KalmanFilter.h_k(self.theta, v_k)],
             [KalmanFilter.alpha_k(beacon_pos, self.state, v_p)]]
        )

    @staticmethod
    def Hk(beacon_pos: np.ndarray, robot_pos: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        beacon_pos
        robot_pos

        Returns
        -------
        Hk = [2x3] Jacobian of the measurement
        """
        x_Li, y_Li = beacon_pos
        x_k, y_k = robot_pos
        r2 = (x_Li - x_k) ** 2 + (y_Li - y_k) ** 2
        return np.array(
            [[0.0, 0.0, 1.0],
             [(y_Li - y_k) / r2, -(x_Li - x_k) / r2, -1.0]
             ]
        )

    @staticmethod
    def Sk(Hk: np.ndarray, C_k_k_1: np.ndarray, Nk: np.ndarray, Vk: np.ndarray) -> np.ndarray:
        """Kovarianz der Innovation"""
        return Hk @ C_k_k_1 @ Hk.T + Vk @ Nk @ Vk.T

    @staticmethod
    def K(Hk: np.ndarray, C_k_k_1: np.ndarray, Sk: np.ndarray) -> np.ndarray:
        """Kalman Gain"""
        return C_k_k_1 @ Hk.T @ np.linalg.inv(Sk)

    @staticmethod
    def h_k(theta_k: float, v_k: float) -> float:
        """
        Returns compass-measured angle to the x axis
        Parameters
        ----------
        theta_k
        v_k

        Returns
        -------
        beta
        """
        return theta_k + v_k

    @staticmethod
    def alpha_k(
            beacon_pos: np.ndarray,
            robot_pos: np.ndarray,
            v_pk: float
    ):
        """
        Parameters
        ----------
        beacon_pos [x_Li, y_Li] of a beacon
        robot_pos [x_k, y_k, theta_k] of a robot
        v_pk - noise

        Returns
        -------
        alpha - angle to the beacon
        """
        x_Li, y_Li = beacon_pos
        x_k, y_k, theta_k = robot_pos
        return np.arctan2(y_Li - y_k, x_Li - x_k) - theta_k - v_pk

    @staticmethod
    def mahalanobis_distance(nu: np.ndarray, Sk: np.ndarray) -> float:
        return (nu.T @ np.linalg.inv(Sk) @ nu).item()

beacons = Beacons(
    positions=np.array(
        [
            [3.0, 3.0],
            [7.0, 10.0],
            [5.0, 3.0]
        ]
    )
)
kf = KalmanFilter(beacons=beacons, state=robot.state.copy(), g=9.0)
kl, kr = 0.001, 0.001


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
    # print(laser_distances)
    # print(hitpoints[:2])
    #print(robot.theta)

    vr = (0.01 if "e" in inputs else -0.01 if "d" in inputs else 0.0) * dt
    vl = (0.01 if "q" in inputs else -0.01 if "a" in inputs else 0.0) * dt

    # Error propagation of odometry.
    robot.C_p = update_c_p(
        robot.C_p, robot.theta, vl, vr, dt, robot.wheels["vr"].l, kl, kr
    )
    api.draw_error_ellipse(robot.C_p, robot.pos)

    # Updating the robot pose using the previous pose, the time step, and motion commands for left and right wheel.
    robot.pos[0], robot.pos[1], robot.theta = update_pose(
        robot.pos[0], robot.pos[1], robot.theta, vl, vr, dt, robot.wheels["vr"].l
    )

    # Draw beacons
    beacons.draw(api)

    # Kalman filter
    # Odometry step
    C_k_k_1 = update_c_p(
        kf.covariance, kf.theta, vl, vr, dt, robot.wheels["vr"].l, kl, kr
    )
    x_k_k_1 = update_pose(
        kf.pos[0], kf.pos[1], kf.theta, vl, vr, dt, robot.wheels["vr"].l
    )
    kf.update(
        x_k_k_1=x_k_k_1,
        C_k_k_1=C_k_k_1,
        beacon_pose=beacons.sample()
    )
    kf.draw(api)
    api.draw_error_ellipse(kf.covariance, kf.pos)


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
