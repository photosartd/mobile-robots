import numpy as np
import json
from typing import Tuple
from scipy.stats import norm

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
    state=np.array([5, 5, 2.0]),
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


class ParticleFilter:
    def __init__(self, num_particles: int = 100, update_frequency: int = 1):
        self.num_particles = num_particles
        self.particle_positions = np.random.uniform(1, 14, size=(num_particles, 3))
        self.importance_factors = np.repeat(1 / num_particles, num_particles)

        self.updates_count = 0
        self.update_frequency = update_frequency

    def sample_particles(self, num_particles: int | None = None) -> np.ndarray:
        """
        Step 1: sample Partickeln aus dem Cloud im Bezug auf ihre Importance Factors
        Parameters
        ----------
        num_particles: how many particles to sample (can be decreased w.r.t. the cloud)

        Returns
        -------

        """
        if num_particles is None:
            num_particles = self.num_particles
        # Returns COPIES, not references to the objects, => we can modify them all without copying
        return self.particle_positions[
            np.random.choice(self.particle_positions.shape[0], size=num_particles, replace=True, p=self.importance_factors)
        ].copy()

    def update(self, new_positions: list, new_unnormalised_if: list) -> None:
        if self.updates_count // self.update_frequency == 1:
            self.updates_count = 0
        else:
            self.updates_count += 1
            return
        # Normalise importances
        importance_factors = np.array(new_unnormalised_if)
        if any(np.isnan(importance_factors)):
            raise ValueError("Some importance factors were nan")
        importance_factors = importance_factors / sum(importance_factors)
        self.particle_positions = np.array(new_positions)
        self.importance_factors = importance_factors

    @staticmethod
    def perception_model(laser_distances_true: np.ndarray, laser_distances_particle: np.ndarray, sigma: float = 1.0) -> float:
        """
        We assume that the noise was Normally distributed
        We assume that _true distances are not noised while particles are (in the implementation it's the other way
        around but it does not play a role for the perception model)
        Returns:
            unnormalised likelyhood to see those measurements
        """
        if np.any(laser_distances_particle == np.nan) or np.any(laser_distances_particle == np.inf):
            raise ValueError("Laser distances had nans in them")
        likelihoods = norm.pdf(laser_distances_particle, loc=laser_distances_true, scale=sigma)
        full_likelyhood = np.prod(likelihoods)
        #full_likelyhood = np.sum(likelihoods) / len(laser_distances_particle)
        if full_likelyhood == np.nan:
            raise ValueError("Likelyhood was nan")
        return full_likelyhood

    @staticmethod
    def add_noise_to_measurements(
            laser_distances: np.ndarray,
            hitpoints: np.ndarray,
            cov: float = 0.001
    ) -> Tuple[np.ndarray, np.ndarray]:
        laser_noise_cov = np.diag([cov] * len(laser_distances))
        laser_noise = np.random.multivariate_normal(mean=[0] * len(laser_distances), cov=laser_noise_cov)
        noised_laser_distances = laser_distances + laser_noise
        # TODO: check this implementation is ok
        noised_hitpoints = hitpoints + laser_noise[:, np.newaxis]
        return noised_laser_distances, noised_hitpoints


N = 100
update_freq = 1
particle_filter = ParticleFilter(N, update_frequency=update_freq)
cov = 0.001


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

    #Noise zu Distanzen hinzufügen
    noised_laser_distances, noised_hitpoints = particle_filter.add_noise_to_measurements(laser_distances, hitpoints, cov=cov)

    api.draw_lasers(robot.pos, noised_hitpoints)
    #print(laser_distances)
    #print(hitpoints[:2])
    print(robot.theta)

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

    # Step 1: sample particles
    particles_positions = particle_filter.sample_particles(num_particles=N)
    new_particle_positions = []
    new_unnormalised_importance_factors = []
    for particle_lt_1 in particles_positions:
        # Step 2: sample P(l_T | a_T, l_T-1)
        particle_lt_stern = update_pose(*particle_lt_1, vl=vl, vr=vr, dt=dt, l=robot.wheels["vr"].l)
        particle_cov = update_c_p(np.diag([cov] * 3), theta=particle_lt_stern[2], vl=vl, vr=vr, dt=dt, l=robot.wheels["vr"].l, kl=0.001, kr=0.001)
        particle_lt = np.random.multivariate_normal(particle_lt_stern, particle_cov)
        # Step 3: (unnormalisiert) IF P(s_T, | l_T)
        particle_laser_dists, _ = shoot_lasers(particle_lt[:2], particle_lt[2], lasers=robot.lasers, world=world, max_value=1e12)
        unnormalised_importance_factor = particle_filter.perception_model(noised_laser_distances, particle_laser_dists, sigma=4.0)

        # Add to the arrays for the future update
        new_particle_positions.append(particle_lt)
        new_unnormalised_importance_factors.append(unnormalised_importance_factor)
    # Step 4: update the ParticleFilter
    print("Particle IFS:", new_unnormalised_importance_factors)
    particle_filter.update(new_particle_positions, new_unnormalised_importance_factors)
    api.draw_particles(particles=particle_filter.particle_positions)


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
