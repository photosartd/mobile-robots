"""
Definitions for Kivy GUI elements and drawing functions, making up the visual frontend of the robot simulator.
"""
import time

import numpy as np
from kivy.app import App
from kivy.clock import Clock
from kivy.core.text import Label as CoreLabel
from kivy.core.window import Window
from kivy.graphics.context_instructions import Color, PushMatrix, Rotate, PopMatrix
from kivy.graphics.vertex_instructions import Line, Rectangle, Point
from kivy.graphics.texture import Texture
from kivy.graphics import Ellipse
from kivy.uix.widget import Widget

from .robots import Robot
from .util import *

# Some default values
DRAW_LABELS = False
METER_TO_PX = 30.0
OFFSET_X = 10.0
OFFSET_Y = 10.0
DEFAULT_WINDOW_SIZE = (900, 900)
MAX_HISTORY = 1000

# Global variables
scale = METER_TO_PX
ui_scale = 1.0


class RoboSimPyApp(App):
    """
    Subclass of Kivy App implementing GUI elements for the robot simulator.
    """

    def __init__(
        self,
        robot: Robot,
        state_update,
        world=None,
        window_size=None,
        meter2px=METER_TO_PX,
        gui_scale=1,
        fixed_timestep=True,
        **kwargs,
    ):
        """

        Parameters
        ----------
        robot : Robot
            The robot to simulate.

        window_size : (int, int)
            Size of the map in pixels.

        state_update : function
            The function for the main simulation loop. Gets called up to 60 times a second.

        world : ndarray or None
            A 2D numpy array with shape n by 4 with the format [[x1,y1,x2,y2], ... ].
            Contains the definition of the walls for the simulation.

        meter2px :
            Scaling factor from meter to pixel.

        ui_scale : float
            Scaling factor for GUI elements. If you are using a high resolution display, you can increase
            the drawing size of simulator elements using this setting. Additionally, change meter2px
            to an appropriate setting for your display.

        fixed_timestep : bool
            Whether or not a fixed timestep is used for simulation.
            If true, all movement speed related settings are made in meter/timestep.
            If this setting is false, the simulator uses wall clock time and all
            movement speed related units are meter/second.

        kwargs
            Optional arguments for superclass.
        """
        global scale, ui_scale

        super().__init__(**kwargs)
        self.robot = robot
        self.state_update = state_update
        self.window_size = window_size
        self.fixed_timestep = fixed_timestep

        if world is not None:
            self.world = np.array(world)
            world_size_x, world_size_y = compute_world_size(self.world)
            window_size_x = int(world_size_x * meter2px) / gui_scale
            window_size_y = int(world_size_y * meter2px) / gui_scale
            self.window_size = (window_size_x, window_size_y)
        else:
            self.world = np.empty((0, 4), float)
            self.window_size = DEFAULT_WINDOW_SIZE

        scale = meter2px
        ui_scale = gui_scale

    def build(self):
        """
        Gets called by Kivy when starting the application.

        Returns
        -------
        RoboSimPyAPI
        """
        ui = RoboSimPyWidget(
            robot=self.robot,
            world=self.world,
            state_update=self.state_update,
            fixed_timestep=self.fixed_timestep,
        )
        Window.size = self.window_size
        Window.clearcolor = (1, 1, 1, 1)
        Clock.schedule_interval(ui.update, 1.0 / 60.0)
        return ui


class RoboSimPyWidget(Widget):
    """
    Subclass of a Kivy widget implementing drawing methods specific to this robot simulator.
    To be able to call functions from an object of this class during simulation, you have to pass it as a
    parameter to your main simulation function (the "state_update"-Argument of RoboSimPyApp).
    """

    _inputs = dict()  # mapping key code to actual character
    _to_be_added = None
    _updates_per_frame = 1
    _history = []
    _collisions = []

    def __init__(self, robot, world, state_update, fixed_timestep, **kwargs):
        """

        Parameters
        ----------
        robot : Robot
            The robot to simulate.

        world : ndarray or None
            A 2D numpy array with shape n by 4 with the format [[x1,y1,x2,y2], ... ].
            Contains the definition of the walls for the simulation.

        state_update : function
            The function for the main simulation loop. Gets called up to 60 times a second.

        kwargs
            Optional arguments for superclass.
        """
        super().__init__(**kwargs)

        self.world = world
        self.robot = robot
        self.user_logic = state_update
        self.dt = 1.0
        self.fixed_timestep = fixed_timestep
        if not self.fixed_timestep:
            self.prev_time = time.time_ns()

        # noinspection PyUnusedLocal
        def _on_keyboard_down(instance, key, scancode, codepoint, modifiers):
            global scale
            # Pfeiltasten separat behandeln
            if key == 273:
                codepoint = "up"
            elif key == 275:
                codepoint = "right"
            elif key == 274:
                codepoint = "down"
            elif key == 276:
                codepoint = "left"
            # Sonst:
            self._inputs[key] = codepoint
            if "0" == codepoint:
                global DRAW_LABELS
                DRAW_LABELS = not DRAW_LABELS
            elif "+" == codepoint:
                self._updates_per_frame += 1
                self.dt += 0.25
            elif "-" == codepoint:
                self._updates_per_frame -= 1
                self.dt = max(0.0, self.dt - 0.25)
                self._updates_per_frame = max(0, self._updates_per_frame)
            elif "9" == codepoint:
                scale += 0.1
                self._history = []
            elif "8" == codepoint:
                scale -= 0.1
                self._history = []

        # noinspection PyUnusedLocal
        def _on_keyboard_up(instance, key, scancode):
            self._inputs.pop(key)

        Window.bind(on_key_down=_on_keyboard_down)
        Window.bind(on_key_up=_on_keyboard_up)

    def on_touch_down(self, touch):
        """
        Gets called when the user clicks/touches the window of the application.

        Parameters
        ----------
        touch
            A touch object.
        """
        self._to_be_added = touch.pos  # first point of new line

    def on_touch_up(self, touch):
        """
        Gets called when the user releases a click/touche on the window of the application.
        This function implements a simple routine to add/remove walls to the active simulation.

        Parameters
        ----------
        touch
            A touch object.
        """

        # add new line to world
        if (touch.pos[0] - self._to_be_added[0]) ** 2 + (
            touch.pos[1] - self._to_be_added[1]
        ) ** 2 >= 40:
            self.world = np.vstack(
                (self.world, np.array([*touch.pos, *self._to_be_added]) / scale)
            )
        else:
            # if clicked on line: delete line
            for i in range(self.world.shape[0]):
                a, b = self.world[i, :2], self.world[i, 2:]
                c = np.array(self._to_be_added) / float(scale)
                cross = np.cross(b - a, c - a)
                eps = 0.1
                if -eps < cross < eps:
                    if 0 < np.dot((b - a), (c - a)) < np.linalg.norm(a - b) ** 2:
                        self.world = np.delete(self.world, i, axis=0)
                        break

            self._to_be_added = None

    def update(self, *args):
        """
        Main update routine. Calls the default drawing functions for the robot, robot path history,
        world, and collision markers as well as the user defined state_update function.
        Is scheduled to run up to 60 times a second.
        """
        if " " in self._inputs.values():  # Space pauses simulation
            return

        prev_state = self.robot.state.copy()

        self.canvas.clear()
        self.user_logic(self, self.world, self._inputs.values())
        with self.canvas:
            self.draw_history()
            self.draw_world()
            self.draw_collisions()
            self.draw_robot()

        did_collide = collision(self.world, self.robot.enclosure, self.robot.state)
        if did_collide != -1:
            self._collisions.extend([self.robot.pos[0], self.robot.pos[1]])
            self.robot.state = prev_state

        if not self.fixed_timestep:
            current_time = time.time_ns()
            diff_in_s = (current_time - self.prev_time) * 1e-9
            self.prev_time = current_time
            self.dt = diff_in_s

    def draw_line(self, x0, y0, x1, y1, pointsize=10, color=(0.0, 0.0, 0.0, 1.0)):
        with self.canvas:
            Color(*color)
            Line(
                points=[x0 * scale, y0 * scale, x1 * scale, y1 * scale],
                width=pointsize / 6 * ui_scale,
                cap="none",
            )

    def draw_point(self, x, y, pointsize, color=(0.0, 0.0, 0.0, 1.0)):
        with self.canvas:
            Color(*color)
            ptsize = pointsize * ui_scale
            Ellipse(
                pos=(x * scale - ptsize / 2, y * scale - ptsize / 2),
                size=(ptsize, ptsize),
            )

    def draw_discrete_belief(self, belief, grid_size, world_size, max_alpha=1.0):
        """
        Draws a grid based discrete probability distribution over a robot pose (that is 2D position and 1D orientation).

        Parameters
        ----------
        belief: ndarray
            3D array with entries describing the probability of the robot being in this cell.
        grid_size: array
            3x1 vector describing the size of the belief grid.
            Entry 1: number of cells in x-direction.
            Entry 2: number of cells in y-direction.
            Entry 3: number of cells for possible angular range.
        world_size: array
            3x1 vector specifying the world size.
            Entry 1: size in y-direction.
            Entry 2: size in x-direction.
            Entry 3: size of angular range (in rad, if unrestricted: 2*np.pi)
        max_alpha:
            Maximum value of alpha channel for display, default is 1.0 (no opacity).
        """
        texture = Texture.create(
            size=grid_size[:2],
            colorfmt="rgba",
            icolorfmt="rgba",
            bufferfmt="float",
            mipmap=False,
        )
        texture.mag_filter = "nearest"

        bel_sum = np.sum(belief, axis=-1).T

        # Calculate RGB value from HSV color space
        angle_space = np.linspace(0, np.pi * 2, num=belief.shape[-1], endpoint=False)
        vec_space = np.stack([np.cos(angle_space), np.sin(angle_space)], -1)
        avg_vecs = np.sum(belief[:, :, :, None] * vec_space[None, None, :, :], 2)
        avg_angles = np.arctan2(avg_vecs[..., 1], avg_vecs[..., 0]) + np.pi

        h = avg_angles.T / (np.pi * 2) * 360.0
        s = np.ones_like(h)  # max_alpha*bel_sum
        v = np.ones_like(h)

        hi = (h // 60).astype(np.int32)
        f = h / 60.0 - hi
        p = 1.0 - s
        q = 1.0 - s * f
        t = 1.0 - s * (1.0 - f)
        color = np.stack([v, t, p], -1)
        color[hi == 1] = np.stack([q, v, p], -1)[hi == 1]
        color[hi == 2] = np.stack([p, v, t], -1)[hi == 2]
        color[hi == 3] = np.stack([p, q, v], -1)[hi == 3]
        color[hi == 4] = np.stack([t, p, v], -1)[hi == 4]
        color[hi == 5] = np.stack([v, p, q], -1)[hi == 5]
        color = color
        alpha = bel_sum * max_alpha

        rgba = np.concatenate([color, alpha[:, :, None]], -1).astype(np.float32)
        texture.blit_buffer(rgba.tobytes(), colorfmt="rgba", bufferfmt="float")
        with self.canvas:
            Color(1, 1, 1, 1)
            size = (world_size[1] * scale, world_size[0] * scale)
            Rectangle(texture=texture, size=size)

    def draw_particles(self, particles, pointsize=10, color=(0.0, 0.0, 0.0, 1.0)):
        """
        Draws particles as a simple way to display a position and orientation, for example
        for Monte Carlo localization.
        The first argument "particles" may be a Nx2 or Nx3 array (with N := number of particles).
        The optional third entry in the second dimension gets interpreted as orientation and is displayed
        via a small line.

        Parameters
        ----------
        particles : ndarray
            2D numpy array with the following structure:
            [[ x1, y1],
             [ x2, y2],
             ...
            ]
            OR
            3D numpy array with the following structure:
            [[ x1, y1, theta1],
             [ x2, y2, theta2],
             ...
            ]
        pointsize : int or None
            Size of the points drawn.
        color : tuple
            RGB or RGBA tuple.

        """
        if len(particles.shape) == 1:
            particles = np.array([particles])

        if (
            particles.shape[len(particles.shape) - 1] == 2
        ):  # [[x1,y1],[x2,y2], ... ] or [x,y]
            for x, y in particles:
                self.draw_point(x, y, pointsize, color)
                # Point(points=(scale * particles.flatten()).tolist(), pointsize=pointsize)
        elif (
            particles.shape[len(particles.shape) - 1] == 3
        ):  # [[x1,y1,theta1],[x2,y2, theta2], ... ] or [x,y, theta]
            
            for x0, y0, theta in particles:
                self.draw_point(x0, y0, pointsize, color)
                with self.canvas:
                    x1 = x0 + np.cos(theta) * pointsize * 0.03
                    y1 = y0 + np.sin(theta) * pointsize * 0.03
                    Line(
                        points=[x0 * scale, y0 * scale, x1 * scale, y1 * scale],
                        width=pointsize / 6 * ui_scale,
                        cap="none",
                    )
                # Point(points=(scale * particles[:, :2].flatten()).tolist(), pointsize=pointsize)
                # for i in range(particles.shape[0]):
                #    p = particles[i, :2]
                #    t = particles[i, 2]
        else:
            raise Exception("Wrong dimensions for drawing particles")

    def draw_error_ellipse(self, c_p, pos):
        """
        Draws a 3x3 covariance matrix C_p at the position pos by projecting into 2D space.

        Parameters
        ----------
        c_p: ndarray
            Matrix to draw.
        pos: ndarray
            Center of the ellipse.
        """
        # Src: https://scipython.com/book/chapter-7-matplotlib/examples/bmi-data-with-confidence-ellipses/
        cov = c_p[:2, :2]
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]

        # The anti-clockwise angle to rotate our ellipse by
        vx, vy = eigvecs[:, 0][0], eigvecs[:, 0][1]
        theta = np.arctan2(vy, vx)

        # Width and height of ellipse to draw
        width, height = scale * np.sqrt(eigvals)

        with self.canvas:
            Color(1.0, 0.0, 0.0)
            PushMatrix()
            Rotate(angle=np.degrees(theta), origin=scale * pos)
            Line(
                ellipse=(
                    (scale * pos[0]) - width / 2.0,
                    (scale * pos[1]) - height / 2.0,
                    width,
                    height,
                ),
                width=1 * ui_scale,
            )
            PopMatrix()

    def draw_lasers(self, robot_state, hitpoints, color=(0.0, 0.4, 0.9)):
        """
        Draws laser beams.

        Parameters
        ----------
        robot_state : ndarray
            Pose (position and orientation) of the robot. This is the origin of the laser beams.
        hitpoints : ndarray
            Positions of intersections of laser beams with walls. May contain as many hitpoints as there are laser beams.

        """
        with self.canvas:
            Color(*color)
            for p in hitpoints:
                if not np.any(np.isinf(p)):
                    Line(
                        points=(scale * np.array([*robot_state[:2], *p])).tolist(),
                        width=1 * ui_scale,
                    )

            if DRAW_LABELS:
                Color(0.0, 0.0, 0.0)
                for p in hitpoints:
                    if not np.any(np.isinf(p)):
                        lbl_pos = (int(p[0] * scale) - 10, int(p[1] * scale) - 8)
                        distance = np.linalg.norm(
                            np.array(robot_state[:2]) - np.array(p)
                        )
                        lbl_text = f"{distance:.2f}"
                        lbl_size = (
                            (20 * ui_scale, 15 * ui_scale)
                            if len(lbl_text) > 2
                            else (15 * ui_scale, 15 * ui_scale)
                        )

                        core_l = CoreLabel()
                        core_l.options["font_size"] = 14 * ui_scale
                        core_l.text = lbl_text
                        core_l.refresh()
                        Rectangle(texture=core_l.texture, pos=lbl_pos, size=lbl_size)

    def draw_robot(self):
        """
        Draws the robot, that is the hull, the wheels, the connecting lines between the
        center of the robot coordinate system and each wheel and the center of the robot coordinate
        system as a green dot.
        """
        robot = self.robot
        self.draw_point(robot.pos[0], robot.pos[1], 6 * ui_scale, (0.0, 0.6, 0.2, 1.0))
        for x1, y1, x2, y2 in robot.enclosure:
            Color(0.0, 0.6, 0.2)
            Line(
                points=(
                    scale
                    * np.array(
                        [
                            (
                                robot.pos
                                + rotate(robot.theta, np.array([x1, y1], order="C"))
                            ),
                            (
                                robot.pos
                                + rotate(robot.theta, np.array([x2, y2], order="C"))
                            ),
                        ]
                    )
                ).tolist(),
                width=1.2 * ui_scale,
            )

        for wheel in robot.wheels.values():
            walpha = float(wheel.alpha)
            wbeta = wheel.beta

            # line l
            Color(0.0, 0.6, 0.2)
            [lx1, ly1] = rotate(
                walpha + robot.theta, np.array([wheel.l, 0.0], order="C")
            )
            rlx, rly = robot.pos[0] + lx1, robot.pos[1] + ly1
            Line(
                points=(
                    scale * np.array([robot.pos[0], robot.pos[1], rlx, rly])
                ).tolist(),
                width=1 * ui_scale,
            )

            # line d
            Color(0.0, 0.0, 1.0)
            [dx2, dy2] = rotate(
                walpha + wbeta + robot.theta, np.array([wheel.d, 0.0], order="C")
            )
            rldx, rldy = rlx + dx2, rly + dy2
            Line(
                points=(scale * np.array([rlx, rly, rldx, rldy])).tolist(),
                width=1 * ui_scale,
            )

            # wheel
            Color(0.0, 0.0, 0.0) if wheel.motor else Color(0.65, 0.65, 0.65)
            [rx, ry] = rotate(
                walpha + wbeta + robot.theta + np.pi / 2,
                np.array([robot.wheel_radius, 0.0], order="C"),
            )
            Line(
                points=(
                    scale * np.array([rldx + rx, rldy + ry, rldx - rx, rldy - ry])
                ).tolist(),
                width=2.0 * ui_scale,
            )
            
            self.draw_point(rldx + rx, rldy + ry, 2.0 * ui_scale, (1., .2, .2))

    def draw_world(self):
        """Draws the world as a collection of lines as defined by the world array."""
        Color(0.1, 0.5, 0.1, 1.0)
        for x1, y1, x2, y2 in self.world:
            Line(
                points=[scale * x1, scale * y1, scale * x2, scale * y2],
                width=1.2 * ui_scale,
            )

    def draw_history(self):
        """Draws the past trajectory of the robot."""
        self._history.extend((self.robot.pos * scale).astype(np.int32))
        if len(self._history) > MAX_HISTORY:
           self._history.pop(0)
           self._history.pop(0)

        Color(0.9, 0.9, 0.9)
        Line(points=self._history, width=1.2 * ui_scale)

    def draw_collisions(self):
        """Gets used to mark the current position of a robot during a collision."""
        Color(1, 0.6, 0.0, 1)
        Point(
            points=(scale * np.array(self._collisions)).tolist(), pointsize=1 * ui_scale
        )
