"""
# Description of the mobile robot simulator RoboSimPy.

RoboSimPy is a 2D mobile robot simulator. It provides functionality for displaying a top-down view of wheel-based robots,
simple worlds comprised of line-based walls, and visualizations for typical algorithms of mobile robotics.

A robot is defined by wheel configurations and supports 2D laser scanners using a simple raytracing method.

Currently, the following functions are provided:
- Collision detection
- Raytracing of single beams for laser scanner from a single and from multiple robot positions
- Visualization functions for:
    - an error ellipse given a covariance matrix
    - basic robot features: hull, wheels, laser beams
    - a simple, line-based world
    - the past robot trajectory and collision points
    - estimated robot poses as particles (e. g. for Monte Carlo and Kalman localization)
    - a 3D belief grid (e. g. for Markov localization)
- Adding and removing walls during runtime using mouse or touch input
- Handling keyboard events in the simulation loop, e. g. for manual control of the robot
- Keyboard shortcuts to control the flow of the simulation and visualization parameters

# Dependencies and installation

RoboSimPy depends on a Python Version >= 3.6, numpy and Kivy >= 2.0.
All dependencies can be installed with pip using the provided `requirements.txt` 
and the command `pip3 install -r requirements.txt`

If you are using Python 3.6, you need to install the additional package `importlib_resources` to run the example script.

# Keyboard shortcuts for the simulator

- pause: space key
- speed up simulation (fixed timestep only): plus key
- slow down simulation (fixed timestep only): minus key
- show distances of laser beams: '0' key (zero key)
- decrease scaling: '8' key
- increase scaling: '9' key

# Further remarks

Generating a documentation in HTML format can be achieved
by using `pdoc` and the following commands, issued in the robosimpy source directory: 
`pdoc ../robosimpy -o docs`.

Authors: Christoph Schulte, Lars Offermann
"""
