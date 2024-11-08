import numpy as np


def rotate(angle, vec):
    ca = np.cos(angle)
    sa = np.sin(angle)
    return np.array([[ca, -sa], [sa, ca]]) @ vec


def shoot_lasers(laser_pos, theta, lasers, world, max_value=np.inf):
    """
    Fragt die Sensordaten für die Laser-Strahlen ab.
    Es werden sowohl die Distanzen der jeweiligen Laser,
    als auch ein Array der Schnittpunkte mit den Wänden zurückgegeben,
    welches für das Zeichnen der Laser benutzt werden kann.

    Parameters
    ----------
    laser_pos : ndarray
        Position von der aus die Laser geschossen werden. (robot.pos)
    theta : float
        Aktuelle Rotation des Roboters
    lasers : ndarray
        Liste von Winkeln, in denen die Laser geschossen werden sollen. ( 0 bis 2*pi )
    world : ndarray
        Das Array, welches die Wände enthält. Format [ [x1,y1, x2,y2], ... ]


    Returns
    -------
    laser_distances : ndarray
        Distanzen für die Laser
    hitpoints : ndarray
        Punkte, an denen die Laser Wände treffen.

    Examples
    --------
    >>> laser_distances, hitpoints = api.shoot_lasers(laser_pos, robot.theta, robot.lasers, world)
    laser_distances enthält nun soviele Einträge wie robot.lasers

    Anschließendes Zeichnen der Laser
    >>> api.draw_lasers(robot.pos, hitpoints)

    """
    assert len(laser_pos.shape) == 1 and laser_pos.shape[0] == 2, "laser_pos should be an ndarray with exactly two entries (position along x and y axis)"
    angle = theta + lasers
    ray_direction = np.stack((np.cos(angle), np.sin(angle)), -1)  # 6, 2
    line_start = world[:, :2]  # 29, 2
    line_end = world[:, 2:]  # 29, 2
    v1 = laser_pos - line_start  # 29, 2
    v2 = line_end - line_start  # 29, 2
    v3 = np.stack((-ray_direction[:, 1], ray_direction[:, 0]), 0)  # 2, 6

    dot = v2 @ v3
    t1 = np.divide(
        np.expand_dims(np.cross(v2, v1), -1),
        dot,
        out=-np.ones_like(dot),
        where=dot != 0,
    )
    t2 = np.divide(v1 @ v3, dot, out=-np.ones_like(dot), where=dot != 0)

    first = t1 >= 0.0
    second = np.logical_and(t2 >= 0.0, t2 <= 1.0)
    does_intersect = np.logical_and(first, second)
    distances = np.min(t1, 0, initial=max_value, where=does_intersect)
    hitpoints = (
        np.expand_dims(laser_pos, 0) + np.expand_dims(distances, -1) * ray_direction
    )
    return distances, hitpoints
    # hitpoints = hitpoints[~np.all(hitpoints == 0, axis=1)]


def shoot_multiple_lasers(laser_pos, theta, lasers, world, max_value=np.inf):
    angle = np.add.outer(theta, lasers)
    ray_direction = np.stack((np.cos(angle), np.sin(angle)), -1)  # 500000, 6, 2
    line_start = world[:, :2]  # 29, 2
    line_end = world[:, 2:]  # 29, 2
    v1 = laser_pos[:, None, :] - line_start[None, :, :]  # 500000, 29, 2
    v2 = line_end - line_start  # 29, 2
    v3 = np.stack((-ray_direction[..., 1], ray_direction[..., 0]), -1)  # 500000, 6, 2

    dot = np.transpose(v3 @ v2.T, [0, 2, 1])  # 500000, 29, 6
    t1 = np.divide(
        np.expand_dims(np.cross(v2[None, :, :], v1), -1),
        dot,
        out=-np.ones_like(dot),
        where=dot != 0,
    )
    t2 = np.divide(
        v1 @ np.transpose(v3, [0, 2, 1]), dot, out=-np.ones_like(dot), where=dot != 0
    )

    first = t1 >= 0.0
    second = np.logical_and(t2 >= 0.0, t2 <= 1.0)
    does_intersect = np.logical_and(first, second)
    distances = np.min(t1, 1, initial=max_value, where=does_intersect)
    hitpoints = laser_pos[:, None, :] + distances[:, :, None] * ray_direction
    return distances, hitpoints


def collision(world, robot_enclosure, robot_state):
    robot_pos, robot_theta = robot_state[:2], robot_state[2]

    # robot world collisions
    w0_x, w0_y = world[:, 0], world[:, 1]
    w1_x, w1_y = world[:, 2], world[:, 3]

    # Rotate and move
    s_theta = np.sin(robot_theta)
    c_theta = np.cos(robot_theta)
    p2_x = (
        c_theta * robot_enclosure[:, 0] - s_theta * robot_enclosure[:, 1] + robot_pos[0]
    )
    p2_y = (
        s_theta * robot_enclosure[:, 0] + c_theta * robot_enclosure[:, 1] + robot_pos[1]
    )
    p3_x = (
        c_theta * robot_enclosure[:, 2] - s_theta * robot_enclosure[:, 3] + robot_pos[0]
    )
    p3_y = (
        s_theta * robot_enclosure[:, 2] + c_theta * robot_enclosure[:, 3] + robot_pos[1]
    )

    s1_x = w1_x - w0_x
    s1_y = w1_y - w0_y
    s2_x = p3_x - p2_x
    s2_y = p3_y - p2_y
    den = -s2_x[None, :] * s1_y[:, None] + s1_x[:, None] * s2_y[None, :]

    s_num = -s1_y[:, None] * np.subtract.outer(w0_x, p2_x) + s1_x[
        :, None
    ] * np.subtract.outer(w0_y, p2_y)
    t_num = s2_x[None, :] * np.subtract.outer(w0_y, p2_y) - s2_y[
        None, :
    ] * np.subtract.outer(w0_x, p2_x)
    den_not_zero = den != 0

    s = np.divide(s_num, den, where=den_not_zero)
    t = np.divide(t_num, den, where=den_not_zero)
    s_cond = np.logical_and(s >= 0.0, s <= 1.0)
    t_cond = np.logical_and(t >= 0.0, t <= 1.0)
    does_intersect = np.logical_and(s_cond, t_cond)

    if np.any(np.logical_and(does_intersect, den_not_zero)):
        return 0
    return -1


def compute_world_size(world):
    world = np.array(world)
    all_x = np.concatenate([world[:, 0], world[:, 2]], -1)
    all_y = np.concatenate([world[:, 1], world[:, 3]], -1)
    max_x = np.max(all_x)
    max_y = np.max(all_y)
    min_x = np.min(all_x)
    min_y = np.min(all_y)
    world_size_x = max_x + min_x
    world_size_y = max_y + min_y
    return world_size_x, world_size_y
