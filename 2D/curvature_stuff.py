import sys

import matplotlib.pyplot as plt
import roboticstoolbox as rtb
from spatialmath import UnitQuaternion, SE3
import numpy as np

def wait_for_user(fig, key=None, timeout_pause=0.1):
    """
    Wait (non-blocking for the GUI) until the user presses any key or mouse button.
    If `key` is provided (string), only that key will continue (e.g. 'q' or 'enter').
    """
    state = {"done": False}

    def _on_event(event):
        # event.key is None for mouse clicks; compare only if a key is requested
        if key is None or event.key == key:
            state["done"] = True

    cid_k = fig.canvas.mpl_connect("key_press_event", _on_event)
    cid_b = fig.canvas.mpl_connect("button_press_event", _on_event)

    try:
        while not state["done"]:
            plt.pause(timeout_pause)
    finally:
        for cid in (cid_k, cid_b):
            try:
                if cid is not None and hasattr(fig, "canvas") and hasattr(fig.canvas, "mpl_disconnect"):
                    fig.canvas.mpl_disconnect(cid)
            except Exception:
                pass

def orientation_error(Rd, Rc):
    # Orientation error
    Re = Rd @ Rc.transpose()
    trace = np.trace(Re)
    val = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(val)

    if abs(theta) < 1e-6:
        return 0, np.zeros(3)

    # skew-symmetric part -> rotation axis (unnormalized)
    rx = np.array([
        Re[2, 1] - Re[1, 2],
        Re[0, 2] - Re[2, 0],
        Re[1, 0] - Re[0, 1]
    ])

    denom = 2.0 * np.sin(theta)
    if abs(denom) < 1e-6:
        # fallback: normalize rx if possible
        n = np.linalg.norm(rx)
        axis = rx / n if n > 0 else np.zeros(3)
    else:
        axis = rx / denom

    return theta, axis

def discretize_circle_with_normals(R, N, center=(0, 0)):
    cx, cy = center
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)

    x = cx + R * np.cos(theta)
    y = cy + R * np.sin(theta)

    nx = np.cos(theta)
    ny = np.sin(theta)

    points = np.column_stack((x, y))
    normals = np.column_stack((nx, ny))

    return points, normals

def _circle_normal(center, point, unit=True):
    """
    Compute the normal vector of a circle at a given point.

    Parameters:
        center : tuple (xc, yc)
        point  : tuple (x, y)
        unit   : bool -> return unit normal if True

    Returns:
        normal vector as numpy array
    """
    xc, yc = center
    x, y = point

    # Radial vector (normal)
    normal = np.array([x - xc, y - yc], dtype=float)

    if unit:
        norm = np.linalg.norm(normal)
        if norm == 0:
            raise ValueError("Point cannot be the center of the circle.")
        normal = normal / norm

    return normal

def _sine_normal(x, unit=True):
    """
    Compute normal vector of sin function at a given x.
    """
    dy_dx = np.cos((x / 0.1) + 0.5)

    # Normal vector (perpendicular to tangent)
    normal = np.array([dy_dx, -1])

    if unit:
        norm = np.linalg.norm(normal)
        normal = normal / norm

    return normal

def sine_curvature(x):
    """
    Compute curvature of the sine function at a given x.
    """
    u = (x / 0.1) + 0.5

    # First derivative
    dy_dx = np.cos(u)

    # Second derivative
    d2y_dx2 = -10 * np.sin(u)

    # Curvature formula
    curvature = d2y_dx2 / (1 + dy_dx ** 2) ** (3 / 2)

    return curvature

def get_normal(which, point, center=None):
    if which == "circle":
        return _circle_normal(point=point, center=center)
    elif which == "sine":
        return _sine_normal(x=point[0])
    else:
        raise ValueError("Invalid option for normal type")

class planar_robot(rtb.Robot):
    """
    4dof 2D planar robot class
    """

    def __init__(self, tld, filepath):

        links, name, urdf_string, urdf_filepath = self.URDF_read(file_path=filepath, tld=tld)
        super().__init__(
            links,
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )

        self.qz = np.array([0,0,0,0])
        self.qs = np.array([-0.2, -0.3, 0.3, 0.5])
        self.qj = np.array([-0.4, -0.6, 1.4, 1.4])
        self.qc = np.array([-0.15, -0.35, 1.83, 1.61])
        self.qsine = np.array([2.1, -0.5, -0.5, -0.5])

        self.addconfiguration("qz", self.qz)
        self.addconfiguration("qs", self.qs)

def main():
    which = "circle"  # "circle" or "sine"
    start_x_sine = -0.1070796327
    reference_traj = False
    # reference_traj = True
    TIME = 100

    FREQ = 100
    DT = 1/FREQ
    STEPS = int(TIME/DT)
    MAX_VEL = 0.8
    MIN_VEL = -0.8
    KV = 1.
    KW = 1.
    SIM_STEP = 5
    OFFSET_DISTANCE = 0.02  # Offset distance from surface

    rx, ry = (0.1, 0.1)
    obs_pos = np.array([0.2, 0.2])

    # Create the robot and some joint angles
    robot = planar_robot("/home/louis/Git/playground/2D/", "four_link_planar.urdf.xacro")
    robot.q = robot.qc if which == "circle" else robot.qsine
    # Plot robot in environment and setup axes
    env = robot.plot(robot.q, backend="pyplot")
    ax = env.ax
    ax.view_init(elev=90, azim=-90)
    ax.set_xlim(0, 0.6)
    ax.set_ylim(-0.2, 0.4)
    fig = plt.gcf()

    if which == "circle":
        # Plot a circle
        u, v = np.mgrid[0:2 * np.pi:50j, 0:np.pi:50j]
        x = rx*np.cos(u) * np.sin(v) + obs_pos[0]
        y = ry*np.sin(u) * np.sin(v) + obs_pos[1]
        z = 0.01*np.cos(v)
        ax.plot_surface(x, y, z, color="r", alpha=0.2)
        circle_points, normal_points = discretize_circle_with_normals(rx, STEPS, center=obs_pos)

        start_pos = circle_points[0]
        start_ori = -normal_points[0]
        Rd = np.array([
            [start_ori[0], -start_ori[1], 0],
            [start_ori[1], start_ori[0], 0],
            [0, 0, 1]
        ])
    elif which == "sine":
        x = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
        y = 0.1*np.sin((x/0.1) + 0.5) + 0.2
        z = np.zeros_like(x)  # Keep curve in XY plane
        ax.plot(x, y, z, color='red')

        start_pos = (start_x_sine, 0.1*np.sin((start_x_sine/0.1) + 0.5) + 0.2)
        start_ori = get_normal(which=which, point=start_pos)
        Rd = np.array([
            [start_ori[0], -start_ori[1], 0],
            [start_ori[1], start_ori[0], 0],
            [0, 0, 1]
        ])
    else:
        raise "Invalid option for which curve to track. Should be either 'circle' or 'sine'"

    # wait_for_user(fig, key='w')
    # Make sure robot is in the right configuration to start with
    while True:
        _q_curr = robot.q
        _x_pose = robot.fkine(_q_curr)

        _x_curr = _x_pose.t[:2]
        _x_error = KV*(start_pos - _x_curr)
        _x_vel = _x_error/DT
        _x_vel[np.where(_x_vel > MAX_VEL)] = MAX_VEL
        _x_vel[np.where(_x_vel < MIN_VEL)] = MIN_VEL

        _theta, _axis = orientation_error(Rd, _x_pose.R)
        _omega = (_theta * _axis)[-1]/DT

        _jacob0 = robot.jacob0(robot.q)[[0, 1, -1], :]
        _qd = np.linalg.pinv(_jacob0) @ np.append(KV*_x_vel, KW*_omega)
        robot.q = _qd*DT + _q_curr

        # Update the environment
        env.step()
        if np.linalg.norm(_x_pose.t[:2] - start_pos) < 1e-6 and abs(_omega) < 1e-6:
            break

    wait_for_user(fig, key='w')

    ang_error_norm = []
    ang_error_perc = []
    ori_error_hist = []

    # Control loop
    for control_idx in range(1, STEPS):
        q_curr = robot.q
        x_pose = robot.fkine(q_curr)
        x_curr = x_pose.t[:2]

        # Replace x_vel with tangential velocity (assume no reference trajectory)
        current_normal = get_normal(which=which, center=obs_pos, point=x_curr)
        tangential_dir = np.array([-current_normal[1], current_normal[0]])
        x_tan = 0.1*tangential_dir

        # TODO: add noise to see how having a normal component in the velocity (for surface tracking) is similar to having a force that keeps the ee on the surface

        # Calculate angular velocity based on curvature
        kappa = 1/rx if which == "circle" else sine_curvature(x_curr[0])
        desired_ori = -get_normal(which=which, center=obs_pos, point=x_curr) if which == "circle" else get_normal(which=which, point=x_curr)
        Rd = np.array([
            [desired_ori[0], -desired_ori[1], 0],
            [desired_ori[1], desired_ori[0], 0],
            [0, 0, 1]
        ])
        theta_e, axis_e = orientation_error(Rd, x_pose.R)
        omega_error = (theta_e*axis_e)[-1] / DT
        omega_curvature = kappa*np.linalg.norm(x_tan) + omega_error

        # Reference trajectory tracking: calculate velocity based on error to the next point on the circle
        if which == "circle":
            if reference_traj:
                desired_pos = circle_points[control_idx]
                x_curr = x_pose.t[:2]
                x_error = desired_pos - x_curr
                x_from_error = x_error/DT
                desired_ori = -normal_points[control_idx]
                Rd = np.array([
                    [desired_ori[0], -desired_ori[1], 0],
                    [desired_ori[1], desired_ori[0], 0],
                    [0, 0, 1]
                ])

                # Calculate angular velocity based on orientation error
                theta, axis = orientation_error(Rd, x_pose.R)
                omega_from_error = (theta * axis)[-1] / DT

                ang_error_norm.append(np.linalg.norm(omega_from_error - omega_curvature))
                ang_error_perc.append(100 * (abs(omega_from_error - omega_curvature) / omega_curvature))
                ori_error_hist.append(theta * axis * 180 / np.pi)

                x_vel = x_from_error
                omega = omega_from_error

            else:
                x_vel = x_tan
                omega = omega_curvature

        elif which == "sine":
            x_vel = x_tan
            omega = omega_curvature

        # Cap the velocity
        # x_vel[np.where(x_vel > MAX_VEL)] = MAX_VEL
        # x_vel[np.where(x_vel < MIN_VEL)] = MIN_VEL

        # Get joint velocities then integrate to get next joint configuration
        jacob0 = robot.jacob0(robot.q)[[0, 1, -1], :]
        qd = np.linalg.pinv(jacob0) @ np.append(x_vel, omega)
        robot.q = qd*DT + q_curr

        # Update the environment
        if control_idx % SIM_STEP == 0:
            env.step()

    if reference_traj:
        print(f"Errors between angular velocity derived from orientation error and curvature-based angular velocity:")
        print(f"max ang error norm: {np.max(ang_error_norm)}, avg error norm: {np.mean(ang_error_norm)}")
        print(f"max ang error %: {np.max(ang_error_perc)}%, avg error %: {np.mean(ang_error_perc)}%")

        print(f"Error between desired and actual orientation:")
        print(f"max ori error: {np.max(ori_error_hist)}deg, avg ori error: {np.mean(ori_error_hist)}deg")

    plt.show(block=1)

main()