import sys

import matplotlib.pyplot as plt
import roboticstoolbox as rtb
from spatialmath import UnitQuaternion, SE3

import numpy as np

from cbf_diff_sq.superquadric import SuperquadricObject
from cbf_diff_sq.velcont import JointVelocityController

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

# def ellipsoid_inside_outside(x, c, R, axes):
#     """
#     Inside-outside function of a rotated ellipsoid.
#
#     Parameters
#     ----------
#     x : (3,) array
#         Query point
#     c : (3,) array
#         Ellipsoid centre
#     R : (3,3) array
#         Rotation matrix
#     axes : (3,) array
#         Semi-axis lengths [a, b, c]
#
#     Returns
#     -------
#     f : float
#         Inside-outside value (negative inside, zero surface, positive outside)
#     """
#
#     D = np.diag(1.0 / axes**2)
#     A = R @ D @ R.T
#
#     d = x - c
#     f = d.T @ A @ d - 1.0
#
#     return f
#
#
# def ellipsoid_df_dc(x, c, R, axes):
#     """
#     Derivative of inside-outside function w.r.t. ellipsoid centre.
#
#     Parameters
#     ----------
#     x : (3,) array
#         Query point
#     c : (3,) array
#         Ellipsoid centre
#     R : (3,3) array
#         Rotation matrix
#     axes : (3,) array
#         Semi-axis lengths [a, b, c]
#
#     Returns
#     -------
#     f : float
#         Inside-outside value (negative inside, zero surface, positive outside)
#
#     """
#
#     D = np.diag(1.0 / axes**2)
#     A = R @ D @ R.T
#
#     d = x - c
#     grad_c = -2.0 * (A @ d)
#
#     return grad_c

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
        self.qf = np.array([-0.30414485, -0.15784471,  1.41410639,  1.34961703])

        self.addconfiguration("qz", self.qz)
        self.addconfiguration("qz", self.qs)

def update_sq(link_poses, link_sqs):
    end_offset = []
    for tf_i, tf in enumerate(link_poses):
        if tf_i == 0:
            tf_prev = link_poses[0]
            continue

        tf_curr = link_poses[tf_i]

        sq_t = (tf_curr.t + tf_prev.t) / 2
        sq_rot = UnitQuaternion(tf_prev.R)
        link_sqs[tf_i - 1].set_pose(sq_t, tuple(sq_rot.vec))
        end_offset.append(tf_prev.inv()@SE3.Rt(tf_prev.R, sq_t))
        tf_prev = tf_curr

    return end_offset

def pad_to_2x4(A):
    target_rows, target_cols = 2, 4
    rows, cols = A.shape

    pad_rows = max(0, target_rows - rows)
    pad_cols = max(0, target_cols - cols)

    # pad only bottom and right
    padded = np.pad(A, ((0, pad_rows), (0, pad_cols)), mode='constant')
    return padded

def main(obs_pos, final_pose, robot, q_start, ee=False):
    # Link lengths (metres)
    link_lengths = [0.2, 0.3, 0.3, 0.2]

    if not ee:
        link_names = ["link1", "link2", "link3", "link4"]
    else:
        link_names = ["link4"]
    # Create the robot and some joint angles
    robot.q = q_start
    # Plot robot in environment
    env = robot.plot(robot.q, backend="pyplot")
    ax = env.ax
    ax.view_init(elev=90, azim=-90)

    if not ee:
        link_sqs = [
            SuperquadricObject(a=link_lengths[0]/2, b=0.05, c=0.01, eps1=1.0, eps2=1.0),
            SuperquadricObject(a=link_lengths[1]/2, b=0.05, c=0.01, eps1=1.0, eps2=1.0),
            SuperquadricObject(a=link_lengths[2]/2, b=0.05, c=0.01, eps1=1.0, eps2=1.0),
            SuperquadricObject(a=link_lengths[3]/2, b=0.05, c=0.01, eps1=1.0, eps2=1.0)
                    ]
        tool_tr = update_sq(robot.fkine_all(robot.q)[2:], link_sqs)
    else:
        link_sqs = [
            SuperquadricObject(a=link_lengths[3]/2, b=0.05, c=0.01, eps1=1.0, eps2=1.0)
                    ]
        tool_tr = update_sq(robot.fkine_all(robot.q)[5:], link_sqs)

    link_sq_handles = [None] * len(link_sqs)
    for i in range(len(link_sq_handles)):
        link_sq_handles[i] = link_sqs[i].plot_sq(ax, 'green')

    # Draw small sphere to represent a point obstacle
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = 0.01*np.cos(u) * np.sin(v) + obs_pos[0]
    y = 0.01*np.sin(u) * np.sin(v) + obs_pos[1]
    z = 0.01*np.cos(v)
    ax.plot_surface(x, y, z, color="r")

    # Draw small sphere to represent the goal position
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = 0.01*np.cos(u) * np.sin(v) + final_pose[0]
    y = 0.01*np.sin(u) * np.sin(v) + final_pose[1]
    z = 0.01*np.cos(v)
    ax.plot_surface(x, y, z, color="g")

    vel_cont = JointVelocityController(ub=0.5, lb=-0.5, jac_shape=robot.jacob0(robot.q)[:2].shape,
                                       njoints=robot.q.shape[0], ndim=2, nconst=len(link_sqs))
    h = np.zeros((len(link_sqs), 1))
    nabla_h = np.zeros((len(link_sqs), robot.q.shape[0]))

    # Control loop
    # wait_for_user(plt.gcf(), key='w')
    plt.pause(1)
    for control_idx in range(STEPS):
        q_curr = robot.q
        x_curr = robot.fkine(q_curr).t[:2]
        x_error = final_pose - x_curr
        x_vel = x_error/DT
        x_vel[np.where(x_vel > MAX_VEL)] = MAX_VEL
        x_vel[np.where(x_vel < MIN_VEL)] = MIN_VEL

        jacob0 = robot.jacob0(robot.q)[:2]

        # qd_ref = np.linalg.pinv(jacob0) @ x_vel

        # Calculate safety function between the links and point obstacle
        for cbf_idx, sqs in enumerate(link_sqs):
            h[cbf_idx] = (sqs.inside_out_function(obs_pos[0], obs_pos[1], 0) - SAFETY_OFFSET)*GAMMA

            # Calculate gradient of safety function using ellipsoid matrix form
            R = sqs.quat.rotation_matrix[:2, :2]
            D = np.diag([1/(sqs.a**2), 1/(sqs.b**2)])
            x = obs_pos - sqs.pos[:2]
            nabla_h[cbf_idx] = (-2*R@D@R.transpose()@x) @ pad_to_2x4(robot.jacob0(robot.q, end=link_names[cbf_idx], tool=tool_tr[cbf_idx])[:2])

        vel_cont.set_param(xd_tgt=x_vel, G_matr=nabla_h, h_matr=h, robot_jac=jacob0); qd = vel_cont.get_solution()

        q_next = qd*DT + q_curr
        robot.q = q_next

        if not ee:
            tool_tr = update_sq(robot.fkine_all(robot.q)[2:], link_sqs)
        else:
            tool_tr = update_sq(robot.fkine_all(robot.q)[5:], link_sqs)

        # Update the environment
        if control_idx % SIM_SKIP == 0:
            # P_null = np.eye(4) - np.linalg.pinv(jacob0) @ jacob0
            # grad_g_null = P_null @ nabla_h.transpose()
            # print(np.linalg.norm(grad_g_null))
            # # 2. Get the residual (slack) of your CBF constraint
            # # residual = Lg_h @ dq.value + Lf_h + gamma * h
            # res = vel_cont.constraints[0].residual
            #
            # # 3. Calculate the Barrier Value
            # # Note: tau is the solver's duality gap, usually very small (e.g., 1e-8)
            # tau = vel_cont.prob.solver_stats.extra_stats['gap'] if 'gap' in vel_cont.prob.solver_stats.extra_stats else 1e-8
            # barrier_val = -tau * np.log(res)
            # print(f"tau: {tau:.2e}")
            # print(f"Constraint Residual: {res:.2e}")
            # print(f"Estimated Barrier Potential: {barrier_val:.2e}")
            # print(vel_cont.prob.parameters)
            for handles_idx in range(len(link_sq_handles)):
                link_sq_handles[handles_idx].remove()
                link_sq_handles[handles_idx] = link_sqs[handles_idx].plot_sq(ax, 'green')
            env.step()


    print(qd)
    print(h)
    print(nabla_h)
    print(robot.q)
    plt.show(block=1)

if __name__ == '__main__':
    FREQ = 100
    DT = 1/FREQ
    TIME = 500
    STEPS = int(TIME*FREQ)
    MAX_VEL = 0.2
    MIN_VEL = -0.2
    SIM_SKIP = 10

    robot2d = planar_robot("/home/louis/Git/playground/2D/", "four_link_planar.urdf.xacro")

    # TODO: Show the motion without the OA constraint

    # Testing motion towards the goal with obstacle in the way. Try playing around with SAFETY_OFFSET and GAMMA
    # GAMMA = 0.5
    # SAFETY_OFFSET = 3.  # Must be geq to 1
    # obs_point = np.array([0.69, 0.28])  # Testing motion towards the goal with obstacle in the way
    # goal_point = np.array([0.5, 0.23])
    # ee_bool = False
    # q_start = robot2d.qs

    # Testing motion towards the goal with obstacle in the way, deadlock occurs
    # GAMMA = 1.
    # SAFETY_OFFSET = 1.3  # Must be geq to 1
    # obs_point = np.array([0.6, 0.2105])  # Testing motion towards the goal with obstacle in the way
    # goal_point = np.array([0.5, 0.23])
    # ee_bool = False
    # q_start = robot2d.qs

    # Testing motion towards the goal with obstacle in the way, tuned Gamma, notice how the robot is still moving after
    # reaching the goal. Will revisit later on how this can be prevented. However, this is an interesting observation
    # since the robot is reconfiguring itself to maximise distance between robot body and obstacles even though we have
    # not explicitly stated this in the objective function
    # GAMMA = 3.
    # SAFETY_OFFSET = 1.3  # Must be geq to 1
    # obs_point = np.array([0.6, 0.2105])  # Testing motion towards the goal with obstacle in the way
    # goal_point = np.array([0.5, 0.23])
    # ee_bool = False
    # q_start = robot2d.qs

    # Testing motion towards the goal with obstacle in the way of the base link, play around with GAMMA to show base
    # link not moving
    # GAMMA = 0.1
    # SAFETY_OFFSET = 1.3  # Must be geq to 1
    # obs_point = np.array([0.08, -0.1])
    # goal_point = np.array([0.5, 0.23])
    # ee_bool = False
    # q_start = robot2d.qs

    # Testing nullspace motion, notice how it breaks objective function, changing objective function helps prevent it
    # from breaking, first add the velocity norm in the objective function, however, this puts preference on translation
    # as this minimises joint velocities. Therefore, penalise tracking error more heavily to prevent translational
    # avoidance
    # SIM_SKIP = 200
    # GAMMA = 0.1
    # SAFETY_OFFSET = 2.3  # Must be geq to 1. Note at 2.3, the safety function is close to zero but constraint is still satisfied. However, the robot still tries to reconfigure itself
    # obs_point = np.array([0.6, 0.2])
    # goal_point = np.array([0.5, 0.2])
    # ee_bool = 0
    # q_start = robot2d.qf
    # TODO: print out constraint value, check the dual variable and see why tracking error is introduced even though
    #  constraint is satisfied
    # Testing motion towards the goal with obstacle in the way, tuned Gamma, notice how changing objective functions
    # change the robot's motion
    # SIM_SKIP = 5
    # GAMMA = 3.
    # SAFETY_OFFSET = 1.3  # Must be geq to 1
    # obs_point = np.array([0.6, 0.2105])
    # # obs_point = np.array([0.08, -0.1]); GAMMA = 0.2  # --> notice what happens when we change gamma
    # goal_point = np.array([0.5, 0.23])
    # ee_bool = False
    # q_start = robot2d.qs

    # Discuss the usefulness of being able to do stuff in the joint space and possible application fields to highlight
    # these benefits. First, we don't need to calculate derivatives of the minimum distance w.r.t the orientation of the
    # robot body. By formulating the problem in the joint space, we are implicitly able to control the orientation of
    # the EE to maximise the distance between the robot body and obstacles. What are the benefits of this? Any cool
    # applications?
    # Can we design an algorithm based on this?
    # Working in joint space allows the robot to reconfigure its body intelligently and automatically to increase safety
    # and clearance, without explicitly handling complex orientation derivatives — making it powerful for redundancy
    # resolution, collision avoidance, and high-performance real-world robotics.
    # “Implicit Orientation Optimization for Collision Avoidance in Redundant Manipulators”
        # Core Idea: Show that joint-space minimum-distance optimization outperforms task-space formulations that
        # explicitly constrain orientation.
        # Experiments: Task-space QP with explicit orientation control and Position-only IK + collision avoidance heuristic

    # Deadlock
    # SIM_SKIP = 4
    # GAMMA = 1
    # SAFETY_OFFSET = 2.3  # Must be geq to 1
    # obs_point = np.array([0.3, 0.19])
    # goal_point = np.array([0.1, 0.2])
    # ee_bool = 0
    # q_start = robot2d.qf

    SIM_SKIP = 100
    GAMMA = 0.1
    SAFETY_OFFSET = 1.3  # Must be geq to 1
    obs_point = np.array([0.3, 0.21])
    goal_point = np.array([0.1, 0.2])
    ee_bool = False
    Rd = np.array([
        [ 0., -1.,  0.],
        [1., 0.,  0.],
        [0.,  0.,  1. ]
    ])
    start_pos = np.array([0.45, 0.2])

    while True:
        _q_curr = robot2d.q
        _x_pose = robot2d.fkine(_q_curr)

        _x_curr = _x_pose.t[:2]
        _x_error = (start_pos - _x_curr)
        _x_vel = _x_error/DT
        _x_vel[np.where(_x_vel > MAX_VEL)] = MAX_VEL
        _x_vel[np.where(_x_vel < MIN_VEL)] = MIN_VEL

        _theta, _axis = orientation_error(Rd, _x_pose.R)
        _omega = (_theta * _axis)[-1]/DT

        _jacob0 = robot2d.jacob0(robot2d.q)[[0, 1, -1], :]
        _qd = np.linalg.pinv(_jacob0) @ np.append(_x_vel, _omega)
        robot2d.q = _qd*DT + _q_curr

        if np.linalg.norm(_x_pose.t[:2] - start_pos) < 1e-6 and abs(_omega) < 1e-6:
            break
    q_start = robot2d.q

    main(obs_point, goal_point, robot2d, q_start, ee=ee_bool)