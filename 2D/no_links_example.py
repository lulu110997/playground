import math
import sys

import matplotlib
matplotlib.use('TkAgg')  # or 'QtAgg'
import matplotlib.pyplot as plt
import roboticstoolbox as rtb
from spatialmath import SE3

import numpy as np

from cbf_diff_sq.mindistcalcmulti import MinDistMulti3D
from cbf_diff_sq.superquadric import SuperquadricObject
from cbf_diff_sq.velcont import VelocityController, VelocityControllerWeighted

plt.ion()


def augment_w_to_3d(w_2d, z_eig=1e-6):
    w_2d = np.array(w_2d, dtype=float)
    if w_2d.shape != (2, 2):
        raise ValueError(f"Expected a 2x2 matrix, got {w_2d.shape}")

    # Keep the matrix symmetric to ensure a real eigendecomposition.
    w_xy = 0.5 * (w_2d + w_2d.T)
    w_3d = np.eye(3) * float(z_eig)
    w_3d[:2, :2] = w_xy
    return w_3d


def plot_small_ellipsoid_from_metric(ax, center, w_2d, robot_abc, scale=1.0, z_radius=1e-6,
                                     color="cyan", alpha=0.18, n_u=80, n_v=80):
    w_2d = np.array(w_2d, dtype=float)
    w_xy = 0.5 * (w_2d + w_2d.T)
    eigvals_xy, eigvecs_xy = np.linalg.eigh(w_xy)
    eigvals_xy = np.clip(eigvals_xy, 1e-12, None)

    # Keep XY shape from W, but decouple from the tiny Z thickness.
    inv_sqrt_xy = 1.0 / np.sqrt(eigvals_xy)
    inv_sqrt_xy = inv_sqrt_xy / np.mean(inv_sqrt_xy)

    robot_abc = np.array(robot_abc, dtype=float)
    base_xy = float(np.mean(robot_abc[:2]))
    radii_xy = scale * base_xy * inv_sqrt_xy
    radii_xy = np.clip(radii_xy, 0.6 * np.min(robot_abc[:2]), 1.4 * np.max(robot_abc[:2]))
    radii = np.array([radii_xy[0], radii_xy[1], max(float(z_radius), 1e-9)])

    rot_3d = np.eye(3)
    rot_3d[:2, :2] = eigvecs_xy

    u = np.linspace(0, 2 * np.pi, n_u)
    v = np.linspace(0, np.pi, n_v)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    sphere = np.stack([x, y, z], axis=0).reshape(3, -1)
    ellipsoid = rot_3d @ (np.diag(radii) @ sphere)
    center = np.array(center, dtype=float).reshape(3, 1)
    ellipsoid = ellipsoid + center

    xe = ellipsoid[0].reshape(x.shape)
    ye = ellipsoid[1].reshape(y.shape)
    ze = ellipsoid[2].reshape(z.shape)
    return ax.plot_surface(xe, ye, ze, color=color, alpha=alpha, linewidth=0)

def main_traj(obs_pos, initial_pos, final_pos):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=90, azim=-90)

    # Set limits
    ax.set_xlim(0.1, 0.5)
    ax.set_ylim(0., 0.4)
    ax.set_zlim(-0.2, 0.2)

    rob_sq = SuperquadricObject(a=0.05/2, b=0.15/2, c=0.01/2, eps1=1.0, eps2=1.0, pos=initial_pos, quat=(1,0,0,0))
    robot_params = [{"c": rob_sq.pos.squeeze().tolist(), "r": rob_sq.get_abc(), "eps": rob_sq.get_eps(), "q": rob_sq.quat}]
    # rob_sq_handle = rob_sq.plot_sq(ax, colour="green")

    # Draw small sphere to represent a point obstacle
    obs_sq = SuperquadricObject(a=0.1/2, b=0.1/2, c=0.01/2, eps1=1.0, eps2=1.0, pos=obs_pos, quat=(1,0,0,0))
    obs_sq.plot_sq(ax=ax, colour="red", alpha=0.5)

    # Draw small sphere to represent the goal position
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = 0.01*np.cos(u) * np.sin(v) + final_pos[0]
    y = 0.01*np.sin(u) * np.sin(v) + final_pos[1]
    z = 0.01*np.cos(v) + final_pos[2]
    ax.plot_surface(x, y, z, color="g")

    obs_params = [{"c": obs_sq.pos.squeeze().tolist(), "r": obs_sq.get_abc(), "eps": obs_sq.get_eps(), "q": obs_sq.quat}]
    dist_calc = MinDistMulti3D(robot_params, obs_params, io_eps=1e-16)

    vel_cont = VelocityControllerWeighted(ub=MAX_VEL, lb=MIN_VEL, ndim=2)

    x_optimal = None
    lam_g0 = False
    prev_vel = None

    # Control loop
    robot_traj = rtb.ctraj(SE3(x=initial_pos[0], y=initial_pos[1], z=initial_pos[2]),
                           SE3(x=final_pos[0], y=final_pos[1], z=final_pos[2]), STEPS)
    x_curr = initial_pos

    plt.show()
    plt.pause(1)

    for control_idx in range(1, len(robot_traj)):
        step_count = ax.text2D(0.05, 0.95, control_idx, transform=ax.transAxes,
                               fontsize=12, color='black')
        next_pos = robot_traj[control_idx].t
        x_error = np.array(next_pos - x_curr)[:2]
        x_vel = x_error/DT
        dist_calc.set_robot_pose(ca=rob_sq.pos.squeeze().tolist(), qa=rob_sq.quat.elements.tolist())
        x_optimal, lam_g0 = dist_calc.get_primal_dual_solutions(x_optimal, lam_g0)
        obs_points = x_optimal[3:]
        obs_normal = obs_sq.surface_normal(obs_points[0], obs_points[1], obs_points[2])
        obs_normal = obs_normal[:2]


        # Calculate safety function between the links and point obstacle
        dist_obs = dist_calc.get_distances()[0]
        h = (dist_obs - SAFETY_OFFSET) * GAMMA
        nabla_h = np.array(dist_calc.sensitivity_analysis()[:2])
        # radians = np.arctan2(np.cross(x_vel, obs_normal), np.dot(x_vel, obs_normal))
        # x_tan = math.exp(-100*dist_obs)*(x_vel - np.dot(x_vel, obs_normal))
        # print(np.dot(x_vel, obs_normal))
        vel_cont.set_param(xd_tgt=x_vel, G_matr=nabla_h, h_matr=h)
        xd = vel_cont.get_solution()
        x_curr = np.append(xd, 0) * DT + x_curr

        rob_sq.set_pose(x_curr, (1,0,0,0))

        # Update the environment
        # if control_idx % SIM_SKIP ==0 and control_idx > 300:
            # print("np." + repr(vel_cont.W.value) + ",")
            # print(np.linalg.eig(vel_cont.W.value).eigenvalues)

        if control_idx % SIM_SKIP == 0 and control_idx > 0.2500:
            # Analyse solution
            normal_unit_vector = (nabla_h / np.linalg.norm(nabla_h)).reshape(2, 1)
            normal_proj = np.outer(normal_unit_vector, normal_unit_vector)
            tangent_proj = np.eye(2) - normal_proj
            xd_n, xd_t = xd @ normal_proj, xd @ tangent_proj
            # print(f"normal: {np.linalg.norm(xd_n)}, tangent: {np.linalg.norm(xd_t)}, distance:{h}")
            print(h, np.linalg.eig(vel_cont.W.value).eigenvalues)

            # tangent (choose one direction)
            tx, ty = -obs_normal[1], obs_normal[0]
            L = 0.5
            x0, y0 = obs_points[0], obs_points[1]
            tangent_line, = ax.plot([x0 - L * tx, x0 + L * tx],
                                   [y0 - L * ty, y0 + L * ty],
                                   color='blue')

            # Plot one tiny ellipsoid per control step from the 2x2 weight matrix.
            ellipsoid_handle = plot_small_ellipsoid_from_metric(
                ax=ax,
                center=rob_sq.pos.squeeze(),
                w_2d=vel_cont.W.value,
                robot_abc= rob_sq.get_abc(),
                scale=1.0,
                z_radius=1e-6,
                color="cyan",
                alpha=0.9,
            )

            normal_vector = ax.quiver(obs_points[0], obs_points[1], 0,
                                      obs_normal[0], obs_normal[1], 0, color='blue', length=dist_obs, normalize=True)
            velocity_vector = ax.quiver(rob_sq.pos[0], rob_sq.pos[1], 0,
                                        xd[0], xd[1], 0, color='green', length=np.linalg.norm(xd)*10, normalize=True)

            desired_pos_handle, = ax.plot(next_pos[0], next_pos[1], 0, 'bx')
            robot_travelled_handle, = ax.plot(x_curr[0], x_curr[1], 0, 'gx')
            rob_sq_handle = rob_sq.plot_sq(ax, 'green')

            # Show plots
            plt.draw()
            plt.pause(0.001)

            # Remove handles
            normal_vector.remove()
            desired_pos_handle.remove()
            rob_sq_handle.remove()
            tangent_line.remove()
            velocity_vector.remove()
            ellipsoid_handle.remove()

            # velocity_vector_diff = np.dot(prev_vel, xd) / (np.linalg.norm(prev_vel) * np.linalg.norm(xd))
            # print(velocity_vector_diff)

        prev_vel = xd
        step_count.remove()

    plt.show(block=True)

if __name__ == '__main__':
    FREQ = 100
    DT = 1/FREQ
    TIME = 60
    STEPS = int(TIME*FREQ)
    MAX_VEL = 0.02
    MIN_VEL = -0.02
    SIM_SKIP = 11
    SAFETY_OFFSET = 1e-3
    GAMMA = 1.

    # starting_point = np.array([0.9, 0.2, 0.]) is kinda a circle
    starting_point = np.array([1.0045, 0.2, 0.])
    # starting_point = np.array([0.45, 0.25, 0.])
    goal_point = np.array([0.1, 0.2, 0.])
    obs_point = np.array([0.3, 0.23, 1e-16])

    # Test 2, north/south points giving different behaviour from west/east points
    # not eccentricity problem, tested with circle robot and sideways fat ellipse
    # starting_point = np.array([0.3, -0.2, 0.])
    # # starting_point = np.array([0.33, 0., 0.])
    # goal_point = np.array([0.3, 0.4, 0.])
    # obs_point = np.array([0.27, 0.2, 1e-16])

    # Test 3, robot changes direction??
    starting_point = np.array([0.43, 0., 0.])
    goal_point = np.array([0.3, 0.4, 0.])
    obs_point = np.array([0.3, 0.2, 1e-16])

    main_traj(obs_point, starting_point, goal_point)
    # main_no_traj(obs_point, starting_point, goal_point)
