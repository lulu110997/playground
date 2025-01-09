import numpy as np
import matplotlib.pyplot as plt
from superquadric import SuperquadricObject
from min_dists import MinDist2D, MinDist3DTransl, MinDist3DRot

Ra = (0.1, 0.2, 0.13)  # Circle radius
Rb = (0.11, 0.0375, 0.11)
eps_a = (1.0, 1.0)
eps_b = (0.5, 0.5)

x_opt_history = np.load('with_ori_3Drot.npy')  # load

xa_init = x_opt_history[0, :3]
qa_init = tuple(x_opt_history[0, 3:])
xb_init = (-0.545, -0.080, 0.15)  # Obstacle position
qb_init = (1.0, 0.0, 0.0, 0.0)  # Obstacle orientation

def static_test():
    # Create sq object
    eps = 1e-4
    idx_we_want = 396
    x_opt_history = np.load('with_ori_3Drot.npy')  # load

    # for i in []

    xa_init = x_opt_history[idx_we_want, :3]
    xa_init[0] += eps
    xa_init[0] = -0.4
    qa_init = tuple(x_opt_history[idx_we_want, 3:])

    s1 = SuperquadricObject(*Ra, *eps_a, pos=xa_init, quat=qa_init)
    s2 = SuperquadricObject(*Rb, *eps_b, pos=xb_init, quat=qb_init)

    # Create optimisation problem
    optimiser = MinDist3DRot(xa_init, xb_init, Ra, Rb, eps_a, eps_b, qa_init, qb_init, objective="NORM")
    xa, lambda_a, xb, lambda_b = optimiser.get_primal_dual_solutions(requires_grad=False)
    xd, yd, zd, q0d, q1d, q2d, q3d = optimiser.sensitivity_analysis()

    ax = plt.subplot(111, projection='3d')
    s1.plot_sq(ax, 'green')
    s2.plot_sq(ax, 'red')
    ax.plot((xa[0], xb[0]), (xa[1], xb[1]), (xa[2], xb[2]), 'ro-')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(0.0, 0.55)
    print(optimiser.get_optimal_value())
    ax.text2D(0.05, 0.95, f" {round(xd, 3)}, {round(yd, 3)}, {round(zd, 3)}   "
                          f" {round(q0d, 3)}, {round(q1d, 3)}, {round(q2d, 3)}, {round(q3d, 3)}",
              transform=ax.transAxes)
    plt.show()

def replay_motion():
    STEPS = x_opt_history.shape[0]
    SKIP = 3

    s1 = SuperquadricObject(*Ra, *eps_a, pos=xa_init, quat=qa_init)
    s2 = SuperquadricObject(*Rb, *eps_b, pos=xb_init, quat=qb_init)

    plt.figure()
    ax = plt.subplot(111, projection='3d')
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(0.0, 0.55)
    s2.plot_sq(ax, 'red')
    ax.plot(x_opt_history[:, 0], x_opt_history[:, 1], x_opt_history[:, 2])

    for idx in range(0, STEPS, SKIP):
        # if x_opt_history[idx, 0] < -0.5:
        #     print(idx)
        #     plt.pause(10)
        s1_handle = s1.plot_sq(ax, 'green')
        plt.pause(1e-16)
        s1_handle.remove()
        s1.set_pose(pos=x_opt_history[idx, :3], quat=tuple(x_opt_history[idx, 3:]))

    s1_handle = s1.plot_sq(ax, 'green')
    plt.show()

# static_test()
replay_motion()