import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from superquadric import SuperquadricObject
from min_dists import MinDist2D, MinDist3DTransl, MinDist3DRot
from casadi_min_dist import MinDist3D as cas_md3d

Ra = (0.1, 0.2, 0.13)  # Circle radius
Rb = (0.11, 0.0375, 0.11)
eps_a = (1.0, 1.0)
eps_b = (0.5, 0.5)

x_opt_history = np.load('with_ori_3Drot.npy')  # load

ca_init = x_opt_history[0, :3]
qa_init = tuple(x_opt_history[0, 3:])
cb_init = (-0.545, -0.080, 0.15)  # Obstacle position
qb_init = (1.0, 0.0, 0.0, 0.0)  # Obstacle orientation

def static_test():
    # Create sq object
    eps = 5e-1
    idx_we_want = 396
    x_opt_history = np.load('with_ori_3Drot.npy')  # load

    ca_init = [-0.2454, 0.5, 0.29]
    cb_init = [-0.45, -0.080, 0.15]
    qa_init = (0.9396926, 0.0, 0.0, 0.3420201)
    qb_init = (1.0, 0.0, 0.0, 0.0)

    s1 = SuperquadricObject(*Ra, *eps_a, pos=ca_init, quat=qa_init)
    s2 = SuperquadricObject(*Rb, *eps_b, pos=cb_init, quat=qb_init)

    # Create optimisation problem
    optimiser = MinDist3DRot(ca_init, cb_init, Ra, Rb, eps_a, eps_b, qa_init, qb_init, objective="NORM")
    c_optimiser = cas_md3d(ca_init, cb_init, Ra, Rb, eps_a, eps_b, list(qa_init), list(qb_init))

    for nmb in range(10):

        optimiser.set_params(ca_init, cb_init, qa_init, qb_init)
        start_time = time.time()
        xa, lambda_a, xb, lambda_b = optimiser.get_primal_dual_solutions(requires_grad=False)
        print("cvxpy ", time.time() - start_time)
        print(list(np.round(xa, 6)) + list(np.round(xb, 6)), np.round(lambda_a, 6), np.round(lambda_b, 6), np.round(optimiser.get_optimal_value(), 6))
        print()

        if nmb > 0:
            x_guess = x_star
        else:
            xa = [0, 0, 0]
            xb = [2.1, -2.1, 0.0]
            x_guess = list(xa) + list(xb)
        c_optimiser.set_params(ca_init, cb_init, list(qa_init), list(qb_init))
        start_time = time.time()
        x_star, lambda_star = c_optimiser.get_primal_dual_solutions(x_guess)
        print("casadi ", time.time() - start_time)
        print(np.round(x_star, 6), np.round(lambda_star, 6), np.round(c_optimiser.get_optimal_value(), 6))
        print()

        ca_init[0] += 0.1


        # start_time = time.time()
        # optimiser_t.set_params(ca_init, cb_init)
        # xa, lambda_a, xb, lambda_b = optimiser_t.get_primal_dual_solutions(requires_grad=False)
        # print("translational ", time.time() - start_time)


    sys.exit()
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
# replay_motion()

def quick_test():
    # Create sq object
    Ra = (1, 1, 1)  # Circle radius
    eps_a = (1.0, 1.0)

    s1 = SuperquadricObject(*Ra, *eps_a, pos=(0,0,0), quat=(1,0,0,0))
    ax = plt.subplot(111, projection='3d')
    s1.plot_sq(ax, 'green')
    ax.plot(1, 1, 1, 'ro-')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-1.1, 1.1)
    plt.show()

static_test()