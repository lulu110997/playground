import sys
import time
from utils import *
import matplotlib.pyplot as plt
import numpy as np
from roboticstoolbox.tools import trajectory
from spatialmath import SE3, SO3, UnitQuaternion
from VelocityControllers import VelocityController
from casadi_min_dist import MinDist3D
from superquadric import SuperquadricObject

REPLAY = 0
SAVE = 0

ndim = 6  # 2 for 2D, 3 for 3D transl, 6 for 3D transl and rotation

# CBF param
GAMMA = 0.4  # Used in CBF calculation

# P gain
Kv = 0.4  # position
Kw = 1.0  # orientation

# Shape param
Ra = (0.12/2, 0.045/2, 0.27/2)
Rb = (0.11, 0.0375, 0.11)
eps_a = (0.1, 0.5)
eps_b = (0.1, 0.5)

# Initial and target shape positions
xb_init = (-0.545, -0.080, 0.15)  # Obstacle position
# qb_init = (0.9659258, 0, 0, -0.258819)  # Obstacle orientation
qb_init = (1.0, 0.0, 0.0, 0.0)  # Obstacle orientation
xa_init = (-0.454, 0.5, 0.29)  # Initial robot position
qa_init = (1.0, 0, 0, 0.0)  # Initial robot orientation
# qa_init = (0.9990482, 0, 0, 0.0436194)  # Initial robot orientation
# qa_init = (0.9396926, 0, 0, 0.3420201)
xa_tgt = (-0.454, -0.5, 0.29)  # Final robot position
qa_tgt = qa_init
# qa_tgt = (0.9396926, 0, 0, 0.3420201)
# qa_tgt = (1, 0, 0, 0)  # Final robot orientation

# Velocity limits
UB = np.array([0.2]*3 + [1.0]*3)  # Upper bound
LB = np.array([-0.2]*3 + [-1.0]*3)  # Lower bound

FREQ = 100.0
TIME = 15.0
TIME_SCALE = 0.1
DT = 1.0/FREQ
STEPS = int(TIME/DT)

SIM_START = 700
SIM_END = 1050
# TODO: desired behaviour when traj is inside the obstacle?? note that the optimiser is tryuing to minimiuse the error as much as possible but this means 'staying' in the curr pos

W = np.diag([100, 100, 100, 1, 1, 1])

########################################################################################################################

if __name__ == '__main__':
    # Create a trajectory
    initial_pose = SE3(xa_init) @ UnitQuaternion(s=qa_init[0], v=qa_init[1:]).SE3()
    final_pose = SE3(xa_tgt) @ UnitQuaternion(s=qa_tgt[0], v=qa_tgt[1:]).SE3()

    x_traj = trajectory.ctraj(initial_pose, final_pose, STEPS)

    # Histories
    x_opt_history = np.zeros((STEPS, ndim+1))
    xd_opt_history = np.zeros((STEPS, ndim))
    lagrange_history = np.zeros((STEPS, 2))
    sqa_closest_history = np.zeros((STEPS, 3))
    sqb_closest_history = np.zeros((STEPS, 3))
    optimisation_h_history = np.zeros((STEPS, 1))
    optimisation_hd_history = np.zeros((STEPS, ndim))

    x_opt_curr = np.array(xa_init)
    qa_curr = qa_init

    x_opt_history[0, :3] = xa_init
    x_opt_history[0, 3:] = qa_init

    if REPLAY:
        x_opt_history = np.load('with_ori_3Drot.npy')  # load
        sqa_closest_history = np.load('sqa_closest_history.npy')
        sqb_closest_history = np.load('sqb_closest_history.npy')
    else:
        # Create optimiser
        obj = MinDist3D(ca=list(xa_init), cb=list(xb_init), ra=Ra, rb=Rb, eps_a=eps_a, eps_b=eps_b, qa=list(qa_init), qb=list(qb_init))
        vel_cont = VelocityController(ndim, 1, W=W)
        cnt = 0
        x_star = [0.1, 0.1, 0.1, 0, 0, 0]

        # Control loop
        for idx in range(1, STEPS):
            # print(idx)
            acnt = time.time()
            # Change sq parameters in optimiser
            obj.set_params(ca=list(x_opt_curr), cb=list(xb_init), qa=list(qa_curr), qb=list(qb_init))
            x_star, lambda_star = obj.get_primal_dual_solutions(x_star)

            # Optimisation approach
            x_error = x_traj[idx].t - x_opt_curr
            theta, a_hat = calc_quat_error(UnitQuaternion(qa_curr[0], qa_curr[1:]), UnitQuaternion(x_traj[idx].R))
            vel = np.array(((Kv*x_error) / DT, (Kw*theta*a_hat) / DT)).reshape(6, 1)

            # CBF stuff
            G_opt = -np.array(obj.sensitivity_analysis())  # CBF derivative
            # if SAVE:
            #     with open("default.txt", "ab") as f:
            #         f.write(b"\n")
            #         np.savetxt(f, G_opt)
            h_opt = GAMMA * obj.get_optimal_value()  # CBF exponential gamma*hx
            vel_cont.set_param(vel, G_opt, h_opt, UnitQuaternion(qa_curr))
            xd_opt_des = vel_cont.get_solution()

            # if no CBF stuff
            # xd_opt_des = vel.squeeze()

            cnt += time.time() - acnt

            # Integrate to obtain forward pose
            next_x_opt = x_opt_curr + xd_opt_des[:3]*DT

            ha = xd_opt_des[3:] * (DT * 0.5)  # half angle
            dR = UnitQuaternion(s=1.0, v=[ha[0], ha[1], ha[2]])
            next_qa = dR * UnitQuaternion(qa_curr)

            # Update current state for next iteration
            x_opt_curr = next_x_opt
            qa_curr = next_qa.vec

            # Save states
            x_opt_history[idx, :3] = x_opt_curr
            x_opt_history[idx, 3:] = qa_curr
            sqa_closest_history[idx, :] = x_star[:3]
            sqb_closest_history[idx, :] = x_star[3::]
            optimisation_h_history[idx, :] = h_opt/GAMMA
            optimisation_hd_history[idx, :3] = G_opt[:3]

        print(f"{1000*(cnt/STEPS)} ms/iter")

    # sys.exit()
    s1 = SuperquadricObject(*Ra, *eps_a, pos=xa_init, quat=qa_init)
    s2 = SuperquadricObject(*Rb, *eps_b, pos=xb_init, quat=qb_init)

    if SAVE:
        np.save('with_ori_3Drot.npy', x_opt_history) # save
        np.save('sqa_closest_history.npy', sqa_closest_history) # save
        np.save('sqb_closest_history.npy', sqb_closest_history) # save
        sys.exit()

    plt.figure()
    ax = plt.subplot(111, projection='3d')
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_xlim(-1.0, 0.2)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(0.0, 0.55)
    s2.plot_sq(ax, 'red')
    ax.plot(x_opt_history[:, 0], x_opt_history[:, 1], x_opt_history[:, 2], color='blue')
    ax.plot(x_traj.t[:,0], x_traj.t[:,1], x_traj.t[:,2], color='g')
    ax.scatter(x_opt_history[::200, 0], x_opt_history[::200, 1], x_opt_history[::200, 2], color='black', marker='x', linewidths=1.5)

    for idx in range(SIM_START, SIM_END, int(TIME*TIME_SCALE)):
        s1_handle = s1.plot_sq(ax, 'green')
        traj_handle = ax.scatter(x_traj[idx].t[0], x_traj[idx].t[1], x_traj[idx].t[2], color='g', marker='o', linewidth=2)

        line_handle = ax.plot((sqa_closest_history[idx, 0], sqb_closest_history[idx, 0]),
                              (sqa_closest_history[idx, 1], sqb_closest_history[idx, 1]),
                              (sqa_closest_history[idx, 2], sqb_closest_history[idx, 2]), 'ro-')
        plt.pause(1e-16)
        s1_handle.remove()
        line_handle[0].remove()
        traj_handle.remove()
        s1.set_pose(pos=x_opt_history[idx, :3], quat=tuple(x_opt_history[idx, 3:]))

    s1_handle = s1.plot_sq(ax, 'green')
    # line_handle = ax.plot((sqa_closest_history[idx, 0], sqb_closest_history[idx, 0]),
    #                       (sqa_closest_history[idx, 1], sqb_closest_history[idx, 1]),
    #                       (sqa_closest_history[idx, 2], sqb_closest_history[idx, 2]), 'ro-')

    # Plots
    dist_fig, dist_ax = plt.subplots(2)
    dist_fig.suptitle('h function')
    dist_ax[0].plot(range(0, STEPS-1), np.round(optimisation_h_history[1:, 0], 3), label="distance", color='r', lw=2)

    dist_ax[1].plot(range(0, STEPS-1), np.round(optimisation_hd_history[1:, 0], 3), label="optimisation x grad", color='r', lw=2)
    dist_ax[1].plot(range(0, STEPS-1), np.round(optimisation_hd_history[1:, 1], 3), label="optimisation y grad", color='g', lw=2)
    dist_ax[1].plot(range(0, STEPS-1), np.round(optimisation_hd_history[1:, 2], 3), label="optimisation z grad", color='b', lw=2)
    dist_ax[0].legend(); dist_ax[1].legend()
    plt.show()
