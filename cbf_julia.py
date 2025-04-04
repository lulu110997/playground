from sklearn.metrics import mean_squared_error
import time
import yaml
import matplotlib.pyplot as plt
import sys
# print(sys.path)
sys.path.append("/home/louis/Git/playground")
import numpy as np
from roboticstoolbox.tools import trajectory
from spatialmath import SE3, SO3, UnitQuaternion
from julia import Main
from VelocityControllers import VelocityController
from superquadric import SuperquadricObject
from utils import *

# plt.ion()
TEST_TYPE = 'compare with tracy'
TEST_DIR = 'test1'
TEST_NAME = 'test1'
REPLAY = 0; cwd = f'test cases/{TEST_TYPE}/{TEST_DIR}/{TEST_NAME}_'
SAVE = 0; sd = f'test cases/{TEST_TYPE}/{TEST_DIR}/{TEST_NAME}_'

with open(f"test cases/{TEST_TYPE}/{TEST_DIR}/{TEST_NAME}.yaml") as file:
    try:
        params = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

ndim = 6  # 2 for 2D, 3 for 3D transl, 6 for 3D transl and rotation

# CBF-QP parameters
GAMMA = params['GAMMA']  # Used in CBF
BETA = params['BETA']  # Used in DCOL
Kv = params['Kv']  # Gain for position
Kw = params['Kw']  # Gain for orientation
UB = np.array(params['VB'])  # Upper bound for linear velocity
LB = -np.array(params['VB'])  # Lower bound for angular velocity
W = np.diag(params['W'])  # Weighting for the velocities

# Superquadric parameters
Ra = params['Ra']
Rb = params['Rb']
eps_a = params['eps_a']
eps_b = params['eps_b']

# Initial and target shape poses
xb_init = params['xb_init']  # Obstacle position
xa_init = params['xa_init']  # Initial robot position
qa_init = params['qa_init']  # Initial robot orientation
qb_init = params['qb_init']  # Obstacle orientation
xa_tgt = params['xa_tgt']  # Final robot position
qa_tgt = params['qa_tgt']  # Final robot orientation

# Simulation parameters
FREQ = params['FREQ']
TIME = params['TIME']
DT = 1.0/FREQ
STEPS = int(TIME/DT)
SIM_START = params['SIM_START']
SIM_END = params['SIM_END']
TIME_SCALE = params['TIME_SCALE']

########################################################################################################################

if __name__ == '__main__':
    # Create the trajectory
    initial_pose = SE3(xa_init) @ UnitQuaternion(s=qa_init[0], v=qa_init[1:]).SE3()
    final_pose = SE3(xa_tgt) @ UnitQuaternion(s=qa_tgt[0], v=qa_tgt[1:]).SE3()
    x_traj = trajectory.ctraj(initial_pose, final_pose, STEPS)

    # Initialise functions required from Julia to obtain CBF using scaling factor
    create_shapes = Main.include("create_shapes.jl")
    get_α_J = Main.include("get_α_J.jl")
    create_shapes()

    if REPLAY:
        # Load histories
        x_opt_history = np.load(cwd + 'x_opt_history.npy')
        xd_opt_history = np.load(cwd + 'xd_opt_history.npy')
        sqa_closest_history = np.load(cwd + 'sqa_closest_history.npy')
        sqb_closest_history = np.load(cwd + 'sqb_closest_history.npy')
        optimisation_h_history = np.load(cwd + 'optimisation_h_history.npy')
        optimisation_hd_history = np.load(cwd + 'optimisation_hd_history.npy')
    else:
        # Histories
        x_opt_history = np.zeros((STEPS, ndim + 1))
        xd_opt_history = np.zeros((STEPS, ndim))
        lagrange_history = np.zeros((STEPS, 2))
        sqa_closest_history = np.zeros((STEPS, 3))
        sqb_closest_history = np.zeros((STEPS, 3))
        optimisation_h_history = np.zeros((STEPS, 1))
        optimisation_hd_history = np.zeros((STEPS, ndim+1))
        tracking_err_history = np.zeros((STEPS, 4))

        x_opt_curr = np.array(xa_init)
        qa_curr = qa_init

        x_opt_history[0, :3] = xa_init
        x_opt_history[0, 3:] = qa_init

        # Create optimiser
        vel_cont = VelocityController(UB, LB, ndim, 1, W=W)
        cnt = 0
        x_star = [0.1, 0.1, 0.1, 0, 0, 0]

        # Control loop
        for idx in range(1, STEPS):
            acnt = time.time()

            # Change parameters of ellipse and polygon
            rs = np.array([*x_opt_curr, *xb_init])
            qs = np.array([*qa_curr, *qb_init])

            # Optimisation approach
            x_error = x_traj[idx].t - x_opt_curr
            theta, a_hat = calc_quat_error(UnitQuaternion(qa_curr[0], qa_curr[1:]), UnitQuaternion(x_traj[idx].R))
            vel = np.array(((Kv*x_error) / DT, (Kw*theta*a_hat) / DT)).reshape(6, 1)

            # CBF stuff
            # alpha, x_inter, G_opt = get_α_J(rs, qs)
            alpha, G_opt = get_α_J(rs, qs)
            # https://github.com/BolunDai0216/DifferentiableOptimizationCBF/issues/6#issuecomment-2708300188
            h_opt = GAMMA * (alpha - BETA) # CBF exponential gamma*hx
            # G_opt = -np.array(G_opt[-1][:7])  # CBF derivative, assuming that object 1 is the ee
            G_opt = -np.array(G_opt[:7])  # CBF derivative, assuming that object 1 is the ee
            vel_cont.set_param(vel, G_opt, h_opt, UnitQuaternion(qa_curr))
            xd_opt_des = vel_cont.get_solution()


            # Integrate to obtain forward pose
            next_x_opt = x_opt_curr + xd_opt_des[:3]*DT

            ha = xd_opt_des[3:] * (DT * 0.5)  # half angle
            dR = UnitQuaternion(s=1.0, v=[ha[0], ha[1], ha[2]])
            next_qa = dR * UnitQuaternion(qa_curr)

            # Update current state for next iteration
            x_opt_curr = next_x_opt
            qa_curr = next_qa.vec

            # Obtain closest points
            # pa = x_opt_curr + (x_inter - x_opt_curr)/alpha
            # pb = np.array(xb_init) + (x_inter - np.array(xb_init))/alpha
            # Save states
            x_opt_history[idx, :3] = x_opt_curr
            x_opt_history[idx, 3:] = qa_curr
            xd_opt_history[idx, :] = xd_opt_des
            # sqa_closest_history[idx, :] = pa
            # sqb_closest_history[idx, :] = pb
            optimisation_h_history[idx, :] = h_opt/GAMMA
            optimisation_hd_history[idx, :] = G_opt
            tracking_err_history[idx, :3] = x_error
            tracking_err_history[idx, 3] = theta
            cnt += time.time() - acnt

        print(f"{1000*(cnt/STEPS)} ms/iter")

    if SAVE:
        np.save(sd + 'x_opt_history.npy', x_opt_history)
        np.save(sd + 'xd_opt_history.npy', xd_opt_history)
        np.save(sd + 'sqa_closest_history.npy', sqa_closest_history)
        np.save(sd + 'sqb_closest_history.npy', sqb_closest_history)
        np.save(sd + 'optimisation_h_history.npy', optimisation_h_history)
        np.save(sd + 'optimisation_hd_history.npy', optimisation_hd_history)
        sys.exit()

    s1 = SuperquadricObject(*Ra, *eps_a, pos=xa_init, quat=qa_init)
    s2 = SuperquadricObject(*Rb, *eps_b, pos=xb_init, quat=qb_init)

    plt.figure()
    ax = plt.subplot(111, projection='3d')
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_xlim(-1.0, 0.2)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(0.0, 0.55)
    ax.set_aspect('equal')
    s2.plot_sq(ax, 'red')
    ax.plot(x_opt_history[:, 0], x_opt_history[:, 1], x_opt_history[:, 2], color='blue')
    ax.plot(x_traj.t[:,0], x_traj.t[:,1], x_traj.t[:,2], color='g')
    ax.scatter(x_opt_history[::200, 0], x_opt_history[::200, 1], x_opt_history[::200, 2], color='black', marker='x', linewidths=1.5)

    ax.view_init(10, -60 ,0)
    for idx in range(SIM_START, SIM_END, int(TIME*TIME_SCALE)):
        s1_handle = s1.plot_sq(ax, 'green')
        traj_handle = ax.scatter(x_traj[idx].t[0], x_traj[idx].t[1], x_traj[idx].t[2], color='g', marker='o', linewidth=2)
        curr_pos_handle = ax.scatter(x_opt_history[idx, 0], x_opt_history[idx, 1], x_opt_history[idx, 2], color='blue', marker='o', alpha=0.5)
        vel_handle = ax.quiver(x_opt_history[idx, 0], x_opt_history[idx, 1], x_opt_history[idx, 2],
                               xd_opt_history[idx, 0], xd_opt_history[idx, 1], xd_opt_history[idx, 2])
        line_handle = ax.plot((sqa_closest_history[idx, 0], sqb_closest_history[idx, 0]),
                              (sqa_closest_history[idx, 1], sqb_closest_history[idx, 1]),
                              (sqa_closest_history[idx, 2], sqb_closest_history[idx, 2]), 'ro-')
        steps_handle = ax.text2D(0.05, 0.95, f"Step #{idx}", transform=ax.transAxes)

        if plt.isinteractive():
            while not plt.waitforbuttonpress():
                plt.pause(1e-6)
        else:
            plt.pause(1e-16)

        if idx < SIM_END-1:
            s1_handle.remove()
            traj_handle.remove()
            curr_pos_handle.remove()
            vel_handle.remove()
            line_handle[0].remove()
            steps_handle.remove()
        s1.set_pose(pos=x_opt_history[idx, :3], quat=tuple(x_opt_history[idx, 3:]))

    # Plots
    # CBF
    cbf_fig, cbf_ax = plt.subplots()
    cbf_fig.suptitle('h value')
    cbf_ax.plot(range(optimisation_h_history.shape[0]-1), np.round(optimisation_h_history[1:, 0], 3), label="CBF value", color='r', lw=2)
    cbf_ax.legend()

    # CBF derivative
    cbfd_fig, cbfd_ax = plt.subplots(2)
    cbfd_fig.suptitle('hd values')
    cbfd_ax[0].plot(range(optimisation_hd_history.shape[0]-1), np.round(optimisation_hd_history[1:, 0], 3), label="optimisation x grad", lw=2)
    cbfd_ax[0].plot(range(optimisation_hd_history.shape[0]-1), np.round(optimisation_hd_history[1:, 1], 3), label="optimisation y grad", lw=2)
    cbfd_ax[0].plot(range(optimisation_hd_history.shape[0]-1), np.round(optimisation_hd_history[1:, 2], 3), label="optimisation z grad", lw=2)
    cbfd_ax[1].plot(range(optimisation_hd_history.shape[0]-1), np.round(optimisation_hd_history[1:, 3], 3), label="optimisation qw grad", lw=2)
    cbfd_ax[1].plot(range(optimisation_hd_history.shape[0]-1), np.round(optimisation_hd_history[1:, 4], 3), label="optimisation qx grad", lw=2)
    cbfd_ax[1].plot(range(optimisation_hd_history.shape[0]-1), np.round(optimisation_hd_history[1:, 5], 3), label="optimisation qy grad", lw=2)
    cbfd_ax[1].plot(range(optimisation_hd_history.shape[0]-1), np.round(optimisation_hd_history[1:, 6], 3), label="optimisation qz grad", lw=2)
    cbfd_ax[0].legend(); cbfd_ax[1].legend()

    # Velocities
    vel_fig, vel_ax = plt.subplots(2)
    vel_fig.suptitle('Velocities')
    vel_ax[0].plot(range(xd_opt_history.shape[0] - 1), np.round(xd_opt_history[1:, 0], 3), label="vx", lw=2)
    vel_ax[0].plot(range(xd_opt_history.shape[0] - 1), np.round(xd_opt_history[1:, 1], 3), label="vy", lw=2)
    vel_ax[0].plot(range(xd_opt_history.shape[0] - 1), np.round(xd_opt_history[1:, 2], 3), label="vz", lw=2)
    vel_ax[1].plot(range(xd_opt_history.shape[0] - 1), np.round(xd_opt_history[1:, 3], 3), label="wx", lw=2)
    vel_ax[1].plot(range(xd_opt_history.shape[0] - 1), np.round(xd_opt_history[1:, 4], 3), label="wy", lw=2)
    vel_ax[1].plot(range(xd_opt_history.shape[0] - 1), np.round(xd_opt_history[1:, 5], 3), label="wz", lw=2)
    vel_ax[0].legend(); vel_ax[1].legend()

    # Tracking error
    tracking_err_fig, tracking_err_ax = plt.subplots(2)
    tracking_err_fig.suptitle('Positional tracking error')
    tracking_err_ax[0].plot(range(tracking_err_history.shape[0]-1), np.round(tracking_err_history[1:, 0], 3), label="x", lw=2)
    tracking_err_ax[0].plot(range(tracking_err_history.shape[0]-1), np.round(tracking_err_history[1:, 1], 3), label="y", lw=2)
    tracking_err_ax[0].plot(range(tracking_err_history.shape[0]-1), np.round(tracking_err_history[1:, 2], 3), label="z", lw=2)
    tracking_err_ax[1].plot(range(tracking_err_history.shape[0]-1), np.round(tracking_err_history[1:, 3], 3), label="theta", color='r', lw=2)
    rms = mean_squared_error(x_traj.t, x_opt_history[:, :3], squared=False)
    tracking_err_ax[0].text(0.01, 0.0, f"RMSE: {np.round(rms*1000, 3)}mm")
    rms = np.linalg.norm(tracking_err_history[1:, 3])
    tracking_err_ax[1].text(0.01, 0.0, f"RMSE: {np.round(rms, 3)}rad")
    tracking_err_ax[0].legend(); tracking_err_ax[1].legend()
    tracking_err_ax[0].legend(); tracking_err_ax[1].legend()

    plt.show(block=True)
