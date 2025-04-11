import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--JULIA', action='store_true')
parser.set_defaults(JULIA=False)
args = parser.parse_args()
JULIA = args.JULIA
sys.path.append("/home/louis/Git/playground")  # Needed for when python-jl is used to run this script

import time
import yaml
import numpy as np
import os
from roboticstoolbox.tools import trajectory
from spatialmath import SE3

if JULIA is True:
    from julia import Main
else:
    from casadi_min_dist import MinDist3D
from VelocityControllers import VelocityController
from utils import *

WD = 'compare with tracy'
SAVE = 0; sd = f'test cases/{WD}/'

with open(f"test cases/{WD}/test.yaml") as file:
    try:
        params = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


NDIM = 6

# CBF-QP parameters
GAMMA = params['GAMMA']  # Used for CBF
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
    xb_locs = np.loadtxt('xb_init.txt')
    counter = 0
    for row in xb_locs:
        obstacles = [row[:3], row[3:]]  # Store obstacle positions

        # Create the trajectory
        initial_pose = SE3(xa_init) @ UnitQuaternion(s=qa_init[0], v=qa_init[1:]).SE3()
        final_pose = SE3(xa_tgt) @ UnitQuaternion(s=qa_tgt[0], v=qa_tgt[1:]).SE3()
        x_traj = trajectory.ctraj(initial_pose, final_pose, STEPS)

        # Histories
        x_opt_history = np.zeros((STEPS, NDIM + 1))
        xd_opt_history = np.zeros((STEPS, NDIM))
        lagrange_history = np.zeros((STEPS, len(obstacles)))
        optimisation_h_history = np.zeros((STEPS, len(obstacles)))
        optimisation_hd_history = np.zeros((STEPS, len(obstacles), NDIM + 1))
        tracking_err_history = np.zeros((STEPS, 2))

        x_opt_curr = np.array(xa_init)
        qa_curr = qa_init

        x_opt_history[0, :3] = xa_init
        x_opt_history[0, 3:] = qa_init

        # Create optimisers
        vel_cont = VelocityController(UB, LB, NDIM, len(obstacles), W=W)  # QP controller
        xd_prev = np.zeros((NDIM,))  # Used to store prev vel  TODO: piecewise first derivative? See Boyd Lec10

        if JULIA:
            create_shapes = Main.include("create_shapes.jl")
            get_α_J = Main.include("get_α_J.jl")
            create_shapes()
        else:
            x_star = [xa_init + xb_init.tolist() for xb_init in obstacles]
            lambda_star = [(0, 0), (0, 0)]
            calculators = []
            for xb_init in obstacles:
                calculators.append(MinDist3D(ca=list(xa_init), cb=list(xb_init),
                                             ra=Ra, rb=Rb, eps_a=eps_a, eps_b=eps_b,
                                             qa=list(qa_init), qb=list(qb_init)))
            # Obtain the optimal solutions before the control loop
            for o in range(len(obstacles)):
                calculators[o].set_params(ca=list(x_opt_curr), cb=list(obstacles[o]),
                                          qa=list(qa_curr), qb=list(qb_init))
                x_star[o], lambda_star[o] = calculators[o].get_primal_dual_solutions(x_star[o], lambda_star[o])

        # Control loop
        toc = 0  # To measure control rate
        for idx in range(1, STEPS):
            tic = time.time()

            # Obtain nominal control
            x_error = x_traj[idx].t - x_opt_curr
            theta, a_hat = calc_quat_error(UnitQuaternion(qa_curr[0], qa_curr[1:]), UnitQuaternion(x_traj[idx].R))
            vel = np.array(((Kv*x_error) / DT, (Kw*theta*a_hat) / DT)).reshape(6, 1)

            # Initialise variables to store h and hdot
            h_opt = np.zeros((len(obstacles),))
            G_opt = np.zeros((len(obstacles), NDIM+1))

            # Calculate h and hdot for each obstacle
            for o in range(len(obstacles)):
                if JULIA:
                    alpha, da_dp = get_α_J(np.array([*x_opt_curr, *obstacles[o]]), np.array([*qa_curr, *qb_init]))
                    # https://github.com/BolunDai0216/DifferentiableOptimizationCBF/issues/6#issuecomment-2708300188
                    h_opt[o] = GAMMA * (alpha - BETA)
                    G_opt[o] = -np.array(da_dp[:7])
                else:
                    calculators[o].set_params(ca=list(x_opt_curr), cb=list(obstacles[o]),
                                              qa=list(qa_curr), qb=list(qb_init))
                    x_star[o], lambda_star[o] = calculators[o].get_primal_dual_solutions(x_star[o], lambda_star[o])
                    h_opt[o] = GAMMA * calculators[o].get_optimal_value()
                    G_opt[o] = -np.array(calculators[o].sensitivity_analysis())
            vel_cont.set_param(vel, xd_prev, G_opt, h_opt, UnitQuaternion(qb_init))  # TODO: change this to be the orientation of the specific obstacle
            xd_opt_des = vel_cont.get_solution()
            xd_prev = xd_opt_des

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
            xd_opt_history[idx, :] = xd_opt_des
            optimisation_h_history[idx, :] = h_opt/GAMMA
            optimisation_hd_history[idx] = G_opt
            tracking_err_history[idx, 0] = np.linalg.norm(x_error)
            tracking_err_history[idx, 1] = theta
            toc += time.time() - tic

        # print(f"{1000*(toc/STEPS)} ms/iter")
        total_pos_err = np.mean(tracking_err_history[1:, 0])
        total_ori_err = np.mean(tracking_err_history[1:, 1])

        if np.linalg.norm(x_traj[-1].t - x_opt_history[-1][:3]) > 1e-3 or theta > 1e-3:
            finished = 0
        else:
            finished = 1

        print(counter, total_pos_err*1000, total_ori_err, (toc/STEPS)*1000, finished)

        if SAVE:
            sp = f'{sd}/RUN{counter}_JULIA{int(JULIA)}/'
            os.makedirs(sp, exist_ok=True)
            np.save(sp + f'x_opt_history_{counter}.npy', x_opt_history)
            np.save(sp + f'xd_opt_history_{counter}.npy', xd_opt_history)
            np.save(sp + f'optimisation_h_history_{counter}.npy', optimisation_h_history)
            np.save(sp + f'optimisation_hd_history_{counter}.npy', optimisation_hd_history)
            np.save(sp + f'tracking_err_history_{counter}.npy', tracking_err_history)
        counter += 1
