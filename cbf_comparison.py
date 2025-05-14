import sys; sys.path.append("/home/louis/Git/playground")  # Needed for when python-jl is used to run this script

from roboticstoolbox.tools import trajectory
from spatialmath import SE3
from tqdm import tqdm
from VelocityControllers import VelocityController
from utils import *
import numpy as np
import os
import argparse
import time
import yaml
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--JULIA', action='store_true')
parser.set_defaults(JULIA=False)
args = parser.parse_args()
JULIA = args.JULIA

WD = 'comparative study 2'
SAVE = 1; sd = f'test cases/{WD}/'

if JULIA is True:
    from julia import Main
    csv_file = f"test cases/{WD}/julia_results.csv"
else:
    from casadi_min_dist import MinDist3D
    # Some docs for warm starting IPOPT
    # https://github.com/casadi/casadi/wiki/FAQ%3A-Warmstarting-with-IPOPT
    # https://www.gams.com/latest/docs/S_IPOPT.html#IPOPT_WARMSTART
    SOLVER = "ipopt"
    if SOLVER == "ipopt":
        solver_options = {"linear_solver":"ma27", "sb":"yes", "print_level":0,
                         "tol": 1e-6 , "warm_start_init_point":"yes" , "warm_start_bound_push": 1e-9,
                         "warm_start_mult_bound_push": 1e-9 , "mu_strategy": "monotone", "mu_init": 1e-9,
                         "bound_relax_factor": 1e-9 , "warm_start_bound_frac": 1e-9 , "warm_start_slack_bound_frac": 1e-9,
                         "warm_start_slack_bound_push": 1e-9}
    elif SOLVER == "snopt":
        os.environ["SNOPT_LICENSE"] = "/home/louis/licenses/snopt7.lic"
        solver_options = {
            'Warm start': "True",
            "Major feasibility tolerance": 1e-6,
            "Major optimality tolerance": 1e-6,
            'Summary file': 0,  # Suppress summary file
            'Major print level': 0,  # Minimal output
            'Minor print level': 0,  # Minimal output
            'Solution': 'No',  # Don't print solution
            'System information': 'No',  # Don't print system info
            'Print frequency': 0,  # Disable iteration output
            'Verify level': 0  # Disable verification output
        }
    csv_file = f"test cases/{WD}/python_results.csv"

if SAVE:
    df = pd.read_csv(csv_file)
    header_df = pd.DataFrame(columns=['idx', 'avg pos error (mm)', 'avg ori error (rad)', 'average control rate', 'finished'])
    header_df.to_csv(csv_file, index=False)

with open(f"test cases/{WD}/test.yaml") as param_file:
    try:
        params = yaml.safe_load(param_file)
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
    xb_locs = load_txt_as_generator("xb_pos.txt")
    xb_oris = load_txt_as_generator("xb_ori.txt")
    counter = 0
    n_obs = np.loadtxt("xb_pos.txt")[0].shape[0] // 3
    n_sim = np.loadtxt("xb_pos.txt").shape[0]

    # Create the trajectory
    initial_pose = SE3(xa_init) @ UnitQuaternion(s=qa_init[0], v=qa_init[1:]).SE3()
    final_pose = SE3(xa_tgt) @ UnitQuaternion(s=qa_tgt[0], v=qa_tgt[1:]).SE3()
    x_traj = trajectory.ctraj(initial_pose, final_pose, STEPS)

    for pos, ori in tqdm(zip(xb_locs, xb_oris), total=n_sim):
        # Store obstacle pose
        obs_locs = [pos[:3], pos[3:]]
        obs_oris = [ori[:4], ori[4:]]

        # Histories
        x_opt_history = np.zeros((STEPS, NDIM + 1))
        xd_opt_history = np.zeros((STEPS, NDIM))
        lagrange_history = np.zeros((STEPS, n_obs))
        optimisation_h_history = np.zeros((STEPS, n_obs))
        optimisation_hd_history = np.zeros((STEPS, n_obs, NDIM + 1))
        tracking_err_history = np.zeros((STEPS, 2))

        x_opt_curr = np.array(xa_init)
        qa_curr = qa_init

        x_opt_history[0, :3] = xa_init
        x_opt_history[0, 3:] = qa_init

        # Create optimisers
        vel_cont = VelocityController(UB, LB, NDIM, n_obs, W=W)  # QP controller
        xd_prev = np.zeros((NDIM,))  # Used to store prev vel  TODO: piecewise first derivative? See Boyd Lec10

        with stdout_redirected():
            if JULIA:
                create_shapes = Main.include("create_shapes.jl")
                get_α_J = Main.include("get_α_J.jl")
                create_shapes()
            else:
                x_star = [xa_init + xb_init.tolist() for xb_init in obs_locs]
                lambda_star = [(0, 0), (0, 0)]
                calculators = []
                for n in range(n_obs):
                    calculators.append(MinDist3D(ca=list(xa_init), cb=list(obs_locs[n]),
                                                 ra=Ra, rb=Rb, eps_a=eps_a, eps_b=eps_b,
                                                 qa=list(qa_init), qb=list(obs_oris[n]), solver=SOLVER))
                    calculators[-1].set_params(ca=list(x_opt_curr), cb=list(obs_locs[n]),
                                               qa=list(qa_curr), qb=list(obs_oris[n]))
                    x_star[n], lambda_star[n] = calculators[-1].get_primal_dual_solutions(x_star[n])

                # Reinitialise the solvers to have warm start settings
                calculators = []
                for n in range(n_obs):
                    calculators.append(MinDist3D(ca=list(xa_init), cb=list(obs_locs[n]),
                                                 ra=Ra, rb=Rb, eps_a=eps_a, eps_b=eps_b,
                                                 qa=list(qa_init), qb=list(obs_oris[n]), solver=SOLVER, solver_options=solver_options))

        # Control loop
        toc = 0  # To measure control rate
        for idx in range(1, STEPS):
            tic = time.time()

            # Obtain nominal control
            x_error = x_traj[idx].t - x_opt_curr
            theta, a_hat = calc_quat_error(UnitQuaternion(qa_curr[0], qa_curr[1:]), UnitQuaternion(x_traj[idx].R))
            vel = np.array(((Kv*x_error) / DT, (Kw*theta*a_hat) / DT)).reshape(6, 1)

            # Initialise variables to store h and hdot
            h_opt = np.zeros((n_obs,))
            G_opt = np.zeros((n_obs, NDIM+1))

            # Calculate h and hdot for each obstacle
            with stdout_redirected():
                for o in range(n_obs):
                    if JULIA:
                        alpha, da_dp = get_α_J(np.array([*x_opt_curr, *obs_locs[o]]), np.array([*qa_curr, *obs_oris[o]]))
                        # https://github.com/BolunDai0216/DifferentiableOptimizationCBF/issues/6#issuecomment-2708300188
                        h_opt[o] = GAMMA * (alpha - BETA)
                        G_opt[o] = -np.array(da_dp[:7])
                    else:
                        calculators[o].set_params(ca=list(x_opt_curr), cb=list(obs_locs[o]),
                                                  qa=list(qa_curr), qb=list(obs_oris[o]))
                        x_star[o], lambda_star[o] = calculators[o].get_primal_dual_solutions(x_star[o], lambda_star[o])
                        # print(calculators[o].get_optimal_value())
                        # print(calculators[o].get_solver_stats()["return_status"])
                        # print(calculators[o].get_solver_stats()["success"])
                        h_opt[o] = GAMMA * calculators[o].get_optimal_value()
                        G_opt[o] = -np.array(calculators[o].sensitivity_analysis())

            vel_cont.set_param(vel, xd_prev, G_opt, h_opt, UnitQuaternion(qa_curr))  # TODO: change this to be the orientation of the specific obstacle
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

        avg_pos_err = np.mean(tracking_err_history[1:, 0])
        avg_ori_err = np.mean(tracking_err_history[1:, 1])

        if np.linalg.norm(x_traj[-1].t - x_opt_history[-1][:3]) > 1e-3 or theta > 1e-3:
            finished = 0
        else:
            finished = 1
        # print((toc/STEPS)*1000)
        df_values = (counter, avg_pos_err*1000, avg_ori_err, (toc/STEPS)*1000, finished)
        new_row = pd.DataFrame([{'idx': df_values[0], 'average pos error (mm)': df_values[1],
                                 'average ori error (rad)': df_values[2], 'average control rate': df_values[3],
                                 'finished': df_values[4]}])

        if SAVE:
            new_row.to_csv(csv_file, mode='a', header=False, index=False)
            sp = f'{sd}/RUN{counter}_JULIA{int(JULIA)}/'
            os.makedirs(sp, exist_ok=True)
            np.save(sp + f'x_opt_history_{counter}.npy', x_opt_history)
            np.save(sp + f'xd_opt_history_{counter}.npy', xd_opt_history)
            np.save(sp + f'optimisation_h_history_{counter}.npy', optimisation_h_history)
            np.save(sp + f'optimisation_hd_history_{counter}.npy', optimisation_hd_history)
            np.save(sp + f'tracking_err_history_{counter}.npy', tracking_err_history)
        counter += 1

    # Compute mean and stddev of pos/ori error and control rate, compute sum of finished column, save as a new file
    if SAVE:
        df = pd.read_csv(csv_file)
        columns = ['avg pos error (mm)', 'avg ori error (rad)', 'average control rate']
        mean_row = df[columns].mean()
        std_row = df[columns].std()
        finished_sum = pd.Series({'finished': df['finished'].sum()}, name='finished_sum')
        mean_row['finished'] = df['finished'].sum()
        df_stats = pd.concat([df, pd.DataFrame([mean_row, std_row])], axis=0)
        df_stats.to_csv(csv_file, header=True, index=False)
