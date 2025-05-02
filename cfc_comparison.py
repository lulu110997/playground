import math
import sys
import time
import os
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm
from utils import *
import numpy as np
from casadi_min_dist import MinDist3D
import pandas as pd
from superquadric import SuperquadricObject

SOLVER = "ipopt"

if SOLVER == "ipopt":
    GUESS_SCALE = 0.7
    # "hsllib":"/usr/local/lib/libcoinhsl.so",
    solver_options = {"linear_solver":"ma27", "sb":"yes",
                     "print_level":0 ,"timing_statistics":"yes" , "tol":1e-6
                     , "bound_relax_factor": 1e-9
                     , "mu_init": 0.05
                     , "bound_push": 1e-4
                     , "bound_frac": 0.2
                     , "slack_bound_push": 1e-4
                     , "slack_bound_frac": 0.2
                     }
elif SOLVER == "snopt":
    os.environ["SNOPT_LICENSE"] = "/home/louis/licenses/snopt7.lic"
    GUESS_SCALE = 0.97
    solver_options = {
                      'Summary file': 0,  # Suppress summary file
                      'Major print level': 0,  # Minimal output
                      'Minor print level': 0,  # Minimal output
                      'Solution': 'No',  # Don't print solution
                      'System information': 'No',  # Don't print system info
                      'Print frequency': 0 , # Disable iteration output
                      'Verify level': 0  # Disable verification output
                     }
elif SOLVER == "knitro":
    os.environ["ARTELYS_LICENSE"] = "/home/louis/licenses/"
    GUESS_SCALE = 0.7
    # 1	Interior/Barrier (default), 2 Active Set SQP, 3	Interior/Barrier + Active Set, 4 Interior/Barrier + Direct Step
    # 5	Feasibility Restoration Phase
    solver_options = {
        "outlev": 0,
        "algorithm": 0,
        # 'xtol': 1e-6,       # Feasibility tolerance (primal feasibility)
        # 'ftol': 1e-6,       # Relative function (objective) tolerance
        # 'opttol': 1e-6,     # Optimality tolerance (KKT conditions)
        # 'feastol': 1e-6,    # Constraint violation tolerance
    }
else:
    raise "Solver type is not supported"

TEST_TYPE = 'compare with tracy'
TEST_DIR = 'test1'
TEST_NAME = 'test1'
REPLAY = 0; cwd = f'test cases/{TEST_TYPE}/{TEST_DIR}/{TEST_NAME}_'
SAVE = 1; sd = f'test cases/{TEST_TYPE}/{TEST_DIR}/{TEST_NAME}_'

WD = 'compare with tracy'
SAVE = 1; sd = f'test cases/{WD}/'

with open(f"test cases/{WD}/test.yaml") as file:
    try:
        params = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

ndim = 6  # for 3D transl and rotation
SQ_dir = "SQ-SQ-2022-08-03" # SQ-SQ
# Load params for sq1
sq1 = pd.read_csv(f'/home/louis/Git/playground/test cases/compare with CFC/{SQ_dir}/bench_config_SQ_s1.csv')
sq1_r = sq1.loc[:, ["Semi_a", "Semi_b", "Semi_c"]].to_numpy()
sq1_e = sq1.loc[:, ["Epsilon_1", "Epsilon_2"]].to_numpy()
sq1_x = sq1.loc[:, ["Pos_x","Pos_y","Pos_z"]].to_numpy()
sq1_q = sq1.loc[:, ["Quat_1","Quat_2","Quat_3","Quat_4"]].to_numpy()

# Load params for sq2
sq2 = pd.read_csv(f'/home/louis/Git/playground/test cases/compare with CFC/{SQ_dir}/bench_config_SQ_s2.csv')
sq2_r = sq2.loc[:, ["Semi_a", "Semi_b", "Semi_c"]].to_numpy()
sq2_e = sq2.loc[:, ["Epsilon_1", "Epsilon_2"]].to_numpy()
sq2_x = sq2.loc[:, ["Pos_x","Pos_y","Pos_z"]].to_numpy()
sq2_q = sq2.loc[:, ["Quat_1","Quat_2","Quat_3","Quat_4"]].to_numpy()

# Load results from FCL
fcl_results = pd.read_csv(f"/home/louis/Git/playground/test cases/compare with CFC/{SQ_dir}/bench_result_SQ_SQ_FCL.csv")
cfc_ls_results = pd.read_csv(f"/home/louis/Git/playground/test cases/compare with CFC/{SQ_dir}/bench_result_SQ_SQ_CFCLeastSquares.csv")
cfc_lscn_results = pd.read_csv(f"/home/louis/Git/playground/test cases/compare with CFC/{SQ_dir}/bench_result_SQ_SQ_CFCLeastSquaresCommonNormal.csv")
cfc_fp_results = pd.read_csv(f"/home/louis/Git/playground/test cases/compare with CFC/{SQ_dir}/bench_result_SQ_SQ_CFCFixedPoint.csv")
is_collision = fcl_results["is_collision"].to_numpy()
cant_solve = [410]

########################################################################################################################
if __name__ == '__main__':
    could_not_solve = []
    max_ratios = []
    e1a_max = []
    e1b_max = []
    our_error = []
    cfc_ls_error = []
    cfc_lscn_error = []
    cfc_fp_error = []
    t_wall = []
    iter_count = []
    in_collision = 0
    for idx in tqdm(range(sq1.shape[0])):
        if is_collision[idx] == 1:
            in_collision +=1
            continue

        s1 = SuperquadricObject(*sq1_r[idx], *sq1_e[idx], pos=sq1_x[idx], quat=list(sq1_q[idx]))
        s2 = SuperquadricObject(*sq2_r[idx], *sq2_e[idx], pos=sq2_x[idx], quat=list(sq2_q[idx]))
        x_guess = (s1.get_poi(*(s2.get_pose()[0]), scale=GUESS_SCALE).tolist() +
                   s2.get_poi(*(s1.get_pose()[0]), scale=GUESS_SCALE).tolist())
        calc = MinDist3D(ca=list(sq1_x[idx]), cb=list(sq2_x[idx]), ra=list(sq1_r[idx]), rb=list(sq2_r[idx]),
                         eps_a=list(sq1_e[idx]), eps_b=list(sq2_e[idx]), qa=list(sq1_q[idx]), qb=list(sq2_q[idx]),
                         x_bounds=(-40,40), solver=SOLVER, solver_options=solver_options)
        with stdout_redirected():
            x_star, __ = calc.get_primal_dual_solutions(x_guess)
        stats = calc.get_solver_stats()
        t_wall.append(stats["t_wall_total"]*1000)
        our_d = calc.get_optimal_value()

        if not calc.get_solver_stats()['success']:
            could_not_solve.append(idx)
            # input()
            # s1_sol = fcl_results.loc[idx, ['closest_point_s1_x', 'closest_point_s1_y', 'closest_point_s1_z']].to_numpy()
            # s2_sol = fcl_results.loc[idx, ['closest_point_s2_x', 'closest_point_s2_y', 'closest_point_s2_z']].to_numpy()
            # ax = plt.subplot(111, projection='3d')
            # x_val = s1.get_pose()[0][0], s2.get_pose()[0][0]
            # y_val = s1.get_pose()[0][1], s2.get_pose()[0][1]
            # z_val = s1.get_pose()[0][2], s2.get_pose()[0][2]
            # ax.plot(x_val, y_val, z_val, marker='o')
            # s1.plot_sq(ax, 'green')
            # s2.plot_sq(ax, 'blue')
            # ax.scatter(*x_guess[:3], color='green')
            # ax.scatter(*x_guess[3:], color='blue')
            # ax.scatter(*x_star[:3], color='red')
            # ax.scatter(*x_star[3:], color='red')
            # ax.scatter(*s1_sol, color='black')
            # ax.scatter(*s2_sol, color='black')
            # ax.set_aspect('equal')
            # plt.show()
            # continue

        base_d = fcl_results['distance'][idx]
        cfc_ls_d = cfc_ls_results['distance'][idx]
        cfc_lscn_d = cfc_lscn_results['distance'][idx]
        cfc_fp_d = cfc_fp_results['distance'][idx]
        our_error.append(100*math.fabs(base_d-our_d)/base_d)
        cfc_ls_error.append(100*math.fabs(base_d-cfc_ls_d)/base_d)
        cfc_lscn_error.append(100*math.fabs(base_d-cfc_lscn_d)/base_d)
        cfc_fp_error.append(100*math.fabs(base_d-cfc_fp_d)/base_d)

        if cfc_fp_results["is_collision"][idx]:
            print(idx)

    print("# of shapes not in collision: ", sq1.shape[0] - in_collision)
    print("# of pairs that could not be solved: ", len(could_not_solve))
    print(could_not_solve)

    print("#### ERROR COMPARISONS (%) ####")
    print("our error: ", np.round(np.mean(our_error), 2), "%")
    print("cfc_ls_error error: ", np.round(np.mean(cfc_ls_error), 2), "%")
    print("cfc_lscn_error error: ", np.round(np.mean(cfc_lscn_error), 2), "%")
    print("cfc_fp_error error: ", np.round(np.mean(cfc_fp_error), 2), "%")

    print("#### STDDEV COMPARISONS ####")
    print("our error's std: ", np.round(np.std(our_error), 2))
    print("cfc_ls error's std: ", np.round(np.std(cfc_ls_error), 2))
    print("cfc_lscn error's std: ", np.round(np.std(cfc_lscn_error), 2))
    print("cfc_fp error's std: ", np.round(np.std(cfc_fp_error), 2))

    print("#### TIME COMPARISONS (ms) ####")
    print("our average query time: ", np.round(np.mean(t_wall), 2))
    print("cfc_ls average query time: ", np.round(np.mean(cfc_ls_results["time_query"].to_numpy()), 2))
    print("cfc_lscn average query time: ", np.round(np.mean(cfc_lscn_results["time_query"].to_numpy()), 2))
    print("cfc_fp average query time: ", np.round(np.mean(cfc_fp_results["time_query"].to_numpy()), 2))


    fig = plt.figure(figsize =(10, 7))
    plt.scatter(range(len(our_error)), our_error, color='black', marker='o', alpha=0.5, label="our error")
    plt.scatter(range(len(our_error)), cfc_ls_error, color='red', marker='x', alpha=0.5, label="cfc_ls_error")
    plt.scatter(range(len(our_error)), cfc_lscn_error, color='green', marker='v', alpha=0.5, label="cfc_lscn_error")
    plt.scatter(range(len(our_error)), cfc_fp_error, color='blue', marker='.', alpha=0.5, label="cfc_fp_error")
    plt.legend()
    plt.grid()
    plt.savefig("cfc_comparison.svg")
    # plt.show()
