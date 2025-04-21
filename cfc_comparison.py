import math
import sys
import time

import matplotlib.pyplot as plt
import yaml
from utils import *
import numpy as np
from casadi_min_dist import MinDist3D
import pandas as pd
from superquadric import SuperquadricObject

ipopt_options = {"linear_solver":"ma27", "hsllib":"/usr/local/lib/libcoinhsl.so", "sb":"yes",
                         "print_level":0, "timing_statistics":"yes"}

# plt.ion()
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

# Load params for sq1
sq1 = pd.read_csv('/home/louis/Git/playground/test cases/compare with CFC/SQ-SQ/bench_config_SQ_s1.csv')
sq1_r = sq1.loc[:, ["Semi_a", "Semi_b", "Semi_c"]].to_numpy()
sq1_e = sq1.loc[:, ["Epsilon_1", "Epsilon_2"]].to_numpy()
sq1_x = sq1.loc[:, ["Pos_x","Pos_y","Pos_z"]].to_numpy()
sq1_q = sq1.loc[:, ["Quat_1","Quat_2","Quat_3","Quat_4"]].to_numpy()

# Load params for sq2
sq2 = pd.read_csv('/home/louis/Git/playground/test cases/compare with CFC/SQ-SQ/bench_config_SQ_s2.csv')
sq2_r = sq2.loc[:, ["Semi_a", "Semi_b", "Semi_c"]].to_numpy()
sq2_e = sq2.loc[:, ["Epsilon_1", "Epsilon_2"]].to_numpy()
sq2_x = sq2.loc[:, ["Pos_x","Pos_y","Pos_z"]].to_numpy()
sq2_q = sq2.loc[:, ["Quat_1","Quat_2","Quat_3","Quat_4"]].to_numpy()

# Load results from FCL
fcl_results = pd.read_csv("/home/louis/Git/playground/test cases/compare with CFC/SQ-SQ/bench_result_SQ_SQ_FCL.csv")
cfc_ls_results = pd.read_csv("/home/louis/Git/playground/test cases/compare with CFC/SQ-SQ/bench_result_SQ_SQ_CFCLeastSquares.csv")
is_collision = fcl_results["is_collision"].to_numpy()
cant_solve = [4, 22, 33, 34, 36, 71, 83, 121, 123, 142, 145, 158, 170, 193, 198, 201, 219, 220, 222, 224, 228, 237,
              253, 256, 289, 294, 328, 339, 342, 357, 362, 391, 394, 398]

########################################################################################################################
if __name__ == '__main__':
    could_not_solve = []
    max_ratios = []
    e1a_max = []
    e1b_max = []
    our_error = []
    cfc_error = []

    lambda_star = [0, 0]
    in_collision = 0
    for idx in range(sq1.shape[0]):
        if is_collision[idx] == 1:
            in_collision +=1
            continue

        s1 = SuperquadricObject(*sq1_r[idx], *sq1_e[idx], pos=sq1_x[idx], quat=list(sq1_q[idx]))
        s2 = SuperquadricObject(*sq2_r[idx], *sq2_e[idx], pos=sq2_x[idx], quat=list(sq2_q[idx]))
        x_guess = (s1.get_poi(*(s2.get_pose()[0]), scale=0.95).tolist() +
                   s2.get_poi(*(s1.get_pose()[0]), scale=0.95).tolist())
        calc = MinDist3D(ca=list(sq1_x[idx]), cb=list(sq2_x[idx]), ra=list(sq1_r[idx]), rb=list(sq2_r[idx]),
                         eps_a=list(sq1_e[idx]), eps_b=list(sq2_e[idx]), qa=list(sq1_q[idx]), qb=list(sq2_q[idx]),
                         x_bounds=(-40,40), ipopt_options=ipopt_options)
        tic = time.time()
        x_star, __ = calc.get_primal_dual_solutions(x_guess, [0.0, 0.0])
        print((time.time() - tic)*1000)
        stats=calc.get_solver_stats()
        print(stats["iter_count"])
        print(stats["t_wall_total"]*1000)
        input()
        our_d = calc.get_optimal_value()

        if not calc.get_solver_stats()['success']:
            could_not_solve.append(idx)
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

            continue
        base_d = fcl_results['distance'][idx]
        cfc_d = cfc_ls_results['distance'][idx]
        our_error.append(math.fabs(base_d-our_d)/base_d)
        cfc_error.append(math.fabs(base_d-cfc_d)/base_d)

    print("# of shapes not in collision: ", sq1.shape[0] - in_collision)
    print(len(could_not_solve))
    print(could_not_solve)
    print(np.mean(our_error)*100)
    print(np.mean(cfc_error)*100)
