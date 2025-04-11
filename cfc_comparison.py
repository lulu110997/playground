import sys
import time
import yaml
from utils import *
import numpy as np
from casadi_min_dist import MinDist3D
import pandas as pd
from superquadric import SuperquadricObject

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

ndim = 6  # 2 for 2D, 3 for 3D transl, 6 for 3D transl and rotation

# Load params for sq1
sq1 = pd.read_csv('/home/louis/Git/playground/test cases/compare with CFC/SQ-SQ/bench_config_SQ_s1.csv')
sq1_r = sq1.loc[:, ["Semi_a", "Semi_b", "Semi_c"]].to_numpy()
sq1_e = sq1.loc[:, ["Epsilon_1", "Epsilon_2"]].to_numpy()
sq1_x = sq1.loc[:, ["Pos_x","Pos_y","Pos_z"]].to_numpy()
sq1_q = sq1.loc[:, ["Quat_1","Quat_2","Quat_3","Quat_4"]].to_numpy()

# Load params for sq2
sq2 = pd.read_csv('/home/louis/Git/playground/test cases/compare with CFC/SQ-SQ/bench_config_SQ_s2.csv')
sq2_r = sq1.loc[:, ["Semi_a", "Semi_b", "Semi_c"]].to_numpy()
sq2_e = sq1.loc[:, ["Epsilon_1", "Epsilon_2"]].to_numpy()
sq2_x = sq1.loc[:, ["Pos_x","Pos_y","Pos_z"]].to_numpy()
sq2_q = sq1.loc[:, ["Quat_1","Quat_2","Quat_3","Quat_4"]].to_numpy()

# Load results from FCL
fcl_results = pd.read_csv("/home/louis/Git/playground/test cases/compare with CFC/SQ-SQ/bench_result_SQ_SQ_FCL.csv")
is_collision = fcl_results["is_collision"].to_numpy()
########################################################################################################################

if __name__ == '__main__':
        x_star = []
        lambda_star = [0, 0]
        num_collisions = 0
        for idx in range(sq1.shape[0]):
            if is_collision[idx] == 1:
                num_collisions += 1
                continue
            calc = MinDist3D(ca=list(sq1_x[idx]), cb=list(sq2_x[idx]), ra=sq1_r[idx], rb=sq2_r[idx], eps_a=sq1_e[idx],
                             eps_b=sq2_e[idx], qa=list(sq1_q[idx]), qb=list(sq2_q[idx]))
            calc.get_primal_dual_solutions(list(sq1_x[idx]) + list(sq1_x[idx]), [0.01, 0.1])
            print(list(sq2_x[idx]))
            print(calc.get_optimal_value())
            print(calc.get_solver_stats())
            sys.exit()
        #     x_star.extend([xa_init + xb_init.tolist()])
        #
        # vel_cont = VelocityController(UB, LB, ndim, len(obstacles), W=W)
        # cnt = 0
        # xd_prev = np.zeros((ndim,))
