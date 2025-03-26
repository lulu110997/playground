from utils import *
from roboticstoolbox.tools import trajectory
from spatialmath import SE3, SO3, UnitQuaternion
from VelocityControllers import VelocityController
from casadi_min_dist import MinDist3D, MinDist3DTransl
from superquadric import SuperquadricObject

import sys
import time
import matplotlib.pyplot as plt
import numpy as np

REPLAY = 0
SAVE = 0

ndim = 3  # 2 for 2D, 3 for 3D transl, 6 for 3D transl and rotation

# CBF param
GAMMA = 0.4  # Used in CBF calculation

# P gain
Kv = 0.4  # position
Kw = 1.0  # orientation

# Shape param
Ra = (0.15, 0.15, 0.15)
Rb = (0.1, 0.1, 0.1)
eps_a = (1.0, 1.0)
eps_b = (1.0, 1.0)

# Initial and target shape positions
xb_init = (-0.545, -0.080, 0.15)  # Obstacle position
qb_init = (1.0, 0.0, 0.0, 0.0)  # Obstacle orientation
xa_init = (-0.454, 0.5, 0.29)  # Initial robot position
qa_init = (1.0, 0, 0, 0.0)  # Initial robot orientation
xa_tgt = (-0.454, -0.5, 0.29)  # Final robot position
qa_tgt = qa_init

# Velocity limits
UB = np.array([0.2]*3 + [1.0]*3)  # Upper bound
LB = np.array([-0.2]*3 + [-1.0]*3)  # Lower bound

FREQ = 100.0
TIME = 15.0
TIME_SCALE = 0.85
DT = 1.0/FREQ
STEPS = int(TIME/DT)
SIM_START = 0

########################################################################################################################
# Create a trajectory
initial_pose = SE3(xa_init) @ UnitQuaternion(s=qa_init[0], v=qa_init[1:]).SE3()
final_pose = SE3(xa_tgt) @ UnitQuaternion(s=qa_tgt[0], v=qa_tgt[1:]).SE3()

x_traj = trajectory.ctraj(initial_pose, final_pose, STEPS)

# Histories
x_opt_history = np.zeros((STEPS, ndim))
sqa_closest_history = np.zeros((STEPS, 3))
sqb_closest_history = np.zeros((STEPS, 3))
optimisation_h_history = np.zeros((STEPS, 1))
optimisation_hd_history = np.zeros((STEPS, ndim))
analytical_h_history = np.zeros((STEPS, 1))
analytical_hd_history = np.zeros((STEPS, ndim))
x_opt_curr = np.array(xa_init)

x_opt_history[0, :3] = xa_init

if REPLAY:
    # Load histories
    x_opt_history = np.load('with_ori_3Drot.npy')
    sqa_closest_history = np.load('sqa_closest_history.npy')
    sqb_closest_history = np.load('sqb_closest_history.npy')
else:
    # Create optimisers
    obj = MinDist3DTransl(ca=list(xa_init), cb=list(xb_init), ra=Ra, rb=Rb, eps_a=eps_a, eps_b=eps_b)
    vel_cont = VelocityController(ndim, 1)

    cnt = 0
    x_star = [0.1, 0.1, 0.1, 0, 0, 0]
    analytical_h, analytical_hd = distance_derivative()

    # Control loop
    for idx in range(1, STEPS):
        acnt = time.time()

        # Change sq parameters in optimiser
        obj.set_params(ca=list(x_opt_curr), cb=list(xb_init))
        x_star, lambda_star = obj.get_primal_dual_solutions(x_star)

        # Optimisation approach
        x_error = x_traj[idx].t - x_opt_curr
        vel = np.array((Kv*x_error) / DT)

        # CBF stuff
        G_opt = -np.array(obj.sensitivity_analysis())  # CBF derivative
        h_opt = GAMMA * obj.get_optimal_value()  # CBF exponential gamma*hx
        vel_cont.set_param(vel, G_opt, h_opt)
        xd_opt_des = vel_cont.get_solution()

        analytical_h_history[idx, :] = analytical_h(x_opt_curr, xb_init, Ra[0], Rb[0])
        analytical_hd_history[idx, :] = -analytical_hd(x_opt_curr, xb_init, Ra[0], Rb[0])

        cnt += time.time() - acnt

        # Integrate to obtain forward pose
        next_x_opt = x_opt_curr + xd_opt_des*DT

        # Update current state for next iteration
        x_opt_curr = next_x_opt

        # Save states
        x_opt_history[idx, :] = x_opt_curr
        sqa_closest_history[idx, :] = x_star[:3]
        sqb_closest_history[idx, :] = x_star[3::]
        optimisation_h_history[idx, :] = h_opt/GAMMA
        optimisation_hd_history[idx, :] = G_opt

    print(f"{1000*(cnt/STEPS)} ms/iter")

print("d error: ", np.linalg.norm(analytical_h_history - optimisation_h_history))
print("dd error:", np.linalg.norm(analytical_hd_history - optimisation_hd_history))

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
ax.set_aspect('equal')
s2.plot_sq(ax, 'red')
ax.plot(x_opt_history[:, 0], x_opt_history[:, 1], x_opt_history[:, 2])

for idx in range(SIM_START, STEPS, int(TIME*TIME_SCALE)):
    s1_handle = s1.plot_sq(ax, 'green')
    line_handle = ax.plot((sqa_closest_history[idx, 0], sqb_closest_history[idx, 0]),
                          (sqa_closest_history[idx, 1], sqb_closest_history[idx, 1]),
                          (sqa_closest_history[idx, 2], sqb_closest_history[idx, 2]), 'ro-')
    plt.pause(1e-16)
    line_handle[0].remove()
    s1_handle.remove()
    s1.set_pose(pos=x_opt_history[idx, :3], quat=qa_init)

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
