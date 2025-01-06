import matplotlib.pyplot as plt
import numpy as np
from roboticstoolbox.tools import trajectory
from spatialmath import SE2
import sympy as sym
from qpsolvers import solve_ls
from min_dists import MinDist2D

# CBF param
GAMMA = 2.0  # Used in CBF calculation

# P gain
K = 1.0  # P gain for position controller

# Shape param
Ra = (0.2, 0.2)  # Circle radius
Rb = (0.3, 0.3)  # Radius around the EE

# Initial and target shape positions
xa_init = (-1.0, -0.0)  # Obstacle position
xb_init = (-0.001, 0.35)  # Obstacle position
xa_tgt = (1.0, 0)  # Are you still okay to meetFinal position

# Velocity limits
UB = np.array([2.0, 2.0])  # Upper bound
LB = np.array([-2.0, -2.0])  # Lower bound

FREQ = 500.0
TIME = 5.0
DT = 1.0/FREQ
STEPS = int(TIME/DT)
SKIP_SCALE = 1.2


def obtain_cbf():
    x_ee = sym.symbols('x_ee:2')
    x_obs = sym.symbols('x_obs:2')
    x_ee = sym.Matrix([[x_ee[0], x_ee[1]]])
    x_obs = sym.Matrix([[x_obs[0], x_obs[1]]])
    Rw = sym.Symbol('Rw')
    R = sym.Symbol('R')

    # This provides a much more complicated hx_dot
    hx = sym.sqrt((x_ee[0] - x_obs[0])**2 + (x_ee[1] - x_obs[1])**2) - (R + Rw)
    hx_dot = sym.diff(hx, x_ee)

    hx = sym.lambdify([x_ee, x_obs, R, Rw], expr=hx)
    hx_dot = sym.lambdify([x_ee, x_obs, R, Rw], expr=hx_dot)
    return hx, hx_dot


########################################################################################################################
# Create a trajectory
initial_pose = SE2(xa_init)
final_pose = SE2(xa_tgt)
x_traj = trajectory.ctraj(initial_pose, final_pose, STEPS).t

# Histories
optimisation_h_history = np.zeros((STEPS, 1))
optimisation_hd_history = np.zeros((STEPS, 2))
analytical_h_history = np.zeros((STEPS, 1))
analytical_hd_history = np.zeros((STEPS, 2))

# TODO: Add histories for motion

analytical_h, analytical_hd = obtain_cbf()

x_opt_curr = np.array(xa_init)

# Create optimiser
obj = MinDist2D(ca=xa_init, cb=xb_init, ra=Ra, rb=Rb, eps_a=(1, 1), eps_b=(1, 1), objective='NORM')

# Control loop using optimisation to solve for distances
for idx in range(1, STEPS):
    # Change sq parameters in optimiser
    obj.set_params(ca=x_opt_curr, cb=xb_init)
    xa_star, lambda_a, xb_star, lambda_b = obj.get_primal_dual_solutions(False)

    # Optimisation approach
    u = (K/DT)*(x_traj[idx] - x_opt_curr)
    G_opt = -np.array(obj.sensitivity_analysis()[:2])  # CBF derivative
    h_opt = GAMMA * obj.get_optimal_value()  # CBF exponential gamma*hx
    xd_opt_des = solve_ls(np.eye(2), u, G_opt, np.array([h_opt]), lb=LB, ub=UB, solver="scs")  # clarabel or proxqp
    next_x_opt = x_opt_curr + xd_opt_des*DT

    # Save states
    optimisation_h_history[idx, :] = h_opt/GAMMA
    optimisation_hd_history[idx, :] = G_opt

    # Update current state for next iteration
    x_opt_curr = next_x_opt

x_opt_curr = np.array(xa_init)  # Reset starting pos
# Control loop using analytical expressions to solve for distances
for idx in range(1, STEPS):
    # Change sq parameters in optimiser
    hx = analytical_h(x_opt_curr, xb_init, Ra[0], Rb[0])
    hx_dot = analytical_hd(x_opt_curr, xb_init, Ra[0], Rb[0])

    # Optimisation approach
    u = (K/DT)*(x_traj[idx] - x_opt_curr)
    G_opt = -np.array(hx_dot)  # CBF derivative
    h_opt = GAMMA * hx  # CBF exponential gamma*hx
    xd_opt_des = solve_ls(np.eye(2), u, G_opt, np.array([h_opt]), lb=LB, ub=UB, solver="scs")  # clarabel or proxqp
    next_x_opt = x_opt_curr + xd_opt_des*DT

    # Save states
    analytical_h_history[idx, :] = h_opt/GAMMA
    analytical_hd_history[idx, :] = G_opt

    # Update current state for next iteration
    x_opt_curr = next_x_opt

circle_b = plt.Circle(xb_init, Rb[0], color='r')

# Plot motion
# plt.figure()
# ax = plt.gca()
# ax.add_patch(circle_b)
# plt.ion()
# ax.plot(x_opt_history[:, 0], x_opt_history[:, 1])
# ax.plot(x_star_a[1:, 0], x_star_a[1:, 1])
#
# for idx in range(0, STEPS, int(TIME*SKIP_SCALE)):
#     circle_opt = plt.Circle(x_opt_history[idx, :], Ra[0], color='g', alpha=0.5)
#     plt.axis('scaled')
#     ax.add_patch(circle_opt)
#     plt.draw()
#     plt.pause(DT)
#     circle_opt.remove()
#
# ax.add_patch(circle_opt)

# Plots
plt.rc('legend', fontsize=6)

dist_fig, dist_ax = plt.subplots(figsize=(3.5, 1.5), dpi=200)
dist_fig.suptitle('Comparison of Analytical and Optimisation Values')

dist_ax.set_xlabel('Steps')  #, fontsize=15)
dist_ax.set_ylabel('Distance')  #, fontsize=15)

# Plot vals
dist_ax.plot(range(0, STEPS-1), np.round(optimisation_h_history[1:, 0], 3), label="Optimisation distance", color='#003f5c', lw=1.7)
dist_ax.plot(range(0, STEPS-1), np.round(analytical_h_history[1:, 0], 3), label="Analytical distance", color='#ffa600', lw=2, linestyle=(0, (5, 5)))
dist_ax.plot(range(0, STEPS-1), np.round(optimisation_hd_history[1:, 0], 3), label="Optimisation gradient", color='#94b83a', lw=1.7)
dist_ax.plot(range(0, STEPS-1), np.round(analytical_hd_history[1:, 0], 3), label="Analytical gradient", color='#ef5675', lw=2, linestyle=(0, (5, 5)))
dist_ax.legend()
plt.savefig('/home/louis/Pictures/2025/sim_plots_results_combined.svg', bbox_inches='tight', transparent=True)

# plt.show(block=True)