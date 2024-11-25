import matplotlib.pyplot as plt
import numpy as np
from roboticstoolbox.tools import trajectory
from spatialmath import SE2
import sympy as sym
from qpsolvers import solve_ls
from min_dists import MinDist2D

# CBF param
GAMMA = 4.0  # Used in CBF calculation

# P gain
K = 1.0  # P gain for position controller

# Shape param
Ra = 0.2  # Circle radius
Rb = 0.25  # Radius around the EE

# Initial and target shape positions
xa_init = (-1.0, -0.0)  # Obstacle position
xb_init = (-0.1, 0.4)  # Obstacle position
xa_tgt = (1.0, 0)  # Final position

# Velocity limits
UB = np.array([10.0, 10.0])  # Upper bound
LB = np.array([-10.0, -10.0])  # Lower bound

FREQ = 500.0
TIME = 1.0
DT = 1.0/FREQ
STEPS = int(TIME/DT)

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
    # print(hx_dot)
    # sym.preview(hx_dot)

    hx = sym.lambdify([x_ee, x_obs, R, Rw], expr=hx)
    hx_dot = sym.lambdify([x_ee, x_obs, R, Rw], expr=hx_dot)
    return hx, hx_dot



########################################################################################################################
# Create a trajectory
initial_pose = SE2(xa_init)
final_pose =SE2(xa_tgt)
x_traj = trajectory.ctraj(initial_pose, final_pose, STEPS).t

# Histories
x_an_history = np.zeros((STEPS, 2))
x_opt_history = np.zeros((STEPS, 2))
xd_an_history = np.zeros((STEPS, 2))
xd_opt_history = np.zeros((STEPS, 2))
lagrange_history = np.zeros((STEPS, 2))
analytical_h_history = np.zeros((STEPS, 1))
analytical_hd_history = np.zeros((STEPS, 2))
optimisation_h_history = np.zeros((STEPS, 1))
optimisation_hd_history = np.zeros((STEPS, 2))

hx, hx_dot = obtain_cbf()

x_an_curr = np.array(xa_init)
x_opt_curr = np.array(xa_init)

x_an_history[0, :] = xa_init
x_opt_history[0, :] = xa_init

# Create optimiser
obj = MinDist2D(cxa=xa_init[0], cya=xa_init[1], cxb=xb_init[0], cyb=xb_init[1],
                aa=Ra, ba=Ra, ab=Rb, bb=Rb)

# Control loop
for idx in range(1, STEPS):
    # Change sq parameters in optimiser
    obj.set_params(cxa=x_opt_curr[0], cya=x_opt_curr[1], cxb=xb_init[0], cyb=xb_init[1])
    xa_star, lambda_a, xb_star, lambda_b = obj.get_primal_dual_solutions()

    # Analytical (SOS) approach
    u = (K/DT)*(x_traj[idx] - x_an_curr)
    G_an = -hx_dot(x_an_curr, xb_init, Ra, Rb)  # CBF derivative
    h_an = GAMMA * hx(x_an_curr, xb_init, Ra, Rb)  # CBF exponential gamma*hx
    xd_an_des = solve_ls(np.eye(2), u, G_an, np.array([h_an]), lb=LB, ub=UB, solver="scs")  # clarabel or proxqp
    next_x_an = x_an_curr + xd_an_des*DT

    # Optimisation approach
    u = (K/DT)*(x_traj[idx] - x_opt_curr)
    G_opt = -np.array(obj.sensitivity_analysis()[:2])  # CBF derivative
    h_opt = GAMMA * obj.get_optimal_value()  # CBF exponential gamma*hx
    xd_opt_des = solve_ls(np.eye(2), u, G_opt, np.array([h_opt]), lb=LB, ub=UB, solver="scs")  # clarabel or proxqp
    next_x_opt = x_opt_curr + xd_opt_des*DT

    # Save states
    x_an_history[idx, :] = x_an_curr
    x_opt_history[idx, :] = x_opt_curr
    xd_an_history[idx, :] = xd_an_des*DT
    xd_opt_history[idx, :] = xd_opt_des*DT
    analytical_h_history[idx, :] = h_an/GAMMA
    analytical_hd_history[idx, :] = G_an.squeeze()
    optimisation_h_history[idx, :] = h_opt/GAMMA
    optimisation_hd_history[idx, :] = G_opt

    # Update current state for next iteration
    x_an_curr = next_x_an
    x_opt_curr = next_x_opt

circle_b = plt.Circle(xb_init, Rb, color='r')

plt.figure()
ax = plt.gca()
ax.add_patch(circle_b)

for idx in range(0, STEPS, 100):
    circle_an = plt.Circle(x_an_history[idx, :], Ra, color='b', alpha=0.5)
    circle_opt = plt.Circle(x_opt_history[idx, :], Ra, color='g', alpha=0.5)
    plt.axis('scaled')
    ax.add_patch(circle_an)
    ax.add_patch(circle_opt)
    plt.draw()
    plt.pause(DT)
    circle_an.remove()
    circle_opt.remove()

ax.add_patch(circle_an)
ax.add_patch(circle_opt)

# Plots
dist_fig, dist_ax = plt.subplots(2)
dist_fig.suptitle('h function')
dist_ax[0].plot(range(0, STEPS-1), np.round(analytical_h_history[1:, 0], 3), label="analytical distance", color='b', lw=2)
dist_ax[0].plot(range(0, STEPS-1), np.round(optimisation_h_history[1:, 0], 3), label="optimisation distance", color='g', lw=2)
dist_ax[1].plot(range(0, STEPS-1), np.round(analytical_hd_history[1:, 0], 3), label="analytical grad", color='b', lw=2)
dist_ax[1].plot(range(0, STEPS-1), np.round(optimisation_hd_history[1:, 0], 3), label="optimisation grad", color='g', lw=2)
dist_ax[0].legend(); dist_ax[1].legend()
plt.show()