from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np
from roboticstoolbox.tools import trajectory
from spatialmath import SE3
import sympy as sym
from qpsolvers import solve_ls
from min_dists import MinDist2D, MinDist3DTransl
from superquadric import SuperquadricObject

REPLAY = True
SAVE = False

ndim = 3  # 2 for 2D, 3 for 3D transl, 6 for 3D transl and rotation

# CBF param
GAMMA = 0.5  # Used in CBF calculation

# P gain
K = 1.0  # P gain for position controller

# Shape param
Ra = (0.1, 0.2, 0.13)  # Circle radius
Rb = (0.11, 0.0375, 0.11)
eps_a = (1.0, 1.0)
eps_b = (0.1, 0.5)

# Initial and target shape positions
xa_init = (-0.454, 0.578, 0.29)  # Obstacle position
xb_init = (-0.545, -0.080, 0.15)  # Obstacle position
xa_tgt = (-0.454, -0.542, 0.29)  # Final position

# Velocity limits
UB = np.array([5.0]*ndim)  # Upper bound
LB = np.array([-5.0]*ndim)  # Lower bound

FREQ = 500.0
TIME = 10.0
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
initial_pose = SE3(xa_init)
final_pose =SE3(xa_tgt)
x_traj = trajectory.ctraj(initial_pose, final_pose, STEPS).t

# Histories
x_an_history = np.zeros((STEPS, ndim))
x_opt_history = np.zeros((STEPS, ndim))
xd_an_history = np.zeros((STEPS, ndim))
xd_opt_history = np.zeros((STEPS, ndim))
lagrange_history = np.zeros((STEPS, 2))
sqa_closest_history = np.zeros((STEPS, ndim))
sqb_closest_history = np.zeros((STEPS, ndim))
optimisation_h_history = np.zeros((STEPS, 1))
optimisation_hd_history = np.zeros((STEPS, ndim))

if ndim == 2:
    hx, hx_dot = obtain_cbf()
else:
    hx = None
    hx_dot = None

x_an_curr = np.array(xa_init)
x_opt_curr = np.array(xa_init)

x_an_history[0, :] = xa_init
x_opt_history[0, :] = xa_init

if REPLAY:
    x_opt_history = np.load('with_ori_3Drot.npy')  # load
    sqa_closest_history = np.load('sqa_closest_history.npy')
    sqb_closest_history = np.load('sqb_closest_history.npy')
else:
    # Create optimiser
    obj = MinDist3DTransl(ca=xa_init, cb=xb_init, ra=Ra, rb=Rb, eps_a=eps_a, eps_b=eps_b, objective="NORM")

    # Control loop
    for idx in range(1, STEPS):
        # Change sq parameters in optimiser
        obj.set_params(ca=x_opt_curr, cb=xb_init)
        xa_star, lambda_a, xb_star, lambda_b = obj.get_primal_dual_solutions(False)

        # Optimisation approach
        u = (K/DT)*(x_traj[idx] - x_opt_curr)
        G_opt = -np.array(obj.sensitivity_analysis())  # CBF derivative
        h_opt = GAMMA * obj.get_optimal_value()  # CBF exponential gamma*hx
        xd_opt_des = solve_ls(np.eye(3), u, G_opt, np.array([h_opt]), lb=LB, ub=UB, solver="clarabel")  # clarabel or proxqp
        next_x_opt = x_opt_curr + xd_opt_des*DT

        # Save states
        x_opt_history[idx, :] = x_opt_curr
        xd_opt_history[idx, :] = xd_opt_des*DT
        optimisation_h_history[idx, :] = h_opt/GAMMA
        optimisation_hd_history[idx, :] = -G_opt
        sqa_closest_history[idx, :] = xa_star
        sqb_closest_history[idx, :] = xb_star

        # Update current state for next iteration
        x_opt_curr = next_x_opt

s1 = SuperquadricObject(*Ra, *eps_a, pos=xa_init, quat=(1, 0, 0, 0))
s2 = SuperquadricObject(*Rb, *eps_b, pos=xb_init, quat=(1, 0, 0, 0))

plt.figure()
ax = plt.subplot(111, projection='3d')
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
s2.plot_sq(ax, 'red')
ax.plot(x_opt_history[:, 0], x_opt_history[:, 1], x_opt_history[:, 2])

for idx in range(0, STEPS, int(TIME*2)):
    s1_handle = s1.plot_sq(ax, 'green')
    line_handle = ax.plot((sqa_closest_history[idx, 0], sqb_closest_history[idx, 0]),
                          (sqa_closest_history[idx, 1], sqb_closest_history[idx, 1]),
                          (sqa_closest_history[idx, 2], sqb_closest_history[idx, 2]), 'ro-')
    plt.pause(1e-16)
    s1_handle.remove()
    line_handle[0].remove()
    s1.set_pose(pos=x_opt_history[idx, :], quat=(1,0,0,0))

s1_handle = s1.plot_sq(ax, 'green')
line_handle = ax.plot((sqa_closest_history[idx, 0], sqb_closest_history[idx, 0]),
                      (sqa_closest_history[idx, 1], sqb_closest_history[idx, 1]),
                      (sqa_closest_history[idx, 2], sqb_closest_history[idx, 2]), 'ro-')

if SAVE:
    np.save('with_ori_3Drot.npy', x_opt_history) # save
    np.save('sqa_closest_history.npy', sqa_closest_history) # save
    np.save('sqb_closest_history.npy', sqb_closest_history) # save

# Plots
dist_fig, dist_ax = plt.subplots(2)
dist_fig.suptitle('h function')
dist_ax[1].plot(range(0, STEPS-1), np.round(optimisation_hd_history[1:, 0], 3), label="optimisation x grad", color='r', lw=2)
dist_ax[1].plot(range(0, STEPS-1), np.round(optimisation_hd_history[1:, 1], 3), label="optimisation y grad", color='g', lw=2)
dist_ax[1].plot(range(0, STEPS-1), np.round(optimisation_hd_history[1:, 2], 3), label="optimisation z grad", color='b', lw=2)
dist_ax[0].legend(); dist_ax[1].legend()
plt.show()
