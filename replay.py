import matplotlib
import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib import animation
from roboticstoolbox.tools import trajectory
from spatialmath import SE3, SO3, UnitQuaternion
from superquadric import SuperquadricObject

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
GAMMA = params['GAMMA']  # Used for CBF
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

matplotlib.rcParams['savefig.dpi'] = 300.0
FIG_SIZE = (12.80, 9.60)

# Create the trajectory
initial_pose = SE3(xa_init) @ UnitQuaternion(s=qa_init[0], v=qa_init[1:]).SE3()
final_pose = SE3(xa_tgt) @ UnitQuaternion(s=qa_tgt[0], v=qa_tgt[1:]).SE3()
x_traj = trajectory.ctraj(initial_pose, final_pose, STEPS)

fig = plt.figure(figsize=FIG_SIZE)
x_opt_history = np.load(cwd + 'x_opt_history.npy')
xd_opt_history = np.load(cwd + 'xd_opt_history.npy')
sqa_closest_history = np.load(cwd + 'sqa_closest_history.npy')
sqb_closest_history = np.load(cwd + 'sqb_closest_history.npy')
optimisation_h_history = np.load(cwd + 'optimisation_h_history.npy')
optimisation_hd_history = np.load(cwd + 'optimisation_hd_history.npy')

s1 = SuperquadricObject(*Ra, *eps_a, pos=xa_init, quat=qa_init)
s2 = SuperquadricObject(*Rb, *eps_b, pos=xb_init, quat=qb_init)

ax = plt.subplot(111, projection='3d')
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_xlim(-1.0, 0.2)
ax.set_ylim(-0.5, 0.5)
ax.set_zlim(0.0, 0.55)
ax.set_aspect('equal')
s2.plot_sq(ax, 'red')
ax.plot(x_opt_history[:, 0], x_opt_history[:, 1], x_opt_history[:, 2], color='blue')
ax.plot(x_traj.t[:, 0], x_traj.t[:, 1], x_traj.t[:, 2], color='g')
ax.scatter(x_opt_history[::200, 0], x_opt_history[::200, 1], x_opt_history[::200, 2], color='black', marker='x',
           linewidths=1.5)
ax.view_init(10, -60, 0)

s1_handle = s1.plot_sq(ax, 'green')
traj_handle = ax.scatter(x_traj[0].t[0], x_traj[0].t[1], x_traj[0].t[2], color='g', marker='o', linewidth=2)
curr_pos_handle = ax.scatter(x_opt_history[0, 0], x_opt_history[0, 1], x_opt_history[0, 2], color='blue',
                             marker='o', alpha=0.5)
vel_handle = ax.quiver(x_opt_history[0, 0], x_opt_history[0, 1], x_opt_history[0, 2],
                       xd_opt_history[0, 0], xd_opt_history[0, 1], xd_opt_history[0, 2])
line_handle = ax.plot((sqa_closest_history[0, 0], sqb_closest_history[0, 0]),
                      (sqa_closest_history[0, 1], sqb_closest_history[0, 1]),
                      (sqa_closest_history[0, 2], sqb_closest_history[0, 2]), 'ro-')
steps_handle = ax.text2D(0.05, 0.95, f"Step #{0}", transform=ax.transAxes)

def simulate(idx):
    """
    Simulate stuff, can also be used to create the animation
    """
    global s1_handle, traj_handle, curr_pos_handle, vel_handle, line_handle, steps_handle
    s1.set_pose(pos=x_opt_history[idx, :3], quat=tuple(x_opt_history[idx, 3:]))

    if idx < SIM_END-1:
        plt.pause(0.001)
        s1_handle.remove()
        traj_handle.remove()
        curr_pos_handle.remove()
        vel_handle.remove()
        line_handle[0].remove()
        steps_handle.remove()

    s1_handle = s1.plot_sq(ax, 'green')
    traj_handle = ax.scatter(x_traj[idx].t[0], x_traj[idx].t[1], x_traj[idx].t[2], color='g', marker='o', linewidth=2)
    curr_pos_handle = ax.scatter(x_opt_history[idx, 0], x_opt_history[idx, 1], x_opt_history[idx, 2], color='blue', marker='o', alpha=0.5)
    vel_handle = ax.quiver(x_opt_history[idx, 0], x_opt_history[idx, 1], x_opt_history[idx, 2],
                           xd_opt_history[idx, 0], xd_opt_history[idx, 1], xd_opt_history[idx, 2])
    line_handle = ax.plot((sqa_closest_history[idx, 0], sqb_closest_history[idx, 0]),
                          (sqa_closest_history[idx, 1], sqb_closest_history[idx, 1]),
                          (sqa_closest_history[idx, 2], sqb_closest_history[idx, 2]), 'ro-')
    steps_handle = ax.text2D(0.05, 0.95, f"Step #{idx}", transform=ax.transAxes)

    # Plots
    # dist_fig, dist_ax = plt.subplots(2)
    # dist_fig.suptitle('h function')
    # dist_ax[0].plot(range(optimisation_h_history.shape[0]-1), np.round(optimisation_h_history[1:, 0], 3), label="distance", color='r', lw=2)
    #
    # dist_ax[1].plot(range(optimisation_hd_history.shape[0]-1), np.round(optimisation_hd_history[1:, 0], 3), label="optimisation x grad", color='r', lw=2)
    # dist_ax[1].plot(range(optimisation_hd_history.shape[0]-1), np.round(optimisation_hd_history[1:, 1], 3), label="optimisation y grad", color='g', lw=2)
    # dist_ax[1].plot(range(optimisation_hd_history.shape[0]-1), np.round(optimisation_hd_history[1:, 2], 3), label="optimisation z grad", color='b', lw=2)
    # dist_ax[0].legend(); dist_ax[1].legend()
    # plt.show(block=True)
    return ax,

ani = animation.FuncAnimation(fig=fig, func=simulate, frames=range(SIM_START, SIM_END, int(TIME*TIME_SCALE)), repeat=False)
plt.close()
ani.save('A.mp4',fps=30, bitrate=-1)