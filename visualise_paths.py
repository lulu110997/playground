import matplotlib
import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib import animation
from roboticstoolbox.tools import trajectory
from spatialmath import SE3, SO3, UnitQuaternion
from superquadric import SuperquadricObject

RUN = 339

YAML_PATH = '/home/louis/Git/playground/test cases/compare with tracy/test.yaml'
JULIA_FILES = f'/home/louis/Git/playground/test cases/compare with tracy/RUN339_JULIA{0}/'
PYTHON_FILES = f'/home/louis/Git/playground/test cases/compare with tracy/RUN339_JULIA{1}/'

with open(YAML_PATH) as file:
    try:
        params = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

ndim = 6  # 2 for 2D, 3 for 3D transl, 6 for 3D transl and rotation

# CBF-QP parameters
GAMMA = params['GAMMA']  # Used for CBF
BETA = params['BETA']  # Used for DCOL

# Superquadric parameters
Ra = params['Ra']
Rb = params['Rb']
eps_a = params['eps_a']
eps_b = params['eps_b']

# Initial and target shape poses
xa_init = params['xa_init']  # Initial robot position
qa_init = params['qa_init']  # Initial robot orientation
xa_tgt = params['xa_tgt']  # Final robot position`
qa_tgt = params['qa_tgt']  # Final robot orientation
xb_init = np.loadtxt('xb_init.txt')
xb_init = xb_init[RUN, :3], xb_init[RUN, 3:]
qb_init = params['qb_init']  # Obstacle orientation

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
x_opt_history_j = np.load(JULIA_FILES + f'x_opt_history_{RUN}.npy')
x_opt_history_p = np.load(PYTHON_FILES + f'x_opt_history_{RUN}.npy')
xd_opt_history_j = np.load(JULIA_FILES + f'xd_opt_history_{RUN}.npy')
xd_opt_history_p = np.load(PYTHON_FILES + f'xd_opt_history_{RUN}.npy')

s1 = [SuperquadricObject(*Ra, *eps_a, pos=xa_init, quat=qa_init),
      SuperquadricObject(*Ra, *eps_a, pos=xa_init, quat=qa_init)]
s2 = []
for loc in xb_init:
    s2.append(SuperquadricObject(*Rb, *eps_b, pos=loc, quat=qb_init))

ax = plt.subplot(111, projection='3d')
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_xlim(-1.0, 0.2)
ax.set_ylim(-0.5, 0.5)
ax.set_zlim(0.0, 0.55)
ax.set_aspect('equal')
for s in s2:
    s.plot_sq(ax, 'red')
ax.plot(x_opt_history_j[:, 0], x_opt_history_j[:, 1], x_opt_history_j[:, 2], color='blue')
ax.plot(x_opt_history_p[:, 0], x_opt_history_p[:, 1], x_opt_history_p[:, 2], color='black')
ax.plot(x_traj.t[:, 0], x_traj.t[:, 1], x_traj.t[:, 2], color='g')

ax.view_init(60, -165, 0)

ph = [ax.scatter(x_opt_history_j[0, 0], x_opt_history_j[0, 1], x_opt_history_j[0, 2],
                                              color='blue', marker='o', alpha=0.5),
      ax.scatter(x_opt_history_p[0, 0], x_opt_history_p[0, 1], x_opt_history_p[0, 2],
                 color='black', marker='o', alpha=0.5)
      ]

sh = [s1[0].plot_sq(ax, 'blue'), s1[1].plot_sq(ax, 'black')]
vh = (ax.quiver(x_opt_history_j[0, 0], x_opt_history_j[0, 1], x_opt_history_j[0, 2],
                xd_opt_history_j[0, 0], xd_opt_history_j[0, 1], xd_opt_history_j[0, 2], color='blue'),
      ax.quiver(x_opt_history_p[0, 0], x_opt_history_p[0, 1], x_opt_history_p[0, 2],
                xd_opt_history_p[0, 0], xd_opt_history_p[0, 1], xd_opt_history_p[0, 2], color="black"))
traj_handle = ax.scatter(x_traj[0].t[0], x_traj[0].t[1], x_traj[0].t[2], color='g', marker='o', linewidth=2)
steps_handle = ax.text2D(0.05, 0.95, f"Step #{0}", transform=ax.transAxes)

def simulate(idx):
    """
    Simulate stuff, can also be used to create the animation
    """
    global ph, sh, vh, traj_handle, steps_handle
    s1[0].set_pose(pos=x_opt_history_j[idx, :3], quat=tuple(x_opt_history_j[idx, 3:]))
    s1[1].set_pose(pos=x_opt_history_p[idx, :3], quat=tuple(x_opt_history_p[idx, 3:]))

    if idx < SIM_END-int(TIME*TIME_SCALE)-1:
        plt.pause(0.0001)
        sh[0].remove()
        sh[1].remove()
        ph[0].remove()
        ph[1].remove()
        vh[0].remove()
        vh[1].remove()
        traj_handle.remove()
        steps_handle.remove()

    ph = [ax.scatter(x_opt_history_j[idx, 0], x_opt_history_j[idx, 1], x_opt_history_j[idx, 2],
                     color='blue', marker='o', alpha=0.5),
          ax.scatter(x_opt_history_p[idx, 0], x_opt_history_p[idx, 1], x_opt_history_p[idx, 2],
                     color='black', marker='o', alpha=0.5)
          ]
    sh[0] = s1[0].plot_sq(ax, 'blue')
    sh[1] = s1[1].plot_sq(ax, 'black')
    traj_handle = ax.scatter(x_traj[idx].t[0], x_traj[idx].t[1], x_traj[idx].t[2], color='g', marker='o', linewidth=2)
    vh = (ax.quiver(x_opt_history_j[idx, 0], x_opt_history_j[idx, 1], x_opt_history_j[idx, 2],
                    xd_opt_history_j[idx, 0], xd_opt_history_j[idx, 1], xd_opt_history_j[idx, 2], color="blue"),
          ax.quiver(x_opt_history_p[idx, 0], x_opt_history_p[idx, 1], x_opt_history_p[idx, 2],
                    xd_opt_history_p[idx, 0], xd_opt_history_p[idx, 1], xd_opt_history_p[idx, 2], color="black"))
    steps_handle = ax.text2D(0.05, 0.95, f"Step #{idx}", transform=ax.transAxes)

    return ax,

ani = animation.FuncAnimation(fig=fig, func=simulate,
                              frames=range(SIM_START, SIM_END, int(TIME*TIME_SCALE)), repeat=False)
plt.close(); ani.save('A.mp4',fps=30, bitrate=-1)
# for i in range(0, STEPS, int(TIME*TIME_SCALE)):
#     simulate(i)

#
# s1 = SuperquadricObject(*Ra, *eps_a, pos=xa_init, quat=qa_init)
# s2 = []
# for i in obstacles:
#     s2.append(SuperquadricObject(*Rb, *eps_b, pos=i, quat=qb_init))

# plt.figure()
# ax = plt.subplot(111, projection='3d')
# ax.set_xlabel('x-axis')
# ax.set_ylabel('y-axis')
# ax.set_xlim(-1.0, 0.2)
# ax.set_ylim(-0.5, 0.5)
# ax.set_zlim(0.0, 0.55)
# ax.set_aspect('equal')
# s2[0].plot_sq(ax, 'red')
# s2[1].plot_sq(ax, 'red')
# ax.plot(x_opt_history[:, 0], x_opt_history[:, 1], x_opt_history[:, 2], color='blue')
# ax.plot(x_traj.t[:,0], x_traj.t[:,1], x_traj.t[:,2], color='g')
# ax.scatter(x_opt_history[::200, 0], x_opt_history[::200, 1], x_opt_history[::200, 2], color='black', marker='x', linewidths=1.5)
# ax.view_init(10, -60 ,0)

# for idx in range(SIM_START, SIM_END, int(TIME*TIME_SCALE)):
#     s1_handle = s1.plot_sq(ax, 'green')
#     traj_handle = ax.scatter(x_traj[idx].t[0], x_traj[idx].t[1], x_traj[idx].t[2], color='g', marker='o', linewidth=2)
#     curr_pos_handle = ax.scatter(x_opt_history[idx, 0], x_opt_history[idx, 1], x_opt_history[idx, 2], color='blue', marker='o', alpha=0.5)
#     vel_handle = ax.quiver(x_opt_history[idx, 0], x_opt_history[idx, 1], x_opt_history[idx, 2],
#                            xd_opt_history[idx, 0], xd_opt_history[idx, 1], xd_opt_history[idx, 2])
#     line_handle = ax.plot((sqa_closest_history[idx, 0], sqb_closest_history[idx, 0]),
#                           (sqa_closest_history[idx, 1], sqb_closest_history[idx, 1]),
#                           (sqa_closest_history[idx, 2], sqb_closest_history[idx, 2]), 'ro-')
#     steps_handle = ax.text2D(0.05, 0.95, f"Step #{idx}", transform=ax.transAxes)
#
#     if plt.isinteractive():
#         while not plt.waitforbuttonpress():
#             plt.pause(1e-16)
#     else:
#         plt.pause(1e-16)
#
#     if idx < SIM_END-1-int(TIME*TIME_SCALE):
#         s1_handle.remove()
#         traj_handle.remove()
#         curr_pos_handle.remove()
#         vel_handle.remove()
#         line_handle[0].remove()
#         steps_handle.remove()
#     s1.set_pose(pos=x_opt_history[idx, :3], quat=tuple(x_opt_history[idx, 3:]))
#
# plt.pause(0.1)
# ax.view_init(0, 0 ,0)
# plt.pause(0.1); input('zy')
# ax.view_init(0, 90, 0)
# plt.pause(0.1); input('zx')
# ax.view_init(90, 90, 0)
# plt.pause(0.1); input('xy')
# # Plots
# # CBF
# cbf_fig, cbf_ax = plt.subplots()
# cbf_fig.suptitle('h value')
# cbf_ax.plot(range(optimisation_h_history.shape[0]-1), np.round(optimisation_h_history[1:, 0], 3), label="CBF value", color='r', lw=2)
# cbf_ax.legend()
#
# # CBF derivative
# cbfd_fig, cbfd_ax = plt.subplots(2)
# cbfd_fig.suptitle('hd values')
# cbfd_ax[0].plot(range(optimisation_hd_history.shape[0]-1), np.round(optimisation_hd_history[1:, 0], 3), label="optimisation x grad", lw=2)
# cbfd_ax[0].plot(range(optimisation_hd_history.shape[0]-1), np.round(optimisation_hd_history[1:, 1], 3), label="optimisation y grad", lw=2)
# cbfd_ax[0].plot(range(optimisation_hd_history.shape[0]-1), np.round(optimisation_hd_history[1:, 2], 3), label="optimisation z grad", lw=2)
# cbfd_ax[1].plot(range(optimisation_hd_history.shape[0]-1), np.round(optimisation_hd_history[1:, 3], 3), label="optimisation qw grad", lw=2)
# cbfd_ax[1].plot(range(optimisation_hd_history.shape[0]-1), np.round(optimisation_hd_history[1:, 4], 3), label="optimisation qx grad", lw=2)
# cbfd_ax[1].plot(range(optimisation_hd_history.shape[0]-1), np.round(optimisation_hd_history[1:, 5], 3), label="optimisation qy grad", lw=2)
# cbfd_ax[1].plot(range(optimisation_hd_history.shape[0]-1), np.round(optimisation_hd_history[1:, 6], 3), label="optimisation qz grad", lw=2)
# cbfd_ax[0].legend(); cbfd_ax[1].legend()
#
# # Velocities
# vel_fig, vel_ax = plt.subplots(2)
# vel_fig.suptitle('Velocities')
# vel_ax[0].plot(range(xd_opt_history.shape[0] - 1), np.round(xd_opt_history[1:, 0], 3), label="vx", lw=2)
# vel_ax[0].plot(range(xd_opt_history.shape[0] - 1), np.round(xd_opt_history[1:, 1], 3), label="vy", lw=2)
# vel_ax[0].plot(range(xd_opt_history.shape[0] - 1), np.round(xd_opt_history[1:, 2], 3), label="vz", lw=2)
# vel_ax[1].plot(range(xd_opt_history.shape[0] - 1), np.round(xd_opt_history[1:, 3], 3), label="wx", lw=2)
# vel_ax[1].plot(range(xd_opt_history.shape[0] - 1), np.round(xd_opt_history[1:, 4], 3), label="wy", lw=2)
# vel_ax[1].plot(range(xd_opt_history.shape[0] - 1), np.round(xd_opt_history[1:, 5], 3), label="wz", lw=2)
# vel_ax[0].legend(); vel_ax[1].legend()
#
# # Tracking error
# tracking_err_fig, tracking_err_ax = plt.subplots(2)
# tracking_err_fig.suptitle('Positional tracking error')
# tracking_err_ax[0].plot(range(tracking_err_history.shape[0]-1), np.round(tracking_err_history[1:, 0], 3), label="x", lw=2)
# tracking_err_ax[0].plot(range(tracking_err_history.shape[0]-1), np.round(tracking_err_history[1:, 1], 3), label="y", lw=2)
# tracking_err_ax[0].plot(range(tracking_err_history.shape[0]-1), np.round(tracking_err_history[1:, 2], 3), label="z", lw=2)
# tracking_err_ax[1].plot(range(tracking_err_history.shape[0]-1), np.round(tracking_err_history[1:, 3], 3), label="theta", color='r', lw=2)
# rms = mean_squared_error(x_traj.t, x_opt_history[:, :3], squared=False)
# tracking_err_ax[0].text(0.01, 0.0, f"RMSE: {np.round(rms*1000, 3)}mm")
# rms = np.linalg.norm(tracking_err_history[1:, 3])
# tracking_err_ax[1].text(0.01, 0.0, f"RMSE: {np.round(rms, 3)}rad")
# tracking_err_ax[0].legend(); tracking_err_ax[1].legend()

# plt.show(block=True)