import time

import matplotlib.pyplot as plt
import numpy as np
from roboticstoolbox.tools import trajectory
from spatialmath import SE3, SO3, UnitQuaternion
from spatialmath.base import trotx, troty, trotz
from VelocityControllers import VelocityController
from min_dists import MinDist3DRot
from superquadric import SuperquadricObject
import math

def calc_quat_error(q_curr: UnitQuaternion, q_desired: UnitQuaternion):
    """
    Calculate quaternion error
    q_curr: UnitQuaternion | Current orientation as a unit quaternion
    q_desired: UnitQuaternion | Desired orientation as a unit quaternion
    """
    if not (isinstance(q_curr, UnitQuaternion) and isinstance(q_desired, UnitQuaternion)):
        raise "Check that variable being passed is a UnitQuaternion class"
    quat_error = q_desired * q_curr.conj()
    norm = np.linalg.norm(quat_error.vec3)

    # Angle axis calc
    if norm > 1e-4:
        axis = quat_error.vec3/norm
        angle = 2*math.acos(quat_error.vec[0])
    else:
        axis = np.array([1, 0, 0])
        angle = 0

    # Limit the angle to [pi pi[
    if angle > math.pi:
        angle = 2*math.pi - angle
        axis = -axis

    # Sanity check
    if (angle < -math.pi) and (angle > math.pi):
        print(angle)
        raise "Axis angle error outside of bounds"

    return angle*axis

REPLAY = 0
SAVE = False

ndim = 6  # 2 for 2D, 3 for 3D transl, 6 for 3D transl and rotation

# CBF param
GAMMA = 0.5  # Used in CBF calculation

# P gain
Kv = 1.0  # P gain for position controller
Kw = 1.0

# Shape param
Ra = (0.1, 0.2, 0.13)  # Circle radius
Rb = (0.11, 0.0375, 0.11)
eps_a = (1.0, 1.0)
eps_b = (0.5, 0.5)

# Initial and target shape positions
xb_init = (-0.545, -0.080, 0.15)  # Obstacle position
qb_init = (-0.2154947, -0.6235768, 0.7284832, 0.1844621)  # Obstacle orientation
xa_init = (-0.454, 1, 0.29)  # Initial robot position
qa_init = (1, 0, 0, 0)  # Initial robot orientation
xa_tgt = (-0.454, -1, 0.29)  # Final robot position
qa_tgt = (1, 0, 0, 0)  # Final robot orientation

# Velocity limits
UB = np.array([5.0]*3 + [2]*3)  # Upper bound
LB = np.array([-5.0]*3 + [-2]*3)  # Lower bound

FREQ = 500.0
TIME = 5.0
DT = 1.0/FREQ
STEPS = int(TIME/DT)

########################################################################################################################
# Create a trajectory
initial_pose = SE3(xa_init) @ SE3(SO3(UnitQuaternion(qa_init[0], qa_init[1:]).R))
final_pose = SE3(xa_tgt) @ SE3(SO3(UnitQuaternion(qa_tgt[0], qa_tgt[1:]).R))

x_traj = trajectory.ctraj(initial_pose, final_pose, STEPS)

# Histories
x_opt_history = np.zeros((STEPS, ndim+1))
xd_opt_history = np.zeros((STEPS, ndim))
lagrange_history = np.zeros((STEPS, 2))
sqa_closest_history = np.zeros((STEPS, ndim))
sqb_closest_history = np.zeros((STEPS, ndim))
optimisation_h_history = np.zeros((STEPS, 1))
optimisation_hd_history = np.zeros((STEPS, ndim))

x_opt_curr = np.array(xa_init)
qa_curr = qa_init

x_opt_history[0, :3] = xa_init
x_opt_history[0, 3:] = qa_init

if REPLAY:
    x_opt_history = np.load('x_opt_history.npy')  # load
    sqa_closest_history = np.load('sqa_closest_history.npy')
    sqb_closest_history = np.load('sqb_closest_history.npy')
else:
    # Create optimiser
    obj = MinDist3DRot(ca=xa_init, cb=xb_init, ra=Ra, rb=Rb, eps_a=eps_a, eps_b=eps_b, qa=qa_init, qb=qb_init, objective="NORM")
    vel_cont = VelocityController(ndim, 1)
    cnt = 0

    # Control loop
    for idx in range(1, STEPS):
        acnt = time.time()
        # Change sq parameters in optimiser
        obj.set_params(ca=x_opt_curr, cb=xb_init, qa=qa_curr, qb=qb_init)
        xa_star, lambda_a, xb_star, lambda_b = obj.get_primal_dual_solutions(False)

        # Optimisation approach
        x_error = x_traj[idx].t - x_opt_curr
        omega = calc_quat_error(UnitQuaternion(x_traj.R), UnitQuaternion(x_traj[idx].R))
        vel = np.array((x_error / DT, omega / DT)).reshape(6, 1)
        # desired * inv(current)
        G_opt = -np.array(obj.sensitivity_analysis())  # CBF derivative
        h_opt = GAMMA * obj.get_optimal_value()  # CBF exponential gamma*hx

        vel_cont.set_param(vel, G_opt, h_opt, UnitQuaternion(qa_curr))
        xd_opt_des = vel_cont.get_solution()
        cnt += time.time() - acnt

        # Integrate to obtain forward pose
        next_x_opt = x_opt_curr + xd_opt_des[:3]*DT
        next_qa = np.array(qa_curr[:3]) + 0.5*np.array(qa_curr[:3])*xd_opt_des[3:]*DT
        next_qa = np.array(qa_curr) / np.sqrt(np.sum(np.array(qa_curr) ** 2))
        
        # Save states
        # x_opt_history[idx, :] = x_opt_curr
        # xd_opt_history[idx, :] = xd_opt_des*DT
        # optimisation_h_history[idx, :] = h_opt/GAMMA
        # optimisation_hd_history[idx, :] = -G_opt
        # sqa_closest_history[idx, :] = xa_star
        # sqb_closest_history[idx, :] = xb_star

        # Update current state for next iteration
        x_opt_curr = next_x_opt
        qa_curr = next_qa
    print(1000*(cnt/STEPS))
s1 = SuperquadricObject(*Ra, *eps_a, pos=xa_init, quat=(1, 0, 0, 0))
s2 = SuperquadricObject(*Rb, *eps_b, pos=xb_init, quat=(1, 0, 0, 0))

plt.figure()
ax = plt.subplot(111, projection='3d')
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
s2.plot_sq(ax, 'red')
ax.plot(x_opt_history[:, 0], x_opt_history[:, 1], x_opt_history[:, 2])
plt.show()

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
    np.save('x_opt_history.npy', x_opt_history) # save
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
