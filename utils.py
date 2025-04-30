from spatialmath import UnitQuaternion
import sympy as sym
import numpy as np
import math
import os
import sys
from contextlib import contextmanager

def skew(x):
    """
    skew operator
    """
    if x.ndim > 1:
        x = x.squeeze()
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

def delta_rotation(angVel, deltaTime):
    # ha = angVel * (deltaTime * 0.5)  # vector of half angle
    # l = np.sqrt(np.sum(ha ** 2))  # magnitude
    #
    # if l > 1e-4:
    #     ha = (ha * math.sin(l)) / l
    #
    # return UnitQuaternion(s=math.cos(l), v=[ha[0], ha[1], ha[2]])
    _Q = np.array([
        [0, -angVel[0], -angVel[1], -angVel[2]],
        [angVel[0], 0, angVel[2], -angVel[1]],
        [angVel[1], -angVel[2], 0, angVel[0]],
        [angVel[2], angVel[1], -angVel[0], 0]
    ])
    delta_r = np.eye(4) + 0.5*_Q*deltaTime
    return delta_r

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

    # Limit the angle to [-pi pi]
    if angle > math.pi:
        angle = 2*math.pi - angle
        axis = -axis
    # print("here", quat_error.angvec(), (angle, axis))

    # Sanity check
    if (angle < -math.pi) and (angle > math.pi):
        print(angle)
        raise "Axis angle error outside of bounds"

    return angle, axis

def distance_derivative():
    """
    Symbolically compute distance and its derivative for spheres
    Returns: tuple | distance_function and distance_derivative_function
    """
    x_ee = sym.symbols('x_ee:3')
    x_obs = sym.symbols('x_obs:3')
    x_ee = sym.Matrix([[x_ee[0], x_ee[1], x_ee[2]]])
    x_obs = sym.Matrix([[x_obs[0], x_obs[1], x_obs[2]]])
    R_ee = sym.Symbol('R_ee')
    R_obs = sym.Symbol('R_obs')

    hx = sym.sqrt(
        (x_ee[0] - x_obs[0])**2 +
        (x_ee[1] - x_obs[1])**2 +
        (x_ee[2] - x_obs[2])**2
    ) - (R_obs + R_ee)

    hx_dot = sym.diff(hx, x_ee)

    hx = sym.lambdify([x_ee, x_obs, R_ee, R_obs], expr=hx)
    hx_dot = sym.lambdify([x_ee, x_obs, R_ee, R_obs], expr=hx_dot)
    return hx, hx_dot

@contextmanager
def stdout_redirected(to=os.devnull):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different
