# To be able to add up the axis-angle representations, the initial orientation must be the Identity element. See ex1

# Aim: After we find the error between two orientation, can we alter the original orientation so that when we find the
# error, we cam use a scaling factor which will change the velocity in an 'expected' manner (eg wz will always be scaled
# up). Similar to how we can add a gain in front of a distance function to scale one of the components.
# Can maybe look into using geodesic distance in the SO3 manifold? Or an equivalent strategy in the Lie algebra since
# it's easier to manipulate objects in the Lie algebra space?

import sys

import roboticstoolbox as rp
import numpy as np
from swift import Swift
import spatialgeometry as sg
from spatialmath import SE3, SO3, UnitQuaternion
from spatialmath.base import trotx, troty, trotz, angvec2tr
from math import pi as PI
import math

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
np.random.seed(1)
SCALE_ANG = 2

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
    if angle > PI:
        angle = 2*PI - angle
        axis = -axis

    # Sanity check
    if (angle < -PI) and (angle > PI):
        print(angle)
        raise "Axis angle error outside of bounds"

    return angle, axis

def skew(x):
    """
    skew operator
    """
    if x.ndim > 1:
        x = x.squeeze()
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

def ex1():
    """
    Show how adding up the angle-axis representation is valid if they are in the Lie algebra (ie R1 = I so that order of
    operation, multiplying left or right, does not matter)
    """
    # Initial orientation
    r1_t = SE3()

    # Final orientation
    r2_t = SE3(0.3, 0.3, 0)@SE3(trotx(PI*(30/180))@trotz(PI*(40/180)))@SE3(troty(PI/-6))

    # Obtain the axis-angle representation required to go from r1_t to r2_t. The angle-axis representation is expressed
    # in the world frame
    theta, a_hat = calc_quat_error(UnitQuaternion(r1_t), UnitQuaternion(r2_t))

    # Obtain rotation matrix based on the axis angle
    dR = np.eye(3) + skew(a_hat) * np.sin(theta) + skew(a_hat) @ skew(a_hat) * (1 - np.cos(theta))

    # Integrate angular velocity to obtain the next position
    final = dR @ r1_t.R
    check = np.abs(final - r2_t.R)
    np.testing.assert_array_less(check, 1e-3)
    print("check1:\n", check)

    # Add up omega and axis angle. Only works if the final rotation is defined wrt to the identity
    # https://stackoverflow.com/questions/4316712/combine-rotation-axis-vectors
    a_final = theta * a_hat + r1_t.angvec()[0] * r1_t.angvec()[1]
    theta_final = np.linalg.norm(a_final)
    a_hat_final = a_final / theta_final
    expected_final = r2_t.angvec()
    print("check2:\n", theta_final - expected_final[0], a_hat_final - expected_final[1])
    print("check2:\n", (theta_final * a_hat_final) - (expected_final[0] * expected_final[1]))

    # Calculate dR w/o using quaternion
    # https://robotics.stackexchange.com/questions/20295/how-to-compute-the-orientation-error-between-two-3d-coordinate-frames
    # Need to perform post multiplication of the inverse of the initial frame to obtain error in the world frame
    rot_error_post = SO3(r2_t.R @ r1_t.R.transpose())
    print("check3:\n", dR - rot_error_post)

def ex2():
    """
    If R1 is not identity, adding up the angle-axis parameters only work when R_error is small? Likely due to the higher
    order terms from the Baker–Campbell–Hausdorff formula?
    https://en.wikipedia.org/wiki/Baker%E2%80%93Campbell%E2%80%93Hausdorff_formula
    """
    # Initial orientation
    r1_t = SE3(trotz(PI*(5/180)))@SE3(troty(PI*(3/180)))@SE3(trotx(PI/-10))

    # Final orientation
    r2_t = SE3(0.3, 0.3, 0)@SE3(trotx(PI*(5/180)))@SE3(troty(PI/12))

    # Obtain the axis-angle representation required to go from r1_t to r2_t. The angle-axis representation is expressed
    # in the world frame
    q_theta, q_a_hat = calc_quat_error(UnitQuaternion(r1_t), UnitQuaternion(r2_t))
    theta, a_hat = SO3(r2_t.R @ r1_t.R.transpose()).angvec()  # eq28
    dR = np.eye(3) + skew(a_hat) * np.sin(theta) + skew(a_hat) @ skew(a_hat) * (1 - np.cos(theta))

    # Go from local tangent space to Lie algebra
    local_tangent_space = SO3(r1_t.R.transpose()@r2_t.R).log()  # eq 26
    l_axis_angle = [-local_tangent_space[1, 2], local_tangent_space[0, 2], -local_tangent_space[0, 1]]
    l_theta = np.linalg.norm(l_axis_angle)
    l_a_hat = l_axis_angle/l_theta
    l_dR = np.eye(3) + skew(l_a_hat) * np.sin(l_theta) + skew(l_a_hat) @ skew(l_a_hat) * (1 - np.cos(l_theta))
    converted_dR = r1_t.R @ l_dR @ r1_t.R.transpose()
    converted_theta, converted_a_hat = SO3(converted_dR).angvec()

    # The calculated omega from the different representations should match up
    assert all(theta*a_hat - converted_theta*converted_a_hat < 1e-3)
    assert all(theta*a_hat - q_theta*q_a_hat < 1e-3)

    # Integrate angular velocity to obtain the next position
    final = dR @ r1_t.R
    check = np.abs(final - r2_t.R)
    print(SO3(r2_t.R).angvec())
    print(SO3(final).angvec())
    np.testing.assert_array_less(check, 1e-3)
    print("check1:\n", check)

    # Add up omega and axis angle. Only works if the final rotation is defined wrt to the identity
    # https://stackoverflow.com/questions/4316712/combine-rotation-axis-vectors
    a_final = theta * a_hat + r1_t.angvec()[0] * r1_t.angvec()[1]
    theta_final = np.linalg.norm(a_final)
    a_hat_final = a_final / theta_final
    expected_final = r2_t.angvec()
    print("check2:\n", theta_final - expected_final[0], a_hat_final - expected_final[1])
    print("check2:\n", (theta_final * a_hat_final) - (expected_final[0] * expected_final[1]))

    # Calculate dR w/o using quaternion
    # https://robotics.stackexchange.com/questions/20295/how-to-compute-the-orientation-error-between-two-3d-coordinate-frames
    # Need to perform post multiplication of the inverse of the initial frame to obtain error in the world frame
    rot_error_post = SO3(r2_t.R @ r1_t.R.transpose())
    print("check3:\n", dR - rot_error_post)

def ex3():
    # Initial orientation
    r1_t = SE3(trotz(PI*(5/180)))@SE3(troty(PI*(3/180)))@SE3(trotx(PI/-8))

    # Final orientation
    r2_t = SE3(0.3, 0.3, 0)@SE3(trotx(PI*(3/180))@trotz(PI*(-5/180)))

    # Obtain rotation matrix error
    dR = SO3(r2_t.R @ r1_t.R.transpose())
    theta, a_hat = dR.angvec()

    # Integrate angular velocity to obtain the next position
    final = dR @ r1_t.R
    check = np.abs(final - r2_t.R)
    np.testing.assert_array_less(check, 1e-3)

    # Scale the angular velocity, where do we end up?
    omega = theta*a_hat
    omega_scaled = np.copy(omega)
    omega_scaled[2] = omega_scaled[2]*SCALE_ANG  # Scale wz
    theta_scaled = np.linalg.norm(omega_scaled)
    a_hat_scaled = omega_scaled/theta_scaled
    print(np.linalg.norm(a_hat_scaled))
    assert abs(1 - np.linalg.norm(a_hat_scaled)) < 1e-3

    # Calculate the scaled orientation
    dR_scaled = np.eye(3) + skew(a_hat_scaled)*np.sin(theta_scaled) + skew(a_hat_scaled)@skew(a_hat_scaled)*(1 - np.cos(theta_scaled))
    final_scaled = dR_scaled@r1_t.R
    print("scaled components", a_hat_scaled, theta_scaled, theta_scaled*a_hat_scaled)
    print("original components", a_hat, theta, theta*a_hat)
    print(dR_scaled, UnitQuaternion(dR_scaled))
    print(dR, UnitQuaternion(dR))
    # Example above shows that we can scale one of the angular velocity components. Just need to reverse engineer it to see
    # what needs to be scaled in the rotation matrix

    # How can we find out what we need to 'scale' the rotation matrix of the sq to be to scale the angular velocity
    scaling_matr = np.linalg.inv(dR)@final_scaled@np.linalg.inv(r1_t.R)
    print("scaling matrix:\n", scaling_matr)
    # check if scaling matrix is orthogonal (ie RR^T = I)
    check = np.abs(scaling_matr@scaling_matr.transpose() - np.eye(3))
    np.testing.assert_array_less(check, 1e-3)

    scaling_matr = SO3(scaling_matr)
    # print(scaling_matr.angvec()[0]*scaling_matr.angvec()[1])
    sys.exit()

    # Make and instance of the Swift simulator and open it
    env = Swift()
    env.launch(realtime=True, browser='firefox')

    r1 = sg.Axes(0.25)
    r2 = sg.Axes(0.25)
    r1.T = r1_t
    r2.T = r2_t
    env.add(r1)
    env.add(r2)

    r2_scaled = sg.Axes(0.35, pose=SE3(r2_t.t) @ SE3(SO3(final_scaled)))
    env.add(r2_scaled)

    # env.hold()

# Rotate the sq at a certain rate
a_hat = np.array([0, 0, 1])
theta = 0.01
Rk = UnitQuaternion(0.9848078, [0.0, 0.0, 0.1736482]).SO3()
q_list = [UnitQuaternion(0.9848078, [0.0, 0.0, 0.1736482]).vec_xyzs.tolist()]
for i in range(180):

    dR = np.eye(3) + skew(a_hat) * np.sin(theta) + skew(a_hat) @ skew(a_hat) * (1 - np.cos(theta))
    R_next = SO3(dR)@Rk
    # print(Rk.rpy(unit="deg"))
    # print(R_next.rpy(unit="deg"))
    # print(UnitQuaternion(R_next).vec_xyzs)
    # print("")
    q_list.append(UnitQuaternion(R_next).vec_xyzs.tolist())
    Rk = R_next
print(R_next.rpy(unit="deg"))

# with open("quaternion_tests.txt", "w") as output:
#     for lines in q_list:
#         output.write(str(lines).replace('[', '').replace(']', '').replace(',', ''))
#         output.write('\n')