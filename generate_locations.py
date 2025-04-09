import math
import sys

import numpy as np
import yaml
from roboticstoolbox.tools import trajectory
from spatialmath import SE3, UnitQuaternion
from casadi_min_dist import MinDist3D
from superquadric import SuperquadricObject
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

def sample_point_from_rotated_box(center, size, rotation_matrix):
    """
    Samples a random point from within a rotated 3D bounding box.

    Parameters:
    - center: (3,) array-like, the center of the bounding box
    - size: (3,) array-like, the box size along each axis (length, width, height)
    - rotation_matrix: (3, 3) numpy array, rotation to apply to the box

    Returns:
    - A sampled 3D point inside the rotated bounding box
    """
    center = np.array(center)
    size = np.array(size)

    # Sample a point in the axis-aligned box (centered at origin)
    local_point = (np.random.rand(3) - 0.5) * size  # uniform in [-0.5*size, 0.5*size]

    # Rotate and translate
    rotated_point = rotation_matrix @ local_point
    world_point = rotated_point + center

    return world_point


def draw_rotated_bounding_box(center, size, rotation_matrix, ax=None, color='blue', face_alpha=0.1):
    """
    Draws a rotated 3D bounding box with transparent faces using matplotlib.

    Parameters:
    - center: (3,) array-like, center of the box
    - size: (3,) array-like, box dimensions (length, width, height)
    - rotation_matrix: (3, 3) numpy array, rotation matrix
    - ax: optional matplotlib 3D axes to plot on
    - color: color of the bounding box
    - face_alpha: transparency of the faces (0 = fully transparent, 1 = opaque)
    """
    center = np.array(center)
    size = np.array(size)

    # Define 8 corner points in local (unrotated) box coordinates
    l, w, h = size / 2
    corners = np.array([
        [-l, -w, -h],
        [+l, -w, -h],
        [+l, +w, -h],
        [-l, +w, -h],
        [-l, -w, +h],
        [+l, -w, +h],
        [+l, +w, +h],
        [-l, +w, +h]
    ])

    # Rotate and translate corners
    rotated_corners = (rotation_matrix @ corners.T).T + center

    # Define edges (pairs of indices)
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]

    # Define faces (as lists of corner indices)
    faces = [
        [0, 1, 2, 3],  # bottom
        [4, 5, 6, 7],  # top
        [0, 1, 5, 4],  # side 1
        [1, 2, 6, 5],  # side 2
        [2, 3, 7, 6],  # side 3
        [3, 0, 4, 7]   # side 4
    ]

    # Plot setup
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # Draw edges
    lines = [(rotated_corners[i], rotated_corners[j]) for i, j in edges]
    lc = Line3DCollection(lines, colors=color, linewidths=1.0)
    ax.add_collection3d(lc)

    # Draw faces with transparency
    face_vertices = [[rotated_corners[i] for i in face] for face in faces]
    pc = Poly3DCollection(face_vertices, facecolors=color, alpha=face_alpha, linewidths=0.5, edgecolors=color)
    ax.add_collection3d(pc)

    # Auto scale
    all_pts = rotated_corners
    max_range = np.ptp(all_pts, axis=0).max() * 0.6
    mean = np.mean(all_pts, axis=0)
    for axis, m in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], mean):
        axis([m - max_range, m + max_range])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # plt.show()


def rotation_matrix_from_line(p1, p2, up=np.array([0, 0, 1])):
    """
    Computes a rotation matrix that aligns the Z-axis to the vector from p1 to p2.

    Parameters:
    - p1, p2: 3D points defining the line segment.
    - up: the preferred 'up' direction (used to define the full orientation).

    Returns:
    - A 3x3 rotation matrix where:
        - column 0 is the X-axis,
        - column 1 is the Y-axis,
        - column 2 is the Z-axis (aligned with p2 - p1).
    """
    p1 = np.array(p1, dtype=np.float64)
    p2 = np.array(p2, dtype=np.float64)
    up = np.array(up, dtype=np.float64)

    # Line direction (target Z-axis)
    z_axis = p2 - p1
    z_axis /= np.linalg.norm(z_axis)

    # Avoid colinearity
    if np.allclose(np.abs(np.dot(z_axis, up)), 1.0):
        up = np.array([1, 0, 0])  # use alternative up if colinear

    # New X-axis
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    # New Y-axis
    y_axis = np.cross(z_axis, x_axis)

    # Rotation matrix: columns are the new axes
    R = np.column_stack([x_axis, y_axis, z_axis])
    return R


# Save and load paths
TEST_TYPE = 'compare with tracy'
TEST_DIR = 'test1'
TEST_NAME = 'test1'
REPLAY = 0; cwd = f'test cases/{TEST_TYPE}/{TEST_DIR}/{TEST_NAME}_'
SAVE = 1; sd = f'test cases/{TEST_TYPE}/{TEST_DIR}/{TEST_NAME}_'

# TODO: Define NUM_SIM in yaml? or just as an arg?
NUM_SIM = 100
NP_SEED = 9

np.random.seed(NP_SEED)

with open(f"test cases/{TEST_TYPE}/{TEST_DIR}/{TEST_NAME}.yaml") as file:
    try:
        params = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# Define lower and upper bounds
lower_bounds1 = np.array([-0.74798409,  0.08765844,  0.10815752])  # 440
upper_bounds1 = np.array([-0.49244496,  0.30491329,  0.21712809])  # 650

lower_bounds2 = np.array([-0.55439293, -0.3062642,  0.23514009])
upper_bounds2 = np.array([ 0.311791851, 0.08765844,  0.35798119])

# Superquadric parameters
Rb = params['Rb']
eps_b = params['eps_b']

# TODO: Some comment
xa_init = params['xa_init']  # Initial robot position
qa_init = params['qa_init']  # Initial robot position
xa_tgt = params['xa_tgt']  # Final robot position
qa_tgt = params['qa_tgt']  # Final robot position
qb_init = params['qb_init']
FREQ = params['FREQ']
TIME = params['TIME']
DT = 1.0/FREQ
STEPS = int(TIME/DT)

# Create the trajectory
initial_pose = SE3(xa_init) @ UnitQuaternion(s=qa_init[0], v=qa_init[1:]).SE3()
final_pose = SE3(xa_tgt) @ UnitQuaternion(s=qa_tgt[0], v=qa_tgt[1:]).SE3()
x_traj = trajectory.ctraj(initial_pose, final_pose, STEPS)
x_traj_line_rot = rotation_matrix_from_line(x_traj[0].t, x_traj[-1].t)

# print(np.linalg.norm(x_traj[1040].t - x_traj[-1].t)); sys.exit()
# c1 = x_traj[600].t; c1[1] += 0.055; c1[2] += 0.055
# c2 = x_traj[940].t; c2[1] -= 0.055; c2[2] -= 0.055
# print(c1, c2); sys.exit()

# Define bb centre and size
c1 = [-0.54447965,  0.17467979,  0.26012008]
c2 = [-0.10218479, -0.20750167,  0.25218813]
size = (0.07, 0.15, 0.35)

for i in range(NUM_SIM):
    if i == 30:
        c1[1] -= 2*0.055
        c2[1] += 2*0.055
    if i == 60:
        c1[2] -= 2*0.055
        c2[2] += 2*0.055

    while True:
        # Sample points
        obstacles = [sample_point_from_rotated_box(c1, size, x_traj_line_rot),
                     sample_point_from_rotated_box(c2, size, x_traj_line_rot)]

        # Check that both shapes have a min dist of 0.05 unit between each other
        calculator = MinDist3D(ca=list(obstacles[0]), cb=list(obstacles[1]), ra=Rb, rb=Rb, eps_a=eps_b, eps_b=eps_b,
                               qa=list(qa_init), qb=list(qb_init))
        x_star_, lambda_star_ = calculator.get_primal_dual_solutions(c1 + c2)
        dist = calculator.get_optimal_value()
        solved = calculator.get_solver_stats()['success']
        if not solved:
            # Solution was not found, might be due to bad initial guess
            print(c1, c2)
            continue
        if dist < 0.2:
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # s1 = SuperquadricObject(*Rb, *eps_b, pos=obstacles[0], quat=qb_init)
            # s1_handle = s1.plot_sq(ax, 'red')
            # s2 = SuperquadricObject(*Rb, *eps_b, pos=obstacles[1], quat=qb_init)
            # s2_handle = s2.plot_sq(ax, 'red')
            # ax.plot(x_traj.t[:, 0], x_traj.t[:, 1], x_traj.t[:, 2], color='g')
            # draw_rotated_bounding_box(c1, size, x_traj_line_rot, ax)
            # draw_rotated_bounding_box(c2, size, x_traj_line_rot, ax)
            # line_handle = ax.plot((x_star_[0], x_star_[3]),
            #                       (x_star_[1], x_star_[4]),
            #                       (x_star_[2], x_star_[5]), 'ro-')
            # ax.view_init(83, 144, 0)
            # ax.set_xlim(-1.0, 0.2)
            # ax.set_ylim(-0.5, 0.5)
            # ax.set_zlim(0.0, 0.55)
            # plt.show()
            continue


        break

    # Append to file (as a single row)
    with open('xb_init.txt', 'a') as f:
        if i == 32 or i ==62 or i ==10:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            s1 = SuperquadricObject(*Rb, *eps_b, pos=obstacles[0], quat=qb_init)
            s1_handle = s1.plot_sq(ax, 'red')
            s2 = SuperquadricObject(*Rb, *eps_b, pos=obstacles[1], quat=qb_init)
            s2_handle = s2.plot_sq(ax, 'red')
            ax.plot(x_traj.t[:, 0], x_traj.t[:, 1], x_traj.t[:, 2], color='g')
            draw_rotated_bounding_box(c1, size, x_traj_line_rot, ax)
            draw_rotated_bounding_box(c2, size, x_traj_line_rot, ax)
            ax.view_init(50, -179, 0)
            ax.set_xlim(-1.0, 0.2)
            ax.set_ylim(-0.5, 0.5)
            ax.set_zlim(0.0, 0.55)
            plt.show()
        combined = np.concatenate([obstacles[0], obstacles[1]]).reshape(1, -1)
        np.savetxt(f, combined)