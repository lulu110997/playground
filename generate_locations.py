from tqdm import tqdm
import yaml
from roboticstoolbox.tools import trajectory
from spatialmath import SE3, UnitQuaternion
from casadi_min_dist import MinDist3D
from superquadric import SuperquadricObject
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from scipy.spatial.transform import Rotation
import numpy as np
import matplotlib.pyplot as plt

def point_to_segment_distance(point, seg_start, seg_end):
    """
    Computes the shortest distance from a point to a line segment in 3D.

    Args:
        point: array-like | coordinates of the point
        seg_start: array-like | coordinates of the start of the segment
        seg_end: array-like | coordinates of the end of the segment

    Returns:
        distance: float | shortest distance from point to segment
    """
    p = np.array(point)
    a = np.array(seg_start)
    b = np.array(seg_end)
    ab = b - a
    ap = p - a

    # Project point p onto the line defined by a–b
    t = np.dot(ap, ab) / np.dot(ab, ab)
    t_clamped = np.clip(t, 0, 1)  # Clamp t to [0, 1] to stay on the segment

    # Find the closest point on the segment
    closest_point = a + t_clamped * ab

    # Distance from point to closest point
    distance = np.linalg.norm(p - closest_point)

    return distance

def sample_point_from_rotated_box(center, size, rotation_matrix):
    """
    Samples a random point from within a rotated 3D bounding box.

    Parameters:
    center: array-like | the center of the bounding box
    size: array-like | the box size along each axis (length, width, height)
    rotation_matrix: numpy array | rotation to apply to the box

    Returns:
        A sampled 3D point inside the rotated bounding box
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

    Args:
        center: array-like | center of the box
        size: array-like | box dimensions (length, width, height)
        rotation_matrix: numpy array | rotation matrix
        ax: optional matplotlib 3D axes to plot on
        color: color of the bounding box
        face_alpha: transparency of the faces (0 = fully transparent, 1 = opaque)
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

    Args:
        p1, p2: 3D points defining the line segment.
        up: the preferred 'up' direction (used to define the full orientation).

    Returns:
        A 3x3 rotation matrix
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


NUM_SIM = 1000
NP_SEED = 9
PLOT = 0
SAVE = 1
TRAJ_D = 0.05  # Min dist of obstacle centre from traj
SQ_D = 0.2  # Min dist between the two obstacles

# Define bb centre and size
c1 = [-0.32333222, -0.01641094, 0.256154105]
size = (0.25, 0.2, 1.0)  # y, z, x

# Save and load paths
TEST_TYPE = 'compare with tracy'
TEST_NAME = 'test'

np.random.seed(NP_SEED)

with open(f"test cases/{TEST_TYPE}/{TEST_NAME}.yaml") as file:
    try:
        params = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# Parameters
Rb = params['Rb']  # Obstacle radii
eps_b = params['eps_b']  # Obstacle epsilon value
xa_init = params['xa_init']  # Initial robot position
qa_init = params['qa_init']  # Initial robot orientation
xa_tgt = params['xa_tgt']  # Final robot position
qa_tgt = params['qa_tgt']  # Final robot orientation
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

solver_options = {"linear_solver": "ma27", "sb": "yes", "print_level": 0, "tol": 1e-4}

for i in tqdm(range(NUM_SIM)):
    while True:
        # Sample points
        obs_pos = [sample_point_from_rotated_box(c1, size, x_traj_line_rot),
                     sample_point_from_rotated_box(c1, size, x_traj_line_rot)]
        random_ori = Rotation.random(2).as_quat()
        ori1 = (random_ori[0][-1], random_ori[0][0], random_ori[0][1], random_ori[0][2])
        ori2 = (random_ori[1][-1], random_ori[1][0], random_ori[1][1], random_ori[1][2])

        calculator = MinDist3D(ca=list(obs_pos[0]), cb=list(obs_pos[1]), ra=Rb, rb=Rb, eps_a=eps_b, eps_b=eps_b,
                               qa=list(ori1), qb=list(ori2), solver_options=solver_options)
        x_star_, lambda_star_ = calculator.get_primal_dual_solutions(list(obs_pos[0]) + list(obs_pos[1]), [0, 0])
        dist = calculator.get_optimal_value()
        solved = calculator.get_solver_stats()['success']

        if PLOT:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            s1 = SuperquadricObject(*Rb, *eps_b, pos=obs_pos[0], quat=ori1)
            s1_handle = s1.plot_sq(ax, 'red')
            s2 = SuperquadricObject(*Rb, *eps_b, pos=obs_pos[1], quat=ori2)
            s2_handle = s2.plot_sq(ax, 'red')
            ax.plot(x_traj.t[:, 0], x_traj.t[:, 1], x_traj.t[:, 2], color='g')
            draw_rotated_bounding_box(c1, size, x_traj_line_rot, ax)
            line_handle = ax.plot((x_star_[0], x_star_[3]),
                                  (x_star_[1], x_star_[4]),
                                  (x_star_[2], x_star_[5]), 'ro-')
            ax.view_init(83, 144, 0)
            ax.set_xlim(-1.0, 0.2)
            ax.set_ylim(-0.5, 0.5)
            ax.set_zlim(0.0, 0.55)
            ax.set_aspect('equal')

            plt.show()

        # Ensure the point centre of SQ is some distance away from the obstacle and check that both shapes have a min
        # dist of 0.05 unit between each other
        # Distance between the two is too small OR Solution was not found, might be due to bad initial guess OR offset
        # the centre of one/both of the obstacle does not satisfy the minimum distance from the trajectory
        if max(point_to_segment_distance(obs_pos[0], xa_init, xa_tgt),
               point_to_segment_distance(obs_pos[1], xa_init, xa_tgt)) < TRAJ_D or dist < SQ_D or not solved:
            continue
        break

    # Append to file (as a single row)
    if SAVE:
        with open('xb_pos.txt', 'a') as f:
            combined = np.concatenate([obs_pos[0], obs_pos[1]]).reshape(1, -1)
            np.savetxt(f, combined)
        with open('xb_ori.txt', 'a') as f:
            combined = np.concatenate([ori1, ori2]).reshape(1, -1)
            np.savetxt(f, combined)