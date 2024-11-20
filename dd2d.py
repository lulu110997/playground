"""
test distance derivative in 2d
"""
import sys

from scipy.spatial import ConvexHull
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np
from spatialmath import UnitQuaternion

def plot_min_point_movement():

    plt.ion()

    p1 = np.loadtxt('points1.txt', delimiter=',')
    p2 = np.loadtxt('points2.txt', delimiter=',')
    angles = np.loadtxt('quaternion_tests.txt')
    angles = [UnitQuaternion(s=i[-1], v=i[:3]).rpy('deg')[2] for i in angles]

    sq1_centre = (-0.1, -0.08)
    sq2_centre = (0.1, 0.0)
    sq_width = 0.11*2
    sq_height = 0.03752*2

    sq2 = Ellipse(sq2_centre, sq_width, sq_height, 0, facecolor='none', edgecolor='b', lw=1)

    a = plt.subplot(111)
    ax = plt.gca()
    ax.set_xlim(-0.25, 0.25)
    ax.set_ylim(-0.25, 0.25)
    a.add_artist(sq2)

    for idx, angle in enumerate(angles):
        p1_h, = plt.plot(p1[idx, 0], p1[idx, 1], 'r.')
        p2_h, = plt.plot(p2[idx, 0], p2[idx, 1], 'r.')
        l_h, = plt.plot([p1[idx, 0], p2[idx, 0]], [p1[idx, 1], p2[idx, 1]], 'r-o')
        e = Ellipse(sq1_centre, sq_width, sq_height, facecolor='none', edgecolor='r', lw=1)
        e.angle = angle
        a.add_artist(e)

        plt.pause(0.05)
        e.remove()
        # p1_h.remove()

    plt.show(block=True)

def discretise_ellipse():
    """
    This is done by creating a bounding box around the ellipse then iterating through points inside the bounding box. If
    the point lies in the ellipse, we save it in a list
    """
    DISC_VAL = 70

    a = plt.subplot(111)
    a.set_xlim(-0.70, 0.70)
    a.set_ylim(-0.70, 0.70)
    a.grid()

    # Create two ellipses
    e1 = Ellipse((0.2, 0.3), 0.2, 0.1, color='g')
    e2 = Ellipse((0.1, 0.1), 0.1, 0.15, color='b')
    a.add_artist(e1)
    a.add_artist(e2)
    e1_points = [[], []]
    e2_points = [[], []]

    # Obtain bounding box
    e1_corners = e1.get_corners()
    e1_x = (e1_corners[0][0], e1_corners[1][0])
    e1_y = (e1_corners[0][1], e1_corners[3][1])
    e2_corners = e2.get_corners()
    e2_x = (e2_corners[0][0], e2_corners[1][0])
    e2_y = (e2_corners[0][1], e2_corners[3][1])

    # Find points inside the bounding box that define each ellipse
    xs = np.linspace(e1_x[0], e1_x[1], num=DISC_VAL)
    ys = np.linspace(e1_y[0], e1_y[1], num=DISC_VAL)
    for i in xs:
        for j in ys:
            if e1.contains_point(a.transData.transform((i, j))):
                e1_points[0].append(i)
                e1_points[1].append(j)

    xs = np.linspace(e2_x[0], e2_x[1], num=DISC_VAL)
    ys = np.linspace(e2_y[0], e2_y[1], num=DISC_VAL)
    for i in xs:
        for j in ys:
            # contains_point uses the internal coordinates to determine whether some point is contained in the patch.
            # Need to transform the point to the screen coordinate system first and then query if it is contained in the
            # patch
            if e2.contains_point(a.transData.transform((i, j))):
                e2_points[0].append(i)
                e2_points[1].append(j)

    # Minkowski difference
    # https://stackoverflow.com/questions/57114574/how-to-subtract-each-element-of-an-array-from-another-array
    A = np.array(e1_points[0])
    B = np.array(e2_points[0])
    x_diff = A - B[:, np.newaxis]
    x_diff = x_diff.flatten()

    A = np.array(e1_points[1])
    B = np.array(e2_points[1])
    y_diff = A - B[:, np.newaxis]
    y_diff = y_diff.flatten()

    points = np.column_stack((x_diff, y_diff))
    hull = ConvexHull(points)

    # Plot the points
    e1.remove()
    e2.remove()
    plt.scatter(e1_points[0], e1_points[1], color='g')
    plt.scatter(e2_points[0], e2_points[1], color='b')
    plt.scatter(x_diff[1::50], y_diff[1::50], alpha=0.05, color='r')
    plt.plot(points[hull.vertices, 0], points[hull.vertices, 1], 'r--', lw=2)
    plt.show()

discretise_ellipse()