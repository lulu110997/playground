"""
Sanity checks
"""
import time

from numpy.ma.testutils import assert_almost_equal
from superquadric import SuperquadricObject
from min_dists import MinDist2D, MinDist3DTransl, MinDist3DRot
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

###################################################### 2D tests ########################################################

def ellipse_2d_handwritten_check(plot=False):
    """
    Checking we coded things correctly by validating code results with handwritten result and optimal points from ipopt
    These tests use SOS as the objective function which is an equivalent problem for what we are looking for
    """
    # Define parameters of SQ
    cxa, cxb = (0.3, 1), (2.5, -0.1)
    ra, rb = (0.2, 0.5), (0.5, 0.4)
    obj = MinDist2D(cxa, cxb, ra, rb)
    xa, lambda_a ,xb, lambda_b = obj.get_primal_dual_solutions(requires_grad=True)
    analytic_tmp = obj.sensitivity_analysis()

    # Check primal dual solutions match with ipopt
    assert_almost_equal(xa, [0.44204, 0.64799], decimal=5)
    assert_almost_equal(xb, [2.0234, 0.020948], decimal=5)
    assert_almost_equal(lambda_a, 0.44534, decimal=5)
    assert_almost_equal(lambda_b, 0.82951, decimal=5)

    # Check nabla_theta matches with handwritten sol
    # Tales into account rounding error since values used to calc nabla_theta were rounded to 5 decimal
    actual = obj.sensitivity_analysis()
    desired = [-3.1628, 1.254, 3.1628, -1.254]
    assert_almost_equal(np.array(actual), desired, decimal=4)

    if plot:
        print(xa, xb)
        plt.figure()
        ax = plt.gca()
        plt.plot((xa[0], xb[0]), (xa[1], xb[1]), 'ro-')
        ax.add_patch(Ellipse(xy=cxa, width=2*ra[0], height=2*ra[1]))
        ax.add_patch(Ellipse(xy=cxb, width=2*rb[0], height=2*rb[1]))
        plt.text(*cxa, np.round(analytic_tmp[:2], 2), fontsize=12)
        plt.text(*cxb, np.round(analytic_tmp[2:], 2), fontsize=12)
        plt.axis('scaled')
        plt.show()

def ellipse_2d_numerical_diff():
    """
    https://en.wikipedia.org/wiki/Numerical_differentiation
    Use two-point formula to check the derivatives we compute is correct
    These tests use SOS as the objective function which is an equivalent problem for what we are looking for
    """
    PERTURBATION = 1e-4

    # Define parameters of SQ
    cxa, cxb = [0.3, 1], [2.5, -0.1]
    ra, rb= (0.2, 0.5), (0.5, 0.4)

    # Text for nabla_cxa
    cxa_perturb = cxa[:]
    cxa_perturb[0] = cxa_perturb[0] - PERTURBATION
    obj = MinDist2D(cxa_perturb, cxb, ra, rb)
    xa, __ ,xb, __ = obj.get_primal_dual_solutions(requires_grad=True)
    dist_minus = (xa[0] - xb[0])**2 + (xa[1] - xb[1])**2

    cxa_perturb = cxa[:]
    cxa_perturb[0] = cxa_perturb[0] + PERTURBATION
    obj.set_params(cxa_perturb, cxb)
    xa, lambda_a ,xb, lambda_b = obj.get_primal_dual_solutions(requires_grad=True)
    dist_plus = (xa[0] - xb[0])**2 + (xa[1] - xb[1])**2

    nabla_cxa = (dist_plus - dist_minus)/(2*PERTURBATION)

    # Text for nabla_cya
    cxa_perturb = cxa[:]
    cxa_perturb[1] = cxa_perturb[1] - PERTURBATION
    obj = MinDist2D(cxa_perturb, cxb, ra, rb)
    xa, __, xb, __ = obj.get_primal_dual_solutions(requires_grad=True)
    dist_minus = (xa[0] - xb[0]) ** 2 + (xa[1] - xb[1]) ** 2

    cxa_perturb = cxa[:]
    cxa_perturb[1] = cxa_perturb[1] + PERTURBATION
    obj.set_params(cxa_perturb, cxb)
    xa, lambda_a, xb, lambda_b = obj.get_primal_dual_solutions(requires_grad=True)
    dist_plus = (xa[0] - xb[0]) ** 2 + (xa[1] - xb[1]) ** 2

    nabla_cya = (dist_plus - dist_minus) / (2 * PERTURBATION)

    # Text for nabla_cxb
    cxb_perturb = cxb[:]
    cxb_perturb[0] = cxb_perturb[0] - PERTURBATION
    obj = MinDist2D(cxa, cxb_perturb, ra, rb)
    xa, __, xb, __ = obj.get_primal_dual_solutions(requires_grad=True)
    dist_minus = (xa[0] - xb[0]) ** 2 + (xa[1] - xb[1]) ** 2

    cxb_perturb = cxb[:]
    cxb_perturb[0] = cxb_perturb[0] + PERTURBATION
    obj.set_params(cxa, cxb_perturb)
    xa, lambda_a, xb, lambda_b = obj.get_primal_dual_solutions(requires_grad=True)
    dist_plus = (xa[0] - xb[0]) ** 2 + (xa[1] - xb[1]) ** 2

    nabla_cxb = (dist_plus - dist_minus) / (2 * PERTURBATION)

    # Text for nabla_cyb
    cxb_perturb = cxb[:]
    cxb_perturb[1] = cxb_perturb[1] - PERTURBATION
    obj = MinDist2D(cxa, cxb_perturb, ra, rb)
    xa, __, xb, __ = obj.get_primal_dual_solutions(requires_grad=True)
    dist_minus = (xa[0] - xb[0]) ** 2 + (xa[1] - xb[1]) ** 2

    cxb_perturb = cxb[:]
    cxb_perturb[1] = cxb_perturb[1] + PERTURBATION
    obj.set_params(cxa, cxb_perturb)
    xa, lambda_a, xb, lambda_b = obj.get_primal_dual_solutions(requires_grad=True)
    dist_plus = (xa[0] - xb[0]) ** 2 + (xa[1] - xb[1]) ** 2

    nabla_cyb = (dist_plus - dist_minus) / (2 * PERTURBATION)

    obj = MinDist2D(cxa, cxb, ra, rb)
    obj.get_primal_dual_solutions(requires_grad=True)

    # Obtain analytic gradient and cvxpy predicted
    analytic_tmp =  obj.sensitivity_analysis()
    tmp = obj.sensitivity_cvxpy()

    # Reformat data for easy reading when printed
    analytic = [round(i, 6) for i in analytic_tmp]
    cvxpy_pred = [round(i, 6) for i in tmp[0].tolist()]
    cvxpy_pred.extend([round(i, 6) for i in tmp[1].tolist()])
    nabla_cxa = round(nabla_cxa, 6)
    nabla_cya = round(nabla_cya, 6)
    nabla_cxb = round(nabla_cxb, 6)
    nabla_cyb = round(nabla_cyb, 6)
    # Print solutions
    print("gradients:")
    print("actual           ", analytic)
    print("cvxpy predicted: ", cvxpy_pred)
    print(f"numerical diff:   [{nabla_cxa}, {nabla_cya}, {nabla_cxb}, {nabla_cyb}]")

    # f(x+2h)
    cxa_perturb = cxa[:]
    cxa_perturb[0] = cxa_perturb[0] + 2*PERTURBATION
    obj.set_params(cxa_perturb, cxb)
    xa, lambda_a ,xb, lambda_b = obj.get_primal_dual_solutions(requires_grad=True)
    dist_2plus = (xa[0] - xb[0])**2 + (xa[1] - xb[1])**2

    # f(x-2h)
    cxa_perturb = cxa[:]
    cxa_perturb[0] = cxa_perturb[0] - 2*PERTURBATION
    obj = MinDist2D(cxa_perturb, cxb, ra, rb)
    xa, __ ,xb, __ = obj.get_primal_dual_solutions(requires_grad=True)
    dist_2minus = (xa[0] - xb[0])**2 + (xa[1] - xb[1])**2

    # f(x+h)
    cxa_perturb = cxa[:]
    cxa_perturb[0] = cxa_perturb[0] + PERTURBATION
    obj.set_params(cxa_perturb, cxb)
    xa, lambda_a ,xb, lambda_b = obj.get_primal_dual_solutions(requires_grad=True)
    dist_plus = (xa[0] - xb[0])**2 + (xa[1] - xb[1])**2

    # f(x-h)
    cxa_perturb = cxa[:]
    cxa_perturb[0] = cxa_perturb[0] - PERTURBATION
    obj = MinDist2D(cxa_perturb, cxb, ra, rb)
    xa, __ ,xb, __ = obj.get_primal_dual_solutions(requires_grad=True)
    dist_minus = (xa[0] - xb[0])**2 + (xa[1] - xb[1])**2

    nabla_cxa = (- dist_2minus + 8*dist_plus - 8*dist_minus + dist_2plus)/(12*PERTURBATION)
    print(nabla_cxa)

def gradient_check_2d(plot=False):
    """
    More 2d tests to check validity of gradient calculation
    Sanity check for gradients
    - the effect of moving one shape in one axis should be equal to the effect of moving another shape but with a
    different sign. This indicates that when the shapes move in the same direction at the same speed, the relative
    motion between them is zero, which means the minimum distance should not change
    - Reducing the radius of a circle while keeping the location constant, the value of derivatives should get bigger
    since the rate of change of the surface becomes more 'intentse' (the rate of change of the curve of a smaller circle
    is more significant than the rate of change of a bigger circle)
    - Keeping the radius constant but moving two circle closer will result in the values decreasing since a small
    movement of the circle will result in a less drastic change in the point moving along the surface

    - These tests use SOS as the objective function which is an equivalent problem for what we are looking for
    """
    # Define centres and radii of SQ
    params = [
        # cxa, cya, cxb, cyb, aa, ba, ab, bb
        # [(1.0, 1.0), (-1.0, -1.0), (1.1, 1.1), (1.2, 1.2)],  # Big circle
        # [(1.0, 1.0), (-1.0, -1.0), (1.1, 1.1), (0.8, 0.8)],  # Smaller circle
        # [(1.0, 1.0), (-1.0, -1.0), (1.1, 1.1), (0.2, 0.2)],  # Even smaller circle
        [(0.2, 0.2), (-1.0, -1.0), (1.1, 1.1), (0.3, 0.3)],  # Circles close to each other
        [(1.0, 1.0), (0.2, 0.15), (0.3, 0.3), (0.8, 0.8)],   # Circles close
        [(1.0, -1.0), (-2.5, 0.15), (1.3, 0.4), (0.75, 0.2)], # Ellipse
        [(1.1, -0.8), (-1.0, 1.15), (0.27, 1.4), (0.75, 0.2)] # Ellipse
    ]
    circle_count = 0

    for ca, cb, ra, rb, in params:
        # Create problem
        obj = MinDist2D(ca, cb, ra, rb)

        # Obtain solution and gradients
        xa, lambda_a, xb, lambda_b = obj.get_primal_dual_solutions(requires_grad=True)
        print(xa, lambda_a, xb, lambda_b)
        analytic_tmp =  obj.sensitivity_analysis()
        tmp = obj.sensitivity_cvxpy()

        # Test that gradients from cvxpy and analytical are similar
        assert_almost_equal(np.array(analytic_tmp[:2]), tmp[0], decimal=5)
        assert_almost_equal(np.array(analytic_tmp[2:]), tmp[1], decimal=5)

        if plot:
            plt.figure()
            ax = plt.gca()
            plt.plot((xa[0], xb[0]), (xa[1], xb[1]), 'ro-')
            ax.add_patch(Ellipse(xy=ca, width=2 * ra[0], height=2 * ra[1]))
            ax.add_patch(Ellipse(xy=cb, width=2 * rb[0], height=2 * rb[1]))
            if circle_count < 3:
                ax.add_patch(Ellipse(xy=(-1.0, -1.0), width=2 * 1.1, height=2 * 1.1, alpha=0.2))
                circle_count += 1
            plt.text(*ca, np.round(analytic_tmp[:2], 2), fontsize=12)
            plt.text(*cb, np.round(analytic_tmp[2:], 2), fontsize=12)
            plt.axis('scaled')
            plt.show()

# ellipse_2d_handwritten_check(True)
# ellipse_2d_numerical_diff()
# gradient_check_2d(True)
###################################################### 3D tests ########################################################

def ellipse_test_transl(plot=False):
    """
    Uses eps=1 and quat=(1,0,0,0). Check distance with only translation
    These tests revealed that we needed to use abs on the numerator of the inside outside function
    """
    params = [
        # ca, cb, ra, rb, eps_a, eps_b
        # [(0, 0.0, 0), (1, 0, 0), (0.1, 0.2, 0.3), (0.25, 0.5, 0.15), (1, 1), (1, 1)], # T1
        # [(0, 0.4, 0), (1, 0, 0), (0.1, 0.2, 0.3), (0.25, 0.5, 0.15), (1, 1), (1, 1)], # T2
        [(0, 0.4, -0.2), (1, 0, 0), (0.1, 0.2, 0.3), (0.25, 0.5, 0.15), (1, 1), (1, 1)], # T3
        # [(-1.0, -3.8, 0.9), (-1.0, 0.6, -1.7),  (1, 1.2, 0.9), (1.25, 1.5, 1.15), (1, 1), (1, 1)],
        # [(-2, -0.5, -0.4), (-2, -0.4, 0.8), (0.2, 0.1, 0.4), (0.15, 0.5, 0.3), (1, 1), (1, 1)],
        # [(-1, -0.9, 0.9), (1, 0.5, -1.7), (1, 1.2, 0.9), (1.25, 1.5, 1.15), (0.5, 1), (1, 1)],
        # [(-1.0, -0.9, 0.9), (-1.0, 0.5, -1.7), (1, 1.2, 0.9), (1.25, 1.5, 1.15), (0.5, 1.0), (1.8, 2.0)],

        [(-1.0, -0.9, 0.9), (-1.0, 0.5, -1.7), (1, 1.2, 0.9), (1.25, 1.5, 1.15), (0.1, 1.0), (2.0, 2.0)],  # case1, Not works_a1
        # [(-1.0, -0.9, 0.9), (-1.0, 0.5, -1.7), (1, 1.2, 0.9), (1.25, 1.5, 1.15), (1.0, 1.0), (1.0, 1.0)],  # Encapsulating ellipsoids
        [(-1.0, -0.9, 0.9), (-1.0, 0.5, -1.7), (1, 1.2, 0.9), (1.25, 1.5, 1.15), (0.2, 1.0), (1.9, 1.9)],  # case2, eps_b less than two works but 0.2 and 0.1 for eps_a[0] does not work (smth to do with curvature??)
        # [(-1.0, -0.9, 0.9), (-1.0, 0.5, -1.7), (1, 1.2, 0.9), (1.25, 1.5, 1.15), (0.1, 1.0), (1.9, 1.9)],   # 0.1
        # [(-1.0, -0.9, 0.9), (-1.0, 0.5, -1.7), (1, 1.2, 0.9), (1.25, 1.5, 1.15), (0.25, 1.0), (1.9, 1.9)],  # i think works?

        [(-1.0, -0.9, 0.9), (-1.0, 0.5, -1.7), (1, 0.5, 0.9), (1.25, 1.5, 1.15), (0.25, 1.0), (0.1, 0.9)],  # case3, Not works_a2
        # [(-1.0, -0.9, 0.9), (-1.0, -0.5, -1.7), (1, 0.5, 0.9), (1.25, 1.5, 1.15), (0.25, 1.0), (0.1, 0.9)],  # Not works_a3
        # [(-1.0, 1.9, 0.9), (-1.0, 0.5, -1.7), (1, 0.5, 0.9), (1.25, 1.5, 1.15), (0.25, 1.0), (0.1, 0.9)],  # Shape on RHS is fine
        # [(-1.0, -0.9, 0.9), (-1.0, 0.5, -1.7), (1, 1.2, 0.9), (1.25, 1.5, 1.15), (1.0, 1.0), (0.9, 0.9)]  # Not works_a4
    ]

    for ca, cb, ra, rb, eps_a, eps_b in params:
        # Create sq object
        s1 = SuperquadricObject(*ra, *eps_a, pos=ca, quat=(1, 0, 0, 0))
        s2 = SuperquadricObject(*rb, *eps_b, pos=cb, quat=(1, 0, 0, 0))

        # Create optimisation problem
        optimiser = MinDist3DTransl(ca, cb, ra, rb, eps_a, eps_b, objective="NORM")
        xa, _, xb, _ = optimiser.get_primal_dual_solutions(requires_grad=False)
        xd, yd, zd = optimiser.sensitivity_analysis()
        print(xd, yd, zd)

        if plot:
            ax = plt.subplot(111, projection='3d')
            s1.plot_sq(ax, 'green')
            s2.plot_sq(ax, 'red')
            ax.plot((xa[0], xb[0]), (xa[1], xb[1]), (xa[2], xb[2]), 'ro-')
            plt.xlabel('x-axis')
            ax.axis('scaled')
            plt.show()

def ellipse_test_rot(plot=False):
    """
    Uses eps=1 and quat=(1,0,0,0). Check distance with only translation
    """
    # Define centres and radii of SQ
    params = [
        # ca, cb, ra, rb, eps_a, eps_b, qa, qb
        [(0, 0.4, -0.2), (1, 0, 0), (0.1, 0.2, 0.3), (0.25, 0.5, 0.15), (1, 1), (1, 1), (1.0, 0, 0, 0.0), (1, 0, 0, 0)],  # T3
        [(-1.0, -0.9, 0.9), (-1.0, 0.5, -1.7), (1, 1.2, 0.9), (1.25, 1.5, 1.15), (0.1, 1.0), (2.0, 2.0), (0.9238795, 0, 0, 0.3826834), (1, 0, 0, 0)], # case1, Not works_a1
        [(-1.0, -0.9, 0.9), (-1.0, 0.5, -1.7), (1, 1.2, 0.9), (1.25, 1.5, 1.15), (0.2, 1.0), (1.9, 1.9), (0.9076734, 0.0885213, 0.2432103, -0.3303661), (0.9238795, 0.0, 0.0, 0.3826834)], # case2, eps_b less than two works but 0.2 and 0.1 for eps_a[0] does not work (smth to do with curvature??)
        [(-1.0, -0.9, 0.9), (-1.0, 0.5, -1.7), (1, 0.5, 0.9), (1.25, 1.5, 1.15), (0.25, 1.0), (0.1, 0.9), (0.7717387, 0.5141426, 0.0270976, 0.3732862), (0.9924039, -0.0868241, -0.0075961, 0.0868241)] # case3, Not works_a2
    ]

    for ca, cb, ra, rb, eps_a, eps_b, qa, qb in params:
        # Create sq object
        s1 = SuperquadricObject(*ra, *eps_a, pos=ca, quat=qa)
        s2 = SuperquadricObject(*rb, *eps_b, pos=cb, quat=qb)

        # Create optimisation problem
        optimiser = MinDist3DRot(ca, cb, ra, rb, eps_a, eps_b, qa, qb, objective="NORM")
        xa, lagr_a, xb, lagr_b = optimiser.get_primal_dual_solutions(requires_grad=False)
        xd, yd, zd, q0d, q1d, q2d, q3d = optimiser.sensitivity_analysis()

        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            s1.plot_sq(ax, 'green')
            s2.plot_sq(ax, 'red')
            ax.plot((xa[0], xb[0]), (xa[1], xb[1]), (xa[2], xb[2]), 'ro-')
            plt.xlabel('x-axis')
            plt.ylabel('y-axis')
            ax.text2D(0.05, 0.95, f" {round(xd, 3)}, {round(yd, 3)}, {round(zd, 3)}   "
                                  f" {round(q0d, 3)}, {round(q1d, 3)}, {round(q2d, 3)}, {round(q3d, 3)}",transform=ax.transAxes)
            plt.show()

def ellipse_test_transl_grad(plot=False):
    """
    Uses eps=1 and quat=(1,0,0,0). Check distance with only translation
    These tests revealed that we needed to use abs on the numerator of the inside outside function
    """
    params = [
        # ca, cb, ra, rb, eps_a, eps_b
        [(0.2, 0.2, 0.001), (-1.0, -1.0, -0.001), (1.1, 1.1, 0.001), (0.3, 0.3, 0.001), (1.0, 1.0), (1.0, 1.0)],  # Circles close to each other
        [(1.1, -0.8, 0.01), (-1.0, 1.15, -0.01), (0.27, 1.4, 0.1), (0.75, 0.2, 0.1), (1.0, 1.0), (1.0, 1.0)], # Ellipse

        [(0, 0.4, -0.2), (1, 0, 0), (0.1, 0.2, 0.3), (0.25, 0.5, 0.15), (1, 1), (1, 1)], # T3
        [(-1.0, -0.9, 0.9), (-1.0, 0.5, -1.7), (1, 1.2, 0.9), (1.25, 1.5, 1.15), (0.1, 1.0), (2.0, 2.0)],  # case1, Not works_a1
        [(-1.0, -0.9, 0.9), (-1.0, 0.5, -1.7), (1, 1.2, 0.9), (1.25, 1.5, 1.15), (0.2, 1.0), (1.9, 1.9)],  # case2, eps_b less than two works but 0.2 and 0.1 for eps_a[0] does not work (smth to do with curvature??)
        [(-1.0, -0.9, 0.9), (-1.0, 0.5, -1.7), (1, 0.5, 0.9), (1.25, 1.5, 1.15), (0.25, 1.0), (0.1, 0.9)],  # case3, Not works_a2
        [(-1.5, -0.1, 0.9), (-1.0, 0.5, -1.7), (1, 0.5, 0.9), (1.25, 1.5, 1.15), (0.25, 1.0), (0.1, 0.9)]
    ]

    for ca, cb, ra, rb, eps_a, eps_b in params:
        # Create sq object
        s1 = SuperquadricObject(*ra, *eps_a, pos=ca, quat=(1, 0, 0, 0))
        s2 = SuperquadricObject(*rb, *eps_b, pos=cb, quat=(1, 0, 0, 0))

        # Create optimisation problem
        optimiser = MinDist3DTransl(ca, cb, ra, rb, eps_a, eps_b, objective="NORM")
        xa, lambda_a, xb, lambda_b = optimiser.get_primal_dual_solutions(requires_grad=False)
        xd, yd, zd = optimiser.sensitivity_analysis()

        if plot:
            ax = plt.subplot(111, projection='3d')
            s1.plot_sq(ax, 'green')
            s2.plot_sq(ax, 'red')
            ax.plot((xa[0], xb[0]), (xa[1], xb[1]), (xa[2], xb[2]), 'ro-')
            plt.xlabel('x-axis')
            plt.ylabel('y-axis')
            ax.text2D(0.05, 0.95, f"{round(xd, 3)}, {round(yd, 3)}, {round(zd, 3)}", transform=ax.transAxes)
            ax.axis('scaled')
            plt.show()


# ellipse_test_transl(1)
ellipse_test_rot(1)
# ellipse_test_transl_grad(0)

###################################################### Sim tests ########################################################

def sim_tester(plot=False):
    """
    Uses eps=1 and quat=(1,0,0,0). Check distance with only translation
    These tests revealed that we needed to use abs on the numerator of the inside outside function
    """
    Ra = (0.6, 0.2, 0.1)  # Circle radius
    Rb = (0.3, 0.6, 0.1)  # Radius around the EE
    eps_a = (0.1, 1)
    eps_b = (1, 1)

    # Initial and target shape positions
    xa_init = (4.0, -0.0, 0.1)  # Obstacle position
    xb_init = (-1.0, 0.6, -0.01)  # Obstacle position
    xa_tgt = (-4.0, 0, 0.1)  # Final position

    params = [
        # ca, cb, ra, rb, eps_a, eps_b
        [xa_tgt, xb_init, Ra, Rb, eps_a, eps_b]
    ]

    for ca, cb, ra, rb, eps_a, eps_b in params:
        # Create sq object
        s1 = SuperquadricObject(*ra, *eps_a, pos=ca, quat=(1, 0, 0, 0))
        s2 = SuperquadricObject(*rb, *eps_b, pos=cb, quat=(1, 0, 0, 0))

        # Create optimisation problem
        optimiser = MinDist3DTransl(ca, cb, ra, rb, eps_a, eps_b, objective="NORM")
        xa, lambda_a, xb, lambda_b = optimiser.get_primal_dual_solutions(requires_grad=False)
        xd, yd, zd = optimiser.sensitivity_analysis()

        if plot:
            ax = plt.subplot(111, projection='3d')
            s1.plot_sq(ax, 'green')
            s2.plot_sq(ax, 'red')
            ax.plot((xa[0], xb[0]), (xa[1], xb[1]), (xa[2], xb[2]), 'ro-')
            plt.xlabel('x-axis')
            plt.ylabel('y-axis')
            ax.text2D(0.05, 0.95, f"{round(xd, 3)}, {round(yd, 3)}, {round(zd, 3)}", transform=ax.transAxes)
            ax.axis('scaled')
            plt.show()

# sim_tester(1)