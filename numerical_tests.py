"""
Sanity checks
"""
from ctypes.wintypes import PRECT

import numpy as np
from numpy.ma.testutils import assert_almost_equal
from scipy.stats import alpha

from min_dists import min_dist_2D
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

def ellipse_2d_handwritten_check(plot=False):
    """
    Checking we coded things correctly by validating code results with handwritten result and optimal points from ipopt
    """
    # Define centres of SQ
    cxa, cya, cxb, cyb = 0.3, 1, 2.5, -0.1
    aa, ab, ba, bb = 0.2, 0.5, 0.5, 0.4
    obj = min_dist_2D(cxa, cya, cxb, cyb, aa, ab, ba, bb)
    xa, lambda_a ,xb, lambda_b = obj.get_primal_dual_solutions()
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
        print(xa)
        plt.figure()
        ax = plt.gca()
        plt.plot((xa[0], xb[0]), (xa[1], xb[1]), 'ro-')
        ax.add_patch(Ellipse(xy=(cxa, cya), width=2*aa, height=2*ba))
        ax.add_patch(Ellipse(xy=(cxb, cyb), width=2*ab, height=2*bb))
        plt.text(cxa, cya, np.round(analytic_tmp[:2], 2), fontsize=12)
        plt.text(cxb, cyb, np.round(analytic_tmp[2:], 2), fontsize=12)
        plt.axis('scaled')
        plt.show()

def ellipse_2d_numerical_diff():
    """
    https://en.wikipedia.org/wiki/Numerical_differentiation
    Use two-point formula to check the derivatives we compute is correct
    """
    PERTURBATION = 1e-4
    # Define centres of SQ
    cxa, cya, cxb, cyb = 0.3, 1, 2.5, -0.1

    # Define radii
    aa, ba, ab, bb = 0.2, 0.5, 0.5, 0.4

    # Text for nabla_cxa
    obj = min_dist_2D(cxa-PERTURBATION, cya, cxb, cyb, aa, ba, ab, bb)
    xa, __ ,xb, __ = obj.get_primal_dual_solutions()
    dist_minus = (xa[0] - xb[0])**2 + (xa[1] - xb[1])**2

    obj.set_params(cxa+PERTURBATION, cya, cxb, cyb)
    xa, lambda_a ,xb, lambda_b = obj.get_primal_dual_solutions()
    dist_plus = (xa[0] - xb[0])**2 + (xa[1] - xb[1])**2

    nabla_cxa = (dist_plus - dist_minus)/(2*PERTURBATION)

    # Text for nabla_cya
    obj = min_dist_2D(cxa, cya - PERTURBATION, cxb, cyb, aa, ba, ab, bb)
    xa, __, xb, __ = obj.get_primal_dual_solutions()
    dist_minus = (xa[0] - xb[0]) ** 2 + (xa[1] - xb[1]) ** 2

    obj.set_params(cxa, cya + PERTURBATION, cxb, cyb)
    xa, lambda_a, xb, lambda_b = obj.get_primal_dual_solutions()
    dist_plus = (xa[0] - xb[0]) ** 2 + (xa[1] - xb[1]) ** 2

    nabla_cya = (dist_plus - dist_minus) / (2 * PERTURBATION)

    # Text for nabla_cxb
    obj = min_dist_2D(cxa, cya, cxb - PERTURBATION, cyb, aa, ba, ab, bb)
    xa, __, xb, __ = obj.get_primal_dual_solutions()
    dist_minus = (xa[0] - xb[0]) ** 2 + (xa[1] - xb[1]) ** 2

    obj.set_params(cxa, cya, cxb + PERTURBATION, cyb)
    xa, lambda_a, xb, lambda_b = obj.get_primal_dual_solutions()
    dist_plus = (xa[0] - xb[0]) ** 2 + (xa[1] - xb[1]) ** 2

    nabla_cxb = (dist_plus - dist_minus) / (2 * PERTURBATION)

    # Text for nabla_cyb
    obj = min_dist_2D(cxa, cya, cxb, cyb - PERTURBATION, aa, ba, ab, bb)
    xa, __, xb, __ = obj.get_primal_dual_solutions()
    dist_minus = (xa[0] - xb[0]) ** 2 + (xa[1] - xb[1]) ** 2

    obj.set_params(cxa, cya, cxb, cyb + PERTURBATION)
    xa, lambda_a, xb, lambda_b = obj.get_primal_dual_solutions()
    dist_plus = (xa[0] - xb[0]) ** 2 + (xa[1] - xb[1]) ** 2

    nabla_cyb = (dist_plus - dist_minus) / (2 * PERTURBATION)

    obj = min_dist_2D(cxa, cya, cxb, cyb, 0.2, 0.5, 0.5, 0.4)
    obj.get_primal_dual_solutions()

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
    """
    # Define centres and radii of SQ
    params = [
        # cxa, cya, cxb, cyb, aa, ba, ab, bb
        [1.0, 1.0, -1.0, -1.0, 1.1, 1.1, 1.2, 1.2],  # Big circle
        [1.0, 1.0, -1.0, -1.0, 1.1, 1.1, 0.8, 0.8],  # Smaller circle
        [1.0, 1.0, -1.0, -1.0, 1.1, 1.1, 0.2, 0.2],  # Even smaller circle
        [0.2, 0.2, -1.0, -1.0, 1.1, 1.1, 0.3, 0.3],  # Circles close to each other
        [1.0, 1.0, 0.2, 0.15, 0.3, 0.3, 0.8, 0.8],   # Circles close
        [1.0, -1.0, -2.5, 0.15, 1.3, 0.4, 0.75, 0.2], # Ellipse
        [1.1, -0.8, -1.0, 1.15, 0.27, 1.4, 0.75, 0.2] # Ellipse
    ]
    circle_count = 0

    for cxa, cya, cxb, cyb, aa, ba, ab, bb in params:
        # Create problem
        obj = min_dist_2D(cxa, cya, cxb, cyb, aa, ba, ab, bb)

        # Obtain solution and gradients
        xa, lambda_a, xb, lambda_b = obj.get_primal_dual_solutions()
        analytic_tmp =  obj.sensitivity_analysis()
        tmp = obj.sensitivity_cvxpy()

        # Test that gradients from cvxpy and analytical are similar
        assert_almost_equal(np.array(analytic_tmp[:2]), tmp[0], decimal=5)
        assert_almost_equal(np.array(analytic_tmp[2:]), tmp[1], decimal=5)

        if plot:
            print(xa)
            plt.figure()
            ax = plt.gca()
            plt.plot((xa[0], xb[0]), (xa[1], xb[1]), 'ro-')
            ax.add_patch(Ellipse(xy=(cxa, cya), width=2 * aa, height=2 * ba))
            ax.add_patch(Ellipse(xy=(cxb, cyb), width=2 * ab, height=2 * bb))
            if circle_count < 3:
                ax.add_patch(Ellipse(xy=(-1.0, -1.0), width=2 * 1.1, height=2 * 1.1, alpha=0.2))
                circle_count += 1
            plt.text(cxa, cya, np.round(analytic_tmp[:2], 2), fontsize=12)
            plt.text(cxb, cyb, np.round(analytic_tmp[2:], 2), fontsize=12)
            plt.axis('scaled')
            plt.show()

# ellipse_2d_handwritten_check(True)
# ellipse_2d_numerical_diff()
# gradient_check_2d(True)