"""
Sanity checks
"""
import numpy as np
from numpy.ma.testutils import assert_almost_equal

from min_dists import min_dist_2D

PERTURBATION = 1e-4

def ellipse_2d_handwritten_check():
    """
    Checking we coded things correctly by validating code results with handwritten result and optimal points from ipopt
    """
    # Define centres of SQ
    cxa, cya, cxb, cyb = 0.3, 1, 2.5, -0.1

    obj = min_dist_2D(cxa, cya, cxb, cyb, 0.2, 0.5, 0.5, 0.4)
    xa, lambda_a ,xb, lambda_b = obj.get_primal_dual_solutions()

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


def ellipse_2d_numerical_diff():
    """
    https://en.wikipedia.org/wiki/Numerical_differentiation
    Use two-point formula to check the derivatives we compute is correct
    """
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

# ellipse_2d_handwritten_check()
# ellipse_2d_numerical_diff()