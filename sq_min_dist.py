# -*- coding: utf-8 -*-
"""
cyipopt: Python wrapper for the Ipopt optimization package, written in Cython.

Copyright (C) 2012-2015 Amit Aides
Copyright (C) 2015-2017 Matthias Kümmerer
Copyright (C) 2017-2024 cyipopt developers

License: EPL 2.0
"""
import sys
import time

# Based on matlab code by Peter Carbonetto.
# Had to build cyipopt==1.4.0 from source since 1.5.0 does not support python3.8. Using pip didn't work

import numpy as np
import cyipopt

class hs071:

    def __init__(self):
        self.cxa, self.cya, self.cza, self.cxb, self.cyb, self.czb = [0.3, 1, 0,
                                                                      2.5, -0.1, 0]

        self.aa, self.ba, self.ca, self.ab, self.bb, self.cb = [0.2, 0.5, 0.1,
                                                                0.5, 0.4, 0.1]

        self.e1a, self.e2a, self.e1b, self.e2b = [1.0, 1.0,
                                                  1.0, 1.0]

    def objective(self, x):
        """
        Objective Function
        Args:
            x: Tuple | the two points that is within or on the surface of each sq

        Returns: Float | objective function value, square of the distance of the two points

        """
        return (x[0] - x[3])**2 + (x[1] - x[4])**2 + (x[2] - x[5])**2

    def gradient(self, x):
        #
        # The callback for calculating the gradient
        #
        return np.array([
            2*(x[0] - x[3]),
            2*(x[1] - x[4]),
            2*(x[2] - x[5]),
            -2*(x[0] - x[3]),
            -2*(x[1] - x[4]),
            -2*(x[2] - x[5])
        ])

    def constraints(self, x):
        #
        # The callback for calculating the constraints
        #
        return np.array((
            ((x[0] - self.cxa)**2) / self.aa**2 + ((x[1] - self.cya)**2) / self.ba**2 + ((x[2] - self.cza)**2) / self.ca**2,
            ((x[3] - self.cxb)**2) / self.ab**2 + ((x[4] - self.cyb)**2) / self.bb**2 + ((x[5] - self.czb)**2) / self.cb**2
        ))

    def jacobian(self, x):
        #
        # The callback for calculating the Jacobian of the constraints
        #
        return np.array((
            (2*(x[0] - self.cxa)) / self.aa**2, (2*(x[1] - self.cya)) / self.ba**2, (2*(x[2] - self.cza)) / self.ca**2,
            0, 0, 0, 0, 0, 0,
            (2*(x[3] - self.cxb)) / self.ab**2, (2*(x[4] - self.cyb)) / self.bb**2, (2*(x[5] - self.czb)) / self.cb**2
        ))

    # def hessianstructure(self):
    #     #
    #     # The structure of the Hessian
    #     # Note:
    #     # The default hessian structure is of a lower triangular matrix. Therefore
    #     # this function is redundant. I include it as an example for structure
    #     # callback.
    #     #
    #
    #     return np.nonzero(np.tril(np.ones((4, 4))))

    # def hessian(self, x, lagrange, obj_factor):
    #     #
    #     # The callback for calculating the Hessian
    #     #
    #     H = obj_factor*np.array((
    #             (2*x[3], 0, 0, 0),
    #             (x[3],   0, 0, 0),
    #             (x[3],   0, 0, 0),
    #             (2*x[0]+x[1]+x[2], x[0], x[0], 0)))
    #
    #     H += lagrange[0]*np.array((
    #             (0, 0, 0, 0),
    #             (x[2]*x[3], 0, 0, 0),
    #             (x[1]*x[3], x[0]*x[3], 0, 0),
    #             (x[1]*x[2], x[0]*x[2], x[0]*x[1], 0)))
    #
    #     H += lagrange[1]*2*np.eye(4)
    #
    #     row, col = self.hessianstructure()
    #
    #     return H[row, col]

    # def intermediate(
    #         self,
    #         alg_mod,
    #         iter_count,
    #         obj_value,
    #         inf_pr,
    #         inf_du,
    #         mu,
    #         d_norm,
    #         regularization_size,
    #         alpha_du,
    #         alpha_pr,
    #         ls_trials
    #         ):

        #
        # Example for the use of the intermediate callback.
        #
        # print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))


def main():
    a = time.time()
    #
    # Define the problem
    #
    x0 = [0.1, 0.1, 0.1, 0.1, 1.1, 0.1]

    """
    As requested I have started the refund process for order number AP1554979310. The amount of AUD 75.99 will be
    returned to the credit card charged. Please allow 5-7 working (business) days to process and it will show up in your
    next billing cycle (within 10 working days depending on your bank). Please, take a note of your refund confirmation
    number RMA4515195.
    89117282
    """

    # Convert sq position from local to world frame, taking into account orientation of the sq

    # Upper and lower bound of the decision variables
    lb = [-10.0, -10.0, -10.0, -10.0, -10.0, -10.0]
    ub = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]

    # Upper and lower bound of the constraints
    cl = [1.0e-8, 1.0e-8]
    cu = [1.0, 1.0]

    nlp = cyipopt.Problem(
        n=len(x0),
        m=len(cl),
        problem_obj=hs071(),
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu,
        )
    # https://github.com/casadi/casadi/issues/2032
    # https://stackoverflow.com/questions/72944883/set-env-variable-for-hsl-so-ipopt-can-use-the-solver-ma57
    # nlp.add_option("linear_solver", "ma27")

    #
    # Set solver options
    #
    #nlp.addOption('derivative_test', 'second-order')
    nlp.add_option('mu_strategy', 'adaptive')
    nlp.add_option('tol', 1e-6)

    #
    # Scale the problem (Just for demonstration purposes)
    #
    # nlp.set_problem_scaling(
    #     obj_scaling=2,
    #     x_scaling=[1, 1, 1, 1, 1, 1]
    #     )
    # nlp.add_option('nlp_scaling_method', 'user-scaling')

    #
    # Solve the problem
    #
    x, info = nlp.solve(x0)

    print("Solution of the primal variables: x=%s\n" % repr(x))

    print("Solution of the dual variables: lambda=%s\n" % repr(info['mult_g']))

    print("Objective=%s\n" % repr(info['obj_val']))
    print(time.time() - a)


if __name__ == '__main__':
    main()