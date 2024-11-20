"""
TODO: Vectorise operations to improve speed. Just need to ensure the vectorisation performs what I want (element-wise or
matrix)
"""
import math
import time
import cvxpy as cp
import numpy as np

class min_dist_2D:
    """
    Class for calculating the minimum distance between two ellipse type shapes
    """
    def __init__(self, cxa, cya, cxb, cyb, aa, ba, ab, bb):
        # Decision variables
        self.xa = cp.Variable((2,1))  # Point on SQ a
        self.xb = cp.Variable((2,1))  # Point on SQ b

        # Position vectors as params of SQs
        self.cxa = cp.Parameter((2, 1), value=np.array([[cxa, cya]]).transpose())
        self.cxb = cp.Parameter((2, 1), value=np.array([[cxb, cyb]]).transpose())

        # Radii
        self.aa = aa
        self.ba = ba
        self.ab = ab
        self.bb = bb

        # Objective
        objective = cp.Minimize(cp.sum_squares(self.xa - self.xb))

        # Constraints, variables must lie on the surface of the sq
        self.constraints = [
            ((self.xa[0, 0] - self.cxa[0, 0]) / aa)**2 + ((self.xa[1, 0] - self.cxa[1, 0]) / ba)**2 -1 <= 0,
            ((self.xb[0, 0] - self.cxb[0, 0]) / ab)**2 + ((self.xb[1, 0] - self.cxb[1, 0]) / bb)**2 -1 <= 0
        ]

        # Problem
        self.prob = cp.Problem(objective, self.constraints)

    def get_primal_dual_solutions(self):
        """
        Get the points where the minimum distance occurs and the corresponding dual variables
        Returns: (points on xa, dual value for xa, points on xb, dual value for xb)
        """
        self.prob.solve(solver='SCS', requires_grad=True)
        # self.prob.solve(solver='CLARABEL')
        return self.xa.value.squeeze(), self.constraints[0].dual_value, self.xb.value.squeeze(), self.constraints[1].dual_value


    def set_params(self, cxa, cya, cxb, cyb):
        """
        Set the new centre positions of each shape
        """
        self.cxa.value = np.array([[cxa, cya]]).transpose()
        self.cxb.value = np.array([[cxb, cyb]]).transpose()

    def sensitivity_analysis(self):
        """
        Return the gradient of Lagrangian wrt problem parameters
        Returns: [nabla_cxa_L, nabla_cya_L, nabla_cya_L, nabla_cyb_L]
        """
        return [-2*self.constraints[0].dual_value*((self.xa.value[0, 0] - self.cxa.value[0, 0]) / self.aa**2),
                -2*self.constraints[0].dual_value*((self.xa.value[1, 0] - self.cxa.value[1, 0]) / self.ba**2),
                -2*self.constraints[1].dual_value*((self.xb.value[0, 0] - self.cxb.value[0, 0]) / self.ab**2),
                -2*self.constraints[1].dual_value*((self.xb.value[1, 0] - self.cxb.value[1, 0]) / self.bb**2)]

    def sensitivity_cvxpy(self):
        """
        TODO: understand if I need backward or derivative
        Sensitivity analysis through cvxpy
        Returns: [nabla_cxa_L, nabla_cya_L, nabla_cya_L, nabla_cyb_L]
        """
        self.cxa.delta = 1e-5
        self.cxb.delta = 1e-5
        self.prob.derivative()
        print(self.xa.delta)
