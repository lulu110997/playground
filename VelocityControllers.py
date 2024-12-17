import math
import sys
import time
import cvxpy as cp
import numpy as np
import sympy

class VelocityController():
    """
    Velocity controller using cvxpy. This class formulates a first order QP problem to obtain an optimal velocity from
    CBF constraints.
    """

    def __init__(self, ndim=3, nconst=1):
        """
        Args:
            ndim: int | number of dimensions we are considering. 2 for xy velocity, 3 for xyz velocity and 6 for xyz
            translational and rotational velocities
            nconst: int | number of inequality constraints
        """
        # Velocities
        self.xd = cp.Variable(ndim)
        self.xd_tgt = cp.Parameter(ndim)

        # CBF constraints in matrix form
        self.G = cp.Parameter((nconst, ndim))
        self.h = cp.Parameter(nconst)

        self.objective = cp.Minimize(cp.norm(self.xd - self.xd_tgt))
        self.constraints = [self.G@self.xd - self.h <= 0]

        self.prob = cp.Problem(self.objective, self.constraints)

    def get_solution(self):
        self.prob.solve(solver='CLARABEL')
        return self.xd.value.squeeze()

    def set_param(self, xd_tgt, G_matr, h_matr):
        self.xd_tgt.value = xd_tgt
        self.G.value = G_matr.reshape(1,3)
        h_matr = np.array([h_matr])
        self.h.value = h_matr
