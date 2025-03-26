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

    def __init__(self, ndim=3, nconst=1, W=None):
        """
        Args:
            ndim: int | number of dimensions we are considering. 2 for xy velocity, 3 for xyz velocity and 6 for xyz
            translational and rotational velocities
            nconst: int | number of inequality constraints
        """
        self.ndim = ndim
        self.nconst = nconst

        if W is None:
            W = np.eye(self.ndim)

        # Velocities
        self.xd = cp.Variable(ndim)
        self.xd_tgt = cp.Parameter(ndim)

        # CBF constraints in matrix form
        self.G = cp.Parameter((nconst, ndim))
        self.h = cp.Parameter(nconst)
        self.objective = cp.Minimize(cp.norm((W**0.5)@(self.xd - self.xd_tgt)))
        self.constraints = [self.G@self.xd - self.h <= 0]

        self.prob = cp.Problem(self.objective, self.constraints)

    def get_solution(self):
        self.prob.solve(solver='CLARABEL')
        return self.xd.value.squeeze()

    def set_param(self, xd_tgt, G_matr, h_matr, q=None):
        if self.ndim == 3:
            self.xd_tgt.value = xd_tgt
            self.G.value = G_matr.reshape(self.nconst, self.ndim)
            h_matr = np.array([h_matr])
            self.h.value = h_matr
        elif self.ndim == 6:
            self.xd_tgt.value = xd_tgt.squeeze()  # vx,vy,vz,wx,wy,wz
            Q = 0.5*np.array([
                    [-q.vec[1], -q.vec[2], -q.vec[3]],
                    [q.vec[0], -q.vec[3], q.vec[2]],
                    [q.vec[3], q.vec[0], -q.vec[1]],
                    [-q.vec[2], q.vec[1], q.vec[0]],
                ])
            # quat_d = G_matr[3:] @ Q  # xd,yd,zd,qwd,qxdy,qyd,qzd
            self.G.value = np.hstack((G_matr[:3], G_matr[3:]@Q)).reshape(self.nconst, self.ndim)  # xd,yd,zd,qwd,qxdy,qyd,qzd
            # h_matr = np.array([h_matr])
            self.h.value = np.array([h_matr])