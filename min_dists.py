"""
TODO: Vectorise operations to improve speed. Just need to ensure the vectorisation performs what I want (element-wise or
matrix)
"""
import math
import time
import cvxpy as cp
import numpy as np

class MinDist2D:
    """
    Class for calculating the minimum distance between two ellipse type shapes
    """
    def __init__(self, cxa, cya, cxb, cyb, aa, ba, ab, bb, objective="SOS"):
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
        if objective == "SOS":
            self.objective = cp.Minimize(cp.sum_squares(self.xa - self.xb))
        else:
            self.objective = cp.Minimize(cp.norm(self.xa - self.xb))

        # Constraints, variables must lie on the surface of the sq
        self.constraints = [
            ((self.xa[0, 0] - self.cxa[0, 0]) / aa)**2 + ((self.xa[1, 0] - self.cxa[1, 0]) / ba)**2 -1 <= 0,
            ((self.xb[0, 0] - self.cxb[0, 0]) / ab)**2 + ((self.xb[1, 0] - self.cxb[1, 0]) / bb)**2 -1 <= 0
        ]

        # Problem
        self.prob = cp.Problem(self.objective, self.constraints)

    def get_primal_dual_solutions(self):
        """
        Get the points where the minimum distance occurs and the corresponding dual variables
        Returns: tuple(np.array, float, np.array, float) | (points on xa, dual value for xa,
                                                            points on xb, dual value for xb)
        """
        self.prob.solve(solver='SCS', requires_grad=True)
        # self.prob.solve(solver='CLARABEL')
        return self.xa.value.squeeze(), self.constraints[0].dual_value, self.xb.value.squeeze(), self.constraints[1].dual_value

    def get_optimal_value(self):
        """
        Obtain objective function given after the optimal solution has been found
        Returns: float | Square of the distance function given the optimal solutions
        """
        return self.objective.value


    def set_params(self, cxa, cya, cxb, cyb):
        """
        Set the new centre positions of each shape
        Args:
            cxa: float | x position vector of shape a
            cya: float | y position vector of shape a
            cxb: float | x position vector of shape b
            cyb: float | y position vector of shape b
        """
        self.cxa.value = np.array([[cxa, cya]]).transpose()
        self.cxb.value = np.array([[cxb, cyb]]).transpose()

    def sensitivity_analysis(self):
        """
        Return the gradient of Lagrangian wrt problem parameters
        Returns: list(float..)  | [nabla_cxa_L, nabla_cya_L, nabla_cya_L, nabla_cyb_L]
        """
        return [-2*self.constraints[0].dual_value*((self.xa.value[0, 0] - self.cxa.value[0, 0]) / self.aa**2),
                -2*self.constraints[0].dual_value*((self.xa.value[1, 0] - self.cxa.value[1, 0]) / self.ba**2),
                -2*self.constraints[1].dual_value*((self.xb.value[0, 0] - self.cxb.value[0, 0]) / self.ab**2),
                -2*self.constraints[1].dual_value*((self.xb.value[1, 0] - self.cxb.value[1, 0]) / self.bb**2)]

    def sensitivity_cvxpy(self):
        """
        Sensitivity analysis through cvxpy
        Returns: tuple(np.array, np.array) | [arr(nabla_cxa_L, nabla_cya_L), arr(nabla_cya_L, nabla_cyb_L)]

        More generally, the backward method can be used to compute the gradient of a scalar-valued function f of the
        optimal variables, with respect to the parameters. If x(p) denotes the optimal value of the variable
        (which might be a vector or a matrix) for a particular value of the parameter p and f(x(p)) is a scalar, then
        backward can be used to compute the gradient of f with respect to p. Let x* = x(p), and say the derivative of f
        with respect to x* is dx. To compute the derivative of f with respect to p, before calling problem.backward(),
        just set x.gradient = dx.

        The backward method can be powerful when combined with software for automatic differentiation. We recommend the
        software package CVXPY Layers, which provides differentiable PyTorch and TensorFlow wrappers for CVXPY problems.
        """
        delta_x = 2*(self.xa - self.xb)  # Gradient of objective function wrt decision variables
        self.xa.gradient = delta_x.value
        self.xb.gradient = -delta_x.value
        self.prob.backward()
        return self.cxa.gradient.squeeze(), self.cxb.gradient.squeeze()

class MinDist3D_delrot:
    """
    Class for calculating the minimum distance between two ellipse type shapes
    """
    def __init__(self, ca, cb, ra, rb, eps_a, eps_b, qa, qb, objective="SOS"):
        """
        Args:
            ca, cb: Tuple[float] | Represents the centre position (xyz) of each superquadric in the world frame
            ra, rb: Tuple[float] | Represents the radii (abc) of each superquadric
            eps_a, eps_b: Tuple[float] | Represents the roundness parameter (eps1, eps2) of each superquadric
            qa, qb: Tuple[float] | Represents the orientation of each superquadric in the world frame as quaternions (wxyz)
            objective: String | The objective function for the minimisation problem. Defaults to sum of squares (SOS)
        """
        # Decision variables
        self.xa = cp.Variable(3)  # Point on SQ a
        self.xb = cp.Variable(3)  # Point on SQ b

        # Position vectors and orientation as params of SQs
        self.ca = cp.Parameter(3, value=np.array(ca))
        self.cb = cp.Parameter(3, value=np.array(cb))
        self.qa = cp.Parameter(4, value=np.array(qa))
        self.qb = cp.Parameter(4, value=np.array(qb))

        # Radii
        self.ra = ra
        self.rb = rb

        # Roundness parameters
        self.eps_a = eps_a
        self.eps_b = eps_b

        # Objective
        if objective == "SOS":
            self.objective = cp.Minimize(cp.sum_squares(self.xa - self.xb))
        else:
            self.objective = cp.Minimize(cp.norm(self.xa - self.xb))

        #  Transform points on each sq
        xrot_a = (  self.xa[0] * (2*(self.qa[0]*self.qa[0] + self.qa[1]*self.qa[1]) - 1)
                  + self.xa[1] *  2*(self.qa[1]*self.qa[2] - self.qa[0]*self.qa[3])
                  + self.xa[2] *  2*(self.qa[1]*self.qa[3] + self.qa[0]*self.qa[2]))

        yrot_a = (  self.xa[0] *  2*(self.qa[1]*self.qa[2] + self.qa[0]*self.qa[3])
                  + self.xa[1] * (2*(self.qa[0]*self.qa[0] + self.qa[2]*self.qa[2]) - 1)
                  + self.xa[2] *  2*(self.qa[2]*self.qa[3] - self.qa[0]*self.qa[1]))

        zrot_a = (  self.xa[0] *  2*(self.qa[1]*self.qa[3] - self.qa[0]*self.qa[2])
                  + self.xa[1] *  2*(self.qa[2]*self.qa[3] + self.qa[0]*self.qa[1])
                  + self.xa[2] * (2*(self.qa[0]*self.qa[0] + self.qa[3]*self.qa[3])-1))

        xrot_b = (  self.xb[0] * (2*(self.qb[0]*self.qb[0] + self.qb[1]*self.qb[1]) - 1)
                  + self.xb[1] *  2*(self.qb[1]*self.qb[2] - self.qb[0]*self.qb[3])
                  + self.xb[2] *  2*(self.qb[1]*self.qb[3] + self.qb[0]*self.qb[2]))

        yrot_b = (  self.xb[0] *  2*(self.qb[1]*self.qb[2] + self.qb[0]*self.qb[3])
                  + self.xb[1] * (2*(self.qb[0]*self.qb[0] + self.qb[2]*self.qb[2]) - 1)
                  + self.xb[2] *  2*(self.qb[2]*self.qb[3] - self.qb[0]*self.qb[1]))

        zrot_b = (  self.xb[0] *  2*(self.qb[1]*self.qb[3] - self.qb[0]*self.qb[2])
                  + self.xb[1] *  2*(self.qb[2]*self.qb[3] + self.qb[0]*self.qb[1])
                  + self.xb[2] * (2*(self.qb[0]*self.qb[0] + self.qb[3]*self.qb[3])-1))


        # Constraints, variables must lie on the surface of the sq
        xa_w = (((xrot_a - self.ca[0]) / self.ra[0])**(2/self.eps_a[1]))
        ya_w = (((yrot_a - self.ca[1]) / self.ra[1])**(2/self.eps_a[1]))
        za_w = (((zrot_a - self.ca[2]) / self.ra[2])**(2/self.eps_a[0]))
        xb_w = (((xrot_b - self.cb[0]) / self.rb[0])**(2/self.eps_b[1]))
        yb_w = (((yrot_b - self.cb[1]) / self.rb[1])**(2/self.eps_b[1]))
        zb_w = (((zrot_b - self.cb[2]) / self.rb[2])**(2/self.eps_b[0]))
        self.constraints = [
            ((xa_w + ya_w)**(self.eps_a[1]/self.eps_a[0])) + za_w -1 <= 0,
            ((xb_w + yb_w)**(self.eps_b[1]/self.eps_b[0])) + zb_w -1 <= 0
        ]

        # Problem
        self.prob = cp.Problem(self.objective, self.constraints)
        # assert self.prob.is_dcp()

    def get_primal_dual_solutions(self):
        """
        Get the points where the minimum distance occurs and the corresponding dual variables
        Returns: tuple(np.array, float, np.array, float) | (points on xa, dual value for xa,
                                                            points on xb, dual value for xb)
        """
        # self.prob.solve(solver='SCS', requires_grad=True)
        # ['CLARABEL', 'CVXOPT', 'DAQP', 'DIFFCP', 'ECOS', 'ECOS_BB', 'GLPK', 'GLPK_MI', 'OSQP', 'PIQP', 'PROXQP', 'SCIPY', 'SCS']
        ret_val = self.prob.solve(solver='CLARABEL')
        if not isinstance(ret_val, float):
            print(ret_val)
            raise "problem cannot be solved"
        return self.xa.value.squeeze(), self.constraints[0].dual_value, self.xb.value.squeeze(), self.constraints[1].dual_value

    def get_optimal_value(self):
        """
        Obtain objective function given after the optimal solution has been found
        Returns: float | Square of the distance function given the optimal solutions
        """
        return self.objective.value


    def set_params(self, ca, cb, qa, qb):
        """
        For each shape, set a new centre position and orientation
        Args:
            ca, cb: Tuple[float] | Represents the centre position (xyz) of each superquadric in the world frame
            qa, qb: Tuple[float] | Represents the orientation of each superquadric in the world frame as quaternions (wxyz)
        """
        self.ca.value = np.array(ca)
        self.cb.value = np.array(cb)
        self.qa.value = np.array(qa)
        self.qb.value = np.array(qb)

    def sensitivity_analysis(self):
        """
        TODO: change for 3D
        Return the gradient of Lagrangian wrt problem parameters
        Returns: list(float..)  | [nabla_cxa_L, nabla_cya_L, nabla_cya_L, nabla_cyb_L]
        """
        return [-2*self.constraints[0].dual_value*((self.xa.value[0, 0] - self.cxa.value[0, 0]) / self.aa**2),
                -2*self.constraints[0].dual_value*((self.xa.value[1, 0] - self.cxa.value[1, 0]) / self.ba**2),
                -2*self.constraints[1].dual_value*((self.xb.value[0, 0] - self.cxb.value[0, 0]) / self.ab**2),
                -2*self.constraints[1].dual_value*((self.xb.value[1, 0] - self.cxb.value[1, 0]) / self.bb**2)]

    def sensitivity_cvxpy(self):
        """
        TODO: change for 3D
        Sensitivity analysis through cvxpy
        Returns: tuple(np.array, np.array) | [arr(nabla_cxa_L, nabla_cya_L), arr(nabla_cya_L, nabla_cyb_L)]

        More generally, the backward method can be used to compute the gradient of a scalar-valued function f of the
        optimal variables, with respect to the parameters. If x(p) denotes the optimal value of the variable
        (which might be a vector or a matrix) for a particular value of the parameter p and f(x(p)) is a scalar, then
        backward can be used to compute the gradient of f with respect to p. Let x* = x(p), and say the derivative of f
        with respect to x* is dx. To compute the derivative of f with respect to p, before calling problem.backward(),
        just set x.gradient = dx.

        The backward method can be powerful when combined with software for automatic differentiation. We recommend the
        software package CVXPY Layers, which provides differentiable PyTorch and TensorFlow wrappers for CVXPY problems.
        """
        delta_x = 2*(self.xa - self.xb)  # Gradient of objective function wrt decision variables
        self.xa.gradient = delta_x.value
        self.xb.gradient = -delta_x.value
        self.prob.backward()
        return self.cxa.gradient.squeeze(), self.cxb.gradient.squeeze()

class MinDist3D_transl:
    """
    Class for calculating the minimum distance between two ellipse type shapes
    """
    def __init__(self, ca, cb, ra, rb, eps_a, eps_b, objective="SOS"):
        """
        Args:
            ca, cb: Tuple[float] | Represents the centre position (xyz) of each superquadric in the world frame
            ra, rb: Tuple[float] | Represents the radii (abc) of each superquadric
            eps_a, eps_b: Tuple[float] | Represents the roundness parameter (eps1, eps2) of each superquadric
            objective: String | The objective function for the minimisation problem. Defaults to sum of squares (SOS)
        """
        # Decision variables
        self.xa = cp.Variable(3)  # Point on SQ a
        self.xb = cp.Variable(3)  # Point on SQ b

        # Position vectors and orientation as params of SQs
        self.ca = cp.Parameter(3, value=np.array(ca))
        self.cb = cp.Parameter(3, value=np.array(cb))

        # Radii
        self.ra = ra
        self.rb = rb

        # Roundness parameters
        self.eps_a = eps_a
        self.eps_b = eps_b

        # Objective
        if objective == "SOS":
            self.objective = cp.Minimize(cp.sum_squares(self.xa - self.xb))
        elif objective == "NORM":
            self.objective = cp.Minimize(cp.norm(self.xa - self.xb))
        else:
            raise "Type of objective is invalid. Choose 'SOS' or 'NORM'"

        # Constraints, variables must lie on the surface of the sq
        xa_w = ((self.xa[0] - self.ca[0]) / self.ra[0])**(2.0/self.eps_a[1])
        ya_w = ((self.xa[1] - self.ca[1]) / self.ra[1])**(2.0/self.eps_a[1])
        za_w = ((self.xa[2] - self.ca[2]) / self.ra[2])**(2.0/self.eps_a[0])
        xb_w = ((self.xb[0] - self.cb[0]) / self.rb[0])**(2.0/self.eps_b[1])
        yb_w = ((self.xb[1] - self.cb[1]) / self.rb[1])**(2.0/self.eps_b[1])
        zb_w = ((self.xb[2] - self.cb[2]) / self.rb[2])**(2.0/self.eps_b[0])
        self.constraints = [
            ((xa_w + ya_w)**(self.eps_a[1]/self.eps_a[0])) + za_w - 1.0 <= 0,
            ((xb_w + yb_w)**(self.eps_b[1]/self.eps_b[0])) + zb_w - 1.0 <= 0
        ]

        # Problem
        self.prob = cp.Problem(self.objective, self.constraints)
        assert self.prob.is_dcp()

    def get_primal_dual_solutions(self):
        """
        Get the points where the minimum distance occurs and the corresponding dual variables
        Returns: tuple(np.array, float, np.array, float) | (points on xa, dual value for xa,
                                                            points on xb, dual value for xb)
        """
        # self.prob.solve(solver='SCS', requires_grad=True)
        # ['CLARABEL', 'CVXOPT', 'DAQP', 'DIFFCP', 'ECOS', 'ECOS_BB', 'GLPK', 'GLPK_MI', 'OSQP', 'PIQP', 'PROXQP', 'SCIPY', 'SCS']
        ret_val = self.prob.solve(solver='CLARABEL', verbose=False)
        if not isinstance(ret_val, float):
            print(ret_val)
            raise "problem cannot be solved"
        return self.xa.value.squeeze(), self.constraints[0].dual_value, self.xb.value.squeeze(), self.constraints[1].dual_value

    def get_optimal_value(self):
        """
        Obtain objective function given after the optimal solution has been found
        Returns: float | Square of the distance function given the optimal solutions
        """
        return self.objective.value


    def set_params(self, ca, cb):
        """
        For each shape, set a new centre position and orientation
        Args:
            ca, cb: Tuple[float] | Represents the centre position (xyz) of each superquadric in the world frame
            qa, qb: Tuple[float] | Represents the orientation of each superquadric in the world frame as quaternions (wxyz)
        """
        self.ca.value = np.array(ca)
        self.cb.value = np.array(cb)

    def sensitivity_analysis(self):
        """
        TODO: change for 3D
        Return the gradient of Lagrangian wrt problem parameters
        Returns: list(float..)  | [nabla_cxa_L, nabla_cya_L, nabla_cya_L, nabla_cyb_L]
        """
        return [-2*self.constraints[0].dual_value*((self.xa.value[0, 0] - self.cxa.value[0, 0]) / self.aa**2),
                -2*self.constraints[0].dual_value*((self.xa.value[1, 0] - self.cxa.value[1, 0]) / self.ba**2),
                -2*self.constraints[1].dual_value*((self.xb.value[0, 0] - self.cxb.value[0, 0]) / self.ab**2),
                -2*self.constraints[1].dual_value*((self.xb.value[1, 0] - self.cxb.value[1, 0]) / self.bb**2)]

    def sensitivity_cvxpy(self):
        """
        TODO: change for 3D
        Sensitivity analysis through cvxpy
        Returns: tuple(np.array, np.array) | [arr(nabla_cxa_L, nabla_cya_L), arr(nabla_cya_L, nabla_cyb_L)]

        More generally, the backward method can be used to compute the gradient of a scalar-valued function f of the
        optimal variables, with respect to the parameters. If x(p) denotes the optimal value of the variable
        (which might be a vector or a matrix) for a particular value of the parameter p and f(x(p)) is a scalar, then
        backward can be used to compute the gradient of f with respect to p. Let x* = x(p), and say the derivative of f
        with respect to x* is dx. To compute the derivative of f with respect to p, before calling problem.backward(),
        just set x.gradient = dx.

        The backward method can be powerful when combined with software for automatic differentiation. We recommend the
        software package CVXPY Layers, which provides differentiable PyTorch and TensorFlow wrappers for CVXPY problems.
        """
        delta_x = 2*(self.xa - self.xb)  # Gradient of objective function wrt decision variables
        self.xa.gradient = delta_x.value
        self.xb.gradient = -delta_x.value
        self.prob.backward()
        return self.cxa.gradient.squeeze(), self.cxb.gradient.squeeze()

# if __name__ == "__main__":
#     ca = (-1, 0, 0)
#     cb = (-2, 0, 0)
#     ra = (0.1, 0.2, 0.1)
#     rb = (0.1, 0.1, 0.2)
#     eps_a = (1, 1)
#     eps_b = (1, 1)
#     qa = (1, 0, 0, 0)
#     qb = (1, 0, 0, 0)
#     obj = MinDist3D(ca, cb, ra, rb, eps_a, eps_b, qa, qb, objective="NORM")
#     print(obj.get_primal_dual_solutions())
#     cb = (2, 0, 0)
#     qa = (0.9512512, -0.0449435, -0.254887, 0.1677313)
#     obj.set_params(ca, cb, qa, qb)
#     print(obj.get_primal_dual_solutions())
