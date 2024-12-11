"""
TODO: Vectorise operations to improve speed. Just need to ensure the vectorisation performs what I want (element-wise or
matrix)
"""
import math
import sys
import time
import cvxpy as cp
import numpy as np

class MinDist:
    def __init__(self, ca, cb, ra, rb, eps_a, eps_b, objective="SOS"):
        """
        Parent class for the other classes to calculate minimum distance between two superquadrics
        Args:
            ca, cb: Tuple[float] | Represents the centre position of each superquadric in the world frame
            ra, rb: Tuple[float] | Represents the radii of each superquadric
            eps_a, eps_b: Tuple[float] | Represents the roundness parameter (eps1, eps2) of each superquadric
            objective: String | The objective function for the minimisation problem. Defaults to sum of squares (SOS)
        """
        self.ndim = len(ca)  # decides if we are doing 2d or 3d distance calcs

        # Decision variables
        self.xa = cp.Variable(self.ndim)  # Point on SQ a
        self.xb = cp.Variable(self.ndim)  # Point on SQ b

        # Position vectors as params of SQs
        self.ca = cp.Parameter(self.ndim, value=np.array(ca))
        self.cb = cp.Parameter(self.ndim, value=np.array(cb))

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
            raise "Choose 'SOS' or 'NORM'"

        self.constraints = None
        self.prob = None

    def _set_problem(self):
        """
        Set the optimisation probelm based on the constraint. Constraint is the inside-outside function and it depends
        on whether the problem is in 2D or 3D and if the orientation is considered
        """
        raise 'Must be setup in the child class'

    def get_primal_dual_solutions(self, requires_grad):
        """
        Get the points where the minimum distance occurs and the corresponding dual variables
        Args:
            requires_grad: bool | Makes it possible to compute gradients of a solution with respect to Parameters
        Returns: tuple(np.array, float, np.array, float) | (points on xa, dual value for xa,
                                                            points on xb, dual value for xb)
        """
        if requires_grad:
            self.prob.solve(solver='SCS', requires_grad=True)
        else:
            self.prob.solve(solver='CLARABEL')
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
        """
        self.ca.value = np.array(ca)
        self.cb.value = np.array(cb)

class MinDist2D(MinDist):
    """
    Class for calculating the minimum distance between two superquadric shapes
    """
    def __init__(self, ca, cb, ra, rb, eps_a=2.0, eps_b=2.0, objective="SOS"):
        """
        Parent class for the other classes to calculate minimum distance between two superquadrics
        Args:
            ca, cb: Tuple[float] | Represents the centre position of each superquadric in the world frame
            ra, rb: Tuple[float] | Represents the radii of each superquadric
            eps_a, eps_b: Tuple[float] | Represents the roundness parameter (eps1, eps2) of each superquadric
            objective: String | The objective function for the minimisation problem. Defaults to sum of squares (SOS)
        """
        super().__init__(ca, cb, ra, rb, eps_a, eps_b, objective)
        self._set_problem()

    def _set_problem(self):
        """
        Inside-outside function for the 2D case considering only translation
        """
        # Constraints, variables must lie on the surface of the sq
        self.constraints = [
            ((self.xa[0] - self.ca[0]) / self.ra[0])**2 + ((self.xa[1] - self.ca[1]) / self.ra[1])**2 -1 <= 0,
            ((self.xb[0] - self.cb[0]) / self.rb[0])**2 + ((self.xb[1] - self.cb[1]) / self.rb[1])**2 -1 <= 0
        ]

        # Problem
        self.prob = cp.Problem(self.objective, self.constraints)

    def sensitivity_analysis(self):
        """
        Return the gradient of Lagrangian wrt problem parameters
        Returns: list(float..)  | [nabla_cxa_L, nabla_cya_L, nabla_cya_L, nabla_cyb_L]
        """
        return [-2*self.constraints[0].dual_value*((self.xa.value[0] - self.ca.value[0]) / self.ra[0]**2),
                -2*self.constraints[0].dual_value*((self.xa.value[1] - self.ca.value[1]) / self.ra[1]**2),
                -2*self.constraints[1].dual_value*((self.xb.value[0] - self.cb.value[0]) / self.rb[0]**2),
                -2*self.constraints[1].dual_value*((self.xb.value[1] - self.cb.value[1]) / self.rb[1]**2)]

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
        return self.ca.gradient.squeeze(), self.cb.gradient.squeeze()

class MinDist3DRot(MinDist):
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
        super().__init__(ca, cb, ra, rb, eps_a, eps_b, objective)
        self.qa = cp.Parameter(4, value=np.array(qa))
        self.qb = cp.Parameter(4, value=np.array(qb))
        self._set_problem()

    def _set_problem(self):
        """
        Inside-outside function for the 3D case considering translation and rotation
        """
        #  Transform points on each sq
        xrot_a = (  (self.xa[0] - self.ca[0]) * (2*(self.qa[0]*self.qa[0] + self.qa[1]*self.qa[1]) - 1)
                  + (self.xa[1] - self.ca[1]) *  2*(self.qa[1]*self.qa[2] + self.qa[0]*self.qa[3])
                  + (self.xa[2] - self.ca[2]) *  2*(self.qa[1]*self.qa[3] - self.qa[0]*self.qa[2]))

        yrot_a = (  (self.xa[0] - self.ca[0]) *  2*(self.qa[1]*self.qa[2] - self.qa[0]*self.qa[3])
                  + (self.xa[1] - self.ca[1]) * (2*(self.qa[0]*self.qa[0] + self.qa[2]*self.qa[2]) - 1)
                  + (self.xa[2] - self.ca[2]) *  2*(self.qa[2]*self.qa[3] + self.qa[0]*self.qa[1]))

        zrot_a = (  (self.xa[0] - self.ca[0]) *  2*(self.qa[1]*self.qa[3] + self.qa[0]*self.qa[2])
                  + (self.xa[1] - self.ca[1]) *  2*(self.qa[2]*self.qa[3] - self.qa[0]*self.qa[1])
                  + (self.xa[2] - self.ca[2]) * (2*(self.qa[0]*self.qa[0] + self.qa[3]*self.qa[3]) - 1))

        xrot_b = (  (self.xb[0] - self.cb[0]) * (2*(self.qb[0]*self.qb[0] + self.qb[1]*self.qb[1]) - 1)
                  + (self.xb[1] - self.cb[1]) *  2*(self.qb[1]*self.qb[2] + self.qb[0]*self.qb[3])
                  + (self.xb[2] - self.cb[2]) *  2*(self.qb[1]*self.qb[3] - self.qb[0]*self.qb[2]))

        yrot_b = (  (self.xb[0] - self.cb[0]) *  2*(self.qb[1]*self.qb[2] - self.qb[0]*self.qb[3])
                  + (self.xb[1] - self.cb[1]) * (2*(self.qb[0]*self.qb[0] + self.qb[2]*self.qb[2]) - 1)
                  + (self.xb[2] - self.cb[2]) *  2*(self.qb[2]*self.qb[3] + self.qb[0]*self.qb[1]))

        zrot_b = (  (self.xb[0] - self.cb[0]) *  2*(self.qb[1]*self.qb[3] + self.qb[0]*self.qb[2])
                  + (self.xb[1] - self.cb[1]) *  2*(self.qb[2]*self.qb[3] - self.qb[0]*self.qb[1])
                  + (self.xb[2] - self.cb[2]) * (2*(self.qb[0]*self.qb[0] + self.qb[3]*self.qb[3]) - 1))


        # Constraints, variables must lie on the surface of the sq
        xa_w = ((cp.abs(xrot_a) / self.ra[0])**(2/self.eps_a[1]))
        ya_w = ((cp.abs(yrot_a) / self.ra[1])**(2/self.eps_a[1]))
        za_w = ((cp.abs(zrot_a) / self.ra[2])**(2/self.eps_a[0]))
        xb_w = ((cp.abs(xrot_b) / self.rb[0])**(2/self.eps_b[1]))
        yb_w = ((cp.abs(yrot_b) / self.rb[1])**(2/self.eps_b[1]))
        zb_w = ((cp.abs(zrot_b) / self.rb[2])**(2/self.eps_b[0]))
        self.constraints = [
            ((xa_w + ya_w)**(self.eps_a[1]/self.eps_a[0])) + za_w - 1 <= 0,
            ((xb_w + yb_w)**(self.eps_b[1]/self.eps_b[0])) + zb_w - 1 <= 0
        ]

        # Problem
        self.prob = cp.Problem(self.objective, self.constraints)
        assert self.prob.is_dcp()


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

class MinDist3DTransl(MinDist):
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
        super().__init__(ca, cb, ra, rb, eps_a, eps_b, objective)
        self._set_problem()

    def _set_problem(self):
        """
        Inside-outside function for the 3D case considering only translation
        """
        # Constraints, variables must lie on the surface of the sq
        xa_w = (cp.abs(self.xa[0] - self.ca[0]) / self.ra[0])**(2.0/self.eps_a[1])
        ya_w = (cp.abs(self.xa[1] - self.ca[1]) / self.ra[1])**(2.0/self.eps_a[1])
        za_w = (cp.abs(self.xa[2] - self.ca[2]) / self.ra[2])**(2.0/self.eps_a[0])
        xb_w = (cp.abs(self.xb[0] - self.cb[0]) / self.rb[0])**(2.0/self.eps_b[1])
        yb_w = (cp.abs(self.xb[1] - self.cb[1]) / self.rb[1])**(2.0/self.eps_b[1])
        zb_w = (cp.abs(self.xb[2] - self.cb[2]) / self.rb[2])**(2.0/self.eps_b[0])
        self.constraints = [
            ((xa_w + ya_w)**(self.eps_a[1]/self.eps_a[0])) + za_w - 1.0 <= 0,
            ((xb_w + yb_w)**(self.eps_b[1]/self.eps_b[0])) + zb_w - 1.0 <= 0
        ]

        # Problem
        self.prob = cp.Problem(self.objective, self.constraints)
        assert self.prob.is_dcp()

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
