"""
TODO: Vectorise operations to improve speed. Just need to ensure the vectorisation performs what I want (element-wise or
matrix)
"""
import math
import sys
import time
import cvxpy as cp
import numpy as np
import sympy

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
            ((self.xa[0] - self.ca[0]) / self.ra[0])**(2/self.eps_a[1]) + ((self.xa[1] - self.ca[1]) / self.ra[1])**(2/self.eps_a[1]) -1 <= 0,
            ((self.xb[0] - self.cb[0]) / self.rb[0])**(2/self.eps_b[1]) + ((self.xb[1] - self.cb[1]) / self.rb[1])**(2/self.eps_b[1]) -1 <= 0
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
        self._get_nabla_L()

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


    def _get_nabla_L(self):
        """
        Use sympy to obtain the derivative of the Lagrangian wrt the problem parameters. Since only the constraint for
        SQ_a contains the problem parameters, we do not need to model the objective function and the contraint for SQ_b.
        Furthermore, the derivative of the Lagrangian wrt the parameters of SQ_b is the same as the derivative wrt SQ_a
        but with opposite signs
        """
        xa = sympy.symbols('xa:3', real=True)
        ca = sympy.symbols('ca:3', real=True)
        ra = sympy.symbols('ra:3', real=True)
        eps_a = sympy.symbols('epsa:2', real=True)
        nu_a = sympy.symbols('nu_a', real=True)
        qa = sympy.symbols('qa:4', real=True)

        xa = sympy.Matrix([[xa[0], xa[1], xa[2]]])
        ca = sympy.Matrix([[ca[0], ca[1], ca[2]]])
        ra = sympy.Matrix([[ra[0], ra[1], ra[2]]])
        eps_a = sympy.Matrix([[eps_a[0], eps_a[1]]])
        qa = sympy.algebras.Quaternion(qa[0], qa[1], qa[2], qa[3])  # w, x, y, z
        qa_vec = qa.to_Matrix()

        SQ_a_rotation = qa.to_rotation_matrix(homogeneous=True)
        SQ_a_pose = SQ_a_rotation.transpose() @ (sympy.Abs(xa - ca)).transpose()

        xa_w = (SQ_a_pose[0] / ra[0]) ** (2.0 / eps_a[1])
        ya_w = (SQ_a_pose[1] / ra[1]) ** (2.0 / eps_a[1])
        za_w = (SQ_a_pose[2] / ra[2]) ** (2.0 / eps_a[0])
        f1 = ((xa_w + ya_w) ** (eps_a[1] / eps_a[0])) + za_w - 1.0

        # xb = sympy.symbols('xb:3', real=True)
        # cb = sympy.symbols('cb:3', real=True)
        # rb = sympy.symbols('rb:3', real=True)
        # eps_b = sympy.symbols('epsb:2', real=True)
        # qb = sympy.symbols('qb:4', real=True)
        # xb = sympy.Matrix([[xb[0], xb[1], xb[2]]])
        # cb = sympy.Matrix([[cb[0], cb[1], cb[2]]])
        # rb = sympy.Matrix([[rb[0], rb[1], rb[2]]])
        # eps_b = sympy.Matrix([[eps_b[0], eps_b[1]]])
        # qb = sympy.algebras.Quaternion(qb[0], qb[1], qb[2], qb[3])  # w, x, y, z
        # qb_vec = qb.to_Matrix()
        # xb_w = (sympy.Abs(xb[0] - cb[0]) / rb[0]) ** (2.0 / eps_b[1])
        # yb_w = (sympy.Abs(xb[1] - cb[1]) / rb[1]) ** (2.0 / eps_b[1])
        # zb_w = (sympy.Abs(xb[2] - cb[2]) / rb[2]) ** (2.0 / eps_b[0])
        # f2 = ((xb_w + yb_w) ** (eps_b[1] / eps_b[0])) + zb_w - 1.0

        f1_diff = nu_a*sympy.Matrix([f1.diff(ca[0]), f1.diff(ca[1]), f1.diff(ca[2]),
                                      f1.diff(qa_vec[0]), f1.diff(qa_vec[1]), f1.diff(qa_vec[2]), f1.diff(qa_vec[3])])
        self.nabla_L = sympy.lambdify((ca, ra, eps_a, qa_vec, xa, nu_a), expr=f1_diff, cse=True)


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
        Return the gradient of Lagrangian wrt problem parameters
        Returns: list(float..)  | [nabla_cxa_L, nabla_cya_L, nabla_cya_L, nabla_cyb_L]
        """
        return self.nabla_L(self.ca.value, self.ra, self.eps_a, self.qa.value, self.xa.value.squeeze(), self.constraints[0].dual_value).squeeze()

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
        xd = math.copysign((2.0 * (cp.abs(self.ca[0] - self.xa[0]) / self.ra[0]) ** (2.0 / self.eps_a[1]) * (
                (cp.abs(self.ca[0] - self.xa[0]) / self.ra[0]) ** (2.0 / self.eps_a[1]) + (cp.abs(self.ca[1] - self.xa[1]) / self.ra[1]) ** (2.0 / self.eps_a[1])) ** (
                self.eps_a[1] / self.eps_a[0])).value, (self.ca[0] - self.xa[0]).value) / (
                self.eps_a[0] * ((cp.abs(self.ca[0] - self.xa[0]) / self.ra[0]) ** (2.0 / self.eps_a[1]) + (cp.abs(self.ca[1] - self.xa[1]) / self.ra[1]) ** (2.0 / self.eps_a[1])) * cp.abs(self.ca[0] - self.xa[0]))
        yd = math.copysign((2.0 * (cp.abs(self.ca[1] - self.xa[1]) / self.ra[1]) ** (2.0 / self.eps_a[1]) * (
                (cp.abs(self.ca[0] - self.xa[0]) / self.ra[0]) ** (2.0 / self.eps_a[1]) + (cp.abs(self.ca[1] - self.xa[1]) / self.ra[1]) ** (2.0 / self.eps_a[1])) ** (
                self.eps_a[1] / self.eps_a[0])).value, (self.ca[1] - self.xa[1]).value) / (
                self.eps_a[0] * ((cp.abs(self.ca[0] - self.xa[0]) / self.ra[0]) ** (2.0 / self.eps_a[1]) + (cp.abs(self.ca[1] - self.xa[1]) / self.ra[1]) ** (2.0 / self.eps_a[1])) * cp.abs(
            self.ca[1] - self.xa[1]))
        zd = math.copysign((2.0 * (cp.abs(self.ca[2] - self.xa[2]) / self.ra[2]) ** (2.0 / self.eps_a[0])).value, (self.ca[2] - self.xa[2]).value) / (self.eps_a[0] * cp.abs(self.ca[2] - self.xa[2]))
        return self.constraints[0].dual_value*xd.value, self.constraints[0].dual_value*yd.value, self.constraints[0].dual_value*zd.value

class MinDistMink2D(MinDist):
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

