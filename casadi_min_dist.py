import math
import sys
import time
import numpy as np
import sympy
import casadi
if casadi.__version__ != "3.7.0":
    raise "incorrect casadi version! Casadi 3.7.0 is required"

class MinDist:

    def __init__(self, ca, cb, ra, rb, eps_a, eps_b, **kwargs):
        """
        Parent class for calculating minimum distance between two superquadrics
        Args:
            ca, cb: List[float] | Represents the centre position of each superquadric in the world frame
            ra, rb: List[float] | Represents the radii of each superquadric
            eps_a, eps_b: List[float] | Represents the roundness parameter (eps1, eps2) of each superquadric
            x_bounds: List[float] | Represents the bounds on the optimal points
            ipopt_options: dict | options for ipopt
        """
        self.ndim = len(ca)  # decides if we are doing 2d or 3d distance calcs

        # Decision variables
        self.xa = casadi.SX.sym("xa", self.ndim) # Point on SQ a
        self.xb = casadi.SX.sym("xb", self.ndim) # Point on SQ b
        self.decision_vars = casadi.vertcat(self.xa, self.xb)

        # Position vectors as params of SQs
        self.ca = ca
        self.cb = cb

        # Radii
        self.ra = ra
        self.rb = rb

        # Roundness parameters
        self.eps_a = eps_a
        self.eps_b = eps_b

        # Objective
        self.objective = casadi.norm_2(self.xa - self.xb)

        self.G_BOUNDS = (-1.0, 1e-10)  # Bounds for the constraint
        self.X_BOUNDS = kwargs.get("x_bounds", (-10.0, 10.0))  # Bounds for the optimal points

        # Solver and solver options
        self.solver = kwargs.get("solver", "ipopt")
        self.solver_options = kwargs.get("solver_options", {})
        self.casadi_options = kwargs.get("casadi_options", {"verbose": False, "print_time":0, "record_time": True,
                                                            "jit": True, self.solver: self.solver_options})
        self.params = None
        self.nlp = None

        self.x_opt = None
        self.nu = None
        self.obj_val = None

    def _set_problem(self):
        """
        Set the optimisation problem based on the constraint. Constraint is the inside-outside function and it depends
        on whether the problem is in 2D or 3D and if the orientation is considered
        """
        raise "Must be setup in the child class"

    def get_primal_dual_solutions(self, x_guess, lam_g0=False):
        """
        Get the points where the minimum distance occurs and the corresponding dual variables
        Args:
            x_guess: List[Float] | initial guess for the optimal solution
            lam_g0 bool or list | initial guess for the dual variables. Defaults to False if no guess is provided

        Returns: tuple(np.array, float, np.array, float) | (points on xa, dual value for xa,
                                                            points on xb, dual value for xb)
        """
        r = self.nlp(p=self.params
                     ,x0=x_guess, lam_g0=lam_g0
                     ,lbg=self.G_BOUNDS[0], ubg=self.G_BOUNDS[1]
                     ,lbx=self.X_BOUNDS[0], ubx=self.X_BOUNDS[1]
                     )
        self.x_opt = r["x"].elements()
        self.nu = r["lam_g"].elements()
        self.obj_val = r["f"].elements()[0]
        return self.x_opt, self.nu

    def get_solver_stats(self):
        """
        Get solver stats from IPOPT
        """
        return self.nlp.stats()

    def get_optimal_value(self):
        """
        Obtain objective function given after the optimal solution has been found
        Returns: float | Square of the distance function given the optimal solutions
        """
        return self.obj_val

    def set_params(self, ca, cb):
        """
        For each shape, set a new centre position and orientation
        Args:
            ca, cb: Tuple[float] | Represents the centre position (xyz) of each superquadric in the world frame
        """
        self.ca = ca
        self.cb = cb
        self.params = self.ca + self.cb


class MinDist3D(MinDist):
    """
    Class for calculating the minimum distance between two ellipse type shapes
    """
    def __init__(self, ca, cb, ra, rb, eps_a, eps_b, qa, qb, **kwargs):
        """
        Args:
            ca, cb: Tuple[float] | Represents the centre position (xyz) of each superquadric in the world frame
            ra, rb: Tuple[float] | Represents the radii (abc) of each superquadric
            eps_a, eps_b: Tuple[float] | Represents the roundness parameter (eps1, eps2) of each superquadric
            qa, qb: Tuple[float] | Represents the orientation of each superquadric in the world frame as quaternions (wxyz)
            x_bounds: List[float] | Represents the bounds on the optimal points
            ipopt_options: dict | options for ipopt
        """
        super().__init__(ca, cb, ra, rb, eps_a, eps_b, **kwargs)
        # Initialise params for this problem
        self.qa = qa
        self.qb = qb
        self.params = self.ca + self.cb + self.qa + self.qb
        self._set_problem()
        self._get_nabla_L()

    def _set_problem(self):
        """
        Sets the constraints, parameters and nlp solver for the 3D case considering translation and rotation
        """
        # Rotated coordinates based on the sq's orientation
        ca = casadi.SX.sym("ca", self.ndim)
        cb = casadi.SX.sym("cb", self.ndim)
        qa = casadi.SX.sym("qa", self.ndim + 1)
        qb = casadi.SX.sym("qb", self.ndim + 1)
        xrot_a = ((self.xa[0] - ca[0]) * (2 * (qa[0] * qa[0] + qa[1] * qa[1]) - 1)
                  + (self.xa[1] - ca[1]) * 2 * (qa[1] * qa[2] + qa[0] * qa[3])
                  + (self.xa[2] - ca[2]) * 2 * (qa[1] * qa[3] - qa[0] * qa[2]))

        yrot_a = ((self.xa[0] - ca[0]) * 2 * (qa[1] * qa[2] - qa[0] * qa[3])
                  + (self.xa[1] - ca[1]) * (2 * (qa[0] * qa[0] + qa[2] * qa[2]) - 1)
                  + (self.xa[2] - ca[2]) * 2 * (qa[2] * qa[3] + qa[0] * qa[1]))

        zrot_a = ((self.xa[0] - ca[0]) * 2 * (qa[1] * qa[3] + qa[0] * qa[2])
                  + (self.xa[1] - ca[1]) * 2 * (qa[2] * qa[3] - qa[0] * qa[1])
                  + (self.xa[2] - ca[2]) * (2 * (qa[0] * qa[0] + qa[3] * qa[3]) - 1))

        xrot_b = ((self.xb[0] - cb[0]) * (2 * (qb[0] * qb[0] + qb[1] * qb[1]) - 1)
                  + (self.xb[1] - cb[1]) * 2 * (qb[1] * qb[2] + qb[0] * qb[3])
                  + (self.xb[2] - cb[2]) * 2 * (qb[1] * qb[3] - qb[0] * qb[2]))

        yrot_b = ((self.xb[0] - cb[0]) * 2 * (qb[1] * qb[2] - qb[0] * qb[3])
                  + (self.xb[1] - cb[1]) * (2 * (qb[0] * qb[0] + qb[2] * qb[2]) - 1)
                  + (self.xb[2] - cb[2]) * 2 * (qb[2] * qb[3] + qb[0] * qb[1]))

        zrot_b = ((self.xb[0] - cb[0]) * 2 * (qb[1] * qb[3] + qb[0] * qb[2])
                  + (self.xb[1] - cb[1]) * 2 * (qb[2] * qb[3] - qb[0] * qb[1])
                  + (self.xb[2] - cb[2]) * (2 * (qb[0] * qb[0] + qb[3] * qb[3]) - 1))

        # Inside-outside function in the world frame as the constraint
        # Change abs(x) to sqrt(x**2 + eps) to improve Hessian calculation (less NaNs)
        # The eps value is used to smooth out the derivatives
        xa_w = (casadi.sqrt(xrot_a**2 + 5e-4) / self.ra[0]) ** (2 / self.eps_a[1])
        ya_w = (casadi.sqrt(yrot_a**2 + 5e-4) / self.ra[1]) ** (2 / self.eps_a[1])
        za_w = (casadi.sqrt(zrot_a**2 + 5e-4) / self.ra[2]) ** (2 / self.eps_a[0])
        xb_w = (casadi.sqrt(xrot_b**2 + 5e-4) / self.rb[0]) ** (2 / self.eps_b[1])
        yb_w = (casadi.sqrt(yrot_b**2 + 5e-4) / self.rb[1]) ** (2 / self.eps_b[1])
        zb_w = (casadi.sqrt(zrot_b**2 + 5e-4) / self.rb[2]) ** (2 / self.eps_b[0])

        # Concatenate constraints into a vector
        c1 = casadi.cse(((xa_w + ya_w) ** (self.eps_a[1] / self.eps_a[0])) + za_w - 1)
        c2 = casadi.cse(((xb_w + yb_w) ** (self.eps_b[1] / self.eps_b[0])) + zb_w - 1)

        # Set problem constraints and parameters
        constraints = casadi.vertcat(c1, c2)
        problem_pars = casadi.vertcat(ca, cb, qa, qb)

        # Define the nlp problem and set the solver
        nlp = {"x": self.decision_vars, "f": self.objective, "g": constraints, "p": problem_pars}
        self.nlp = casadi.nlpsol("S", self.solver, nlp, self.casadi_options)

    def _get_nabla_L(self):
        """
        Use sympy to obtain the derivative of the Lagrangian wrt the problem parameters.
        """
        xa = sympy.symbols("xa:3", real=True)
        ca = sympy.symbols("ca:3", real=True)
        ra = sympy.symbols("ra:3", real=True)
        eps_a = sympy.symbols("epsa:2", real=True)
        nu_a = sympy.symbols("nu_a", real=True)
        qa = sympy.symbols("qa:4", real=True)

        xa = sympy.Matrix([[xa[0], xa[1], xa[2]]])
        ca = sympy.Matrix([[ca[0], ca[1], ca[2]]])
        ra = sympy.Matrix([[ra[0], ra[1], ra[2]]])
        eps_a = sympy.Matrix([[eps_a[0], eps_a[1]]])
        qa = sympy.algebras.Quaternion(qa[0], qa[1], qa[2], qa[3])  # w, x, y, z
        qa_vec = qa.to_Matrix()

        SQ_a_rotation = qa.to_rotation_matrix(homogeneous=True)
        SQ_a_pose = SQ_a_rotation.transpose() @ (xa - ca).transpose()

        xa_w = (sympy.Abs(SQ_a_pose[0]) / ra[0]) ** (2.0 / eps_a[1])
        ya_w = (sympy.Abs(SQ_a_pose[1]) / ra[1]) ** (2.0 / eps_a[1])
        za_w = (sympy.Abs(SQ_a_pose[2]) / ra[2]) ** (2.0 / eps_a[0])
        f1 = ((xa_w + ya_w) ** (eps_a[1] / eps_a[0])) + za_w - 1.0

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
        self.ca = ca
        self.cb = cb
        self.qa = qa
        self.qb = qb
        self.params = self.ca + self.cb + self.qa + self.qb


    def sensitivity_analysis(self):
        """
        Return the gradient of Lagrangian wrt problem parameters
        Returns: list(float..)  | [nabla_cxa_L, nabla_cya_L, nabla_cya_L, nabla_cyb_L]
        """
        return self.nabla_L(self.ca, self.ra, self.eps_a, self.qa, self.x_opt[:self.ndim], self.nu[0]).squeeze()

class MinDist3DTransl(MinDist):
    """
    Class for calculating the minimum distance between two ellipse type shapes
    """
    def __init__(self, ca, cb, ra, rb, eps_a, eps_b, x_bounds=None, ipopt_options=None):
        """
        Args:
            ca, cb: Tuple[float] | Represents the centre position (xyz) of each superquadric in the world frame
            ra, rb: Tuple[float] | Represents the radii (abc) of each superquadric
            eps_a, eps_b: Tuple[float] | Represents the roundness parameter (eps1, eps2) of each superquadric
            x_bounds: List[float] | Represents the bounds on the optimal points
            ipopt_options: dict | options for ipopt
        """
        super().__init__(ca, cb, ra, rb, eps_a, eps_b, x_bounds=None, ipopt_options=None)
        self.params = self.ca + self.cb
        self._set_problem()

    def _set_problem(self):
        """
        Inside-outside function for the 3D case considering only translation
        """
        # Define parameters
        ca = casadi.SX.sym("ca", self.ndim)
        cb = casadi.SX.sym("cb", self.ndim)

        # Constraints, variables must lie on the surface of the sq
        xa_w = (casadi.fabs(self.xa[0] - ca[0]) / self.ra[0])**(2.0/self.eps_a[1])
        ya_w = (casadi.fabs(self.xa[1] - ca[1]) / self.ra[1])**(2.0/self.eps_a[1])
        za_w = (casadi.fabs(self.xa[2] - ca[2]) / self.ra[2])**(2.0/self.eps_a[0])
        xb_w = (casadi.fabs(self.xb[0] - cb[0]) / self.rb[0])**(2.0/self.eps_b[1])
        yb_w = (casadi.fabs(self.xb[1] - cb[1]) / self.rb[1])**(2.0/self.eps_b[1])
        zb_w = (casadi.fabs(self.xb[2] - cb[2]) / self.rb[2])**(2.0/self.eps_b[0])

        # Concatenate constraints into a vector
        c1 = casadi.cse(((xa_w + ya_w) ** (self.eps_a[1] / self.eps_a[0])) + za_w - 1)
        c2 = casadi.cse(((xb_w + yb_w) ** (self.eps_b[1] / self.eps_b[0])) + zb_w - 1)


        # Set attributes
        constraints = casadi.vertcat(c1, c2)
        problem_pars = casadi.vertcat(ca, cb)


        # Define the nlp problem and set the solver
        nlp = {"x": self.decision_vars, "f": self.objective, "g": constraints, "p": problem_pars}
        self.nlp = casadi.nlpsol("S", "ipopt", nlp,
                         {"ipopt":{"linear_solver": "ma27", "hsllib": "/usr/local/lib/libcoinhsl.so", "sb": "yes",
                                   "print_level":0, "mu_strategy":"adaptive", "tol":1e-6}, "verbose": False,  "print_time":0})
        self.constraints = [
            ((xa_w + ya_w)**(self.eps_a[1]/self.eps_a[0])) + za_w - 1.0 <= 0,
            ((xb_w + yb_w)**(self.eps_b[1]/self.eps_b[0])) + zb_w - 1.0 <= 0
        ]

    def sensitivity_analysis(self):
        """
        Return the gradient of Lagrangian wrt problem parameters
        Returns: list(float..)  | [nabla_cxa_L, nabla_cya_L, nabla_cya_L, nabla_cyb_L]
        """
        xd = math.copysign((2.0 * (casadi.fabs(self.ca[0] - self.x_opt[0]) / self.ra[0]) ** (2.0 / self.eps_a[1]) * (
                (casadi.fabs(self.ca[0] - self.x_opt[0]) / self.ra[0]) ** (2.0 / self.eps_a[1]) + (casadi.fabs(self.ca[1] - self.x_opt[1]) / self.ra[1]) ** (2.0 / self.eps_a[1])) ** (
                self.eps_a[1] / self.eps_a[0])), (self.ca[0] - self.x_opt[0])) / (
                self.eps_a[0] * ((casadi.fabs(self.ca[0] - self.x_opt[0]) / self.ra[0]) ** (2.0 / self.eps_a[1]) + (casadi.fabs(self.ca[1] - self.x_opt[1]) / self.ra[1]) ** (2.0 / self.eps_a[1])) * casadi.fabs(self.ca[0] - self.x_opt[0]))
        yd = math.copysign((2.0 * (casadi.fabs(self.ca[1] - self.x_opt[1]) / self.ra[1]) ** (2.0 / self.eps_a[1]) * (
                (casadi.fabs(self.ca[0] - self.x_opt[0]) / self.ra[0]) ** (2.0 / self.eps_a[1]) + (casadi.fabs(self.ca[1] - self.x_opt[1]) / self.ra[1]) ** (2.0 / self.eps_a[1])) ** (
                self.eps_a[1] / self.eps_a[0])), (self.ca[1] - self.x_opt[1])) / (
                self.eps_a[0] * ((casadi.fabs(self.ca[0] - self.x_opt[0]) / self.ra[0]) ** (2.0 / self.eps_a[1]) + (casadi.fabs(self.ca[1] - self.x_opt[1]) / self.ra[1]) ** (2.0 / self.eps_a[1])) * casadi.fabs(
            self.ca[1] - self.x_opt[1]))
        zd = math.copysign((2.0 * (casadi.fabs(self.ca[2] - self.x_opt[2]) / self.ra[2]) ** (2.0 / self.eps_a[0])), (self.ca[2] - self.x_opt[2])) / (self.eps_a[0] * casadi.fabs(self.ca[2] - self.x_opt[2]))
        return self.nu[0]*xd, self.nu[0]*yd, self.nu[0]*zd