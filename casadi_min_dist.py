import math
import sys
import time
import numpy as np
import sympy
import casadi

class MinDist:

    def __init__(self, ca, cb, ra, rb, eps_a, eps_b):
        """
        Parent class for the other classes to calculate minimum distance between two superquadrics
        Args:
            ca, cb: List[float] | Represents the centre position of each superquadric in the world frame
            ra, rb: List[float] | Represents the radii of each superquadric
            eps_a, eps_b: List[float] | Represents the roundness parameter (eps1, eps2) of each superquadric
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

        self.G_BOUNDS = (-1.0, 0.0)  # Bounds for the constraint
        self.X_BOUNDS = (-50.0, 50.0)  # Bounds for the points

        self.params = None
        self.nlp = None

        self.x_opt = np.zeros((1, 6))
        self.nu = None
        self.obj_val = None

    def _set_problem(self):
        """
        Set the optimisation problem based on the constraint. Constraint is the inside-outside function and it depends
        on whether the problem is in 2D or 3D and if the orientation is considered
        """
        raise 'Must be setup in the child class'

    def get_primal_dual_solutions(self, x_guess, lam_g0):
        """
        Get the points where the minimum distance occurs and the corresponding dual variables
        Args:
            x_guess: bool | Makes it possible to compute gradients of a solution with respect to Parameters
            lam_g0

        Returns: tuple(np.array, float, np.array, float) | (points on xa, dual value for xa,
                                                            points on xb, dual value for xb)
        """
        r = self.nlp(p=self.params
                     ,x0=x_guess, lam_g0=lam_g0
                     ,lbg=self.G_BOUNDS[0], ubg=self.G_BOUNDS[1]
                     ,lbx=self.X_BOUNDS[0], ubx=self.X_BOUNDS[1]
                     )
        self.x_opt = r['x'].elements()
        self.nu = r['lam_g'].elements()
        self.obj_val = r['f'].elements()[0]
        return self.x_opt, self.nu

    def get_solver_stats(self):
        """
        Get solver stats
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
    def __init__(self, ca, cb, ra, rb, eps_a, eps_b, qa, qb):
        """
        Args:
            ca, cb: Tuple[float] | Represents the centre position (xyz) of each superquadric in the world frame
            ra, rb: Tuple[float] | Represents the radii (abc) of each superquadric
            eps_a, eps_b: Tuple[float] | Represents the roundness parameter (eps1, eps2) of each superquadric
            qa, qb: Tuple[float] | Represents the orientation of each superquadric in the world frame as quaternions (wxyz)
        """
        super().__init__(ca, cb, ra, rb, eps_a, eps_b)
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
        xa_w = casadi.cse((casadi.fabs(xrot_a) / self.ra[0]) ** (2 / self.eps_a[1]))
        ya_w = casadi.cse((casadi.fabs(yrot_a) / self.ra[1]) ** (2 / self.eps_a[1]))
        za_w = casadi.cse((casadi.fabs(zrot_a) / self.ra[2]) ** (2 / self.eps_a[0]))
        xb_w = casadi.cse((casadi.fabs(xrot_b) / self.rb[0]) ** (2 / self.eps_b[1]))
        yb_w = casadi.cse((casadi.fabs(yrot_b) / self.rb[1]) ** (2 / self.eps_b[1]))
        zb_w = casadi.cse((casadi.fabs(zrot_b) / self.rb[2]) ** (2 / self.eps_b[0]))

        # Concatenate constraints into a vector
        c1 = casadi.cse(((xa_w + ya_w) ** (self.eps_a[1] / self.eps_a[0])) + za_w - 1)
        c2 = casadi.cse(((xb_w + yb_w) ** (self.eps_b[1] / self.eps_b[0])) + zb_w - 1)


        # Set attributes
        constraints = casadi.vertcat(c1, c2)
        problem_pars = casadi.vertcat(ca, cb, qa, qb)


        # Define the nlp problem and set the solver
        # Some docs for warm starting IPOPT
        # https://github.com/casadi/casadi/wiki/FAQ%3A-Warmstarting-with-IPOPT
        # https://www.gams.com/latest/docs/S_IPOPT.html#IPOPT_WARMSTART
        nlp = {'x': self.decision_vars, 'f': self.objective, 'g': constraints, 'p': problem_pars}
        self.nlp = casadi.nlpsol('S', 'ipopt', nlp,
                         {"verbose": False, 'print_time':0, 'jit': True,
                          'ipopt':{'linear_solver':'ma27', "hsllib":"/usr/local/lib/libcoinhsl.so", "sb":'yes',
                                   'print_level':0, 'warm_start_init_point':'yes'
                                   # ,"mu_strategy":"monotone", "mu_init": 0.0001
                                   # ,"warm_start_bound_push": 1e-6
                                   # ,"warm_start_bound_frac": 1e-6
                                   # ,"warm_start_slack_bound_frac": 1e-6
                                   # ,"warm_start_slack_bound_push": 1e-6
                                   # ,"warm_start_mult_bound_push": 1e-6
                                   # ,"nlp_scaling_method": "none"
                                   # ,"bound_relax_factor": 1e-7
                                   }})

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
    def __init__(self, ca, cb, ra, rb, eps_a, eps_b):
        """
        Args:
            ca, cb: Tuple[float] | Represents the centre position (xyz) of each superquadric in the world frame
            ra, rb: Tuple[float] | Represents the radii (abc) of each superquadric
            eps_a, eps_b: Tuple[float] | Represents the roundness parameter (eps1, eps2) of each superquadric
        """
        super().__init__(ca, cb, ra, rb, eps_a, eps_b)
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
        nlp = {'x': self.decision_vars, 'f': self.objective, 'g': constraints, 'p': problem_pars}
        self.nlp = casadi.nlpsol('S', 'ipopt', nlp,
                         {'ipopt':{'linear_solver': 'ma27', "hsllib": "/usr/local/lib/libcoinhsl.so", "sb": 'yes',
                                   'print_level':0, "mu_strategy":"adaptive", "tol":1e-6}, "verbose": False,  'print_time':0})
        self.constraints = [
            ((xa_w + ya_w)**(self.eps_a[1]/self.eps_a[0])) + za_w - 1.0 <= 0,
            ((xb_w + yb_w)**(self.eps_b[1]/self.eps_b[0])) + zb_w - 1.0 <= 0
        ]

        # Problem
        # self.prob = casadi.Problem(self.objective, self.constraints)

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


if __name__ == "__main__":
    def nlp_problem(ra, rb, eps_a, eps_b):
        """
        Define the problem
        Args:
            ra, rb: Tuple[float] | Represents the radii of each superquadric
            eps_a, eps_b: Tuple[float] | Represents the roundness parameter (eps1, eps2) of each superquadric
        Returns: nlp solver

        Different linear solvers for IPOPT
        ma27: use the Harwell routine MA27
        ma57: use the Harwell routine MA57
        ma77: use the Harwell routine HSL_MA77
        ma86: use the Harwell routine HSL_MA86
        ma97: use the Harwell routine HSL_MA97
        spral: use the Spral package
        mumps: use the Mumps package
        """
        # Define variables and parameters
        xa = casadi.SX.sym("xa", 3)
        xb = casadi.SX.sym("xb", 3)
        qa = casadi.SX.sym("qa", 4)
        qb = casadi.SX.sym("qb", 4)
        ca = casadi.SX.sym("ca", 3)
        cb = casadi.SX.sym("cb", 3)
        decision_vars = casadi.vertcat(xa, xb)
        problem_pars = casadi.vertcat(ca, cb, qa, qb)

        # Rotated coordinates based on the sq's orientation
        xrot_a = ((xa[0] - ca[0]) * (2 * (qa[0] * qa[0] + qa[1] * qa[1]) - 1)
                  + (xa[1] - ca[1]) * 2 * (qa[1] * qa[2] + qa[0] * qa[3])
                  + (xa[2] - ca[2]) * 2 * (qa[1] * qa[3] - qa[0] * qa[2]))

        yrot_a = ((xa[0] - ca[0]) * 2 * (qa[1] * qa[2] - qa[0] * qa[3])
                  + (xa[1] - ca[1]) * (2 * (qa[0] * qa[0] + qa[2] * qa[2]) - 1)
                  + (xa[2] - ca[2]) * 2 * (qa[2] * qa[3] + qa[0] * qa[1]))

        zrot_a = ((xa[0] - ca[0]) * 2 * (qa[1] * qa[3] + qa[0] * qa[2])
                  + (xa[1] - ca[1]) * 2 * (qa[2] * qa[3] - qa[0] * qa[1])
                  + (xa[2] - ca[2]) * (2 * (qa[0] * qa[0] + qa[3] * qa[3]) - 1))

        xrot_b = ((xb[0] - cb[0]) * (2 * (qb[0] * qb[0] + qb[1] * qb[1]) - 1)
                  + (xb[1] - cb[1]) * 2 * (qb[1] * qb[2] + qb[0] * qb[3])
                  + (xb[2] - cb[2]) * 2 * (qb[1] * qb[3] - qb[0] * qb[2]))

        yrot_b = ((xb[0] - cb[0]) * 2 * (qb[1] * qb[2] - qb[0] * qb[3])
                  + (xb[1] - cb[1]) * (2 * (qb[0] * qb[0] + qb[2] * qb[2]) - 1)
                  + (xb[2] - cb[2]) * 2 * (qb[2] * qb[3] + qb[0] * qb[1]))

        zrot_b = ((xb[0] - cb[0]) * 2 * (qb[1] * qb[3] + qb[0] * qb[2])
                  + (xb[1] - cb[1]) * 2 * (qb[2] * qb[3] - qb[0] * qb[1])
                  + (xb[2] - cb[2]) * (2 * (qb[0] * qb[0] + qb[3] * qb[3]) - 1))

        # Inside-outside function in the world frame as the constraint
        xa_w = casadi.cse((casadi.fabs(xrot_a) / ra[0]) ** (2 / eps_a[1]))
        ya_w = casadi.cse((casadi.fabs(yrot_a) / ra[1]) ** (2 / eps_a[1]))
        za_w = casadi.cse((casadi.fabs(zrot_a) / ra[2]) ** (2 / eps_a[0]))
        xb_w = casadi.cse((casadi.fabs(xrot_b) / rb[0]) ** (2 / eps_b[1]))
        yb_w = casadi.cse((casadi.fabs(yrot_b) / rb[1]) ** (2 / eps_b[1]))
        zb_w = casadi.cse((casadi.fabs(zrot_b) / rb[2]) ** (2 / eps_b[0]))

        # Concatenate contraints into a vector
        c1 = casadi.cse(((xa_w + ya_w)**(eps_a[1]/eps_a[0])) + za_w - 1)
        c2 = casadi.cse(((xb_w + yb_w)**(eps_b[1]/eps_b[0])) + zb_w - 1)
        constraints = casadi.vertcat(c1, c2)

        # Define objective function
        objective = casadi.norm_2(xa - xb)

        # Define the problem
        nlp = {'x':decision_vars, 'f':objective, 'g':constraints, 'p':problem_pars}

        return casadi.nlpsol('S', 'ipopt', nlp,
                             {'ipopt':{'linear_solver': 'ma27', "hsllib": "/usr/local/lib/libcoinhsl.so", "sb": 'yes',
                                       'print_level':0, "mu_strategy":"adaptive"}, "verbose": False,  'print_time':0})


    def main():
        ra = (0.1, 0.2, 0.13)  # Circle radius
        rb = (0.11, 0.0375, 0.11)
        eps_a = (1.0, 1.0)
        eps_b = (0.5, 0.5)
        g_bounds = (-1.0, 0.0)
        x_bounds = (-2.0, 2.0)
        x_guess = [-0.20301610361083608, 0.34511759398032515, 0.272650496038, -0.3585103113410, -0.05026070374510, 0.21552559962]
        S = nlp_problem(ra, rb, eps_a, eps_b)

        ca = [-0.2454, 0.5, 0.29 ]
        cb = [-0.45, -0.080, 0.15]
        qa = [0.9396926, 0.0, 0.0, 0.3420201]
        qb = [1.0, 0.0, 0.0, 0.0]

        for nmbs in range(10):
            ca[0] += 0.1
            r = S(x0=x_guess,
                  p=ca + cb + qa + qb,
                  lbg=g_bounds[0], ubg=g_bounds[1], lbx=x_bounds[0], ubx=x_bounds[1])

            x_opt = r['x']
            nu = r['lam_g']
            obj_val = r['f']
            print('x_opt: ', x_opt)
            print('nu ', nu)
            print('obj_val', obj_val)

    main()