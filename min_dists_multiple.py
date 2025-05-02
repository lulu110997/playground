import math
import sys
import time
import numpy as np
import sympy
import casadi
if casadi.__version__ != "3.7.0":
    raise "incorrect casadi version! Casadi 3.7.0 is required"

class MinDistMulti:

    def __init__(self, ee_params, obs_params, **kwargs):
        """
        Parent class for calculating minimum distance between two superquadrics
        Args:
            ee_params dict : Dictionary which contains the sq parameters of the ee
            obs_params List[dict] : List of dict which contains the sq parameters of each obstacle
            x_bounds: List[float] | Represents the bounds on the optimal points
            ipopt_options: dict | options for ipopt
        """
        self.ndim = len(ee_params["c"])  # Decides if between 2d or 3d distance calcs
        self.nobs = len(obs_params)  # Number of obstacles in the scene

        # Decision variables
        self.xa = casadi.SX.sym("xa", self.nobs*self.ndim) # Point on the ee
        self.xb = casadi.SX.sym("xb", self.nobs*self.ndim) # Point on the obstacle
        self.decision_vars = casadi.vertcat(self.xa, self.xb)

        # Store SQ params of ee and obstacle
        self.ca = ee_params["c"]
        self.ra = ee_params["r"]
        self.eps_a = ee_params["eps"]

        self.cb = []
        self.rb = []
        self.eps_b = []
        for p in obs_params:
            self.cb.extend(p["c"])
            self.rb.extend(p["r"])
            self.eps_b.extend(p["eps"])

        # Objective
        self.objective = []
        start_ee = 0
        start_obs = 0
        for o in range(0, self.xb.shape[0], 3):
            self.objective.append(casadi.norm_2(self.xa[start_ee:o+3] - self.xb[start_obs:o+3]))
            start_obs += 3
            start_ee += 3

        self.objective = casadi.sum(casadi.vertcat(*self.objective))

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


class MinDistMulti3D(MinDistMulti):
    """
    Class for calculating the minimum distance between two ellipse type shapes
    """
    def __init__(self, ee_params, obs_params, **kwargs):
        """
        Args:
            ca, cb: Tuple[float] | Represents the centre position (xyz) of each superquadric in the world frame
            ra, rb: Tuple[float] | Represents the radii (abc) of each superquadric
            eps_a, eps_b: Tuple[float] | Represents the roundness parameter (eps1, eps2) of each superquadric
            qa, qb: Tuple[float] | Represents the orientation of each superquadric in the world frame as quaternions (wxyz)
            x_bounds: List[float] | Represents the bounds on the optimal points
            ipopt_options: dict | options for ipopt
        """
        super().__init__(ee_params, obs_params, **kwargs)
        # Initialise params for this problem
        self.qa = ee_params["q"]
        self.qb = []
        for p in obs_params:
            self.qb.extend(p["q"])
        self.params = self.ca + self.cb + self.qa + self.qb
        self._set_problem()
        self._get_nabla_L()

    def _set_problem(self):
        """
        Sets the constraints, parameters and nlp solver for the 3D case considering translation and rotation
        """
        # Set-up the constraint for the ee based on the global inside-outside function
        # Declare the parameters
        ca = casadi.SX.sym("ca", self.ndim)
        qa = casadi.SX.sym("qa", self.ndim + 1)

        # Transform points
        x_idx = 0
        c1 = []
        for i in range(self.nobs):
            xrot_a = ((self.xa[x_idx] - ca[0]) * (2 * (qa[0] * qa[0] + qa[1] * qa[1]) - 1)
                      + (self.xa[x_idx+1] - ca[1]) * 2 * (qa[1] * qa[2] + qa[0] * qa[3])
                      + (self.xa[x_idx+2] - ca[2]) * 2 * (qa[1] * qa[3] - qa[0] * qa[2]))

            yrot_a = ((self.xa[x_idx] - ca[0]) * 2 * (qa[1] * qa[2] - qa[0] * qa[3])
                      + (self.xa[x_idx+1] - ca[1]) * (2 * (qa[0] * qa[0] + qa[2] * qa[2]) - 1)
                      + (self.xa[x_idx+2] - ca[2]) * 2 * (qa[2] * qa[3] + qa[0] * qa[1]))

            zrot_a = ((self.xa[x_idx] - ca[0]) * 2 * (qa[1] * qa[3] + qa[0] * qa[2])
                      + (self.xa[x_idx+1] - ca[1]) * 2 * (qa[2] * qa[3] - qa[0] * qa[1])
                      + (self.xa[x_idx+2] - ca[2]) * (2 * (qa[0] * qa[0] + qa[3] * qa[3]) - 1))

            # Compute the xyz components of the inside outside function
            xa_w = (casadi.sqrt(xrot_a**2 + 5e-4) / self.ra[0]) ** (2 / self.eps_a[1])
            ya_w = (casadi.sqrt(yrot_a**2 + 5e-4) / self.ra[1]) ** (2 / self.eps_a[1])
            za_w = (casadi.sqrt(zrot_a**2 + 5e-4) / self.ra[2]) ** (2 / self.eps_a[0])

            # Write inside-outside function as a constraint
            c1.append(casadi.cse(((xa_w + ya_w) ** (self.eps_a[1] / self.eps_a[0])) + za_w - 1))

            x_idx += 3

        # Set-up the constraint for each obstacle
        cb = casadi.SX.sym("cb", self.ndim*self.nobs)
        qb = casadi.SX.sym("qb", (self.ndim + 1)*self.nobs)

        x_idx = 0
        e_idx = 0
        c_idx = 0
        q_idx = 0
        c2 = []
        for i in range(self.nobs):
            xrot_b = ((self.xb[x_idx] - cb[c_idx]) * (2 * (qb[q_idx] * qb[q_idx] + qb[q_idx+1] * qb[q_idx+1]) - 1)
                      + (self.xb[x_idx+1] - cb[c_idx+1]) * 2 * (qb[q_idx+1] * qb[q_idx+2] + qb[q_idx] * qb[q_idx+3])
                      + (self.xb[x_idx+2] - cb[c_idx+2]) * 2 * (qb[q_idx+1] * qb[q_idx+3] - qb[q_idx] * qb[q_idx+2]))

            yrot_b = ((self.xb[x_idx] - cb[c_idx]) * 2 * (qb[q_idx+1] * qb[q_idx+2] - qb[q_idx] * qb[q_idx+3])
                      + (self.xb[x_idx+1] - cb[c_idx+1]) * (2 * (qb[q_idx] * qb[q_idx] + qb[q_idx+2] * qb[q_idx+2]) - 1)
                      + (self.xb[x_idx+2] - cb[c_idx+2]) * 2 * (qb[q_idx+2] * qb[q_idx+3] + qb[q_idx] * qb[q_idx+1]))

            zrot_b = ((self.xb[x_idx] - cb[c_idx]) * 2 * (qb[q_idx+1] * qb[q_idx+3] + qb[q_idx] * qb[q_idx+2])
                      + (self.xb[x_idx+1] - cb[c_idx+1]) * 2 * (qb[q_idx+2] * qb[q_idx+3] - qb[q_idx] * qb[q_idx+1])
                      + (self.xb[x_idx+2] - cb[c_idx+2]) * (2 * (qb[q_idx] * qb[q_idx] + qb[q_idx+3] * qb[q_idx+3]) - 1))

            xb_w = (casadi.sqrt(xrot_b**2 + 5e-4) / self.rb[c_idx]) ** (2 / self.eps_b[e_idx+1])
            yb_w = (casadi.sqrt(yrot_b**2 + 5e-4) / self.rb[c_idx+1]) ** (2 / self.eps_b[e_idx+1])
            zb_w = (casadi.sqrt(zrot_b**2 + 5e-4) / self.rb[c_idx+2]) ** (2 / self.eps_b[e_idx])

            c2.append(casadi.cse(((xb_w + yb_w) ** (self.eps_b[e_idx+1] / self.eps_b[e_idx])) + zb_w - 1))

            x_idx += 3
            c_idx += 3
            q_idx += 4
            e_idx += 2

        # Set and concatenate problem constraints and parameters into a vector
        constraints = casadi.vertcat(*c1, *c2)
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

import os
from superquadric import SuperquadricObject
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# SOLVER = "knitro"
SOLVER = "ipopt"
# SOLVER = "snopt"

if SOLVER == "ipopt":
    GUESS_SCALE = 0.7
    # "hsllib":"/usr/local/lib/libcoinhsl.so",
    solver_options = {"linear_solver":"ma57", "sb":"yes",
                     "print_level":0 ,"timing_statistics":"yes" , "tol":1e-6
                     # , "bound_relax_factor": 1e-9
                     # , "mu_init": 0.05
                     # , "bound_push": 1e-4
                     # , "bound_frac": 0.2
                     # , "slack_bound_push": 1e-4
                     # , "slack_bound_frac": 0.2
                     }
elif SOLVER == "snopt":
    os.environ["SNOPT_LICENSE"] = "/home/louis/licenses/snopt7.lic"
    GUESS_SCALE = 0.97
    solver_options = {
        'Summary file': 0,  # Suppress summary file
        'Major print level': 0,  # Minimal output
        'Minor print level': 0,  # Minimal output
        'Solution': 'No',  # Don't print solution
        'System information': 'No',  # Don't print system info
        'Print frequency': 0 , # Disable iteration output
        'Verify level': 0  # Disable verification output
                     }
elif SOLVER == "knitro":
    os.environ["ARTELYS_LICENSE"] = "/home/louis/licenses/"
    GUESS_SCALE = 0.97
    # 1	Interior/Barrier (default), 2 Active Set SQP, 3	Interior/Barrier + Active Set, 4 Interior/Barrier + Direct Step
    # 5	Feasibility Restoration Phase
    solver_options = {
        "outlev": 1,
        "algorithm": 0,

        # 'xtol': 1e-6,       # Feasibility tolerance (primal feasibility)
        # 'ftol': 1e-6,       # Relative function (objective) tolerance
        # 'opttol': 1e-6,     # Optimality tolerance (KKT conditions)
        # 'feastol': 1e-6,    # Constraint violation tolerance
    }
else:
    raise "Given solver is not supported"

ee_p = {"c": [0, 0, 0],
        "q": [1, 0, 0, 0],
        "r": [0.1, 0.1, 0.1],
        "eps": [1.0, 1.0]}

NUM_OBS = 100

np.random.seed(9)

eps1 = np.random.uniform(0.1, 2.0, size=NUM_OBS)
eps2 = np.random.uniform(0.1, 1.5, size=NUM_OBS)
random_ori = Rotation.random(NUM_OBS)
random_xyz = np.random.uniform(-5, 5, size=(NUM_OBS, 3))
random_rad = np.random.uniform(0.1, 0.5, size=(NUM_OBS, 3))
obs_p = []

for idx, q in enumerate(random_ori):
    ori = q.as_quat().tolist()
    ori = (ori[-1], ori[0], ori[1], ori[2])
    pos = random_xyz[idx].tolist()
    rad = random_rad[idx].tolist()
    obs_p.append({"c": pos, "q": ori, "r": rad, "eps": [eps1[idx], eps2[idx]]})

sq = [SuperquadricObject(*ee_p["r"], *ee_p["eps"], ee_p["c"], ee_p["q"])]
for i in obs_p:
    sq.append(SuperquadricObject(*i["r"], *i["eps"], i["c"], i["q"]))

ee_guess = []
obs_guess = []


for s in sq[1:]:
    ee_guess.extend(sq[0].get_poi(*(s.get_pose()[0]), scale=GUESS_SCALE).tolist())
    obs_guess.extend(s.get_poi(*(sq[0].get_pose()[0]), scale=GUESS_SCALE).tolist())

obj = MinDistMulti3D(ee_p, obs_p, solver=SOLVER, solver_options=solver_options, x_bounds=(-7, 7))
tic = time.time()
x, __ = obj.get_primal_dual_solutions(ee_guess+obs_guess)
toc = 1000*(time.time()-tic)
s = obj.get_solver_stats()

print("time (ms):", s['t_wall_total']*1000, toc)
print("max freq (Hz):", 1/s['t_wall_total'])

ax = plt.subplot(111, projection='3d')
#
# dec_vars = len(x)
# split_num = dec_vars//2
# three = 0
# start=0
# for i in range(0, len(x), 6):
#     ax.plot((x[three], x[split_num+three]),
#             (x[three+1], x[split_num+1+three]),
#             (x[three+2], x[split_num+2+three]), 'ro-')
#     three+=3
#     if three > split_num-3:
#         break
#
# sq[0].plot_sq(ax, 'green')
# for s in sq[1:]:
#     s.plot_sq(ax, 'red')
# plt.show()


# obs_p = [{"c": [0.3, 0.3, 0.3], "q": [1, 0, 0, 0], "r": [0.1, 0.1, 0.1], "eps": [eps1[0], eps2[0]]},
#          {"c": [-0.3, 0.35, 0.3], "q": [1, 0, 0, 0], "r": [0.1, 0.1, 0.0932], "eps": [eps1[1], eps2[1]]},
#          {"c": [0.3, -0.35, 0.3], "q": [1, 0, 0, 0], "r": [0.1, 0.1, 0.1], "eps": [eps1[2], eps2[2]]},
#          {"c": [0.3, 0.35, -0.3], "q": [1, 0, 0, 0], "r": [0.1, 0.1, 0.1], "eps": [eps1[3], eps2[3]]},
#          {"c": [0.3, -0.35, -0.3], "q": [1, 0, 0, 0], "r": [0.1, 0.1, 0.1], "eps": [eps1[4], eps2[4]]},
#          {"c": [-0.3, -0.35, 0.3], "q": [1, 0, 0, 0], "r": [0.1, 0.1, 0.1], "eps": [eps1[5], eps2[5]]},
#          {"c": [-0.3, 0.35, -0.3], "q": [1, 0, 0, 0], "r": [0.1, 0.1, 0.1], "eps": [eps1[6], eps2[6]]},
#          {"c": [-0.3, 0.0, -0.3], "q": [1, 0, 0, 0], "r": [0.1, 0.1, 0.1], "eps": [eps1[7], eps2[7]]},
#
#          {"c": [-1.3, 0.35, 0.3], "q": [1, 0, 0, 0], "r": [0.13, 0.12, 0.11], "eps": [eps1[8], eps2[8]]},
#          {"c": [1.3, -0.35, 0.3], "q": [1, 0, 0, 0], "r": [0.13, 0.12, 0.115], "eps": [eps1[9], eps2[9]]},
#          {"c": [1.3, 0.35, -0.3], "q": [1, 0, 0, 0], "r": [0.13, 0.12, 0.115], "eps": [eps1[10], eps2[10]]},
#          {"c": [1.3, -0.35, -0.3], "q": [1, 0, 0, 0], "r": [0.13, 0.12, 0.115], "eps": [eps1[11], eps2[11]]},
#          {"c": [-1.3, -0.35, 0.3], "q": [1, 0, 0, 0], "r": [0.13, 0.12, 0.115], "eps": [eps1[12], eps2[12]]},
#          {"c": [-1.3, 0.35, -0.3], "q": [1, 0, 0, 0], "r": [0.13, 0.12, 0.115], "eps": [eps1[13], eps2[13]]},
#          {"c": [-1.3, 0.0, -0.3], "q": [1, 0, 0, 0], "r": [0.13, 0.12, 0.115], "eps": [eps1[14], eps2[14]]},
#
#          {"c": [-1.3, 1.35, 0.3], "q": [1, 0, 0, 0], "r": [0.19, 0.12, 0.11], "eps": [eps1[15], eps2[15]]},
#          {"c": [1.3, -1.35, 0.3], "q": [1, 0, 0, 0], "r": [0.19, 0.12, 0.15], "eps": [eps1[16], eps2[16]]},
#          {"c": [1.3, 1.35, -0.3], "q": [1, 0, 0, 0], "r": [0.19, 0.12, 0.15], "eps": [eps1[17], eps2[17]]},
#          {"c": [1.3, -1.35, -0.3], "q": [1, 0, 0, 0], "r": [0.19, 0.12, 0.15], "eps": [eps1[18], eps2[18]]},
#          {"c": [-1.3, -1.35, 0.3], "q": [1, 0, 0, 0], "r": [0.19, 0.12, 0.15], "eps": [eps1[19], eps2[19]]},
#          {"c": [-1.3, 1.35, -0.3], "q": [1, 0, 0, 0], "r": [0.19, 0.12, 0.15], "eps": [eps1[20], eps2[20]]},
#          {"c": [-1.3, 1.0, -0.3], "q": [1, 0, 0, 0], "r": [0.19, 0.12, 0.15], "eps": [eps1[21], eps2[21]]},
#
#          {"c": [-1.3, 1.35, 1.3], "q": [1, 0, 0, 0], "r": [0.19, 0.12, 0.11], "eps": [eps1[22], eps2[22]]},
#          {"c": [1.3, -1.35, 1.3], "q": [1, 0, 0, 0], "r": [0.19, 0.12, 0.15], "eps": [eps1[23], eps2[23]]},
#          {"c": [1.3, 1.35, -1.3], "q": [1, 0, 0, 0], "r": [0.19, 0.12, 0.15], "eps": [eps1[24], eps2[24]]},
#          {"c": [1.3, -1.35, -1.3], "q": [1, 0, 0, 0], "r": [0.19, 0.12, 0.15], "eps": [eps1[25], eps2[25]]},
#          {"c": [-1.3, -1.35, 1.3], "q": [1, 0, 0, 0], "r": [0.19, 0.12, 0.15], "eps": [eps1[26], eps2[26]]},
#          {"c": [-1.3, 1.35, -1.3], "q": [1, 0, 0, 0], "r": [0.19, 0.12, 0.15], "eps": [eps1[27], eps2[27]]},
#          {"c": [-1.3, 1.0, -1.3], "q": [1, 0, 0, 0], "r": [0.19, 0.12, 0.15], "eps": [eps1[28], eps2[28]]},
#
#          {"c": [-1.3, 0.35, 1.3], "q": [1, 0, 0, 0], "r": [0.19, 0.12, 0.11], "eps": [eps1[29], eps2[29]]},
#          {"c": [1.3, -0.35, 1.3], "q": [1, 0, 0, 0], "r": [0.19, 0.12, 0.15], "eps": [eps1[30], eps2[30]]},
#          {"c": [1.3, 0.35, -1.3], "q": [1, 0, 0, 0], "r": [0.19, 0.12, 0.15], "eps": [eps1[31], eps2[31]]},
#          {"c": [1.3, -0.35, -1.3], "q": [1, 0, 0, 0], "r": [0.19, 0.12, 0.15], "eps": [eps1[32], eps2[32]]},
#          {"c": [-1.3, -0.35, 1.3], "q": [1, 0, 0, 0], "r": [0.19, 0.12, 0.15], "eps": [eps1[33], eps2[33]]},
#          {"c": [-1.3, 0.35, -1.3], "q": [1, 0, 0, 0], "r": [0.19, 0.12, 0.15], "eps": [eps1[34], eps2[34]]},
#          {"c": [-1.3, 0.0, -1.3], "q": [1, 0, 0, 0], "r": [0.19, 0.12, 0.15], "eps": [eps1[35], eps2[35]]},
#
#          {"c": [-1.3, -0.35, -0.3], "q": [1, 0, 0, 0], "r": [0.1, 0.12, 0.115], "eps": [eps1[-1], eps2[-1]]}]