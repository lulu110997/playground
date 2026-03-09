import sys

import matplotlib.pyplot as plt
from spatialmath import SO3, SE3
import spatialmath.base as spb
import numpy as np
import traceback
from spatialmath.base import transl, trotx
from scipy.linalg import expm
import numpy as np
import sympy as sp

def skew_symmetric(v):
    x, y, z = v
    return np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])

def rodrigues_from_angular_velocity(omega, dt=1.0):
    """
    Compute rotation matrix from angular velocity using Rodrigues formula.
    Overtime, this function can cause discretisation errors to build up causing R to not be in SO(3).
    In this case, R_next needs to be re-orthonormalized
    U, _, Vt = np.linalg.svd(R_next)
    R_next = U @ Vt

    Parameters
    ----------
    omega : array-like (3,)
        Angular velocity vector (rad/s)
    dt : float
        Time step (seconds)

    Returns
    -------
    R : ndarray (3,3)
        Rotation matrix
    """
    omega = np.asarray(omega, dtype=float)

    theta = np.linalg.norm(omega) * dt
    if theta < 1e-8:
        # print("here")
        # # first-order exponential approximation
        # wx, wy, wz = omega
        # w_hat = np.array([
        #     [0, -wz, wy],
        #     [wz, 0, -wx],
        #     [-wy, wx, 0]
        # ])
        # return np.eye(3) + w_hat * dt
        return np.eye(3)

    axis = omega / np.linalg.norm(omega)

    ux, uy, uz = axis

    # Skew-symmetric matrix
    u_hat = np.array([
        [0, -uz, uy],
        [uz, 0, -ux],
        [-uy, ux, 0]
    ])

    R = np.eye(3) + np.sin(theta) * u_hat + (1 - np.cos(theta)) * (u_hat @ u_hat)

    return R

def rotation_from_normal_tangent(n, x):
    """
    Create rotation matrix R such that:
        R[:,2] = normalized normal vector (z)
        R[:,0] = tangent direction projected onto plane of z (x)

    Parameters
    ----------
    n : array-like (3,)
        Normal direction (z-axis)
    x : array-like (3,)
        Tangent direction (approximate x-axis)

    Returns
    -------
    R : ndarray (3,3)
        Rotation matrix
    """

    n = np.asarray(n, dtype=float)
    x = np.asarray(x, dtype=float)

    # Normalize normal
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-8:
        raise ValueError("Normal vector too small")
    z_axis = n / n_norm

    # Remove component of x along z (Gram-Schmidt)
    x_axis = x - np.dot(x, z_axis) * z_axis
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-8:
        raise ValueError("Tangent is parallel to normal")

    x_axis /= x_norm

    # Right-handed frame
    y_axis = np.cross(z_axis, x_axis)

    R = np.column_stack((x_axis, y_axis, z_axis))
    return R

def principal_curvatures_sphere(R, point, center=(0, 0, 0), inward=True):
    """
    Principal curvatures and directions of a sphere
    with arbitrary center.

    Parameters
    ----------
    R : float
        Radius of sphere
    point : array-like (3,)
        Point on sphere
    center : array-like (3,)
        Center of sphere
    inward : bool
        If True → inward normal (positive curvature)
        If False → outward normal (negative curvature)

    Returns
    -------
    k1, k2 : float
        Principal curvatures
    d1, d2 : ndarray (3,)
        Orthonormal principal directions
    normal : ndarray (3,)
        Unit normal vector
    """

    p = np.array(point, dtype=float)
    c = np.array(center, dtype=float)

    v = p - c
    dist = np.linalg.norm(v)

    if not np.isclose(dist, R):
        raise ValueError("Point is not on the sphere with given center and radius.")

    # Unit normal
    normal = v / dist

    if inward:
        normal = -normal
        k = 1.0 / R
    else:
        k = -1.0 / R

    # Build tangent basis
    # Choose vector not parallel to normal
    if abs(normal[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = np.array([0.0, 1.0, 0.0])

    # First tangent direction
    d1 = ref - np.dot(ref, normal) * normal
    d1 /= np.linalg.norm(d1)

    # Second tangent direction
    d2 = np.cross(normal, d1)

    return k, k, d1, d2, normal

def wait_for_user(fig, key=None, timeout_pause=0.1):
    """
    Wait (non-blocking for the GUI) until the user presses any key or mouse button.
    If `key` is provided (string), only that key will continue (e.g. 'q' or 'enter').
    """
    state = {"done": False}

    def _on_event(event):
        # event.key is None for mouse clicks; compare only if a key is requested
        if key is None or event.key == key:
            state["done"] = True

    cid_k = fig.canvas.mpl_connect("key_press_event", _on_event)
    cid_b = fig.canvas.mpl_connect("button_press_event", _on_event)

    try:
        while not state["done"]:
            plt.pause(timeout_pause)
    finally:
        for cid in (cid_k, cid_b):
            try:
                if cid is not None and hasattr(fig, "canvas") and hasattr(fig.canvas, "mpl_disconnect"):
                    fig.canvas.mpl_disconnect(cid)
            except Exception:
                pass

def orientation_error(Rd, Rc):
    # Orientation error
    Re = Rd @ Rc.transpose()
    trace = np.trace(Re)
    val = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(val)

    if abs(theta) < 1e-6:
        return 0, np.zeros(3)

    # skew-symmetric part -> rotation axis (unnormalized)
    rx = np.array([
        Re[2, 1] - Re[1, 2],
        Re[0, 2] - Re[2, 0],
        Re[1, 0] - Re[0, 1]
    ])

    denom = 2.0 * np.sin(theta)
    if abs(denom) < 1e-6:
        # fallback: normalize rx if possible
        n = np.linalg.norm(rx)
        axis = rx / n if n > 0 else np.zeros(3)
    else:
        axis = rx / denom

    return theta, axis

def discretise_circle_with_normals(R, N, center=(0, 0, 0), plane_normal=(0, 0, 1)):
    """
    Discretize a circle in 3D and compute outward normals at each point.

    Parameters
    ----------
    R : float
        Radius of the circle
    N : int
        Number of discretization points
    center : tuple (cx, cy, cz)
        Center of the circle
    plane_normal : tuple (nx, ny, nz)
        Normal vector defining the plane of the circle

    Returns
    -------
    points : (N, 3) ndarray
    normals : (N, 3) ndarray
    """

    center = np.asarray(center, dtype=float)
    n = np.asarray(plane_normal, dtype=float)

    # Normalize plane normal
    n_norm = np.linalg.norm(n)
    if n_norm == 0:
        raise ValueError("plane_normal cannot be zero.")
    n = n / n_norm

    # Create orthonormal basis (u, v) for the plane
    # Pick a vector not parallel to n
    if np.allclose(n, [1, 0, 0]):
        a = np.array([0, 1, 0])
    else:
        a = np.array([1, 0, 0])

    u = np.cross(n, a)
    u /= np.linalg.norm(u)

    v = np.cross(n, u)

    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)

    # Points on circle
    points = np.array([
        center + R * (np.cos(t) * u + np.sin(t) * v)
        for t in theta
    ])

    # Outward radial normals (in-plane)
    normals = np.array([
        (np.cos(t) * u + np.sin(t) * v)
        for t in theta
    ])

    return points, normals

def _sphere_normal(center, point, unit=True):
    """
    Compute the normal vector of a sphere at a given point (3D).

    Parameters:
        center : tuple (xc, yc, zc)
        point  : tuple (x, y, z)
        unit   : bool -> return unit normal if True

    Returns:
        normal vector as numpy array
    """
    xc, yc, zc = center
    x, y, z = point

    # Radial vector (normal)
    normal = np.array([x - xc, y - yc, z - zc], dtype=float)

    if unit:
        norm = np.linalg.norm(normal)
        if norm == 0:
            raise ValueError("Point cannot be the center of the sphere.")
        normal = normal / norm

    return normal

def _sine_normal(x, unit=True):
    """
    Compute normal vector of sin function at a given x.
    """
    dy_dx = np.cos((x / 0.1) + 0.5)

    # Normal vector (perpendicular to tangent)
    normal = np.array([dy_dx, -1])

    if unit:
        norm = np.linalg.norm(normal)
        normal = normal / norm

    return normal

def sine_curvature(x):
    """
    Compute curvature of the sine function at a given x.
    """
    u = (x / 0.1) + 0.5

    # First derivative
    dy_dx = np.cos(u)

    # Second derivative
    d2y_dx2 = -10 * np.sin(u)

    # Curvature formula
    curvature = d2y_dx2 / (1 + dy_dx ** 2) ** (3 / 2)

    return curvature


def implicit_curve_fundamental_forms(F, x=None, y=None, z=None, normal_sign=1):
    """
    Build symbolic differential-geometry quantities for an implicit surface F(x, y, z) = 0.

    Notes
    -----
    - Uses a local graph chart z = z(x, y), so Fz != 0 at evaluation points.
    - First and second fundamental forms are returned as 2x2 matrices I and II.

    Parameters
    ----------
    F : sympy.Expr or sympy.Equality
        Implicit surface expression (F = 0) or equation (lhs = rhs).
    x, y, z : sympy.Symbol, optional
        Symbols used in F. If omitted, inferred from F.free_symbols (must be 3 symbols).
    normal_sign : int
        +1 or -1 to choose normal orientation.

    Returns
    -------
    dict
        Symbolic expressions and callables for forms, normal, and curvature quantities.
    """
    if isinstance(F, sp.Equality):
        F = F.lhs - F.rhs

    if x is None or y is None or z is None:
        symbols = sorted(F.free_symbols, key=lambda s: s.name)
        if len(symbols) != 3:
            raise ValueError("Could not infer exactly three symbols; pass x, y, z explicitly.")
        x, y, z = symbols

    if normal_sign not in (-1, 1):
        raise ValueError("normal_sign must be +1 or -1")

    Fx = sp.diff(F, x)
    Fy = sp.diff(F, y)
    Fz = sp.diff(F, z)

    # Second derivatives used for implicit differentiation and Hessian-based quantities.
    Fxx = sp.diff(Fx, x)
    Fxy = sp.diff(Fx, y)
    Fxz = sp.diff(Fx, z)
    Fyy = sp.diff(Fy, y)
    Fyz = sp.diff(Fy, z)
    Fzz = sp.diff(Fz, z)

    zx = sp.simplify(-Fx / Fz)
    zy = sp.simplify(-Fy / Fz)

    zxx = sp.simplify(-(Fxx + 2 * Fxz * zx + Fzz * zx ** 2) / Fz)
    zyy = sp.simplify(-(Fyy + 2 * Fyz * zy + Fzz * zy ** 2) / Fz)
    zxy = sp.simplify(-(Fxy + Fxz * zy + Fyz * zx + Fzz * zx * zy) / Fz)

    E = sp.simplify(1 + zx ** 2)
    Fcoef = sp.simplify(zx * zy)
    G = sp.simplify(1 + zy ** 2)
    I_form = sp.Matrix([[E, Fcoef], [Fcoef, G]])

    W = sp.sqrt(1 + zx ** 2 + zy ** 2)
    e = sp.simplify(normal_sign * zxx / W)
    f = sp.simplify(normal_sign * zxy / W)
    g = sp.simplify(normal_sign * zyy / W)
    II_form = sp.Matrix([[e, f], [f, g]])

    S = sp.simplify(I_form.inv() * II_form)
    mean_curvature = sp.simplify(sp.trace(S) / 2)
    gaussian_curvature = sp.simplify(S.det())

    grad = sp.Matrix([Fx, Fy, Fz])
    grad_norm = sp.sqrt(Fx ** 2 + Fy ** 2 + Fz ** 2)
    normal = sp.simplify(normal_sign * grad / grad_norm)

    F_callable = sp.lambdify((x, y, z), F, "numpy")
    Fz_callable = sp.lambdify((x, y, z), Fz, "numpy")
    grad_callable = sp.lambdify((x, y, z), grad, "numpy")

    eval_callables = {
        "zx": sp.lambdify((x, y, z), zx, "numpy"),
        "zy": sp.lambdify((x, y, z), zy, "numpy"),
        "zxx": sp.lambdify((x, y, z), zxx, "numpy"),
        "zxy": sp.lambdify((x, y, z), zxy, "numpy"),
        "zyy": sp.lambdify((x, y, z), zyy, "numpy"),
        "I": sp.lambdify((x, y, z), I_form, "numpy"),
        "II": sp.lambdify((x, y, z), II_form, "numpy"),
        "shape_operator": sp.lambdify((x, y, z), S, "numpy"),
        "mean_curvature": sp.lambdify((x, y, z), mean_curvature, "numpy"),
        "gaussian_curvature": sp.lambdify((x, y, z), gaussian_curvature, "numpy"),
        "normal": sp.lambdify((x, y, z), normal, "numpy"),
    }

    def evaluate_at(x_val, y_val, z_val, check_on_surface=True, atol=1e-8):
        F_val = float(F_callable(x_val, y_val, z_val))
        if check_on_surface and abs(F_val) > atol:
            raise ValueError("Point is not on F(x, y, z) = 0 within tolerance.")
        if abs(float(Fz_callable(x_val, y_val, z_val))) <= atol:
            raise ValueError("Fz is ~0 at this point; z(x, y) chart is singular here.")

        n = np.asarray(eval_callables["normal"](x_val, y_val, z_val), dtype=float).reshape(3)
        I_val = np.asarray(eval_callables["I"](x_val, y_val, z_val), dtype=float)
        II_val = np.asarray(eval_callables["II"](x_val, y_val, z_val), dtype=float)
        S_val = np.asarray(eval_callables["shape_operator"](x_val, y_val, z_val), dtype=float)

        eigvals, eigvecs = np.linalg.eig(S_val)
        eigvals = np.real(eigvals)
        eigvecs = np.real(eigvecs)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # Convert principal directions from (dx, dy) chart coordinates to 3D tangent directions.
        r_x = np.array([1.0, 0.0, float(eval_callables["zx"](x_val, y_val, z_val))])
        r_y = np.array([0.0, 1.0, float(eval_callables["zy"](x_val, y_val, z_val))])

        d1 = eigvecs[0, 0] * r_x + eigvecs[1, 0] * r_y
        d2 = eigvecs[0, 1] * r_x + eigvecs[1, 1] * r_y
        d1 = d1 / np.linalg.norm(d1)
        d2 = d2 / np.linalg.norm(d2)

        return {
            "F": F_val,
            "I": I_val,
            "II": II_val,
            "shape_operator": S_val,
            "normal": n,
            "principal_curvatures": eigvals,
            "k1": float(eigvals[0]),
            "k2": float(eigvals[1]),
            "mean_curvature": float(np.real(eval_callables["mean_curvature"](x_val, y_val, z_val))),
            "gaussian_curvature": float(np.real(eval_callables["gaussian_curvature"](x_val, y_val, z_val))),
            "principal_directions": np.column_stack((d1, d2)),
        }

    return {
        "symbols": (x, y, z),
        "F": F,
        "F_callable": F_callable,
        "Fx": Fx,
        "Fy": Fy,
        "Fz": Fz,
        "grad": grad,
        "normal": normal,
        "zx": zx,
        "zy": zy,
        "zxx": zxx,
        "zxy": zxy,
        "zyy": zyy,
        "first_fundamental_form": I_form,
        "second_fundamental_form": II_form,
        "shape_operator": S,
        "mean_curvature": mean_curvature,
        "gaussian_curvature": gaussian_curvature,
        "callables": eval_callables,
        "evaluate_at": evaluate_at,
        "gradient_callable": grad_callable,
    }

def get_normal(which, point, center=None):
    if which == "circle":
        return _sphere_normal(point=point, center=center)
    elif which == "sine":
        return _sine_normal(x=point[0])
    else:
        raise ValueError("Invalid option for normal type")

which = "circle"  # "circle" or "sine"

circle_radius = 1.
circle_pos = np.array([0.5, 0.5, 0])
start_x_sine = -0.1070796327
reference_traj = False
# reference_traj = True
TIME = 250

FREQ = 100
DT = 1 / FREQ
STEPS = int(TIME / DT)
MAX_VEL = 0.8
MIN_VEL = -0.8
KV = 1.
KW = 1.
SIM_STEP = 5
OFFSET_DISTANCE = 0.02  # Offset distance from surface

# Create an axes
ax = plt.axes(projection="3d")

# Function to create the axes
# spb.trplot( spb.transl(1,2,3), frame='A', rviz=True, width=1, dims=[0, 10, 0, 10, 0, 10], ax=ax)

# Plot the surface and obtain initial pose required to reach the surface
if which == "circle":
    # Plot a circle
    u, v = np.mgrid[0:2 * np.pi:50j, 0:np.pi:50j]
    x = circle_radius * np.cos(u) * np.sin(v) + circle_pos[0]
    y = circle_radius * np.sin(u) * np.sin(v) + circle_pos[1]
    z = circle_radius * np.cos(v)
    ax.plot_surface(x, y, z, color="r", alpha=0.2)
    ax.view_init(elev=23, azim=-22)

    u, v = np.mgrid[0:2 * np.pi:50j, 0:np.pi:50j]
    x = 0.025 * np.cos(u) * np.sin(v) + circle_pos[0]
    y = 0.025 * np.sin(u) * np.sin(v) + circle_pos[1]
    z = 0.025 * np.cos(v)
    ax.plot_surface(x, y, z, color="r", alpha=1.)

    x_curr = [0.5, 1.5, 0. ]
    k1, k2, d1, d2, normal = principal_curvatures_sphere(R=circle_radius, center=circle_pos, point=x_curr)
    Rd = rotation_from_normal_tangent(normal, d2)

elif which == "sine":
    pass
    # x = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
    # y = 0.1 * np.sin((x / 0.1) + 0.5) + 0.2
    # z = np.zeros_like(x)  # Keep curve in XY plane
    # ax.plot(x, y, z, color='red')
    #
    # start_pos = (start_x_sine, 0.1 * np.sin((start_x_sine / 0.1) + 0.5) + 0.2, 0)
    # start_ori = get_normal(which=which, point=start_pos)
    # Rd = np.array([
    #     [start_ori[0], -start_ori[1], 0],
    #     [start_ori[1], start_ori[0], 0],
    #     [0, 0, 1]
    # ])
else:
    raise "Invalid option for which curve to track. Should be either 'circle' or 'sine'"

T_curr = SE3().Rt(Rd, x_curr)

ang_error_norm = []
ang_error_perc = []
ori_error_hist = []
T_history = [T_curr.A]

# spb.trplot(T_curr.A, frame='A', rviz=True, width=1, dims=[0, 10, 0, 10, 0, 10], ax=ax)
# plt.show()
sumofang = 0.

try:
    # Control loop
    for control_idx in range(1, STEPS):
        x_curr = T_curr.t
        R_curr = T_curr.R

        # Compute principal curvature and direction
        k1, k2, d1, d2, normal = principal_curvatures_sphere(R=circle_radius, center=circle_pos, point=x_curr)
        # Compute tangential velocity
        x_tan = 0.1 * d2
        R_des = rotation_from_normal_tangent(normal, d2)
        theta, axis = orientation_error(R_des, R_curr)
        sumofang += np.linalg.norm(theta*axis)
        omega = 1.*theta*axis/DT + k2*0.1*d1

        # TODO: Lots of issues with discretisation to obtain the next pose. However, the focus should not be here since
        # we ultimately send velocities to the robot
        # Get joint velocities then integrate to get next joint configuration
        # Drifting occurs since tangential velocity integration occurs in Euclidean space but surface is a constraint manifold
        # x_next = x_curr + (x_tan + normal*abs(circle_radius - np.linalg.norm(circle_pos - x_curr)))*DT
        x_next = x_curr + x_tan * DT  # move tangentially
        x_next = circle_pos + circle_radius * (x_next - circle_pos) / np.linalg.norm(x_next - circle_pos)
        R_next = expm(skew_symmetric(omega * DT))@R_curr
        U, _, Vt = np.linalg.svd(R_next); R_next = U @ Vt

        T_curr = SE3().Rt(R_next, x_next)

        T_history.append(T_curr.A)

except Exception as e:
    print(np.linalg.det(R_next))
    print(np.linalg.det(T_history[0][:3, :3]))
    print(f"Failed at iteration #{control_idx}")
    print(f"An exception occurred: {e}")
    traceback.print_exc()

print(sumofang)
anim = spb.tranimate(T_history[::20], interval=0.1)
plt.show()