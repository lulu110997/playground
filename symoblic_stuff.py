import numpy as np
from sympy.matrices import Matrix, zeros, diag, MatrixSymbol
from sympy import symbols, diff, sqrt


def attempt1():
    xa, xb, ya, yb = symbols("xa, xb, ya, yb")
    cxa, cxb, cya, cyb = symbols("cxa, cxb, cya, cyb")
    aa, ba, ab, bb = symbols("aa, ab, ab, bb")
    v1, v2 = symbols("v1, v2")

    f0 = (xa - xb) ** 2 + (ya - yb) ** 2  # Objective function
    f1 = (xa - cxa) ** 2 / aa ** 2 + (ya - cya) ** 2 / ba ** 2 - 1
    f2 = (xb - cxb) ** 2 / ab ** 2 + (yb - cyb) ** 2 / bb ** 2 - 1

    L = f0 + v1 * f1 + v2 * f2
    F = Matrix([f1, f2])

    nabla_L_xa = diff(L, xa)
    nabla_L_ya = diff(L, ya)
    nabla_L_xb = diff(L, xb)
    nabla_L_yb = diff(L, yb)

    nabla_L = Matrix([nabla_L_xa, nabla_L_ya, nabla_L_xb, nabla_L_yb])
    Dx_nabla_L = nabla_L.jacobian([xa, ya, xb, yb])

    Dx_F = F.jacobian([xa, ya, xb, yb])

    Dxh = zeros(4)[0,:]

    diag_v_Dx_F = diag([v1, v2], unpack=True)@Dx_F

    diag_f = diag([f1, f2], unpack=True)

    Dz_g = Matrix([[Dx_nabla_L, Dx_F.transpose(), Dxh.transpose()],
                   [diag_v_Dx_F, diag_f, zeros(2)[0,:].transpose()],
                   [Dxh, zeros(2)[0,:], Matrix([0])]])

    Dtheta_nabla_L = nabla_L.jacobian([cxa, cya, cxb, cyb])

    diag_v_Dtheta_F = diag([v1, v2], unpack=True)@F.jacobian([cxa, cxb, cya, cyb])

    Dtheta_g = Matrix([Dtheta_nabla_L, diag_v_Dtheta_F, Dxh])

    Dz_g = Dz_g.subs([(xa, 0.5), (ya, 0), (xb, 2), (yb, 0),
                      (cxa, 0), (cya, 0), (cxb, 2.5), (cyb,0),
                      (aa, 0.5), (ba, 0.5), (ab, 0.5), (bb, 0.5),
                      (v1, 0.75), (v2, 0.75)])

    Dtheta_g = Dtheta_g.subs([(xa, 0.5), (ya, 0), (xb, 2), (yb, 0),
                      (cxa, 0), (cya, 0), (cxb, 2.5), (cyb,0),
                      (aa, 0.5), (ba, 0.5), (ab, 0.5), (bb, 0.5),
                      (v1, 0.75), (v2, 0.75)])

    # Shape does not work out. CBF should be 2x2 but we get a 7x4. Why is the shape like this???
    print(Dz_g.shape, Dtheta_g.shape)

def get_nabla_x_and_nabla_theta_of_L():
    xa, xb, ya, yb = symbols("xa, xb, ya, yb")
    cxa, cxb, cya, cyb = symbols("cxa, cxb, cya, cyb")
    aa, ba, ab, bb = symbols("aa, ab, ab, bb")
    v1, v2 = symbols("v1, v2")

    f0 = (xa - xb) ** 2 + (ya - yb) ** 2  # Objective function
    f1 = (xa - cxa) ** 2 / aa ** 2 + (ya - cya) ** 2 / ba ** 2 - 1
    f2 = (xb - cxb) ** 2 / ab ** 2 + (yb - cyb) ** 2 / bb ** 2 - 1

    L = f0 + v1 * f1 + v2 * f2
    F = Matrix([f1, f2])

    nabla_x_L_xa = diff(L, xa)
    nabla_x_L_ya = diff(L, ya)
    nabla_x_L_xb = diff(L, xb)
    nabla_x_L_yb = diff(L, yb)

    nabla_x_L = Matrix([nabla_x_L_xa, nabla_x_L_ya, nabla_x_L_xb, nabla_x_L_yb])
    Dx_nabla_L = nabla_x_L.jacobian([xa, ya, xb, yb])

    Dx_nabla_L = Dx_nabla_L.subs([(xa, 0.5), (ya, 0), (xb, 2), (yb, 0),
                      (cxa, 0), (cya, 0), (cxb, 2.5), (cyb,0),
                      (aa, 0.5), (ba, 0.5), (ab, 0.5), (bb, 0.5),
                      (v1, 0.75), (v2, 0.75)])


    nabla_L_xa = diff(L, cxa)
    nabla_L_ya = diff(L, cya)
    nabla_L_xb = diff(L, cxb)
    nabla_L_yb = diff(L, cyb)
    Dtheta_nabla_L = Matrix([nabla_L_xa, nabla_L_ya, nabla_L_xb, nabla_L_yb])

    Dtheta_nabla_L = Dtheta_nabla_L.subs([(xa, 0.5), (ya, 0), (xb, 2), (yb, 0),
                      (cxa, 0), (cya, 0), (cxb, 2.5), (cyb,0),
                      (aa, 0.5), (ba, 0.5), (ab, 0.5), (bb, 0.5),
                      (v1, 0.75), (v2, 0.75)])

    a=np.array([[3, -3], [1e-4, -1e-6]])
    print(a)
    print(np.linalg.inv(a))

def kkt_conditions(xa_sol, xb_sol, ca_sol, cb_sol, ra_sol, rb_sol, eps_a_sol, eps_b_sol, nu_sol):
    xa = MatrixSymbol('xa', 1, 3)
    xb = MatrixSymbol('xb' , 1, 3)
    ca = MatrixSymbol('ca', 1, 3)
    cb = MatrixSymbol('cb', 1, 3)
    ra = MatrixSymbol('ra', 1, 3)
    rb = MatrixSymbol('rb', 1, 3)
    eps_a = MatrixSymbol('eps_a', 1, 2)
    eps_b = MatrixSymbol('eps_b', 1, 2)
    nu = MatrixSymbol('nu', 1, 2)

    # Check sympy distance calculation matches numpy
    f0 = sqrt((xa - xb)@(xa - xb).transpose())
    assert np.abs(f0.subs({xa: Matrix(xa_sol), xb: Matrix(xb_sol),
                       ca: Matrix(ca_sol), cb: Matrix(cb_sol),
                       ra: Matrix(ra_sol), rb: Matrix(rb_sol),
                       eps_a: Matrix(eps_a_sol), eps_b: Matrix(eps_b_sol),
                       nu: Matrix(nu_sol)
                       }).doit()[0] - np.linalg.norm(xa_sol-xb_sol)) < 1e-8

    # Constraints
    xa_w = ((xa[0] - ca[0]) / ra[0]) ** (2.0 / eps_a[1])
    ya_w = ((xa[1] - ca[1]) / ra[1]) ** (2.0 / eps_a[1])
    za_w = ((xa[2] - ca[2]) / ra[2]) ** (2.0 / eps_a[0])
    xb_w = ((xb[0] - cb[0]) / rb[0]) ** (2.0 / eps_b[1])
    yb_w = ((xb[1] - cb[1]) / rb[1]) ** (2.0 / eps_b[1])
    zb_w = ((xb[2] - cb[2]) / rb[2]) ** (2.0 / eps_b[0])

    f1 = ((xa_w + ya_w) ** (eps_a[1] / eps_a[0])) + za_w - 1
    f2 = ((xb_w + yb_w) ** (eps_b[1] / eps_b[0])) + zb_w - 1

    # Dual feasibility
    print("dual feasibility (nu >=0):", nu.subs({nu: Matrix(nu_sol)}).doit())

    # Complementary slackness
    f1_nu = f1*nu[0]
    f1_nu = f1_nu.subs({xa: Matrix(xa_sol), xb: Matrix(xb_sol),
                        ca: Matrix(ca_sol), cb: Matrix(cb_sol),
                        ra: Matrix(ra_sol), rb: Matrix(rb_sol),
                        eps_a: Matrix(eps_a_sol), eps_b: Matrix(eps_b_sol),
                        nu: Matrix(nu_sol)
                        }).doit()
    f2_nu = f2*nu[0]
    f2_nu = f2_nu.subs({xa: Matrix(xa_sol), xb: Matrix(xb_sol),
                        ca: Matrix(ca_sol), cb: Matrix(cb_sol),
                        ra: Matrix(ra_sol), rb: Matrix(rb_sol),
                        eps_a: Matrix(eps_a_sol), eps_b: Matrix(eps_b_sol),
                        nu: Matrix(nu_sol)
                        }).doit()
    print("Complementary slackness (f1*nu1+f2*nu2=0)", f1_nu+f2_nu)


    # Primal feasibility
    print(f"primal feasibility for SQ_a",
          f1.subs({xa: Matrix(xa_sol), xb: Matrix(xb_sol),
                   ca: Matrix(ca_sol), cb: Matrix(cb_sol),
                   ra: Matrix(ra_sol), rb: Matrix(rb_sol),
                   eps_a: Matrix(eps_a_sol), eps_b: Matrix(eps_b_sol),
                   nu: Matrix(nu_sol)
                   }).doit())
    print(f"primal feasibility for SQ_b",
          f2.subs({xa: Matrix(xa_sol), xb: Matrix(xb_sol),
                   ca: Matrix(ca_sol), cb: Matrix(cb_sol),
                   ra: Matrix(ra_sol), rb: Matrix(rb_sol),
                   eps_a: Matrix(eps_a_sol), eps_b: Matrix(eps_b_sol),
                   nu: Matrix(nu_sol)
                   }).doit())

    # Stationarity condition
    f0_diff = f0.diff(xa)
    f1_diff = Matrix([f1.diff(xa[0]), f1.diff(xa[1]), f1.diff(xa[2])])
    f2_diff = Matrix([f2.diff(xa[0]), f2.diff(xa[1]), f2.diff(xa[2])])
    dL_dxa = f0_diff + nu[0]*f1_diff + nu[1]*f2_diff
    print("stationarity condition")
    print(dL_dxa.subs({xa: Matrix(xa_sol), xb: Matrix(xb_sol),
                       ca: Matrix(ca_sol), cb: Matrix(cb_sol),
                       ra: Matrix(ra_sol), rb: Matrix(rb_sol),
                       eps_a: Matrix(eps_a_sol), eps_b: Matrix(eps_b_sol),
                       nu: Matrix(nu_sol)
                       }).doit())

    print()

