import numpy as np
from sympy.matrices import Matrix, eye, zeros, ones, diag
from sympy import symbols, Symbol, diff

xa, xb, ya, yb = symbols("xa, xb, ya, yb")
cxa, cxb, cya, cyb = symbols("cxa, cxb, cya, cyb")
aa, ba, ab, bb = symbols("aa, ab, ab, bb")
v1, v2 =symbols("v1, v2")

f0 = (xa-xb)**2 + (ya-yb)**2  # Objective function
f1 = (xa-cxa)**2/aa**2 + (ya-cya)**2/ba**2 - 1
f2 = (xb-cxb)**2/ab**2 + (yb-cyb)**2/bb**2 - 1

L = f0 + v1*f1 + v2*f2
F = Matrix([f1, f2])

def attempt1():
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

#####################################################
# Get nabla_x and nabla_theta of L
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