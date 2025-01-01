import sys

import numpy as np

def q_to_rot_matr(q0, q1, q2, q3):
    return np.array([[2*(q0**2 + q1**2)-1, 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
                     [2*(q1*q2+q0*q3), 2*(q0**2 + q2**2) - 1, 2*(q2*q3 - q0*q1)],
                     [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 2*(q0**2 + q3**2)-1]])

def inside_outside_function(c, r, eps, x):
    xa_w = (((x[0] - c[0]) / r[0]) ** (2.0 / eps[1]))
    ya_w = (((x[1] - c[1]) / r[1]) ** (2.0 / eps[1]))
    za_w = (((x[2] - c[2]) / r[2]) ** (2.0 / eps[0]))
    return ((xa_w + ya_w) ** (eps[1] / eps[0])) + za_w - 1

print(q_to_rot_matr(0.9238795, 0, 0, -0.3826834))
sys.exit()
c = (-1, 0.5, -1.7)
r = (1.25, 1.5, 1.15)
eps = (1.8, 2)
x = [-0.39641155, -0.22430613, -0.55]
print(inside_outside_function(c, r, eps, x))
##########################################
c = (-1, -0.9, 0.9)
r = (1, 1.2, 0.9)
eps = (0.1, 1)
x = [-1.34607501, -1.31957271,  1.02173112]
print(inside_outside_function(c, r, eps, x))

c = (-1, 0.5, -1.7)
r = (1.25, 1.5, 1.15)
eps = (2, 2)
x = [-1.34607501, -1.31957271,  1.02173112]
print(inside_outside_function(c, r, eps, x))