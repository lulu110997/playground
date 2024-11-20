from sympy import *

init_printing(use_unicode=True)

xa, ya, za, xb, yb, zb  = symbols('x[0] x[1] x[2] x[3] x[4] x[5]')
cxa, cya, cza, cxb, cyb, czb = symbols('cxa cya cza cxb cyb czb')
aa, ba, ca, ab, bb, cb = symbols('aa ba ca ab bb cb')
e1a, e2a, e1b, e2b  = symbols('e1a e2a e1b e2b')

# F = ()**2/e2a