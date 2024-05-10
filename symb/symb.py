#! /home/sesha/anaconda3/bin/python

import sympy as sp


### Symbol Definitions ---------------------------------------------------------
# Nominal Parameters
f13 = sp.Symbol(r'\bar f_{13}', domain='QQ')
f23 = sp.Symbol(r'\bar f_{23}', domain='QQ')
f32 = sp.Symbol(r'\bar f_{32}', domain='QQ')
f31 = sp.Symbol(r'\bar f_{31}', domain='QQ')
f24 = sp.Symbol(r'\bar f_{24}', domain='QQ')
g1 = sp.Symbol(r'\bar g_{1}', domain='QQ')
g2 = sp.Symbol(r'\bar g_2', domain='QQ')
g3 = sp.Symbol(r'\bar g_3', domain='QQ')
g23 = sp.Symbol(r'\bar g_{23}', domain='QQ')
g32 = sp.Symbol(r'\bar g_{32}', domain='QQ')
g4 = sp.Symbol(r'g_4', domain='QQ')
b11 = sp.Symbol(r'\bar b_{11}', domain='QQ')
b42 = sp.Symbol(r'\bar b_{42}', domain='QQ')

# Perturbation parameters
df13 = sp.Symbol(r'\delta f_{13}', domain='QQ')
df23 = sp.Symbol(r'\delta f_{23}', domain='QQ')
df32 = sp.Symbol(r'\delta f_{32}', domain='QQ')
df31 = sp.Symbol(r'\delta f_{31}', domain='QQ')
df24 = sp.Symbol(r'\delta f_{24}', domain='QQ')
dg1 = sp.Symbol(r'\delta g_{1}', domain='QQ')
dg2F = sp.Symbol(r'\delta g_{2_F}', domain='QQ')
dg2T = sp.Symbol(r'\delta g_{2_T}', domain='QQ')
dg3 = sp.Symbol(r'\delta g_3', domain='QQ')
dg23 = sp.Symbol(r'\delta g_{23}', domain='QQ')
dg32 = sp.Symbol(r'\delta g_{32}', domain='QQ')
db11 = sp.Symbol(r'\delta b_{11}', domain='QQ')
db42 = sp.Symbol(r'\delta b_{42}', domain='QQ')

# Volume, flow-rate and urea cut off frequency, Storage
V = sp.Symbol(r'V', domain='QQ')
bv = 1/V    # sp.Symbol(r'b_v', domain='QQ')
wu = sp.Symbol(r'\omega_u', domain='QQ')
bu = sp.Symbol(r'b_u', domain='QQ')
theta = sp.Symbol(r'\Theta', domain='QQ')


# Nominal states
x1 = sp.Symbol(r'\bar x_1', domain='QQ')
x2 = sp.Symbol(r'\bar x_2', domain='QQ')
x3 = sp.Symbol(r'\bar x_3', domain='QQ')
x4 = sp.Symbol(r'\bar x_4', domain='QQ')

# nominal inputs
u1 = sp.Symbol(r'\bar u_1', domain='QQ')
u2 = sp.Symbol(r'\bar u_2', domain='QQ')
T = sp.Symbol(r'\bar T', domain='QQ')
F = sp.Symbol(r'\bar F', domain='QQ')


### Acutual system parameters --------------------------------------------------
# Nominal rate constants
k1 = sp.Symbol(r'\bar k_1', domain='QQ')
k3 = sp.Symbol(r'\bar k_3', domain='QQ')
k4F = sp.Symbol(r'\bar k_{4F}', domain='QQ')
k4R = sp.Symbol(r'k_{4R}', domain='QQ')

# The temperature coeffients of rate constants
p1 = sp.Symbol(r'p_1', domain='QQ')
p3 = sp.Symbol(r'p_3', domain='QQ')
p4F =sp.Symbol(r'p_{4F}', domain='QQ')
p4R = sp.Symbol(r'p_{4R}', domain='QQ')


#### Substitution dictionary----------------------------------------------------
subs_dict = dict()

# Nominal Parameters
subs_dict[f13] = k1
subs_dict[f23] = k4F
subs_dict[f32] = k4F * theta
subs_dict[f31] = k1 * V
subs_dict[f24] = bv * F
subs_dict[g1] = bv * F
subs_dict[g2] = bv*F + k4F*theta
subs_dict[g3] = k4R + k3
subs_dict[g23] = k4R*bv
subs_dict[g32] = k4F * V * theta
subs_dict[g4] = wu
subs_dict[b11] = bv * F
subs_dict[b42] = (wu*bu)/F

# Purturbed parameters
subs_dict[df13] = p1 * k1
subs_dict[df23] = p4F * k4F
subs_dict[df32] = p4F * k4F * theta
subs_dict[df31] = p1 * k1 * V
subs_dict[df24] = bv
subs_dict[dg1] = bv
subs_dict[dg2F] = bv
subs_dict[dg2T] = p4F*k4F*theta
subs_dict[dg3] = p4R*k4R + p3*k3
subs_dict[dg23] = p4R*k4R*bv
subs_dict[dg32] = p4F*k4F*V*theta
subs_dict[db11] = bv
subs_dict[db42] = -(wu*bu)/(F**2)


### State-space matrices--------------------------------------------------------

# A matrix
A = sp.Matrix([[ -(f13 * x3 + g1), 0, -f13*x1, 0],
               [ 0,  (-g2 + f23*x3), (f23*x2 + g23), f24],
               [-f31*x3, (-f32*x3 + g32), (-f32*x2 - g3 - f31*x1), 0],
               [0, 0, 0, -g4]], domain=
            'QQ')

# B matrix
B = sp.Matrix([[b11, 0, df13*x1*x3, db11*u1 - dg1*x1],
               [0, 0, (-dg2T*x2 + df23*x2*x3 + dg23*x3), (-dg2F*x2 + df24*x4)],
               [0, 0, (-df32*x2*x3 - dg3*x3 - df31*x1*x3 + dg32*x2), 0],
               [0, b42, 0, db42*u2]], domain='QQ')


#C matrix
C = sp.Matrix([[1, 0, 0, 0],
              [0, 0, 0, 0]])
### (sI - A) matrix ------------------------------------------------------------

s = sp.Symbol(r's', domain='QQ')
sIA = (s*sp.eye(4) - A)
