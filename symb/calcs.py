from symb import *

### Calculating the determinant
det_sIA = sIA.det()
D = det_sIA
sIA_TL = sIA[0:3, 0:3]
det_factor = sIA_TL.det()
det_factor_coeffs = (det_factor.as_poly(s)).coeffs()
# Getting the detailed coefficient expressions
# det_factor_coeffs[3].as_poly(x3)
# (det_factor_coeffs[3].subs(subs_dict)).as_poly(theta, x3)


adj_sIA = (sIA.cofactor_matrix()).T
M = adj_sIA[0:2, :]
# Getting relavent matrices
N = M*B
# Getting relavent matrices
# Mfact = M[1,3].factor()
# Mfact
# flist = sp.factor_list(Mfact)
# f2 = flist[1][1][0]
# f2.as_poly(s)


G = N/D
