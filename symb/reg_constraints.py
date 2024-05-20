import sympy as sp

a, b, z, w = sp.symbols('alpha beta xi omega', real=True)
a3, a2, a1, a0 = sp.symbols('a_3 a_2 a_1 a_0', real=True)

# Equations

eq1 = a3 - ( (a + b) + (2*z*w) )
eq2 = a2 - ( (a*b) + w**2 +  (a + b)*(2*z*w) )
eq3 = a1 - ( (2*z*w)*(a*b) + (a + b)*(w**2) )
eq4 = a0 - ( (a*b)*(w**2) )
