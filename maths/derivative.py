import sympy as sp
x = sp.Symbol('x')
expr = (x**2)*sp.acos(x)+sp.log(x)
derivative = sp.diff(expr, x)
print(derivative)
