#2-var
import sympy as sp

# 1. Define symbolic variables
# 'l' represents the Lagrange multiplier lambda (as 'lambda' is a reserved keyword in Python)
x, y, l = sp.symbols('x y l', real=True)

# 2. Define the objective and constraint functions
f = 81*x**2 + y**2
g = 4*x**2 + y - 1

# 3. Construct the Lagrangian function: L = f + l * g
L = f + l * g

# 4. Compute partial derivatives
grad_L = [sp.diff(L, var) for var in (x, y, l)]

# 5. Solve the system of equations where all partial derivatives equal 0
solutions = sp.solve(grad_L, (x, y, l), dict=True)

# 6. Display results
for sol in solutions:
    print(f"Optimal Point: x = {sol[x]}, y = {sol[y]}")
    print(f"Lagrange Multiplier (lambda) = {sol[l]}")
    print(f"Minimum Value = {f.subs(sol)}")


#3-var
import sympy as sp

# 1. Define symbolic variables
# 'l' represents the Lagrange multiplier lambda (as 'lambda' is a reserved keyword in Python)
x, y, z, l = sp.symbols('x y z l', real=True)

# 2. Define the objective and constraint functions
f = x*y*z
g = x + 9*y**2 + z**2 - 4

# 3. Construct the Lagrangian function: L = f + l * g
L = f + l * g

# 4. Compute partial derivatives
grad_L = [sp.diff(L, var) for var in (x, y, z, l)]

# 5. Solve the system of equations where all partial derivatives equal 0
solutions = sp.solve(grad_L, (x, y, z, l), dict=True)

# 6. Display results
for sol in solutions:
    print(f"Optimal Point: x = {sol[x]}, y = {sol[y]}, z = {sol[z]}")
    print(f"Lagrange Multiplier (lambda) = {sol[l]}")
    print(f"Minimum Value = {f.subs(sol)}")
