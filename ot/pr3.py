import numpy as np
from scipy.optimize import minimize

# 1. Define objective function
def objective(variables):
    x, y = variables
    return -x*y

# 2. Define multiple inequality constraints (must be >= 0 format)
def constraint1(variables):
    x, y = variables
    return -x -y**2 + 2

def constraint2(variables):
    x, y = variables
    return -x - 2*y + 6
def constraint3(variables):
    x, y = variables
    return x 
def constraint4(variables):
    y = variables
    return y
# 3. Format constraints for SciPy
constraints = [
    {'type': 'ineq', 'fun': constraint1},
    {'type': 'ineq', 'fun': constraint2},
    {'type': 'ineq', 'fun': constraint3},
    {'type': 'ineq', 'fun': constraint4}
]

# 4. Set initial guess and run optimizer
initial_guess = [0.0, 0.0]
result = minimize(objective, initial_guess, constraints=constraints, method='SLSQP')

print("Success:", result.success)
print("Optimal Solution (x, y):", result.x)
print("Minimum Value:", result.fun)

#Prac Ques
#Q1
import numpy as np
from scipy.optimize import minimize

# 1. Define objective function
def objective(variables):
    x, y, z = variables
    return x**2+(2*y**2)+(3*z**3)

# 2. Define multiple inequality constraints (must be >= 0 format)
def constraint1(variables):
    x, y, z = variables
    return (5*x)-y-(3*z)-3

def constraint2(variables):
    x, y, z = variables
    return (2*x)+y+(2*z)-6
# 3. Format constraints for SciPy
constraints = [
    {'type': 'ineq', 'fun': constraint1},
    {'type': 'ineq', 'fun': constraint2}
]

# 4. Set initial guess and run optimizer
initial_guess = [0.0, 0.0,0.0]
result = minimize(objective, initial_guess, constraints=constraints, method='SLSQP')

print("Success:", result.success)
print("Optimal Solution (x, y, z):", result.x)
print("Minimum Value:", result.fun)

#Q2
import numpy as np
from scipy.optimize import minimize

# 1. Define objective function
def objective(variables):
    x, y = variables
    return (x-3)**2+(y-2)**2

# 2. Define multiple inequality constraints (must be >= 0 format)
def constraint1(variables):
    x, y = variables
    return (-x**2) -(y**2) + 5

def constraint2(variables):
    x, y = variables
    return -x - (2*y) + 4
def constraint3(variables):
    x= variables
    return x 
def constraint4(variables):
    y = variables
    return y
# 3. Format constraints for SciPy
constraints = [
    {'type': 'ineq', 'fun': constraint1},
    {'type': 'ineq', 'fun': constraint2},
    {'type': 'ineq', 'fun': constraint3},
    {'type': 'ineq', 'fun': constraint4}
]

# 4. Set initial guess and run optimizer
initial_guess = [0.0, 0.0]
result = minimize(objective, initial_guess, constraints=constraints, method='SLSQP')

print("Success:", result.success)
print("Optimal Solution (x, y):", result.x)
print("Minimum Value:", result.fun)
