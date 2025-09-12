import numpy as np
from fractions import Fraction
mat = ([1, 0, 1],
       [0, 2, 3],
       [1, 2, 1])
inv_mat = np.linalg.inv(np.array(mat))
inv_mat_frac = np.vectorize(lambda x: Fraction(x).limit_denominator())(inv_mat)
det=np.linalg.det(mat)
cofactor = inv_mat.T * det
adjacent = inv_mat * det
print("Determinant:",det)
print(cofactor)
print(adjacent)
print(inv_mat_frac)
