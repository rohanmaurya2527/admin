import numpy as np
from sympy import Matrix, symbols
mat=([2,-3,0],[2,-5,0],[0,0,3])
lamda = symbols('lambda')
A = Matrix(mat)
char_poly = A.charpoly(lamda)
print("Char eq:",char_poly)
eival,eivec=np.linalg.eig(mat)
for i in range(len(eival)):
    v = eivec[:, i]
    v_scaled = v / np.min(np.abs(v[v!=0]))
    print(eival[i],":",v_scaled)
