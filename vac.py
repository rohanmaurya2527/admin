'''
Q.1 Chinese Problem
count of heads:chicken+rabbits=35
count of legs:chicken+rabbits=94
'''
for i in range(50):
    for j in range(50):
        if (i+j)==35 and (2*i+4*j)==94:
            chi=i
            rab=j
            break
print(chi)
print(rab)

'''
Q.2 Indian Rangoli Problem
'''
def print_rangoli(size):
    import string
    alpha = string.ascii_lowercase

    width = 4 * size - 3

    for i in range(size):
        s = "-".join(alpha[i:size])
        line = (s[::-1] + s[1:]).center(width, "-")
    print(line)
    for i in range(size-2, 0,-1):
        s = "-".join(alpha[i:size])
        line = (s[::-1] + s[1:]).center(width, "-")
        print(line)
    for i in range(size):
        s = "-".join(alpha[i:size])
        line = (s[::-1] + s[1:]).center(width, "-")
        print(line)
n = int(input())
print_rangoli(n)

'''
Q.3 Time print using time.time()
'''
import time
print(time.time())
for i in range(101):
    print('1+1')
print(time.time())

'''
Q.4 Matrix Normalization
'''
import random

# create 5x5 random matrix
matrix = [[1,2,3,4,5],[8,7,6,5,3],[21,34,6,8,9],[43,5,78,32,12],[43,78,21,87,9]]

print("Original Matrix:")
for row in matrix:
    print(row)

# find minimum and maximum manually
min_val = matrix[0][0]
max_val = matrix[0][0]

for i in range(5):
    for j in range(5):
        if matrix[i][j] < min_val:
            min_val = matrix[i][j]
        if matrix[i][j] > max_val:
            max_val = matrix[i][j]

# normalize matrix
norm = []
for i in range(5):
    row = []
    for j in range(5):
        value = (matrix[i][j] - min_val) / (max_val - min_val)
        row.append(value)
    norm.append(row)

print("\nNormalized Matrix:")
for row in norm:
    print(row)

'''
Q.5 Methods to fetch integer part of array
'''
import numpy as np

# create random array
arr = np.random.uniform(0,10,5)

print("Original Array:")
print(arr)

# Method 1: using astype(int)
m1 = arr.astype(int)

# Method 2: using floor
m2 = np.floor(arr)

# Method 3: using trunc
m3 = np.trunc(arr)

# Method 4: using list comprehension with int()
m4 = [int(x) for x in arr]

print("\nMethod 1 (astype):", m1)
print("Method 2 (floor):", m2)
print("Method 3 (trunc):", m3)
print("Method 4 (int()):", m4)

'''
Q.6
'''



'''
Q.7 Checking 2 arrays are equal(using list as array not avail in Py)
'''
a=[1,6,4,7,9]
b=[4,7,2,1,89]
equal='Equal'
if len(a)==len(b):
    for i in range(len(a)):
        if a[i]!=b[i]:
            equal='Not equal'
            break
    print(equal)
else:
    print(equal)
