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
Q.6 Generator function that generates 10 integers
'''
# Generator function that generates 10 integers
def generate_numbers():
    for i in range(1, 11):  # generates numbers from 1 to 10
        val=random.randint(0,10)
        yield val

# Use the generator to build an array (list)
gen = generate_numbers()
array = list(gen)  # convert generator to list

print("Generated array:", array)


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

'''
Q.8 Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates.
'''
import random
import math
cartesian = [[random.randint(0, 10), random.randint(0, 10)] for _ in range(10)]
polar = []
for x, y in cartesian:
    r = (x**2 + y**2)**0.5
    theta = math.atan2(y, x)
    polar.append([r, theta])
print("Cartesian coordinates:")
for point in cartesian:
    print(point)
print("\nPolar coordinates (r, theta in radians):")
for point in polar:
    print(point)

'''
Q.9 Histogram Distribution of marks using matplotlib
'''
import random
import matplotlib.pyplot as plt

# Generate 100 random exam scores between 0 and 100
scores = [random.randint(0, 100) for _ in range(100)]

# Plot histogram
plt.hist(scores, bins=10, edgecolor='black')  # bins=10 for 10 intervals
plt.xlabel('Marks')                            # x-axis label
plt.ylabel('Number of Students')               # y-axis label
plt.title('Distribution of Exam Scores')       # plot title
plt.grid(axis='y', alpha=0.75)                # optional: add grid for clarity
plt.show()

'''
Q.11 Creating dashboard using subplots
'''
import matplotlib.pyplot as plt

# Sample Data
months = ["Jan","Feb","Mar","Apr","May"]
sales = [200, 250, 300, 280, 350]

products = ["Product A","Product B","Product C","Product D"]
profit = [40, 55, 70, 50]

distribution = [35, 25, 20, 20]

# Create subplots
plt.figure(figsize=(12,5))

# 1. Sales Line Chart
plt.subplot(1,3,1)
plt.plot(months, sales, marker='o')
plt.title("Sales Line Chart")
plt.xlabel("Month")
plt.ylabel("Sales")

# 2. Profit Bar Chart
plt.subplot(1,3,2)
plt.bar(products, profit)
plt.title("Profit Bar Chart")
plt.xlabel("Products")
plt.ylabel("Profit")

# 3. Product Distribution Pie Chart
plt.subplot(1,3,3)
plt.pie(distribution, labels=products, autopct='%1.1f%%')
plt.title("Product Distribution")

plt.tight_layout()
plt.show()

'''
Q.12 & 15  Histogram distribution of marks using seaborn with pandas dataframe
            By showing Mean and Std_Dev
'''
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Generate 200 random scores between 0 and 100
scores = np.random.randint(0, 101, 200)

# Create DataFrame
df = pd.DataFrame(scores, columns=["Exam_Scores"])

print(df.head())

sns.histplot(df["Exam_Scores"], bins=10, kde=True)

plt.title("Histogram of Exam Scores")
plt.xlabel("Scores")
plt.ylabel("Frequency")

plt.show()
print("Mean:",df['Exam_Scores'].mean())
print("Standard Deviation:",df['Exam_Scores'].std())

