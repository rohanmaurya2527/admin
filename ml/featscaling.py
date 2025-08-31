import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load the data
df = pd.read_csv("Rizwan/Datasets/Social_Network_Ads.csv")

# Apply Standardization and Normalization
std = StandardScaler()
norm = MinMaxScaler()

df[['std_Age', 'std_Salary']] = std.fit_transform(df[['Age', 'EstimatedSalary']])
df[['norm_Age', 'norm_Salary']] = norm.fit_transform(df[['Age', 'EstimatedSalary']])

# Plot 1: Original Data
plt.figure(figsize=(5, 4))
plt.scatter(df['Age'], df['EstimatedSalary'], color='blue')
plt.xlabel("Age")
plt.ylabel("Salary")
plt.title("Original Data")
plt.grid(True)
plt.show()

# Plot 2: After Standardization
plt.figure(figsize=(5, 4))
plt.scatter(df['std_Age'], df['std_Salary'], color='green')
plt.xlabel("Standardized Age")
plt.ylabel("Standardized Salary")
plt.title("After Standardization")
plt.grid(True)
plt.show()

# Plot 3: After Normalization
plt.figure(figsize=(5, 4))
plt.scatter(df['norm_Age'], df['norm_Salary'], color='red')
plt.xlabel("Normalized Age")
plt.ylabel("Normalized Salary")
plt.title("After Normalization")
plt.grid(True)
plt.show()
