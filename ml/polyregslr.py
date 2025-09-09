import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load the dataset
data = pd.read_excel("Datasets/PolyData.xlsx")
data.drop("Unnamed: 0", axis=1, inplace=True)

# Split into input (X) and output (y)
X = data.iloc[:, 0].values.reshape(-1, 1)
y = data.iloc[:, -1].values

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Create polynomial features (degree = 2)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Fit the polynomial regression model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Predict
y_pred = model.predict(X_test_poly)

# Evaluate
print("R2 Score:", r2_score(y_test, y_pred))

# Plotting
plt.scatter(X, y, color='blue', label="Data Points")
X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_plot_poly = poly.transform(X_plot)
y_plot = model.predict(X_plot_poly)
plt.plot(X_plot, y_plot, color='red', label="Polynomial Fit")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.title("Polynomial Regression (Degree 2)")
plt.show()
