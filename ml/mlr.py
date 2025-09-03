# 📦 Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

# 🧪 Generate synthetic data with 2 features
X, y = make_regression(n_samples=100, n_features=2, noise=30, random_state=1)

# 📊 Create DataFrame for easy handling
df = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
df["Target"] = y
""" 🗃 Load your dataset
df = pd.read_csv("your_dataset.csv")

# 🎯 Define features and target
X = df[["Feature1", "Feature2"]]  # Replace with real feature names
y = df["Target"]  # Replace with real target column
"""
# ✂️ Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[["Feature 1", "Feature 2"]], df["Target"], test_size=0.2, random_state=42)

# 🤖 Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 🔮 Predict on test data
y_pred = model.predict(X_test)

# 📈 Print model performance
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# 📉 2D Visualization: Feature 1 vs Target
plt.figure(figsize=(8, 5))
plt.scatter(X_test["Feature 1"], y_test, color='blue', label="Actual")
plt.scatter(X_test["Feature 1"], y_pred, color='red', label="Predicted")
plt.title("Feature 1 vs Target (Actual vs Predicted)")
plt.xlabel("Feature 1")
plt.ylabel("Target")
plt.legend()
plt.grid(True)
plt.show()

# 🌀 3D Visualization: Feature 1, Feature 2 vs Target
fig = px.scatter_3d(df, x='Feature 1', y='Feature 2', z='Target', title="3D Scatter of Features vs Target")
fig.show()
