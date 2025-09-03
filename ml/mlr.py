# 📦 Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# 🧪 Generate synthetic data with 2 features
X, y = make_regression(n_samples=100, n_features=2, noise=30, random_state=1)

# 📊 Create DataFrame for easy handling
df = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
df["Target"] = y

# 🎯 Define features and target
X = df[["Feature 1", "Feature 2"]]
y = df["Target"]

""" 
#🗃 Load your dataset
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
X_np = X.values  # Convert DataFrame to NumPy
x_range = np.linspace(X_np[:, 0].min(), X_np[:, 0].max(), 50)
y_range = np.linspace(X_np[:, 1].min(), X_np[:, 1].max(), 50)
X_mesh, Y_mesh = np.meshgrid(x_range, y_range)
Z_mesh = model.predict(np.c_[X_mesh.ravel(), Y_mesh.ravel()]).reshape(X_mesh.shape)

# Plot data and regression surface
fig = go.Figure([
    go.Scatter3d(x=X_np[:, 0], y=X_np[:, 1], z=y, mode='markers', name='Data'),
    go.Surface(x=X_mesh, y=Y_mesh, z=Z_mesh, opacity=0.7, name='Regression Plane')
])
fig.update_layout(title="3D Regression Surface", scene=dict(
    xaxis_title='Feature 1',
    yaxis_title='Feature 2',
    zaxis_title='Target'
))
fig.show()
