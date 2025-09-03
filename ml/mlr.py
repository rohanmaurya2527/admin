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

# Generate synthetic data
X, y = make_regression(n_samples=100, n_features=2, noise=30, random_state=1)
df = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
df["Target"] = y
"""
1. Using predefined data then following lines by skipping above lines
from sklearn.datasets import load_diabetes
data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["Target"] = data.target

2. 🗃 Load your dataset
df = pd.read_csv("your_dataset.csv")
"""

"""
# 🎯 Define features and target
X = df[["Feature1", "Feature2"]]  # Replace with real feature names
y = df["Target"]  # Replace with real target column
"""
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df[["Feature 1", "Feature 2"]], df["Target"], test_size=0.2, random_state=42)

# Train model
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

# Print metrics
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# 2D Plot: Feature 1 vs Target
plt.scatter(X_test["Feature 1"], y_test, color='blue', label="Actual")
plt.scatter(X_test["Feature 1"], y_pred, color='red', label="Predicted")
plt.xlabel("Feature 1")
plt.ylabel("Target")
plt.title("Actual vs Predicted (Feature 1)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 3D Plot with regression surface
xg, yg = np.meshgrid(
    np.linspace(df["Feature 1"].min(), df["Feature 1"].max(), 20),
    np.linspace(df["Feature 2"].min(), df["Feature 2"].max(), 20)
)
zg = model.predict(np.c_[xg.ravel(), yg.ravel()]).reshape(xg.shape)

fig = px.scatter_3d(df, x="Feature 1", y="Feature 2", z="Target", title="3D Regression")
fig.add_trace(go.Surface(x=xg, y=yg, z=zg, opacity=0.5, name='Regression Surface'))
fig.show()
