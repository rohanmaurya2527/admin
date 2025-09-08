import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")
from sklearn.datasets import load_diabetes
data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["Target"] = data.target
# 🎯 Define features and target
X = df[["age", "bmi"]]  # Replace with real feature names
y = df["Target"]  # Replace with real target column
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


# 3D Plot with regression surface
xg, yg = np.meshgrid(
    np.linspace(df["age"].min(), df["age"].max(), 20),
    np.linspace(df["bmi"].min(), df["bmi"].max(), 20)
)
zg = model.predict(poly.transform(np.c_[xg.ravel(), yg.ravel()])).reshape(xg.shape)

fig = px.scatter_3d(df, x="age", y="bmi", z="Target", title="3D Regression")
fig.add_trace(go.Surface(x=xg, y=yg, z=zg, opacity=0.5, name='Regression Surface'))
fig.show()
