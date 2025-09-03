import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv("Arya/Datasets/Ice Cream.csv")
X = df["Temperature"].to_numpy()
y = df["Revenue"].to_numpy()

# Train-test split and scikit-learn regression (for comparison)
X_train, X_test, y_train, y_test = train_test_split(X.reshape(-1, 1), y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
print("Sklearn Coefficient:", model.coef_[0])

# Initialize parameters for Gradient Descent
m, b = 0, 3
lr = 0.001
epochs = 50

# History tracking
m_history, b_history, loss_history = [], [], []

# Gradient Descent Loop
for _ in range(epochs):
    y_pred = m * X + b
    error = y - y_pred

    # Gradients
    grad_m = -2 * np.dot(error, X)
    grad_b = -2 * np.sum(error)

    # Parameter updates
    m -= lr * grad_m
    b -= lr * grad_b

    # Save history
    m_history.append(m)
    b_history.append(b)
    loss = np.sqrt(mean_squared_error(y, y_pred))
    loss_history.append(loss)

# Final results
print(f"Gradient Descent → Slope: {m:.4f}, Intercept: {b:.4f}, RMSE: {loss:.4f}")

# Plot results
plt.scatter(X, y, color='red', label='Data Points')
plt.plot(X, m * X + b, color='blue', label='Fitted Line')
plt.title("Linear Regression using Gradient Descent")
plt.xlabel("Temperature")
plt.ylabel("Revenue")
plt.legend()
plt.grid(True)
plt.show()
