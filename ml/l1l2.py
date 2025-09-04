import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, load_diabetes
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Generate synthetic regression data
x, y = make_regression(n_samples=100, noise=20, n_features=1, random_state=2)

# Visualization of data
plt.scatter(x, y, c='purple')
plt.title("Generated Data: y vs x")
plt.show()

# Train Linear, Lasso, and Ridge regression models with different alpha values
alphas = [0.1, 1, 10, 100]  # Removed alpha=0 from Lasso
plt.plot(x, y, 'b.', label='Data')

# Linear Regression model
lr = LinearRegression()
lr.fit(x, y)
plt.plot(x, lr.predict(x), 'y-', label="Linear Regression")

# Lasso and Ridge models with different alphas
for alpha in alphas:
    lasso = Lasso(alpha=alpha)
    ridge = Ridge(alpha=alpha)
    lasso.fit(x, y)
    ridge.fit(x, y)
    plt.plot(x, lasso.predict(x), label=f'Lasso alpha={alpha}')
    plt.plot(x, ridge.predict(x), label=f'Ridge alpha={alpha}')

plt.legend()
plt.title("Linear vs Lasso vs Ridge Regression")
plt.show()

# Load diabetes dataset and split into train-test
data = load_diabetes()
x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=2)

# Coefficients and R2 scores for Lasso and Ridge
coefs, r2_scores = {'Lasso': [], 'Ridge': []}, {'Lasso': [], 'Ridge': []}

for alpha in [0.1, 1, 10, 100]:  # Removed alpha=0 from Lasso
    lasso = Lasso(alpha=alpha).fit(x_train, y_train)
    ridge = Ridge(alpha=alpha).fit(x_train, y_train)
    
    coefs['Lasso'].append(lasso.coef_)
    coefs['Ridge'].append(ridge.coef_)
    
    r2_scores['Lasso'].append(r2_score(y_test, lasso.predict(x_test)))
    r2_scores['Ridge'].append(r2_score(y_test, ridge.predict(x_test)))

# Plot coefficients for Lasso and Ridge
plt.figure(figsize=(14, 9))
for i, alpha in enumerate([0.1, 1, 10, 100]):  # Removed alpha=0 from Lasso
    plt.subplot(3, 2, i+1)  # 3x2 grid
    plt.bar(data.feature_names, coefs['Lasso'][i], alpha=0.7, label='Lasso')
    plt.bar(data.feature_names, coefs['Ridge'][i], alpha=0.3, label='Ridge')
    plt.title(f'Alpha={alpha}, Lasso R²={round(r2_scores["Lasso"][i], 2)}, Ridge R²={round(r2_scores["Ridge"][i], 2)}')
    plt.legend()

plt.tight_layout()
plt.show()
