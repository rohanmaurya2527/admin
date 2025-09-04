import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge
from sklearn.datasets import make_regression,load_diabetes
from sklearn.metrics import r2_score
import pandas as pd
# Generate synthetic data (100 samples, 10 features)
#X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['TARGET'] = data.target
X,y=data.data,data.target
# Define a list of alphas to test
alphas = [0.01, 0.1, 1, 10, 100]

# Create subplots for coefficients and R² comparison
plt.figure(figsize=(16, 12))

# Store R² scores for each alpha value for both models
lasso_r2_scores = []
ridge_r2_scores = []
lasso_coefs = []
ridge_coefs = []

for i, alpha in enumerate(alphas):
    # Apply Lasso (L1 regularization) with current alpha
    lasso = Lasso(alpha=alpha)
    lasso.fit(X, y)
    lasso_r2 = r2_score(y, lasso.predict(X))
    lasso_r2_scores.append(lasso_r2)
    lasso_coefs.append(lasso.coef_.tolist())

    # Apply Ridge (L2 regularization) with current alpha
    ridge = Ridge(alpha=alpha)
    ridge.fit(X, y)
    ridge_r2 = r2_score(y, ridge.predict(X))
    ridge_r2_scores.append(ridge_r2)
    ridge_coefs.append(ridge.coef_.tolist())

    # Plot Lasso coefficients
    plt.subplot(2, len(alphas), i + 1)
    plt.title(f"Lasso (L1), α={alpha}\nR² = {lasso_r2:.3f}", fontsize=12)
    plt.bar(range(len(lasso.coef_)), lasso.coef_, color='blue')
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Value')

    # Plot Ridge coefficients
    plt.subplot(2, len(alphas), len(alphas) + i + 1)
    plt.title(f"Ridge (L2), α={alpha}\nR² = {ridge_r2:.3f}", fontsize=12)
    plt.bar(range(len(ridge.coef_)), ridge.coef_, color='red')
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Value')

lasso_np_arr = np.array(lasso_coefs)
lasso_coef_df = pd.DataFrame(lasso_np_arr,columns=data.feature_names)
lasso_coef_df['alpha'] = alphas
lasso_coef_df.set_index('alpha',inplace=True)

ridge_np_arr = np.array(ridge_coefs)
ridge_coef_df = pd.DataFrame(ridge_np_arr,columns=data.feature_names)
ridge_coef_df['alpha'] = alphas
ridge_coef_df.set_index('alpha',inplace=True)
