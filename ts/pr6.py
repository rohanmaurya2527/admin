import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
# Generate synthetic time series
np.random.seed(42)
n = 100
ts = pd.Series(50 + 0.8*np.arange(n) + np.random.normal(0, 5, n))
# Train-test split
train = ts[:80]
test = ts[80:]
# Fit AR(5) model
model = AutoReg(train, lags=5).fit()
# Predict
predictions = model.predict(start=len(train), end=len(ts)-1)
# Evaluate
mse = mean_squared_error(test, predictions)
print("MSE:", mse)
# Plot results
plt.plot(test.index, test, label="Actual")
plt.plot(test.index, predictions, "--", label="Predicted")
plt.legend()
plt.show()
