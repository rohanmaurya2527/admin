import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# Generate non-stationary time series
np.random.seed(42)
n = 100
ts = pd.Series(np.cumsum(np.random.normal(size=n)))

# Function to plot series and run ADF test
def plot_and_adf(series, title="Time Series", show_diff=False):
    plt.figure(figsize=(10,4))
    plt.plot(series, label=title)
    plt.title(title)
    plt.legend()
    plt.show()
    
    result = adfuller(series.dropna())
    print(f"\nADF Test for {title}")
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"  {key}: {value:.4f}")
    print("Stationary" if result[1] <= 0.05 else "Not stationary")
    
    # Optionally return differenced series
    if show_diff:
        diff_series = series.diff().dropna()
        return diff_series

# --- Usage ---

# Original series
diff_ts = plot_and_adf(ts, title="Original Series", show_diff=True)

# Differenced series
plot_and_adf(diff_ts, title="Differenced Series")
