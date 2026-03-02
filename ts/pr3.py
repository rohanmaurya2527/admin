import pandas as pd
from statsmodels.tsa.stattools import adfuller
# Create DataFrame
df = pd.DataFrame({
    "date": pd.date_range(start="2020-01-01", periods=100, freq="D"),
    "value": [x * 1.1 for x in range(100)]
})
# Perform ADF test
adf_stat, p_value, _, _, critical_values, _ = adfuller(df["value"])
print("ADF Statistic:", adf_stat)
print("p-value:", p_value)
print("Critical Values:")
for k, v in critical_values.items():
    print(f"   {k}: {v}")
print("The Series is Stationary." if p_value <= 0.05 
      else "The Series is not Stationary.")
