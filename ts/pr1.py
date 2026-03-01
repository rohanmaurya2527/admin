import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
# 1. Load the dataset
df = pd.read_csv("/kaggle/input/datasets/chirag19/air-passengers/AirPassengers.csv")
# 2. Plot original time series
plt.figure(figsize=(7,4))
plt.plot(df['#Passengers'], color='green')
plt.title("Air Passengers Dataset")
plt.xlabel("Time")
plt.ylabel("Number of Passengers")
plt.show()
# 3. Apply STL decomposition
stl = STL(df['#Passengers'], period=13, robust=True)
result = stl.fit()
# 4. Plot Trend
plt.figure(figsize=(7,2))
plt.plot(result.trend, color='red')
plt.title("Trend Component")
plt.show()
# 5. Plot Seasonal
plt.figure(figsize=(7,2))
plt.plot(result.seasonal, color='blue')
plt.title("Seasonal Component")
plt.show()
# 6. Plot Residual
plt.figure(figsize=(7,2))
plt.plot(result.resid, color='black')
plt.title("Residual Component")
plt.show()
