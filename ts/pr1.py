import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

# Load dataset
df = pd.read_csv("AirPassengers.csv")

# STL decomposition
stl = STL(df['#Passengers'], period=13, robust=True)
result = stl.fit()

# Store all components including original
components = [
    ("Original Time Series", df['#Passengers'], "green"),
    ("Trend Component", result.trend, "red"),
    ("Seasonal Component", result.seasonal, "blue"),
    ("Residual Component", result.resid, "black")
]

# Loop for plotting
for title, data, color in components:
    plt.figure(figsize=(7,3))
    plt.plot(data, '-', color=color)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Passengers")
    plt.show()
