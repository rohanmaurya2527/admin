import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.ar_model import AutoReg
df = pd.read_csv('AirPassengers.csv', parse_dates=['Month'], index_col='Month')
# Original Time Series Plot
plt.figure(figsize=(7,4))
plt.plot(df.index, df['#Passengers'], label='Original Time Series', color='green')
plt.title('Air Passengers Dataset')
plt.xlabel('Year')
plt.ylabel('Number of Passengers')
plt.legend()
plt.show()
# ACF and PACF plots using loop
plots = [
    (plot_acf, "Autocorrelation Function (ACF)"),
    (plot_pacf, "Partial Autocorrelation Function (PACF)")
]
for func, title in plots:
    func(df['#Passengers'], lags=40, alpha=0.05)
    plt.title(title)
    plt.show()
# ACF and PACF values
acf_values = acf(df['#Passengers'], nlags=40, fft=True)
pacf_values = pacf(df['#Passengers'], nlags=40, method='ywm')
print("Autocorrelation Values:")
print(acf_values)
print("\nPartial Autocorrelation Values:")
print(pacf_values)
# AIC calculation for lag selection
aic_values = []
for lag in range(1, 50):
    model = AutoReg(df['#Passengers'], lags=lag).fit()
    aic_values.append(model.aic)
best_lag = np.argmin(aic_values) + 1
for lag, aic in enumerate(aic_values, 1):
    print(f"Lag {lag}: AIC = {aic}")
print("\nBest Lag based on AIC:", best_lag)
