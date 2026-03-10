import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import autocorrelation_plot 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
df = pd.read_csv("AirPassengers.csv",parse_dates=['Month'], index_col='Month')
plt.figure(figsize=(7,4))
plt.plot(df.index, df['#Passengers'], label='Original time series', color='green')
plt.xlabel('Year')
plt.ylabel('Number of Passenger')
plt.title('Air Passenger Dataset')
plt.legend()
plt.show()
def adf_test(series):
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] <= 0.05:
        print("Data is stationary.")
    else:
        print("Data is not stationary.")
adf_test(df['#Passengers'])
df['#Passengers_diff'] = df['#Passengers'].diff(2).dropna()
adf_test(df['#Passengers_diff'].dropna())
autocorrelation_plot(df['#Passengers_diff'].dropna())
plt.show()
acf_diff=plot_acf(df['#Passengers'])
pacf_diff=plot_pacf(df['#Passengers'], method='ywm')
import pmdarima
from pmdarima import auto_arima
model = auto_arima(df['#Passengers'], seasonal=False, stepwise=True, trace=True)
print(model.summary())
model = ARIMA(df['#Passengers'], order=(4,2,3)) 
model_fit = model.fit()
print(model_fit.summary())
forecast_steps = 12
forecast = model_fit.forecast(steps=forecast_steps)
plt.figure(figsize=(10,5))
plt.plot(df.index, df['#Passengers'], label='Actual')
plt.plot(pd.date_range(df.index[-1], periods=forecast_steps+1, freq='ME')[1:], forecast, label='Forcast', color='red')
plt.legend()
plt.title('ARIMA Model Forecast')
plt.show()
