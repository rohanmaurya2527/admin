import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import autocorrelation_plot
from pmdarima import auto_arima

# Load dataset
df = pd.read_csv("AirPassengers.csv", parse_dates=['Month'], index_col='Month')

# Plot original time series
plt.figure(figsize=(7,4))
plt.plot(df.index, df['#Passengers'], color='green', label='Original time series')
plt.xlabel('Year')
plt.ylabel('Number of Passengers')
plt.title('Air Passenger Dataset')
plt.legend()
plt.show()

# Function for ADF test
def adf_test(series):
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    print("Data is stationary." if result[1] <= 0.05 else "Data is not stationary.")

# Check stationarity
adf_test(df['#Passengers'])

# Differencing to make series stationary
df['#Passengers_diff'] = df['#Passengers'].diff(2)
adf_test(df['#Passengers_diff'].dropna())

# Plot autocorrelation
autocorrelation_plot(df['#Passengers_diff'].dropna())
plt.show()
plot_acf(df['#Passengers'])
plot_pacf(df['#Passengers'], method='ywm')
plt.show()

# Auto ARIMA to determine best parameters
model_auto = auto_arima(df['#Passengers'], seasonal=False, stepwise=True, trace=True)
print(model_auto.summary())

# Fit ARIMA model with chosen order
model = ARIMA(df['#Passengers'], order=(4,2,3))
model_fit = model.fit()
print(model_fit.summary())

# Forecast next 12 months
forecast_steps = 12
forecast = model_fit.forecast(steps=forecast_steps)

plt.figure(figsize=(10,5))
plt.plot(df.index, df['#Passengers'], label='Actual')
plt.plot(pd.date_range(df.index[-1], periods=forecast_steps+1, freq='ME')[1:], 
         forecast, color='red', label='Forecast')
plt.title('ARIMA Model Forecast')
plt.legend()
plt.show()
