import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load dataset
df = pd.read_csv("AirPassengers.csv",parse_dates=['Month'], index_col='Month')

# Plot original time series
plt.figure(figsize=(7,4))
plt.plot(df.index, df['#Passengers'], color='green', label='Original time series')
plt.xlabel('Year')
plt.ylabel('Number of Passengers')
plt.title('Air Passenger Dataset')
plt.legend()
plt.show()

# Auto ARIMA to find best seasonal model
stepwise_fit = auto_arima(df['#Passengers'], seasonal=True, m=12, trace=True)
print(stepwise_fit.summary())

# Fit SARIMA model
model = SARIMAX(df['#Passengers'], order=(2,1,1), seasonal_order=(0,1,0,12))
sarima_result = model.fit()
print(sarima_result.summary())

# Forecast next 12 periods
forecast = sarima_result.get_forecast(steps=12)
conf_int = forecast.conf_int()

# Plot forecast with confidence intervals
plt.figure(figsize=(10,5))
plt.plot(df['#Passengers'], label='Observed')
plt.plot(forecast.predicted_mean, color='red', label='Forecast')
plt.fill_between(conf_int.index, conf_int.iloc[:,0], conf_int.iloc[:,1], color='pink', alpha=0.3)
plt.legend()
plt.title('SARIMA Forecast')
plt.show()
