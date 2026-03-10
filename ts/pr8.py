import pandas as pd
from pmdarima import auto_arima
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
df2 = pd.read_csv(r"C:\Users\User 2\Downloads\AirPassengers (1).csv",parse_dates=['Month'], index_col='Month')
plt.figure(figsize=(7,4))
plt.plot(df2.index, df['Passengers'], label='Original time series', color='green')
plt.xlabel('Year')
plt.ylabel('Number of Passenger')
plt.title('Air Passenger Dataset')
plt.legend()
plt.show()
stepwise_fit = auto_arima(df['Passengers'], seasonal=True, m=12, trace=True)
stepwise_fit.summary()
model = SARIMAX(df['Passengers'], orders=(2,1,1), seasonal_orders= (0,1,0,12))
sarima_result = model.fit()
print(sarima_result.summary())
forecast = sarima_result.get_forecast(steps=12)
conf_int = forecast.conf_int()
plt.figure(figsize=(10,5))
plt.plot(df, label="observed")
plt.plot(forecast.predicted_mean, label="Forecast", color='red')
plt.fill_between(conf_int.index, conf_int.iloc[:,0], conf_int.iloc[:,1], color='pink', alpha=0.3)
plt.legend()
plt.show()
