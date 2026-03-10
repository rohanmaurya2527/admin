import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing
# Load data
data = pd.read_csv("AirPassengers.csv", parse_dates=['Month'], index_col='Month')
print(data.head())
series = data.squeeze()   # Convert to Series
plt.figure()
plt.plot(series, label="Original Data")
plt.show()
# Function to fit, forecast and plot
def fit_and_plot(model, title):
    fit = model.fit()
    forecast = fit.forecast(40)
    plt.figure()
    plt.plot(series, label="Original Data")
    plt.plot(fit.fittedvalues, label="Fitted Values")
    plt.plot(forecast, label="Forecast")
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Number of Passengers")
    plt.legend()
    plt.show()
# Single Exponential Smoothing
fit_and_plot(SimpleExpSmoothing(series), "Single Exponential Smoothing")
# Double Exponential Smoothing (Holt)
fit_and_plot(Holt(series), "Double Exponential Smoothing")
# Triple Exponential Smoothing (Holt-Winters)
fit_and_plot(
    ExponentialSmoothing(series, seasonal_periods=12, trend='add', seasonal='add'),
    "Triple Exponential Smoothing"
)
