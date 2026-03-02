import pandas as pd
import matplotlib.pyplot as plt
# Create data
dates = pd.date_range(start='2020-01-01', end='2020-01-31', freq='D')
numbers = [43, 31, 1, 10, 20, 24, 26, 27, 34, 35, 36, 37, 31, 21, 20,
           19, 18, 19, 20, 24, 28, 29, 20, 32, 34, 35, 36, 30, 32, 30, 23]
# Create single DataFrame
df = pd.DataFrame({'Dates': dates, 'Price': numbers})
# Original line plot
df.plot.line(x='Dates', y='Price')
plt.show()
# Add moving averages
for w in [3, 4, 5]:
    df[f'{w} - Moving Average'] = df['Price'].rolling(window=w).mean()
# Plot each moving average separately
for w in [3, 4, 5]:
    df.plot.line(x='Dates', y=['Price', f'{w} - Moving Average'])
    plt.show()
