import pandas as pd

dates = pd.date_range(start='2020-01-01', end='2020-01-31', freq='D').tolist()
numbers = [43, 31, 1, 10, 20, 24, 26, 27, 34, 35, 36, 37, 31, 21, 20, 19, 
           18, 19, 20, 24, 28, 29, 20, 32, 34, 35, 36, 30, 32, 30, 23]

# Original dataframe
df = pd.DataFrame({'Dates': dates, 'Price': numbers})
df.plot.line(x='Dates', y='Price')

# Moving averages using loop
for w in [3, 4, 5]:
    df_ma = df.copy()
    col_name = f"{w} - Moving Average"
    df_ma[col_name] = df_ma['Price'].rolling(window=w).mean()
    df_ma.plot.line(x='Dates', y=['Price', col_name])
