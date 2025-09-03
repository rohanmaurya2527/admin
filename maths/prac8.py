import pandas as pd
df=pd.read_csv('Desktop/DSAI/Datasets/AAPL.csv')
df['Volume'].plot()

df.plot(subplots=True,figsize=(6,6))

df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')
df_month=df.resample("M").mean()
fig,ax=plt.subplots(figsize=(6,6))
ax.bar(df_month['2015':].index,df_month.loc['2015':,"Volume"],width=25,align='center')
