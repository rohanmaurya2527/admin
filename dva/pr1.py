#Button code
from bokeh.io import show
from bokeh.models import Button, CheckboxGroup, RadioGroup, CustomJS
from bokeh.layouts import column

labels = ["first", "second", "third"]

# Create widgets
btn = Button(label="Click Me", button_type="success")
chk = CheckboxGroup(labels=labels, active=[0, 2])
rad = RadioGroup(labels=labels, active=0)

# Add JavaScript callbacks
btn.js_on_click(CustomJS(code="alert('Button Clicked')"))
chk.js_on_change("active", CustomJS(code="console.log('Checked:', this.active);"))
rad.js_on_change("active", CustomJS(code="alert('Selected: ' + this.labels[this.active]);"))

# Layout and show
show(column(btn, chk, rad))

#Plotly
import plotly.express as px
df=px.data.tips()
fig=px.line(df,y="tip",line_dash='sex',color='sex')
fig1=px.bar(df,x='day',y='tip',color='sex')
fig2=px.scatter(df,x='total_bill',y='tip',color='time')
fig3=px.histogram(df,x="total_bill", color='sex')
fig4=px.pie(df,values="total_bill",names="day")
fig.show()
fig1.show()
fig2.show()
fig3.show()
fig4.show()

#Matplot
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
df=pd.read_csv('tips.csv')
plt.bar(df['day'],df['tip'],color='green')
plt.show()
plt.scatter(df['total_bill'],df['tip'])
plt.show()
x,y=df[['total_bill']],df[['tip']]
lr=LinearRegression()
lr.fit(x,y)
plt.plot(x,lr.predict(x),c='r')
plt.show()
plt.hist(y,bins=10)
plt.show()
plt.pie([1,3,5,7,9],labels=['a','b','c','d','e'],autopct='%1.1f%%')
plt.show()

#Seaborn
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# reading the database
data = pd.read_csv("tips.csv")
sns.scatterplot(x=df['total_bill'], y=df['tip'], hue=df['sex'])
plt.show()
sns.histplot(x=df['total_bill'],kde=True,hue=df['sex'])
plt.show()
sns.barplot(x=df['day'],y=df['tip'],ci=None)
plt.show()
sns.lineplot(x=df['total_bill'],y=df['tip'])
plt.show()
sns.pairplot(data=df,hue='sex')
plt.show()
