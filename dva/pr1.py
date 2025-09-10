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

#Bokeh
from bokeh.plotting import figure,show
g=figure(title="Bokeh Bar Chart")
g.vbar(x=data['total_bill'],top=data['tip'],legend_label="bill vs tip",color='red')
g.vbar(x=data['tip'],top=data['size'],legend_label="tip vs size",color='green')
g.legend.click_policy="hide"
show(g)
#Bokeh

# Import necessary libraries
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
from bokeh.io import output_notebook
import numpy as np
import pandas as pd

# 1. Line Plot using Bokeh
# Generate data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a ColumnDataSource
source = ColumnDataSource(data=dict(x=x, y=y))

# Create a figure
p = figure(title="Bokeh Line Plot", x_axis_label='X', y_axis_label='Y')

# Add a line renderer
p.line('x', 'y', source=source, line_width=2, color="blue", legend_label="Sine Wave")

# Show the plot
output_notebook()  # To display the plot inline in Jupyter Notebooks
show(p)

# 2. Scatter Plot using Bokeh
# Generate random data
x = np.random.rand(50)
y = np.random.rand(50)

# Create a ColumnDataSource
source = ColumnDataSource(data=dict(x=x, y=y))

# Create a figure
p = figure(title="Bokeh Scatter Plot", x_axis_label='X', y_axis_label='Y')

# Add a scatter renderer
p.scatter('x', 'y', source=source, size=8, color="red", legend_label="Random Points")

# Show the plot
show(p)

# 3. Bar Plot using Bokeh
# Data for bar plot
categories = ['A', 'B', 'C', 'D', 'E']
values = [10, 24, 18, 6, 12]

# Create a ColumnDataSource
source = ColumnDataSource(data=dict(categories=categories, values=values))

# Create a figure
p = figure(x_range=categories, title="Bokeh Bar Plot", x_axis_label='Category', y_axis_label='Value')

# Add a bar renderer
p.vbar(x='categories', top='values', source=source, width=0.5, color="green", legend_label="Values")

# Show the plot
show(p)

# 4. Histogram using Bokeh
# Generate random data
data = np.random.randn(1000)

# Create a figure
p = figure(title="Bokeh Histogram", x_axis_label='Value', y_axis_label='Frequency', tools="pan,box_zoom,reset")

# Create histogram and add it to the figure
hist, edges = np.histogram(data, bins=30)

# Add the histogram as a quad renderer
p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="purple", line_color="white", legend_label="Histogram")

# Show the plot
show(p)

# 5. Pie Chart using Bokeh
# Data for pie chart
sizes = [20, 30, 40, 10]
labels = ['A', 'B', 'C', 'D']

# Create a figure
p = figure(title="Bokeh Pie Chart")

# Pie chart data
angles = [2 * np.pi * (i / sum(sizes)) for i in sizes]
p.wedge(x=0, y=1, radius=0.4, start_angle=0, end_angle=angles[0], color="red", legend_label=labels[0])

# Add additional wedges
angle_offset = angles[0]
for i in range(1, len(sizes)):
    p.wedge(x=0, y=1, radius=0.4, start_angle=angle_offset, end_angle=angle_offset + angles[i], color="blue", legend_label=labels[i])
    angle_offset += angles[i]

# Show the plot
show(p)

# 6. Heatmap using Bokeh
# Create a random correlation matrix for heatmap
data = np.random.rand(10, 10)

# Create a ColumnDataSource from the matrix
source = ColumnDataSource(data=dict(x=np.tile(np.arange(10), 10), y=np.repeat(np.arange(10), 10), value=data.flatten()))

# Create a figure
p = figure(title="Bokeh Heatmap", x_axis_label='X', y_axis_label='Y', tools="hover")

# Add the heatmap as a square renderer
p.square(x='x', y='y', source=source, size=10, color='value', legend_label="Heatmap")

# Show the plot
show(p)

# 7. Line Plot with multiple lines using Bokeh
# Generate data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create a ColumnDataSource
source = ColumnDataSource(data=dict(x=x, y1=y1, y2=y2))

# Create a figure
p = figure(title="Bokeh Line Plot with Multiple Lines", x_axis_label='X', y_axis_label='Y')

# Add two lines
p.line('x', 'y1', source=source, line_width=2, color="blue", legend_label="Sine Wave")
p.line('x', 'y2', source=source, line_width=2, color="orange", legend_label="Cosine Wave")

# Show the plot
show(p)
