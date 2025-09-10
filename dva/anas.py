#Matplotlib

# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')
# 1. Line Plot
# Create data for plotting
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Plot the data
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='Sine Wave', color='blue')
plt.title('Line Plot: Sine Wave')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
plt.show()

# 2. Bar Plot
# Data for bar plot
categories = ['A', 'B', 'C', 'D', 'E']
values = [10, 24, 18, 6, 12]

# Plot the data
plt.figure(figsize=(8, 6))
plt.bar(categories, values, color='green')
plt.title('Bar Plot')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.show()

# 3. Scatter Plot
# Data for scatter plot
x = np.random.rand(50)
y = np.random.rand(50)

# Plot the data
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='red', marker='o')
plt.title('Scatter Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

# 4. Histogram
# Data for histogram
data = np.random.randn(1000)

# Plot the data
plt.figure(figsize=(8, 6))
plt.hist(data, bins=30, color='purple', edgecolor='black')
plt.title('Histogram of Random Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# 5. Pie Chart
# Data for pie chart
labels = ['A', 'B', 'C', 'D']
sizes = [20, 30, 40, 10]

# Plot the data
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title('Pie Chart')
plt.show()

#Seaborn

# Import necessary libraries
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Line Plot using Seaborn
# Create a simple dataset
x = np.linspace(0, 10, 100)
y = np.sin(x)
data = pd.DataFrame({'x': x, 'y': y})

# Plot the data
plt.figure(figsize=(8, 6))
sns.lineplot(x='x', y='y', data=data, color='blue')
plt.title('Seaborn Line Plot')
plt.show()

# 2. Scatter Plot using Seaborn
# Create random data
x = np.random.rand(100)
y = np.random.rand(100)
data = pd.DataFrame({'x': x, 'y': y})

# Plot the data
plt.figure(figsize=(8, 6))
sns.scatterplot(x='x', y='y', data=data, color='red')
plt.title('Seaborn Scatter Plot')
plt.show()

# 3. Bar Plot using Seaborn
# Data for bar plot
categories = ['A', 'B', 'C', 'D', 'E']
values = [10, 24, 18, 6, 12]
data = pd.DataFrame({'Category': categories, 'Value': values})

# Plot the data
plt.figure(figsize=(8, 6))
sns.barplot(x='Category', y='Value', data=data, palette='viridis')
plt.title('Seaborn Bar Plot')
plt.show()

# 4. Box Plot using Seaborn
# Create random data for box plot
data = pd.DataFrame({'Category': np.random.choice(['A', 'B', 'C'], 100),
                     'Value': np.random.randn(100)})

# Plot the data
plt.figure(figsize=(8, 6))
sns.boxplot(x='Category', y='Value', data=data, palette='coolwarm')
plt.title('Seaborn Box Plot')
plt.show()

# 5. Heatmap using Seaborn
# Create correlation matrix
corr_matrix = np.random.rand(10, 10)
corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make it symmetric

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Seaborn Heatmap')
plt.show()

# 6. Pair Plot using Seaborn
# Load the Iris dataset
iris = sns.load_dataset('iris')

# Plot the pairplot
sns.pairplot(iris, hue='species', palette='Set1')
plt.title('Seaborn Pair Plot')
plt.show()

# 7. Violin Plot using Seaborn
# Data for violin plot
tips = sns.load_dataset('tips')

# Plot the data
plt.figure(figsize=(8, 6))
sns.violinplot(x='day', y='total_bill', data=tips, inner='quart')
plt.title('Seaborn Violin Plot')
plt.show()

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

#Plotly

# Import necessary libraries
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

# 1. Line Plot using Plotly
# Generate data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a line plot
fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines', name='Sine Wave', line=dict(color='blue')))

# Add titles and labels
fig.update_layout(title='Plotly Line Plot', xaxis_title='X', yaxis_title='Y')

# Show the plot
fig.show()

# 2. Scatter Plot using Plotly
# Generate random data
x = np.random.rand(50)
y = np.random.rand(50)

# Create a scatter plot
fig = go.Figure(data=go.Scatter(x=x, y=y, mode='markers', marker=dict(color='red', size=10)))

# Add titles and labels
fig.update_layout(title='Plotly Scatter Plot', xaxis_title='X', yaxis_title='Y')

# Show the plot
fig.show()

# 3. Bar Plot using Plotly
# Data for bar plot
categories = ['A', 'B', 'C', 'D', 'E']
values = [10, 24, 18, 6, 12]

# Create a bar plot
fig = go.Figure(data=go.Bar(x=categories, y=values, marker_color='green'))

# Add titles and labels
fig.update_layout(title='Plotly Bar Plot', xaxis_title='Category', yaxis_title='Value')

# Show the plot
fig.show()

# 4. Histogram using Plotly
# Generate random data
data = np.random.randn(1000)

# Create a histogram
fig = go.Figure(data=go.Histogram(x=data, nbinsx=30, marker_color='purple'))

# Add titles and labels
fig.update_layout(title='Plotly Histogram', xaxis_title='Value', yaxis_title='Frequency')

# Show the plot
fig.show()

# 5. Pie Chart using Plotly
# Data for pie chart
labels = ['A', 'B', 'C', 'D']
sizes = [20, 30, 40, 10]

# Create a pie chart
fig = go.Figure(data=go.Pie(labels=labels, values=sizes, hole=0.3))

# Add title
fig.update_layout(title='Plotly Pie Chart')

# Show the plot
fig.show()

# 6. Heatmap using Plotly
# Create a random correlation matrix for heatmap
data = np.random.rand(10, 10)

# Create a heatmap
fig = go.Figure(data=go.Heatmap(z=data, colorscale='Viridis'))

# Add title and axis labels
fig.update_layout(title='Plotly Heatmap', xaxis_title='X', yaxis_title='Y')

# Show the plot
fig.show()

# 7. 3D Scatter Plot using Plotly
# Generate random data for 3D plot
x = np.random.rand(100)
y = np.random.rand(100)
z = np.random.rand(100)

# Create a 3D scatter plot
fig = go.Figure(data=go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=5, color='orange')))

# Add titles
fig.update_layout(title='Plotly 3D Scatter Plot', scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))

# Show the plot
fig.show()

# 8. Box Plot using Plotly Express
# Load a sample dataset
df = px.data.tips()

# Create a box plot
fig = px.box(df, x='day', y='total_bill', points='all')

# Add title
fig.update_layout(title='Plotly Box Plot')

# Show the plot
fig.show()

# 9. Area Plot using Plotly Express
# Generate data for area plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create an area plot
fig = px.area(x=x, y=y, title='Plotly Area Plot')

# Show the plot
fig.show()

# 10. Violin Plot using Plotly Express
# Create a violin plot
fig = px.violin(df, y='total_bill', box=True, points='all')

# Add title
fig.update_layout(title='Plotly Violin Plot')

# Show the plot
fig.show()
