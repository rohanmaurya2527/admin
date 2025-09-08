import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
x=np.array([1,2,3,4,5])
y=np.array([7,14,15,18,19])
n=np.size(x)
x_mean=np.mean(x)
y_mean=np.mean(y)
x_mean,y_mean

Sxy=np.sum(x*y)-n*x_mean*y_mean
Sxx=np.sum(x*x)-n*x_mean*x_mean
b1=Sxy/Sxx
b0=y_mean-b1*x_mean
print('Slope b1 is',b1)
print('Intercept b0 is',b0)

plt.scatter(x,y)
plt.xlabel('Independent Variable X')
plt.ylabel('Dependent Variable y')
plt.show()

y_pred=b1*x+b0

plt.scatter(x,y,color='red')
plt.plot(x,y_pred,color='green')
plt.xlabel('X')
plt.ylabel('y')
