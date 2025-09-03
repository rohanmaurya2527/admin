#Normal Distribution
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats
mean=78
std_dev=25
tot_stud=100
score=60
z_score=(score-mean)/std_dev
prob=norm.cdf(z_score)
percent=prob*100
print("Percentage of students who got less than 60 marks:",round(percent,2),"%")

mean=78
std_dev=25
tot_stud=100
score=70
z_score=(score-mean)/std_dev
prob=norm.cdf(z_score)
percent=(1-prob)*100
print("Percentage of students who scored more than 70 marks:",round(percent,2),"%")

#Binomial Distribution
from scipy.stats import binom
n=6
p=0.6
r_val=list(range(n+1))
dist=[binom.pmf(r,n,p) for r in r_val]
print(dist)

plt.bar(r_val,dist)
plt.show()

#Q. Fashion Bazaar 
#Expected number of successful trials=5
#total number of trails=30
#Probability of success=0.15
stats.binom.pmf(5,30,0.15)

#Poisson Distribution
#Q. The number of calls arriving at a call center follows a Poisson distribution at 20 calls per hour. Calculate the probability that the number of calls will be maximum 10.

stats.poisson.cdf(10,20)

#Chi-Square Distribution
from scipy.stats import chi2
# Parameters for the Chi-Square distribution
degrees_of_freedom = 5  # Degrees of freedom
# Generate a range of x values
x = np.linspace(0, 20, 1000)
# Calculate the probability density function (PDF) for the Chi-Square distribution
pdf = chi2.pdf(x, degrees_of_freedom)
# Plot the PDF
plt.plot(x, pdf, label=f'Chi-Square (df={degrees_of_freedom})')
plt.xlabel('x')
plt.ylabel('PDF')
plt.legend()
plt.title('Chi-Square Distribution')
plt.grid()
plt.show()
