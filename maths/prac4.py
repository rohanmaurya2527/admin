#t-test
import numpy as np
from scipy import stats
grp1=np.array([85,90,88,92,95])
grp2=np.array([78,80,84,83,82])
t_stat,p_val=stats.ttest_ind(grp1,grp2)
print(f"t-Statistic:{t_stat}")
print(f"P-Value:{p_val}")
alpha=0.05
if p_val<alpha:
    print("Reject the null hypothesis: There is a significant difference between the two groups.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference between the two groups.")

#z-test
import numpy as np
from scipy import stats
sam_height=np.array([66,68,70,65,69])
pop_mean=67
pop_std=2
sam_mean=np.mean(sam_height)
z_stat=(sam_mean-pop_mean)/(pop_std/np.sqrt(len(sam_height)))
p_val=2*(1-stats.norm.cdf(np.abs(z_stat))) #2-tailed test
print(f"z-Statistic:{z_stat}")
print(f"P-Value:{p_val}")
alpha=0.05
if p_val < alpha:
    print("Reject the null hypothesis: The sample mean is significantly different from the population mean.")
else:
    print("Fail to reject the null hypothesis: The sample mean is not significantly different from the population mean.")


