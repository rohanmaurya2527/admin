#Sign-Rank Test
import numpy as np
from scipy.stats import binom
data=[(4,3),(5,6),(7,6),(4,5),
     (3,2),(6,6),(8,7),(5,4),(6,6),(4,4)]
diff=[x-y for x,y in data]
pos_diff=sum(1 for differ in diff if differ>0)
neg_diff=sum(1 for differ in diff if differ<0)
n=len(data)
k=pos_diff
p_val=binom.cdf(k,n,0.5)+binom.sf(k-1,n,0.5)
print(n)
print(k)
print(f"P-value:{p_val}")

#Wilcoxon Signed Rank Test
import numpy as np
from scipy.stats import wilcoxon
before=[35,42,38,46,33,29,42,37,40,32]
after=[30,40,36,44,28,24,41,35,38,30]
diff=[after[i]-before[i] for i in range(len(before))]
stat,p_val=wilcoxon(diff)
print(f"Test Statistic:{stat}")
print(f"P-Value:{p_val}")

#Kruskal Wali’s Test
import scipy.stats as stats
grp_A=[23,27,21,19,25]
grp_B=[18,16,20,24,22]
grp_C=[10,13,15,11,12]
H,p_val=stats.kruskal(grp_A,grp_B,grp_C)
print(f"Kruskal-Wallis H:{H}")
print(f"P_value:{p_val}")

#Mann Whitney U Test
from scipy import stats
sam1=[23,45,67,89,12,34,56]
sam2=[13,25,37,49,61,73,85]
U_stat,p_val=stats.mannwhitneyu(sam1,sam2,alternative='two-sided')
print(f"U-Statistic:{U_stat}")
print(f"P-Value:{p_val}")
alpha=0.05
if p_val<alpha:
    print("Difference bet 2 samples is statistically significant")
else:
    print("Difference bet 2 samples is not statistically significant")



