#One-Way Anova
import numpy as np
mth1=np.array([85,88,92,94,90])
mth2=np.array([78,80,83,77,79])
mth3=np.array([91,89,94,96,92])
from scipy import stats
f_stat,p_val=stats.f_oneway(mth1,mth2,mth3)
print(f"F-Statistic:{f_stat}")
print(f"P-Value:{p_val}")

#Two-Way Anova
import pandas as pd
data = pd.DataFrame({
    'Score': [85, 78, 91, 88, 80, 89, 92, 83, 94, 94, 77, 96, 90, 79, 92],
    'Method': ['Method1', 'Method1', 'Method1', 'Method2', 'Method2', 'Method2', 'Method3', 'Method3', 'Method3',
               'Method1', 'Method2', 'Method3', 'Method1', 'Method2', 'Method3'],
    'Gender': ['Female', 'Female', 'Female', 'Male', 'Male', 'Male', 'Female', 'Female', 'Female',
               'Male', 'Male', 'Male', 'Female', 'Female', 'Female']
})
import statsmodels.api as sm
from statsmodels.formula.api import ols
model=ols('Score ~ C(Method) * C(Gender)',data=data).fit()
#C(Method): Test effect of teaching method
#C(Gender): Test effect of gender
#C(Method) * C(Gender): Test interaction effect bet teaching method and gender
anova_table=sm.stats.anova_lm(model,typ=2)
print(anova_table)

