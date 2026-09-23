
# check figure Figure2/TEED_feature_changes_1.pdf
# number of significant changes across vs TEED0 = 15 / 27
# number of significant changes across vs TEEDne0 = 3 / 27

from scipy.stats import binomtest

# Condition 1
res1 = binomtest(23, 27, p=0.05, alternative='greater')
print(res1.pvalue)
# 1.7190898582339303e-26

# Condition 2
res2 = binomtest(18, 27, p=0.05, alternative='greater')
print(res2.pvalue)
# 1.1555056765941719e-17