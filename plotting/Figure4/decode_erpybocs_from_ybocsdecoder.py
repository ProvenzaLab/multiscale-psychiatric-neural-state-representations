import pandas as pd
import numpy as np
#import cebra

import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from scipy.stats import permutation_test
import seaborn as sns
from sklearn import linear_model, model_selection
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
from sklearn import linear_model
from sklearn.linear_model import MultiTaskElasticNet, MultiTaskElasticNetCV
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
from joblib import Parallel, delayed
from sklearn.neighbors import KNeighborsRegressor
import pickle
#import utils
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import sys
import os
import argparse
# set arial with fontsize 5
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 5

df_res = pd.read_csv("plotting/Figure4/source_data/decoding_results_sudsybocs_ybocssuds_decoding_1.csv")
print(df_res.groupby("YBOCS_TO_SUDS")["pearson_corr"].mean())
print(df_res.groupby("YBOCS_TO_SUDS")["pearson_corr"].std())
# run a permutation test against zero for each YBOCS_TO_SUDS
def mean_stat(x):
    return np.mean(x)

for ybocs_to_suds in [True, False]:
    df_res_sub = df_res.query("YBOCS_TO_SUDS == @ybocs_to_suds")
    corr_vals = df_res_sub["pearson_corr"].dropna().to_numpy()
    res = permutation_test(
        (corr_vals,),
        statistic=mean_stat,
        permutation_type="samples",
        alternative="two-sided",
        n_resamples=10000,
        random_state=0,
    )
    print(f"YBOCS_TO_SUDS {ybocs_to_suds} p-value: {res.pvalue:.4f}")
# YBOCS_TO_SUDS -0.09 pm 0.32, p=0.73
# SUDS_TO_YBOCS -0.20 pm 0.4, p=0.25

plt.figure(figsize=(2.5, 1.5))

ax = sns.barplot(
    data=df_res,
    x="subject",
    y="pearson_corr",
    hue="YBOCS_TO_SUDS",
    errorbar=None,
    linewidth=0.6,
    edgecolor=None,
)

# Add significance stars
for patch, (_, row) in zip(ax.patches, df_res.iterrows()):
    if row["pearson_pval"] < 0.05:
        x = patch.get_x() + patch.get_width() / 2
        y = patch.get_height()

        ax.text(
            x,
            y + 0.02,
            "*",
            ha="center",
            va="bottom",
            color="red",
            fontsize=5,
        )

plt.title("Pearson correlation between predicted and true values for each subject", fontsize=5)
plt.ylabel("Pearson correlation", fontsize=5)
plt.xlabel("Subject", fontsize=5)
plt.legend(title="YBOCS_TO_SUDS", fontsize=5, title_fontsize=5)
ax.tick_params(axis="x", which="major", length=1.0, width=0.6, labelsize=5)
ax.tick_params(axis="y", which="major", length=1.0, width=0.6, labelsize=5)
for spine in ax.spines.values():
    spine.set_linewidth(0.6)

sns.despine()
plt.tight_layout()
plt.savefig("figures/Figure4/results_sudsybocs_ybocs_suds_decoding_1.pdf")


