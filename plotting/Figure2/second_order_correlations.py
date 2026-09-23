import pandas as pd
import numpy as np
from scipy import stats

import matplotlib as mpl
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 10,
    'axes.unicode_minus': False
})

from matplotlib import pyplot as plt
import seaborn as sns




feature_names = [
                 "delta",
                 "theta",
                 "alpha",
                 "beta",
                 "gamma",
                 "offset",
                 "exponent",
                 "raw",
                 "RawHjorth_Activity",
                 "RawHjorth_Complexity",
                 "RawHjorth_Mobility",
                 "Sharpwave_Max_prominence_range_1_12",
                 "Sharpwave_Max_prominence_range_1_5",
                 "Sharpwave_Max_sharpness_range_1_12",
                 "Sharpwave_Max_sharpness_range_1_5",
                 "Sharpwave_Mean_interval_range_1_12",
                 "Sharpwave_Mean_interval_range_1_5",
                 "burst_amplitude_delta",
                 "burst_amplitude_theta",
                 "burst_amplitude_alpha",
                 "burst_amplitude_beta",
                 "burst_amplitude_gamma",
"burst_duration_delta_ms",
"burst_duration_theta_ms",
                 "burst_duration_alpha_ms",
                 "burst_duration_beta_ms",
                 
                 
                 "burst_duration_gamma_ms",
                 ]

regions = ["SC_L", "SC_R", "C_L_1", "C_L_2", "C_R_1", "C_R_2"] 

subs_ = ["004", "005", "007", "008", "009", "010", "011", "012"]
subs_suds = ["004", "005", "007", "009", "010", "011", "012"] # no SUDS for 012

#PATH_ = "/Users/Timon/Documents/Houston/OCD_RCS/OCD_RCS/correlation_SUDS_coefficients_patient_ind.npy"
arr_suds_orig = np.load("plotting/Figure2/source_data/suds_correlation_scores_npy_1.npy")
# dimensions: regions x features x subjects

#arr_ybocs = np.load("corr_score_region_feature_sub.npy")
arr_ybocs_orig = np.load("plotting/Figure2/source_data/ybocs_correlation_scores_npy_1.npy")
# dimensions: scores x regions x features x subjects (score (0): YBOCS, )
# delete the 008 subject from arr_ybocs to match arr_suds
arr_ybocs_orig = np.delete(arr_ybocs_orig, 3, axis=2)



df_res_region_subs = []

#plt.figure(figsize=(22, 12))
# rows regions, cols subjects, skip C_ regions if only SC
for idx_region, region in enumerate(regions):
    for idx_sub, sub in enumerate(subs_suds):
        if region in ["C_L_1", "C_L_2", "C_R_1", "C_R_2"] and sub in ["004", "005", "007"]:
            continue  # these subjects don't have cortical electrodes
        #plt.subplot(len(regions), len(subs_suds) + 1, idx_region * (len(subs_suds) + 1) + idx_sub + 1)
        arr_ybocs_sub = arr_ybocs_orig[idx_region, :, idx_sub].flatten()
        arr_suds_sub =  arr_suds_orig[idx_region, :, idx_sub].flatten()
        mask = ~np.isnan(arr_ybocs_sub) & ~np.isnan(arr_suds_sub)
        corr, p_value = stats.pearsonr(arr_ybocs_sub[mask], arr_suds_sub[mask])
        # use sns regplot
        #sns.regplot(x=arr_ybocs_sub[mask], y=arr_suds_sub[mask], label=f"r={corr:.2f}\np={p_value:.3f}")
        #plt.legend()
        df_res_region_subs.append({
            "subject" : sub,
            "region" : region,
            "p" : p_value,
            "corr" : corr,
        })

df_res_region_subs = pd.DataFrame(df_res_region_subs)

df_comb = df_res_region_subs.pivot(index="subject", columns="region", values="corr").reset_index()
df_comb = df_comb.astype("float")

# VCVS: df_comb.iloc[:, -2:].to_numpy().mean(): -0.02 pm 0.29
# OFC: df_comb.iloc[3:, 1:5].to_numpy().mean(): -0.28 pm 0.37

#permutation test if vcvs is signifcantly different from 0
from scipy.stats import permutation_test
def statistic(x, y):
    return np.mean(x) - np.mean(y)
vcvs_corrs = df_comb.iloc[:, -2:].to_numpy().flatten()
ofc_corrs = df_comb.iloc[3:, 1:5].to_numpy().flatten()
res_vcvs = permutation_test((vcvs_corrs, np.zeros_like(vcvs_corrs)), statistic, vectorized=False, n_resamples=10000)
res_ofc = permutation_test((ofc_corrs, np.zeros_like(ofc_corrs)), statistic, vectorized=False, n_resamples=10000)
print(f"VCVS: mean={vcvs_corrs.mean():.3f}, p={res_vcvs.pvalue:.4f}") # 0.913
print(f"OFC: mean={ofc_corrs.mean():.3f}, p={res_ofc.pvalue:.4f}") # 0.0020

plt.figure()
sns.heatmap(data=df_comb.iloc[:, 1:][regions], cmap="coolwarm",  vmin=-0.8, vmax=0.8, annot=False, cbar_kws={"label": "Pearson r"})
plt.yticks(np.arange(len(df_comb["subject"].values)), df_comb["subject"].values, )
# add a star to significant correlations (p < 0.05)
for i, sub in enumerate(df_res_region_subs["subject"].values):
    for j, region in enumerate(regions):
        p = df_res_region_subs[(df_res_region_subs["subject"] == sub) & (df_res_region_subs["region"] == region)]["p"]
        if p.empty:
            continue
        p_val = p.values[0]
        r = df_res_region_subs[(df_res_region_subs["subject"] == sub) & (df_res_region_subs["region"] == region)]["corr"].values[0]
        if p_val < 0.05:
            plt.text(j + 0.5, i + 0.5, f"*", ha="center", va="center", fontsize=12)
        #else:
            #plt.text(j + 0.5, i + 0.5, f"{np.round(r, 2)}", ha="center", va="center", fontsize=12)
plt.savefig("figures/Figure2/second_order_corr_heatmap_1.pdf")

