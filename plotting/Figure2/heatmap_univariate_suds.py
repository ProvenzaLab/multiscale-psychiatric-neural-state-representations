from scipy import stats
from sklearn import cross_decomposition
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
import re
import seaborn as sns

import random
import copy


def permutationTest(x, y, plot_distr=True, x_unit=None, p=5000):
    """
    Calculate permutation test
    https://towardsdatascience.com/how-to-assess-statistical-significance-in-your-data-with-permutation-tests-8bb925b2113d

    x (np array) : first distr.
    y (np array) : first distr.
    plot_distr (boolean) : if True: plot permutation histplot and ground truth
    x_unit (str) : histplot xlabel
    p (int): number of permutations

    returns:
    gT (float) : estimated ground truth, here absolute difference of
    distribution means
    p (float) : p value of permutation test

    """
    # Compute ground truth difference
    gT = np.abs(np.average(x) - np.average(y))

    pV = np.concatenate((x, y), axis=0)
    pS = copy.copy(pV)
    # Initialize permutation:
    pD = []
    # Permutation loop:
    for i in range(0, p):
        # Shuffle the data:
        random.shuffle(pS)
        # Compute permuted absolute difference of your two sampled
        # distributions and store it in pD:
        pD.append(
            np.abs(
                np.average(pS[0 : int(len(pS) / 2)])
                - np.average(pS[int(len(pS) / 2) :])
            )
        )

    # Calculate p-value
    if gT < 0:
        p_val = len(np.where(pD <= gT)[0]) / p
    else:
        p_val = len(np.where(pD >= gT)[0]) / p

    if plot_distr:
        plt.hist(pD, bins=30, label="permutation results")
        plt.axvline(gT, color="orange", label="ground truth")
        plt.title("ground truth " + x_unit + "=" + str(gT) + " p=" + str(p_val))
        plt.xlabel(x_unit)
        plt.legend()
        plt.show()
    return gT, p_val


def natural_keys(text):
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

df_features = pd.read_csv("plotting/Figure1/source_data/neural_features_suds.csv")
df_features["time"] = pd.to_datetime(df_features["time"])

regions = ["SC_L", "SC_R", "C_L_1", "C_L_2", "C_R_1", "C_R_2"] 
num_subject = df_features["subject"].nunique()
num_regions = len(regions)
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
                 "burst_duration_alpha_ms",
                 "burst_duration_beta_ms",
                 "burst_duration_delta_ms",
                 "burst_duration_theta_ms",
                 "burst_duration_gamma_ms",
                 ]
num_features = len(feature_names)


arr_coef = np.full((num_regions, num_features, num_subject), np.nan)
arr_pval = np.full((num_regions, num_features, num_subject), np.nan)
#for sub_idx, sub in enumerate(df_features["subject"].unique()):
#    df_s = df_features[df_features["subject"] == sub]
for region_idx, region in enumerate(regions):
    for sub_idx, sub in enumerate(df_features["subject"].unique()):
        df_s = df_features[df_features["subject"] == sub]
        if (sub == 4 or sub == 5 or sub == 7) and region.startswith("C_"):
           continue

        df_r = df_s[[c for c in df_s.columns if c.startswith(f"{region}_") or c == "subject" or c == "score"]]  # or c == "time"

        if df_r.empty:
            continue

        df_r = df_r.drop(columns=["subject"], errors='ignore')

        df_r = df_r.dropna(axis=1, how='all')

        if df_r.empty:
            continue
        mask = df_r.notna().all(axis=1) & df_features["score"].notna()

        X = df_r.loc[mask]
        feature_names_sel = [f"{region}_{c}" for c in feature_names]

        X = X[feature_names_sel].values

        Y = df_r["score"].loc[mask].values

        for feature_idx, feature in enumerate(feature_names_sel):
            arr_coef[region_idx, feature_idx, sub_idx] = stats.pearsonr(X[:, feature_idx], Y)[0]
            arr_pval[region_idx, feature_idx, sub_idx] = stats.pearsonr(X[:, feature_idx], Y)[1]

np.save("plotting/Figure2/source_data/suds_correlation_scores_npy_1.npy", arr_coef)
#np.load("plotting/Figure2/source_data/suds_correlation_scores_npy.npy")
# iterate through subjects and show the SC_L and SC_R correlations for all features, it should be an image
subjects = df_features["subject"].unique()

rows = []

for sub_idx, subject in enumerate(subjects):
    for region_idx, region in enumerate(regions):
        for feature_idx, feature in enumerate(feature_names):
            coef = arr_coef[region_idx, feature_idx, sub_idx]
            pval = arr_pval[region_idx, feature_idx, sub_idx]

            rows.append({
                "subject": subject,
                "region": region,
                "feature": feature,
                "correlation": coef,
                "p_value": pval
            })

df_coefs = pd.DataFrame(rows)
df_coefs["corr_abs"] = df_coefs["correlation"].abs()
df_coefs.groupby(["region", "feature"])["correlation"].mean().reset_index().sort_values("correlation", ascending=False)
# print std of the best mean correlation per region and feature
df_coefs.query("region == 'C_L_1' and feature == 'burst_amplitude_theta'")["corr_abs"].std()
#C_L_1_burst_amplitude_theta: 0.40 +m 0.29

df_coefs.query("region == 'C_L_1' and feature == 'RawHjorth_Activity'")["correlation"].std()
# C_L_1 RawHjorth_Activity: 0.36 pm 0.36

#df_coefs.to_csv("plotting/Figure2/source_data/feature_region_subject_correlations_suds.csv", index=False)

best_corrs_vcvs = []
highest_abs_corrs = np.nan_to_num(np.abs(arr_coef[[0, 1], :, :])).max(axis=(0,1))
for sub in range(arr_coef.shape[2]):
    max_idx = np.unravel_index(np.nanargmax(np.abs(arr_coef[[0, 1], :, sub])), (arr_coef[[0, 1], :, :].shape[0], arr_coef.shape[1]))
    region = regions[[0, 1][max_idx[0]]]
    feature = feature_names[max_idx[1]]
    corr_value = arr_coef[[0, 1][max_idx[0]], max_idx[1], sub]
    print(f"Subject {subjects[sub]}: Highest abs corr {corr_value:.3f} for region {region} and feature {feature}")
    best_corrs_vcvs.append({
        "subject": subjects[sub],
        "region": region,
        "feature": feature,
        "correlation": corr_value
    })

df_best_corrs = pd.DataFrame(best_corrs_vcvs)
#df_best_corrs.to_csv("best_feature_region_subject_correlations_vcvs_suds.csv", index=False)

best_corrs_c = []
highest_abs_corrs = np.nan_to_num(np.abs(arr_coef[[2, 3, 4, 5], :, :])).max(axis=(0,1))
for sub in range(arr_coef.shape[2]):
    sub_name = subjects[sub]
    if sub_name in [4, 5, 7]:
        continue
    max_idx = np.unravel_index(np.nanargmax(np.abs(arr_coef[[2, 3, 4, 5], :, sub])), (arr_coef[[2, 3, 4, 5], :, :].shape[0], arr_coef.shape[1]))
    region = regions[[2, 3, 4, 5][max_idx[0]]]
    feature = feature_names[max_idx[1]]
    corr_value = arr_coef[[2, 3, 4, 5][max_idx[0]], max_idx[1], sub]
    print(f"Subject {subjects[sub]}: Highest abs corr {corr_value:.3f} for region {region} and feature {feature}")
    best_corrs_c.append({
        "subject": subjects[sub],
        "region": region,
        "feature": feature,
        "correlation": corr_value
    })

df_best_corrs = pd.DataFrame(best_corrs_c)
#df_best_corrs.to_csv("best_feature_region_subject_correlations_c_suds.csv", index=False)

#df_coefs.to_csv("plotting/Figure2/source_data/feature_region_subject_correlations.csv", index=False)

# ECOG plot

# Subjects of interest
ecog_subjects = [9, 10, 11, 12]
subjects_all = df_features["subject"].unique()
subj_to_idx = {int(s): i for i, s in enumerate(subjects_all)}

# Region indices for cortical ECoG contacts
ecog_regions = [2, 3, 4, 5]
ecog_labels  = ["C_L_1", "C_L_2", "C_R_1", "C_R_2"]

n_cols = len(ecog_subjects) + 1
fig, axes = plt.subplots(1, n_cols, figsize=(14, 7), sharey=True)

if n_cols == 1:
    axes = [axes]

for col_idx, sub in enumerate(ecog_subjects):
    sidx = subj_to_idx[sub]
    arr_plot = arr_coef[ecog_regions, :, sidx]   # shape (4, features)
    arr_p    = arr_pval[ecog_regions, :, sidx]

    # Build annotation matrix with stars
    vals  = arr_plot.T    # (features, 4)
    pvals = arr_p.T
    annot = np.empty_like(vals, dtype=object)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            reject, pvals_corrected, _, _ = multipletests(pvals[:, j].flatten(), alpha=0.05, method='fdr_bh')
            p = pvals_corrected[i]
            #p = pvals_corrected[i * vals.shape[1] + j]
            #v = vals[i, j]
            #p = pvals[i, j]
            star = "*" if (np.isfinite(p) and p < 0.05) else ""
            annot[i, j] = f"{star}" # {v:.2f}

    ax = axes[col_idx]
    sns.heatmap(vals, annot=annot, cmap="coolwarm",
                vmin=-0.5, vmax=0.5,
                yticklabels=feature_names,
                xticklabels=ecog_labels,
                fmt="", cbar=False,
                annot_kws={"ha": "center", "va": "center", "size": 8},
                ax=ax)

    #ax.invert_yaxis()
    ax.set_title(f"Subject {sub}", fontsize=8)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)

    # # Only keep y labels on first subplot
    # if col_idx != 0:
    #     ax.set_ylabel("")
    #     ax.set_yticks([])
    #     ax.set_yticklabels([])

# plot the subject average in the last subplot
arr_plot = np.nanmean(arr_coef[ecog_regions, :, :][:, :, [subj_to_idx[s] for s in ecog_subjects]], axis=2)   # shape (4, features)
#arr_p    = np.nanmean(arr_pval[ecog_regions, :, [subj_to_idx[s] for s in ecog_subjects]], axis=2)   # same shape
# Build annot matrix (features x 2) with "\n*" if p < 0.05
vals   = arr_plot.T                                   # (features, 4)
pvals  = arr_p.T                                      # (features, 4)
annot  = np.empty_like(vals, dtype=object)


p_vals_avg = np.zeros(vals.shape)
corr_vals_avg = np.zeros(vals.shape)
for i in range(vals.shape[0]):
    for j in range(vals.shape[1]):
        
        v = vals[i, j]
        p = pvals[i, j]
        # compute a permutation test p-value for this feature across subjects
        feature_ = feature_names[i]
        region_ = ecog_labels[j]
        corrs_ = df_coefs.query("region == @region_ and feature == @feature_")["correlation"].values[3:]
        p_val = permutationTest(corrs_, np.zeros(corrs_.shape), plot_distr=False, x_unit="correlation", p=5000)[1]
        
        #reject, pvals_corrected, _, _ = multipletests(pvals[:, j], alpha=0.05, method='fdr_bh')
        #p_val = pvals_corrected[i]
        p_vals_avg[i, j] = p_val
        corr_vals_avg[i, j] = np.mean(corrs_)

p_val_corr_arr = np.zeros(vals.shape)
for i in range(vals.shape[0]):
    for j in range(vals.shape[1]):
        reject, pvals_corrected, _, _ = multipletests(p_vals_avg[:, j].flatten(), alpha=0.05, method='fdr_bh')
        p_val = pvals_corrected[i]
        #p_val = p_vals_avg[i, j]
        star = "*" if (isinstance(p_val, (float, np.floating)) and np.isfinite(p_val) and p_val < 0.05) else ""
        annot[i, j] = f"{star}" #{v:.2f}
        p_val_corr_arr[i, j] = p_val
ax = axes[-1]
sns.heatmap(vals, annot=annot, cmap="coolwarm",
            vmin=-0.5, vmax=0.5,
            yticklabels=feature_names,
            xticklabels=ecog_labels,
            #cbar=(col_idx==0),
            cbar=False,
            fmt="",
            annot_kws={"ha": "center", "va": "center", "size": 8},
            ax=ax)
ax.set_title(f"Average", fontsize=8)
#cbar = axes[-1].collections[0].colorbar
#cbar.ax.set_position([0.92, 0.15, 0.02, 0.7])

#plt.tight_layout(rect=[0, 0, 0.9, 1])
ax.invert_yaxis()
#plt.savefig("figures/Figure2/heatmap_suds_ofc.pdf")
#plt.show()



fig, axes = plt.subplots(1, 8, figsize=(14, 7), sharey=True)

for sub_idx, sub in enumerate(subjects[:7]):
    arr_plot = arr_coef[[0, 1], :, sub_idx]              # shape: (2, features)
    arr_p    = arr_pval[[0, 1], :, sub_idx]              # same shape

    # Build annot matrix (features x 2) with "\n*" if p < 0.05
    vals   = arr_plot.T                                   # (features, 2)
    pvals  = arr_p.T                                      # (features, 2)
    annot  = np.empty_like(vals, dtype=object)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            reject, pvals_corrected, _, _ = multipletests(pvals[:, j].flatten(), alpha=0.05, method='fdr_bh')
            p = pvals_corrected[i]            
            #p = pvals[i, j]
            star = "*" if (isinstance(p, (float, np.floating)) and np.isfinite(p) and p < 0.05) else ""
            annot[i, j] = f"{star}"  # {v:.2f}

    ax = axes[sub_idx]
    sns.heatmap(vals, annot=annot, cmap="coolwarm",
                vmin=-0.5, vmax=0.5,
                yticklabels=feature_names,
                xticklabels=["SC_L", "SC_R"],
                fmt="", cbar=False,
                annot_kws={"ha": "center", "va": "center", "size": 8},
                ax=ax)

    ax.set_title(f"Subject {sub}", fontsize=8)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)

# plot the subject average in the last subplot
arr_plot = np.nanmean(arr_coef[[0, 1], :, :7], axis=2)   # shape (2, features)
#arr_p    = np.nanmean(arr_pval[[0, 1], :, :7], axis=2)   # same shape
# Build annot matrix (features x 2) with "\n*" if p < 0.05
vals   = arr_plot.T                                   # (features, 2)
pvals  = arr_p.T                                      # (features, 2)
annot  = np.empty_like(vals, dtype=object)
p_vals_avg = np.zeros(vals.shape)
for i in range(vals.shape[0]):
    for j in range(vals.shape[1]):
        v = vals[i, j]
        p = pvals[i, j]
        # compute a permutation test p-value for this feature across subjects
        feature_ = feature_names[i]
        region_ = regions[j]
        corrs_ = df_coefs.query("region == @region_ and feature == @feature_")["correlation"].values
        p_val = permutationTest(corrs_, np.zeros(corrs_.shape), plot_distr=False, x_unit="correlation", p=5000)[1]
        p_vals_avg[i, j] = p_val
        #star = "*" if (isinstance(p_val, (float, np.floating)) and np.isfinite(p_val) and p_val < 0.05) else ""
        #annot[i, j] = f"{star}"  # {v:.2f}

for i in range(vals.shape[0]):
    for j in range(vals.shape[1]):
        reject, pvals_corrected, _, _ = multipletests(p_vals_avg[:, j].flatten(), alpha=0.05, method='fdr_bh')
        p_val = pvals_corrected[i]
        #p_val = p_vals_avg[i, j]
        star = "*" if (isinstance(p_val, (float, np.floating)) and np.isfinite(p_val) and p_val < 0.05) else ""
        annot[i, j] = f"{star}" #{v:.2f}
    
ax = axes[7]
sns.heatmap(vals, annot=annot, cmap="coolwarm",
            vmin=-0.5, vmax=0.5,
            yticklabels=feature_names,
            xticklabels=["SC_L", "SC_R"],
            cbar=False,#(sub_idx==0),
            fmt="",
            annot_kws={"ha": "center", "va": "center", "size": 8},
            ax=ax)
ax.set_title(f"Average", fontsize=8)
ax.invert_yaxis()
# single shared colorbar to the right
#cbar = axes[0].collections[0].colorbar
#cbar.ax.set_position([0.92, 0.15, 0.02, 0.7])
#plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.savefig("figures/Figure2/heatmap_suds_vcvs.pdf")
