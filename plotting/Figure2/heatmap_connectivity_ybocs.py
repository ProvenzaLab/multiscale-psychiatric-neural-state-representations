from scipy import stats
from sklearn import cross_decomposition
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
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

#df_features_ = pd.read_csv("plotting/Figure1/source_data/neural_features_suds.csv")
df_features = pd.read_csv("plotting/Figure1/source_data/neural_features_ybocs_with_coh_1.csv")

# replace in column subject 'aDBS004' with 4, 'aDBS005' with 5, 'aDBS007' with 7, 'aDBS009' with 9, 'aDBS010' with 10, 'aDBS011' with 11, 'aDBS012' with 12 and 'aDBS008' with 08

df_features["subject"] = df_features["subject"].replace({"aDBS004": 4, "aDBS005": 5, "aDBS007": 7, "aDBS008": 8, "aDBS009": 9, "aDBS010": 10, "aDBS011": 11, "aDBS012": 12})

# replace column "date" with "time"
df_features = df_features.rename(columns={"date": "time"})
df_features = df_features.rename(columns={"YBOCS II Total Score": "score"})

df_features["time"] = pd.to_datetime(df_features["time"])

num_subject = df_features["subject"].nunique()
# remove cols that have only NaN
df_features = df_features.dropna(axis=1, how='all')
feature_cols_candidate = [c for c in df_features.columns if "psd" not in c and c.startswith("coh")or c == "YBOCS II Total Score"]
# cut everything off after alpha, theta, delta, beta, gamma
coh_channels = ["coh_SC_left_right", 
                "coh_SC_left_C_left",
                "coh_SC_right_C_right",
                "coh_SC_left_C_right",
                "coh_SC_right_C_left",
                "coh_C_left_right", ]

features_names = ["coherence_delta",
                  "coherence_theta",
                  "coherence_alpha",
                  "coherence_beta",
                  "coherence_gamma",
                  "fooof_a_exp", # _corr
                  "fooof_a_offset"] # _corr

#df_features.query("subject == 10")[["coh_SC_right_C_left_coherence_delta", "coh_SC_right_C_left_coherence_beta", "coh_SC_right_C_left_coherence_gamma"]]
df_features.query("subject == 12")[["coh_SC_right_C_left_coherence_delta", "coh_SC_left_C_left_coherence_delta", "coh_SC_right_C_right_coherence_delta"]]

num_features = len(features_names)
num_regions = len(coh_channels)
regions = coh_channels


arr_coef = np.full((num_regions, num_features, num_subject), np.nan)
arr_pval = np.full((num_regions, num_features, num_subject), np.nan)

for region_idx, region in enumerate(coh_channels):

    for sub_idx, sub in enumerate(df_features["subject"].unique()):
        df_s = df_features[df_features["subject"] == sub]
        if (sub == 4 or sub == 5 or sub == 7) and region.startswith("C_"):
           continue
        if sub == 10:
            print("here")

        df_r = df_s[[c for c in df_s.columns if c.startswith(f"{region}_") or c == "subject" or c == "score"]]  # or c == "time"

        if df_r.empty:
            continue

        df_r = df_r.drop(columns=["subject"], errors='ignore')

        df_r = df_r.dropna(axis=1, how='all')

        if df_r.empty:
            continue
        mask = df_r.notna().all(axis=1) & df_features["score"].notna()

        X = df_r.loc[mask]
        #feature_names_sel = [f"{region}_{c}_{region[4:]}" for c in features_names]
        feature_names_sel = [f"{region}_{c}" for c in features_names]
        if not all(f in X.columns for f in feature_names_sel):
            continue
        X = X[feature_names_sel].values

        Y = df_r["score"].loc[mask].values

        for feature_idx, feature in enumerate(feature_names_sel):
            arr_coef[region_idx, feature_idx, sub_idx] = stats.pearsonr(X[:, feature_idx], Y)[0]
            arr_pval[region_idx, feature_idx, sub_idx] = stats.pearsonr(X[:, feature_idx], Y)[1]

# iterate through subjects and show the SC_L and SC_R correlations for all features, it should be an image
subjects = df_features["subject"].unique()

rows = []

for sub_idx, subject in enumerate(subjects):
    for region_idx, region in enumerate(regions):
        for feature_idx, feature in enumerate(features_names):
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
#df_coefs.to_csv("plotting/Figure2/source_data/feature_region_subject_correlations_ybocs_coherence.csv", index=False)

coh_delta = df_coefs.query("feature == 'coherence_delta'")["correlation"]
coh_delta_not_na = coh_delta[coh_delta.notna()]
p_delta_all = permutationTest(
    coh_delta_not_na,
    np.zeros(coh_delta_not_na.shape[0]),
    False, None, 5000
)[1]
# p=0.0036, 0.15 pm 30

regions_not_use = ["coh_SC_left_right", "coh_C_left_right"]
coh_delta_SC_C = df_coefs.query("feature == 'coherence_delta' and region not in @regions_not_use")["correlation"]
coh_delta_SC_C_not_na = coh_delta_SC_C[coh_delta_SC_C.notna()]
p_delta_C_SC = permutationTest(
    coh_delta_SC_C_not_na,
    np.zeros(coh_delta_SC_C_not_na.shape[0]),
    False, None, 5000
)[1]
# p=0.0392, 0.14 pm 0.29


highest_abs_corrs = np.nan_to_num(np.abs(arr_coef)).max(axis=(0,1))
# get the region and feature for each highest abs corr
best_corrs_coh = []
for sub in range(arr_coef.shape[2]):
    max_idx = np.unravel_index(np.nanargmax(np.abs(arr_coef[:, :, sub])), (arr_coef.shape[0], arr_coef.shape[1]))
    region = regions[max_idx[0]]
    feature = features_names[max_idx[1]]
    corr_value = arr_coef[max_idx[0], max_idx[1], sub]
    print(f"Subject {subjects[sub]}: Highest abs corr {corr_value:.3f} for region {region} and feature {feature}")
    best_corrs_coh.append({
        "subject": subjects[sub],
        "region": region,
        "feature": feature,
        "correlation": corr_value
    })
df_best_corrs_coh = pd.DataFrame(best_corrs_coh)
#df_best_corrs_coh.to_csv("best_feature_region_subject_correlations_ybocs_coherence.csv", index=False)
#df_coefs.to_csv("plotting/Figure2/source_data/feature_region_subject_correlations_ybocs.csv", index=False)

fig, axes = plt.subplots(1, 9, figsize=(14.3, 3), sharey=True)

for sub_idx, sub in enumerate(subjects):
    arr_plot = arr_coef[:, :, sub_idx]              # shape: (2, features)
    arr_p    = arr_pval[:, :, sub_idx]              # same shape

    # Build annot matrix (features x 2) with "\n*" if p < 0.05
    vals   = arr_plot.T                                   # (features, 2)
    pvals  = arr_p.T                                      # (features, 2)
    annot  = np.empty_like(vals, dtype=object)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            #p = pvals[i, j]
            reject, pvals_corrected, _, _ = multipletests(pvals[:, j].flatten(), alpha=0.05, method='fdr_bh')
            p = pvals_corrected[i]
            #p = pvals[i, j]  # use uncorrected p-value for individual subject plots
            star = "*" if (isinstance(p, (float, np.floating)) and np.isfinite(p) and p < 0.05) else ""
            annot[i, j] = f"{star}"  # {v:.2f}

    ax = axes[sub_idx]
    sns.heatmap(vals, annot=annot, cmap="coolwarm",
                vmin=-0.5, vmax=0.5,
                yticklabels=features_names,
                #xticklabels=["SC_L", "SC_R"],
                xticklabels = regions,
                fmt="", cbar=False,
                annot_kws={"ha": "center", "va": "center", "size": 15},
                ax=ax)

    ax.set_title(f"Subject {sub}", fontsize=8)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)

# plot the subject average in the last subplot
arr_plot = np.nanmean(arr_coef[:, :, :], axis=2)   # shape (2, features)
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
        feature_ = features_names[i]
        region_ = regions[j]
        corrs_ = df_coefs.query("region == @region_ and feature == @feature_")["correlation"].values
        test_vals = corrs_[~np.isnan(corrs_)]
        test_zeros = np.zeros(test_vals.shape)
        p_val = permutationTest(test_vals, test_zeros, plot_distr=False, x_unit="correlation", p=5000)[1]
        p_vals_avg[i, j] = p_val
        #star = "*" if (isinstance(p_val, (float, np.floating)) and np.isfinite(p_val) and p_val < 0.05) else ""
        #annot[i, j] = f"{star}"  # {v:.2f}

for i in range(vals.shape[0]):
    for j in range(vals.shape[1]):
        reject, pvals_corrected, _, _ = multipletests(pvals[:, j].flatten(), alpha=0.05, method='fdr_bh')
        p = pvals_corrected[i]
        star = "*" if (isinstance(p, (float, np.floating)) and np.isfinite(p) and p < 0.05) else ""
        annot[i, j] = f"{star}"  # {v:.2f}
ax = axes[8]
sns.heatmap(vals, annot=annot, cmap="coolwarm",
            vmin=-0.5, vmax=0.5,   # vmin=-0.3, vmax=0.3,
            yticklabels=features_names,
            xticklabels=regions,
            cbar=False,#(sub_idx==0),
            fmt="",
            annot_kws={"ha": "center", "va": "center", "size": 15},
            ax=ax)
ax.set_title(f"Average", fontsize=8)
ax.invert_yaxis()
ax.tick_params(axis='x', labelsize=8)
# single shared colorbar to the right
#cbar = axes[0].collections[0].colorbar
#cbar.ax.set_position([0.92, 0.15, 0.02, 0.7])
#plt.tight_layout()
plt.savefig("figures/Figure2/heatmap_ybocs_coherence_1.pdf")
plt.show()