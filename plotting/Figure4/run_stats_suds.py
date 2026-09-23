import pandas as pd
from scipy import stats
import numpy as np
import pickle
import matplotlib
import seaborn as sns
# set arial font and fontsize to be 10
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 5
import matplotlib.pyplot as plt

subjects = [4, 5, 7, 8, 9, 10, 11, 12]

df = pd.read_csv(f"plotting/Figure4/source_data/suds_decoding_results_neural_audio_fau_10.csv")


def set_condition(row):
    if row["INCLUDE_AU"] and row["INCLUDE_AUDIO"]:
        return "SUDS+AU+Audio"
    elif row["INCLUDE_AU"] and not row["INCLUDE_AUDIO"]:
        return "SUDS+AU"
    elif not row["INCLUDE_AU"] and row["INCLUDE_AUDIO"]:
        return "SUDS+Audio"
    else:
        return "SUDS only"
df["condition"] = df.apply(set_condition, axis=1)

with open(f"plotting/Figure4/source_data/suds_decoding_results_neural_audio_fau_10.pkl", "rb") as f:
    results = pickle.load(f)

def get_tr_pr(region="all", RS=False, subject=5, ZS=True):

    res_df = results[(region, not RS)]['df_res']
    res_df["condition"] = res_df.apply(set_condition, axis=1)
    res_df = res_df.query("SHUFFLE == False")
    ind_sub_suds_only = res_df[(res_df["condition"] == "SUDS only") & (res_df["subject"] == subject)].index[0]
    ind_sub_suds_au_audio = res_df[(res_df["condition"] == "SUDS+AU+Audio") & (res_df["subject"] == subject)].index[0]

    y_tr_suds_only = results[(region,  not RS)]["y_true"][ind_sub_suds_only][:, 0]
    y_tr_suds_au_audio = results[(region,  not RS)]["y_true"][ind_sub_suds_au_audio][:, 0]
    y_pr_suds_only = results[(region,  not RS)]["y_pred"][ind_sub_suds_only]
    y_pr_suds_au_audio = results[(region,  not RS)]["y_pred"][ind_sub_suds_au_audio]

    y_tr_suds_only_zs = (y_tr_suds_only - np.mean(y_tr_suds_only)) / np.std(y_tr_suds_only)
    y_tr_suds_au_audio_zs = (y_tr_suds_au_audio - np.mean(y_tr_suds_au_audio)) / np.std(y_tr_suds_au_audio)
    y_pr_suds_only_zs = (y_pr_suds_only - np.mean(y_pr_suds_only)) / np.std(y_pr_suds_only)
    y_pr_suds_au_audio_zs = (y_pr_suds_au_audio - np.mean(y_pr_suds_au_audio)) / np.std(y_pr_suds_au_audio)


    return y_tr_suds_only_zs, y_tr_suds_au_audio_zs, y_pr_suds_only_zs, y_pr_suds_au_audio_zs


plt.figure(figsize=(2, 1.25))
RS = False
y_tr_suds_only_zs, y_tr_suds_au_audio_zs, y_pr_suds_only_zs, y_pr_suds_au_audio_zs = \
            get_tr_pr(region="all", RS=RS, subject=12, ZS=True)

plt.subplot(1, 2, 1)
plt.title(f"Exemplary SUDS prediction subject 12", fontsize=5)
plt.plot(y_tr_suds_only_zs, label="True SUDS", color="black", linewidth=0.6, marker="o", markersize=1)
#plt.plot(y_pr_suds_only_zs, label="Predicted SUDS only", color="#3C6682", linewidth=0.6, marker="o", markersize=1)
plt.plot(y_pr_suds_au_audio_zs, label="Predicted SUDS+AU+Audio", color="#45A778", linewidth=0.6, marker="o", markersize=1)
plt.ylabel("SUDS z-score [a.u.]", fontsize=5)
plt.xlabel("Time [a.u.]", fontsize=5)
ax = plt.gca()
ax.tick_params(axis="x", which="major", length=1.0, width=0.6, labelsize=5)
ax.tick_params(axis="y", which="major", length=1.0, width=0.6, labelsize=5)
for spine in ax.spines.values():
    spine.set_linewidth(0.6)
sns.despine()

plt.subplot(1, 2, 2)
df_plot = pd.DataFrame({
    "True SUDS": y_tr_suds_only_zs,
    "Predicted SUDS only": y_pr_suds_only_zs,
    "Predicted SUDS+AU+Audio": y_pr_suds_au_audio_zs,
})
#sns.regplot(data=df_plot, x="True SUDS", y="Predicted SUDS only", label="SUDS only", color="#3C6682", scatter_kws={"s": 6}, line_kws={"linewidth": 0.6})
sns.regplot(data=df_plot, x="True SUDS", y="Predicted SUDS+AU+Audio", label="SUDS+AU+Audio", color="#45A778", scatter_kws={"s": 6}, line_kws={"linewidth": 0.6})
ax = plt.gca()
ax.tick_params(axis="x", which="major", length=1.0, width=0.6, labelsize=5)
ax.tick_params(axis="y", which="major", length=1.0, width=0.6, labelsize=5)
for spine in ax.spines.values():
    spine.set_linewidth(0.6)
sns.despine()
plt.savefig(f"figures/Figure4/exemplar_subject_12_suds_1.pdf")
# r = 0.80, p = 1.2e-5

# RS=False
#   Region: SC, Condition: SUDS only, Mean: 0.058, Std: 0.227
#   Region: SC, Condition: SUDS+AU+Audio, Mean: 0.120, Std: 0.268
#   Region: C, Condition: SUDS only, Mean: 0.042, Std: 0.200
#   Region: C, Condition: SUDS+AU+Audio, Mean: 0.195, Std: 0.342
#   Region: all, Condition: SUDS only, Mean: 0.056, Std: 0.245
#   Region: all, Condition: SUDS+AU+Audio, Mean: 0.189, Std: 0.283

# compare within statistics RS=False, SUDS only vs SUDS+AU+Audio
# permutation test non-paired
# I want to compare the regions against each other
for condition in ["SUDS only", "SUDS+AU+Audio"]:
    for region in ["SC", "C", "all"]:
        for region_2 in ["SC", "C", "all"]:
            if region >= region_2:
                continue
            subset_1 = df.query(f"RS == False and region == '{region}' and condition == '{condition}' and SHUFFLE == False")["r"]
            subset_2 = df.query(f"RS == False and region == '{region_2}' and condition == '{condition}' and SHUFFLE == False")["r"]
            res = stats.permutation_test(
                (subset_1, subset_2),
                statistic=lambda x, y: np.mean(x) - np.mean(y),
                permutation_type="independent",
                alternative="two-sided",
                n_resamples=10000,
                random_state=0
            )
            print(f"Permutation test p-value (SUDS only, RS=False, {condition}, {region} vs {region_2}): {res.pvalue}")

# Permutation test p-value (SUDS only, RS=False, SUDS only, SC vs all): 0.8041958041958042
# Permutation test p-value (SUDS only, RS=False, SUDS only, C vs SC): 0.40606060606060607
# Permutation test p-value (SUDS only, RS=False, SUDS only, C vs all): 0.4727272727272727
# Permutation test p-value (SUDS only, RS=False, SUDS+AU+Audio, SC vs all): 0.10081585081585082
# Permutation test p-value (SUDS only, RS=False, SUDS+AU+Audio, C vs SC): 0.1393939393939394
# Permutation test p-value (SUDS only, RS=False, SUDS+AU+Audio, C vs all): 0.9030303030303031
