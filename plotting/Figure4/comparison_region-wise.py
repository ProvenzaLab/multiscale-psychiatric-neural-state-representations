import pandas as pd
from scipy import stats
import numpy as np
import pickle
import matplotlib
import seaborn as sns
# set arial font and smaller figure text
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 5
import matplotlib.pyplot as plt

df_suds = pd.read_csv(f"plotting/Figure4/source_data/suds_decoding_results_neural_audio_fau_10.csv")
df_rs = pd.read_csv(f"plotting/Figure4/source_data/ybocs_decoding_results_neural_audio_fau_49.csv")

def set_condition(row):
    if row["INCLUDE_AU"] and row["INCLUDE_AUDIO"]:
        return "SUDS+AU+Audio"
    elif row["INCLUDE_AU"] and not row["INCLUDE_AUDIO"]:
        return "SUDS+AU"
    elif not row["INCLUDE_AU"] and row["INCLUDE_AUDIO"]:
        return "SUDS+Audio"
    else:
        return "SUDS only"
df_rs["condition"] = df_rs.apply(set_condition, axis=1)
df_suds["condition"] = df_suds.apply(set_condition, axis=1)
df_rs = df_rs.query("condition == 'SUDS only' or condition == 'SUDS+AU+Audio'")
df_suds = df_suds.query("condition == 'SUDS only' or condition == 'SUDS+AU+Audio'")


# second comparison now, set hue to be region
plt.figure(figsize=(3, 1.5))
for rs_idx, RS_value in enumerate([False, True]):
    if RS_value:
        df = df_rs.copy().query("RS == True and SHUFFLE == False")
    else:
        df = df_suds.copy().query("SHUFFLE == False")

    plt.subplot(1, 2, rs_idx + 1)
    plt.title("RS decoding" if RS_value else "SUDS decoding")
    sns.boxplot(
        data=df, hue="condition", x="region", y="r",
        order=["SC", "C", "all"],
        hue_order=["SUDS only", "SUDS+AU+Audio"],
        palette="viridis",
        showmeans=True,
        showfliers=False,
        linewidth=0.6,
        meanprops={"marker": "^", "markersize": 2},
        fliersize=0
    )
    sns.swarmplot(
        data=df, hue="condition", x="region", y="r",
        order=["SC", "C", "all"],
        hue_order=["SUDS only", "SUDS+AU+Audio"],
        dodge=True,
        color=".25",
        size=2,
        linewidth=0.6,
        alpha=1,
        legend=False
    )
    ax = plt.gca()
    ax.tick_params(axis="x", which="major", length=1.0, width=0.6, labelsize=5)
    ax.tick_params(axis="y", which="major", length=1.0, width=0.6, labelsize=5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    ax.set_xlabel("Region", fontsize=5)
    ax.set_ylabel("Decoding r", fontsize=5)
    ax.set_title("RS decoding" if RS_value else "SUDS decoding", fontsize=5)
    sns.despine()
    print(f"RS={RS_value}")
    for region in ["SC", "C", "all"]:
        for condition in ["SUDS only", "SUDS+AU+Audio"]:
            subset = df.query(f"region == '{region}' and condition == '{condition}'")
            mean = subset["r"].mean()
            std = subset["r"].std()
            print(f"  Region: {region}, Condition: {condition}, Mean: {mean:.3f}, Std: {std:.3f}")
    # compare with a scipy permutation test for each region the two conditions
    for region in ["SC", "C", "all"]:
        subset1 = df.query(f"region == '{region}' and condition == 'SUDS only'")["r"]
        subset2 = df.query(f"region == '{region}' and condition == 'SUDS+AU+Audio'")["r"]

        perm_result = stats.permutation_test((subset2.values - subset1.values, np.zeros_like(subset1)),
                            statistic=lambda x, y: np.mean(x) - np.mean(y),
                            n_resamples=5000, alternative='greater')
        print(f"  Region: {region}, Permutation test p-value: {perm_result.pvalue:.4f}")
plt.savefig(f"figures/Figure4/comparison_region-wise_1.pdf")

# RS=True
#   Region: SC, Condition: SUDS only, Mean: 0.349, Std: 0.321
#   Region: SC, Condition: SUDS+AU+Audio, Mean: 0.408, Std: 0.580
#   Region: C, Condition: SUDS only, Mean: 0.328, Std: 0.357
#   Region: C, Condition: SUDS+AU+Audio, Mean: 0.502, Std: 0.326
#   Region: all, Condition: SUDS only, Mean: 0.270, Std: 0.304
#   Region: all, Condition: SUDS+AU+Audio, Mean: 0.545, Std: 0.239
#   Region: SC, Permutation test p-value: 0.4543
#   Region: C, Permutation test p-value: 0.0635
#   Region: all, Permutation test p-value: 0.0026

# RS=False
#   Region: SC, Condition: SUDS only, Mean: 0.078, Std: 0.199
#   Region: SC, Condition: SUDS+AU+Audio, Mean: 0.167, Std: 0.207
#   Region: C, Condition: SUDS only, Mean: -0.032, Std: 0.235
#   Region: C, Condition: SUDS+AU+Audio, Mean: 0.405, Std: 0.258
#   Region: all, Condition: SUDS only, Mean: 0.115, Std: 0.329
#   Region: all, Condition: SUDS+AU+Audio, Mean: 0.387, Std: 0.253
#   Region: SC, Permutation test p-value: 0.2002
#   Region: C, Permutation test p-value: 0.0143
#   Region: all, Permutation test p-value: 0.0003
