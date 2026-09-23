import pandas as pd
from scipy import stats
import numpy as np
import pickle
import matplotlib
import seaborn as sns
# set arial font and smaller global figure text size
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 5
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

subjects = [4, 5, 7, 8, 9, 10, 11, 12]

def set_condition(row):
    if row["INCLUDE_AU"] and row["INCLUDE_AUDIO"]:
        return "SUDS+AU+Audio"
    elif row["INCLUDE_AU"] and not row["INCLUDE_AUDIO"]:
        return "SUDS+AU"
    elif not row["INCLUDE_AU"] and row["INCLUDE_AUDIO"]:
        return "SUDS+Audio"
    else:
        return "SUDS only"

RS_ = False
if RS_ == True:
    df = pd.read_csv(f"plotting/Figure4/source_data/ybocs_decoding_results_neural_audio_fau_49.csv")
    # df["SHUFFLE"] = False
    # df_shuffled = pd.read_csv(f"plotting/Figure3/source_data/ybocs_decoding_results_neural_audio_fau_SHUFFLED.csv")
    # df = pd.concat([df, df_shuffled], ignore_index=True)
    
else:
    #df = pd.read_csv(f"plotting/Figure4/source_data/suds_decoding_results_neural_audio_fau.csv")
    df = pd.read_csv(f"plotting/Figure4/source_data/suds_decoding_results_neural_audio_fau_10.csv")

df["condition"] = df.apply(set_condition, axis=1)


df_plt = df.query("region == 'all' and RS == @RS_").copy()
df_comp = df_plt.query("condition == 'SUDS Only' or condition == 'SUDS+AU+Audio'")
# make a table with r, p, and r_shuffled, p_shuffled for each subject just with RS == False
df_wide = pd.DataFrame()
for subject in subjects:
    for condition in ["SUDS Only", "SUDS+AU+Audio"]:
        row_true = df_comp.query("subject == @subject and condition == @condition and SHUFFLE == False")
        row_shuf  = df_comp.query("subject == @subject and condition == @condition and SHUFFLE == True")
        if row_true.shape[0] == 0 or row_shuf.shape[0] == 0:
            continue
        df_wide = pd.concat([df_wide, pd.DataFrame({
            "subject": [subject],
            "condition": [condition],
            "r": [row_true["r"].values[0]],
            "p": [row_true["p"].values[0]],
            "r_shuffled": [row_shuf["r"].values[0]],
            "p_shuffled": [row_shuf["p"].values[0]],
        })], ignore_index=True)


plt.figure(figsize=(1.9, 1.6))

df_plt = df.query("region == 'all' and RS == @RS_").copy()
order = ["SUDS only", "SUDS+AU", "SUDS+Audio", "SUDS+AU+Audio"]
hue_order = [True, False]   # keep what you want

ax = sns.boxplot(
    data=df_plt,
    x="condition",
    y="r",
    order=order,
    hue="SHUFFLE",
    hue_order=hue_order,
    palette="viridis",
    showmeans=True,
    showfliers=False,
    linewidth=0.6,
    fliersize=0,
    meanprops={"marker": "^", "markersize": 2}
)

sns.swarmplot(
    data=df_plt,
    x="condition",
    y="r",
    order=order,
    hue="SHUFFLE",
    hue_order=hue_order,
    dodge=True,
    color=".25",
    size=2,
    linewidth=0.6,
    alpha=1,
    ax=ax
)

# ---- CONNECT SUBJECT PAIRS ----
xticks = {cond: i for i, cond in enumerate(order)}
dodge = 0.4
offset = dodge / 2

# map hue value -> x offset based on hue_order
hue_to_offset = {
    hue_order[0]: -offset,  # first hue goes left
    hue_order[1]: +offset   # second hue goes right
}

for subject, df_sub in df_plt.groupby("subject"):
    for cond, df_sc in df_sub.groupby("condition"):
        if not ({True, False}.issubset(set(df_sc["SHUFFLE"]))):
            continue

        x_center = xticks[cond]
        y_true  = df_sc.loc[df_sc["SHUFFLE"] == True,  "r"].iloc[0]
        y_false = df_sc.loc[df_sc["SHUFFLE"] == False, "r"].iloc[0]

        x_true  = x_center + hue_to_offset[True]
        x_false = x_center + hue_to_offset[False]

        ax.plot(
            [x_true, x_false],
            [y_true, y_false],
            color="gray",
            alpha=0.4,
            linewidth=0.6,
            zorder=0
        )

# set ylim to -0.5 to 1.0
ax.set_ylim(-0.6, 1.0)
# ---- CLEAN LEGEND (remove duplicates from swarmplot) ----
handles, labels = ax.get_legend_handles_labels()
# First two correspond to hue_order in most seaborn versions
ax.legend(handles[:2], labels[:2], title="SHUFFLE", fontsize=5, title_fontsize=5)

ax.set_xlabel("Condition", fontsize=5)
ax.set_ylabel("Decoding r", fontsize=5)
#ax.grid(axis="y", alpha=0.2)
ax.tick_params(axis="x", which="major", length=1.0, width=0.6, labelsize=5)
ax.tick_params(axis="y", which="major", length=1.0, width=0.6, labelsize=5)
for spine in ax.spines.values():
    spine.set_linewidth(0.6)
plt.suptitle(f"Region: all, RS: {RS_}", fontsize=5)
sns.despine()
plt.tight_layout()
plt.savefig(f"figures/Figure4/decoding_vs_shuffled_rs_{RS_}_1.pdf")

# compute statistics, relative permutation test between SHUFFLE and non-SHUFFLE using scipy.stats.permutation_test
#for RS_ in [True, False]:

df_plt = df.query("region == 'all' and RS == @RS_").copy()
print(f"RS: {RS_}")
order = ["SUDS only", "SUDS+AU", "SUDS+Audio", "SUDS+AU+Audio"]
for cond in order:
    df_cond = df_plt.query("condition == @cond")
    r_true = df_cond.loc[df_cond["SHUFFLE"] == False, "r"].values
    r_shuf = df_cond.loc[df_cond["SHUFFLE"] == True,  "r"].values

    def statistic(x, y):
        return np.mean(x) - np.mean(y)

    res = stats.permutation_test(
        (r_true, r_shuf),
        statistic,
        vectorized=False,
        n_resamples=5000,
        alternative='greater'
    )
    print(f"Condition: {cond}, p-value (permutation test): {res.pvalue}")

    # print mean and std for each condition and shuffle
    mean_true = np.mean(r_true)
    std_true = np.std(r_true)
    mean_shuf = np.mean(r_shuf)
    std_shuf = np.std(r_shuf)
    print(f"  SHUFFLE=False: Mean: {mean_true:.3f}, Std: {std_true:.3f}")
    print(f"  SHUFFLE=True: Mean: {mean_shuf:.3f}, Std: {std_shuf:.3f}")

print()

# SUDS:
# Condition: SUDS only, p-value (permutation test): 0.05856643356643357
#   SHUFFLE=False: Mean: 0.115, Std: 0.305
#   SHUFFLE=True: Mean: -0.140, Std: 0.207
# Condition: SUDS+AU, p-value (permutation test): 0.24737762237762237
#   SHUFFLE=False: Mean: 0.121, Std: 0.178
#   SHUFFLE=True: Mean: 0.056, Std: 0.147
# Condition: SUDS+Audio, p-value (permutation test): 0.08304195804195805
#   SHUFFLE=False: Mean: 0.202, Std: 0.282
#   SHUFFLE=True: Mean: -0.024, Std: 0.250
# Condition: SUDS+AU+Audio, p-value (permutation test): 0.0026223776223776225
#   SHUFFLE=False: Mean: 0.387, Std: 0.234
#   SHUFFLE=True: Mean: -0.033, Std: 0.172

# RS: True
# Condition: SUDS only, p-value (permutation test): 0.002599480103979204
#   SHUFFLE=False: Mean: 0.270, Std: 0.284
#   SHUFFLE=True: Mean: -0.148, Std: 0.162
# Condition: SUDS+AU, p-value (permutation test): 0.0057988402319536095
#   SHUFFLE=False: Mean: 0.439, Std: 0.365
#   SHUFFLE=True: Mean: -0.102, Std: 0.255
# Condition: SUDS+Audio, p-value (permutation test): 0.0057988402319536095
#   SHUFFLE=False: Mean: 0.551, Std: 0.387
#   SHUFFLE=True: Mean: 0.035, Std: 0.230
# Condition: SUDS+AU+Audio, p-value (permutation test): 0.0007998400319936012
#   SHUFFLE=False: Mean: 0.545, Std: 0.224
#   SHUFFLE=True: Mean: -0.056, Std: 0.284