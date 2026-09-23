import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy import stats

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

#SC_L_features = [f"SC_L_{feat}" for feat in feature_names]
#SC_R_features = [f"SC_R_{feat}" for feat in feature_names]

hue_colors = {
    "within_day_0": "#3274A1",
    "within_day_not0": "#45A778",
    "across_days": "#6C6C6C"
}


def add_significance_bars(ax, available_groups, comparisons):
    group_order = ["within_day_0", "within_day_not0", "across_days"]
    positions = {label: idx for idx, label in enumerate(group_order) if label in available_groups}
    if len(positions) < 2:
        return

    y_min, y_max = ax.get_ylim()
    span = y_max - y_min
    baseline = y_max + 0.08 * span
    max_bar_top = baseline

    for idx, (pair, p_value) in enumerate(comparisons.items()):
        group_a, group_b = pair
        if group_a not in positions or group_b not in positions:
            continue

        x1 = positions[group_a]
        x2 = positions[group_b]
        y_top = baseline + idx * 0.2 * span
        max_bar_top = max(max_bar_top, y_top + 0.1 * span)

        ax.plot([x1, x1, x2, x2], [y_top, y_top + 0.03 * span, y_top + 0.03 * span, y_top],
                color="black", linewidth=0.8, solid_capstyle="round")
        label = "*" if p_value < 0.05 else "n.s."
        ax.text((x1 + x2) / 2, y_top + 0.035 * span, label, ha="center", va="bottom",
                fontsize=5)

    ax.set_ylim(y_min, max_bar_top + 0.05 * span)


df = pd.read_csv("plotting/Figure2/source_data/ybocs_feature_stim_merged_per_sess.csv")
df["TEED_left"] = df["Amplitude_mA_left"] * (df["Frequency_Hz_left"]) * (df["PulseWidth_us_left"] * 0.001)
df["TEED_right"] = df["Amplitude_mA_right"] * (df["Frequency_Hz_right"]) * (df["PulseWidth_us_right"]* 0.001)
df["TEED_mean"] = df[["TEED_left", "TEED_right"]].mean(axis=1)

patients = df["subject"].unique()
prev_rs2_df = None
TEED_prev = None

diff_all = []
rs_comp_all = []
teeds_ = []

for patient in patients:
    df_sub = df[df["subject"] == patient]
    days = df_sub["date"].unique()
    if patient == "aDBS012":
        print("check")  # aDBS012 had always the same parameters for rs1 and rs2, do TEED diff always 0

    for day in days:
        df_sub_day = df_sub[df_sub["date"] == day]
        rs_names = df_sub_day["rs_name"].unique()
        if "resting-state1" not in rs_names or "resting-state2" not in rs_names:
            continue
        df_rs1 = df_sub_day[df_sub_day["rs_name"] == "resting-state1"]
        df_rs2 = df_sub_day[df_sub_day["rs_name"] == "resting-state2"]

        for ch in ["SC_L", "SC_R", "C_L_1", "C_R_1", "C_L_2", "C_R_2"]:
            if "L" in ch:
                hem = "left"
            else:
                hem = "right"
            feature_cols = [f"{ch}_{feat}" for feat in feature_names]
            TEED_diff_within_day = df_rs2[f"TEED_{hem}"].values[0] - df_rs1[f"TEED_{hem}"].values[0]
            df_rs1_features = df_rs1[feature_cols].reset_index(drop=True)
            df_rs2_features = df_rs2[feature_cols].reset_index(drop=True)

            diff_within_day = np.abs(df_rs2_features - df_rs1_features)
            diff_within_day.columns = feature_names
            diff_within_day["TEED_diff"] = TEED_diff_within_day
            diff_within_day["hem"] = hem
            diff_within_day["sub"] = patient
            diff_within_day["ch"] = ch
            diff_within_day["day"] = day
            diff_within_day["type"] = "within_day"
            #if diff_within_day["TEED_diff"].sum() != 0:
            diff_all.append(diff_within_day)
        
        # df_add = pd.concat([diff_within_day_L, diff_within_day_R], ignore_index=True)
        # df_add["sub"] = patient
        # df_add["day"] = day
        # df_add["type"] = "within_day"

        if prev_rs2_df is not None:
            for ch in ["SC_L", "SC_R", "C_L_1", "C_R_1", "C_L_2", "C_R_2"]:
                feature_cols = [f"{ch}_{feat}" for feat in feature_names]
                if "L" in ch:
                    hem = "left"
                    df_rs2_features_L = df_rs2[feature_cols].reset_index(drop=True)
                    prev_rs2 =  prev_rs2_df[feature_cols].reset_index(drop=True)
                    TEED_prev = TEED_L_prev
                else:
                    hem = "right"
                    df_rs2_features_R = df_rs2[feature_cols].reset_index(drop=True)
                    prev_rs2 = prev_rs2_df[feature_cols].reset_index(drop=True)
                    TEED_prev = TEED_R_prev

                TEED_diff_across_days = df_rs1[f"TEED_{hem}"].values[0] - TEED_prev
                diff_across_days = np.abs(df_rs1[feature_cols].reset_index(drop=True) - prev_rs2)
                diff_across_days.columns = feature_names
                diff_across_days["TEED_diff"] = TEED_diff_across_days
                diff_across_days["hem"] = hem
                diff_across_days["ch"] = ch
                diff_across_days["sub"] = patient
                diff_across_days["day"] = day
                diff_across_days["type"] = "across_days"
            
                if diff_across_days["TEED_diff"].sum() == 0:
                    diff_all.append(diff_across_days)
        
        prev_rs2_df = df_rs2.reset_index(drop=True)
        #prev_rs2_R = df_rs2_features_R.reset_index(drop=True)
        TEED_L_prev = df_rs2["TEED_left"].values[0]
        TEED_R_prev = df_rs2["TEED_right"].values[0]


df_diff = pd.concat(diff_all, ignore_index=True)
# check where TEED_diff is not NaN
df_diff = df_diff[df_diff["TEED_diff"].notna()]
diff_diff = df_diff[df_diff["delta"].notna()]

# pivot table s.t. features are in one column
df_diff_melted = df_diff.melt(id_vars=["sub", "day", "type", "ch", "TEED_diff"], 
                              value_vars=feature_names,
                              var_name="feature",
                              value_name="feature_value")

df_diff_melted_g = df_diff_melted.groupby(["feature", "type", "ch", "sub"])["feature_value"].mean().reset_index()
# for each feature, type, hem and sub, divide feature_value from across_days by within_day
df_diff_ratio = df_diff_melted_g.pivot_table(index=["feature", "sub", "ch"], columns="type", values="feature_value").reset_index()
df_diff_ratio["ratio_across_within"] = df_diff_ratio["across_days"] / (df_diff_ratio["across_days"]  + df_diff_ratio["within_day"])
df_diff_ratio["ratio_across_within"] = (df_diff_ratio["across_days"] - df_diff_ratio["within_day"]) / (df_diff_ratio["across_days"] + df_diff_ratio["within_day"]) # 

diffs_ = []
for sub in df_diff["sub"].unique():
    for hem in ["left", "right"]:
        df_sub = df_diff[(df_diff["sub"] == sub) & (df_diff["hem"] == hem)]
        mean_ = df_sub.groupby("type")[feature_names + ["TEED_diff"]].mean().reset_index()
        mean_["sub"] = sub
        mean_["hem"] = hem
        diffs_.append(mean_)
df_mean_diffs = pd.concat(diffs_, ignore_index=True)

df_means = []
df_means_within_comp = []
for ch in df_diff_melted["ch"].unique():
    df_test = df_diff_melted.query("ch == @ch")
    for sub in df_diff_melted["sub"].unique():
        if ch.startswith("C") and sub in ["aDBS004", "aDBS005", "aDBS007"]:
            continue  # these subjects don't have cortical electrodes
        df_sub = df_test[df_test["sub"] == sub]
        for feature in feature_names:
            # z-score normalize feature values
            df_sub_feature = df_sub[df_sub["feature"] == feature]
            df_sub_feature[f"{feature}_norm"] = (df_sub_feature["feature_value"] - df_sub_feature["feature_value"].mean()) / df_sub_feature["feature_value"].std()
            # mean for across_days and within_day
            mean_across = df_sub_feature[df_sub_feature["type"] == "across_days"][f"{feature}_norm"].mean()
            mean_within = df_sub_feature[df_sub_feature["type"] == "within_day"][f"{feature}_norm"].mean()
            mean_within_TEED_diff_0 = df_sub_feature[(df_sub_feature["type"] == "within_day") & (df_sub_feature["TEED_diff"] == 0)][f"{feature}_norm"].mean()
            mean_within_TEED_diff_not0 = df_sub_feature[(df_sub_feature["type"] == "within_day") & (df_sub_feature["TEED_diff"] != 0)][f"{feature}_norm"].mean()
            df_means.append({
                "sub": sub,
                "feature": feature,
                "ch": ch,
                "diff": mean_across,
                "type": "across",
            })
            df_means.append({
                "sub": sub,
                "feature": feature,
                "ch": ch,
                "diff": mean_within,
                "type": "within",
            })

            if not np.isnan(mean_within_TEED_diff_0) and not np.isnan(mean_within_TEED_diff_not0):
                df_means_within_comp.append({
                    "sub": sub,
                    "feature": feature,
                    "ch": ch,
                    "diff": mean_within_TEED_diff_0,
                    "type": "within_TEED_0",
                })
                df_means_within_comp.append({
                    "sub": sub,
                    "feature": feature,
                    "ch": ch,
                    "diff": mean_within_TEED_diff_not0,
                    "type": "within_TEED_not0",
                })

df_means_final = pd.DataFrame(df_means)
df_means_final["diff_abs"] = df_means_final["diff"].abs()
df_means_within_comp_final = pd.DataFrame(df_means_within_comp)
df_means_within_comp_final["diff_abs"] = df_means_within_comp_final["diff"].abs()


across_ = df_means_final[df_means_final["type"] == "across"]["diff_abs"]
across_ = across_[~across_.isna()]
within_ = df_means_final[df_means_final["type"] == "within"]["diff_abs"]
within_ = within_[~within_.isna()]
stats_result = stats.permutation_test(
    (across_.values, within_.values),
    statistic=lambda x, y: np.mean(x) - np.mean(y),
    alternative='greater',
    n_resamples=5000,
    random_state=42
)
p_val = stats_result.pvalue
# print mean and std of both conditions
mean_across = across_.mean()
std_across = across_.std()
mean_within = within_.mean()
std_within = within_.std()
print(f"Mean across days: {mean_across:.3f} +/- {std_across:.3f}")
print(f"Mean within days: {mean_within:.3f} +/- {std_within:.3f}")
print(f"Permutation test p-value: {p_val:.5f}")

# Mean across days: 0.570 +/- 0.505
# Mean within days: 0.138 +/- 0.118
# Permutation test p-value: 0.00020

# run the same for TEED differences
ch_plt = "SC_R"
df_plt_TEED = df_diff_melted.query("ch == @ch_plt and feature == 'delta'")
df_plt_TEED["TEED_diff_abs"] = df_plt_TEED["TEED_diff"].abs()
df_plt_TEED_across = df_plt_TEED[df_plt_TEED["type"] == "across_days"]["TEED_diff_abs"]
df_plt_TEED_across = df_plt_TEED_across[~df_plt_TEED_across.isna()]
df_plt_TEED_within = df_plt_TEED[df_plt_TEED["type"] == "within_day"]["TEED_diff_abs"]
df_plt_TEED_within = df_plt_TEED_within[~df_plt_TEED_within.isna()]

stats_result_TEED = stats.permutation_test(
    (df_plt_TEED_across.values, df_plt_TEED_within.values),
    statistic=lambda x, y: np.mean(x) - np.mean(y),
    alternative='less',
    n_resamples=5000,
    random_state=42
)
p_val_TEED = stats_result_TEED.pvalue
mean_TEED_across = df_plt_TEED_across.mean()
std_TEED_across = df_plt_TEED_across.std()
mean_TEED_within = df_plt_TEED_within.mean()
std_TEED_within = df_plt_TEED_within.std()
print(f"Mean TEED across days: {mean_TEED_across:.3f}  +/- {std_TEED_across:.3f}")
print(f"Mean TEED within days: {mean_TEED_within:.3f} +/- {std_TEED_within:.3f}")
print(f"Permutation test TEED p-value: {p_val_TEED:.5f}")

# Mean TEED across days: 0.000  +/- 0.000
# Mean TEED within days: 8.062 +/- 15.662
# Permutation test TEED p-value: 0.00020


plt.figure(figsize=(8, 5))

PLT_DIFF_MEAN_PATIENTS = False
# if PLT_DIFF_MEAN_PATIENTS:
#     df_use = df_mean_diffs
# else:
ch_plt = "SC_L"
ch_plt = "C_L_1"
font_size_ylabel = 5
#df_use = df_diff_melted.query("ch == @ch_plt")
df_use = df_diff_melted.copy()
# set a new column type_, that is "within_day" or "across_days"
# but if within day, set wtihin_day_0 if TEED_diff == 0 else within_day_not0
def set_type(row):
    if row["type"] == "within_day":
        if row["TEED_diff"] == 0:
            return "within_day_0"
        else:
            return "within_day_not0"
    else:
        return "across_days"
df_use["type_"] = df_use.apply(set_type, axis=1)

plt.subplot(4, 8, 1)
df_plt_TEED = df_diff_melted.query("ch == @ch_plt and feature == 'delta'")
df_plt_TEED["type_"] = df_plt_TEED.apply(set_type, axis=1)
# make mean symbol smaller
sns.boxplot(
    data=df_plt_TEED, y="TEED_diff", x="type_", showmeans=True, showfliers=False,
    order=["within_day_0", "within_day_not0", "across_days"],
    palette=hue_colors, meanprops={"markersize":2}
)
sns.despine()
df_acrossday = df_plt_TEED.query("type_ == 'across_days'")["TEED_diff"]
df_withinday_0 = df_plt_TEED.query("type_ == 'within_day_0'")["TEED_diff"]
df_withinday_not0 = df_plt_TEED.query("type_ == 'within_day_not0'")["TEED_diff"]
df_acrossday = df_acrossday[~df_acrossday.isna()]
df_withinday_0 = df_withinday_0[~df_withinday_0.isna()]
df_withinday_not0 = df_withinday_not0[~df_withinday_not0.isna()]
# run permuation test
stat = stats.permutation_test((df_acrossday.values, df_withinday_0.values),
                                statistic=lambda x, y: np.mean(x) - np.mean(y),
                                alternative='two-sided',
                                n_resamples=5000,
                                random_state=42)
p_a_vs_w0 = stat.pvalue
stat = stats.permutation_test((df_acrossday.values, df_withinday_not0.values),
                                statistic=lambda x, y: np.mean(x) - np.mean(y),
                                alternative='two-sided',
                                n_resamples=5000,
                                random_state=42)
p_a_vs_wnot0 = stat.pvalue
stat = stats.permutation_test((df_withinday_0.values, df_withinday_not0.values),
                                statistic=lambda x, y: np.mean(x) - np.mean(y),
                                alternative='two-sided',
                                n_resamples=5000,
                                random_state=42)
p_w0_vs_wnot0 = stat.pvalue
#plt.title(f"TEED_diff\np_a_vs_w0: {p_a_vs_w0:.3f}\n p_a_vs_wnot0: {p_a_vs_wnot0:.3f}\n p_w0_vs_wnot0: {p_w0_vs_wnot0:.3f}", fontsize=9)
# set ytick fontsize
plt.ylabel("TEED_diff", fontsize=font_size_ylabel)
plt.tick_params(axis='y', labelsize=font_size_ylabel)
# remove xlabel
plt.xlabel("")
# remove x tick labels
plt.xticks([])
# keep only middle xtick

# add significance bars
add_significance_bars(
    plt.gca(),
    set(df_plt_TEED["type_"].dropna().unique()),
    {
        ("across_days", "within_day_0"): p_a_vs_w0,
        ("across_days", "within_day_not0"): p_a_vs_wnot0,
        ("within_day_0", "within_day_not0"): p_w0_vs_wnot0,
    },
)

for i, feature in enumerate(feature_names):
    ax = plt.subplot(4, 8, i+2)
    
    if feature == "burst_duration_gamma_ms":
        print("check")

    df_acrossday = df_use.query("type == 'across_days' and feature == @feature")["feature_value"]
    df_withinday = df_use.query("type == 'within_day' and feature == @feature")["feature_value"]
    df_withinday0 = df_use.query("type_ == 'within_day_0' and feature == @feature")["feature_value"]
    df_withindaynot0 = df_use.query("type_ == 'within_day_not0' and feature == @feature")["feature_value"]

    df_acrossday = df_acrossday[~df_acrossday.isna()]
    df_withinday0 = df_withinday0[~df_withinday0.isna()]
    df_withindaynot0 = df_withindaynot0[~df_withindaynot0.isna()]

    # run permuation test
    stat = stats.permutation_test((df_acrossday.values, df_withinday0.values),
                                    statistic=lambda x, y: np.median(x) - np.median(y),
                                    alternative='greater',
                                    n_resamples=5000,
                                    random_state=42)
    p_a_vs_w0 = stat.pvalue
    stat = stats.permutation_test((df_acrossday.values, df_withindaynot0.values),
                                    statistic=lambda x, y: np.median(x) - np.median(y),
                                    alternative='greater',
                                    n_resamples=5000,
                                    random_state=42)
    p_a_vs_wnot0 = stat.pvalue
    stat = stats.permutation_test((df_withindaynot0.values, df_withinday0.values),
                                    statistic=lambda x, y: np.median(x) - np.median(y),
                                    alternative='two-sided',
                                    n_resamples=5000,
                                    random_state=42)
    p_w0_vs_wnot0 = stat.pvalue

    # normalize feature
    df_plt_use = df_use.query("feature == @feature").copy()
    #df_plt_use[f"feature_value_norm"] = (df_plt_use["feature_value"] - df_plt_use["feature_value"].mean()) / df_plt_use["feature_value"].std()
    sns.boxplot(
        data=df_plt_use, y=f"feature_value",  # df_mean_diffs to show just one dot per subject
        x="type_", showmeans=True, ax=ax, showfliers=False,
        palette=hue_colors, meanprops={"markersize":2},
        order=["within_day_0", "within_day_not0", "across_days"]
    )
    if PLT_DIFF_MEAN_PATIENTS:
        sns.swarmplot(
            data=df_use, y=f"{feature}_norm",  # df_mean_diffs
            hue="type", dodge=True, color=".25", ax=ax
        )
    plt.ylabel(f"{feature}", fontsize=font_size_ylabel)
    #ax.set_title(f"p_a_vs_w0: {p_a_vs_w0:.3f}\n p_a_vs_wnot0: {p_a_vs_wnot0:.3f}\n p_w0_vs_wnot0: {p_w0_vs_wnot0:.3f}", fontsize=9)
    add_significance_bars(
        ax,
        set(df_plt_use["type_"].dropna().unique()),
        {
            ("across_days", "within_day_0"): p_a_vs_w0,
            ("across_days", "within_day_not0"): p_a_vs_wnot0,
            ("within_day_0", "within_day_not0"): p_w0_vs_wnot0,
        },
    )
    # set fontsize of ylabel
    ax.set_ylabel(ax.get_ylabel(), fontsize=font_size_ylabel)
    # set ytick fontsize
    ax.tick_params(axis='y', labelsize=font_size_ylabel)
    # remove xlabel
    ax.set_xlabel("")
    # remove x tick labels
    ax.set_xticklabels([])
    # make x and y tick length smaller
    ax.tick_params(axis='both', which='major', length=2)
    

    # Remove legend from EVERY subplot
    if i != 0:
        ax.legend([], [], frameon=False)

    sns.despine(ax=ax)


plt.tight_layout()
plt.savefig("figures/Figure2/TEED_feature_changes_1.pdf")
