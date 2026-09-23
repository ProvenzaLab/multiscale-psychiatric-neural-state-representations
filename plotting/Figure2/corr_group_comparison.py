import pandas as pd
import matplotlib as mpl
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 5,
    'axes.unicode_minus': False
})
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

df_ybocs = pd.read_csv("plotting/Figure2/source_data/feature_region_subject_correlations_ybocs_1.csv")
df_ybocs["YBOCS"] = True
# set column "group" to "VCVS" if region starts with SC_ else "Cortex_" 
df_ybocs["group"] = df_ybocs["region"].apply(lambda x: "VCVS" if x.startswith("SC_") else "Cortex")

df_suds = pd.read_csv("plotting/Figure2/source_data/feature_region_subject_correlations_suds.csv")
df_suds["YBOCS"] = False
df_suds["group"] = df_suds["region"].apply(lambda x: "VCVS" if x.startswith("SC_") else "Cortex")


df_coh_ybocs = pd.read_csv("plotting/Figure2/source_data/feature_region_subject_correlations_ybocs_coherence_1.csv")
df_coh_ybocs["YBOCS"] = True
df_coh_ybocs["group"] = "coherence"
df_coh_suds = pd.read_csv("plotting/Figure2/source_data/feature_region_subject_correlations_suds_coherence.csv")
df_coh_suds["YBOCS"] = False
df_coh_suds["group"] = "coherence"

df_all = pd.concat([df_ybocs, df_suds, df_coh_ybocs, df_coh_suds], ignore_index=True)
df_all["corr_abs"] = df_all["correlation"].abs()
df_all["hem"] = df_all["region"].apply(lambda x: x.split("_")[1])


df_corr_lat_comp = df_all.dropna(subset=["corr_abs"]).query("hem == 'L' or hem == 'R'").groupby(["subject", "hem", "YBOCS", "feature", "group"])["corr_abs"].max().reset_index()
df_corr_lat_comp_mean = df_corr_lat_comp.groupby([ "YBOCS", "group", "hem",])["corr_abs"].mean()

plt.figure(figsize=(3, 1.5))
font_size = 5

def set_compact_ticks(ax):
    ax.tick_params(axis="x", which="major", length=1.0, width=0.5)
    ax.tick_params(axis="y", which="major", length=1.0, width=0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)


def add_group_significance_bars(
    ax,
    groups,
    comparisons,
    y_max=1.0,
    top=0.97,
    step=0.08,
    bar_height=0.015,
    fontsize=5,
):
    """
    Add significance bars above categorical groups.

    Parameters
    ----------
    ax : matplotlib Axes
    groups : list
        Groups in their plotted x-axis order.
        Example: ["Left", "Right"]
        or ["coherence", "VCVS", "Cortex"].
    comparisons : dict
        Dict mapping (group1, group2) -> p-value.
    y_max : float
        Upper y-axis limit.
    top : float
        Position of highest significance bar as fraction of y_max.
    step : float
        Vertical spacing between stacked bars as fraction of y_max.
    bar_height : float
        Height of vertical ticks on the significance bar.
    fontsize : float
        Significance-label font size.
    """

    positions = {group: i for i, group in enumerate(groups)}

    # only keep valid comparisons
    valid_comparisons = [
        (pair, p)
        for pair, p in comparisons.items()
        if pair[0] in positions and pair[1] in positions
    ]

    # Put longer comparisons higher so bars don't cross
    valid_comparisons.sort(
        key=lambda item: abs(
            positions[item[0][1]] - positions[item[0][0]]
        )
    )

    n = len(valid_comparisons)

    for i, ((group_a, group_b), p_value) in enumerate(valid_comparisons):

        x1 = positions[group_a]
        x2 = positions[group_b]

        # stack bars upward
        y = y_max * (top - step * (n - 1 - i))
        h = y_max * bar_height

        ax.plot(
            [x1, x1, x2, x2],
            [y - h, y, y, y - h],
            color="black",
            linewidth=0.5,
            clip_on=False,
        )

        # significance label
        #if p_value < 0.0001:
        #    label = "****"
        #elif p_value < 0.001:
        #    label = "***"
        #elif p_value < 0.01:
        #    label = "**"
        if p_value < 0.05:
            label = "*"
        else:
            label = "n.s."

        ax.text(
            (x1 + x2) / 2,
            y + 0.01 * y_max,
            label,
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )

    ax.set_ylim(0, y_max)
# plot for each group (VCVS, Cortex) and YBOCS being (True, False) in a subplot swarmplot with overlayed boxplot
for group in ["VCVS", "Cortex"]:
    for ybocs in [True, False]:
        plt.subplot(1, 4, 1 if group == "VCVS" and ybocs == True else 2 if group == "Cortex" and ybocs == True else 3 if group == "VCVS" and ybocs == False else 4)
        df_plt = df_corr_lat_comp.query("group == @group and YBOCS == @ybocs")
        sns.boxplot(data=df_plt, x="hem", y="corr_abs", showmeans=True, showfliers=False, width=0.6, palette="viridis", linewidth=0.4, meanprops={"marker": "^", "markersize": 1.5})
        sns.swarmplot(data=df_plt, x="hem", y="corr_abs", color=".25", alpha=0.2,
                      size=0.5, linewidth=0.4)
        # run a paired permutation test for difference in means between L and R
        wide_ = df_plt.pivot(index=["subject", "feature"], columns="hem", values="corr_abs").dropna(subset=["L", "R"])
        diffs_ = wide_["L"] - wide_["R"]
        res_ = stats.permutation_test(
            (diffs_,),
            statistic=lambda x: np.mean(x),
            permutation_type="samples",   # <-- paired test
            n_resamples=5000,
            alternative="two-sided"
        )
        t_stat = res_.statistic
        p_val = res_.pvalue
        mean_L = df_plt.query("hem == 'L'")["corr_abs"].mean()
        std_L = df_plt.query("hem == 'L'")["corr_abs"].std()
        mean_R = df_plt.query("hem == 'R'")["corr_abs"].mean()
        std_R = df_plt.query("hem == 'R'")["corr_abs"].std()
        plt.title(f"{group} {'YBOCS' if ybocs else 'SUDS'}\np= {p_val:.4f}\nL: {mean_L:.2f} ± {std_L:.2f}\nR: {mean_R:.2f} ± {std_R:.2f}", fontsize=5)
        plt.ylabel("Pearson correlation (abs)", fontsize=5)
        # change x tick labels to L and R
        plt.xticks([0, 1], ["Left", "Right"])
        # remove xlabel
        plt.xlabel("")
        sns.despine()
        add_group_significance_bars(
            plt.gca(),
            ["Left", "Right"],
            {("Left", "Right"): p_val}
        )
        plt.ylim(0, 1)
        set_compact_ticks(plt.gca())

plt.tight_layout()
plt.savefig("figures/Figure2/corr_lateralization_1.pdf")
#plt.show()

df_plt  = df_all.dropna(subset=["corr_abs"]).query("hem == 'L' or hem == 'R'").groupby(["YBOCS", "subject", "hem", "feature"])["corr_abs"].max().reset_index()

# run permutation test for difference in means between YBOCS and SUDS

# run a paried test for difference in means between YBOCS and SUDS
# paired per feature, region, subject
wide_ = df_plt.pivot(index=["subject", "feature", "hem"], columns="YBOCS", values="corr_abs").dropna(subset=[False, True])
diffs_ = wide_[True] - wide_[False]
res_ = stats.permutation_test(
    (diffs_,),
    statistic=lambda x: np.mean(x),
    permutation_type="samples",   # <-- paired test
    n_resamples=10000,
    alternative="two-sided"
)
corrs_ybocs = df_plt.query("YBOCS == True")["corr_abs"]
corrs_suds = df_plt.query("YBOCS == False")["corr_abs"]

t = res_.statistic
p_val = res_.pvalue
print(f"SUDS vs YBOCS: t={t:.4f}, p={p_val:.4f}")
# mean and std for YBOCS and SUDS
mean_ybocs = corrs_ybocs.mean()
std_ybocs = corrs_ybocs.std()
mean_suds = corrs_suds.mean()
std_suds = corrs_suds.std()
print(f"YBOCS: mean={mean_ybocs:.3f}, std={std_ybocs:.3f}")
print(f"SUDS: mean={mean_suds:.3f}, std={std_suds:.3f}")

# SUDS vs YBOCS: t=0.193, p=0.0002
# YBOCS: mean=0.46, std=0.23
# SUDS: mean=0.244, std=0.168

plt.figure(figsize=(4.5, 2)) # 9
plt.subplot(1,5,1)
sns.boxplot(data=df_plt, x="YBOCS", y="corr_abs", showmeans=True, showfliers=False, width=0.6, palette="viridis", linewidth=0.4, meanprops={"marker": "^", "markersize": 1.5})
#sns.swarmplot(data=df_all, x="YBOCS", y="corr_abs", color=".25", alpha=0.2, size=3)
plt.ylim(0, 1)
plt.title("SUDS vs YBOCS", fontsize=5)
sns.despine()
add_group_significance_bars(plt.gca(), [False, True], { (False, True): p_val })


plt.subplot(1,5,2)
df_plt = df_all.query("YBOCS == False and group == 'VCVS'").groupby(["subject", "hem", "feature"])["corr_abs"].max().reset_index()
sns.boxplot(data=df_plt, x="hem", y="corr_abs", width=0.6,
            showmeans=True, showfliers=False, palette="viridis", linewidth=0.4, meanprops={"marker": "^", "markersize": 1.5})
sns.swarmplot(data=df_plt, x="hem", y="corr_abs", color=".25", alpha=0.2, size=0.5, linewidth=0.4)
sns.despine()
df_plt.groupby("hem")["corr_abs"].mean()
df_plt.groupby("hem")["corr_abs"].std()
wide_ = df_plt.pivot(index=["subject", "feature"], columns="hem", values="corr_abs").dropna(subset=["L", "R"])
diffs_ = wide_["L"] - wide_["R"]
res_ = stats.permutation_test(
    (diffs_,),
    statistic=lambda x: np.mean(x),
    permutation_type="samples",   # <-- paired test
    n_resamples=10000,
    alternative="greater"
)
t_stat = res_.statistic
p_val = res_.pvalue
print(f"L {wide_['L'].mean():.3f} ± {wide_['L'].std():.3f}, R {wide_['R'].mean():.3f} ± {wide_['R'].std():.3f}, p={p_val:.4f}")
# L 0.209 ± 0.153, R 0.113 ± 0.100, p=0.0001
plt.title("SUDS VCVS\nL vs R", fontsize=5)
add_group_significance_bars(plt.gca(), ["Left", "Right"], {("Left", "Right"): p_val})
plt.ylim(0, 1)

plt.subplot(1,5,3)
df_plt = df_all.query("YBOCS == True and group == 'VCVS'").groupby(["subject", "hem", "feature"])["corr_abs"].max().reset_index()
sns.boxplot(data=df_plt, x="hem", y="corr_abs", showmeans=True, showfliers=False, width=0.6, palette="viridis", linewidth=0.4, meanprops={"marker": "^", "markersize": 1.5})
sns.swarmplot(data=df_plt, x="hem", y="corr_abs", color=".25", alpha=0.2, size=0.5, linewidth=0.4)
plt.ylim(0, 1)
sns.despine()

df_plt.groupby("hem")["corr_abs"].mean()
df_plt.groupby("hem")["corr_abs"].std()
wide_ = df_plt.pivot(index=["subject", "feature"], columns="hem", values="corr_abs").dropna(subset=["L", "R"])
diffs_ = wide_["L"] - wide_["R"]
res_ = stats.permutation_test(
    (diffs_,),
    statistic=lambda x: np.mean(x),
    permutation_type="samples",   # <-- paired test
    n_resamples=10000,
    alternative="greater"
)
t_stat = res_.statistic
p_val = res_.pvalue
print(f"L {wide_['L'].mean():.3f} ± {wide_['L'].std():.3f}, R {wide_['R'].mean():.3f} ± {wide_['R'].std():.3f}, p={p_val:.4f}")
# pre: l: 0.40 pm 0.25, r: 0.38 pm 0.24, p=0.16
# L 0.327 ± 0.216, R 0.359 ± 0.233, p=0.9749

plt.title("YBOCS VCVS\nL vs R", fontsize=5)
add_group_significance_bars(plt.gca(), ["Left", "Right"], {("Left", "Right"): p_val})
plt.ylim(0, 1)

plt.subplot(1,5,4)
df_plt = df_all.query("YBOCS == False").groupby(["subject", "group", "feature"])["corr_abs"].max().reset_index()
sns.boxplot(data=df_plt, x="group", y="corr_abs", showmeans=True, showfliers=False, width=0.6, palette="viridis", order=["coherence", "VCVS", "Cortex"], linewidth=0.4, meanprops={"marker": "^", "markersize": 1.5})
sns.swarmplot(data=df_plt, x="group", y="corr_abs", color=".25", alpha=0.2, size=0.5, linewidth=0.4, order=["coherence", "VCVS", "Cortex"])
plt.ylim(0, 1)
plt.title("SUDS", fontsize=5)
sns.despine()

RELATIVE_ = False
df_plt = df_plt.dropna()
all_group_pvals = {}
for group_1 in ["VCVS", "Cortex", "coherence"]:
    for group_2 in ["VCVS", "Cortex", "coherence"]:
        if group_1 == group_2:
            continue
        corrs_1 = df_plt.query("group == @group_1")["corr_abs"]
        corrs_2 = df_plt.query("group == @group_2")["corr_abs"]
        if RELATIVE_:
            df_group1 = df_plt.query("group == @group_1")
            df_group2 = df_plt.query("group == @group_2")
            # get only the ones from both where subject and feature are the same
            if group_1 != "coherence" and group_2 != "coherence":
                df_group1 = df_group1.merge(df_group2[["subject", "feature"]], on=["subject", "feature"], how="inner")
                df_group2 = df_group2.merge(df_group1[["subject", "feature"]], on=["subject", "feature"], how="inner")
                corrs_1 = df_group1["corr_abs"]
                corrs_2 = df_group2["corr_abs"]
            else:
                # limit only to subject that are present in both groups
                subjects_1 = set(df_group1["subject"].unique())
                subjects_2 = set(df_group2["subject"].unique())
                subjects_common = subjects_1.intersection(subjects_2)
                df_group1 = df_group1[df_group1["subject"].isin(subjects_common)]
                df_group2 = df_group2[df_group2["subject"].isin(subjects_common)]
                corrs_1 = df_group1["corr_abs"]
                corrs_2 = df_group2["corr_abs"]
            if len(corrs_1) == 0 or len(corrs_2) == 0:
                #print(f"SUDS: {group_1} vs {group_2}: not enough data for relative comparison")
                continue
      
        res_ = stats.permutation_test(
            (corrs_1, corrs_2),
            statistic=lambda x, y: np.mean(x) - np.mean(y),
            vectorized=False,
            n_resamples=10000,
            alternative="two-sided"
        )
        t = res_.statistic
        p_val = res_.pvalue
        all_group_pvals[(group_1, group_2)] = p_val
        print(f"SUDS: {group_1} vs {group_2}: t={t:.4f}, p={p_val:.4f}")

add_group_significance_bars(
    plt.gca(),
    ["coherence", "VCVS", "Cortex"],
    {
        ("coherence", "VCVS"): all_group_pvals.get(("coherence", "VCVS"), 1.0),
        ("coherence", "Cortex"): all_group_pvals.get(("coherence", "Cortex"), 1.0),
        ("VCVS", "Cortex"): all_group_pvals.get(("VCVS", "Cortex"), 1.0),
    }
)

# plot mean and std for each group in one line 
means = df_plt.groupby("group")["corr_abs"].mean()
stds = df_plt.groupby("group")["corr_abs"].std()
for group in ["VCVS", "Cortex", "coherence"]:
    print(f"SUDS {group}: mean={means[group]:.3f}, std={stds[group]:.3f}")

# SUDS VCVS: mean=0.233, std=0.142
# SUDS Cortex: mean=0.322, std=0.179
# SUDS coherence: mean=0.218, std=0.134

# SUDS: VCVS vs Cortex: t=-0.0885, p=0.0002
# SUDS: VCVS vs coherence: t=0.0148, p=0.5139
# SUDS: Cortex vs VCVS: t=0.0885, p=0.0002
# SUDS: Cortex vs coherence: t=0.1034, p=0.0002
# SUDS: coherence vs VCVS: t=-0.0148, p=0.5319
# SUDS: coherence vs Cortex: t=-0.1034, p=0.0004

# relative:
# SUDS VCVS: mean=0.233, std=0.142
# SUDS Cortex: mean=0.322, std=0.179
# SUDS coherence: mean=0.218, std=0.134

# SUDS: VCVS vs Cortex: t=-0.1002, p=0.0002
# SUDS: VCVS vs coherence: t=0.0148, p=0.5203
# SUDS: Cortex vs VCVS: t=0.1002, p=0.0002
# SUDS: Cortex vs coherence: t=0.0341, p=0.3304
# SUDS: coherence vs VCVS: t=-0.0148, p=0.5141
# SUDS: coherence vs Cortex: t=-0.0341, p=0.3390



plt.subplot(1,5,5)
df_plt = df_all.query("YBOCS == True").groupby(["subject", "group", "feature"])["corr_abs"].max().reset_index()
sns.boxplot(data=df_plt, x="group", y="corr_abs", showmeans=True, showfliers=False, width=0.6, palette="viridis", order=["coherence", "VCVS", "Cortex"], linewidth=0.4, meanprops={"marker": "^", "markersize": 1.5})
sns.swarmplot(data=df_plt, x="group", y="corr_abs", color=".25", alpha=0.2, size=0.5, linewidth=0.4, order=["coherence", "VCVS", "Cortex"])
# set limit from 0 to 1
plt.ylim(0, 1)
plt.title("YBOCS", fontsize=5)
sns.despine()
df_plt = df_plt.dropna()
all_group_pvals = {}
for group_1 in ["VCVS", "Cortex", "coherence"]:
    for group_2 in ["VCVS", "Cortex", "coherence"]:
        if group_1 == group_2:
            continue
        if RELATIVE_:
            df_group1 = df_plt.query("group == @group_1")
            df_group2 = df_plt.query("group == @group_2")
            # get only the ones from both where subject and feature are the same
            if group_1 != "coherence" and group_2 != "coherence":
                df_group1 = df_group1.merge(df_group2[["subject", "feature"]], on=["subject", "feature"], how="inner")
                df_group2 = df_group2.merge(df_group1[["subject", "feature"]], on=["subject", "feature"], how="inner")
                corrs_1 = df_group1["corr_abs"]
                corrs_2 = df_group2["corr_abs"]
            else:
                # limit only to subject that are present in both groups
                subjects_1 = set(df_group1["subject"].unique())
                subjects_2 = set(df_group2["subject"].unique())
                subjects_common = subjects_1.intersection(subjects_2)
                df_group1 = df_group1[df_group1["subject"].isin(subjects_common)]
                df_group2 = df_group2[df_group2["subject"].isin(subjects_common)]
                corrs_1 = df_group1["corr_abs"]
                corrs_2 = df_group2["corr_abs"]
            if len(corrs_1) == 0 or len(corrs_2) == 0:
                #print(f"SUDS: {group_1} vs {group_2}: not enough data for relative comparison")
                continue
        else:
            corrs_1 = df_plt.query("group == @group_1")["corr_abs"]
            corrs_2 = df_plt.query("group == @group_2")["corr_abs"]
        res_ = stats.permutation_test(
            (corrs_1, corrs_2),
            statistic=lambda x, y: np.mean(x) - np.mean(y),
            vectorized=False,
            n_resamples=10000,
            alternative="two-sided"
        )
        t = res_.statistic
        p_val = res_.pvalue
        all_group_pvals[(group_1, group_2)] = p_val
        print(f"YBOCS: {group_1} vs {group_2}: t={t:.4f}, p={p_val:.4f}")

add_group_significance_bars(
    plt.gca(),
    ["coherence", "VCVS", "Cortex"],
    {
        ("coherence", "VCVS"): all_group_pvals.get(("coherence", "VCVS"), 1.0),
        ("coherence", "Cortex"): all_group_pvals.get(("coherence", "Cortex"), 1.0),
        ("VCVS", "Cortex"): all_group_pvals.get(("VCVS", "Cortex"), 1.0),
    }
)

# plot mean and std for each group in one line 
means = df_plt.groupby("group")["corr_abs"].mean()
stds = df_plt.groupby("group")["corr_abs"].std()
for group in ["VCVS", "Cortex", "coherence"]:
    print(f"YBOCS {group}: mean={means[group]:.3f}, std={stds[group]:.3f}")

for ax in plt.gcf().axes:
    set_compact_ticks(ax)

plt.tight_layout()
if RELATIVE_:
    plt.savefig("figures/Figure2/corr_group_comparison_relative_1.pdf")
else:
    plt.savefig("figures/Figure2/corr_group_comparison_1.pdf")
# PRE:
# YBOCS: VCVS vs Cortex: t=-0.1583, p=0.0002
# YBOCS: VCVS vs coherence: t=0.0912, p=0.0074
# YBOCS: Cortex vs VCVS: t=0.1583, p=0.0002
# YBOCS: Cortex vs coherence: t=0.2495, p=0.0002
# YBOCS: coherence vs VCVS: t=-0.0912, p=0.0066
# YBOCS: coherence vs Cortex: t=-0.2495, p=0.0002
# YBOCS VCVS: mean=0.506, std=0.229
# YBOCS Cortex: mean=0.664, std=0.222
# YBOCS coherence: mean=0.415, std=0.191

# NEW:
# YBOCS: VCVS vs Cortex: t=-0.0633, p=0.0112
# YBOCS: VCVS vs coherence: t=0.1682, p=0.0002
# YBOCS: Cortex vs VCVS: t=0.0633, p=0.0096
# YBOCS: Cortex vs coherence: t=0.2316, p=0.0002
# YBOCS: coherence vs VCVS: t=-0.1682, p=0.0002
# YBOCS: coherence vs Cortex: t=-0.2316, p=0.0002
# YBOCS VCVS: mean=0.441, std=0.229
# YBOCS Cortex: mean=0.505, std=0.238
# YBOCS coherence: mean=0.273, std=0.188


# relative:
# YBOCS VCVS vs Cortex: t=-0.2297, p=0.0002
# YBOCS VCVS vs coherence: t=0.0912, p=0.0054
# YBOCS Cortex vs VCVS: t=0.2297, p=0.0002
# YBOCS Cortex vs coherence: t=0.1769, p=0.0002
# YBOCS coherence vs VCVS: t=-0.0912, p=0.0092
# YBOCS coherence vs Cortex: t=-0.1769, p=0.0002
# YBOCS VCVS: mean=0.506, std=0.229
# YBOCS Cortex: mean=0.664, std=0.222
# YBOCS coherence: mean=0.415, std=0.191