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
from pathlib import Path

df_ybocs = pd.read_csv("plotting/Figure2/source_data/feature_region_subject_correlations_ybocs_1.csv")
df_ybocs["YBOCS"] = True

df_suds = pd.read_csv("plotting/Figure2/source_data/feature_region_subject_correlations_suds.csv")
df_suds["YBOCS"] = False

df_all = pd.concat([df_suds, df_ybocs], ignore_index=True)
df_all["corr_abs"] = df_all["correlation"].abs()
df_all["hem"] = df_all["region"].apply(lambda x: x.split("_")[1])

REGION_GROUPS = [
    ("Subcortex", "SC_L", "SC_R"),
    ("Cortex Medial", "C_L_1", "C_R_1"),
    ("Cortex Lateral", "C_L_2", "C_R_2"),
]
GROUP_ORDER = [label for label, _, _ in REGION_GROUPS]


def set_compact_ticks(ax):
    ax.tick_params(axis="x", which="major", length=1.0, width=0.5)
    ax.tick_params(axis="y", which="major", length=1.0, width=0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)


def prepare_condition_df(df_condition):
    frames = []
    for group_label, left_region, right_region in REGION_GROUPS:
        df_grp = df_condition[df_condition["region"].isin([left_region, right_region])].copy()
        df_grp = df_grp.groupby(["subject", "feature", "hem"], as_index=False)["corr_abs"].max()
        df_grp["group"] = group_label
        frames.append(df_grp)
    return pd.concat(frames, ignore_index=True)


def compute_group_pvalues(df_plt):
    p_values = {}
    for group_label in GROUP_ORDER:
        wide_ = (
            df_plt[df_plt["group"] == group_label]
            .pivot(index=["subject", "feature"], columns="hem", values="corr_abs")
            .dropna(subset=["L", "R"])
        )
        diffs_ = wide_["L"] - wide_["R"]
        res_ = stats.permutation_test(
            (diffs_,),
            statistic=lambda x: np.mean(x),
            permutation_type="samples",
            n_resamples=10000,
            alternative="two-sided",
        )
        p_values[group_label] = res_.pvalue
    return p_values


def summarize_and_plot(ax, df_condition, condition_label):
    df_plt = prepare_condition_df(df_condition)
    p_values = compute_group_pvalues(df_plt)

    sns.boxplot(
        data=df_plt,
        x="group",
        y="corr_abs",
        hue="hem",
        order=GROUP_ORDER,
        hue_order=["L", "R"],
        showmeans=True,
        showfliers=False,
        width=0.6,
        palette="viridis",
        linewidth=0.4,
        meanprops={"marker": "^", "markersize": 1.5},
        ax=ax,
    )
    # sns.swarmplot(
    #     data=df_plt,
    #     x="group",
    #     y="corr_abs",
    #     hue="hem",
    #     order=GROUP_ORDER,
    #     hue_order=["L", "R"],
    #     dodge=True,
    #     color=".25",
    #     alpha=0.2,
    #     size=0.5,
    #     linewidth=0.4,
    #     ax=ax,
    #     legend=False,
    # )

    for i, group_label in enumerate(GROUP_ORDER):
        p_val = p_values[group_label]
        y_max_grp = df_plt.loc[df_plt["group"] == group_label, "corr_abs"].max()
        label = "*" if p_val < 0.05 else "n.s."
        ax.text(i, y_max_grp + 0.03, label, ha="center", va="bottom", fontsize=5)
        print(f"{condition_label} {group_label}: p={p_val:.4f}")

    ax.set_title(condition_label, fontsize=6)
    ax.set_ylabel("Pearson correlation (abs)", fontsize=5)
    ax.set_xlabel("")
    ax.set_ylim(-0.05, 1.1)
    sns.despine(ax=ax)
    set_compact_ticks(ax)
    ax.legend(title="", fontsize=4, frameon=False, loc="upper right", handletextpad=0.3)


Path("figures/Figure2").mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(3.0, 1.8), sharey=True)

for col, (condition_label, is_ybocs) in enumerate([("SUDS", False), ("YBOCS", True)]):
    ax = axes[col]
    df_condition = df_all[df_all["YBOCS"] == is_ybocs]
    summarize_and_plot(ax, df_condition, condition_label)
    if col != 0:
        ax.set_ylabel("")

plt.tight_layout()
plt.savefig("figures/Figure2/corr_group_lat_met.pdf")

# SUDS Subcortex: p=0.0002
# SUDS Cortex Medial: p=0.0010
# SUDS Cortex Lateral: p=0.5107
# YBOCS Subcortex: p=0.0288
# YBOCS Cortex Medial: p=0.3306
# YBOCS Cortex Lateral: p=0.0004

