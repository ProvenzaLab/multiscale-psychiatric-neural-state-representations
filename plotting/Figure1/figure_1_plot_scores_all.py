import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 5
plt.rcParams["lines.linewidth"] = 0.5


# Load all scores and identify the scores used in the original plot.
df_YBOCS = pd.read_csv("plotting/Figure1/source_data/all_ybocs_scores_from_sheet.csv")
df_YBOCS["date"] = pd.to_datetime(df_YBOCS["date"])
df_YBOCS["ybocs_score"] = pd.to_numeric(df_YBOCS["ybocs_score"], errors="coerce")

df_original = pd.read_csv(
    "plotting/Figure1/source_data/neural_features_ybocs_with_coh_1.csv",
    usecols=["subject", "date", "YBOCS II Total Score"],
)
df_original["date"] = pd.to_datetime(df_original["date"])
df_original["YBOCS II Total Score"] = pd.to_numeric(
    df_original["YBOCS II Total Score"], errors="coerce"
)

original_scores = df_original.dropna(subset=["YBOCS II Total Score"])
plotted_score_keys = set(
    zip(
        original_scores["subject"],
        original_scores["date"],
        original_scores["YBOCS II Total Score"],
    )
)

# Include original-plot scores that are absent from the full-score CSV.
original_scores_for_plot = original_scores.rename(
    columns={"YBOCS II Total Score": "ybocs_score"}
)[["subject", "date", "ybocs_score"]]
df_YBOCS = pd.concat(
    [df_YBOCS.drop(columns="is_in_original_plot", errors="ignore"), original_scores_for_plot],
    ignore_index=True,
).drop_duplicates(subset=["subject", "date", "ybocs_score"])
df_YBOCS["is_in_original_plot"] = [
    (subject, date, score) in plotted_score_keys
    for subject, date, score in zip(
        df_YBOCS["subject"], df_YBOCS["date"], df_YBOCS["ybocs_score"]
    )
]

df_SUDS = pd.read_csv("plotting/Figure1/source_data/neural_features_suds.csv")
df_SUDS["time"] = pd.to_datetime(df_SUDS["time"])
df_SUDS["subject"] = df_SUDS["subject"].replace(
    {
        4: "aDBS004",
        5: "aDBS005",
        7: "aDBS007",
        8: "aDBS008",
        9: "aDBS009",
        10: "aDBS010",
        11: "aDBS011",
        12: "aDBS012",
    }
)

fig, axes = plt.subplots(1, 8, figsize=(5, 1.1), sharey=False)

unique_subjects = df_YBOCS["subject"].unique()
color_viridis = sns.color_palette("viridis", len(unique_subjects))
subject_color_map = {sub: color_viridis[i] for i, sub in enumerate(unique_subjects)}

col_score = "ybocs_score"

for i, sub in enumerate(unique_subjects):
    ax1 = axes[i]
    ax2 = ax1.twinx()
    ax1.set_ylim(0, 60)

    df_sub_ybocs = df_YBOCS.loc[
        (df_YBOCS["subject"] == sub) & df_YBOCS[col_score].notna()
    ].sort_values("date").copy()
    df_sub_suds = df_SUDS.loc[df_SUDS["subject"] == sub].sort_values("time").copy()

    if df_sub_ybocs.empty and df_sub_suds.empty:
        ax1.set_title(f"Subject {sub}\n(no data)")
        ax1.axis("off")
        continue

    t_candidates = []
    if not df_sub_ybocs.empty:
        t_candidates.append(df_sub_ybocs["date"].min())
    if not df_sub_suds.empty:
        t_candidates.append(df_sub_suds["time"].min())
    t0 = min(t_candidates)

    if not df_sub_ybocs.empty:
        x_days_ybocs = (df_sub_ybocs["date"] - t0).dt.days.to_numpy()
        y_scores = df_sub_ybocs[col_score].to_numpy()
        ax1.plot(
            x_days_ybocs,
            y_scores,
            color=subject_color_map[sub],
            marker="o",
            linewidth=0.5,
            zorder=2,
            markersize=0.5,
        )

        original_mask = df_sub_ybocs["is_in_original_plot"].to_numpy()
        ax1.scatter(
            x_days_ybocs[original_mask],
            y_scores[original_mask],
            facecolors="none",
            edgecolors="black",
            linewidths=0.5,
            s=8,
            zorder=3,
        )

    if not df_sub_suds.empty:
        t_start = df_sub_suds["time"].min()
        t_end = df_sub_suds["time"].max()
        left_days = (t_start - t0).days
        width_days = max(1, (t_end - t_start).days)
        ax2.barh(
            width=width_days,
            left=left_days,
            y=0,
            color=subject_color_map[sub],
            alpha=0.35,
            zorder=0,
        )

    ax1.set_title(f"{sub}")
    ax2.set_yticks([])
    ax2.set_ylabel("")

    for spine in ("right", "top"):
        ax1.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)

    if i == 0:
        ax1.set_ylabel(col_score)
    else:
        ax1.set_ylabel("")
        ax2.set_ylabel("")
        ax1.tick_params(left=False)
        ax2.tick_params(left=False)
        ax1.set_yticklabels([])
        ax2.set_yticklabels([])
        ax1.spines["left"].set_visible(False)
        ax2.spines["left"].set_visible(False)

    ax1.tick_params(axis="x", length=2)
    ax1.tick_params(axis="y", length=2)
    ax1.set_yticks(np.arange(0, 60, 10))
    ax1.set_ylim(0, 50)


fig.suptitle("SUDS and All YBOCS Scores Over Time (per Subject)", fontsize=5)
plt.savefig("figures/Figure1/fig1_ybocs_progression_per_patient_all.pdf", bbox_inches="tight")
plt.show()