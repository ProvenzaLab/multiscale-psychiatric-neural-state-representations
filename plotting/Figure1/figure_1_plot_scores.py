import pandas as pd
import numpy as np
import seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt
# set font size to 5 
# set font Arial

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 5
# set line width to 0.5
plt.rcParams['lines.linewidth'] = 0.5


#df_YBOCS = pd.read_csv("plotting/Figure1/source_data/neural_features_ybocs.csv")
df_YBOCS = pd.read_csv("plotting/Figure1/source_data/neural_features_ybocs_with_coh_1.csv")
df_YBOCS["date"] = pd.to_datetime(df_YBOCS["date"])

# df_YBOCS.groupby("subject").count()["date"].mean()
# df_YBOCS.groupby("subject").count()["date"].std()
# 19.75 pm 2.71

# group by subject, and get the sum of the elapses days (last date - first date) for each subject
df_YBOCS.groupby("subject").apply(lambda x: (x["date"].max() - x["date"].min()).days).sum()


df_SUDS = pd.read_csv("plotting/Figure1/source_data/neural_features_suds.csv")
df_SUDS["time"] = pd.to_datetime(df_SUDS["time"])
df_SUDS["subject"] = df_SUDS["subject"].replace({
    4: "aDBS004", 5: "aDBS005", 7: "aDBS007", 8: "aDBS008", 9: "aDBS009", 
    10: "aDBS010", 11: "aDBS011", 12: "aDBS012"
})

fig, axes = plt.subplots(1, 8, figsize=(5, 1.1), sharey=False)

unique_subjects = df_YBOCS["subject"].unique()
color_viridis = sns.color_palette("viridis", len(unique_subjects))
subject_color_map = {sub: color_viridis[i] for i, sub in enumerate(unique_subjects)}

col_score = "YBOCS II Total Score"
col_SUDS  = "score"

for i, sub in enumerate(unique_subjects):
    ax1 = axes[i]
    ax2 = ax1.twinx()  # SUDS overlay (bars), but same x-scale
    ax1.set_ylim(0, 60)  # YBOCS range

    # ----- slice data -----
    df_sub_ybocs = df_YBOCS.loc[df_YBOCS["subject"] == sub].sort_values("date").copy()
    df_sub_suds  = df_SUDS.loc[df_SUDS["subject"] == sub].sort_values("time").copy()

    # skip empty panels safely
    if df_sub_ybocs.empty and df_sub_suds.empty:
        ax1.set_title(f"Subject {sub}\n(no data)")
        ax1.axis("off")
        continue

    # ----- define common time zero (per subject) -----
    # choose earliest timestamp across both series
    t_candidates = []
    if not df_sub_ybocs.empty: t_candidates.append(df_sub_ybocs["date"].min())
    if not df_sub_suds.empty:  t_candidates.append(df_sub_suds["time"].min())
    t0 = min(t_candidates)
    y_trace = df_sub_ybocs[col_score].values
    # remove nans
    y_nan_idx = np.isnan(y_trace)
    y_trace = y_trace[~y_nan_idx]
    # ----- X for YBOCS: days since t0 -----
    if not df_sub_ybocs.empty:
        x_days_ybocs = (df_sub_ybocs["date"] - t0).dt.days
        x_days_ybocs = x_days_ybocs.values[~y_nan_idx]  # align with y_trace after nan removal

        ax1.plot(
            x_days_ybocs, y_trace,
            color=subject_color_map[sub], marker="o",
            linewidth=0.5, zorder=2, markersize=0.5
        )

    # ----- X for SUDS coverage: bar from first to last SUDS timestamp -----
    if not df_sub_suds.empty:
        t_start = df_sub_suds["time"].min()
        t_end   = df_sub_suds["time"].max()
        left_days = (t_start - t0).days
        width_days = max(1, (t_end - t_start).days)  # at least 1 day to be visible

        # a thin horizontal bar at y=0 (twin y-axis)
        ax2.barh(
            width=width_days, left=left_days, y=0,
            color=subject_color_map[sub], alpha=0.35, zorder=0
        )

        # optionally, show scattered SUDS points over time (commented)
        # x_days_suds = (df_sub_suds["time"] - t0).dt.days
        # ax2.scatter(x_days_suds, np.zeros_like(x_days_suds), s=12,
        #             color=subject_color_map[sub], alpha=0.6, zorder=1)

        # keep SUDS y-range tight


    # ----- cosmetics -----
    ax1.set_title(f"{sub}")
    # remove cluttered x ticks/labels inside each small panel
    #ax1.set_xticks([])
    #ax1.set_xlabel("")
    ax2.set_yticks([])
    ax2.set_ylabel("")

    # borders
    for spine in ("right", "top"):
        ax1.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)
        

    # only first subplot gets y-labels
    if i == 0:
        ax1.set_ylabel(col_score)
    else:
        ax1.set_ylabel("")
        ax2.set_ylabel("")
        # remove axis ticks
        ax1.tick_params(left=False)
        ax2.tick_params(left=False)
        # remove labels 
        ax1.set_yticklabels([])
        ax2.set_yticklabels([])
        ax1.spines["left"].set_visible(False)
        ax2.spines["left"].set_visible(False)
    # reduce length of x-ticks and y-ticks
    ax1.tick_params(axis='x', length=2)
    ax1.tick_params(axis='y', length=2)

    # x ticks every 10 values
    ax1.set_yticks(np.arange(0, 60, 10))
    # set limit to 50 
    ax1.set_ylim(0, 50)


fig.suptitle("SUDS and YBOCS Scores Over Time (per Subject)", fontsize=5)
plt.savefig("figures/Figure1/fig1_ybocs_progression_per_patient_1.pdf", bbox_inches="tight")
plt.show()
