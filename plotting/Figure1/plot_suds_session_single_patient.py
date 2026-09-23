import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

df_SUDS = pd.read_csv("plotting/Figure1/source_data/neural_features_suds.csv")
df_SUDS["time"] = pd.to_datetime(df_SUDS["time"])

df_sub = df_SUDS.loc[df_SUDS["subject"] == 5.0]
df_sub["date"] = df_sub["time"].dt.date
unique_dates = df_sub["date"].unique()


plt.figure(figsize=(6, 3))
for idx in range(4):
    plt.subplot(1, 4, idx + 1)
    date = unique_dates[idx]
    df_day = df_sub.loc[df_sub["date"] == date]
    time_ = df_day["time"]- df_day["time"].min()
    time_minutes = time_.dt.total_seconds() / 60.0
    plt.plot(time_minutes, df_day["score"], marker='o')
    plt.title(date)
    # turn off right and top border
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    if idx == 0:
        plt.ylabel("SUDS Score")
    if idx != 0:
        plt.yticks([])
        plt.gca().spines['left'].set_visible(False)
    plt.ylim(4, 9)
plt.savefig("figures/Figure1/fig1_suds_session_example.pdf", bbox_inches='tight')
plt.show()

