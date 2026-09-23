import pandas as pd
from scipy import stats
import numpy as np
import pickle
import matplotlib
import seaborn as sns

# set arial font and compact figure styling
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 5
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

PATH_SUDS = "/Users/Timon/Documents/Houston/OCD_RCS/OCD_RCS/neural_audio_fau_combined.csv"
PATH_RS = "/Users/Timon/Documents/Houston/YBOCS/plotting/Figure1/source_data/ybocs_audio_neural_features_combined_1.csv"


def get_df_features(region: str, feature: str, READ_RS: bool = False):
    if READ_RS:
        PATH_DATA = PATH_RS
    else:
        PATH_DATA = PATH_SUDS
    df_features = pd.read_csv(PATH_DATA)
    if READ_RS is False:
        df_features["time"] = pd.to_datetime(df_features["time"])
    else:   
        df_features["date"] = pd.to_datetime(df_features["date"])
        #df_features = df_features.drop(columns=["subject"])
        #df_features = df_features.rename(columns={"sub": "subject"})
        # if columns start with FAU_, rename the FAU_
        df_features = df_features.rename(columns={c: c[4:] for c in df_features.columns if c.startswith("FAU_")})
        # remove cols that contain "corr"
        df_features = df_features[[c for c in df_features.columns if "corr" not in c and "psd" not in c]]

    if feature == "fft_only":
        if region != "all":
            df_features = df_features[[c for c in df_features.columns if c.startswith(region) and "fft" in c and "fft_psd" not in c or c == "subject" or c == "score_fau" or c == "score" or c == "score_normed" or c == "time" or c == 'YBOCS II Total Score' or c.startswith("AU") or c in l_audio_features]]
        else:
            df_features = df_features[[c for c in df_features.columns if "fft" in c and "fft_psd" not in c or c == "subject" or c == "score_fau" or c == "score" or c == 'YBOCS II Total Score' or c == "score_normed" or c == "time" or c.startswith("AU") or c in l_audio_features]]

    elif feature != "all":
        if region != "all":
            df_features = df_features[[c for c in df_features.columns if (c.startswith(region) and feature in c) or c in ["subject", "score", "score_normed", "time", 'YBOCS II Total Score'] or c.startswith("AU") or c in l_audio_features]]
        else:
            df_features = df_features[[c for c in df_features.columns if feature in c or c == 'YBOCS II Total Score' or c == "subject" or c == "score" or c == "score_normed" or c == "time" or c.startswith("AU") or c in l_audio_features]]
    elif feature == "all" and region != "all":
        df_features = df_features[[c for c in df_features.columns if c.startswith(region) or c == "subject" or c == 'YBOCS II Total Score' or c == "score" or c == "score_feat" or c == "score_normed" or c == "time" or c == "date" or c.startswith("AU") or c in l_audio_features]]
    #else:
        # select all regions
    #    df_features = df_features[[c for c in df_features.columns if c == "subject" or c == "score" or c == "score_normed" or c == "time" or c.startswith("AU") or c in l_audio_features]]
    # if region starts with C_, remove subjects 4, 5, 7
    subs = df_features["subject"].unique()
    if region.startswith("C"):
        subs = [s for s in subs if s not in [4, 5, 7]]
        df_features = df_features[df_features["subject"].isin(subs)]
    if region != "all":
        # remove columns that have coherence or corr in their name
        df_features = df_features[[c for c in df_features.columns if not "coherence" in c and not "corr" in c]]
    return df_features, subs



subjects = [4, 5, 7, 8, 9, 10, 11, 12]

df = pd.read_csv(f"plotting/Figure4/source_data/ybocs_decoding_results_neural_audio_fau_49.csv")

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

def get_tr_pr(region="all", RS=False, subject=5, ZS=True):

    res_df = results[(region, not RS)]['df_res']
    res_df["condition"] = res_df.apply(set_condition, axis=1)
    ind_sub_suds_only = res_df[(res_df["condition"] == "SUDS only") & (res_df["subject"] == subject)].query("SHUFFLE == False").index[0]
    ind_sub_suds_au_audio = res_df[(res_df["condition"] == "SUDS+AU+Audio") & (res_df["subject"] == subject)].query("SHUFFLE == False").index[0]

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
if RS is False:
    with open(f"plotting/Figure3/source_data/suds_decoding_results_neural_audio_fau.pkl", "rb") as f:
        results = pickle.load(f)

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
sns.regplot(data=df_plot, x="True SUDS", y="Predicted SUDS only", label="SUDS only", color="#3C6682", scatter_kws={"s": 6}, line_kws={"linewidth": 0.6})
sns.regplot(data=df_plot, x="True SUDS", y="Predicted SUDS+AU+Audio", label="SUDS+AU+Audio", color="#45A778", scatter_kws={"s": 6}, line_kws={"linewidth": 0.6})
ax = plt.gca()
ax.tick_params(axis="x", which="major", length=1.0, width=0.6, labelsize=5)
ax.tick_params(axis="y", which="major", length=1.0, width=0.6, labelsize=5)
for spine in ax.spines.values():
    spine.set_linewidth(0.6)
sns.despine()
plt.savefig(f"figures/Figure4/exemplar_subject_12_suds_1.pdf")

plt.figure(figsize=(2, 1.25))

df_rs, subs_rs = get_df_features("all", "all", READ_RS=True)
subject = 12
X_sub = df_rs.query("subject == @subject")
dates_sub = X_sub["date"]

RS = True
if RS:
    with open(f"plotting/Figure4/source_data/ybocs_decoding_results_neural_audio_fau_49.pkl", "rb") as f:
        results = pickle.load(f)

y_tr_suds_only_zs, y_tr_suds_au_audio_zs, y_pr_suds_only_zs, y_pr_suds_au_audio_zs = \
            get_tr_pr(region="all", RS=RS, subject=subject, ZS=True)

plt.subplot(1, 2, 1)
plt.title(f"Subject {subject}", fontsize=5)
plt.plot(y_tr_suds_only_zs, label="True SUDS", color="black", linestyle="-", marker="o", linewidth=0.6, markersize=1)
#plt.plot(y_pr_suds_only_zs, label="Predicted SUDS only", color="#3C6682", linestyle="-", marker="o")
plt.plot(y_pr_suds_au_audio_zs, label="Predicted SUDS+AU+Audio", color="#45A778", linestyle="-", marker="o", linewidth=0.6, markersize=1)
plt.ylabel("Y-BOCS z-score [a.u.]", fontsize=5)
plt.xlabel("Time [a.u.]", fontsize=5)
ax = plt.gca()
ax.tick_params(axis="x", which="major", length=1.0, width=0.6, labelsize=5)
ax.tick_params(axis="y", which="major", length=1.0, width=0.6, labelsize=5)
for spine in ax.spines.values():
    spine.set_linewidth(0.6)
sns.despine()

plt.subplot(1, 2, 2)
df_plot = pd.DataFrame({
    "True SUDS": y_tr_suds_only_zs[:-1],
    "Predicted SUDS only": y_pr_suds_only_zs[:-1],
    "Predicted SUDS+AU+Audio": y_pr_suds_au_audio_zs,
})
#sns.regplot(data=df_plot, x="True SUDS", y="Predicted SUDS only", label="SUDS only", color="#3C6682")
sns.regplot(data=df_plot, x="True SUDS", y="Predicted SUDS+AU+Audio", label="SUDS+AU+Audio", color="#45A778", scatter_kws={"s": 6}, line_kws={"linewidth": 0.6})
ax = plt.gca()
ax.tick_params(axis="x", which="major", length=1.0, width=0.6, labelsize=5)
ax.tick_params(axis="y", which="major", length=1.0, width=0.6, labelsize=5)
for spine in ax.spines.values():
    spine.set_linewidth(0.6)
plt.savefig(f"figures/Figure4/exemplar_subject_{subject}_rs.pdf")

