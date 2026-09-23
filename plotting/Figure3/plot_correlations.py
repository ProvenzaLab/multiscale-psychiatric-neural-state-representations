import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.manifold import TSNE
matplotlib.rcParams.update({'font.size': 5, 'font.family': 'Arial'})

def remove_axes_labels_ticks():
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.xlabel("")
    plt.ylabel("")
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)


df_corr = pd.read_csv("plotting/Figure3/source_data/behav_corrs_1.csv")
df_corr = df_corr.query("name.str.startswith('Dim') == False", engine="python")

df_ybocs = pd.read_csv("plotting/Figure1/source_data/ybocs_audio_neural_features_combined_1.csv")
df_suds = pd.read_csv("plotting/Figure3/source_data/suds_neural_audio_fau_combined.csv")

subjects = df_corr["subject"].unique()
l_audio = list(df_corr.query("behav_type == 'Audio' and name.str.startswith('AU_') == False and name.str.startswith('Dim') == False", engine="python")["name"].unique())
l_video = list(df_corr.query("behav_type == 'FAU' and name.str.startswith('AU_') == True", engine="python")["name"].unique())

COMPUTE_SECOND_ORDER_CORRS = True
if COMPUTE_SECOND_ORDER_CORRS:
    df_so_data = df_corr.query("SHUFFLE == False")
    l_so = []
    for behav_type in df_so_data["behav_type"].unique():
        for subj in subjects:
            df_sub_suds = df_so_data.query("subject == @subj and behav_type == @behav_type and RS == False")
            df_sub_ybocs = df_so_data.query("subject == @subj and behav_type == @behav_type and RS == True")
            merged = pd.merge(df_sub_suds, df_sub_ybocs, on="name", suffixes=("_suds", "_ybocs"))
            if len(merged) < 2:
                continue
            r, p = pearsonr(merged["corr_suds"], merged["corr_ybocs"])
            l_so.append({
                "subject": subj,
                "corr_suds_ybocs": r,
                "p_value": p,
                "behav_type": behav_type
            })
    df_so = pd.DataFrame(l_so)
    df_so.to_csv("plotting/Figure3/source_data/second_order_behav_corrs_1.csv", index=False)
    # 
else:
    df_so = pd.read_csv("plotting/Figure3/source_data/second_order_behav_corrs_1.csv")

# TSNE 

df_tsne_l = []
for RS in [True, False]:
    for behav_dim in ["FAU", "Audio"]:
        if RS:
            # query based on behav_dim
            df_tsne_ = df_ybocs.copy()
        else:
            df_tsne_ = df_suds.copy()
        if behav_dim == "FAU":
            df_tsne = df_tsne_[l_video]
        else:
            df_tsne = df_tsne_[l_audio]
            
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)

        df_zs_mean = df_tsne.mean(axis=0)
        df_zs_std = df_tsne.std(axis=0)
        df_zs = (df_tsne - df_zs_mean) / df_zs_std
        df_zs["subject"] = df_tsne_["subject"]

        df_zs = df_zs.dropna()
        tsne_results = tsne.fit_transform(df_zs.drop(columns=["subject"], errors="ignore"))
        df_tsne_out = pd.DataFrame(tsne_results, columns=["tsne-2d-one", "tsne-2d-two"])
        df_tsne_out["subject"] = df_zs["subject"].values
        df_tsne_out["RS"] = RS
        df_tsne_out["behav_type"] = behav_dim
        df_tsne_l.append(df_tsne_out)
df_tsne = pd.concat(df_tsne_l, axis=0, ignore_index=True)


ALPHA = 0.64
hue_colors = ["#3C6682", "#FF0000"]
palette = sns.color_palette(hue_colors)

plt.figure(figsize=(7, 3))
plt.subplot(2, 5, 3)
sns.histplot(data=df_corr.query("RS == False and behav_type == 'FAU'"), x="p", bins=30, hue="SHUFFLE", alpha=ALPHA, palette=palette)
plt.ylim(0, 60)
plt.xlabel("p-value")
plt.ylabel("Bin count [a.u.]")
plt.title(f"Momentary distress\n(SUDS)", fontsize=5)
sns.despine()
plt.subplot(2, 5, 4)
sns.histplot(data=df_corr.query("RS == True and behav_type == 'FAU'"), x="p", bins=30, hue="SHUFFLE", alpha=ALPHA, palette=palette, legend=False)
sns.despine()
plt.ylim(0, 60)
plt.xlabel("p-value")
plt.title(f"OCD severity\nYBOCS-II", fontsize=5)

plt.subplot(2, 5, 8)
sns.histplot(data=df_corr.query("RS == False and behav_type == 'Audio'"), x="p", bins=30, hue="SHUFFLE", alpha=ALPHA, palette=palette, legend=True)
plt.xlabel("p-value")
plt.title(f"Momentary distress\n(SUDS)", fontsize=5)
sns.despine()
plt.ylabel("Bin count [a.u.]")
plt.title(f"Momentary distress\n(SUDS)", fontsize=5)
sns.despine()
plt.subplot(2, 5, 9)
sns.histplot(data=df_corr.query("RS == True and behav_type == 'Audio'"), x="p", bins=30, hue="SHUFFLE", alpha=ALPHA, palette=palette, legend=False)
sns.despine()
plt.xlabel("p-value")
plt.title(f"OCD severity\nYBOCS-II", fontsize=5)

plt.subplot(2, 5, 5)
df_so_fau = df_so.query("behav_type == 'FAU'").sort_values(by="subject").reset_index(drop=True)
sns.barplot(data=df_so_fau, x="subject", y="corr_suds_ybocs")
for i, row in df_so_fau.iterrows():
    if row["p_value"] < 0.05:
        plt.text(i, row["corr_suds_ybocs"] + 0.02, "*", ha='center', va='bottom', fontsize=20)
plt.xlabel("Subject")
mean_corr = df_so_fau["corr_suds_ybocs"].mean()
std_corr = df_so_fau["corr_suds_ybocs"].std()
plt.title(f"SUDS vs YBOCS FAU correlation\nMean r={mean_corr:.2f} ± {std_corr:.2f}")
sns.despine()

plt.subplot(2, 5, 10)
df_so_audio = df_so.query("behav_type == 'Audio'").sort_values(by="subject").reset_index(drop=True)
sns.barplot(data=df_so_audio, x="subject", y="corr_suds_ybocs")
for i, row in df_so_audio.iterrows():
    if row["p_value"] < 0.05:
        plt.text(i, row["corr_suds_ybocs"] + 0.02, "*", ha='center', va='bottom', fontsize=20)
plt.xlabel("Subject")
mean_corr = df_so_audio["corr_suds_ybocs"].mean()
std_corr = df_so_audio["corr_suds_ybocs"].std()
plt.title(f"SUDS vs YBOCS Audio correlation\nMean r={mean_corr:.2f} ± {std_corr:.2f}")
sns.despine()

# reduce line width of the axes
for axis in plt.gcf().axes:
    axis.spines['top'].set_linewidth(0.5)
    axis.spines['right'].set_linewidth(0.5)
    axis.spines['bottom'].set_linewidth(0.5)
    axis.spines['left'].set_linewidth(0.5)
# make the ticks smaller
for axis in plt.gcf().axes:
    axis.tick_params(axis='both', which='major', labelsize=5, width=0.5, length=1)
    axis.tick_params(axis='both', which='minor', labelsize=5, width=0.5, length=1)


plt.subplot(2, 5, 1)

df_tsne_plt = df_tsne.query("RS == False and behav_type == 'FAU'")
plt.scatter(df_tsne_plt["tsne-2d-one"],
            df_tsne_plt["tsne-2d-two"], c=pd.factorize(df_tsne_plt["subject"])[0],
            cmap='Accent', alpha=0.7, s=2.5)
plt.title("SUDS")
remove_axes_labels_ticks()
plt.xlabel("TSNE Dim. 1")
plt.ylabel("TSNE Dim. 2")

plt.subplot(2, 5, 2)

df_tsne_plt = df_tsne.query("RS == True and behav_type == 'FAU'")
plt.scatter(df_tsne_plt["tsne-2d-one"],
            df_tsne_plt["tsne-2d-two"], c=pd.factorize(df_tsne_plt["subject"])[0],
            cmap='Accent', alpha=0.7, s=10)

cbar = plt.colorbar(ticks=range(len(df_tsne_plt["subject"].unique())))
cbar.ax.set_yticklabels(df_tsne_plt["subject"].unique())
cbar.ax.tick_params(labelsize=5)
cbar.ax.tick_params(length=1)
cbar.outline.set_visible(False)

plt.title("YBOCS-II")
remove_axes_labels_ticks()

plt.subplot(2, 5, 6)

df_tsne_plt = df_tsne.query("RS == False and behav_type == 'Audio'")
plt.scatter(df_tsne_plt["tsne-2d-one"],
            df_tsne_plt["tsne-2d-two"], c=pd.factorize(df_tsne_plt["subject"])[0],
            cmap='Accent', alpha=0.7, s=2.5)
remove_axes_labels_ticks()
plt.title("SUDS")

plt.subplot(2, 5, 7)

df_tsne_plt = df_tsne.query("RS == True and behav_type == 'Audio'")
plt.scatter(df_tsne_plt["tsne-2d-one"],
            df_tsne_plt["tsne-2d-two"], c=pd.factorize(df_tsne_plt["subject"])[0],
            cmap='Accent', alpha=0.7, s=10)

cbar = plt.colorbar(ticks=range(len(df_tsne_plt["subject"].unique())))
cbar.ax.set_yticklabels(df_tsne_plt["subject"].unique())
cbar.ax.tick_params(labelsize=5)
cbar.ax.tick_params(length=1)
cbar.outline.set_visible(False)
remove_axes_labels_ticks()
plt.title("YBOCS-II")
plt.savefig("figures/Figure3/behavior_tsne_pvaldist_secondorderbehav_1.pdf", bbox_inches='tight', dpi=300)
