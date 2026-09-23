from feat.plotting import plot_face
from feat.plotting import animate_face
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

# /Users/Timon/Documents/Houston/OCD_RCS/OCD_RCS/plot_redo_FAUs.py

# REPO to get FAU's: /Users/Timon/Documents/Houston/video_features/extracting_FAUs/

muscle_names = [
    "bucc_l",
    "bucc_r",
    "masseter_l",
    "masseter_r",
    "temporalis_l",
    "temporalis_r",
    "dep_lab_inf_l",
    "dep_lab_inf_r",
    "dep_ang_or_l",
    "dep_ang_or_r",
    "mentalis_l",
    "mentalis_r",
    "risorius_l",
    "risorius_r",
    "frontalis_l",
    "frontalis_inner_l",
    "frontalis_r",
    "frontalis_inner_r",
    "cor_sup_r",
    "orb_oc_l_outer",
    "orb_oc_r_outer",
    "lev_lab_sup_l",
    "lev_lab_sup_r",
    "lev_lab_sup_an_l",
    "lev_lab_sup_an_r",
    "zyg_maj_l",
    "zyg_maj_r",
    "orb_oc_l",
    "orb_oc_r",
    "orb_oc_l",
    "orb_oc_l_inner",
    "orb_oc_r_inner",
    "orb_oris_l",
    "orb_oris_u",
    "cor_sup_l",
    "pars_palp_l",
    "pars_palp_r",
    "masseter_l_rel",
    "masseter_r_rel",
    "temporalis_l_rel",
    "temporalis_r_rel",
]

def get_heat(muscle, au, log):
    """Function to create heatmap from au vector

    Args:
        muscle (string): string representation of a muscle
        au (list): vector of action units
        log (boolean): whether the action unit values are on a log scale


    Returns:
        color of muscle according to its au value
    """
    q = sns.color_palette("coolwarm", 101) #bwr 
    
    unit = 0
    aus = {
        "masseter_l": 15,
        "masseter_r": 15,
        "temporalis_l": 15,
        "temporalis_r": 15,
        "dep_lab_inf_l": 14,
        "dep_lab_inf_r": 14,
        "dep_ang_or_l": 10,
        "dep_ang_or_r": 10,
        "mentalis_l": 11,
        "mentalis_r": 11,
        "risorius_l": 12,
        "risorius_r": 12,
        "frontalis_l": 1,
        "frontalis_r": 1,
        "frontalis_inner_l": 0,
        "frontalis_inner_r": 0,
        "cor_sup_l": 2,
        "cor_sup_r": 2,
        "lev_lab_sup_l": 7,
        "lev_lab_sup_r": 7,
        "lev_lab_sup_an_l": 6,
        "lev_lab_sup_an_r": 6,
        "zyg_maj_l": 8,
        "zyg_maj_r": 8,
        "bucc_l": 9,
        "bucc_r": 9,
        "orb_oc_l_outer": 4,
        "orb_oc_r_outer": 4,
        "orb_oc_l": 5,
        "orb_oc_r": 5,
        "orb_oc_l_inner": 16,
        "orb_oc_r_inner": 16,
        "orb_oris_l": 13,
        "orb_oris_u": 13,
        "pars_palp_l": 19,
        "pars_palp_r": 19,
        "masseter_l_rel": 17,
        "masseter_r_rel": 17,
        "temporalis_l_rel": 17,
        "temporalis_r_rel": 17,
    }
    if muscle in aus:
        unit = aus[muscle]
    if log:
        num = int(100 * (1.0 / (1 + 10.0 ** -(au[unit]))))
    else:
        num = int(au[unit])
    # set alpha (opacity)
    
    #alpha = au[unit] / 100
    
    # color = colors.to_hex(q[num])
    # return str(color)
    color = colors.to_rgba(q[num]) #, alpha=alpha)
    return color

AU_labels = [1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 17, 20, 23, 24, 25, 26, 28, 43]

PLOT_EXAMPLE = True

if PLOT_EXAMPLE:
    fau_npy_example = np.load("plotting/Figure3/source_data/au_predictions_example_image.npy")
    df_fau_corrs = pd.read_csv("plotting/Figure3/source_data/suds_fau_corrs.csv")
    df_fau_corrs["AU_label"] = df_fau_corrs["AU"].apply(lambda x: f"{x[3:]}")
    vals = []
    for au in AU_labels:
        au_str = str(au)
        if au_str in df_fau_corrs["AU_label"].unique():
            au_idx = df_fau_corrs["AU_label"].unique().tolist().index(au_str)
            vals.append(fau_npy_example[au_idx])
        else:
            vals.append(0.0)
            
    vals = np.array(vals)
    #fau_range = (vals * 100 + 100) / 2
    # die activations gehen von 0 bis ca 5
    # scale to 0-100
    vals = (vals / 1.5* 100).astype(int)
    muscles = {}
    for muscle in muscle_names:
        muscles[muscle] = get_heat(muscle, vals, log=False)

    plot_face(
        au=vals,
        muscles=muscles,
        #title=f"SUDS corr",
        feature_range=(0, 2)
    )
    plt.savefig("figures/Figure3/FAU_example.pdf")


SUDS_CORRS_ = False
df_corr = pd.read_csv("plotting/Figure3/source_data/behav_corrs.csv")
if SUDS_CORRS_:
    #df_fau_corrs = pd.read_csv("plotting/Figure3/source_data/suds_fau_corrs.csv")
    #df_fau_corrs["AU_label"] = df_fau_corrs["AU"].apply(lambda x: f"{x[3:]}")
    df_fau_corrs = df_corr.query("RS == False and behav_type == 'FAU'").copy()
else:
    df_fau_corrs = df_corr.query("RS == True and behav_type == 'FAU'").copy()
subjects = df_fau_corrs["subject"].unique().tolist()

def face_inputs_for_subject(df_all, subject, AU_labels, muscle_names):
    """Build (faus_faus, muscles, fau_range) for one subject and sign."""
    df_sub = df_all[df_all["subject"] == subject]
    # if sign == "positive":
    #     df_sign = df_sub[df_sub["corr"] > 0]
    # else:
    #     df_sign = df_sub[df_sub["corr"] < 0]

    # collect corr values in AU_labels order (0 if AU missing)
    vals = []
    for au in AU_labels:
        au_str = str(au)
        if au_str in df_sub["AU_label"].values:
            vals.append(df_sub.loc[df_sub["AU_label"] == au_str, "corr"].values[0])
        else:
            vals.append(0.0)

    vals = np.array(vals)
    #faus_faus = vals * 15  # your scaling for plot_face(au=...)
    # range mapping you used (to 0..100-ish, centered 50)
    fau_range = (vals * 100 + 100) / 2

    muscles = {}
    for muscle in muscle_names:
        muscles[muscle] = get_heat(muscle, fau_range, log=False)

    return vals, muscles

#rows = ["positive", "negative"]
ncols = len(subjects) + 1  # subjects + average column
fig, axes = plt.subplots(
    nrows=1, ncols=ncols,
    figsize=(12, 2.5),
    constrained_layout=True
)

for c, subj in enumerate(subjects):
    #for r, sign in enumerate(rows):
    ax = axes[c]
    faus_faus, muscles = face_inputs_for_subject(
        df_fau_corrs, subj, AU_labels, muscle_names
    )
    try:
        plot_face(
            au=faus_faus,
            muscles=muscles,
            #title=f"SUDS corr",
            ax=ax,
            feature_range=(0, 2.5)
        )
    except TypeError:
        plt.sca(ax)
        plot_face(au=faus_faus, muscles=muscles, #title=f"SUDS corr"
        )

    #if r == 0:
    ax.set_title(f"Subject {subj}", fontsize=12, pad=6)
    #if c == 0:
    #    ax.set_ylabel(sign.capitalize(), fontsize=12)

    # clean up
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# --- average column ---
#for r, sign in enumerate(rows):
ax = axes[-1]  # last column
# collect all subjects for this row
faus_all = []
for subj in subjects:
    faus_faus, muscles = face_inputs_for_subject(
        df_fau_corrs, subj, AU_labels, muscle_names
    )
    faus_all.append(faus_faus)

mean_faus = np.mean(faus_all, axis=0)
fau_range = (mean_faus * 100 + 100) / 2
muscles = {m: get_heat(m, fau_range, log=False) for m in muscle_names}

try:
    plot_face(
        au=mean_faus,
        muscles=muscles,
        title="Average",
        ax=ax,
        feature_range=(0, 2.5) # 2.5
    )
except TypeError:
    plt.sca(ax)
    plot_face(au=mean_faus, muscles=muscles, title="Average")

#if r == 0:
ax.set_title("Average", fontsize=12, pad=6)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

if SUDS_CORRS_:
    fig.suptitle("Facial AUs vs SUDS Correlation", fontsize=14, y=1.02)
    plt.savefig("figures/Figure3/FAUS_suds.pdf")
else:
    fig.suptitle("Facial AUs vs Resting State Correlation", fontsize=14, y=1.02)
    plt.savefig("figures/Figure3/FAUS_ybocs.pdf")
plt.show()