import seaborn as sns
import pandas as pd
import numpy as np
import os
import pickle
from tqdm import tqdm
from scipy import stats
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from datetime import datetime


def get_date(f, ):
    idx_start = f.find("resting-state")
    idx_start = idx_start + f[idx_start:].find("_") + 1
    idx_sync_or_toggle = f.find("sync") if "sync" in f else f.find("toggle")
    if "BSoff" in f or "BSon" in f:
        idx_start = f.find("BSoff") + len("BSoff") + 1 if "BSoff" in f else f.find("BSon") + len("BSon") + 1
    return f[idx_start : idx_sync_or_toggle-1]

def convert_to_datetime(d):
    if d.startswith("20"):
        dt_ = datetime.strptime(d, "%Y%m%d%H%M%S%f")
    else:
        dt_ = datetime.fromtimestamp(int(d) / 1000)
    # keep only Ymd
    dt_ = dt_.strftime("%Y-%m-%d")
    return dt_

df_annot = pd.read_excel("preprocessing/resting-state/annotations_rcs.xlsx", sheet_name="Sheet1")
duplicates = df_annot["Duplicate"].dropna().unique()
all_wrong = df_annot.query("Index == 'ALL'")["Filename"].unique()

df_scores = pd.read_csv("preprocessing/resting-state/map_scores/scores_date_mapped_1.csv")
df_scores["file_short"] = [os.path.basename(f)[:-4] for f in df_scores["file"] ]
df_scores["file_short"] = df_scores["file_short"].str.replace(r'_stim_(left|right|NA)$', '', regex=True)
df_scores["file_short"] = df_scores["file_short"].str.replace(r'_(left|right)$', '', regex=True)

#SCORE_METRIC = 'YBOCS II Total Score'  # 'YBOCS II-Obsessions Sub-score', 'YBOCS II-Compulsions Sub-score'
SCORE_COLUMNS = list(df_scores.columns[:-4])
mapping_loc = {
    "C_0_left" : "Cortex",
    "C_1_left" : "Cortex",
    "C_0_right": "Cortex",
    "C_1_right": "Cortex",
    "SC_0_left": "VCVS",
    "SC_1_left": "VCVS",
    "SC_0_right": "VCVS",
    "SC_1_right": "VCVS",
    "SC_left_C_left": "conn",
    "SC_right_C_right": "conn",
    "SC_left_C_right": "conn",
    "SC_right_C_left": "conn",
    "C_left_right" : "conn",
    "SC_left_right": "conn"
}

ch_mapping = {
    "C_0_left" : "C_L_1",
    "C_1_left" : "C_L_2",
    "C_0_right": "C_R_1",
    "C_1_right": "C_R_2",
    "SC_0_left": "SC_L",
    "SC_1_left": "SC_L",
    "SC_0_right": "SC_R",
    "SC_1_right": "SC_R",
    "SC_left_C_left": "SC_left_C_left",
    "SC_right_C_right": "SC_right_C_right",
    "SC_left_C_right": "SC_left_C_right",
    "SC_right_C_left": "SC_right_C_left",
    "C_left_right" : "C_left_right",
    "SC_left_right": "SC_left_right"
}

f_bands = [[0, 4], [4, 8], [8, 15], [15, 30], [30, 55]]
f_bands_names = ["delta", "theta", "alpha", "beta", "gamma"]

def add_coh_features(df, path_):
    # replace end of path_ with "_coh.pkl"
    path_coh = path_.replace("_features_prep.csv", "_coh_spectra.pkl")
    with open(path_coh, "rb") as file:
        d_coh = pickle.load(file)
    # ["subject", "channel", "idx", "feature_name", "feature_value", "fs", "score"]
    idxs_in_df = df["idx"].unique()
    l_add = []
    for idx in idxs_in_df:
        if idx not in d_coh:
            continue
        d_coh_idx = d_coh[idx]
        
        for ch in d_coh_idx.keys():
            f_d_coh_idx = d_coh_idx[ch][0]
            coh_d_coh_idx = d_coh_idx[ch][1]
            for band, band_name in zip(f_bands, f_bands_names):
                coh_band = np.mean(coh_d_coh_idx[(f_d_coh_idx >= band[0]) & (f_d_coh_idx < band[1])])
                l_add.append({
                    "subject": df.query("idx == @idx")["subject"].values[0],
                    "channel": ch,
                    "idx": idx,
                    "feature_name": f"coherence_{band_name}",
                    "feature_value": coh_band,
                    "fs": df.query("idx == @idx")["fs"].values[0],
                    #"score": df.query("idx == @idx")["score"].values[0]
                })
    df_add = pd.DataFrame(l_add)
    df_out = pd.concat([df, df_add], ignore_index=True)
    return df_out

READ_FEATURES_COMBINED = False

if READ_FEATURES_COMBINED:
    PATH_READ = "/Users/Timon/Documents/Houston/resting_state_OCD/features_out"
    files = [f for f in os.listdir(PATH_READ) if "features_prep.csv" in f]
    dfs_ = []
    for file in tqdm(files[::-1]):
        f_name_find = file[:file.find("_features_prep")]
        if f_name_find in duplicates or f_name_find in all_wrong:
            continue
        df = pd.read_csv(os.path.join(PATH_READ, file))
        df = add_coh_features(df, os.path.join(PATH_READ, file))
        df_annot_file = df_annot.query("Filename == @f_name_find")

        df_filtered = df[~df["idx"].isin(df_annot_file["Index"].values)]
        if df_filtered.empty:
            print(f"File {file} is empty after filtering. Skipping.")
            continue
        df_g = df_filtered.groupby(["subject", "channel", "feature_name"])["feature_value"].median().reset_index()
        df_g["loc"] = df_g["channel"].apply(
            lambda x: mapping_loc[x] if x in mapping_loc else "Unknown"
        )
            
        df_g["hemisphere"] = df_g["channel"].apply(
            lambda x: "Left" if "left" in x and "right" not in x else
                    "Right" if "right" in x and "left" not in x else
                    "Both"
        )

        df_g["new_ch"] = df_g["channel"].apply(
            lambda x: ch_mapping[x] if x in ch_mapping else "Unknown"
        )

        df_g["file"] = f_name_find
        subject = f_name_find.split("_")[0]
        # replace combined_timeseries or combined_timeseries_lr with nothing
        f_name_find_name = f_name_find.replace("_combined_timeseries", "").replace("_combined_lr_timeseries", "")
        #df_g["score"] = df_filtered["score"].unique()[0]
        try:
            scores_series = pd.Series(
                df_scores.query("subject == @subject and file_short == @f_name_find_name")[SCORE_COLUMNS].values[0], 
                index=SCORE_COLUMNS)
        except IndexError:
            print(f"Scores not found for subject {subject} and file {f_name_find_name}. Skipping.")
            continue
        df_g["subject"] = subject
        # add scores_series to df_g
        for col in SCORE_COLUMNS:
            df_g[col] = scores_series[col]

        #df_g["score"] = scores_series


        dfs_.append(df_g)
    df_ = pd.concat(dfs_, ignore_index=True)
    df_.to_csv("preprocessing/resting-state/features_prep_combined_1.csv", index=False)
else:
    df_ = pd.read_csv("preprocessing/resting-state/features_prep_combined_1.csv")

GROUP_FEATURES_FOR_DECODING = False

if GROUP_FEATURES_FOR_DECODING:
    def get_date(f, ):
        idx_start = f.find("resting-state")
        idx_start = idx_start + f[idx_start:].find("_") + 1
        idx_sync_or_toggle = f.find("sync") if "sync" in f else f.find("toggle")
        if "BSoff" in f or "BSon" in f:
            idx_start = f.find("BSoff") + len("BSoff") + 1 if "BSoff" in f else f.find("BSon") + len("BSon") + 1
        return f[idx_start : idx_sync_or_toggle-1]

    def convert_to_datetime(d):
        if d.startswith("20"):
            dt_ = datetime.strptime(d, "%Y%m%d%H%M%S%f")
        else:
            dt_ = datetime.fromtimestamp(int(d) / 1000)
        # keep only Ymd
        dt_ = dt_.strftime("%Y-%m-%d")
        return dt_


    # --- Load data ---
    df_ = pd.read_csv("preprocessing/resting-state/features_prep_combined_1.csv")

    # --- 1. Ensure SCORE_COLUMNS are numeric in one shot ---
    df_[SCORE_COLUMNS] = df_[SCORE_COLUMNS].apply(pd.to_numeric, errors='coerce')

    # --- 2. Parse info from "file" column once ---
    # assumes "file" looks like "<subject>_<rs_name>_..."
    file_parts = df_["file"].str.split("_", n=2, expand=True)
    df_["subject"] = file_parts[0]
    df_["rs_name"] = file_parts[1]

    # date: keep your functions but call them only once on df_
    df_["date"] = df_["file"].apply(lambda x: convert_to_datetime(get_date(x)))

    # --- 3. Aggregate per file / feature / new_ch ---
    group_cols = ["file", "feature_name", "new_ch"]

    # dictionary for aggregations: mean for scores + feature_value, first for channel
    agg_dict = {c: "mean" for c in SCORE_COLUMNS + ["feature_value"]}
    agg_dict["channel"] = "first"

    df_features = (
        df_
        .groupby(group_cols, as_index=False)
        .agg(agg_dict)
    )

    df_features["subject"] = df_features["file"].apply(lambda x: x.split("_")[0])
    df_features["rs_name"] = df_features["file"].apply(lambda x: x.split("_")[1])
    df_features["date"] = df_features["file"].apply(lambda x: convert_to_datetime(get_date(x)))

    df_features_wide_per_rs_sess = df_features.pivot_table(
        index=["subject", "date", "file", "channel"],
        columns=["feature_name"],
        values="feature_value"
    ).reset_index()

    df_features_wide_per_rs_sess["rs_name"] = df_features_wide_per_rs_sess["file"].apply(lambda x: x.split("_")[1])

    # add SCORE_COLUMNS to df_features_wide_per_rs_sess
    df_scores = (
        df_[["subject", "date", "file"] + SCORE_COLUMNS]
        .groupby(["subject", "date", "file"], as_index=False)
        .mean()
    )

    df_features_wide_per_rs_sess = df_features_wide_per_rs_sess.merge(df_scores, on=["subject", "date", "file"], how="left")
    df_features_wide_per_rs_sess["new_ch"] = df_features_wide_per_rs_sess["channel"].apply(
        lambda x: ch_mapping[x] if x in ch_mapping else "Unknown"
    )
    # drop duplicates based on subject, date, new_ch
    df_features_wide_per_rs_sess = df_features_wide_per_rs_sess.drop_duplicates(subset=["subject", "date", "new_ch"])
    df_features_wide_per_rs_sess.to_csv("plotting/Figure1/source_data/ybocs_features_wide_per_rs_sess.csv", index=False)

    # group a second time, but just subject, date, new_ch, feature_name
    df_features_mean_date = (
        df_features
        .groupby(["subject", "date", "new_ch", "feature_name"], as_index=False)
        .agg(agg_dict)
    )

    # if new_ch is "Misc", set new_ch to channel value
    df_features_mean_date.loc[df_features_mean_date["new_ch"] == "Misc", "new_ch"] = df_features_mean_date["channel"]

    feature_names = df_features_mean_date["feature_name"].unique()

    # now I want to pivot the the dataframe, such that each column is a {new_ch}_{feature_name} combination, and each row is a unique (subject, date) pair. Other columns are kept
    df_features_wide = df_features_mean_date.pivot_table(
        index=["subject", "date"],
        columns=["new_ch", "feature_name"],
        values="feature_value"
    ).reset_index()

    # combine the multi-index columns into single level, with format "{new_ch}_{feature_name}"
    df_features_wide.columns = [f"{ch}_{feat}" if ch not in ["subject", "date"] else ch for ch, feat in df_features_wide.columns]
    # delete all columns with Uuknown in their name
    df_features_wide = df_features_wide[[c for c in df_features_wide.columns if "Unknown" not in c]]

    df_features_wide = df_features_wide.merge(df_scores, on=["subject", "date"], how="left")

    # replace column names in df_features_wide
    # feature_name = col.replace(f"{region}_", "") if region else col
    # and feature_name.replace(f"_{region}", "")
    for col in df_features_wide.columns:
        if "coh" in col:
            for region_candidate in list(ch_mapping.values()):
                if col.startswith(f"{region_candidate}_"):
                    region = region_candidate
                    feature_name = col.replace(f"{region}_", "")
                    if region in feature_name:
                        feature_name = feature_name.replace(f"_{region}", "")
                    new_col_name = f"coh_{region}_{feature_name}"
                    df_features_wide.rename(columns={col: new_col_name}, inplace=True)
                    break
    # if corr in col, remove everything after corr
    df_features_wide.rename(columns={col: col.split("_corr")[0] for col in df_features_wide.columns if "corr" in col}, inplace=True)

    # if col starts with "SC_left_C_left", "SC_right_C_right", "SC_left_C_right", "SC_right_C_left", "C_left_right", "SC_left_right", then replace name to start with "coh_"
    df_features_wide.rename(columns={col: f"coh_{col}" for col in df_features_wide.columns if col.startswith(("SC_left_C_left", "SC_right_C_right", "SC_left_C_right", "SC_right_C_left", "C_left_right", "SC_left_right"))}, inplace=True)
    # drop duplicates
    df_features_wide = df_features_wide.drop_duplicates(subset=["subject", "date"])
    df_features_wide.to_csv("/Users/Timon/Documents/Houston/YBOCS/plotting/Figure1/source_data/neural_features_ybocs_with_coh_1.csv", index=False)

COMPUTE_CORRELATIONS = True

if COMPUTE_CORRELATIONS:

    corr_results_coh = []
    corr_results_uni = []
    score_column = 'YBOCS II Total Score'
    #df = pd.read_csv("features_prep_combined_wide_with_coh_1.csv")
    df = pd.read_csv("/Users/Timon/Documents/Houston/YBOCS/plotting/Figure1/source_data/neural_features_ybocs_with_coh_1.csv")
    subjects = df["subject"].unique()
    for subject in subjects:
        df_sub = df.query("subject == @subject")
        score_series = df_sub[score_column]
        columns_corr = [col for col in df_sub.columns if col not in ["subject", "date", "file"] + SCORE_COLUMNS and "psd" not in col]
        for col in columns_corr:
            feature_series = df_sub[col]
            # drop rows with NaN in either series
            valid_idx = score_series.notna() & feature_series.notna()
            if valid_idx.sum() < 2:
                continue

            region = None
            for region_candidate in list(ch_mapping.values()):
                if col.startswith(f"{region_candidate}_") or col.startswith(f"coh_{region_candidate}_"):
                    region = region_candidate
                    break
            feature_name = col.replace(f"{region}_", "") if region else col
            if "coh_" in col:
                feature_name = feature_name.replace(f"coh_", "")
            if "fft" in feature_name:
                continue
            print(f"{col} - {region} - {feature_name}")
            corr, p_value = stats.pearsonr(score_series[valid_idx], feature_series[valid_idx])



            if "coh" in col or "corr" in col:
                corr_results_coh.append({
                    "subject": subject,
                    "region": "coh_" + region,
                    "feature": feature_name,
                    "correlation": corr,
                    "corr_abs" : abs(corr),
                    "p_value": p_value,
                })
            else:
                corr_results_uni.append({
                    "subject": subject,
                    "region": region,
                    "feature": feature_name,
                    "correlation": corr,
                    "corr_abs" : abs(corr),
                    "p_value": p_value,
                })
    
    df_corr_results_coh = pd.DataFrame(corr_results_coh)
    df_corr_results_uni = pd.DataFrame(corr_results_uni)
    PATH_SAVE = "/Users/Timon/Documents/Houston/YBOCS/plotting/Figure2/source_data"
    df_corr_results_coh["subject"] = df_corr_results_coh["subject"].apply(lambda x: int(x[-3:]))
    df_corr_results_uni["subject"] = df_corr_results_uni["subject"].apply(lambda x: int(x[-3:]))
    df_corr_results_coh.to_csv(os.path.join(PATH_SAVE, "feature_region_subject_correlations_ybocs_coherence_1.csv"), index=False)
    df_corr_results_uni.to_csv(os.path.join(PATH_SAVE, "feature_region_subject_correlations_ybocs_1.csv"), index=False)