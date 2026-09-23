import pandas as pd
import numpy as np

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

PATH_SETTINGS = "preprocessing/resting-state/get_stim_parameters/StimParams_AllPatients_20260828_0724.csv"
PATH_FEATURES = "plotting/Figure1/source_data/ybocs_features_wide_per_rs_sess.csv"

df_settings = pd.read_csv(PATH_SETTINGS)
df_features = pd.read_csv(PATH_FEATURES)
df_features["PatientID"] = df_features["subject"].apply(lambda x: int(x[-3:]))
# resplace the xe in sessName with resting-state
df_settings["sessName"] = df_settings["sessName"].str.replace("xe", "resting-state")
df_settings = df_settings.rename(columns={"SessDate": "date", "sessName": "rs_name"})
df_settings["date"] = df_settings["SessionDate"].apply(lambda x: x[1:].replace("_", "-"))
df_settings["PatientID"] = df_settings["PatientID"].apply(lambda x: int(x[1:]))
# replace rs_name in df_features if it is resting-state-DBSoff with resting-stateDBSoff
df_features["rs_name"] = df_features["rs_name"].str.replace("resting-state-DBSoff", "resting-stateDBSoff")
df_features["Hemisphere"] = df_features["channel"].apply(lambda x: x.split("_")[-1])
df_features["new_ch"] = df_features["channel"].apply(lambda x: ch_mapping.get(x, x))

df_merge = pd.merge(df_features, df_settings, left_on=["PatientID", "date", "rs_name", "Hemisphere"],
                    right_on=["PatientID", "date", "rs_name", "Hemisphere"], how="left")

features_names = df_merge.columns[2:551]
ind_chs = df_merge["channel"].unique()

def get_new_hem_name(row):
    hem = 'None'
    if 'left' in row['channel']:
        hem = 'left'
    if 'right' in row['channel']:
        hem = 'right'
    if 'left' in row['channel'] and 'right' in row['channel']:
        hem = 'both'
    return hem

df_merge["Hemisphere"] = df_merge.apply(get_new_hem_name, axis=1)
stim_columns = ['Amplitude_mA', 'PulseWidth_us', 'Frequency_Hz', "contact", "Anode+", "Cathode-"]

df_concat = []
for sub in df_merge["subject"].unique():
    l_concat_sub = []
    for sub_date in df_merge[df_merge["subject"] == sub]["date"].unique():
        for rs_sess_date_sub in df_merge[(df_merge["subject"] == sub) & (df_merge["date"] == sub_date)]["rs_name"].unique():
            df_sub_date_rs = df_merge[(df_merge["subject"] == sub) & (df_merge["date"] == sub_date) & (df_merge["rs_name"] == rs_sess_date_sub)]
            # remove dupblicate entries by index subject, date, rs_name
            df_sub_date_rs = df_sub_date_rs.drop_duplicates(subset=["subject", "date", "rs_name", "new_ch"])
            df_pivot = df_sub_date_rs.pivot(index=["subject", "date", "rs_name"], columns="new_ch", values=features_names)
            # flatten multiindex
            df_pivot.columns = [f"{ch}_{feat}" for feat, ch in df_pivot.columns]
            df_pivot = df_pivot.reset_index()
            # add score columns
            score_cols = df_sub_date_rs.columns[551:]
            for col_score in score_cols:
                if col_score in stim_columns:
                    for hem_ in ['left', 'right']:
                        df_hem = df_sub_date_rs[df_sub_date_rs["Hemisphere"] == hem_]
                        if not df_hem.empty:
                            df_pivot[f"{col_score}_{hem_}"] = df_hem[col_score].values[0]
                else:
                    df_pivot[col_score] = df_sub_date_rs[col_score].values[0]
            l_concat_sub.append(df_pivot)
    df_concat_sub = pd.concat(l_concat_sub)
    df_concat.append(df_concat_sub)
df_final = pd.concat(df_concat).reset_index(drop=True)

# drop columns with all NaN values
df_final = df_final.dropna(axis=1, how='all')

df_final.to_csv("preprocessing/resting-state/get_stim_parameters/df_merge_feature_stim_scores.csv")