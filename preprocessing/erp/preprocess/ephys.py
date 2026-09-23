import pandas as pd
import os
import numpy as np

def read_mapping_neural_data(sub, PATH_EPHYS = f"/Users/Timon/Desktop/aDBS004Home", diff_hours: int=0) -> list:

    PATH_SUDS = f"{sub}/{sub}_suds_ratings.csv"
    df_suds = pd.read_csv(PATH_SUDS)
    df_suds["ts_datetime"] = pd.to_datetime(df_suds["ts_datetime"])

    suds_start_time = df_suds["timestamp"].min() + diff_hours * 3600
    suds_end_time = df_suds["timestamp"].max() + diff_hours * 3600

    sessions_ = [int(f[len("Session"):]) / 1000 for f in os.listdir(PATH_EPHYS) if f.startswith("Session") and os.path.isdir(os.path.join(PATH_EPHYS, f))]
    sessions_ = np.array(sorted(sessions_))
    idx_bigger = np.where(sessions_ >= suds_start_time)[0][0]
    idx_smaller = np.where(sessions_ <= suds_end_time)[0][-1]
    if "011" in sub:
        sessions_fit = sessions_#[idx_bigger:idx_smaller+3] # idx_bigger-1
    else:
        sessions_fit = sessions_[idx_bigger-3:idx_smaller+3] # idx_bigger-1

    return sessions_fit

def concat_all_sessions_fit(sessions_fit: list, PATH_EPHYS: str, PATH_OUT: str):

    # read all sessions_fit
    dfs_ = []
    for s_idx, s in enumerate(sessions_fit):
        print(f"Processing session {s} - {s_idx+1}/{len(sessions_fit)}")
        PATH_SESSION = os.path.join(PATH_EPHYS, f"Session{s*1000:06.0f}")
        if len(os.listdir(PATH_SESSION)) == 0:
            continue
        first_folder = [f for f in os.listdir(PATH_SESSION) if os.path.isdir(os.path.join(PATH_SESSION, f))][0]
        PATH_FIRST_FOLDER = os.path.join(PATH_SESSION, first_folder)
        if "NeuralTimeDomain.csv" in os.listdir(PATH_FIRST_FOLDER):
            df = pd.read_csv(os.path.join(PATH_FIRST_FOLDER, "NeuralTimeDomain.csv"))
            dfs_.append(df)
    df_sessions = pd.concat(dfs_, ignore_index=True)
    # sort by timestamp
    df_sessions = df_sessions.sort_values(by="timestamp").reset_index(drop=True)

    df_sessions.to_csv(PATH_OUT, index=False)
    return df_sessions

# # add the ratings to the df_sessions
# df_sessions["ratings"] = np.nan
# # set the ratings always to the greater timestamp

# for idx, row in df_suds.iterrows():
#     print(idx)
#     ts = row["timestamp"]
#     rating = row["ratings"]
#     # find the closest timestamp in df_sessions that is greater than ts
#     idx_ = (df_sessions["timestamp"] >= ts).index.min()
    
#     if idx_ is not None:
#         df_sessions.at[idx_, "ratings"] = rating


# df_sessions.to_csv("005/005_ephys_sessions.csv", index=False)