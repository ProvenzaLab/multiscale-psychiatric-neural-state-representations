import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from scipy.signal import welch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import pickle

def comb_df_with_suds(PATH_EPHYS_COMB_: str, PATH_SUDS: str, PATH_OUT: str, 
            ch_used:str="time_domain_2_0", extraction_window_s: int = 120,
            hour_offset: int = 5, time_extraction_before_suds_s: int = 60,
            GET_ALL_WINDOW_WITH_NANS: bool = True) -> pd.DataFrame:

    sub = PATH_EPHYS_COMB_.split("/")[0]
    df_ = pd.read_csv(PATH_EPHYS_COMB_)
    df_["timestamp"] = df_["timestamp"] / 1000
    df_.set_index("timestamp", inplace=True)

    df_ = df_.sort_index()
    timestamps = df_.index.values
    if "011" in sub:
        timestamps[idx_more] += 1 * 3600
    else:
        timestamps += hour_offset * 3600
    #pd.Timestamp(timestamps[0], unit="s")

    df_ = df_.reset_index()

    df_suds = pd.read_csv(PATH_SUDS)
    df_suds["ts_datetime"] = pd.to_datetime(df_suds["ts_datetime"])

    dfs_ = []

    idx_cnt = 0
    for idx, row in df_suds.iterrows():
        print(f"Processing row {idx+1}/{df_suds.shape[0]}")
        ts = row["timestamp"] - time_extraction_before_suds_s
        rating = row["ratings"]

        start_idx = np.searchsorted(timestamps, ts, side="left")
        # check difference between start_idx and ts
        if start_idx == len(timestamps):
            print(f"Skipping row {idx+1}/{df_suds.shape[0]} because start index is out of bounds")
            continue
        if np.abs(timestamps[start_idx]  - ts) > 5:
            print(f"Skipping row {idx+1}/{df_suds.shape[0]} because start index is too far from timestamp")
            print(f"Difference: {(timestamps[start_idx] - ts) / 3600} h")
            continue
    
        end_idx  = np.searchsorted(timestamps, timestamps[start_idx] + extraction_window_s, side="right")
        print(f"Start index: {start_idx}, End index: {end_idx}, Timestamp: {ts}, Rating: {rating}")
        print(f"Start timestamp: {timestamps[start_idx]}, End timestamp: {timestamps[end_idx-1]}")
        df_range = df_.iloc[start_idx:end_idx].reset_index()
        if end_idx - start_idx < (30 * 250):
            print(f"Skipping row {idx+1}/{df_suds.shape[0]} because range is too small: {(end_idx - start_idx) / 250 } s")
            continue
        df_range.index = pd.to_datetime(df_range["timestamp"], unit='s')
        df_range = df_range.resample("4ms").mean()  # 250 Hz = 4 ms
        df_range = df_range.fillna(method='ffill', limit=5)
        df_range["idx_counter"] = idx_cnt
        df_range["rating"] = rating

        # from matplotlib import pyplot as plt
        # plt.plot(df_range["time_domain_9_8"])
        # plt.xlabel("Time")
        # plt.ylabel("Voltage [uV]")
        # plt.title("sub 008 ch 11-10")

        # get largest consecutive range without NaN values
        if GET_ALL_WINDOW_WITH_NANS == False:
            df_range["valid"] = df_range[ch_used].notna()
            df_range["valid_group"] = (df_range["valid"] != df_range["valid"].shift()).cumsum()
            largest_group = df_range.groupby("valid_group").size().idxmax()
            largest_range = df_range[df_range["valid_group"] == largest_group]
            df_range_largest = largest_range[largest_range["valid"]]

            #df_range_largest["rating"] = rating
            if df_range_largest.empty or df_range_largest.shape[0] < 1500:
                print(f"Skipping empty df_range_largest at index {idx}")
                continue
            df_range = df_range_largest
        #df_range_largest["idx_counter"] = idx_cnt
        idx_cnt += 1
        dfs_.append(df_range)
    print(f"Number of valid dataframes: {len(dfs_)} out of {df_suds.shape[0]}")
    df_comb_ = pd.concat(dfs_, ignore_index=True)
    df_comb_.to_csv(PATH_OUT, index=False)
    return df_comb_

def combine_L_R(sub: str):

    df_comb_L = pd.read_csv(f"{sub}L/comb_mapped_{sub}L.csv")
    df_comb_R = pd.read_csv(f"{sub}R/comb_mapped_{sub}R.csv")

    df_comb_L = df_comb_L[~df_comb_L["timestamp"].isin([np.inf, -np.inf, np.nan])]
    df_comb_R = df_comb_R[~df_comb_R["timestamp"].isin([np.inf, -np.inf, np.nan])]

    df_comb_L["timestamp"] = (df_comb_L["timestamp"] *1000).astype(int)
    df_comb_R["timestamp"] = (df_comb_R["timestamp"] *1000).astype(int)

    mean_timestamp_idxs_L = list((df_comb_L.groupby("idx_counter")["timestamp"].mean()).astype(int).values)
    mean_timestamp_idxs_R = list((df_comb_R.groupby("idx_counter")["timestamp"].mean()).astype(int).values)
    list_R_already_used = []

    idx_cnt_L = list(df_comb_L.groupby("idx_counter")["timestamp"].mean().keys())
    idx_cnt_R = list(df_comb_R.groupby("idx_counter")["timestamp"].mean().keys())

    idx_cnt_both = 0
    dict_res_idx = {}

    for ts_idx, ts_left in enumerate(mean_timestamp_idxs_L):
        diff_with_R = np.abs(np.array(mean_timestamp_idxs_R) - ts_left)
        min_abs_diff = np.min(diff_with_R)
        min_diff_idx_R = np.argmin(diff_with_R)
        mean_ts_R = mean_timestamp_idxs_R[min_diff_idx_R]
        if int(min_abs_diff) < 5000:
            # match
            list_R_already_used.append(mean_ts_R) # remove this one
            # now concatenate the two
            df_L_idx = df_comb_L[df_comb_L["idx_counter"] == idx_cnt_L[ts_idx]]
            df_R_idx = df_comb_R[df_comb_R["idx_counter"] == idx_cnt_R[min_diff_idx_R]]

            df_L_idx = df_L_idx.set_index("timestamp")
            df_R_idx = df_R_idx.set_index("timestamp")

            df_L_idx.index = pd.to_datetime(df_L_idx.index, unit='ms')
            df_R_idx.index = pd.to_datetime(df_R_idx.index, unit='ms')
            df_timestamps_L_start = df_L_idx.index[0]

            df_L_idx = df_L_idx.sort_index()
            df_R_idx = df_R_idx.sort_index()

            df_L_resampled = df_L_idx.resample("4ms", origin=df_timestamps_L_start).mean()
            df_R_resampled = df_R_idx.resample("4ms", origin=df_timestamps_L_start).mean()

            # now merge the two dataframes
            df_comb_L_R = pd.merge(df_L_resampled, df_R_resampled, left_index=True, right_index=True, suffixes=("_L", "_R"))
            df_comb_L_R = df_comb_L_R.drop(columns=["index_L", "index_R", "rating_L",
                                                    "idx_counter_L", "idx_counter_R"])
            df_comb_L_R = df_comb_L_R.rename(columns={"rating_R": "rating"})
            df_comb_L_R = df_comb_L_R.dropna(axis=1, how='all')
            dict_res_idx[idx_cnt_both] = df_comb_L_R
        else:
            # no match, just add the left one
            df_L_idx = df_comb_L[df_comb_L["idx_counter"] == idx_cnt_L[ts_idx]]
            df_L_idx = df_L_idx.set_index("timestamp")
            df_L_idx.index = pd.to_datetime(df_L_idx.index, unit='ms')
            df_L_idx = df_L_idx.sort_index()
            df_L_resampled = df_L_idx.resample("4ms").mean()
            df_L_resampled = df_L_resampled.drop(columns=["index", "idx_counter"])
            df_L_resampled = df_L_resampled.dropna(axis=1, how='all')
            df_L_resampled = df_L_resampled.rename(
                columns={col: f"{col}_L" for col in df_L_resampled.columns if "time_domain" in col}
            )
            dict_res_idx[idx_cnt_both] = df_L_resampled
        idx_cnt_both += 1

    for ts_idx, ts_right in enumerate(mean_timestamp_idxs_R):
        if ts_right in list_R_already_used:
            continue
        df_R_idx = df_comb_R[df_comb_R["idx_counter"] == idx_cnt_R[ts_idx]]
        df_R_idx = df_R_idx.set_index("timestamp")
        df_R_idx.index = pd.to_datetime(df_R_idx.index, unit='ms')
        df_R_idx = df_R_idx.sort_index()
        df_R_resampled = df_R_idx.resample("4ms").mean()
        df_R_resampled = df_R_resampled.drop(columns=["index", "idx_counter"])
        df_R_resampled = df_R_resampled.dropna(axis=1, how='all')
        df_R_resampled = df_R_resampled.rename(
            columns={col: f"{col}_R" for col in df_R_resampled.columns if "time_domain" in col}
        )
        dict_res_idx[idx_cnt_both] = df_R_resampled
        idx_cnt_both += 1

    with open(f"{sub}/{sub}_comb_data.pickle", 'wb') as f:
        pickle.dump(dict_res_idx, f)

def rewrite_data_to_pickle(sub: str):
    df_comb = pd.read_csv(f"{sub}/comb_mapped_{sub}.csv")

    df_comb = df_comb[~df_comb["timestamp"].isin([np.inf, -np.inf, np.nan])]

    df_comb["timestamp"] = (df_comb["timestamp"] *1000).astype(int)

    mean_timestamp_idxs = list((df_comb.groupby("idx_counter")["timestamp"].mean()).astype(int).values)

    idx_cnt = list(df_comb.groupby("idx_counter")["timestamp"].mean().keys())

    idx_cnt_both = 0
    dict_res_idx = {}

    for ts_idx, ts_left in enumerate(mean_timestamp_idxs):
        df_idx = df_comb[df_comb["idx_counter"] == idx_cnt[ts_idx]]
        df_idx = df_idx.set_index("timestamp")
        df_idx.index = pd.to_datetime(df_idx.index, unit='ms')
        df_idx = df_idx.sort_index()
        df_resampled = df_idx.resample("4ms").mean()
        df_resampled = df_resampled.drop(columns=["index", "idx_counter"])
        df_resampled = df_resampled.dropna(axis=1, how='all')
        dict_res_idx[idx_cnt_both] = df_resampled
        idx_cnt_both += 1

    with open(f"{sub}/{sub}_comb_data.pickle", 'wb') as f:
        pickle.dump(dict_res_idx, f)