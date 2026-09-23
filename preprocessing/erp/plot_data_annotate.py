import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy import stats
import mne

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, signal
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from preprocessing.erp.preprocess.utils_params import process_dict
import pickle
from joblib import Parallel, delayed

patients_ = ["011", "012"] # ["004", "005", "007", "009", "010"] 
df_ = []
fs = 250
FILTER = False
PLOT_ = True

def plot_ts_and_split_df(patient):

    if FILTER:
        pdf_path = f"{patient}/ts_all_figures_{patient}_bandstopfiltered100Hz.pdf"
    else:
        pdf_path = f"{patient}/ts_all_figures_{patient}.pdf"
    print(f"Processing {patient}...")
    if PLOT_ is True:
        pdf_ = PdfPages(pdf_path)
    # read 004_comb_data.pickle
    with open(f"{patient}/{patient}_comb_data.pickle", "rb") as f:
        dict_sub = pickle.load(f)
    dict_sub_out = {}
    df_sub_ = []
    for idx_cnt in tqdm(dict_sub.keys()):
        df_range_ = dict_sub[idx_cnt]
        rating_ = df_range_["rating"].iloc[0]
        dict_sub_out[idx_cnt] = {}
        dict_sub_out[idx_cnt]["rating"] = rating_
        dict_sub_out[idx_cnt]["mean_time"] = df_range_.index.mean()

        # split the df again in 5 sec ranges
        df_5s_ranges = []
        cnt_5s_range = 0
        for start in range(0, len(df_range_), 5 * fs):
            end = start + 5 * fs
            df_5s_range = df_range_[start:end][[c for c in df_range_.columns if c.startswith("time_domain")]]
            if len(df_5s_range) < 3 * fs:
                continue
            if df_5s_range.isnull().values.any():
                continue
            df_5s_ranges.append(df_5s_range)
            if "data" not in dict_sub_out[idx_cnt]:
                dict_sub_out[idx_cnt]["data"] = {}
            dict_sub_out[idx_cnt]["data"][cnt_5s_range] = df_5s_range
            cnt_5s_range += 1
        if PLOT_ is False:
            continue
        for df_range_idx, df_range_ in enumerate(df_5s_ranges):
            ch_names = [f for f in df_range_.columns if f.startswith("time_domain")]
            height = len(ch_names) + 2
            plt.figure(figsize=(12, height))
            plt.suptitle(f"{patient} - idx: {idx_cnt} df_range_idx: {df_range_idx}\nRating: {rating_} " +
                         f"timestamp: {df_range_.index[0]}\n{', '.join(ch_names)}")
            for ch_idx, ch_name in enumerate(ch_names):
                data_ = df_range_[ch_name].to_numpy()
                # if data has NaN continue

                # fill nans with 0
                if FILTER:
                    data_raw = np.nan_to_num(data_, nan=0.0)
                    data_filtered = mne.filter.filter_data(
                        data_raw,
                        sfreq=fs,
                        l_freq=105,
                        h_freq=95,
                        method='iir',
                        verbose=False
                    )
                    data_filtered = mne.filter.filter_data(
                        data_filtered,
                        sfreq=fs,
                        l_freq=65,
                        h_freq=55,
                        method='iir',
                        verbose=False
                    )
                    data_filtered = mne.filter.filter_data(
                        data_filtered,
                        sfreq=fs,
                        l_freq=0.5,
                        h_freq=None,
                        method='iir',
                        verbose=False
                    )
                    data_ = data_filtered

                plt.subplot(len(ch_names), 3, ch_idx*3 + 1)
                plt.plot(np.arange(0, data_.shape[0]/fs, 1/fs), data_, linewidth=0.5)
                plt.gca().spines['right'].set_visible(False); plt.gca().spines['top'].set_visible(False)
                plt.xlabel("Time [s]")
                plt.ylabel(f"{ch_name}")
                plt.subplot(len(ch_names), 3, ch_idx*3 + 2)
                plt.plot(np.arange(0, 250/fs, 1/fs), data_[:250], linewidth=0.5)
                plt.gca().spines['right'].set_visible(False); plt.gca().spines['top'].set_visible(False)
                plt.xlabel("Time [s]")
                plt.ylabel("Amplitude [a.u.]")
                plt.subplot(len(ch_names), 3, ch_idx*3 + 3)
                if FILTER is False:
                    data_ = np.nan_to_num(data_, nan=0.0)
                f, Pxx = signal.welch(data_, fs=fs, nperseg=250)
                plt.plot(f, np.log(Pxx), linewidth=0.5)
                plt.gca().spines['right'].set_visible(False); plt.gca().spines['top'].set_visible(False)
                plt.xlabel("Frequency [Hz]")
                plt.ylabel("PSD [a.u.]")
            plt.tight_layout()
            pdf_.savefig(bbox_inches='tight')
            plt.close()
    if PLOT_ is True:
        pdf_.close()

    with open(f"{patient}/{patient}_comb_data_5s.pickle", "wb") as f:
        pickle.dump(dict_sub_out, f)

# Parallel processing for all patients
plot_ts_and_split_df("012")  # Test with the first patient to ensure everything works
Parallel(n_jobs=-1)(delayed(plot_ts_and_split_df)(patient) for patient in patients_)