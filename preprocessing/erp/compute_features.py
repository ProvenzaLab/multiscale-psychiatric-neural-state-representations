import pandas as pd
import numpy as np
from scipy import signal
import mne
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import pickle
from matplotlib import pyplot as plt
from fooof import FOOOF, Bands
import py_neuromodulation as nm

from preprocess.preprocess_data import preprocess_data
from compute import run_py_neuro, compute_directionality, compute_bursts


f_bands = [[0, 4], [4, 8], [8, 15], [15, 30], [30, 55]]
f_bands_names = ["delta", "theta", "alpha", "beta", "gamma"]
fooof_bands = Bands({"delta": [0, 4], "theta": [4, 8], "alpha": [8, 15], "beta": [15, 30], "gamma": [30, 55]})
sum_range = [1, 80]
fooof_range = [10, 50]
out_path_py_neuro = "py_neuro_out"
fs = 250

df_art_annotations = pd.read_excel("Annotations_ephys.xlsx", sheet_name="Sheet1")
# replace 4 with "004", "5" with "005", etc.
df_art_annotations["Subject"] = df_art_annotations["Subject"].astype(str).str.zfill(3)


def compute_sub(sub):
    path_out = f"{sub}/out_features"
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    with open(f"{sub}/{sub}_comb_data_5s.pickle", "rb") as f:
        dict_sub = pickle.load(f)

    idxs_outer = list(dict_sub.keys())
    scores = np.array([float(dict_sub[k]["rating"]) for k in dict_sub.keys()])
    times = np.array([dict_sub[k]["mean_time"] for k in dict_sub.keys()])

    d_features = []
    spectra_ = {}
    coh_spectra = {}
    for cnt_idx, idx_outer in enumerate(idxs_outer):
        if "data" not in dict_sub[idx_outer]:
            continue
        idxs_inner = list(dict_sub[idx_outer]["data"].keys())
        score = scores[cnt_idx]
        time = times[cnt_idx]

        for idx_inner in idxs_inner:
            df_q_annot = df_art_annotations.query(f"Subject == '{sub}' and Idx_inner == {idx_inner} and Idx_outer == {idx_outer}")
            if len(df_q_annot) > 0:
                continue
            df_d_idx = dict_sub[idx_outer]["data"][idx_inner]
            df_d_idx = df_d_idx.drop(columns=["rating"], errors='ignore')

            data_l = []
            chs = df_d_idx.columns.tolist()
            for ch in chs:

                # annotation for artifact only in ch0 for sub005
                if sub == "005" and ch == "time_domain_2_0" and idx_outer >= 158 and idx_outer < 193:
                    continue
                ch_name_ = ch.replace("time_domain_", "")
                data_ = df_d_idx[ch].values
                data_filtered = preprocess_data(data_, fs)

                data_l.append(data_filtered)
                data_ = data_filtered

                f, Zxx_ = signal.welch(data_, fs=fs, nperseg=fs)
                Zxx = np.log(Zxx_)
                if idx_outer not in spectra_:
                    spectra_[idx_outer] = {}
                if idx_inner not in spectra_[idx_outer]:
                    spectra_[idx_outer][idx_inner] = {}
                spectra_[idx_outer][idx_inner][ch] = (f, Zxx)

                # plt.figure()
                # plt.plot(f, Zxx, linewidth=0.5)
                # plt.savefig("test_psd.pdf")

                sum_range_idx_ = np.where((f >= sum_range[0]) & (f <= sum_range[1]))[0]
                total_power = np.sum(Zxx[sum_range_idx_])

                fm = FOOOF()
                fm.fit(f, Zxx_, freq_range=fooof_range)
                offset, exponent = fm.aperiodic_params_
                d_features.append({
                    "subject": sub,
                    "channel": ch_name_,
                    "idx_outer": idx_outer,
                    "idx_inner": idx_inner,
                    "score" : score,
                    "time": time,
                    "feature_name" : "offset",
                    "feature_value": offset,
                })
                d_features.append({
                    "subject": sub,
                    "channel": ch_name_,
                    "idx_outer": idx_outer,
                    "idx_inner": idx_inner,
                    "score" : score,
                    "time": time,
                    "feature_name" : "exponent",
                    "feature_value": exponent,
                })

                for i, (f_band, f_band_name) in enumerate(zip(f_bands, f_bands_names)):
                    idx_band = np.where((f >= f_band[0]) & (f <= f_band[1]))[0]
                    if len(idx_band) == 0:
                        continue
                    band_power = np.mean(Zxx[idx_band]) / total_power
                    d_features.append({
                        "subject": sub,
                        "channel": ch_name_,
                        "idx_outer": idx_outer,
                        "idx_inner": idx_inner,
                        "score": score,
                        "time": time,
                        "feature_name": f_band_name,
                        "feature_value": band_power,
                    })

            data_pynm = np.array(data_l)

            ### BURST COMPUTATION
            for ch_idx, ch in enumerate(chs):
                ch_name_ = ch.replace("time_domain_", "")
                for i, (f_band, f_band_name) in enumerate(zip(f_bands, f_bands_names)):
                    try:
                        bursts_df = compute_bursts(data_pynm[ch_idx].copy(), fs, filter_low=f_band[0], filter_high=f_band[1])
                        d_features.append({
                            "subject": sub,
                            "channel": ch_name_,
                            "idx_outer": idx_outer,
                            "idx_inner": idx_inner,
                            "score": score,
                            "time": time,
                            "feature_name": f"burst_duration_{f_band_name}_ms",
                            "feature_value": bursts_df["duration"].mean()*20,
                        })
                        d_features.append({
                            "subject": sub,
                            "channel": ch_name_,
                            "idx_outer": idx_outer,
                            "idx_inner": idx_inner,
                            "score": score,
                            "time": time,
                            "feature_name": f"burst_amplitude_{f_band_name}",
                            "feature_value": bursts_df["mean_amplitude"].mean(),
                        })
                    except Exception as e:
                        continue

            ### Py-NEURO FEATURE COMPUTATION
            features = run_py_neuro(data_pynm, idx_outer, idx_inner,
                                    fs, chs, fooof_range,
                                    out_path_py_neuro=f"{sub}/out_py_neuro")

            if features is None:
                continue
            if sub == "004":
                chs_sc_left = ["time_domain_2_0"]
                chs_sc_right = ["time_domain_10_8"]
                chs_c_left = []
                chs_c_right = []
            elif sub == "005":
                chs_sc_left = ["time_domain_2_0"]
                chs_sc_right = ["time_domain_10_8"]
                chs_c_left = []
                chs_c_right = []
            elif sub == "007":
                chs_sc_left = ["time_domain_2_0"]
                chs_sc_right = ["time_domain_11_10"]
                chs_c_left = []
                chs_c_right = []
            elif sub == "011" or sub == "012":
                chs_sc_left = []
                chs_sc_right = []
                chs_c_left = []
                chs_c_right = []
                if "time_domain_2_0_a_L" in chs:
                    chs_sc_left.append("time_domain_2_0_a_L")
                if "time_domain_2_0_b_L" in chs:
                    chs_sc_left.append("time_domain_2_0_b_L")
                if "time_domain_2_0_a_R" in chs:
                    chs_sc_right.append("time_domain_2_0_a_R")
                if "time_domain_2_0_b_R" in chs:
                    chs_sc_right.append("time_domain_2_0_b_R")
                if "time_domain_11_10_L" in chs:
                    chs_c_left.append("time_domain_11_10_L")
                if "time_domain_9_8_L" in chs:
                    chs_c_left.append("time_domain_9_8_L")
                if "time_domain_11_10_R" in chs:
                    chs_c_right.append("time_domain_11_10_R")
                if "time_domain_9_8_R" in chs:
                    chs_c_right.append("time_domain_9_8_R")
            elif sub == "009":
                chs_sc_left = []
                if "time_domain_2_0_L" in chs:
                    chs_sc_left.append("time_domain_2_0_L")
                
                chs_sc_right = []
                if "time_domain_2_0_R" in chs:
                    chs_sc_right.append("time_domain_2_0_R")
                
                chs_c_left = []
                if "time_domain_9_8_L" in chs:
                    chs_c_left.append("time_domain_9_8_L")
                if "time_domain_11_10_L" in chs:
                    chs_c_left.append("time_domain_11_10_L")
                chs_c_right = []
                if "time_domain_9_8_R" in chs:
                    chs_c_right.append("time_domain_9_8_R")
                if "time_domain_11_10_R" in chs:
                    chs_c_right.append("time_domain_11_10_R")
            elif sub == "010":
                chs_sc_left = []
                if "time_domain_3_1_L" in chs:
                    chs_sc_left.append("time_domain_3_1_L")
                
                chs_sc_right = []
                if "time_domain_2_0_R" in chs:
                    chs_sc_right.append("time_domain_2_0_R")
                
                chs_c_left = []
                if "time_domain_11_10_L" in chs:
                    chs_c_left.append("time_domain_11_10_L")
                if "time_domain_9_8_L" in chs:
                    chs_c_left.append("time_domain_9_8_L")
                chs_c_right = []
                if "time_domain_11_10_R" in chs:
                    chs_c_right.append("time_domain_11_10_R")
                if "time_domain_9_8_R" in chs:
                    chs_c_right.append("time_domain_9_8_R")

            # chs_sc_left = [c for c in chs if c.startswith("SC_") and c.endswith("_left")]
            # chs_sc_right = [c for c in chs if c.startswith("SC_") and c.endswith("_right")]
            # chs_c_left = [c for c in chs if c.startswith("C_") and c.endswith("_left")]
            # chs_c_right = [c for c in chs if c.startswith("C_") and c.endswith("_right")]
            
            combs = [
                (chs_sc_left, chs_sc_right),
                (chs_c_left, chs_c_right),
                (chs_sc_left, chs_c_left),
                (chs_sc_right, chs_c_right),
                (chs_sc_left, chs_c_right),
                (chs_sc_right, chs_c_left),
            ]
            comb_names = [
                "SC_left_right",
                "C_left_right",
                "SC_left_C_left",
                "SC_right_C_right",
                "SC_left_C_right",
                "SC_right_C_left",
            ]
            for i, (chs1, chs2) in enumerate(combs):
                corrs_ex, corrs_off, coh_, f_coh_ = compute_directionality(data_pynm, fs, chs, chs1, chs2, features)
                if len(corrs_ex) > 0:
                    d_features.append({
                        "subject": sub,
                        "channel": comb_names[i],
                        "idx_outer": idx_outer,
                        "idx_inner": idx_inner,
                        "score": score,
                        "time": time,
                        "feature_name": f"fooof_a_exp_corr_{comb_names[i]}",
                        "feature_value": np.mean(corrs_ex),
                    })
                    d_features.append({
                        "subject": sub,
                        "channel": comb_names[i],
                        "idx_outer": idx_outer,
                        "idx_inner": idx_inner,
                        "score": score,
                        "time": time,
                        "feature_name": f"fooof_a_offset_corr_{comb_names[i]}",
                        "feature_value": np.mean(corrs_off),
                    })

                    for j, f_band_name in enumerate(f_bands_names):
                        if f_coh_ is not None:
                            idx_band = np.where((f_coh_ >= f_bands[j][0]) & (f_coh_ <= f_bands[j][1]))[0]
                            if len(idx_band) > 0:
                                coh_mean = np.mean(coh_[:, idx_band], axis=1)
                                d_features.append({
                                    "subject": sub,
                                    "channel": comb_names[i],
                                    "idx_outer": idx_outer,
                                    "idx_inner": idx_inner,
                                    "score": score,
                                    "time": time,
                                    "feature_name": f"coherence_{f_band_name}_{comb_names[i]}",
                                    "feature_value": coh_mean.mean(),
                                })

                    if idx_outer not in coh_spectra:
                        coh_spectra[idx_outer] = {}
                    if idx_inner not in coh_spectra[idx_outer]:
                        coh_spectra[idx_outer][idx_inner] = {}
                    if comb_names[i] not in coh_spectra[idx_outer][idx_inner]:
                        coh_spectra[idx_outer][idx_inner][comb_names[i]] = {}
                    coh_spectra[idx_outer][idx_inner][comb_names[i]] = (f_coh_, np.array(coh_).mean(axis=0))

            features_median = features.median(axis=0)
            cols_features_py_neuro_add = [c for c in features.columns if c != "time"]
            for col in cols_features_py_neuro_add:
                if "fooof_a_" in col:
                    col_name_median = f"{col}_median"
                else:
                    col_name_median = col
                ch_name_pyneuro_feature = col[col.find("time_domain")+len("time_domain_"):]
                # find the second _
                if "_L_" in ch_name_pyneuro_feature or "_R_" in ch_name_pyneuro_feature:
                    idx_find_hem = ch_name_pyneuro_feature.find("_L_") if "_L_" in ch_name_pyneuro_feature else ch_name_pyneuro_feature.find("_R_")
                    ch_name_pyneuro_feature = ch_name_pyneuro_feature[:idx_find_hem+len("_L")]
                else:
                    idx_add_ = 1
                    ch_name_pyneuro_feature = ch_name_pyneuro_feature[:ch_name_pyneuro_feature.find("_", ch_name_pyneuro_feature.find("_")+idx_add_)]
                
                feature_name = col_name_median[col_name_median.find(ch_name_pyneuro_feature)+len(ch_name_pyneuro_feature)+1:]
                d_features.append({
                    "subject": sub,
                    "channel": ch_name_pyneuro_feature,
                    "idx_outer": idx_outer,
                    "idx_inner": idx_inner,
                    "score": score,
                    "time": time,
                    "feature_name": feature_name,
                    "feature_value": features_median[col],
                })

    d_features_comb = pd.DataFrame(d_features)
    d_features_comb.to_csv(os.path.join(path_out, f"{sub}_features_prep.csv"), index=False)

    with open(os.path.join(path_out, f"{sub}_spectra.pkl"), "wb") as f:
        pickle.dump(spectra_, f)
    with open(os.path.join(path_out, f"{sub}_coh_spectra.pkl"), "wb") as f:
        pickle.dump(coh_spectra, f)

# use joblib to run in parallel^
subs = ["004", "005", "007", "009", "010"] # 
subs = ["011", "012"]  # ["004", "005", "007", "009", "010"]
#compute_sub("011")  # Test with the first file to ensure everything works
Parallel(n_jobs=-1)(delayed(compute_sub)(sub) for sub in subs)

