import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import linear_model, model_selection
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import MultiTaskElasticNet, MultiTaskElasticNetCV
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

l_audio_features = ["Loudness_sma3","alphaRatio_sma3","hammarbergIndex_sma3","slope0-500_sma3","slope500-1500_sma3","spectralFlux_sma3","mfcc1_sma3","mfcc2_sma3","mfcc3_sma3","mfcc4_sma3","F0semitoneFrom27.5Hz_sma3nz","jitterLocal_sma3nz","shimmerLocaldB_sma3nz","HNRdBACF_sma3nz","logRelF0-H1-H2_sma3nz","logRelF0-H1-A3_sma3nz","F1frequency_sma3nz","F1bandwidth_sma3nz","F1amplitudeLogRelF0_sma3nz","F2frequency_sma3nz","F2bandwidth_sma3nz","F2amplitudeLogRelF0_sma3nz","F3frequency_sma3nz","F3bandwidth_sma3nz","F3amplitudeLogRelF0_sma3nz","F0semitoneFrom27.5Hz_sma3nz_amean","F0semitoneFrom27.5Hz_sma3nz_stddevNorm","F0semitoneFrom27.5Hz_sma3nz_percentile20.0","F0semitoneFrom27.5Hz_sma3nz_percentile50.0","F0semitoneFrom27.5Hz_sma3nz_percentile80.0","F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2","F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope","F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope","F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope","F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope","loudness_sma3_amean","loudness_sma3_stddevNorm","loudness_sma3_percentile20.0","loudness_sma3_percentile50.0","loudness_sma3_percentile80.0","loudness_sma3_pctlrange0-2","loudness_sma3_meanRisingSlope","loudness_sma3_stddevRisingSlope","loudness_sma3_meanFallingSlope","loudness_sma3_stddevFallingSlope","spectralFlux_sma3_amean","spectralFlux_sma3_stddevNorm","mfcc1_sma3_amean","mfcc1_sma3_stddevNorm","mfcc2_sma3_amean","mfcc2_sma3_stddevNorm","mfcc3_sma3_amean","mfcc3_sma3_stddevNorm","mfcc4_sma3_amean","mfcc4_sma3_stddevNorm","jitterLocal_sma3nz_amean","jitterLocal_sma3nz_stddevNorm","shimmerLocaldB_sma3nz_amean","shimmerLocaldB_sma3nz_stddevNorm","HNRdBACF_sma3nz_amean","HNRdBACF_sma3nz_stddevNorm","logRelF0-H1-H2_sma3nz_amean","logRelF0-H1-H2_sma3nz_stddevNorm","logRelF0-H1-A3_sma3nz_amean","logRelF0-H1-A3_sma3nz_stddevNorm","F1frequency_sma3nz_amean","F1frequency_sma3nz_stddevNorm","F1bandwidth_sma3nz_amean","F1bandwidth_sma3nz_stddevNorm","F1amplitudeLogRelF0_sma3nz_amean","F1amplitudeLogRelF0_sma3nz_stddevNorm","F2frequency_sma3nz_amean","F2frequency_sma3nz_stddevNorm","F2bandwidth_sma3nz_amean","F2bandwidth_sma3nz_stddevNorm","F2amplitudeLogRelF0_sma3nz_amean","F2amplitudeLogRelF0_sma3nz_stddevNorm","F3frequency_sma3nz_amean","F3frequency_sma3nz_stddevNorm","F3bandwidth_sma3nz_amean","F3bandwidth_sma3nz_stddevNorm","F3amplitudeLogRelF0_sma3nz_amean","F3amplitudeLogRelF0_sma3nz_stddevNorm","alphaRatioV_sma3nz_amean","alphaRatioV_sma3nz_stddevNorm","hammarbergIndexV_sma3nz_amean","hammarbergIndexV_sma3nz_stddevNorm","slopeV0-500_sma3nz_amean","slopeV0-500_sma3nz_stddevNorm","slopeV500-1500_sma3nz_amean","slopeV500-1500_sma3nz_stddevNorm","spectralFluxV_sma3nz_amean","spectralFluxV_sma3nz_stddevNorm","mfcc1V_sma3nz_amean","mfcc1V_sma3nz_stddevNorm","mfcc2V_sma3nz_amean","mfcc2V_sma3nz_stddevNorm","mfcc3V_sma3nz_amean","mfcc3V_sma3nz_stddevNorm","mfcc4V_sma3nz_amean","mfcc4V_sma3nz_stddevNorm","alphaRatioUV_sma3nz_amean","hammarbergIndexUV_sma3nz_amean","slopeUV0-500_sma3nz_amean","slopeUV500-1500_sma3nz_amean","spectralFluxUV_sma3nz_amean","loudnessPeaksPerSec","VoicedSegmentsPerSec","MeanVoicedSegmentLengthSec","StddevVoicedSegmentLengthSec","MeanUnvoicedSegmentLength","StddevUnvoicedSegmentLength","equivalentSoundLevel_dBp","arousal","dominance","valence"] + [f"Dim {i}" for i in range(1024)] + ["duration"]
PATH_SUDS = "plotting/Figure4/source_data/suds_audio_neural_features_combined.csv"
PATH_RS = "plotting/Figure1/source_data/ybocs_audio_neural_features_combined_1.csv"


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
        df_features = df_features.drop(columns=["subject"])
        df_features = df_features.rename(columns={"sub": "subject"})
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


def get_df_features_video_neural_comb(region: str, feature: str, ):
    #if GROUP_NEURAL_AND_VIDEO:
    df_features = pd.read_csv("plotting/Figure1/source_data/neural_features_suds.csv")
    #df_merged = pd.read_csv("neural_audio_fau_combined.csv")

    df_features["time"] = pd.to_datetime(df_features["time"])
    #if GROUP_NEURAL_AND_VIDEO:
    # for sub 009 remove 5 hours from time, for sub 010 remove 6 hours
    df_features.loc[df_features["subject"] == 9, "time"] -= pd.Timedelta(hours=5)
    df_features.loc[df_features["subject"] == 10, "time"] -= pd.Timedelta(hours=6)
    df_features.loc[df_features["subject"] == 12, "time"] -= pd.Timedelta(hours=6)
    df_features.loc[df_features["subject"] == 11, "time"] -= pd.Timedelta(hours=6)

    df_faus = pd.read_csv("FAU_features.csv")
    df_faus["time"] = pd.to_datetime(df_faus["time"])   

    if region != "all":
        cols_neural_features = [c for c in df_features.columns if c.startswith(region) or c == "score" or c == "score_normed" or c == "subject" or c == "time"]  # c not in ["subject", "time", "score", "score_normed"] and 
    else:
        cols_neural_features = df_features.columns.tolist()
    df_merged = pd.merge(df_features[cols_neural_features], df_faus, on=["subject", "time"], suffixes=("_feat", "_fau"))
    # remove columns that start with C_
    #df_merged = df_merged[[c for c in df_merged.columns if not c.startswith("C_")]]
    # if col starts with SC_L_, then keep only ones that contain fft, but not fft_psd
    if feature == "fft_only":
        if region != "all":
            df_merged = df_merged[[c for c in df_merged.columns if not c.startswith(region) or "fft" in c and "fft_psd" not in c]]
        else:
            df_merged = df_merged[[c for c in df_merged.columns if "fft" in c and "fft_psd" not in c or c == "subject" or c == "score_fau" or c == "score_normed" or c == "time" or c.startswith("AU") or c in l_audio_features]]

    elif feature != "all":
        if region != "all":
            df_merged = df_merged[[c for c in df_merged.columns if not c.startswith(region) or feature in c]]
        else:
            df_merged = df_merged[[c for c in df_merged.columns if feature in c or c == "subject" or c == "score_fau" or c == "score_normed" or c == "time" or c.startswith("AU") or c in l_audio_features]]
    # if region starts with C_, remove subjects 4, 5, 7
    subs = df_merged["subject"].unique()
    if region.startswith("C_"):
        subs = [s for s in subs if s not in [4, 5, 7]]
        df_merged = df_merged[df_merged["subject"].isin(subs)]
    return df_merged, subs

def get_train_data(df_merged, sub, INCLUDE_AU: bool = True, INCLUDE_AUDIO: bool = False, INCLUDE_SUDS: bool = True):
    X_sub = df_merged.query("subject == @sub")
    X_sub["date"] = X_sub["time"].dt.date
    X_sub["session_id"] = X_sub["date"].astype("category").cat.codes
    X_neural = X_sub[[c for c in X_sub.columns if c.startswith("SC_") or c.startswith("C_")]]
    X_neural["session_id"] = X_sub["session_id"]
    if INCLUDE_AU is True and INCLUDE_AUDIO is False:
        Y_label = X_sub[[c for c in X_sub.columns if c.startswith("AU") or c == "score"]]
    elif INCLUDE_AUDIO is True and INCLUDE_AU is False:
        Y_label = X_sub[[c for c in X_sub.columns if c in l_audio_features or c == "score"]]
    elif INCLUDE_AUDIO is True and INCLUDE_AU is True:
        Y_label = X_sub[["score"] + l_audio_features + [c for c in X_sub.columns if c.startswith("AU")]]
    else:
        Y_label = X_sub[["score"]]

    Y_label["session_id"] = X_sub["session_id"]

    if sub in [4, 5, 7]:
        # remove columns that contain C_
        X_neural = X_neural[[c for c in X_neural.columns if not c.startswith("C_") and not "_C_" in c]]

    if INCLUDE_SUDS is False:
        Y_label = Y_label.drop(columns=["score"])
    return X_neural, Y_label

def plot_example_predictions(y_pred_direct_sub, y_true_sub, sub, cca_dim, pearson_per_col_direct):
    # sub 12, cca_dims 7
    plt.figure(figsize=(5, 4.5))
    plt.subplot(1, 2, 1)
    plt.plot(stats.zscore(y_pred_direct_sub[:, 0]), label="pred direct")
    plt.plot(stats.zscore(y_true_sub[:, 0]), label="true")
    plt.xlabel("SUDS sample")
    plt.ylabel("SUDS (z-scored)")
    plt.legend()
    sns.despine()
    plt.title(f"sub: {sub} cca dims: {cca_dim} {pearson_per_col_direct[0]:.2f}")
    plt.subplot(1, 2, 2)
    sns.regplot(x=y_true_sub[:, 0], y=y_pred_direct_sub[:, 0])
    sns.despine()
    plt.xlabel("true")
    plt.ylabel("pred direct")
    plt.savefig(f"cca_eval/fig_cca_sub{sub}_dims{cca_dim}_direct.pdf")

def get_per(Y_test, Y_pred):
    r2_per_col = r2_score(Y_test, Y_pred, multioutput='raw_values')
    if Y_test.shape[1] == 1:
        Y_test_ = Y_test[:,0]
        Y_pred_ = Y_pred[:,0]
        mse_per_col = [mean_squared_error(Y_test_, Y_pred_)]
        r, p_val = pearsonr(Y_test_, Y_pred_)
        pearson_per_col = [r]
        p_val = [p_val]
        return r2_per_col, mse_per_col, pearson_per_col, p_val

    else:

        mse_per_col = ((Y_test - Y_pred) ** 2).mean(axis=0)

        # Pearson r per target (optional)
        pearson_per_col = []
        p_val = []
        for j in range(Y_test.shape[1]):
            try:
                if Y_test.shape[1] == 1:
                    r, p_val_ = pearsonr(Y_test[:, 0], Y_pred)
                else:
                    r, p_val_ = pearsonr(Y_test[:, j], Y_pred[:, j])
            except Exception:
                r = np.nan
                p_val_ = np.nan
            pearson_per_col.append(r)
            p_val.append(p_val_)
    return r2_per_col, mse_per_col, pearson_per_col, p_val

def compute_pers(df_merged,
                subs,
                cca_dims: list = [1, 2, 3, 4, 5, 6],
                Z_SCORE: bool = False,
                INCLUDE_AU: bool = True,
                INCLUDE_AUDIO: bool = False,
                INCLUDE_SUDS: bool = True,
                SHUFFLE = False,
                REPORT_SESSION: bool = False,
                GET_ONLY_SUDS: bool = True
):

    df_res = []
    for sub in subs:
        for cca_dim in cca_dims:
            X_neural, Y_label = get_train_data(df_merged, sub,
                                           INCLUDE_AU=INCLUDE_AU,
                                           INCLUDE_AUDIO=INCLUDE_AUDIO,
                                           INCLUDE_SUDS=INCLUDE_SUDS)
            if sub in [4, 5, 7]:
                X_neural = X_neural[[c for c in X_neural.columns if not c.startswith("C_") and not "_C_" in c]]

            y_true_sub = []
            y_pred_sub = []
            for test_sess_id in X_neural["session_id"].unique():
                X_train = X_neural[X_neural["session_id"] != test_sess_id]
                X_test = X_neural[X_neural["session_id"] == test_sess_id]
                Y_train_label = Y_label[Y_label["session_id"] != test_sess_id]
                Y_test_label = Y_label[Y_label["session_id"] == test_sess_id]

                X_train = X_train.drop(columns=["session_id"])
                Y_train = Y_train_label.drop(columns=["session_id"])
                X_test = X_test.drop(columns=["session_id"])
                Y_test = Y_test_label.drop(columns=["session_id"])

                # drop rows with NaNs
                idx_nan = X_train.isna().any(axis=1) | Y_train.isna().any(axis=1)
                X_train = X_train[~idx_nan]
                Y_train = Y_train[~idx_nan]
                idx_nan_test = X_test.isna().any(axis=1) | Y_test.isna().any(axis=1)
                if sum(idx_nan_test) == len(idx_nan_test):
                    continue
                X_test = X_test[~idx_nan_test]
                Y_test = Y_test[~idx_nan_test]
                score_col = "score"
                score_idx = Y_train.columns.get_loc(score_col)

                if Z_SCORE is True:
                    x_scaler = StandardScaler().fit(X_train)
                    y_scaler = StandardScaler().fit(Y_train)

                    Xtr = x_scaler.transform(X_train)
                    Ytr = y_scaler.transform(Y_train)
                    Yte = y_scaler.transform(Y_test)
                    Xte = x_scaler.transform(X_test)
                else:
                    Xtr = X_train.values
                    Ytr = Y_train.values
                    Xte = X_test.values
                    Yte = Y_test.values

                cca = CCA(n_components=cca_dim, max_iter=500, scale=False)
                try:
                    cca.fit(Xtr, Ytr)
                except Exception as e:
                    continue

                B_inv = np.linalg.pinv(cca.y_weights_)

                Y_pred = Xte @ cca.x_weights_ @ B_inv

                if Z_SCORE is True:
                    Y_pred = y_scaler.inverse_transform(Y_pred)

                y_true_sub.append(Yte)
                y_pred_sub.append(Y_pred)

                if REPORT_SESSION:
                    if SHUFFLE is True:
                        # shuffle the true values
                        np.random.shuffle(Yte)
                    if GET_ONLY_SUDS:
                        Yte = Yte[:, score_idx].reshape(-1, 1)
                        Y_pred = Y_pred[:, score_idx].reshape(-1, 1)
                
                    if not INCLUDE_AU and not INCLUDE_AUDIO:
                        Y_pred = Y_pred[:, 0]

                    if Y_pred.shape[0] < 3:
                        continue
                    r2_per_col_direct, mse_per_col_direct, pearson_per_col_direct, p_val_direct = get_per(Yte, Y_pred)
                    df_add_sess = pd.DataFrame({
                        "target": Y_train.columns if not GET_ONLY_SUDS else [score_col],
                        "R2": r2_per_col_direct,
                        "MSE": mse_per_col_direct,
                        "Pearson_r": pearson_per_col_direct,
                        "p_val": p_val_direct,
                        "subject": sub,
                        "cca_dims": cca_dim,
                        "test_session_id": test_sess_id,
                        "sample_size" : len(Yte)
                    })
                    df_res.append(df_add_sess)

            if REPORT_SESSION is False:
                if len(y_true_sub) == 0:
                    continue
                y_true_sub = np.concatenate(y_true_sub)
                if INCLUDE_AU or INCLUDE_AUDIO:
                    y_pred_sub = np.vstack(y_pred_sub)
                else:
                    y_pred_sub = np.concatenate(y_pred_sub)[:, 0]

                if SHUFFLE is True:
                    # shuffle the true values
                    np.random.shuffle(y_true_sub)
                r2_per_col, mse_per_col, pearson_per_col, p_val = get_per(y_true_sub, y_pred_sub)

                df_add = pd.DataFrame({
                    "target": Y_train.columns,
                    "R2": r2_per_col,
                    "MSE": mse_per_col,
                    "Pearson_r": pearson_per_col,
                    "p_val": p_val,
                    "subject": sub,
                    "cca_dims": cca_dim
                })
                df_res.append(df_add)

    if len(df_res) > 0:
        df_res = pd.concat(df_res).reset_index(drop=True)
    else:
        df_res = None

    return df_res