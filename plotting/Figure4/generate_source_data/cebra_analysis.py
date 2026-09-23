import pandas as pd
import numpy as np
import cebra

import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import linear_model, model_selection
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
from sklearn import linear_model
from sklearn.linear_model import MultiTaskElasticNet, MultiTaskElasticNetCV
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
from joblib import Parallel, delayed
from sklearn.neighbors import KNeighborsRegressor
import pickle
import utils
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import sys
import os
import argparse
import random
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

l_audio_features = ["Loudness_sma3", "alphaRatio_sma3", "hammarbergIndex_sma3", "slope0-500_sma3", "slope500-1500_sma3", "spectralFlux_sma3","mfcc1_sma3","mfcc2_sma3","mfcc3_sma3","mfcc4_sma3","F0semitoneFrom27.5Hz_sma3nz","jitterLocal_sma3nz","shimmerLocaldB_sma3nz","HNRdBACF_sma3nz","logRelF0-H1-H2_sma3nz","logRelF0-H1-A3_sma3nz","F1frequency_sma3nz","F1bandwidth_sma3nz","F1amplitudeLogRelF0_sma3nz","F2frequency_sma3nz","F2bandwidth_sma3nz","F2amplitudeLogRelF0_sma3nz","F3frequency_sma3nz","F3bandwidth_sma3nz","F3amplitudeLogRelF0_sma3nz","F0semitoneFrom27.5Hz_sma3nz_amean","F0semitoneFrom27.5Hz_sma3nz_stddevNorm","F0semitoneFrom27.5Hz_sma3nz_percentile20.0","F0semitoneFrom27.5Hz_sma3nz_percentile50.0","F0semitoneFrom27.5Hz_sma3nz_percentile80.0","F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2","F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope","F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope","F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope","F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope","loudness_sma3_amean","loudness_sma3_stddevNorm","loudness_sma3_percentile20.0","loudness_sma3_percentile50.0","loudness_sma3_percentile80.0","loudness_sma3_pctlrange0-2","loudness_sma3_meanRisingSlope","loudness_sma3_stddevRisingSlope","loudness_sma3_meanFallingSlope","loudness_sma3_stddevFallingSlope","spectralFlux_sma3_amean","spectralFlux_sma3_stddevNorm","mfcc1_sma3_amean","mfcc1_sma3_stddevNorm","mfcc2_sma3_amean","mfcc2_sma3_stddevNorm","mfcc3_sma3_amean","mfcc3_sma3_stddevNorm","mfcc4_sma3_amean","mfcc4_sma3_stddevNorm","jitterLocal_sma3nz_amean","jitterLocal_sma3nz_stddevNorm","shimmerLocaldB_sma3nz_amean","shimmerLocaldB_sma3nz_stddevNorm","HNRdBACF_sma3nz_amean","HNRdBACF_sma3nz_stddevNorm","logRelF0-H1-H2_sma3nz_amean","logRelF0-H1-H2_sma3nz_stddevNorm","logRelF0-H1-A3_sma3nz_amean","logRelF0-H1-A3_sma3nz_stddevNorm","F1frequency_sma3nz_amean","F1frequency_sma3nz_stddevNorm","F1bandwidth_sma3nz_amean","F1bandwidth_sma3nz_stddevNorm","F1amplitudeLogRelF0_sma3nz_amean","F1amplitudeLogRelF0_sma3nz_stddevNorm","F2frequency_sma3nz_amean","F2frequency_sma3nz_stddevNorm","F2bandwidth_sma3nz_amean","F2bandwidth_sma3nz_stddevNorm","F2amplitudeLogRelF0_sma3nz_amean","F2amplitudeLogRelF0_sma3nz_stddevNorm","F3frequency_sma3nz_amean","F3frequency_sma3nz_stddevNorm","F3bandwidth_sma3nz_amean","F3bandwidth_sma3nz_stddevNorm","F3amplitudeLogRelF0_sma3nz_amean","F3amplitudeLogRelF0_sma3nz_stddevNorm","alphaRatioV_sma3nz_amean","alphaRatioV_sma3nz_stddevNorm","hammarbergIndexV_sma3nz_amean","hammarbergIndexV_sma3nz_stddevNorm","slopeV0-500_sma3nz_amean","slopeV0-500_sma3nz_stddevNorm","slopeV500-1500_sma3nz_amean","slopeV500-1500_sma3nz_stddevNorm","spectralFluxV_sma3nz_amean","spectralFluxV_sma3nz_stddevNorm","mfcc1V_sma3nz_amean","mfcc1V_sma3nz_stddevNorm","mfcc2V_sma3nz_amean","mfcc2V_sma3nz_stddevNorm","mfcc3V_sma3nz_amean","mfcc3V_sma3nz_stddevNorm","mfcc4V_sma3nz_amean","mfcc4V_sma3nz_stddevNorm","alphaRatioUV_sma3nz_amean","hammarbergIndexUV_sma3nz_amean","slopeUV0-500_sma3nz_amean","slopeUV500-1500_sma3nz_amean","spectralFluxUV_sma3nz_amean","loudnessPeaksPerSec","VoicedSegmentsPerSec","MeanVoicedSegmentLengthSec","StddevVoicedSegmentLengthSec","MeanUnvoicedSegmentLength","StddevUnvoicedSegmentLength","equivalentSoundLevel_dBp","arousal","dominance","valence"] + [f"Dim {i}" for i in range(1024)]# + ["duration"]

if len(sys.argv) == 1:  # only file
    DEBUG = True
else:
    DEBUG = False

def run_sub(df_merged, sub, region, SUDS=True, INCLUDE_AU=True, INCLUDE_AUDIO=True, NORMALIZE=False,
            RUN_SUDS: bool = False, SHUFFLE: bool = False):

    X_sub = df_merged.query("subject == @sub")

    if RUN_SUDS:
        X_sub["date"] = X_sub["time"].dt.date
        #X_sub["session_id"] = X_sub["date"].astype("category").cat.codes
        X_sub["session_id"] = X_sub["time"].astype("category").cat.codes  # this line makes each sample a different session
        X_neural = X_sub[[c for c in X_sub.columns if c.startswith("SC_") or c.startswith("C_")]]
        X_neural["session_id"] = X_sub["session_id"]

    else:
        X_sub["session_id"] = (X_sub["subject"].astype(str) + "_" + X_sub["date"].astype(str)).astype("category").cat.codes
        X_neural = X_sub[[c for c in X_sub.columns if c.startswith("SC_") or c.startswith("C_")]]
        X_neural["session_id"] = X_sub["session_id"]

    cols_include = []
    if INCLUDE_AU:
        cols_include += [c for c in X_sub.columns if c.startswith("AU")]
    if INCLUDE_AUDIO:
        cols_include += l_audio_features
    if SUDS:
        if RUN_SUDS:
            var_decode = 'score_feat'
        else:
            var_decode = 'YBOCS II Total Score'
        cols_include += [var_decode]
        cols_wo_voi = [c for c in cols_include if c != var_decode]          

    Y_fau = X_sub[cols_include]
    Y_fau["session_id"] = X_sub["session_id"]

    if sub in [4, 5, 7]:
        X_neural = X_neural[[c for c in X_neural.columns if not c.startswith("C_") and not "_C_" in c]]
    
    y_true_sub = []
    y_pred_sub = []
    for test_sess_id in X_sub["session_id"].unique():
        X_train = X_neural[X_neural["session_id"] != test_sess_id]
        X_test = X_neural[X_neural["session_id"] == test_sess_id]
        Y_train = Y_fau[Y_fau["session_id"] != test_sess_id]
        Y_test = Y_fau[Y_fau["session_id"] == test_sess_id]

        X_train  = X_train.drop(columns=["session_id"])
        Y_train  = Y_train.drop(columns=["session_id"])
        X_test = X_test.drop(columns=["session_id"])
        Y_test = Y_test.drop(columns=["session_id"])

        idx_nan = X_train.isna().any(axis=1) | Y_train.isna().any(axis=1)
        X_train = X_train[~idx_nan]
        Y_train = Y_train[~idx_nan]
        idx_nan_test = X_test.isna().any(axis=1) | Y_test.isna().any(axis=1)
        if sum(idx_nan_test) == len(idx_nan_test):
            continue
        X_test = X_test[~idx_nan_test]
        Y_test = Y_test[~idx_nan_test]
        idx_var_decode = Y_test.columns.get_loc(var_decode)
        
        if NORMALIZE:
            scaler_X = StandardScaler()
            scaler_X.fit(X_train)
            X_train = pd.DataFrame(scaler_X.transform(X_train), columns=X_train.columns, index=X_train.index)
            X_test = pd.DataFrame(scaler_X.transform(X_test), columns=X_test.columns, index=X_test.index)

            scaler_Y = StandardScaler()
            Y_train = pd.DataFrame(scaler_Y.fit_transform(Y_train), columns=Y_train.columns, index=Y_train.index)
            Y_test = pd.DataFrame(scaler_Y.transform(Y_test), columns=Y_test.columns, index=Y_test.index)

        try:
            if  INCLUDE_AU or INCLUDE_AUDIO:
                conditional = "delta" #  for YBOCS
                conditional = "time_delta"
                Y_use_cebra = Y_train[cols_wo_voi].values
            else:
                conditional = "time"

            if RUN_SUDS:
                cebra_model = cebra.CEBRA(
                    model_architecture="offset1-model", 
                    batch_size=200,
                    learning_rate=3e-3,
                    temperature=1.0,
                    max_iterations=100,
                    distance="cosine",
                    conditional=conditional,
                    output_dimension=50,
                    num_hidden_units=128,
                    verbose=True,
                    delta=0.1,
                )
            else:
                cebra_model = cebra.CEBRA(
                    model_architecture="offset1-model", 
                    batch_size=50,
                    learning_rate=3e-3,
                    temperature=1.0,
                    max_iterations=100,
                    distance="cosine",
                    conditional=conditional,
                    verbose=True,
                    delta=0.1,
                )
        except:
            continue
        if conditional == "time":
           cebra_model.fit(X_train.values)
        else:
            cebra_model.fit(X_train.values, Y_use_cebra)

        X_emb_train = cebra_model.transform(X_train.values)
        X_emb_test = cebra_model.transform(X_test.values)

        downstream_model = linear_model.LinearRegression()

        if SUDS is True:
            downstream_model.fit(X_emb_train, Y_train[var_decode].values)
        else:
            downstream_model.fit(X_emb_train, Y_train.values)

        Y_te_pred = downstream_model.predict(X_emb_test)

        y_true_sub.append(Y_test.values)
        y_pred_sub.append(Y_te_pred)

        #fig = cebra.plot_embedding(X_emb_train, embedding_labels=Y_train.values[:,0], title = "CEBRA-Time (full)", markersize=3, cmap = "rainbow")
        #fig.show()
        # plot the loss curve
        #ax = cebra.plot_loss(cebra_model)
    
    if len(y_true_sub) == 0:
        print(f"No data for sub {sub}, SUDS {SUDS}, INCLUDE_AU {INCLUDE_AU}, INCLUDE_AUDIO {INCLUDE_AUDIO}")
        return pd.DataFrame()
    y_true_sub = np.concatenate(y_true_sub)
    y_pred_sub = np.concatenate(y_pred_sub)

    if SHUFFLE:
        np.random.shuffle(y_true_sub)
    df_res = []
    for i, col in tqdm(enumerate(Y_train.columns)):
        if col != var_decode:
            continue
        r, p = stats.pearsonr(y_true_sub[:, idx_var_decode], y_pred_sub) #
        
        df_res.append({"subject": sub, "AU": col, "r": r, "p": p})

    df_res = pd.DataFrame(df_res)
    df_res["SUDS"] = SUDS
    df_res["INCLUDE_AU"] = INCLUDE_AU
    df_res["INCLUDE_AUDIO"] = INCLUDE_AUDIO
    df_res["RUN_SUDS"] = RUN_SUDS
    df_res["region"] = region
    df_res["SHUFFLE"] = SHUFFLE

    return df_res, y_true_sub, y_pred_sub

if __name__ == "__main__":

    l_all = []
    dict_predictions = {}
    #for SHUFFLE in [True, False]:
    for RUN_SUDS in [False, ]: # True
        for region in ["all", "SC", "C",]:
            df_merged, subs = utils.get_df_features(region, "all", READ_RS=not RUN_SUDS)
            subs = [int(sub) for sub in subs if not pd.isna(sub)]

            pers_ = []
            combination_list = []

            for SHUFFLE in [True, False]:
                for sub in subs:
                    combination_list.append({
                        "sub": sub,
                        "SUDS": True,
                        "INCLUDE_AU": True,
                        "INCLUDE_AUDIO": True,
                        "SHUFFLE": SHUFFLE,
                    })

                    combination_list.append({
                        "sub": sub,
                        "SUDS": True,
                        "INCLUDE_AU": False,
                        "INCLUDE_AUDIO": True,
                        "SHUFFLE": SHUFFLE,

                    })

                    combination_list.append({
                        "sub": sub,
                        "SUDS": True,
                        "INCLUDE_AU": True,
                        "INCLUDE_AUDIO": False,
                        "SHUFFLE": SHUFFLE,

                    })

                    combination_list.append({
                        "sub": sub,
                        "SUDS": True,
                        "INCLUDE_AU": False,
                        "INCLUDE_AUDIO": False,
                        "SHUFFLE": SHUFFLE,
                    })

            df_comb = pd.DataFrame(combination_list)

            num_combinations = len(combination_list)

            # debugging
            idx_run = 10
            d, ytr, ypr = run_sub(df_merged, sub=combination_list[idx_run]["sub"],
                    region=region,
                    SUDS=combination_list[idx_run]["SUDS"],
                    INCLUDE_AU=combination_list[idx_run]["INCLUDE_AU"],
                    INCLUDE_AUDIO=combination_list[idx_run]["INCLUDE_AUDIO"],
                    RUN_SUDS=RUN_SUDS,
                    NORMALIZE=True,
                    SHUFFLE=False,
                )

            res = Parallel(n_jobs=-1)(delayed(run_sub)(
                df_merged, 
                sub=combination_list[idx_run]["sub"],
                region=region,
                SUDS=combination_list[idx_run]["SUDS"],
                INCLUDE_AU=combination_list[idx_run]["INCLUDE_AU"],
                INCLUDE_AUDIO=combination_list[idx_run]["INCLUDE_AUDIO"],
                RUN_SUDS=RUN_SUDS,
                NORMALIZE=True,
                SHUFFLE=combination_list[idx_run]["SHUFFLE"],
            ) for idx_run in tqdm(range(num_combinations)))

            df_res_all = []
            ytr_all = []
            ypr_all = []
            for r in res:
                if r is None or r[0].empty:
                    continue
                df_res_sub, ytr_sub, ypr_sub = r
                df_res_all.append(df_res_sub)
                ytr_all.append(ytr_sub)
                ypr_all.append(ypr_sub)

            df_res_all_ = pd.concat(df_res_all, ignore_index=True)
            df_res_all_["RS"] = not RUN_SUDS
            l_all.append(df_res_all_)

            dict_predictions[(region, RUN_SUDS)] = {
                "y_true": ytr_all,
                "y_pred": ypr_all,
                "df_res": df_res_all_,
            }

        df_final = pd.concat(l_all, ignore_index=True)
        if RUN_SUDS:
            df_final.to_csv(f"plotting/Figure4/source_data/suds_decoding_results_neural_audio_fau.csv", index=False)
            with open(f"plotting/Figure4/suds_decoding_results_neural_audio_fau.pkl", "wb") as f:
                pickle.dump(dict_predictions, f)
        else:
            df_final.to_csv(f"plotting/Figure4/source_data/ybocs_decoding_results_neural_audio_fau.csv", index=False)
            with open(f"plotting/Figure4/ybocs_decoding_results_neural_audio_fau.pkl", "wb") as f:
                pickle.dump(dict_predictions, f)

    # s1 = df_final.query("condition == 'SUDS+AU+Audio' and SHUFFLE == True")
    # s2 = df_final.query("condition == 'SUDS+AU+Audio' and SHUFFLE == False")
    # # run a relative permutation test based on sign differences

    # from scipy import stats
    # r_1 = s1["r"]
    # r_2 = s2["r"]
    # r_ = r_2.values - r_1.values
    # r_z = np.zeros_like(r_)
    # print(stats.permutation_test((r_, r_z), statistic=lambda x, y: np.mean(x) - np.mean(y), n_resamples=10000, alternative='two-sided'))

    # from scipy import stats
    # r_1 = df_final.query("AU == 'score_feat' and condition == 'SUDS+AU+Audio' and RS == False and region == 'all'")["r"]
    # r_2 = df_final.query("AU == 'score_feat' and condition == 'SUDS only' and RS == False  and region == 'all'")["r"]
    # r_ = r_1.values-r_2.values
    # r_z = np.zeros_like(r_)
    # print(stats.permutation_test((r_, r_z), statistic=lambda x, y: np.mean(x) - np.mean(y), n_resamples=10000, alternative='two-sided'))

    # from scipy import stats
    # r_1 = df_final.query("AU == 'YBOCS II Total Score' and condition == 'SUDS+AU+Audio' and RS == True and region == 'all'")["r"]
    # r_2 = df_final.query("AU == 'YBOCS II Total Score' and condition == 'SUDS only' and RS == True  and region == 'all'")["r"]
    # r_ = r_1.values-r_2.values
    # r_z = np.zeros_like(r_)
    # print(stats.permutation_test((r_, r_z), statistic=lambda x, y: np.mean(x) - np.mean(y), n_resamples=10000, alternative='two-sided'))

