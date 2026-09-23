import pandas as pd
import numpy as np
import cebra

import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from scipy.stats import permutation_test
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

# df_res = pd.read_csv("results_sudsybocs_ybocs_suds_decoding.csv")
# df_res.groupby("YBOCS_TO_SUDS")["pearson_corr"].mean()
# df_res.groupby("YBOCS_TO_SUDS")["pearson_corr"].std()
# # run a permutation test against zero for each YBOCS_TO_SUDS
# def mean_stat(x):

#     return np.mean(x)

# for ybocs_to_suds in [True, False]:
#     df_res_sub = df_res.query("YBOCS_TO_SUDS == @ybocs_to_suds")
#     corr_vals = df_res_sub["pearson_corr"].dropna().to_numpy()
#     res = permutation_test(
#         (corr_vals,),
#         statistic=mean_stat,
#         permutation_type="samples",
#         alternative="two-sided",
#         n_resamples=10000,
#         random_state=0,
#     )
#     print(f"YBOCS_TO_SUDS {ybocs_to_suds} p-value: {res.pvalue:.4f}")
# YBOCS_TO_SUDS -0.06 pm 0.16, p=0.41
# SUDS_TO_YBOCS -0.09 pm 0.40, p=0.59


l_audio_features = ["Loudness_sma3","alphaRatio_sma3","hammarbergIndex_sma3","slope0-500_sma3","slope500-1500_sma3","spectralFlux_sma3","mfcc1_sma3","mfcc2_sma3","mfcc3_sma3","mfcc4_sma3","F0semitoneFrom27.5Hz_sma3nz","jitterLocal_sma3nz","shimmerLocaldB_sma3nz","HNRdBACF_sma3nz","logRelF0-H1-H2_sma3nz","logRelF0-H1-A3_sma3nz","F1frequency_sma3nz","F1bandwidth_sma3nz","F1amplitudeLogRelF0_sma3nz","F2frequency_sma3nz","F2bandwidth_sma3nz","F2amplitudeLogRelF0_sma3nz","F3frequency_sma3nz","F3bandwidth_sma3nz","F3amplitudeLogRelF0_sma3nz","F0semitoneFrom27.5Hz_sma3nz_amean","F0semitoneFrom27.5Hz_sma3nz_stddevNorm","F0semitoneFrom27.5Hz_sma3nz_percentile20.0","F0semitoneFrom27.5Hz_sma3nz_percentile50.0","F0semitoneFrom27.5Hz_sma3nz_percentile80.0","F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2","F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope","F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope","F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope","F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope","loudness_sma3_amean","loudness_sma3_stddevNorm","loudness_sma3_percentile20.0","loudness_sma3_percentile50.0","loudness_sma3_percentile80.0","loudness_sma3_pctlrange0-2","loudness_sma3_meanRisingSlope","loudness_sma3_stddevRisingSlope","loudness_sma3_meanFallingSlope","loudness_sma3_stddevFallingSlope","spectralFlux_sma3_amean","spectralFlux_sma3_stddevNorm","mfcc1_sma3_amean","mfcc1_sma3_stddevNorm","mfcc2_sma3_amean","mfcc2_sma3_stddevNorm","mfcc3_sma3_amean","mfcc3_sma3_stddevNorm","mfcc4_sma3_amean","mfcc4_sma3_stddevNorm","jitterLocal_sma3nz_amean","jitterLocal_sma3nz_stddevNorm","shimmerLocaldB_sma3nz_amean","shimmerLocaldB_sma3nz_stddevNorm","HNRdBACF_sma3nz_amean","HNRdBACF_sma3nz_stddevNorm","logRelF0-H1-H2_sma3nz_amean","logRelF0-H1-H2_sma3nz_stddevNorm","logRelF0-H1-A3_sma3nz_amean","logRelF0-H1-A3_sma3nz_stddevNorm","F1frequency_sma3nz_amean","F1frequency_sma3nz_stddevNorm","F1bandwidth_sma3nz_amean","F1bandwidth_sma3nz_stddevNorm","F1amplitudeLogRelF0_sma3nz_amean","F1amplitudeLogRelF0_sma3nz_stddevNorm","F2frequency_sma3nz_amean","F2frequency_sma3nz_stddevNorm","F2bandwidth_sma3nz_amean","F2bandwidth_sma3nz_stddevNorm","F2amplitudeLogRelF0_sma3nz_amean","F2amplitudeLogRelF0_sma3nz_stddevNorm","F3frequency_sma3nz_amean","F3frequency_sma3nz_stddevNorm","F3bandwidth_sma3nz_amean","F3bandwidth_sma3nz_stddevNorm","F3amplitudeLogRelF0_sma3nz_amean","F3amplitudeLogRelF0_sma3nz_stddevNorm","alphaRatioV_sma3nz_amean","alphaRatioV_sma3nz_stddevNorm","hammarbergIndexV_sma3nz_amean","hammarbergIndexV_sma3nz_stddevNorm","slopeV0-500_sma3nz_amean","slopeV0-500_sma3nz_stddevNorm","slopeV500-1500_sma3nz_amean","slopeV500-1500_sma3nz_stddevNorm","spectralFluxV_sma3nz_amean","spectralFluxV_sma3nz_stddevNorm","mfcc1V_sma3nz_amean","mfcc1V_sma3nz_stddevNorm","mfcc2V_sma3nz_amean","mfcc2V_sma3nz_stddevNorm","mfcc3V_sma3nz_amean","mfcc3V_sma3nz_stddevNorm","mfcc4V_sma3nz_amean","mfcc4V_sma3nz_stddevNorm","alphaRatioUV_sma3nz_amean","hammarbergIndexUV_sma3nz_amean","slopeUV0-500_sma3nz_amean","slopeUV500-1500_sma3nz_amean","spectralFluxUV_sma3nz_amean","loudnessPeaksPerSec","VoicedSegmentsPerSec","MeanVoicedSegmentLengthSec","StddevVoicedSegmentLengthSec","MeanUnvoicedSegmentLength","StddevUnvoicedSegmentLength","equivalentSoundLevel_dBp","arousal","dominance","valence"] + [f"Dim {i}" for i in range(1024)]# + ["duration"]

INCLUDE_AU = True
INCLUDE_AUDIO = True


df_rs, subs_rs = utils.get_df_features("all", "all", READ_RS=True)
df_suds, subs_suds = utils.get_df_features("all", "all", READ_RS=False)

res_ = []

for YBOCS_TO_SUDS in [True, False]:
    

    if YBOCS_TO_SUDS:
        train_var = "YBOCS II Total Score"
        test_var = "score"
    else:
        train_var = "score"
        test_var = "YBOCS II Total Score"

    for sub in subs_suds:
        if np.isnan(sub) or sub == 8:
            continue

        if YBOCS_TO_SUDS:
            X_sub = df_rs.query("subject == @sub")
            X_sub["session_id"] = (X_sub["subject"].astype(str) + "_" + X_sub["date"].astype(str)).astype("category").cat.codes
            X_neural = X_sub[[c for c in X_sub.columns if c.startswith("SC_") or c.startswith("C_")]]
            X_neural["session_id"] = X_sub["session_id"]
        else:
            X_sub = df_suds.query("subject == @sub")
            X_sub["date"] = X_sub["time"].dt.date
            #X_sub["session_id"] = X_sub["date"].astype("category").cat.codes
            X_sub["session_id"] = X_sub["time"].astype("category").cat.codes
            X_neural = X_sub[[c for c in X_sub.columns if c.startswith("SC_") or c.startswith("C_")]]
            X_neural["session_id"] = X_sub["session_id"]

        cols_include = []
        if INCLUDE_AU:
            cols_include += [c for c in X_sub.columns if c.startswith("AU")]
        if INCLUDE_AUDIO:
            cols_include += l_audio_features

        var_decode = train_var

        cols_include += [var_decode]
        cols_wo_voi = [c for c in cols_include if c != var_decode]          

        Y_fau = X_sub[cols_include]
        Y_fau["session_id"] = X_sub["session_id"]

        if sub in [4, 5, 7]:
            # remove columns that contain C_
            X_neural = X_neural[[c for c in X_neural.columns if not c.startswith("C_") and not "_C_" in c]]

        X_train  = X_neural.drop(columns=["session_id"])
        Y_train  = Y_fau.drop(columns=["session_id"])

        idx_nan = X_train.isna().any(axis=1) | Y_train.isna().any(axis=1)
        X_train = X_train[~idx_nan]
        Y_train = Y_train[~idx_nan]
        x_train_columns = X_train.columns

        if YBOCS_TO_SUDS:
            X_sub_suds = df_suds.query("subject == @sub")
            #X_sub_suds["session_id"] = (X_sub_suds["subject"].astype(str) + "_" + X_sub_suds["date"].astype(str)).astype("category").cat.codes
            X_neural_suds = X_sub_suds[[c for c in X_sub_suds.columns if c.startswith("SC_") or c.startswith("C_")]]
        else:
            X_sub_suds = df_rs.query("subject == @sub")
            #X_sub_suds["session_id"] = (X_sub_suds["subject"].astype(str) + "_" + X_sub_suds["date"].astype(str)).astype("category").cat.codes
            X_neural_suds = X_sub_suds[[c for c in X_sub_suds.columns if c.startswith("SC_") or c.startswith("C_")]]
        
        if sub in [4, 5, 7]:
            # remove columns that contain C_
            X_neural_suds = X_neural_suds[[c for c in X_neural_suds.columns if not c.startswith("C_") and not "_C_" in c]]

        idx_nan = X_neural_suds.isna().any(axis=1)
        X_neural_suds = X_neural_suds[~idx_nan]
        y_neural_suds = X_sub_suds[test_var]
        y_neural_suds = y_neural_suds[~idx_nan]

        
        x_test_columns = X_neural_suds.columns
        columns_join = x_train_columns.intersection(x_test_columns)

        X_neural_suds = X_neural_suds[columns_join]
        X_train = X_train[columns_join]

        cebra_model = cebra.CEBRA(
            model_architecture="offset1-model", 
            batch_size=200,
            learning_rate=3e-3,
            temperature=1.0,
            max_iterations=100,
            distance="cosine",
            conditional="time_delta",
            #output_dimension=15,
            output_dimension=50, # 50
            num_hidden_units=128, # 128 
            verbose=True,
            delta=0.1,
            #hybrid=True if INCLUDE_AU or INCLUDE_AUDIO else False,
        )

        y_test = Y_train[var_decode].values
        Y_train = Y_train.drop(columns=[var_decode])

        # scale X_train
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        cebra_model.fit(X_train, Y_train.values)

        X_emb_train = cebra_model.transform(X_train)

        downstream_model = linear_model.LinearRegression()

        downstream_model.fit(X_emb_train, y_test)

        X_neural_suds = X_neural_suds[columns_join]
        X_neural_suds = scaler.transform(X_neural_suds)

        X_emb_suds = cebra_model.transform(X_neural_suds)

        y_pred_suds = downstream_model.predict(X_emb_suds)

        idx_nan = np.isnan(y_pred_suds) | np.isnan(y_neural_suds.values)
        y_pred_suds = y_pred_suds[~idx_nan]
        y_neural_suds = y_neural_suds[~idx_nan]
        corr_val, p_val = pearsonr(y_neural_suds.values, y_pred_suds)
        res_.append({
            "subject" : sub,
            "YBOCS_TO_SUDS" : YBOCS_TO_SUDS,
            "pearson_corr" : corr_val,
            "pearson_pval" : p_val,
        })

df_res = pd.DataFrame(res_)

# make a barplot for each subject, showing the pearson correlation for YBOCS_TO_SUDS True and False as hue,
# show significance p<0.05 as a star on top of the bar
plt.figure(figsize=(5, 3))

ax = sns.barplot(
    data=df_res,
    x="subject",
    y="pearson_corr",
    hue="YBOCS_TO_SUDS",
    errorbar=None,  # ci=None is deprecated
)

# Add significance stars
for patch, (_, row) in zip(ax.patches, df_res.iterrows()):
    if row["pearson_pval"] < 0.05:
        x = patch.get_x() + patch.get_width() / 2
        y = patch.get_height()

        ax.text(
            x,
            y + 0.02,
            "*",
            ha="center",
            va="bottom",
            color="red",
            fontsize=20,
        )

plt.title("Pearson correlation between predicted and true values for each subject")
plt.ylabel("Pearson correlation")
plt.xlabel("Subject")
plt.legend(title="YBOCS_TO_SUDS")

sns.despine()
plt.tight_layout()
df_res.to_csv("results_sudsybocs_ybocs_suds_decoding.csv", index=False)
plt.savefig("results_sudsybocs_ybocs_suds_decoding.pdf")
# plot the resulots
y_true_ybocssuds = {
    "4" : 38,
    "5" : 22,
    "7" : 32,
    "9" : 30,
    "10" : 8,
    "11" : 33,
    "12" : 35
}
plt.figure()
# boxplot of predictions for each subject
sns.boxplot(data=pred_)
sns.swarmplot(data=pred_, color=".25")
# x ticks are subject ids
plt.xticks(ticks=np.arange(len(subs_suds)), labels=subs_suds)
plt.xlabel("Subject")
plt.ylabel("Predicted YBOCS II Total Score")
plt.title("Predicted YBOCS II Total Score for each subject")
# plot as stars the true YBOCS II Total Score for each subject
for i, sub in enumerate(subs_suds):
    if str(sub) in y_true_ybocssuds:
        plt.plot(i, y_true_ybocssuds[str(sub)], marker="*", color="red", markersize=10, label="True YBOCS II Total Score" if i == 0 else "")
sns.despine()
plt.show()
        

