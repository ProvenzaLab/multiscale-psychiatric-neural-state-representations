import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
from scipy.stats import zscore

l_audio_features = ["Loudness_sma3","alphaRatio_sma3","hammarbergIndex_sma3","slope0-500_sma3","slope500-1500_sma3","spectralFlux_sma3","mfcc1_sma3","mfcc2_sma3","mfcc3_sma3","mfcc4_sma3","F0semitoneFrom27.5Hz_sma3nz","jitterLocal_sma3nz","shimmerLocaldB_sma3nz","HNRdBACF_sma3nz","logRelF0-H1-H2_sma3nz","logRelF0-H1-A3_sma3nz","F1frequency_sma3nz","F1bandwidth_sma3nz","F1amplitudeLogRelF0_sma3nz","F2frequency_sma3nz","F2bandwidth_sma3nz","F2amplitudeLogRelF0_sma3nz","F3frequency_sma3nz","F3bandwidth_sma3nz","F3amplitudeLogRelF0_sma3nz","F0semitoneFrom27.5Hz_sma3nz_amean","F0semitoneFrom27.5Hz_sma3nz_stddevNorm","F0semitoneFrom27.5Hz_sma3nz_percentile20.0","F0semitoneFrom27.5Hz_sma3nz_percentile50.0","F0semitoneFrom27.5Hz_sma3nz_percentile80.0","F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2","F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope","F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope","F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope","F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope","loudness_sma3_amean","loudness_sma3_stddevNorm","loudness_sma3_percentile20.0","loudness_sma3_percentile50.0","loudness_sma3_percentile80.0","loudness_sma3_pctlrange0-2","loudness_sma3_meanRisingSlope","loudness_sma3_stddevRisingSlope","loudness_sma3_meanFallingSlope","loudness_sma3_stddevFallingSlope","spectralFlux_sma3_amean","spectralFlux_sma3_stddevNorm","mfcc1_sma3_amean","mfcc1_sma3_stddevNorm","mfcc2_sma3_amean","mfcc2_sma3_stddevNorm","mfcc3_sma3_amean","mfcc3_sma3_stddevNorm","mfcc4_sma3_amean","mfcc4_sma3_stddevNorm","jitterLocal_sma3nz_amean","jitterLocal_sma3nz_stddevNorm","shimmerLocaldB_sma3nz_amean","shimmerLocaldB_sma3nz_stddevNorm","HNRdBACF_sma3nz_amean","HNRdBACF_sma3nz_stddevNorm","logRelF0-H1-H2_sma3nz_amean","logRelF0-H1-H2_sma3nz_stddevNorm","logRelF0-H1-A3_sma3nz_amean","logRelF0-H1-A3_sma3nz_stddevNorm","F1frequency_sma3nz_amean","F1frequency_sma3nz_stddevNorm","F1bandwidth_sma3nz_amean","F1bandwidth_sma3nz_stddevNorm","F1amplitudeLogRelF0_sma3nz_amean","F1amplitudeLogRelF0_sma3nz_stddevNorm","F2frequency_sma3nz_amean","F2frequency_sma3nz_stddevNorm","F2bandwidth_sma3nz_amean","F2bandwidth_sma3nz_stddevNorm","F2amplitudeLogRelF0_sma3nz_amean","F2amplitudeLogRelF0_sma3nz_stddevNorm","F3frequency_sma3nz_amean","F3frequency_sma3nz_stddevNorm","F3bandwidth_sma3nz_amean","F3bandwidth_sma3nz_stddevNorm","F3amplitudeLogRelF0_sma3nz_amean","F3amplitudeLogRelF0_sma3nz_stddevNorm","alphaRatioV_sma3nz_amean","alphaRatioV_sma3nz_stddevNorm","hammarbergIndexV_sma3nz_amean","hammarbergIndexV_sma3nz_stddevNorm","slopeV0-500_sma3nz_amean","slopeV0-500_sma3nz_stddevNorm","slopeV500-1500_sma3nz_amean","slopeV500-1500_sma3nz_stddevNorm","spectralFluxV_sma3nz_amean","spectralFluxV_sma3nz_stddevNorm","mfcc1V_sma3nz_amean","mfcc1V_sma3nz_stddevNorm","mfcc2V_sma3nz_amean","mfcc2V_sma3nz_stddevNorm","mfcc3V_sma3nz_amean","mfcc3V_sma3nz_stddevNorm","mfcc4V_sma3nz_amean","mfcc4V_sma3nz_stddevNorm","alphaRatioUV_sma3nz_amean","hammarbergIndexUV_sma3nz_amean","slopeUV0-500_sma3nz_amean","slopeUV500-1500_sma3nz_amean","spectralFluxUV_sma3nz_amean","loudnessPeaksPerSec","VoicedSegmentsPerSec","MeanVoicedSegmentLengthSec","StddevVoicedSegmentLengthSec","MeanUnvoicedSegmentLength","StddevUnvoicedSegmentLength","equivalentSoundLevel_dBp","arousal","dominance","valence"] + [f"Dim {i}" for i in range(1024)] + ["duration"]


l_audio_groups = {
    "LLDs":     # Intensity low level descriptors
        ["Loudness_sma3",
        # Spectral shape / tilt
        "alphaRatio_sma3",
        "hammarbergIndex_sma3",
        "slope0-500_sma3",
        "slope500-1500_sma3",
        "spectralFlux_sma3",
        # Cepstral features (MFCCs)
        "mfcc1_sma3",
        "mfcc2_sma3",
        "mfcc3_sma3",
        "mfcc4_sma3",
        # Pitch & prosody
        "F0semitoneFrom27.5Hz_sma3nz",
        # Voice quality
        "jitterLocal_sma3nz",
        "shimmerLocaldB_sma3nz",
        "HNRdBACF_sma3nz",
        # Harmonics
        "logRelF0-H1-H2_sma3nz",
        "logRelF0-H1-A3_sma3nz",
        # Formants
        "F1frequency_sma3nz",
        "F1bandwidth_sma3nz",
        "F1amplitudeLogRelF0_sma3nz",
        "F2frequency_sma3nz",
        "F2bandwidth_sma3nz",
        "F2amplitudeLogRelF0_sma3nz",
        "F3frequency_sma3nz",
        "F3bandwidth_sma3nz",
        "F3amplitudeLogRelF0_sma3nz"],
    "Functionals": l_audio_features[25:113],  # functionals: basic stats (mean, std, etc.), percentiles, slope relates /temporal, higher order moments (skewness)
    "VAD": l_audio_features[113:116],
    "Embeddings": l_audio_features[116:-1],
}

READ_RS = True

PATH_BASE = "/Users/Timon/Library/CloudStorage/Box-Box/missing-ocd-videos"

folders = [f for f in os.listdir(PATH_BASE) if os.path.isdir(os.path.join(PATH_BASE, f))]

df_ = []
for folder in folders:
    file_features = [os.path.join(PATH_BASE, folder, f) for f in os.listdir(os.path.join(PATH_BASE, folder)) if f.endswith('.csv') and "audio_features" in f][0]
    df = pd.read_csv(file_features)
    df_ = df_ + [df]
df_audio = pd.concat(df_, axis=0, ignore_index=True)

df_audio["date"] = pd.to_datetime(df_audio["date"])
# rename subject to sub column
df_audio = df_audio.rename(columns={"subject": "sub"})

PATH_FEATURES = "/Users/Timon/Documents/Houston/resting_state_OCD/FAUS_rs/fau_neural_combined_2.csv"
df_features = pd.read_csv(PATH_FEATURES)
df_features["date"] = pd.to_datetime(df_features["date"])

# iterate through df_features, and extract for each row the plus minus 1 minute from df_audio, which also has a time column

READ_AUDIO_FEATURES = True
if READ_AUDIO_FEATURES:
    dfs = []
    dfs_speech = []
    dfs_non_speech = []
    for i, row in tqdm(df_features.iterrows()):
        sub = row["subject"]
        date = row["date"]
        df_audio_sub = df_audio.query("sub == @sub and date == @date")
        if df_audio_sub.empty:
            continue
        df_audio_sub_date_ = df_audio_sub.drop(columns=["folder", "sub", "SPEAKER_ID", "text", "file", "date"], errors="ignore")
        mean_features = df_audio_sub_date_.mean(axis=0)
        # combine row and mean_features
        combined = pd.concat([row, mean_features])

        dfs = dfs + [combined]

    df_audio_features_comb = pd.DataFrame(dfs)

    # previous file
    df_audio_features_com_pre = pd.read_csv("plotting/Figure3/source_data/audio_neural_features_combined_rs.csv")

    # columns different across both dataframes
    cols_diff = set(df_audio_features_comb.columns) - set(df_audio_features_com_pre.columns)
    #remove columns ["duration", "start", "end"] from df_audio_features_comb, and then concatenate
    df_audio_features_comb = df_audio_features_comb.drop(columns=["duration", "start", "end"], errors="ignore") 

    df_audio_features_comb = pd.concat([df_audio_features_com_pre, df_audio_features_comb], axis=0, ignore_index=True)
    # replace every column starting with FAU_AU_ with AU_
    df_audio_features_comb = df_audio_features_comb.rename(columns={c: c[4:] for c in df_audio_features_comb.columns if c.startswith("FAU_")})

    # change subject column from aDBS012 to int 12 or aDBS001 to int 1
    df_audio_features_comb["subject"] = df_audio_features_comb["subject"].apply(lambda x: int(x[4:])) 

    df_audio_features_comb.to_csv("plotting/Figure1/source_data/ybocs_audio_neural_features_combined.csv", index=False)
