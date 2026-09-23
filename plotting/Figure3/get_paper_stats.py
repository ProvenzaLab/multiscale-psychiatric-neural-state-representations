import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.stats import t
from statsmodels.stats.multitest import multipletests
# import permutation tests
from scipy.stats import permutation_test

def statistic(x, y):
    return np.mean(x) - np.mean(y)

df_so = pd.read_csv("plotting/Figure3/source_data/second_order_behav_corrs_1.csv")
for behav_type in df_so["behav_type"].unique():
    df_so_sub = df_so[df_so["behav_type"] == behav_type]
    r_mean = df_so_sub["corr_suds_ybocs"].mean()
    r_std = df_so_sub["corr_suds_ybocs"].std()
    p_val = permutation_test((df_so_sub["corr_suds_ybocs"], np.zeros_like(df_so_sub["corr_suds_ybocs"])), statistic, n_resamples=5000, alternative='two-sided')
    print(f"Second order correlations for {behav_type}: mean={r_mean:.3f}, std={r_std:.3f}, p-value={p_val.pvalue:.5f}")


# Second order correlations for FAU: mean=0.038, std=0.213, p-value=0.65326
# Second order correlations for Audio: mean=0.025, std=0.200, p-value=0.75117

N_PERM = 5000  # 5000
MULTIPLE_COMPARISON_CORRECTION = True
FDR_ALPHA = 0.05
RAW_ALPHA = 0.01
N_JOBS = -1
RANDOM_SEED = 42

def fast_corr_p(x, y):
    """
    Fast Pearson correlation and two-sided p-value.
    Equivalent to scipy.stats.pearsonr for normal cases.
    """

    n = len(x)

    if n < 4:
        return np.nan, np.nan

    x = x - np.mean(x)
    y = y - np.mean(y)

    xx = np.dot(x, x)
    yy = np.dot(y, y)

    if xx == 0 or yy == 0:
        return np.nan, np.nan

    r = np.dot(x, y) / np.sqrt(xx * yy)
    r = np.clip(r, -0.999999999999, 0.999999999999)

    df = n - 2
    t_stat = r * np.sqrt(df / (1 - r ** 2))
    p = 2 * t.sf(np.abs(t_stat), df)

    return r, p

# =========================
# Permutation test
# =========================

def one_permutation(seed):
    rng = np.random.default_rng(seed)

    pvals = np.empty(len(pairs))

    for i, pair in enumerate(pairs):
        x = pair["x"]
        y = pair["y"]

        y_perm = rng.permutation(y)

        _, p = fast_corr_p(x, y_perm)
        pvals[i] = p

    valid = ~np.isnan(pvals)

    rejected_perm = np.zeros(len(pvals), dtype=bool)

    if np.sum(valid) > 0:
        if MULTIPLE_COMPARISON_CORRECTION:
            rejected_valid, _, _, _ = multipletests(
                pvals[valid],
                alpha=FDR_ALPHA,
                method="fdr_bh"
            )
        else:
            rejected_valid = pvals[valid] < RAW_ALPHA
        rejected_perm[valid] = rejected_valid

    return int(rejected_perm.sum())

for FAUS in [True, False]:
    for YBOCS in [True, False]:

        if YBOCS: 
            CSV_PATH = "plotting/Figure1/source_data/ybocs_audio_neural_features_combined_1.csv"
        else:
            CSV_PATH = "plotting/Figure3/source_data/suds_neural_audio_fau_combined.csv"

        df = pd.read_csv(CSV_PATH)

        if YBOCS:
            col_decode = "YBOCS II Total Score"
        else:
            col_decode = "score"

        subs = df["subject"].unique()
        if FAUS:
            behav_cols = [col for col in df.columns if col.startswith("AU_")]
        else:
            behav_cols = ["Loudness_sma3","alphaRatio_sma3","hammarbergIndex_sma3","slope0-500_sma3","slope500-1500_sma3","spectralFlux_sma3","mfcc1_sma3","mfcc2_sma3","mfcc3_sma3","mfcc4_sma3","F0semitoneFrom27.5Hz_sma3nz","jitterLocal_sma3nz","shimmerLocaldB_sma3nz","HNRdBACF_sma3nz","logRelF0-H1-H2_sma3nz","logRelF0-H1-A3_sma3nz","F1frequency_sma3nz","F1bandwidth_sma3nz","F1amplitudeLogRelF0_sma3nz","F2frequency_sma3nz","F2bandwidth_sma3nz","F2amplitudeLogRelF0_sma3nz","F3frequency_sma3nz","F3bandwidth_sma3nz","F3amplitudeLogRelF0_sma3nz","F0semitoneFrom27.5Hz_sma3nz_amean","F0semitoneFrom27.5Hz_sma3nz_stddevNorm","F0semitoneFrom27.5Hz_sma3nz_percentile20.0","F0semitoneFrom27.5Hz_sma3nz_percentile50.0","F0semitoneFrom27.5Hz_sma3nz_percentile80.0","F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2","F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope","F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope","F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope","F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope","loudness_sma3_amean","loudness_sma3_stddevNorm","loudness_sma3_percentile20.0","loudness_sma3_percentile50.0","loudness_sma3_percentile80.0","loudness_sma3_pctlrange0-2","loudness_sma3_meanRisingSlope","loudness_sma3_stddevRisingSlope","loudness_sma3_meanFallingSlope","loudness_sma3_stddevFallingSlope","spectralFlux_sma3_amean","spectralFlux_sma3_stddevNorm","mfcc1_sma3_amean","mfcc1_sma3_stddevNorm","mfcc2_sma3_amean","mfcc2_sma3_stddevNorm","mfcc3_sma3_amean","mfcc3_sma3_stddevNorm","mfcc4_sma3_amean","mfcc4_sma3_stddevNorm","jitterLocal_sma3nz_amean","jitterLocal_sma3nz_stddevNorm","shimmerLocaldB_sma3nz_amean","shimmerLocaldB_sma3nz_stddevNorm","HNRdBACF_sma3nz_amean","HNRdBACF_sma3nz_stddevNorm","logRelF0-H1-H2_sma3nz_amean","logRelF0-H1-H2_sma3nz_stddevNorm","logRelF0-H1-A3_sma3nz_amean","logRelF0-H1-A3_sma3nz_stddevNorm","F1frequency_sma3nz_amean","F1frequency_sma3nz_stddevNorm","F1bandwidth_sma3nz_amean","F1bandwidth_sma3nz_stddevNorm","F1amplitudeLogRelF0_sma3nz_amean","F1amplitudeLogRelF0_sma3nz_stddevNorm","F2frequency_sma3nz_amean","F2frequency_sma3nz_stddevNorm","F2bandwidth_sma3nz_amean","F2bandwidth_sma3nz_stddevNorm","F2amplitudeLogRelF0_sma3nz_amean","F2amplitudeLogRelF0_sma3nz_stddevNorm","F3frequency_sma3nz_amean","F3frequency_sma3nz_stddevNorm","F3bandwidth_sma3nz_amean","F3bandwidth_sma3nz_stddevNorm","F3amplitudeLogRelF0_sma3nz_amean","F3amplitudeLogRelF0_sma3nz_stddevNorm","alphaRatioV_sma3nz_amean","alphaRatioV_sma3nz_stddevNorm","hammarbergIndexV_sma3nz_amean","hammarbergIndexV_sma3nz_stddevNorm","slopeV0-500_sma3nz_amean","slopeV0-500_sma3nz_stddevNorm","slopeV500-1500_sma3nz_amean","slopeV500-1500_sma3nz_stddevNorm","spectralFluxV_sma3nz_amean","spectralFluxV_sma3nz_stddevNorm","mfcc1V_sma3nz_amean","mfcc1V_sma3nz_stddevNorm","mfcc2V_sma3nz_amean","mfcc2V_sma3nz_stddevNorm","mfcc3V_sma3nz_amean","mfcc3V_sma3nz_stddevNorm","mfcc4V_sma3nz_amean","mfcc4V_sma3nz_stddevNorm","alphaRatioUV_sma3nz_amean","hammarbergIndexUV_sma3nz_amean","slopeUV0-500_sma3nz_amean","slopeUV500-1500_sma3nz_amean","spectralFluxUV_sma3nz_amean","loudnessPeaksPerSec","VoicedSegmentsPerSec","MeanVoicedSegmentLengthSec","StddevVoicedSegmentLengthSec","MeanUnvoicedSegmentLength","StddevUnvoicedSegmentLength","equivalentSoundLevel_dBp","arousal","dominance","valence"] #+ [f"Dim {i}" for i in range(1024)]# + ["duration"]


        # =========================
        # Precompute valid subject × AU pairs
        # =========================

        pairs = []

        for sub in subs:
            df_sub = df[df["subject"] == sub]

            y_all = df_sub[col_decode].to_numpy(dtype=float)

            for behav_col in behav_cols:
                x_all = df_sub[behav_col].to_numpy(dtype=float)

                valid = ~(np.isnan(x_all) | np.isnan(y_all))

                x = x_all[valid]
                y = y_all[valid]

                if len(x) >= 4:
                    pairs.append({
                        "subject": sub,
                        "behav_col": behav_col,
                        "x": x,
                        "y": y,
                        "n": len(x),
                    })

        #print(f"Number of valid subject × AU tests: {len(pairs)}")


        # =========================
        # Observed correlations
        # =========================

        observed_results = []

        for pair in pairs:
            r, p = fast_corr_p(pair["x"], pair["y"])

            observed_results.append({
                "subject": pair["subject"],
                "behav_col": pair["behav_col"],
                "n": pair["n"],
                "corr": r,
                "p": p,
            })

        df_corr = pd.DataFrame(observed_results)

        valid_p = df_corr["p"].notna().values

        rejected = np.zeros(len(df_corr), dtype=bool)
        pvals_corrected = np.full(len(df_corr), np.nan)

        if MULTIPLE_COMPARISON_CORRECTION:
            rejected_valid, pvals_corrected_valid, _, _ = multipletests(
                df_corr.loc[valid_p, "p"].values,
                alpha=FDR_ALPHA,
                method="fdr_bh"
            )
            rejected[valid_p] = rejected_valid
            pvals_corrected[valid_p] = pvals_corrected_valid
        else:
            raw_p = df_corr.loc[valid_p, "p"].values
            rejected[valid_p] = raw_p < RAW_ALPHA
            pvals_corrected[valid_p] = raw_p

        df_corr["p_fdr"] = pvals_corrected
        df_corr["significant_fdr"] = rejected

        observed_nsig = int(df_corr["significant_fdr"].sum())

        #print(f"Observed number of FDR-significant FAU correlations: {observed_nsig}")

        seeds = RANDOM_SEED + np.arange(N_PERM)

        one_permutation(0)
        num_significant_shuffled = Parallel(
            n_jobs=N_JOBS,
            verbose=0
        )(
            delayed(one_permutation)(seed)
            for seed in seeds
        )

        num_significant_shuffled = np.array(num_significant_shuffled)

        p_value_global = np.mean(num_significant_shuffled >= observed_nsig)
        
        print(f"Permutation p-value for number of FDR-significant FAUs: {p_value_global:.6f}")
        str_ = "FAUS" if FAUS else "Audio"
        str_ += " YBOCS" if YBOCS else " SUDS"
        print(f"Results for {str_}:")
        print(f"p={p_value_global:.6f} n={observed_nsig}, shuffled={num_significant_shuffled.mean():.2f} ± {num_significant_shuffled.std():.2f}")

# FAUS SUDS:
# p=0.000000 n=41, 0.06 ± 0.26

# Audio SUDS:
# p=0.000000 n=19, 0.05 ± 0.25

# FAUS YBOCS:
# p=1.000000 n=0, 0.28 ± 0.58

# Audio YBOCS:
# p=0.344800 n=1, 0.45 ± 0.72




# Previous results:
###############################
# FAUs
# RS: p=0.039 n=1, 0.13 pm 1.35; SUDS: p<10^-5 n=5.86 pm 8.67

# Audio
# RS: p<10^-5 n=9, 1.12 pm 2.41; SUDS: p<10^-5, n=18, 2.57 pm 4.31

# get the mean count and std of significant_fdr across subjects
count_per_subject = df_corr.groupby("subject")["significant_fdr"].sum()
mean_count = count_per_subject.mean()
std_count = count_per_subject.std()

# =========================
# Save outputs
# =========================

df_null = pd.DataFrame({
    "perm_index": np.arange(N_PERM),
    "num_significant": num_significant_shuffled,
})


# =========================
# Optional summary
# =========================

print("\nTop observed correlations:")
print(
    df_corr.sort_values("p_fdr")
    .head(20)
    [["subject", "behav_col", "n", "corr", "p", "p_fdr", "significant_fdr"]]
)