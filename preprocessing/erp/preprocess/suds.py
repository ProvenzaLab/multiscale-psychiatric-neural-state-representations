import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def write_suds(sub, sheet_name="Sheet1"):
    PATH_RATINGS = f"{sub}/aDBS{sub[:3]}-CBT-ratings"  # SUDS are always in LEFT folder

    files_ = [os.path.join(PATH_RATINGS, f) for f in os.listdir(PATH_RATINGS) if f.endswith('.xlsx') and f.startswith('~') is False and f.startswith(".") is False and "dates" not in f]

    l_scores = []
    for f in files_:
        print(f)
        df = pd.read_excel(f, sheet_name=sheet_name)
        if df.shape[0] == 0:
            continue
        l_scores.append(df)

    df_scores = pd.concat(l_scores, ignore_index=True)
    df_scores = df_scores.rename(columns={"Unix": "timestamp"})
    df_scores = df_scores.sort_values(by="timestamp").reset_index(drop=True)
    df_scores["ts_datetime"] = pd.to_datetime(df_scores["timestamp"], unit='s')
    if "Ratings" in df_scores.columns:
        df_scores = df_scores.rename(columns={"Ratings": "ratings"})
    df_scores.to_csv(f"{sub}/{sub}_suds_ratings.csv", index=False)
    return df_scores


def plot_suds_ratings(df_scores, patient_id):
    plt.figure(figsize=(4,3))
    plt.plot(df_scores["ts_datetime"], df_scores["ratings"], "o", markersize=4, label="SUDS Ratings")
    plt.xlabel("Timestamp")
    plt.ylabel("SUDS Ratings")
    plt.xticks(rotation=45)
    plt.title(f"Patient {patient_id}")
    plt.tight_layout()
    plt.savefig(f"figures/{patient_id}_suds_ratings.pdf", bbox_inches='tight')
    plt.show()
