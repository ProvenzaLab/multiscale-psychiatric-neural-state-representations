import pandas as pd
from tqdm import tqdm
import os

path_faus = "/Users/Timon/Documents/Houston/video_features/extracting_FAUs/outdir"
recording_ids = os.listdir(path_faus)

df_mapping = pd.read_csv("/Users/Timon/Documents/Houston/resting_state_OCD/FAUS_rs/fau_date_video_mapping_2.csv")

df_features = pd.read_csv("/Users/Timon/Documents/Houston/resting_state_OCD/features_prep_combined_wide_1.csv")
df_features = pd.read_csv("plotting/Figure1/source_data/neural_features_ybocs_with_coh_1.csv")

rows_ = []
for recording_id in tqdm(recording_ids):
    video_name = recording_id + ".MP4"
    date = df_mapping.query("video == @video_name")["date"].values[0]
    sub = df_mapping.query("video == @video_name")["sub"].values[0]

    df_FAU_rec = pd.read_csv(os.path.join(path_faus, recording_id, "full_au_results.csv"))
    AU_cols = [col for col in df_FAU_rec.columns if col.startswith("AU")]
    df_FAU_sub_mean = df_FAU_rec.query("face_detected == 1")[AU_cols].mean()

    sub_str = str(sub)
    sub_str = sub_str.zfill(3)
    sub_adbs_name = f"aDBS{sub_str}"
    row_use = df_features.query("subject == @sub_adbs_name and date == @date")
    if len(row_use) == 0:
        print(f"No matching row for sub {sub_adbs_name} and date {date}")
        continue
    
    row = row_use.iloc[0].to_dict()
    row.update({f"FAU_{col}": getattr(df_FAU_sub_mean, col) for col in AU_cols})
    rows_.append(row)

df_fau = pd.DataFrame(rows_)
df_fau.to_csv("/Users/Timon/Documents/Houston/resting_state_OCD/FAUS_rs/FAUS_rs/fau_neural_combined_2.csv", index=False)