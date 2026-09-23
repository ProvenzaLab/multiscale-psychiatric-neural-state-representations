import sys
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from preprocessing.erp.preprocess.suds import write_suds, plot_suds_ratings
from preprocessing.erp.preprocess.ephys import read_mapping_neural_data, concat_all_sessions_fit
from preprocessing.erp.preprocess.mapping import comb_df_with_suds, combine_L_R, rewrite_data_to_pickle
from scipy.signal import welch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

process_dict = {
    "004": {"time_offset": -5, "ch_names_1": "time_domain_2_0", "ch_names_2": "time_domain_10_8", "loc_1": "NAc_L", "loc_2": "NAc_R", "hem_1": "L", "hem_2": "R"},
    "005": {"time_offset": -5, "ch_names_1": "time_domain_2_0", "ch_names_2": "time_domain_10_8", "loc_1": "NAc_L", "loc_2": "BNST_R", "hem_1": "L", "hem_2": "R"},
    "007": {"time_offset": -5, "ch_names_1": "time_domain_2_0", "ch_names_2": "time_domain_11_10", "loc_1": "BNST_L", "loc_2": "BNST_R", "hem_1": "L", "hem_2": "R"},
    #"008L": {"time_offset": 1, "ch_names_1": "time_domain_2_0", "ch_names_2": "time_domain_11_10", "loc_1": "NAc_L", "loc_2": "OFC_L", "hem_1": "L", "hem_2": "L"},
    #"008R": {"time_offset": 1, "ch_names_1": "time_domain_2_0", "ch_names_2": "time_domain_11_10", "loc_1": "NAc_R", "loc_2": "OFC_R", "hem_1": "R", "hem_2": "R"},
    "009L": {"time_offset": 1, "ch_names_1": "time_domain_2_0", "ch_names_2": "time_domain_11_10", "loc_1": "NAc_L", "loc_2": "OFC_L", "hem_1": "L", "hem_2": "L"},
    "009R": {"time_offset": 1, "ch_names_1": "time_domain_2_0", "ch_names_2": "time_domain_11_10", "loc_1": "NAc_R", "loc_2": "OFC_R", "hem_1": "R", "hem_2": "R"},
    "010L": {"time_offset": 0, "ch_names_1": "time_domain_3_1", "ch_names_2": "time_domain_11_10", "loc_1": "BNST_L", "loc_2": "OFC_L", "hem_1": "L", "hem_2": "L"},
    "010R": {"time_offset": 0, "ch_names_1": "time_domain_2_0", "ch_names_2": "time_domain_11_10", "loc_1": "NAc_R", "loc_2": "OFC_R", "hem_1": "R", "hem_2": "R"},
    "011L": {"time_offset": 0, "ch_names_1": "time_domain_2_0", "ch_names_2": "time_domain_11_10", "loc_1": "BNST_L", "loc_2": "OFC_L", "hem_1": "L", "hem_2": "L"},
    "011R": {"time_offset": 0, "ch_names_1": "time_domain_2_0", "ch_names_2": "time_domain_11_10", "loc_1": "BNST_R", "loc_2": "OFC_R", "hem_1": "R", "hem_2": "R"},
    "012L": {"time_offset": 1, "ch_names_1": "time_domain_2_0", "ch_names_2": "time_domain_11_10", "loc_1": "NAc_L", "loc_2": "OFC_L", "hem_1": "L", "hem_2": "L"},
    "012R": {"time_offset": 1, "ch_names_1": "time_domain_2_0", "ch_names_2": "time_domain_11_10", "loc_1": "NAc_R", "loc_2": "OFC_R", "hem_1": "R", "hem_2": "R"}
}


WRITE_SUDS = False
WRITE_COMB_EPHYS = False
MAP_EPHYS = False
COMBINE_LEFT_RIGHT = True


for sub in list(process_dict.keys())[-1:]:
    print(f"Processing subject {sub}")
    #sub = "005"
    ## WRITE SUDS
    if WRITE_SUDS:
        if sub[:3] == "009":
            df_suds = write_suds(sub, "Sheet1")
        elif sub[:3] == "008" or sub[:3] == "010" or sub[:3] == "011" or sub[:3] == "012":
            df_suds = write_suds(sub, sheet_name="SUDS Ratings")
        else:
            df_suds = write_suds(sub)
        if sub == "007":
            ratings_ = df_suds["ratings"].fillna(df_suds["rating"])
            df_suds["ratings"] = ratings_
            df_suds = df_suds.drop(columns=["rating"])
            df_suds.to_csv(f"{sub}/{sub}_suds_ratings.csv", index=False)
        elif sub[:3] == "008" or sub[:3] == "010":
            df_suds = df_suds.rename(columns={"Ratings": "ratings"})
            df_suds.to_csv(f"{sub}/{sub[:3]}_suds_ratings.csv", index=False)
        elif sub[:3] == "009":
            df_suds.to_csv(f"{sub}/{sub[:3]}_suds_ratings.csv", index=False)

    ## WRITE COMBINED EPHYS
    if WRITE_COMB_EPHYS:
        PATH_EPHYS = f"/Volumes/labworlds/Provenza/OCD_RCS_at_home/upload/{sub}"  # Home am Ende bei sub < 011
        if sub != "004" and sub != "005" and sub != "007" and sub[:3] != "011" and sub[:3] != "012":
            PATH_EPHYS = f"{PATH_EPHYS}_{sub[-1]}"
        PATH_OUT = f"{sub}/{sub}_ephys_sessions.csv"
        sessions_fit = read_mapping_neural_data(sub, PATH_EPHYS, diff_hours=process_dict[sub]["time_offset"])
        df_sessions = concat_all_sessions_fit(sessions_fit, PATH_EPHYS, PATH_OUT)

    if MAP_EPHYS:
        PATH_EPHYS = f"{sub}/{sub}_ephys_sessions.csv"
        PATH_SUDS = f"{sub}/{sub}_suds_ratings.csv"
        ch_name = process_dict[sub]["ch_names_1"]
        df_comb = comb_df_with_suds(PATH_EPHYS,
                        PATH_SUDS,
                        PATH_OUT = f"{sub}/comb_mapped_{sub}.csv",
                        ch_used = ch_name,
                        extraction_window_s=120, hour_offset=process_dict[sub]["time_offset"],
                        time_extraction_before_suds_s=60)

    if COMBINE_LEFT_RIGHT:
        if sub in ["009L", "010L", "011R", "012R"]:
            sub = sub[:-1]
            combine_L_R(sub)

        # rewrite also other subjects in pickle format
        if sub in ["004", "005", "007"]:
            rewrite_data_to_pickle(sub)
