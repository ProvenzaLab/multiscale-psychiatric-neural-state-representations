1. read_data.m -> run on dry tortugas (Documents/Timon/)
2. read_data.py # read from "restingstate_data" write to "rs_prep" on wrangell
3. /scratch/timonmerk/pre/OCDRCSResting/map_scores/get_scores_from_sheet.py
4. /scratch/timonmerk/pre/OCDRCSResting/map_scores/map_scores.py
5. plot raw data and write out clean data pickle files: plot_raw_data.py -> read data from /timonmerk/"rs_prep" from elias
6. map_scores -> get_scores_from_sheet.py
7. manually perform annotations and write to annotations_rcs.xlsx
8. compute_features.py # writes to features_out/ on elias: RCSRestingv2, 
9. write out features_prep_combined.csv and correlation_features.csv -> read_res_df.py
