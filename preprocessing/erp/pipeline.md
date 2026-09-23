Preprocessing pipeline
----------------------
PATH_PREPROCESSED: /Users/Timon/Documents/Houston/OCD_RCS/OCD_RCS/preprocess

 - preprocessing/convert_json_batch_callable.m
   - required https://github.com/openmind-consortium/Analysis-rcs-data analysi-rcs-data matlab toolbox. Reads in raw .json files and parses them into .csv files
 - preprocessing/check_available_data.py: find time-offset between SUDS scores and available neural data
   - output will be a plot with time-ranges and a .csv file with all session durations
   - note: run iteratively for different subjects
 - process.py
   - write out suds csv table (merged from different sheets)
   - combine with ephys timeseries (120s for each SUDS value)
   - combine L + R; resample to 250 Hz
 - plot_data_annotate.py
   - plot time-series and write out pickle file
  
Features
--------
 - compute_features.py
   - this ran on elias; includes Annotations_ephys.xlsx
   - output is {sub}_features_prep.csv -> repeat this computation and compare results
   - is there a potential bug in the coherence computation? if not, trust the results and report coh results
 - combine_features


