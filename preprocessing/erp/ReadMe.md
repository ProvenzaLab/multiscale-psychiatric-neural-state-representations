1. call convert_patient.m
    adjust the PATH_CONVERT and PATH_OUT
    PATH_CONVERT calls raw rc+s .json directory (/Volumes/datalake/aDBS-49155/aDBS012-at-home/LFP/aDBS012R)
    PATH_OUT writes to datalake: labworlds/Provenza/OCD_RCS_at_home/upload/aDBS012L
    Needs to be repeated for both hemispheres for OFC patients
    Requirements:
     - Matlab2024b
     - Install toolbox: https://github.com/openmind-consortium/Analysis-rcs-data
2. check_available_data.py
3. process.py
4. plot_data_annotate.py (plot data and write out data as pickle)
5. compute_features.py 
6. combine_features_all_patients.py --> plotting/Figure1/source_data/neural_features_suds.csv

7. decode.py (NEURAL)
8. decode_within_patient.py (NEURAL)
   and write out coordinates combined with individual decoding