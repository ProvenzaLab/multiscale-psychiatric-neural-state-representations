import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#df_params_merged = pd.read_csv("/Users/Timon/Documents/Houston/resting_state_OCD/stim params/StimParams_AllPatients_20251124_0831.csv")
df_params_merged = pd.read_csv("/Users/Timon/Documents/Houston/YBOCS/preprocessing/resting-state/get_stim_parameters/StimParams_AllPatients_20260828_0724.csv")

stim_split = df_params_merged['string_stimParams'].str.split(',', expand=True)

# Assign column names
stim_split.columns = ['Contact', 'Amplitude_mA', 'PulseWidth_us', 'Frequency_Hz']

def apply_contact_split(contact):
    anode = None
    cathode = None

    if type(contact) != float and contact != '' and contact != "Disabled":
        anode = contact[0]
        cathode = contact[2]

    return anode, cathode

# Apply the function to split Contact column
stim_split[['Anode+', 'Cathode-']] = stim_split['Contact'].apply(apply_contact_split).apply(pd.Series)
stim_split = stim_split.apply(lambda x: x.str.strip())
stim_split['Amplitude_mA'] = stim_split['Amplitude_mA'].str.replace('mA', '', regex=False).astype(float)
stim_split['PulseWidth_us'] = stim_split['PulseWidth_us'].str.replace('us', '', regex=False).astype(float)
stim_split['Frequency_Hz'] = stim_split['Frequency_Hz'].str.replace('Hz', '', regex=False).astype(float)
stim_split['Contact'] 

# Merge back with original DataFrame
df_params_merged = pd.concat([df_params_merged, stim_split], axis=1)

# I expect only unique BheaviorStartUnix values
behav_starts = df_params_merged['BehavStartUnix'].unique().tolist()
# for each behav_starts, there are multiple rows with stim_parameters; select the earliest one
df_params_merged['SessDate'] = pd.to_datetime(df_params_merged['SessionDate'].str.replace('d', '').str.replace('_', '-'))
df_params_merged["PatientID"] = df_params_merged["PatientID"].apply(lambda x: int(x[1:]))
df_params_merged["exact_time"] = pd.to_datetime(df_params_merged["string_HostUnixTime"], unit='ms')

stim_params_list = []
missed_list = []
for ts in behav_starts:
    for hem in ['left', 'right']:
        ts_dt = pd.to_datetime(ts, unit='ms')
        df_subset = df_params_merged[df_params_merged['BehavStartUnix'] == ts]
        df_subset_hem = df_subset[df_subset['Hemisphere'] == hem]

        df_earlier = df_subset_hem[df_subset_hem['exact_time'] <= ts_dt]
        if df_earlier.empty:
            missed_list.append(ts)
            continue

        df_stim_params = df_earlier.sort_values(by='exact_time').tail(1)
        time_diff = ts_dt - df_stim_params['exact_time'].values[0]
        df_stim_params["time_diff"] = time_diff
        df_stim_params["timestamp_behavior"] = ts
        if time_diff < pd.Timedelta(minutes=120):
            stim_params_list.append(df_stim_params)
        else:
            missed_list.append(ts)

# this can be adapted, for each bheavior timestamp I practically just
# need the most recent stim settings

df_stim_params_final = pd.concat(stim_params_list).reset_index(drop=True)
# if string_therapyStatus is 0, then set Amplitude_mA, PulseWidth_us, Frequency_Hz to 0
df_stim_params_final.loc[df_stim_params_final['string_therapyStatus'] == 0, ['Amplitude_mA', 'PulseWidth_us', 'Frequency_Hz']] = 0
df_stim_params_final.to_csv("plotting/Figure2/source_data/StimParams_Processed_1.csv", index=False)