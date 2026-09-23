import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# set font size to 5 and font family to Arial
plt.rcParams.update({'font.size': 5, 'font.family': 'Arial'})

df_stim_params_final = pd.read_csv("plotting/Figure2/source_data/StimParams_Processed_1.csv")
df_stim_params_final["SessDate"] = pd.to_datetime(df_stim_params_final["SessDate"])
# sort by SessDate and PatientID
df_stim_params_final = df_stim_params_final.sort_values(by=["PatientID", "SessDate"])
# so, now plot the params for left and right hemisphere over time for each patient
df_use = df_stim_params_final.copy()
df_use["TEED"] = df_use["Amplitude_mA"] * df_use["PulseWidth_us"] * df_use["Frequency_Hz"] / 1000  # in uW
df_use["Days_Since_First_Session"] = (df_use["SessDate"] - df_use.groupby("PatientID")["SessDate"].transform('min')).dt.days
# remove ind with 0 Amplitude (DBS off)
df_use = df_use[df_use["Amplitude_mA"] > 0]

plt.figure(figsize=(6, 4))
for i, sub in enumerate(df_use['PatientID'].unique()):
    
    for j, col in enumerate(["Amplitude_mA", "PulseWidth_us", "Frequency_Hz", "TEED"]):
        plt.subplot(8, 4, i * 4 + j + 1)
        for hem in ['left', 'right']:
            df_sub_hem = df_use[(df_use['PatientID'] == sub) & (df_use['Hemisphere'] == hem)]
            plt.plot(df_sub_hem['Days_Since_First_Session'], df_sub_hem[col], marker='o', label=f'{hem}', markersize=1, linewidth=0.5)
        # remove upper and right spines
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        if j == 0:
            plt.ylabel(f'{sub}')
            
        else:
            plt.ylabel('')
            # remove x ticks and 
            #plt.gca().set_xticklabels([])
        if i == 0:
            plt.title(f"{col}")
        plt.xlabel('Days Since First Session')
        if i != len(df_use['PatientID'].unique()) - 1:
            plt.gca().set_xticklabels([])
            plt.xlabel('')
        if i == 0 and j == 0:
            plt.legend(title='Hemisphere')

        # reduce tick length
        plt.tick_params(axis='both', which='major', length=2)

#plt.tight_layout()
plt.savefig("figures/Figure2/StimParams.pdf")
