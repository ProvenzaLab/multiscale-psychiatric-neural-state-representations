from matplotlib import pyplot as plt

import pandas as pd

PATH_SUDS = "neural_audio_fau_combined.csv"
df = pd.read_csv(PATH_SUDS)
trace1 = df.query("subject == 5")["AU_1"].values
trace2 = df.query("subject == 5")["AU_L4"].values
trace_1_nonan = trace1[~pd.isna(trace1)]
trace_2_nonan = trace2[~pd.isna(trace2)]

plt.plot(trace_1_nonan)
plt.plot(trace_2_nonan)
plt.savefig("figures/Figure3/example_fau_suds.pdf")

PATH_RS = "plotting/Figure3/source_data/ybocs_audio_neural_features_combined.csv"
df = pd.read_csv(PATH_RS)
trace1 = df.query("sub == 5")["FAU_AU_1"].values
trace2 = df.query("sub == 5")["FAU_AU_L4"].values
trace_1_nonan = trace1[~pd.isna(trace1)]
trace_2_nonan = trace2[~pd.isna(trace2)]
plt.figure()
plt.plot(trace_1_nonan)
plt.plot(trace_2_nonan)
plt.savefig("figures/Figure3/example_fau_ybocs.pdf")