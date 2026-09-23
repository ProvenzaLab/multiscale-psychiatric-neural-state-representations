from pathlib import Path
import sys

# Allow running this file directly via `python plotting/Figure1/...py`.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from plotting.Figure4.generate_source_data import utils
import pandas as pd
import matplotlib.pyplot as plt

df_merged, subs = utils.get_df_features("all", "all", READ_RS=True)

arr = df_merged.query("subject==4")
arr["date"] = pd.to_datetime(arr["date"])
arr["days_since_start"] = (arr["date"] - arr["date"].min()).dt.days
arr_plt = arr[[c for c in arr.columns if c.startswith("SC_")]]
days_ = arr["days_since_start"].values
# z-score normalization
arr_plt_zs = (arr_plt - arr_plt.mean()) / arr_plt.std()
# remove rows with only NaN rows
arr_plt_zs = arr_plt_zs.dropna(how='all')
plt.figure()
plt.imshow(arr_plt_zs.T, aspect='auto', cmap='viridis')
plt.colorbar(label='Feature Value')
plt.xticks(ticks=range(len(arr_plt_zs)), labels=days_[arr_plt_zs.index], rotation=90)
plt.savefig("figures/Figure1/fig1_exemplar_ybocs_feature_trace_subject4.pdf")