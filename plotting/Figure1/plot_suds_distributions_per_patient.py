import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

df_SUDS = pd.read_csv("plotting/Figure1/source_data/neural_features_suds.csv")
df_SUDS["time"] = pd.to_datetime(df_SUDS["time"])
# replace subject column 4 with aDBS004, 5 aDBS005, 7 aDBS007, 8 aDBS008, 9 aDBS009, 10 aDBS010, 11 aDBS011, 12 aDBS012
df_SUDS["subject"] = df_SUDS["subject"].replace({
    4: "aDBS004", 5: "aDBS005", 7: "aDBS007", 8: "aDBS008", 9: "aDBS009", 
    10: "aDBS010", 11: "aDBS011", 12: "aDBS012"
})

plt.figure(figsize=(4, 3))
sns.boxplot(x="subject", y='score', data=df_SUDS, showmeans=True, showfliers=False,
            boxprops=dict(alpha=0.6),
            palette="viridis")
sns.swarmplot(x="subject", y='score', data=df_SUDS, color=".25", size=4, alpha=0.4,)
plt.ylabel("SUDS")
plt.xticks(rotation=90)
# turn off upper and right border
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.savefig("figures/Figure1/fig1_suds_distributions.pdf", bbox_inches='tight')
plt.show()
