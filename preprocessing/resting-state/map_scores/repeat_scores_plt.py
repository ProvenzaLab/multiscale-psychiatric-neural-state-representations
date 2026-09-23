import pandas as pd
import os
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
import seaborn as sns

def convert_to_datetime(d):
    if d.startswith("20"):
        dt_ = datetime.strptime(d, "%Y%m%d%H%M%S%f")
    else:
        dt_ = datetime.fromtimestamp(int(d) / 1000)
    return dt_

SCORE_ROWS = {
    "YBOCS II-Obsessions Sub-score": 13,
    "YBOCS II-Compulsions Sub-score": 14,
    "YBOCS II Insight Sub-Score ": 15,
    "YBOCS II Reliability Sub-Score": 16,
    "YBOCS II Global Severity Sub-Score": 17,
    "YBOCS II Global Improvement Sub-Score": 18,
    "YBOCS II Total Score": 19,
    "YBOCS I Obsessions Sub-Score": 20,
    "YBOCS I Compulsions Sub-Score": 21,
    "YBOCS I Insight Sub-Score": 22,
    "YBOCS I Total Score": 23,
    "HDRS Total Score": 24,
    "YMRS Total Score": 25,
    "Clinical Global Impression-Severity": 26,
    "Clinical Global Impression-Improvement": 27,
    "BDI-Total Score": 28,
    "BAI-Total Score": 29,
    "Composed-Anxious Sub Score": 31,
    "Agreeable-Hostile Sub Score": 32,
    "Elated-Depressed Sub Score": 33,
    "Confident-Unsure Sub Score": 34,
    "Energetic-Tired Sub Score": 35,
    "Clearheaded-Confused Sub Score": 36,
    "POMS Total Score": 37,
    "Category 1:Concerns about Germs and Cotamination- Subscale Score": 39,
    "Category 2: Concerns about being Responsible for Harm, Injury, or Bad Luck- Subscale Scale": 40,
    "Category 3: Unacceptable Thoughts-Subscale Score": 41,
    "Category 4: Concerns about Symmetry, Completeness": 42,
    "Category 5: Sexually Intrusive Thoughts- Subscale Score": 43,
    "Category 6: Intrusive Violent Thoughts- Subscale Score": 44,
    "Category 7: Immoral and Scrupulous Thoughts- Subscale Score": 45,
    "DOCS Total Score": 46,
    "SDS Total Score" : 47,
    "Attentional Impulsiveness-Subscale Score": 51,
    "Motor Impulsiveness- Subscale Score": 52,
    "Nonplanning Impulsiveness-Subscale Score": 53,
    "BIS Total Score": 54,
    "IUS Total Score": 55

}

df_ybocs_timon = pd.read_csv("features_prep_combined_wide.csv")
subjects = df_ybocs_timon["subject"].unique()

df_scores  = {str(sub): pd.read_excel("aDBS Clinical Outcomes Master Database (all subjects).xlsx",
            sheet_name=sub, engine='openpyxl') for sub in np.unique(subjects)}

df_ybocs_all = []
for sub in subjects:
    
    df = df_scores[sub]

    dates = df.iloc[4, :].values
    dates = [
        np.datetime64(f, 'D') if isinstance(f, datetime) else None
        for f in dates
    ]
    # get ybocs for each date and subject, just use the SCORE_ROWS for YBOCS II Total Score
    for date in dates:
        if date is None:
            continue
        # search in df for the row corresponding to YBOCS II Total Score
        ybocs_row = SCORE_ROWS["YBOCS II Total Score"]
        ybocs_score = df.iloc[ybocs_row, np.where(dates == date)[0][0]]
        # if ybocs_score is not nan, print the subject,
        if not pd.isna(ybocs_score):
            print(f"Subject: {sub}, Date: {date}, YBOCS II Total Score: {ybocs_score}")
        df_ybocs_all.append({
            "subject": sub,
            "date": date,
            "YBOCS II Total Score": ybocs_score
        })
df_ybocs_all = pd.DataFrame(df_ybocs_all)

# rename "unique_dates" to "date" in df_ybocs_all
df_ybocs_timon = df_ybocs_timon.rename(columns={"unique_dates": "date"})
df_ybocs_timon["date"] = pd.to_datetime(df_ybocs_timon["date"])
df_ybocs_all["date"] = pd.to_datetime(df_ybocs_all["date"])
# make a plot for each subject, save to pdf
from matplotlib.backends.backend_pdf import PdfPages
# plot both the entries in df_ybocs_timon and df_ybocs_all, with different colors, for each subject
pdf_ = PdfPages("map_scores/ybocs_scores_all_subjects.pdf")
for sub in subjects:
    df_timon = df_ybocs_timon.query("subject == @sub")
    df_all = df_ybocs_all.query("subject == @sub")
    # remove nan entries from YBOCS II Total Score
    df_timon = df_timon.dropna(subset=["YBOCS II Total Score"])
    df_all = df_all.dropna(subset=["YBOCS II Total Score"])
    plt.figure(figsize=(10, 5))
    plt.plot(df_timon["date"], df_timon["YBOCS II Total Score"], marker='o', label="Neural Data used", color='blue', linestyle='')
    plt.plot(df_all["date"], df_all["YBOCS II Total Score"], marker='x', label="Score available", color='orange')
    plt.title(f"YBOCS II Total Score for subject {sub}")
    plt.xlabel("Date")
    plt.ylabel("YBOCS II Total Score")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    pdf_.savefig()
pdf_.close()
print("")