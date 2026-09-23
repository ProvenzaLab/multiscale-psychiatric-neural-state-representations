import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('QtAgg')

subs = ["004", "005", "007", "008L", "008R", "009L", "009R", "010L", "010R", "011L", "011R", "012L", "012R"]

def get_hour_diff(sub, dt_min):
    hour_diff = {
        "004" : -5,
        "005" : -5,
        "007" : -5,
        "008L" : 1,
        "008R" : 1,
        "009L" : 1,
        "009R" : 1,
        "010L" : 0,
        "010R" : 0,
        "011L" : 1,
        "011R" : 1,
        "012L" : 1,
        "012R" : 1
    }

    return hour_diff[sub]

from matplotlib.backends.backend_pdf import PdfPages

pdf_file = PdfPages("figures/suds_ratings_sessions_sub11_and_12.pdf")

for _, sub in enumerate(subs):

    if not os.path.exists(f"{sub}/{sub}_suds_ratings.csv"):
        print(f"No SUDS ratings found for {sub}. Skipping...")
        continue

    suds = pd.read_csv(f"{sub}/{sub}_suds_ratings.csv")
    suds["ts_datetime"] = pd.to_datetime(suds["ts_datetime"])
    # sort by ts_datetime
    suds = suds.sort_values(by="ts_datetime").reset_index(drop=True)

    # subtract 5h from ts_datetime
    suds["matched"] = False

    #file_path = f"/Users/Timon/Desktop/aDBS{sub}Home"
    file_path = f"/Users/Timon/Desktop/convert_path/aDBS{sub}"
    #if "L" in sub or "R" in sub:
    #    file_path = f"/Users/Timon/Desktop/aDBS{sub[:3]}Home_{sub[-1]}"
    sessions = [f for f in os.listdir(file_path) if f.startswith("Session") and os.path.isdir(os.path.join(file_path, f))]
    #pd.to_datetime(pd.Series([s[len("Session"):] for s in sessions]), unit="ms").sort_values().iloc[:10]
    #pd.Series([s[len("Session"):] for s in sessions]).loc[31]

    df_dur = []
    for session in np.sort(sessions):
        
        first_folder = [f for f in os.listdir(os.path.join(file_path, session)) if os.path.isdir(os.path.join(file_path, session, f))]
        if len(first_folder) == 0:
            continue
        else:
            first_folder = first_folder[0]
        path_first_folder = os.path.join(file_path, session, first_folder)
        if "NeuralTimeDomain.csv" in os.listdir(path_first_folder):
            df = pd.read_csv(os.path.join(path_first_folder, "NeuralTimeDomain.csv"))
            ts_min =  df["timestamp"].min()
            ts_dt_min = pd.to_datetime(ts_min, unit='ms')
            ts_max = df["timestamp"].max()
            ts_dt_max = pd.to_datetime(ts_max, unit='ms')

            hour_diff_sub = get_hour_diff(sub, ts_dt_min)
            df["timestamp"] = df["timestamp"] + hour_diff_sub * 3600 * 1000  # convert to milliseconds and add hour difference
            df_dur.append({
                "session": session,
                "ts_min" : ts_min,
                "ts_max" : ts_max,
                "ts_min_dt": ts_dt_min,
                "ts_max_dt": ts_dt_max
            })
            # print ts_min_dt and ts_max_dt datetime
            print(f"dt_min: {pd.to_datetime(df["timestamp"].min(), unit='ms')}, dt_max: {pd.to_datetime(df["timestamp"].max(), unit='ms')}")
            print()
            for idx, row in suds.iterrows():
                ts = row["timestamp"]
                if df["timestamp"].min()/1000 <= ts <= df["timestamp"].max()/1000:
                    suds.at[idx, "matched"] = True
                    suds.at[idx, "session"] = session
    df_dur = pd.DataFrame(df_dur)
    df_dur["ts_min_dt"] = pd.to_datetime(df_dur["ts_min"], unit='ms')
    df_dur.to_csv(f"{sub}/{sub}_sessions_durations.csv", index=False)
    df_dur["ts_max_dt_plot_plus_20h"] = pd.to_datetime(df_dur["ts_max"], unit='ms') + pd.Timedelta(hours=20)  # add 1 ms to max to include the last timestamp


    # make a plot were the ranges are shown shaded on a x-axis timeline

    plt.figure(figsize=(15, 6))
    for plt_idx in range(2):
        plt.subplot(2, 1, plt_idx + 1)
        for idx, row in df_dur.iterrows():
            if plt_idx == 0:
                plt.axvspan(row["ts_min_dt"], row["ts_max_dt_plot_plus_20h"], alpha=1)
            else:
                plt.axvspan(row["ts_min_dt"], row["ts_max_dt"], alpha=1)
        plt.plot(suds.query("matched == True")["ts_datetime"], suds.query("matched == True")["ratings"],
                "o", markersize=5, label="SUDS Ratings", color="green")
        plt.plot(suds.query("matched == False")["ts_datetime"], suds.query("matched == False")["ratings"],
                "o", markersize=5, label="SUDS Ratings (unmatched)", color="red")
        plt.xlabel("Timestamp")
        plt.ylabel("SUDS Ratings")
        plt.xticks(rotation=45)
        plt.title(f"Patient {sub} - Ephys Sessions and SUDS Ratings")
        # limit to range of first and last suds
        if plt_idx == 1:
            plt.xlim(suds["ts_datetime"].min() - pd.Timedelta(hours=10), suds["ts_datetime"].max()+ pd.Timedelta(hours=10))
        # place legend outside the plot
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
        plt.tight_layout()
    plt.title(f"Patient {sub}")
    pdf_file.savefig()  # saves the current figure into a pdf page
    #plt.show(block=True)
    plt.close()
pdf_file.close()
    #plt.savefig(f"figures/{sub}_suds_ratings_sessions.pdf", bbox_inches='tight')
    #plt.show(block=True)
