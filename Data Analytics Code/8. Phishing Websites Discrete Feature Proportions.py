import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_phishing_strong_moderate_features(
    input_csv: str = "cybersecurity_game/data/Phishing Websites/Phishing Websites/Training Dataset_converted_optimized.csv",
    output_dir: str = "cybersecurity_game/docs/Data Analytics Results"
):

    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[DEBUG] Reading phishing dataset from: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"[DEBUG] Data shape: {df.shape}")

    if 'Label' not in df.columns:
        print("[ERROR] 'Label' column not found. Aborting.")
        return
    
    strong_features = [
        "Prefix_Suffix",
        "having_Sub_Domain",
        "SSLfinal_State",
        "web_traffic",
        "Links_pointing_to_page",
    ]

    moderate_features = [
        "Domain_registeration_length",
        "URL_of_Anchor",
        "Request_URL",
        "SFH",
        "Redirect",
        "Abnormal_URL",
        "age_of_domain",
        "DNSRecord",
        "Page_Rank"
    ]

    selected_features = strong_features + moderate_features

    selected_features = [f for f in selected_features if f in df.columns]
    print(f"[INFO] Plotting {len(selected_features)} total features (strong + moderate).")

    chunk_size = 5
    for start_idx in range(0, len(selected_features), chunk_size):
        subset = selected_features[start_idx : start_idx + chunk_size]

        fig, axes = plt.subplots(
            nrows=1, 
            ncols=len(subset), 
            figsize=(5*len(subset), 4),
            sharey=True
        )
        if len(subset) == 1:
            axes = [axes]

        for idx, feature in enumerate(subset):
            ax = axes[idx]
            ctab = pd.crosstab(df[feature], df['Label'], normalize='columns', dropna=False)
            ctab_t = ctab.transpose()

            ctab_t.plot(
                kind='bar', stacked=True, ax=ax,
                legend=False,
                colormap='Set2'
            )
            ax.set_title(feature)
            ax.set_xlabel("Label (0=Benign, 1=Malicious)")
            if idx == 0:
                ax.set_ylabel("Proportion of Feature Values")
            ax.tick_params(axis='x', labelrotation=0)

        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, title="Feature Values", bbox_to_anchor=(1.05, 0.5), loc='center left')
        plt.tight_layout()

        start_num = start_idx + 1
        end_num = start_idx + len(subset)
        file_name = f"Phishing_StrongModerate_{start_num}_to_{end_num}.png"
        out_path = os.path.join(output_dir, file_name)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        print(f"[DEBUG] Saved figure for features {subset} => {out_path}")

    print("\n[INFO] Done. Created grouped plots for strong + moderate features in sets of 5.")

if __name__ == "__main__":
    analyze_phishing_strong_moderate_features()



