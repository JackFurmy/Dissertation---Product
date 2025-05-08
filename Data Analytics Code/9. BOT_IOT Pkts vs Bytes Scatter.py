import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_bot_iot(
    input_csv: str = "cybersecurity_game/data/Dynamic Threat Simulation/Bot-IoT/Dataset/Training/final_validated/UNSW_2018_IoT_Botnet_Dataset_1_cleaned_step3_fe_optimized_final.csv",
    output_dir: str = "cybersecurity_game/docs/Data Analytics Results"
):
 
    os.makedirs(output_dir, exist_ok=True)

    print(f"[DEBUG] Reading CSV from: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"[DEBUG] Data shape: {df.shape}")
    print("[DEBUG] Columns:", df.columns.tolist())

    if 'label' not in df.columns and 'attack' in df.columns:
        df.rename(columns={'attack': 'label'}, inplace=True)
        print("[DEBUG] Renamed 'attack' -> 'label'.")

    needed = {'label','pkts','bytes'}
    missing = needed - set(df.columns)
    if missing:
        print(f"[ERROR] Missing columns: {missing}. Aborting plot.")
        return

    print("\n[DEBUG] df.head(10):")
    print(df.head(10))

    print("\n[DEBUG] 'pkts' descriptive stats:")
    print(df['pkts'].describe())
    print(f"Min pkts: {df['pkts'].min()} | Max pkts: {df['pkts'].max()}")

    plt.figure(figsize=(6,4))
    sns.scatterplot(data=df, x='pkts', y='bytes', hue='label')
    plt.title("Bot-IoT: pkts vs. bytes (colored by label)")
    plt.xlabel("pkts")
    plt.ylabel("bytes")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "BotIOT_pkts_vs_bytes_scatter.png")
    plt.savefig(out_path, dpi=300)
    plt.show()
    plt.close()

    print(f"[INFO] Scatter plot saved to: {out_path}")

if __name__ == "__main__":
    analyze_bot_iot()



