import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def analyze_comet_split_panels(file_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(file_path)
    print(f"Data loaded from: {file_path}")
    print(f"DataFrame shape: {df.shape}\n")

    df_label0 = df[df['label'] == 0]
    df_label1 = df[df['label'] == 1]
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    sns.scatterplot(
        data=df_label0, x='relative_time', y='energy_deposit', 
        ax=axes[0], color='blue', alpha=0.6
    )
    axes[0].set_title("Comet: Label=0")
    axes[0].set_xlabel("Relative Time")
    axes[0].set_ylabel("Energy Deposit")

    sns.scatterplot(
        data=df_label1, x='relative_time', y='energy_deposit', 
        ax=axes[1], color='orange', alpha=0.6
    )
    axes[1].set_title("Comet: Label=1")
    axes[1].set_xlabel("Relative Time")
    axes[1].set_ylabel("Energy Deposit")

    plt.tight_layout()

    out_file = os.path.join(output_dir, "comet_energy_vs_time_split.png")
    plt.savefig(out_file, dpi=120)
    plt.show()

    print(f"Scatter plots saved to: {out_file}\n")


if __name__ == "__main__":
    input_file = "cybersecurity_game/data/Concept Drift and Adaptive Learning/OpenML/comet_converted.csv"
    output_path = "cybersecurity_game/docs/Data Analytics Results"
    
    analyze_comet_split_panels(input_file, output_path)


