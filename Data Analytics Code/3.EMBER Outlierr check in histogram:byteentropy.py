import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def analyze_ember_with_extra_columns(
    input_csv: str = "cybersecurity_game/data/Malware Attack Patterns/Non-Institutional/EMBER/extracted/ember_2017_2/data_optimization/train_features_5_cleaned_step3_fe_optimized.csv",
    output_dir: str = "cybersecurity_game/docs/Data Analytics Results"
):
    os.makedirs(output_dir, exist_ok=True)

# Load data
    df = pd.read_csv(input_csv)
    print("Data loaded:", df.shape)
    needed_cols = ['histogram', 'byteentropy', 'general', 'strings', 'imports', 'label']
    missing = set(needed_cols) - set(df.columns)
    if missing:
        print(f"Missing columns: {missing}. Please adapt code or your CSV.")
        return

# Correlation Matrix & Heatmap
    numeric_cols = ['histogram', 'byteentropy', 'general', 'strings', 'imports']
    corr_matrix = df[numeric_cols].corr()
    print("Correlation matrix:\n", corr_matrix)

    plt.figure(figsize=(6,5))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
    plt.title("Correlation among EMBER features")
    heatmap_path = os.path.join(output_dir, "EMBER_corr_heatmap.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.show()
    plt.close()
    print(f"Correlation heatmap saved to: {heatmap_path}")

# Pair Plot (sampled) 
    sample_df = df.sample(min(len(df), 5000), random_state=42)
    sns.pairplot(
        data=sample_df,
        vars=numeric_cols,
        hue='label',
        palette='Set2',
        diag_kind='kde'
    )
    pairplot_path = os.path.join(output_dir, "EMBER_pairplot.png")
    plt.savefig(pairplot_path, dpi=300)
    plt.close()
    print(f"Pair plot (sampled) saved to: {pairplot_path}")

# Label-Based Distributions 
    for col in numeric_cols:
        plt.figure(figsize=(6,4))
        sns.histplot(data=df, x=col, hue='label', kde=True, element='step')
        plt.title(f"EMBER: Distribution of '{col}' by label")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()

        dist_path = os.path.join(output_dir, f"EMBER_{col}_by_label.png")
        plt.savefig(dist_path, dpi=300)
        plt.show()
        plt.close()
        print(f"Label-based distribution for {col} saved to: {dist_path}")

if __name__ == "__main__":
    analyze_ember_with_extra_columns()

