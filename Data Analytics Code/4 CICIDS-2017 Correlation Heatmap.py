import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def analyze_cicids_correlation_with_label(
    input_csv: str = "cybersecurity_game/data/Network Traffic Data/Institutional/CICIDS 2017/Traffic Label/cicids2017_final_validated.csv",
    output_dir: str = "cybersecurity_game/docs/Data Analytics Results"
):

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_csv)
    print("Data loaded. Shape:", df.shape)


    if df['label'].dtype == object:
        df['label_numeric'] = df['label'].apply(lambda x: 0 if x.upper() == "BENIGN" else 1)
        label_col = 'label_numeric'
    else:
        label_col = 'label'

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print("Numeric columns found:", numeric_cols)

    corr_matrix = df[numeric_cols].corr()
    print("Correlation matrix shape:", corr_matrix.shape)

# plot the correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        cmap='coolwarm',
        square=False,   
        annot=False,     
        cbar_kws={"shrink": .8}
    )
    plt.title('CICIDS-2017: Correlation Heatmap (with Label)')
    plt.tight_layout()

    heatmap_path = os.path.join(output_dir, "CICIDS2017_CorrelationHeatmap_withLabel.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.show()
    plt.close()

    print(f"Correlation heatmap saved to: {heatmap_path}")

if __name__ == "__main__":
    analyze_cicids_correlation_with_label()
