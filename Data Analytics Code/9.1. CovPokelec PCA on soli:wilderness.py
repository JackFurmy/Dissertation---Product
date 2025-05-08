import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def analyze_covpokelec(
    input_csv: str = "cybersecurity_game/data/Concept Drift and Adaptive Learning/OpenML/CovPokElec_converted.csv",
    output_dir: str = "cybersecurity_game/docs/Data Analytics Results"
):

    os.makedirs(output_dir, exist_ok=True)

    print(f"[DEBUG] Reading CovPokElec data from: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"[DEBUG] Data shape: {df.shape}")
    print("[DEBUG] Columns:", df.columns.tolist())

    soil_cols = [c for c in df.columns if "Soil_Type" in c]
    wild_cols = [c for c in df.columns if "Wilderness_Area" in c]
    combined_cols = soil_cols + wild_cols
    print(f"[DEBUG] Found {len(soil_cols)} soil cols, {len(wild_cols)} wilderness cols.")

    if not combined_cols:
        print("[ERROR] No Soil_Type or Wilderness_Area columns found. Aborting PCA.")
        return
    
    X = df[combined_cols].astype(float)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X)  
    print("[DEBUG] PCA explained variance ratio:", pca.explained_variance_ratio_)

    hue_col = None
    if "class" in df.columns:
        hue_col = df["class"]

    plt.figure(figsize=(6,4))
    if hue_col is not None:
        sns.scatterplot(x=pca_result[:,0], y=pca_result[:,1], hue=hue_col, alpha=0.5)
    else:
        plt.scatter(pca_result[:,0], pca_result[:,1], alpha=0.5)

    plt.title("CovPokElec: PCA on Soil/Wilderness Features")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "CovPokElec_SoilWilderness_PCA.png")
    plt.savefig(out_path, dpi=300)
    plt.show()
    plt.close()

    print(f"[INFO] PCA scatter saved to: {out_path}")

if __name__ == "__main__":
    analyze_covpokelec()



