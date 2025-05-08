import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA

# Configuration
CHUNK_SIZE        = 50_000   
N_COMPONENTS      = 2       
LABEL_COL         = "label"  
TFIDF_PREFIX      = "tfidf_"

INPUT_FILE = "cybersecurity_game/data/Phishing Websites/Phishing Attack URL - non institutional/dataset/big/feature_engineered/train_step3_tfidf.csv"
OUTPUT_DIR = "cybersecurity_game/docs/Data Analytics Results/Phishing Attack URL"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_TOTAL_SAMPLES = 100_000   
SAMPLE_FRAC       = 0.10      

# PASS ONE: partial_fit the IncrementalPCA
def pass_one_partial_fit(csv_path, chunk_size=CHUNK_SIZE, n_components=N_COMPONENTS):ipca = IncrementalPCA(n_components=n_components)
    total_rows = 0

    print(f"[PASS1] partial_fit => {csv_path}, chunk_size={chunk_size}")
    reader = pd.read_csv(csv_path, chunksize=chunk_size, low_memory=True)
    for i, chunk in enumerate(reader):
        tfidf_cols = [c for c in chunk.columns if c.startswith(TFIDF_PREFIX)]
        if not tfidf_cols:
            continue

        X = chunk[tfidf_cols].values
        if len(X) == 0:
            continue

        ipca.partial_fit(X)
        total_rows += len(X)
        print(f"  [Chunk {i}] partial_fit size={len(X)}, total_rows={total_rows}")

    print(f"[PASS1] Done partial_fit. total_rows={total_rows}")
    return ipca

# PASS TWO: transform each chunk, sample in memory
def pass_two_transform_sample(csv_path, ipca, chunk_size=CHUNK_SIZE,max_total_samples=MAX_TOTAL_SAMPLES,sample_frac=SAMPLE_FRAC):

    rows_collected = []
    total_collected = 0

    print("[PASS2] => transform each chunk, random sample, store in memory.")
    reader = pd.read_csv(csv_path, chunksize=chunk_size, low_memory=True)
    for i, chunk in enumerate(reader):
        tfidf_cols = [c for c in chunk.columns if c.startswith(TFIDF_PREFIX)]
        if not tfidf_cols:
            continue

        X = chunk[tfidf_cols].values
        if len(X) == 0:
            continue

        X_ipca = ipca.transform(X)

        if LABEL_COL in chunk.columns:
            y = chunk[LABEL_COL].values
        else:
            y = np.zeros(len(X_ipca))

        chunk_len = len(X_ipca)
        sample_n = int(chunk_len * sample_frac)
        if sample_n <= 0:
            continue

        idx = np.random.choice(chunk_len, size=sample_n, replace=False)
        pc_sample = X_ipca[idx, :]
        label_sample = y[idx]

        merged_sample = np.column_stack([pc_sample, label_sample])
        rows_collected.append(merged_sample)

        total_collected += sample_n
        print(f"  [Chunk {i}] chunk_len={chunk_len}, sample_n={sample_n}, total_collected={total_collected}")
        if total_collected >= max_total_samples:
            print(f"[PASS2] Reached max_total_samples={max_total_samples}, stopping sampling.")
            break

    if not rows_collected:
        print("[PASS2] No data collected at all.")
        return pd.DataFrame(columns=["PC1","PC2","label"])

    all_data = np.vstack(rows_collected)
    df_pca = pd.DataFrame(all_data, columns=["PC1","PC2","label"])
    print(f"[PASS2] final shape of sampled PC data => {df_pca.shape}")
    return df_pca

# Plot & Save
def plot_pca_2d_and_save(df_pca, out_dir=OUTPUT_DIR, fname="pca_phishing.png"):
    if df_pca.empty:
        print("[PLOT] No data to plot, skipping.")
        return

    print(f"[PLOT] We'll plot {len(df_pca)} points from the sample.")
    pc1 = df_pca["PC1"].values
    pc2 = df_pca["PC2"].values
    y   = df_pca["label"].values

    plt.figure(figsize=(6,5))
    plt.scatter(pc1, pc2, c=y, cmap="bwr", alpha=0.6, edgecolor='k')
    plt.title("Incremental PCA (PC1 vs. PC2) - Sampled\n(Red=1=Phish, Blue=0=Legit)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()

    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=300)
    print(f"[PLOT] Saved PCA scatter => {out_path}")
    plt.show()

# Main
def main():
    ipca = pass_one_partial_fit(INPUT_FILE)
    df_pca = pass_two_transform_sample(INPUT_FILE, ipca)
    plot_pca_2d_and_save(df_pca, out_dir=OUTPUT_DIR, fname="pca_phishing.png")

if __name__ == "__main__":
    main()
