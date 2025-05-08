import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_iot23_duration_vs_bytes_chunked(
    input_csv: str = "cybersecurity_game/data/Network Traffic Data/Institutional/IoT-23/Malicous and Benign/conn.log_converted_final_malicious.csv",
    output_dir: str = "cybersecurity_game/docs/Data Analytics Results",
    chunk_size: int = 100_000,
    sample_per_chunk: int = 1_000
):
    os.makedirs(output_dir, exist_ok=True)

    sampled_dfs = []

    chunk_index = 0
    total_rows_processed = 0
    required_cols = {"duration", "orig_bytes", "resp_bytes", "label"}

# Read in Chunks & Sample 
    for chunk in pd.read_csv(input_csv, chunksize=chunk_size):
        chunk_index += 1
        rows_in_chunk = len(chunk)
        total_rows_processed += rows_in_chunk

        print(f"[DEBUG] Chunk {chunk_index} loaded with {rows_in_chunk} rows.")
        print("[DEBUG] Columns in chunk:", chunk.columns.tolist())

        missing = required_cols - set(chunk.columns)
        if missing:
            print(f"[DEBUG] Missing columns in chunk {chunk_index}: {missing}")
            continue

        if rows_in_chunk == 0:
            print(f"[DEBUG] Chunk {chunk_index} is empty, skipping.")
            continue

# Randomly sample
        sample_size = min(sample_per_chunk, rows_in_chunk)
        print(f"[DEBUG] Sampling {sample_size} rows from chunk {chunk_index}.")
        sample_df = chunk.sample(sample_size, random_state=42)
        sampled_dfs.append(sample_df)

    print(f"\n[DEBUG] Total rows processed from file: {total_rows_processed}")

    if not sampled_dfs:
        print("[DEBUG] No data frames were sampled. Possibly missing columns or empty file.")
        return

    df_small = pd.concat(sampled_dfs, ignore_index=True)
    print(f"[DEBUG] Final sampled DataFrame shape: {df_small.shape}")

    for col in ["duration", "orig_bytes", "resp_bytes"]:
        df_small = df_small[df_small[col] > 0]

    for col in ["duration", "orig_bytes", "resp_bytes"]:
        upper99 = df_small[col].quantile(0.99)
        df_small = df_small[df_small[col] <= upper99]

    print(f"[DEBUG] After removing <=0 and clipping top 1%, DF shape: {df_small.shape}")

    if df_small.empty:
        print("[DEBUG] All data got filtered out. Possibly extreme outliers or zero values.")
        return

# Boxplots with Log Scale

# Duration Boxplot
    plt.figure(figsize=(6,4))
    sns.boxplot(data=df_small, x='label', y='duration', showfliers=False)
    plt.yscale('log')
    plt.title('IoT-23: Duration by Label (Boxplot, Clipped, Log Scale)')
    plt.tight_layout()
    box_path_dur = os.path.join(output_dir, "IoT23_Boxplot_Duration_LogY_Clipped.png")
    plt.savefig(box_path_dur, dpi=300)
    plt.show()
    plt.close()

# Orig Bytes Boxplot
    plt.figure(figsize=(6,4))
    sns.boxplot(data=df_small, x='label', y='orig_bytes', showfliers=False)
    plt.yscale('log')
    plt.title('IoT-23: OrigBytes by Label (Boxplot, Clipped, Log Scale)')
    plt.tight_layout()
    box_path_orig = os.path.join(output_dir, "IoT23_Boxplot_OrigBytes_LogY_Clipped.png")
    plt.savefig(box_path_orig, dpi=300)
    plt.show()
    plt.close()

# Resp Bytes Boxplot
    plt.figure(figsize=(6,4))
    sns.boxplot(data=df_small, x='label', y='resp_bytes', showfliers=False)
    plt.yscale('log')
    plt.title('IoT-23: RespBytes by Label (Boxplot, Clipped, Log Scale)')
    plt.tight_layout()
    box_path_resp = os.path.join(output_dir, "IoT23_Boxplot_RespBytes_LogY_Clipped.png")
    plt.savefig(box_path_resp, dpi=300)
    plt.show()
    plt.close()

# Step 5: Optional Scatter (Log-Log)
    plt.figure(figsize=(6,4))
    sns.scatterplot(data=df_small, x='duration', y='orig_bytes', hue='label', alpha=0.6)
    plt.xscale('log')
    plt.yscale('log')
    plt.title("IoT-23: Duration vs OrigBytes (Sampled, Clipped, Log-Log Scatter)")
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, "IoT23_Scatter_Duration_vs_OrigBytes_LogLog_Clipped.png")
    plt.savefig(scatter_path, dpi=300)
    plt.show()
    plt.close()

    print("[DEBUG] Boxplots & scatter complete for IoT-23 (clipped, log scale).")


if __name__ == "__main__":
    analyze_iot23_duration_vs_bytes_chunked()

