import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def analyze_windows10(file_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(file_path, low_memory=False)
    print(f"Data loaded from: {file_path}")
    print(f"DataFrame shape: {df.shape}\n")
    print("Columns in DataFrame:\n", df.columns.tolist(), "\n")
    
    cpu_col = "Processor_pct_ User_Time"
    mem_col = "Process_Working_Set_ Private"
    
    if "label" in df.columns:
        hue_col = "label"
    elif "type" in df.columns:
        hue_col = "type"
    else:
        hue_col = None  
    
    missing_cols = [c for c in [cpu_col, mem_col, hue_col] if c and c not in df.columns]
    if missing_cols:
        print(f"ERROR: The following columns are not in the CSV: {missing_cols}")
        return
    
    plt.figure(figsize=(5,4))
    if hue_col:
        sns.scatterplot(data=df, x=cpu_col, y=mem_col, hue=hue_col, alpha=0.7)
    else:
        sns.scatterplot(data=df, x=cpu_col, y=mem_col, alpha=0.7)
    
    plt.title("Windows 10: CPU vs Memory")
    if hue_col:
        plt.title(f"Windows 10: CPU vs Memory by {hue_col}")
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "windows10_cpu_mem_correlation.png")
    plt.savefig(plot_path, dpi=120)
    plt.show()
    
    print(f"Scatter plot saved to: {plot_path}\n")


if __name__ == "__main__":
    input_file = (
        "cybersecurity_game/data/Concept Drift and Adaptive Learning/TON_IoT Dataset/Training/Windows/final_validated/windows10_dataset_cleaned_step3_fe_optimized_final.csv"
    )
    output_path = (
        "cybersecurity_game/docs/Data Analytics Results"
    )
    
    analyze_windows10(input_file, output_path)
