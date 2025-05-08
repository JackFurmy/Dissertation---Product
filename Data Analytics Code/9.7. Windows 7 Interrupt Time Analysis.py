import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def analyze_windows7(file_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(file_path, low_memory=False)
    print(f"Data loaded from: {file_path}")
    print(f"DataFrame shape: {df.shape}\n")
    
    interrupt_col = "Processor(_Total) pct_ Interrupt Time"
    
    df[interrupt_col] = (
        df[interrupt_col]
        .astype(str)
        .str.replace(r"[^\d.\-eE]", "", regex=True)   
        .replace(r"^\s*$", np.nan, regex=True)       
        .astype(float, errors="raise")               
    )
    
    df.dropna(subset=[interrupt_col], inplace=True)
    
    if "label" in df.columns:
        hue_col = "label"
    elif "type" in df.columns:
        hue_col = "type"
    else:
        hue_col = None
    
    plt.figure(figsize=(6,4))
    sns.kdeplot(
        data=df,
        x=interrupt_col,
        hue=hue_col,
        fill=True,
        common_norm=False
    )
    
    title_str = "Windows 7: Interrupt Time (KDE)"
    if hue_col:
        title_str += f" by {hue_col}"
    plt.title(title_str)
    plt.tight_layout()
    
    out_file = os.path.join(output_dir, "windows7_interrupt_time_kde.png")
    plt.savefig(out_file, dpi=120)
    plt.show()
    
    print(f"KDE plot saved to: {out_file}\n")


# Example usage:
if __name__ == "__main__":
    input_file = (
        "cybersecurity_game/data/Concept Drift and Adaptive Learning/TON_IoT Dataset/Training/Windows/final_validated/windows7_dataset_cleaned_step3_fe_optimized_final.csv"
    )
    output_path = "cybersecurity_game/docs/Data Analytics Results"
    
    analyze_windows7(input_file, output_path)
