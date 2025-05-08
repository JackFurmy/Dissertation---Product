import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def analyze_gas(file_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(file_path)
    print(f"Data loaded from: {file_path}")
    print(f"DataFrame shape: {df.shape}\n")
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    
    sns.boxplot(data=df, x='Class', y='V1', showfliers=False, ax=axes[0])
    axes[0].set_yscale('log')  
    axes[0].set_title("Gas (OPENML): V1 by Class (Log Scale, Outliers Hidden)")
    axes[0].set_ylabel("V1 (log scale)")
    
    sns.boxplot(data=df, x='Class', y='V2', showfliers=False, ax=axes[1])
    axes[1].set_yscale('log')  
    axes[1].set_title("Gas (OPENML): V2 by Class (Log Scale, Outliers Hidden)")
    axes[1].set_ylabel("V2 (log scale)")
    
    plt.tight_layout()
    
    out_file = os.path.join(output_dir, "Gas_V1_V2_boxplots_log.png")
    plt.savefig(out_file, dpi=120)
    plt.show()
    
    print(f"Boxplots saved to: {out_file}\n")


if __name__ == "__main__":
    input_file = "cybersecurity_game/data/Concept Drift and Adaptive Learning/OpenML/gas_converted.csv"
    output_path = "cybersecurity_game/docs/Data Analytics Results"
    
    analyze_gas(input_file, output_path)


