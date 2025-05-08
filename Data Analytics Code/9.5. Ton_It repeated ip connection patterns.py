import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def analyze_ton_it(file_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(file_path)
    print(f"Data loaded from: {file_path}")
    print(f"DataFrame shape: {df.shape}\n")
 
    malicious_df = df[df['label'] == 1]
    print(f"Malicious flows count: {len(malicious_df)}\n")
    
    pair_counts = (
        malicious_df
        .groupby(['src_ip', 'dst_ip'])
        .size()
        .reset_index(name='count')
        .sort_values('count', ascending=False)
    )
    
    print("TON_IT: Top 10 repeated malicious IP pairs:")
    print(pair_counts.head(10), "\n")
    
    top_10 = pair_counts.head(10)
    plt.figure(figsize=(8,5))
    sns.barplot(
        data=top_10, 
        x='count', 
        y=top_10.apply(lambda row: f"{row['src_ip']} -> {row['dst_ip']}", axis=1),
        palette='viridis'
    )
    plt.xlabel("Count of Malicious Connections")
    plt.ylabel("Src IP -> Dst IP")
    plt.title("Top 10 Repeated Malicious IP Pairs (label=1)")
    plt.tight_layout()
    
    out_file = os.path.join(output_dir, "ton_it_repeated_ip_pairs.png")
    plt.savefig(out_file, dpi=120)
    plt.show()
    print(f"Bar chart saved to: {out_file}\n")


if __name__ == "__main__":
    input_file = "cybersecurity_game/data/Concept Drift and Adaptive Learning/TON_IoT Dataset/Training/Network/final_validated/Network_dataset_6_cleaned_step3_fe_optimized_final.csv"
    output_path = "cybersecurity_game/docs/Data Analytics Results"
    
    analyze_ton_it(input_file, output_path)

