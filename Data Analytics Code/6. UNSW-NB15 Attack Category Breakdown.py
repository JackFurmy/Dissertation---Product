import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_unsw_nb15(
    input_csv: str = "cybersecurity_game/data/Network Traffic Data/Institutional/UNSW-NB15/unsw_final_validated.csv",
    output_dir: str = "cybersecurity_game/docs/Data Analytics Results",
    benign_label: int = 0
):
    os.makedirs(output_dir, exist_ok=True)

    print(f"[DEBUG] Reading UNSW-NB15 data from: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"[DEBUG] Data shape: {df.shape}")

    if 'attack_cat' not in df.columns:
        print("[ERROR] 'attack_cat' column not found. Cannot compute category breakdown.")
        return
    
    old_benign_count = (df['attack_cat'] == -1).sum()
    if old_benign_count > 0:
        print(f"[INFO] Replacing {old_benign_count} occurrences of 'attack_cat'=-1 with 0.")
        df['attack_cat'].replace(-1, 0, inplace=True)

# ALL Categories
    cat_counts_all = df['attack_cat'].value_counts(dropna=False).sort_values(ascending=False)
    total_records_all = len(df)
    cat_percent_all = (cat_counts_all / total_records_all) * 100.0

    print("\n[ALL] UNSW-NB15: Attack Category Counts (including benign=0)")
    print(cat_counts_all)
    print("\n[ALL] UNSW-NB15: Attack Category Percentages")
    print(cat_percent_all.round(2))
    
    cat_df_all = pd.DataFrame({
        "Category": cat_counts_all.index,
        "Count": cat_counts_all.values,
        "Percent": cat_percent_all.round(2)
    })
    print("\n[ALL] Category breakdown table:")
    print(cat_df_all.to_string(index=False))

# Bar Plot - All categories
    plt.figure(figsize=(8,4))
    sns.barplot(x=cat_counts_all.index, y=cat_counts_all.values, color='blue')
    plt.xticks(rotation=45, ha='right')
    plt.title("UNSW-NB15: Attack Category Distribution (ALL)")
    plt.xlabel("Attack Category")
    plt.ylabel("Number of Records")
    plt.tight_layout()
    plot_path_all = os.path.join(output_dir, "UNSWNB15_AttackCategoryDistribution_All.png")
    plt.savefig(plot_path_all, dpi=300)
    plt.show()
    plt.close()
    print(f"[DEBUG] All-categories bar plot saved to: {plot_path_all}")

# EXCLUDING BENIGN=0 
    df_attacks_only = df[df['attack_cat'] != benign_label].copy()
    total_records_attack = len(df_attacks_only)

    if total_records_attack == 0:
        print(f"[DEBUG] No records remain after excluding attack_cat={benign_label}. No second plot to show.")
        return

    cat_counts_attack = df_attacks_only['attack_cat'].value_counts(dropna=False).sort_values(ascending=False)
    cat_percent_attack = (cat_counts_attack / total_records_attack) * 100.0

    print(f"\n[ATTACK-ONLY] Excluding benign label={benign_label}")
    print("[ATTACK-ONLY] Attack Category Counts:")
    print(cat_counts_attack)
    print("\n[ATTACK-ONLY] Attack Category Percentages:")
    print(cat_percent_attack.round(2))

    cat_df_attack = pd.DataFrame({
        "Category": cat_counts_attack.index,
        "Count": cat_counts_attack.values,
        "Percent": cat_percent_attack.round(2)
    })
    print("\n[ATTACK-ONLY] Category breakdown table:")
    print(cat_df_attack.to_string(index=False))

# Bar Plot 
    plt.figure(figsize=(8,4))
    sns.barplot(x=cat_counts_attack.index, y=cat_counts_attack.values, color='orange')
    plt.xticks(rotation=45, ha='right')
    plt.title(f"UNSW-NB15: Attack Category Distribution (Excluding {benign_label})")
    plt.xlabel("Attack Category")
    plt.ylabel("Number of Records")
    plt.tight_layout()
    plot_path_attack = os.path.join(output_dir, "UNSWNB15_AttackCategoryDistribution_NoBenign.png")
    plt.savefig(plot_path_attack, dpi=300)
    plt.show()
    plt.close()
    print(f"[DEBUG] Attack-only bar plot saved to: {plot_path_attack}")

    print("UNSW-NB15 attack category analysis complete (both ALL and EXCLUDING benign=0).")

if __name__ == "__main__":
    analyze_unsw_nb15()
