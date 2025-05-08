import os
import pandas as pd
from pathlib import Path

# 1) Configuration
LINUX_OPTIMIZED_FOLDER = Path('/data/Concept Drift and Adaptive Learning/TON_IoT Dataset/Testing/Train_Test_Linux_dataset/data_optimization')

FINAL_FOLDER = LINUX_OPTIMIZED_FOLDER.parent / "final_validated"
FINAL_FOLDER.mkdir(exist_ok=True)

# 2) Validation & Final Export
def final_validation_export(df: pd.DataFrame, file_name: str) -> pd.DataFrame:
    print(f"\n[INFO] Final Validation Checks for {file_name}:")

    missing_sum = df.isna().sum()
    print("  Missing values:\n", missing_sum)

    duplicates_count = df.duplicated().sum()
    print(f"  Duplicates: {duplicates_count}")
    
    if "ts" in df.columns:
        df.sort_values(by="ts", inplace=True)
        print("[INFO] Sorted rows by 'ts'.")
    else:
        print("[WARN] No 'ts' column found; skipping sort.")

    return df

# 3) Save final data
def save_final_data(df: pd.DataFrame, out_path: Path):
    df.to_csv(out_path, index=False)
    print(f"[DONE] Exported final => {out_path.name}")

# 4) Main
def main():
    for csv_file in LINUX_OPTIMIZED_FOLDER.glob("*optimized.csv"):
        if not csv_file.is_file():
            continue
        
        if csv_file.name.startswith("._"):
            print(f"[SKIP] Resource-fork file => {csv_file.name}")
            continue

        print(f"[INFO] Reading optimized data => {csv_file.name}")
        df = pd.read_csv(csv_file, low_memory=False)
        if df.empty:
            print(f"[WARN] {csv_file.name} is empty. Skipping.")
            continue

        final_df = final_validation_export(df, csv_file.name)
        print(f"[INFO] Final shape for {csv_file.name}: {final_df.shape}")

        out_name = csv_file.stem + "_final.csv"
        out_path = FINAL_FOLDER / out_name
        save_final_data(final_df, out_path)

    print("\n[ALL DONE] Linux final validation complete.")

if __name__ == "__main__":
    main()

