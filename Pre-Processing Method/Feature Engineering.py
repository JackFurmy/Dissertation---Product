import os
import pandas as pd
import numpy as np
from pathlib import Path


# 1) Folders
EMBER_MISSING_HANDLED = Path("cybersecurity_game/data/Malware Attack Patterns/Non-Institutional/EMBER/extracted/ember_2017_2/missing_handled")
FEATURE_FOLDER = EMBER_MISSING_HANDLED.parent / "feature_engineered"
FEATURE_FOLDER.mkdir(exist_ok=True)

# 2) Simple Hash Function
def hash_str(val: str, mod: int = 10**6) -> int:
    if pd.isna(val) or val == "":
        return -1
    return abs(hash(val)) % mod

def feature_engineering_ember(csv_path: Path):
    if not csv_path.is_file():
        print(f"[SKIP] Not a file: {csv_path}")
        return

    if csv_path.name.startswith("._"):
        print(f"[SKIP] Resource-fork => {csv_path.name}")
        return

    try:
        df = pd.read_csv(csv_path, low_memory=False)
        if df.empty:
            print(f"[WARN] {csv_path.name} is empty. Skipping.")
            return

        print(f"[INFO] Loaded '{csv_path.name}': {df.shape[0]} rows, {df.shape[1]} columns.")

        non_num_cols = df.select_dtypes(exclude=[np.number]).columns
        for col in non_num_cols:
            print(f"[INFO] Hashing column '{col}' in {csv_path.name}...")
            df[col] = df[col].astype(str).apply(hash_str)

        out_name = csv_path.stem + "_fe.csv"
        out_path = FEATURE_FOLDER / out_name
        df.to_csv(out_path, index=False)
        print(f"[DONE] => Wrote feature-engineered Ember CSV => {out_name}")

    except Exception as e:
        print(f"[ERROR] Could not process {csv_path.name}: {e}")

# 4) Main
def main():
    csv_files = list(EMBER_MISSING_HANDLED.glob("*.csv"))
    if not csv_files:
        print(f"[WARN] No CSV files found in {EMBER_MISSING_HANDLED}")
        return

    for csv_file in csv_files:
        feature_engineering_ember(csv_file)

    print("\n[DONE] Feature engineering (hashing strings) for Ember CSVs complete.")

if __name__ == "__main__":
    main()
