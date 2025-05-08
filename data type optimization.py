import os
import pandas as pd
import numpy as np
from pathlib import Path

# 1) Configuration
EMBER_FE_FOLDER = Path("data/Malware Attack Patterns/Non-Institutional/EMBER/extracted/ember_2017_2/feature_engineered")

OPTIMIZED_FOLDER = EMBER_FE_FOLDER.parent / "data_optimization"
OPTIMIZED_FOLDER.mkdir(exist_ok=True)


# 2) Dtype Optimization Function
def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:

    print("[INFO] Starting dtype optimization...")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        col_dtype = str(df[col].dtype)
        if 'float' in col_dtype:
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif 'int' in col_dtype:
            df[col] = pd.to_numeric(df[col], downcast='integer')
    obj_cols = df.select_dtypes(include=['object']).columns
    for col in obj_cols:
        unique_count = df[col].nunique(dropna=True)
        if unique_count < 10_000:
            print(f"[INFO] Converting '{col}' to category (unique={unique_count})")
            df[col] = df[col].astype('category')

    print("[INFO] Dtype optimization done.")
    return df

# 3) Process One Ember CSV
def optimize_ember_csv(csv_path: Path):
    if not csv_path.is_file():
        print(f"[SKIP] Not a file => {csv_path}")
        return

    # Skip resource-fork files on mac
    if csv_path.name.startswith("._"):
        print(f"[SKIP] Resource-fork file: {csv_path.name}")
        return

    try:
        df = pd.read_csv(csv_path, low_memory=False)
        if df.empty:
            print(f"[WARN] {csv_path.name} is empty. Skipping.")
            return

        before_mem = df.memory_usage(deep=True).sum() / 1024**2
        df = optimize_dtypes(df)
        after_mem = df.memory_usage(deep=True).sum() / 1024**2
        savings = before_mem - after_mem

        out_name = csv_path.stem + "_optimized.csv"
        out_path = OPTIMIZED_FOLDER / out_name
        df.to_csv(out_path, index=False)

        print(f"[DONE] => {csv_path.name}: {before_mem:0.2f} MB -> {after_mem:0.2f} MB (saved {savings:0.2f} MB). Wrote {out_name}")

    except Exception as e:
        print(f"[ERROR] Could not process {csv_path.name}: {e}")

# 4) Main Execution
def main():
    csv_files = list(EMBER_FE_FOLDER.glob("*.csv"))
    if not csv_files:
        print(f"[WARN] No CSV files found in {EMBER_FE_FOLDER}")
        return

    for csv_file in csv_files:
        optimize_ember_csv(csv_file)

    print("\n[DONE] Data type optimization for Ember CSVs complete.")

if __name__ == "__main__":
    main()


