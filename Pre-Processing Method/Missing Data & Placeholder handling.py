import os
import pandas as pd
from pathlib import Path

# 1) Paths
EMBER_CLEANED_FOLDER = Path("data/Malware Attack Patterns/Non-Institutional/EMBER/extracted/ember_2017_2/initial_cleaned")

MISSING_HANDLED_FOLDER = EMBER_CLEANED_FOLDER.parent / "missing_handled"
MISSING_HANDLED_FOLDER.mkdir(exist_ok=True)

# 2) handle_missing_data
def handle_missing_data(df: pd.DataFrame, file_name: str) -> pd.DataFrame:
    missing_summary = df.isna().sum().to_frame("MissingCount")
    missing_summary["MissingPct"] = (missing_summary["MissingCount"] / len(df)) * 100
    print(f"\n[INFO] Missing data summary for {file_name}:")
    print(missing_summary.sort_values("MissingCount", ascending=False))
 
    placeholders = ["?", "#######", "-", ""]
    for ph in placeholders:
        df = df.replace(ph, pd.NA)

    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        median_val = df[col].median(skipna=True)
        df[col] = df[col].fillna(median_val)

    if "label" in df.columns:
        before = len(df)
        df.dropna(subset=["label"], inplace=True)
        after = len(df)
        dropped = before - after
        if dropped > 0:
            print(f"[INFO] Dropped {dropped} rows missing 'label' from {file_name}")

    return df


# 3) Process One CSV
def process_ember_csv(csv_path: Path):
    if not csv_path.is_file():
        print(f"[SKIP] Not a file: {csv_path}")
        return

    # Skip resource-fork
    if csv_path.name.startswith("._"):
        print(f"[SKIP] Resource-fork => {csv_path.name}")
        return

    try:
        df = pd.read_csv(csv_path, low_memory=False)
        if df.empty:
            print(f"[WARN] {csv_path.name} is empty or unreadable.")
            return

        cleaned_df = handle_missing_data(df, csv_path.name)

        out_name = csv_path.stem + "_step3.csv"
        out_path = MISSING_HANDLED_FOLDER / out_name
        cleaned_df.to_csv(out_path, index=False)
        print(f"[INFO] => Saved missing-data-handled CSV => {out_name}")

    except Exception as e:
        print(f"[ERROR] Could not process {csv_path.name}: {e}")


# 4) Main
def main():
    csv_files = list(EMBER_CLEANED_FOLDER.glob("*.csv"))
    if not csv_files:
        print(f"[WARN] No CSV files found in {EMBER_CLEANED_FOLDER}")
        return

    for csv_file in csv_files:
        process_ember_csv(csv_file)

    print("\n[DONE] Missing data & placeholder handling for Ember CSVs.")

if __name__ == "__main__":
    main()
