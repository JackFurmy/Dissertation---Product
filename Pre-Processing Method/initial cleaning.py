import os
import pandas as pd
from pathlib import Path


# 1) Configuration
EMBER_CSV_FOLDER = Path("data/Malware Attack Patterns/Non-Institutional/EMBER/extracted/ember_2017_2/converted_csv")

OUTPUT_FOLDER = EMBER_CSV_FOLDER.parent / "initial_cleaned"
OUTPUT_FOLDER.mkdir(exist_ok=True)


KEEP_COLS = [
    "sha256",
    "label",
    "histogram",
    "byteentropy",
    "general",
    "strings",   
    "imports"   
]


# 2) Clean a single CSV
def clean_ember_csv(csv_path: Path):
    if not csv_path.is_file():
        print(f"[SKIP] Not a file => {csv_path}")
        return
    if csv_path.name.startswith("._"):
        print(f"[SKIP] Resource-fork => {csv_path.name}")
        return

    try:
        df = pd.read_csv(csv_path, low_memory=True)
        if df.empty:
            print(f"[WARN] {csv_path.name} is empty or unreadable.")
            return

        print(f"[INFO] Loaded '{csv_path.name}': {len(df)} rows, {len(df.columns)} columns.")


        existing_cols = [c for c in KEEP_COLS if c in df.columns]
        df = df[existing_cols] 

        df.replace('-', pd.NA, inplace=True)

        out_name = csv_path.stem + "_cleaned.csv"
        out_path = OUTPUT_FOLDER / out_name
        df.to_csv(out_path, index=False)
        print(f"[DONE] => Saved cleaned CSV => {out_name}")

    except Exception as ex:
        print(f"[ERROR] Could not process {csv_path.name}: {ex}")

# 3) Main
def main():
    csv_files = list(EMBER_CSV_FOLDER.glob("*.csv"))
    if not csv_files:
        print(f"[WARN] No CSV files found in {EMBER_CSV_FOLDER}")
        return

    for csv_file in csv_files:
        clean_ember_csv(csv_file)

    print("\n[ALL DONE] Ember CSV initial cleaning complete.")

if __name__ == "__main__":
    main()

