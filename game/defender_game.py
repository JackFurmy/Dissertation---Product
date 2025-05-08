from __future__ import annotations

import argparse
import os, sys, random, tempfile, re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import requests

from ml_integration_defender import (            
    DEFENDER_SCENARIOS as SCENARIOS,
)
from user_session import get_user_score, update_user_score


API_BASE = os.getenv(
    "CYBER_ML_API_BASE",
    "URL"  
).rstrip("/")


def backend_ingest(scenario_key: str, chunk_url: str) -> Dict[str, float]:
    resp = requests.post(f"{API_BASE}/defender/update",
                         json={"scenario_key": scenario_key,
                               "chunk_csv_url": chunk_url},
                         timeout=900)
    resp.raise_for_status()
    return resp.json()

def resource_path(rel_path: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.abspath(os.path.dirname(__file__)))
    return os.path.join(base, rel_path)


if getattr(sys, "frozen", False):                   
    os.chdir(sys._MEIPASS)                           

def _labelcol(df: pd.DataFrame) -> str:
    return "label" if "label" in df.columns else "Label"


def get_csv(url: str, nrows: Optional[int] = None) -> pd.DataFrame:
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        for chunk in r.iter_content(1 << 16):
            tmp.write(chunk)
        return pd.read_csv(tmp.name, nrows=nrows)


def _delta_to_points(delta: float, difficulty: float) -> int:
    return max(0, int(round(delta * 1000 * difficulty)))


def delta_to_points(delta: float, difficulty: float) -> int:      
    return _delta_to_points(delta, difficulty)


def play_round(uid: str, key: str) -> Dict[str, Any]:
    cfg = SCENARIOS[key]
    chunks = list(cfg["chunks"])
    random.shuffle(chunks)

    print("Chunks:")
    for i, u in enumerate(chunks, 1):
        df = get_csv(u, nrows=3)
        pos_pct = int((df[_labelcol(df)] == 1).mean() * 100)
        print(f"{i:>2}. {Path(u).name}   (pos {pos_pct:2d}%)")

    picks = input("Numbers to ingest ➜ ").strip()
    chosen = [c for i, c in enumerate(chunks, 1)
              if str(i) in {x.strip() for x in picks.split(",")}]

    if not chosen:
        print("[INFO] Nothing chosen – aborting round.")
        return {"delta": 0.0, "pts": 0, "strikes": 0, "total": get_user_score(uid)}


    first_res   = backend_ingest(key, chosen[0])
    last_score  = first_res["value"] - first_res["far"]
    total_delta = 0.0
    strikes     = 0


    for idx, url in enumerate(chosen[1:], start=2):
        res   = backend_ingest(key, url)
        score = res["value"] - res["far"]
        delta = score - last_score
        last_score = score

        if delta <= 0:
            strikes += 1
        else:
            total_delta += delta

        print(f"[{idx}] Δ={delta:+.4f}  score={score:.4f}  strikes={strikes}")

    pts = 0
    if strikes < 3 and total_delta > 0:
        pts = _delta_to_points(total_delta, cfg["difficulty"])
        if pts:
            update_user_score(uid, pts)

    return {"delta": total_delta,
            "pts": pts,
            "strikes": strikes,
            "total": get_user_score(uid)}


def main() -> None:
    if len(sys.argv) == 1:
        sys.argv += ["--user", "demo_user"]

    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    uid = ap.parse_args().user

    print("Scenarios:")
    for i, k in enumerate(SCENARIOS, 1):
        print(f"{i:>2}. {k}")
    pick = input("Number ➜ ").strip()
    if not pick.isdigit() or not (1 <= int(pick) <= len(SCENARIOS)):
        sys.exit("Invalid choice.")

    key  = list(SCENARIOS)[int(pick) - 1]
    res  = play_round(uid, key)

    print("\n=== Result ===")
    print(f"Δ Score : {res['delta']:+.4f}")
    print(f"+Points : {res['pts']}")
    print(f"Strikes : {res['strikes']}/3")
    print(f"Total   : {res['total']}")


if __name__ == "__main__":
    os.environ.setdefault("CYBER_ML_API_BASE",
                          "URL")
    try:
        main()
    except KeyboardInterrupt:
        print("\nbye")

