from __future__ import annotations

import importlib, pathlib, sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

def _patch_parents_index_error() -> None:
    original = pathlib.Path.parents.__getitem__

    def safe_getitem(self: pathlib.Path, idx: int):         
        try:
            return original(self, idx)
        except IndexError:                                
            return self.parent

    pathlib.Path.parents.__getitem__ = safe_getitem


try:                                                        
    import ml_integration
except IndexError as exc:                                   
    if "parents" in str(exc):
        _patch_parents_index_error()
        if "ml_integration" in sys.modules:
            del sys.modules["ml_integration"]
        ml_integration = importlib.import_module("ml_integration")
    else:
        raise


import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    from river.metrics import F1                    
except (ImportError, ModuleNotFoundError):
    class F1:                                        
        bigger_is_better = True
        def __init__(self):
            self.tp = self.fp = self.fn = 0
        def update(self, y, yp, **kw):
            self.tp += y == yp == 1
            self.fp += yp == 1 and y == 0
            self.fn += yp == 0 and y == 1
            return self
        def get(self) -> float:
            p = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0
            r = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
            return 2 * p * r / (p + r) if (p + r) else 0.0
        def works_with(self, _): return True
    print("[WARN] River not found – using stub F1 implementation")

from ml_integration import (                      
    load_baseline_model,
    fetch_dataset_via_url,
    fetch_dataset_from_firebase,
    train_dataset_with_removed_cols,
)
from ml_integration_defender import (         
    IncrementalModelManager,
    DEFENDER_SCENARIOS,
    _dl_csv,
)

ml_integration.BASELINE_MODELS_DIR = Path(__file__).parent / "models" / "models"
print("[INFO] BASELINE_MODELS_DIR →", ml_integration.BASELINE_MODELS_DIR)

app = FastAPI(title="Cyber-ML Game backend")

class AttackReq(BaseModel):
    dataset: str
    removed_cols: List[str] = []

class AttackResp(BaseModel):
    accuracy: float
    metrics: Dict[str, float]

class DefReq(BaseModel):
    scenario_key: str
    chunk_csv_url: str

class DefResp(BaseModel):
    metric: str          
    value:  float       
    far:    float       

_ATTACK_CACHE: Dict[str, Tuple[List[str], float, Dict[str, float], str | None]] = {}
_DEFENDER_CACHE: Dict[str, IncrementalModelManager] = {}

def _labelcol(df: pd.DataFrame) -> str:
    return "label" if "label" in df.columns else "Label"

@lru_cache(maxsize=16)
def _cached_validation_df(scenario_key: str) -> pd.DataFrame:

    url = DEFENDER_SCENARIOS[scenario_key]["val"]
    print(f"[CACHE] loading validation split → {scenario_key}")
    df = _dl_csv(url)
    if len(df) > 25_000:                                
        df = df.sample(n=25_000, random_state=42)
    return df

def _process_remote_chunk(
    mgr: IncrementalModelManager,
    scenario_key: str,
    chunk_url: str,
) -> Tuple[str, float, float]:

    df_chunk = _dl_csv(chunk_url)
    lbl_c    = _labelcol(df_chunk)
    for _, row in df_chunk.iterrows():
        mgr.update_one(row.drop(labels=[lbl_c]).to_dict(),
                       int(row[lbl_c] or 0))

    df_val = _cached_validation_df(scenario_key)
    lbl_v  = _labelcol(df_val)

    f1 = F1(); tn = fp = 0
    for _, row in df_val.iterrows():
        y_true = int(row[lbl_v] or 0)
        y_pred = mgr.predict_one(row.drop(labels=[lbl_v]).to_dict()) or 0
        f1.update(y_true, y_pred)
        if y_true == 0:
            tn += y_pred == 0
            fp += y_pred == 1

    far = fp / (tn + fp) if (tn + fp) else 0.0
    mgr.save()
    return "f1", float(f1.get()), float(far)

@app.post("/attacker/round", response_model=AttackResp)
def attacker_round(req: AttackReq):
    if not req.dataset:
        raise HTTPException(400, "dataset required")

    if req.dataset not in _ATTACK_CACHE:
        model, cols, acc, met, _, url = load_baseline_model(req.dataset)
        if model is None:
            raise HTTPException(404, f"unknown dataset {req.dataset}")
        _ATTACK_CACHE[req.dataset] = (cols, acc, met, url)

    cols, base_acc, base_met, url = _ATTACK_CACHE[req.dataset]
    df = fetch_dataset_via_url(url) if url else fetch_dataset_from_firebase(req.dataset)
    if df is None:
        raise HTTPException(500, "dataset download failed")

    tmp_path = Path("/tmp") / f"{req.dataset}.pkl"
    res = train_dataset_with_removed_cols(
        req.dataset, df, tmp_path, removed_cols=req.removed_cols
    )
    if res is None:
        raise HTTPException(500, "training failed")

    return AttackResp(
        accuracy=float(res["baseline_acc"]),
        metrics={k: float(v) for k, v in res["metrics"].items()},
    )

@app.post("/defender/update", response_model=DefResp)
def defender_update(req: DefReq):
    if req.scenario_key not in DEFENDER_SCENARIOS:
        raise HTTPException(404, "unknown scenario")

    mgr = _DEFENDER_CACHE.get(req.scenario_key)
    if mgr is None:
        mgr = IncrementalModelManager(req.scenario_key)   
        _DEFENDER_CACHE[req.scenario_key] = mgr

    try:
        metric, f1_val, far_val = _process_remote_chunk(
            mgr, req.scenario_key, req.chunk_csv_url
        )
    except Exception as exc:
        raise HTTPException(500, f"chunk failed: {exc}")

    return DefResp(metric=metric, value=f1_val, far=far_val)

@app.get("/ping")
def ping() -> Dict[str, bool]:
    return {"ok": True}

asgi_app = app
