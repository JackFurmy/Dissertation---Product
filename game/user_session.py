

from __future__ import annotations

import os
import sys
import joblib
from pathlib import Path
from typing import Dict, Tuple

from firebase_init import get_firestore_db


_user_models_cache: Dict[Tuple[str, str], object] = {}


def _default_models_dir() -> Path:
    env_override = os.getenv("CYBER_GAME_USER_MODELS")
    if env_override:
        return Path(env_override).expanduser().resolve()

    if sys.platform == "win32":
        root = Path(os.getenv("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support"
    else:                                    
        root = Path.home() / ".local" / "share"

    return (root / "cyber_game" / "user_models").resolve()

USER_MODEL_FOLDER: Path = _default_models_dir()
USER_MODEL_FOLDER.mkdir(parents=True, exist_ok=True)


def _model_path(user_id: str, model_type: str) -> Path:
    return USER_MODEL_FOLDER / f"user_{user_id}_{model_type}.pkl"


def create_fresh_model_for_new_user(local_id: str) -> None:
    placeholder = {"model_name": "placeholder_incremental_model"}
    set_user_model(local_id, placeholder, "baseline")
    print(f"[INFO] Created fresh baseline model for new user {local_id}")


def get_user_model(user_id: str, model_type: str = "baseline"):
    key = (user_id, model_type)
    if key in _user_models_cache:
        return _user_models_cache[key]

    path = _model_path(user_id, model_type)
    if not path.exists():
        print(f"[WARN] No {model_type} model on disk for user {user_id}.")
        return None

    try:
        model = joblib.load(path)
        _user_models_cache[key] = model
        print(f"[INFO] Loaded {model_type} model for {user_id} ← {path}")
        return model
    except Exception as exc:
        print(f"[ERROR] Could not load model {path}: {exc}")
        return None


def set_user_model(user_id: str, model, model_type: str = "baseline") -> None:
    key = (user_id, model_type)
    _user_models_cache[key] = model

    path = _model_path(user_id, model_type)
    try:
        joblib.dump(model, path)
        print(f"[INFO] Saved {model_type} model for {user_id} → {path}")
    except Exception as exc:
        print(f"[ERROR] set_user_model: {exc}")

def get_user_score(user_id: str) -> int:
    db = get_firestore_db()
    snap = db.collection("scoreboards").document(user_id).get()
    return snap.to_dict().get("score", 0) if snap.exists else 0


def update_user_score(user_id: str, points: int) -> None:
    db = get_firestore_db()
    ref = db.collection("scoreboards").document(user_id)
    total = get_user_score(user_id) + points
    ref.set({"score": total}, merge=True)
    print(f"[INFO] +{points} points → {user_id} (total {total})")

def _ensure_attacker_score_fields(d: dict) -> dict:
    d.setdefault("attacker_scores", {}).setdefault("overall", 0)
    d["attacker_scores"].setdefault("datasets", {})
    return d


def add_attacker_score(user_id: str, ds: str, pts: int) -> None:
    db = get_firestore_db()
    ref = db.collection("scoreboards").document(user_id)
    data = _ensure_attacker_score_fields(ref.get().to_dict() or {})

    data["attacker_scores"]["overall"] += pts
    ds_map = data["attacker_scores"]["datasets"]
    ds_map[ds] = ds_map.get(ds, 0) + pts

    ref.set(data, merge=True)
    print(f"[INFO] attacker {user_id} +{pts} on {ds}")


def remove_attacker_score(user_id: str, ds: str, pts: int) -> None:
    db = get_firestore_db()
    ref = db.collection("scoreboards").document(user_id)
    data = _ensure_attacker_score_fields(ref.get().to_dict() or {})

    data["attacker_scores"]["overall"] = max(0, data["attacker_scores"]["overall"] - pts)
    ds_map = data["attacker_scores"]["datasets"]
    ds_map[ds] = max(0, ds_map.get(ds, 0) - pts)

    ref.set(data, merge=True)
    print(f"[INFO] attacker {user_id} −{pts} on {ds}")

def _ensure_defender_score_fields(d: dict) -> dict:
    d.setdefault("defender_scores", {}).setdefault("overall", 0)
    d["defender_scores"].setdefault("categories", {})
    return d


def _split_scenario_key(key: str) -> tuple[str, str]:
    return tuple(key.split("/", 1)) if "/" in key else ("misc", key)


def add_defender_score(user_id: str, scenario_key: str, pts: int) -> None:
    db = get_firestore_db()
    ref = db.collection("scoreboards").document(user_id)
    data = _ensure_defender_score_fields(ref.get().to_dict() or {})

    cat, ds = _split_scenario_key(scenario_key)
    data["defender_scores"]["overall"] += pts

    cat_block = data["defender_scores"]["categories"].get(cat, {"overall": 0, "datasets": {}})
    cat_block["overall"] += pts
    cat_block["datasets"][ds] = cat_block["datasets"].get(ds, 0) + pts
    data["defender_scores"]["categories"][cat] = cat_block

    ref.set(data, merge=True)
    print(f"[INFO] defender {user_id} +{pts} on {cat}/{ds}")


def remove_defender_score(user_id: str, scenario_key: str, pts: int) -> None:
    db = get_firestore_db()
    ref = db.collection("scoreboards").document(user_id)
    data = _ensure_defender_score_fields(ref.get().to_dict() or {})

    cat, ds = _split_scenario_key(scenario_key)
    data["defender_scores"]["overall"] = max(0, data["defender_scores"]["overall"] - pts)

    cat_block = data["defender_scores"]["categories"].get(cat, {"overall": 0, "datasets": {}})
    cat_block["overall"] = max(0, cat_block["overall"] - pts)
    cat_block["datasets"][ds] = max(0, cat_block["datasets"].get(ds, 0) - pts)
    data["defender_scores"]["categories"][cat] = cat_block

    ref.set(data, merge=True)
    print(f"[INFO] defender {user_id} −{pts} on {cat}/{ds}")
