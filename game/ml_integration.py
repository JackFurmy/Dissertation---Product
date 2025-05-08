


from __future__ import annotations

import io
import os
import sys                                    
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

def _project_root() -> Path:
    if getattr(sys, "_MEIPASS", None):             
        return Path(sys._MEIPASS)                
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT         = _project_root()
BASELINE_MODELS_DIR  = PROJECT_ROOT / "game" / "game" / "models" / "models"
BASELINE_MODELS_DIR.mkdir(parents=True, exist_ok=True)


FIREBASE_URLS: Dict[str, str] = {
    "cicids2017": (
        "https://firebasestorage.googleapis.com/v0/b/mldiss.firebasestorage.app/"
    ),
    "bot_iot": (
        "https://firebasestorage.googleapis.com/v0/b/mldiss.firebasestorage.app/"
    ),
    "ctu13": (
        "https://firebasestorage.googleapis.com/v0/b/mldiss.firebasestorage.app/"
    ),
    "iot23": (
        "https://firebasestorage.googleapis.com/v0/b/mldiss.firebasestorage.app/"
    ),
    "unsw_nb15": (
        "https://firebasestorage.googleapis.com/v0/b/mldiss.firebasestorage.app/"
    ),
    "ember": (
        "https://firebasestorage.googleapis.com/v0/b/mldiss.firebasestorage.app/"
    ),
    "malware": (
        "https://firebasestorage.googleapis.com/v0/b/mldiss.firebasestorage.app/"
}


def load_baseline_model(
    dataset_name: str,
) -> Tuple[Any, list[str], float, dict, Any, str | None]:
    pkl_path = BASELINE_MODELS_DIR / f"{dataset_name}_baseline.pkl"
    print("DBG  path we will look in:", pkl_path)
    print("DBG  directory listing    :", list(pkl_path.parent.glob("*_baseline.pkl")))

    if not pkl_path.exists():
        print(f"[ERROR] Missing baseline pkl ⇒ {pkl_path}")
        return (None, None, None, None, None, None)

    try:
        data = joblib.load(pkl_path)

        model_obj    = data.get("model")
        scaler       = data.get("scaler")
        columns_list = data.get("columns", [])
        baseline_acc = data.get("baseline_acc", 0.0)
        metrics      = data.get("metrics", {})
        dataset_url  = data.get("dataset_url")

        if model_obj is None:
            all_models = data.get("models")
            best_name  = data.get("best_name")
            if all_models and best_name and best_name in all_models:
                best_info   = all_models[best_name]
                model_obj   = best_info.get("model")
                scaler      = best_info.get("scaler")
                baseline_acc = best_info.get("baseline_acc", baseline_acc)
                metrics     = best_info.get("metrics", metrics)

        if model_obj is None:
            print("[ERROR] 'model' not found inside pickle.")
            return (None, None, None, None, None, None)

        print(
            f"[INFO] Loaded baseline ⇒ {pkl_path.name} "
            f"(features={len(columns_list)}, acc={baseline_acc:.3f})"
        )
        return (model_obj, columns_list, baseline_acc, metrics, scaler, dataset_url)

    except Exception as exc:
        print(f"[ERROR] Could not load {pkl_path.name}: {exc}")
        return (None, None, None, None, None, None)


def _download_csv(url: str) -> pd.DataFrame:
    print(f"[INFO] Downloading CSV ⇒ {url}")
    resp = requests.get(url)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), low_memory=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df


def fetch_cicids2017_df(): return _download_csv(FIREBASE_URLS["cicids2017"])
def fetch_unsw_nb15_df():  return _download_csv(FIREBASE_URLS["unsw_nb15"])
def fetch_ctu13_df():      return _download_csv(FIREBASE_URLS["ctu13"])
def fetch_iot23_df():      return _download_csv(FIREBASE_URLS["iot23"])
def fetch_bot_iot_df():    return _download_csv(FIREBASE_URLS["bot_iot"])
def fetch_ember_df():      return _download_csv(FIREBASE_URLS["ember"])
def fetch_malware_df():    return _download_csv(FIREBASE_URLS["malware"])


def fetch_dataset_via_url(dataset_url: str | None) -> pd.DataFrame | None:
    if not dataset_url:
        print("[ERROR] dataset_url is None")
        return None
    try:
        return _download_csv(dataset_url)
    except Exception as exc:
        print(f"[ERROR] fetch_dataset_via_url ⇒ {exc}")
        return None


def fetch_dataset_from_firebase(dataset_name: str) -> pd.DataFrame | None:
    name = dataset_name.lower().strip()
    if name == "cicids2017":
        return fetch_cicids2017_df()
    if name == "unsw_nb15":
        return fetch_unsw_nb15_df()
    if name == "ctu13":
        return fetch_ctu13_df()
    if name == "iot23":
        return fetch_iot23_df()
    if name == "bot_iot":
        return fetch_bot_iot_df()
    if name == "ember":
        return fetch_ember_df()
    if name in ("malware", "malware_detection"):
        return fetch_malware_df()
    print(f"[ERROR] Unknown dataset ⇒ {dataset_name}")
    return None


def train_baseline_cicids2017_in_memory(df, out_path=None):
    return generic_in_memory_training(df, out_path)

def train_baseline_unsw_in_memory(df, out_path=None):
    return generic_in_memory_training(df, out_path)

def train_baseline_ctu13_in_memory(df, out_path=None):
    return generic_in_memory_training(df, out_path)

def train_baseline_iot23_in_memory(df, out_path=None):
    return generic_in_memory_training(df, out_path)

def train_baseline_bot_iot_in_memory(df, out_path=None):
    return generic_in_memory_training(df, out_path)

def train_baseline_ember_in_memory(df, out_path=None):
    return generic_in_memory_training(df, out_path)

def train_baseline_malware_in_memory(df, out_path=None):
    return generic_in_memory_training(df, out_path)


def generic_in_memory_training(df: pd.DataFrame, out_path: str | Path | None = None):
    for col in list(df.columns):
        if col.lower().strip() == "label":
            if col != "label":
                df.rename(columns={col: "label"}, inplace=True)

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df.replace({"label": {-1: 1}}, inplace=True)
    df.dropna(subset=["label"], inplace=True)
    df = df[df["label"].isin([0, 1])]

    df_no_lbl = df.drop(columns=["label"], errors="ignore")
    numeric_cols = df_no_lbl.select_dtypes(include=[np.number]).columns
    df_num = df_no_lbl[numeric_cols]

    if len(numeric_cols) > 1:
        corr = df_num.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop_cols = [c for c in upper.columns if any(upper[c] > 0.95)]
        if drop_cols:
            df_num.drop(columns=drop_cols, inplace=True, errors="ignore")

    X = df_num
    y = df["label"]

    if X.empty:
        print("[ERROR] No numeric features after cleaning.")
        return None

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler().fit(X_tr.values)
    X_tr_sc = scaler.transform(X_tr.values)
    X_val_sc = scaler.transform(X_val.values)

    models = {
        "SGD": SGDClassifier(loss="hinge", random_state=42),
        "NB":  GaussianNB(),
        "DT":  DecisionTreeClassifier(max_depth=5, min_samples_leaf=50, random_state=42),
        "RF":  RandomForestClassifier(
            n_estimators=30, max_depth=5, min_samples_leaf=50, random_state=42
        ),
        "XGB": xgb.XGBClassifier(
            random_state=42,
            eval_metric="mlogloss",
            max_depth=5,
            min_child_weight=200,
            gamma=10,
            subsample=0.8,
            colsample_bytree=0.8,
        ),
    }

    best_acc = -1.0
    best_name = ""
    best_model = None

    for name, clf in models.items():
        clf.fit(X_tr_sc, y_tr.values)
        acc = accuracy_score(y_val.values, clf.predict(X_val_sc))
        if acc > best_acc:
            best_acc, best_model, best_name = acc, clf, name

    y_pred = best_model.predict(X_val_sc)
    y_proba = (
        best_model.predict_proba(X_val_sc)[:, 1]
        if hasattr(best_model, "predict_proba")
        else None
    )

    metrics_dict = {
        "accuracy": float(best_acc),
        "macro_f1": float(f1_score(y_val, y_pred, average="macro")),
        "roc_auc": (
            float(roc_auc_score(y_val, y_proba)) if y_proba is not None else None
        ),
        "pr_auc": (
            float(average_precision_score(y_val, y_proba))
            if y_proba is not None
            else None
        ),
    }

    tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[0, 1]).ravel()
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    far = fp / (tn + fp) if (tn + fp) else 0.0
    metrics_dict.update({"tnr": float(tnr), "far": float(far)})

    result = {
        "model": best_model,
        "scaler": scaler,
        "columns": list(X.columns),
        "baseline_acc": best_acc,
        "metrics": metrics_dict,
    }

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(result, out_path)
        print(f"[INFO] Re-trained model saved ⇒ {out_path}")

    return result

def train_dataset_with_removed_cols(
    dataset_name: str,
    df: pd.DataFrame,
    out_path: str | Path,
    removed_cols: list[str] | None = None,
):
    removed_cols = removed_cols or []
    df_new = df.drop(columns=removed_cols, errors="ignore")

    name = dataset_name.lower().strip()
    if name == "cicids2017":
        return train_baseline_cicids2017_in_memory(df_new, out_path)
    if name == "unsw_nb15":
        return train_baseline_unsw_in_memory(df_new, out_path)
    if name == "ctu13":
        return train_baseline_ctu13_in_memory(df_new, out_path)
    if name == "iot23":
        return train_baseline_iot23_in_memory(df_new, out_path)
    if name == "bot_iot":
        return train_baseline_bot_iot_in_memory(df_new, out_path)
    if name == "ember":
        return train_baseline_ember_in_memory(df_new, out_path)
    if name in ("malware", "malware_detection"):
        return train_baseline_malware_in_memory(df_new, out_path)

    print(f"[ERROR] Unhandled dataset ⇒ {dataset_name}")
    return None

