import argparse, io, requests, joblib
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from river import tree, linear_model, naive_bayes, metrics
from river.metrics import ConfusionMatrix

WINDOWS7_FIREBASE_URL = (
    ""
)

MODEL_DIR = Path(
    "cybersecurity_game/game/game/models/models"
)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL   = "label"
MODEL_NAMES = ["HoeffdingTree", "LogisticRegression", "NaiveBayes"]

def unify_feats(x: Dict, feat_set: set) -> Tuple[Dict, set]:
    x_full = {f: x.get(f, 0) for f in feat_set}
    for f, v in x.items():
        x_full[f] = v
        if f not in feat_set:
            feat_set.add(f)
    return x_full, feat_set


def label_encode_value(col: str, val, encoders: Dict[str, Dict[str, int]]):
    if val is None:
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        pass
    enc = encoders.setdefault(col, {})
    s   = str(val)
    if s not in enc:
        enc[s] = len(enc) + 1
    return float(enc[s])


def stream_rows(csv_url: str, chunk: int, encoders: Dict[str, Dict[str, int]],
                feat_set: set, label: str = LABEL_COL):
    print(f"[INFO] downloading CSV … {csv_url}")
    df = (
        pd.read_csv(io.StringIO(requests.get(csv_url, verify=True).text), low_memory=True)
        .dropna(subset=[label])
        .drop_duplicates()
    )

    for start in range(0, len(df), chunk):
        for _, row in df.iloc[start:start+chunk].iterrows():
            y   = int(row[label])
            raw = row.drop(labels=[label, "ts"], errors="ignore").to_dict()  # drop 'ts'
            x   = {c: label_encode_value(c, v, encoders) for c, v in raw.items()}
            x, feat_set = unify_feats(x, feat_set)
            yield x, y, feat_set


def bin_scores(cm: ConfusionMatrix):
    tp = cm[1][1]
    fn = cm[1][0] if 0 in cm[1] else 0
    fp = cm[0][1] if 1 in cm[0] else 0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def cm_as_dict(cm: ConfusionMatrix) -> Dict[int, Dict[int, int]]:
    return {a: {p: cm[a][p] for p in (0, 1)} for a in (0, 1)}


def bundle_path() -> Path:
    return MODEL_DIR / "windows7.pkl"


def model_split_path(user_id: str, name: str) -> Path:
    return MODEL_DIR / f"user_{user_id}_{name}.pkl"


def train_windows7(
    chunk_size    : int  = 50_000,
    load_existing : bool = False,
    split_models  : bool = False,
    user_id       : str  = "win7_user_default",
):
    prev_models = {}
    if load_existing and bundle_path().exists():
        data = joblib.load(bundle_path())
        prev_models = {n: data["models"][n]["model"] for n in MODEL_NAMES}
        print(f"[INFO] continuing from bundle ⇒ {bundle_path()}")

    models = {
        "HoeffdingTree":      prev_models.get("HoeffdingTree", tree.HoeffdingTreeClassifier()),
        "LogisticRegression": prev_models.get("LogisticRegression", linear_model.LogisticRegression()),
        "NaiveBayes":         prev_models.get("NaiveBayes",      naive_bayes.GaussianNB()),
    }

    acc = {n: metrics.Accuracy() for n in MODEL_NAMES}
    cm  = {n: ConfusionMatrix()  for n in MODEL_NAMES}

    feats, rows = set(), 0
    label_encoders: Dict[str, Dict[str, int]] = {}

    for x, y, feats in stream_rows(WINDOWS7_FIREBASE_URL,
                                   chunk_size, label_encoders, feats):
        rows += 1
        for n, m in models.items():
            p = m.predict_one(x)
            if p is not None:
                acc[n].update(y, p)
                cm[n].update(y, p)
            m.learn_one(x, y)

        if rows % 50_000 == 0:
            print(f"[INFO] processed {rows:,} rows …")

    print("\n=== Training finished ===")
    print(f"Rows processed: {rows:,}\n")

    bundle_models, best_name, best_f1 = {}, None, -1.0
    for n in MODEL_NAMES:
        a = float(acc[n].get())
        p, r, f1 = bin_scores(cm[n])
        print(f"{n:>18}:  acc={a:.4f}  prec={p:.4f}  rec={r:.4f}  F1={f1:.4f}")

        bundle_models[n] = {
            "model":   models[n],
            "metrics": {
                "accuracy":   a,
                "precision_1": p,
                "recall_1":    r,
                "f1_1":        f1,
                "conf_matrix": cm_as_dict(cm[n]),
            },
        }
        if f1 > best_f1:
            best_f1, best_name = f1, n

    joblib.dump(
        {
            "models":       bundle_models,
            "best_name":    best_name,
            "best_f1":      best_f1,
            "dataset_url":  WINDOWS7_FIREBASE_URL,
            "rows_trained": rows,
        },
        bundle_path(),
    )
    print(f"[INFO] bundle saved ⇒ {bundle_path()}")

    if split_models:
        for n in MODEL_NAMES:
            p = model_split_path(user_id, n)
            joblib.dump(models[n], p)
            print(f"[INFO] split pickle ⇒ {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-size",    type=int, default=50_000)
    ap.add_argument("--load-existing", action="store_true",
                    help="Continue training from previous bundle")
    ap.add_argument("--split-models",  action="store_true",
                    help="Also save three individual model pickles")
    ap.add_argument("--user-id",       default="win7_user_default")
    args = ap.parse_args()

    train_windows7(
        chunk_size   = args.chunk_size,
        load_existing= args.load_existing,
        split_models = args.split_models,
        user_id      = args.user_id,
    )
