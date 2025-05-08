import os
import io
import requests
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score
)
from sklearn.feature_selection import mutual_info_classif

CTU13_FIREBASE_URL = (
    ""
)
LABEL_COL = "label"
LEAKY_COLUMNS = []
PARTIAL_FIT_MODELS = ["SGD", "NB"] 
BATCH_SIZE = 100_000
N_EPOCHS   = 1

SAVE_BASELINE = True
BASELINE_MODEL_PATH = (
    "cybersecurity_game/game/game/models/models/ctu13_baseline.pkl"
)

MI_FIG_DIR  = "cybersecurity_game/game/game/mi figure"
MI_FIG_NAME = "ctu13_mi.png"

def load_data_via_requests(firebase_url, label_col=LABEL_COL, drop_leak=LEAKY_COLUMNS):
    print(f"[INFO] Downloading CTU-13 CSV from {firebase_url} ...")
    resp = requests.get(firebase_url, verify=True)
    resp.raise_for_status()

    csv_text = resp.text
    filelike = io.StringIO(csv_text)

    df = pd.read_csv(filelike, low_memory=True)
    print(f"[INFO] Loaded data: {df.shape[0]} rows, {df.shape[1]} cols from request.")

    for c in drop_leak:
        if c in df.columns:
            df.drop(columns=[c], inplace=True, errors='ignore')

    if label_col not in df.columns:
        print(f"[ERROR] Label column '{label_col}' not found.")
        return None, None

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)
    print(f"[INFO] Removed {before - after} duplicates => now {after} rows.")

    df[label_col] = pd.to_numeric(df[label_col], errors='coerce')
    df.dropna(subset=[label_col], inplace=True)

    X = df.drop(columns=[label_col], errors='ignore')
    y = df[label_col]

    print(f"[INFO] Final shape => X={X.shape}, y={y.shape}")
    return X, y

def remove_correlated_features(X, threshold=0.95):
    if X.shape[1] < 2:
        return X

    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    high_corr_feats = [col for col in upper_tri.columns if any(upper_tri[col] > threshold)]
    if high_corr_feats:
        print(f"[INFO] Dropping highly correlated: {high_corr_feats}")
        X = X.drop(columns=high_corr_feats)
    return X

def plot_mutual_info(X, y):
    X_num = X.select_dtypes(include=[np.number])
    if X_num.shape[1] < 1:
        print("[WARN] No numeric columns => skipping MI.")
        return None

    print(f"[INFO] Computing mutual info on {X_num.shape[1]} numeric cols...")
    mi_scores = mutual_info_classif(X_num, y, discrete_features='auto', random_state=42)
    mi_df = pd.DataFrame({'feature': X_num.columns, 'mi_score': mi_scores})
    mi_df.sort_values(by='mi_score', ascending=False, inplace=True)

    plt.figure(figsize=(10,5))
    plt.bar(mi_df['feature'], mi_df['mi_score'], color='skyblue')
    plt.xticks(rotation=90)
    plt.ylabel("Mutual Info vs. Label")
    plt.title("CTU-13: Feature 'Leakage' Score")
    plt.tight_layout()

    os.makedirs(MI_FIG_DIR, exist_ok=True)
    out_path = os.path.join(MI_FIG_DIR, MI_FIG_NAME)
    plt.savefig(out_path)
    print(f"[INFO] MI figure saved => {out_path}")
    plt.close()

    return mi_df

def train_and_evaluate_all_models(
    X_train, y_train,
    X_val,   y_val,
    partial_fit_models=PARTIAL_FIT_MODELS,
    batch_size=BATCH_SIZE,
    n_epochs=N_EPOCHS
):
    models = {
        "SGD": SGDClassifier(loss='log_loss', random_state=42),
        "NB":  GaussianNB(),
        "DT":  DecisionTreeClassifier(max_depth=5, min_samples_leaf=50, random_state=42),
        "RF":  RandomForestClassifier(n_estimators=30, max_depth=5, min_samples_leaf=50, random_state=42),
        "XGB": xgb.XGBClassifier(
            random_state=42,
            eval_metric='mlogloss',
            max_depth=5,
            min_child_weight=200,
            gamma=10,
            subsample=0.8,
            colsample_bytree=0.8
        )
    }

    scaler = StandardScaler()
    X_train_np = X_train.values
    y_train_np = y_train.values
    X_val_np   = X_val.values
    y_val_np   = y_val.values

    scaler.fit(X_train_np)
    X_train_sc = scaler.transform(X_train_np)
    X_val_sc   = scaler.transform(X_val_np)

    n_samples = len(X_train_sc)
    classes   = np.array([0,1])

    results = {}

    for name, clf in models.items():
        print(f"\n=== Training {name} ===")
        if name in partial_fit_models:
            for epoch in range(n_epochs):
                print(f"[INFO] {name}: epoch {epoch+1}/{n_epochs}")
                idx = np.arange(n_samples)
                np.random.shuffle(idx)

                start=0
                while start < n_samples:
                    end = min(start+batch_size, n_samples)
                    batch_idx = idx[start:end]
                    X_batch   = X_train_sc[batch_idx]
                    y_batch   = y_train_np[batch_idx]

                    if hasattr(clf, 'partial_fit'):
                        clf.partial_fit(X_batch, y_batch, classes=classes)
                    else:
                        clf.fit(X_batch, y_batch)
                    start = end

                y_val_pred = clf.predict(X_val_sc)
                acc = accuracy_score(y_val_np, y_val_pred)
                mf1= f1_score(y_val_np, y_val_pred, average='macro', labels=[0,1])
                print(f"[Epoch {epoch+1}] => ACC={acc:.4f}, Macro-F1={mf1:.4f}")
        else:
            clf.fit(X_train_sc, y_train_np)

        scoreboard = gather_model_metrics(clf, X_val_sc, y_val_np)
        results[name] = {
            "model": clf,
            "scaler": scaler,
            "metrics": scoreboard
        }

    return results

def gather_model_metrics(clf, X_val_sc, y_val_np):
    y_pred = clf.predict(X_val_sc)
    acc = accuracy_score(y_val_np, y_pred)
    macro_f1 = f1_score(y_val_np, y_pred, average='macro', labels=[0,1])

    rocAUC = None
    prAUC  = None
    if hasattr(clf, 'predict_proba'):
        y_proba = clf.predict_proba(X_val_sc)
        if y_proba.shape[1] == 2:
            y_prob1 = y_proba[:,1]
            rocAUC  = roc_auc_score(y_val_np, y_prob1)
            prAUC   = average_precision_score(y_val_np, y_prob1)

    y_true_bin = np.where(y_val_np==0, 0, 1)
    y_pred_bin = np.where(y_pred==0, 0, 1)
    cm_bin = confusion_matrix(y_true_bin, y_pred_bin, labels=[0,1])
    tn, fp, fn, tp = cm_bin.ravel()
    tnr = tn/(tn+fp) if (tn+fp)>0 else 0
    far = fp/(tn+fp) if (tn+fp)>0 else 0

    cm_full = confusion_matrix(y_val_np, y_pred, labels=[0,1])

    return {
        "accuracy":   float(acc),
        "macro_f1":   float(macro_f1),
        "roc_auc":    float(rocAUC) if rocAUC else None,
        "pr_auc":     float(prAUC)  if prAUC  else None,
        "tnr":        float(tnr),
        "far":        float(far),
        "conf_matrix": cm_full.tolist()
    }

def main():
    X, y = load_data_via_requests(CTU13_FIREBASE_URL)
    if X is None or y is None:
        print("[ERROR] Unable to load CTU-13 dataset from Firebase.")
        return

    X = remove_correlated_features(X, threshold=0.95)

    mi_df = plot_mutual_info(X, y)
    if mi_df is not None:
        print("\nTop columns by MI:\n", mi_df.head(15))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    print(f"[INFO] Stratified => Train={X_train.shape}, Val={X_val.shape}")

    results_dict = train_and_evaluate_all_models(
        X_train, y_train,
        X_val,   y_val,
        partial_fit_models=PARTIAL_FIT_MODELS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS
    )
    if not results_dict:
        print("[ERROR] No models were trained.")
        return

    best_name = None
    best_score = -1.0

    for m_name, info in results_dict.items():
        scoreboard = info["metrics"]
        macro_f1   = scoreboard.get("macro_f1", 0.0)
        if macro_f1 > best_score:
            best_score = macro_f1
            best_name  = m_name

    print(f"\n[INFO] Best model => {best_name}, macro_f1={best_score:.4f}")
    best_model_info = results_dict[best_name]
    best_scoreboard = best_model_info["metrics"]
    baseline_acc    = best_scoreboard["accuracy"]

    data_dict = {
        "models": results_dict,   
        "best_name": best_name,   
        "best_acc": baseline_acc, 
        "columns": list(X_train.columns),
        "baseline_acc": baseline_acc, 
        "metrics": best_scoreboard,
        "dataset_url": CTU13_FIREBASE_URL
    }

    if SAVE_BASELINE:
        os.makedirs(os.path.dirname(BASELINE_MODEL_PATH), exist_ok=True)
        joblib.dump(data_dict, BASELINE_MODEL_PATH)
        print(f"[INFO] Dictionary-based baseline saved => {BASELINE_MODEL_PATH}")

if __name__ == "__main__":
    main()
