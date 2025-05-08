import os
import numpy as np
import pandas as pd

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
import matplotlib.pyplot as plt


TRAIN_DATA_PATH = "cybersecurity_game/data/Concept Drift and Adaptive Learning/OpenML/HYPERPLANE_01_converted_integer_label.csv"

HISTORICAL_FILE = "cybersecurity_game/models/1. Baseline Offline Model/Concept Drift: Adatpvie Learning/concept Drift_historical.xlsx"


LABEL_COL = "class"

BINARY_LABELS = [0, 1]

LEAKY_COLUMNS = [
    "global_id", "event_id", "wire_id",      
    "StartTime", "date", "day", "period",     
    "V1","V2","V3",  
]

PARTIAL_FIT_MODELS = ["SGD","NB"]
BATCH_SIZE = 100_000
N_EPOCHS   = 1


def load_data(file_path, label_col=LABEL_COL, drop_leak=LEAKY_COLUMNS):
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return None, None

    df = pd.read_csv(file_path, low_memory=True)
    print(f"[INFO] Loaded data: {df.shape[0]} rows, {df.shape[1]} columns from {file_path}")

    for c in drop_leak:
        if c in df.columns:
            df.drop(columns=[c], inplace=True, errors='ignore')

    if label_col not in df.columns:
        print(f"[ERROR] Label col '{label_col}' not found after dropping. Exiting.")
        return None, None

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)
    print(f"[INFO] Removed {before - after} exact duplicates. Now {after} rowsdf[label_col] = pd.to_numeric(df[label_col], errors='coerce')
    df.dropna(subset=[label_col], inplace=True)

    X = df.drop(columns=[label_col])
    y = df[label_col]

    print(f"[INFO] Final shape after cleaning: X={X.shape}, y={y.shape}")
    return X, y

def remove_correlated_features(X, threshold=0.95):
    if X.shape[1] < 2:
        return X
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_feats = [col for col in upper_tri.columns if any(upper_tri[col] > threshold)]
    if high_corr_feats:
        print(f"[INFO] Dropping highly correlated features: {high_corr_feats}")
        X = X.drop(columns=high_corr_feats)
    return X

def plot_mutual_info(X, y):
    X_num = X.select_dtypes(include=[np.number])
    print(f"[INFO] Computing MI on {X_num.shape[1]} numeric columns...")
    if X_num.empty:
        print("[WARN] No numeric columns found.")
        return None

    mi_scores = mutual_info_classif(X_num, y, discrete_features='auto', random_state=42)
    mi_df = pd.DataFrame({"feature": X_num.columns, "mi_score": mi_scores})
    mi_df.sort_values(by='mi_score', ascending=False, inplace=True)

    plt.figure(figsize=(10,5))
    plt.bar(mi_df['feature'], mi_df['mi_score'], color='skyblue')
    plt.xticks(rotation=90)
    plt.ylabel("Mutual Info wrt label")
    plt.title("Feature 'Leakage' Score via MI - ConceptDrift DS")
    plt.tight_layout()
    plt.show()

    return mi_df

def train_and_evaluate_models(X_train, y_train, X_val, y_val,partial_fit_models=PARTIAL_FIT_MODELS,batch_size=BATCH_SIZE,n_epochs=N_EPOCHS):

    models = {
        "SGD": SGDClassifier(loss='hinge', random_state=42),
        "NB" : GaussianNB(),
        "DT" : DecisionTreeClassifier(max_depth=5, min_samples_leaf=50, random_state=42),
        "RF" : RandomForestClassifier(n_estimators=30, max_depth=5, min_samples_leaf=50, random_state=42),
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
    classes = np.unique(y_train_np)  

    for name, clf in models.items():
        print(f"\n=== Training {name} ===")

        if name in partial_fit_models:
            for epoch in range(n_epochs):
                print(f"[INFO] {name}: Starting epoch {epoch+1}/{n_epochs}")
                idx = np.arange(n_samples)
                np.random.shuffle(idx)

                start=0
                while start < n_samples:
                    end = min(start+batch_size, n_samples)
                    batch_idx = idx[start:end]

                    X_batch = X_train_sc[batch_idx]
                    y_batch = y_train_np[batch_idx]
                    if hasattr(clf, 'partial_fit'):
                        clf.partial_fit(X_batch, y_batch, classes=classes)
                    else:
                        clf.fit(X_batch, y_batch)
                    start = end

                y_val_pred = clf.predict(X_val_sc)
                acc = accuracy_score(y_val_np, y_val_pred)
                f1m = f1_score(y_val_np, y_val_pred, average='macro')
                print(f"[Epoch {epoch+1}] {name} => Accuracy: {acc:.4f}, Macro-F1: {f1m:.4f}")

            y_pred = clf.predict(X_val_sc)
            y_proba= clf.predict_proba(X_val_sc) if hasattr(clf, "predict_proba") else None
            compute_metrics(y_val_np, y_pred, y_proba)
        else:
            if name=="XGB":
                clf.fit(X_train_sc, y_train_np)
                y_pred = clf.predict(X_val_sc)
                y_proba= clf.predict_proba(X_val_sc) if hasattr(clf, "predict_proba") else None
            else:
                clf.fit(X_train_sc, y_train_np)
                y_pred = clf.predict(X_val_sc)
                y_proba= clf.predict_proba(X_val_sc) if hasattr(clf, "predict_proba") else None

            compute_metrics(y_val_np, y_pred, y_proba)

def compute_metrics(y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")

    if y_proba is not None and len(np.unique(y_true)) == 2 and y_proba.shape[1] == 2:
        y_prob1 = y_proba[:,1]
        rocAUC  = roc_auc_score(y_true, y_prob1)
        prAUC   = average_precision_score(y_true, y_prob1)
        print(f"ROC AUC: {rocAUC:.4f}")
        print(f"PR AUC : {prAUC:.4f}")
    print("--------------------------------------------------")

def batch_retraining_demo():
    if not os.path.exists(HISTORICAL_FILE):
        print("[WARN] No historical dataset found.")
        return

    df_hist = pd.read_excel(HISTORICAL_FILE)
    df_hist.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_hist.dropna(how='all', inplace=True)

    if df_hist.empty:
        print("[WARN] Historical file is empty. Skipping retraining.")
        return

    if LABEL_COL not in df_hist.columns:
        print(f"[WARN] '{LABEL_COL}' not found in historical. Skipping.")
        return

    df_hist[LABEL_COL] = pd.to_numeric(df_hist[LABEL_COL], errors='coerce')
    df_hist.dropna(subset=[LABEL_COL], inplace=True)
    if df_hist.empty:
        print("[WARN] No valid labeled rows in historical. Skipping.")
        return

    X_h = df_hist.drop(columns=[LABEL_COL])
    y_h = df_hist[LABEL_COL]

    if len(X_h) == 0:
        print("[WARN] 0 samples in historical. Cannot retrain.")
        return

    clf = RandomForestClassifier(
        n_estimators=30,
        max_depth=5,
        min_samples_leaf=50,
        random_state=42
    )
    clf.fit(X_h, y_h)
    print("[INFO] Batch retraining complete. Model updated with historical data.")


def main():
    X, y = load_data(TRAIN_DATA_PATH)
    if X is None or y is None:
        return

    X = remove_correlated_features(X, threshold=0.95)

    mi_df = plot_mutual_info(X, y)
    if mi_df is not None:
        print("\nTop 15 columns by MI:\n", mi_df.head(15))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,  
        random_state=42
    )
    print(f"[INFO] Stratified split => Train size: {X_train.shape[0]}, Val size: {X_val.shape[0]}")

    train_and_evaluate_models(X_train, y_train, X_val, y_val)

if __name__ == "__main__":
    main()
    batch_retraining_demo()
