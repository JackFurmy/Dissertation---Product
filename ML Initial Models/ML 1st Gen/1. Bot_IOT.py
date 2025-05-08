import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
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


TRAIN_DATA_PATH = (
    'cybersecurity_game/data/Dynamic Threat Simulation/Bot-IoT/Dataset/Training/final_validated/bot_iot_val_reduced.csv'
)

HISTORICAL_FILE = (
    "cybersecurity_game/models/1. Baseline Offline Model/Network IoT: Dynamic Threat/Network IoT_historical.xlsx"
)

LABEL_COL = "attack" 

LEAKY_COLUMNS = ["category", "subcategory"]

DROP_MI_FEATURES = ["dport", "daddr_oct4", "dur","saddr_oct4", "stime", "daddr_oct3"]

PARTIAL_FIT_MODELS = ["SGD","NB"]
BATCH_SIZE = 100_000
N_EPOCHS   = 1

def parse_ip_octets(df: pd.DataFrame, colname: str):
    if colname not in df.columns:
        return df

    def ip_to_octets(ip_str):
        try:
            parts = ip_str.split('.')
            if len(parts) == 4:
                return [int(x) for x in parts]
            else:
                return [0,0,0,0]
        except:
            return [0,0,0,0]

    octet_cols = [f"{colname}_oct{i}" for i in range(1,5)]
    ip_parsed = df[colname].astype(str).apply(ip_to_octets)

    octet_df = pd.DataFrame(ip_parsed.tolist(), columns=octet_cols, index=df.index)
    df = pd.concat([df, octet_df], axis=1)
    df.drop(columns=[colname], inplace=True)
    return df

def label_encode_column(df: pd.DataFrame, colname: str):
    if colname not in df.columns:
        return df
    if pd.api.types.is_numeric_dtype(df[colname]):
        return df
    lbl = LabelEncoder()
    df[colname] = lbl.fit_transform(df[colname].astype(str))
    return df


def load_data(file_path, label_col=LABEL_COL, drop_leak=LEAKY_COLUMNS):
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return None, None

    df = pd.read_csv(file_path, low_memory=True)
    print(f"[INFO] Loaded data: {df.shape[0]} rows, {df.shape[1]} columns from '{file_path}'")

    for c in drop_leak:
        if c in df.columns:
            df.drop(columns=[c], inplace=True, errors='ignore')

    if label_col not in df.columns:
        print(f"[ERROR] Label column '{label_col}' not found in CSV.")
        return None, None

    for ip_c in ["saddr","daddr"]:
        if ip_c in df.columns:
            df = parse_ip_octets(df, ip_c)


    for col in ["proto","flgs","sport","dport"]:
        df = label_encode_column(df, col)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    print(f"[INFO] Final shape after cleaning: {df.shape[0]} rows, {df.shape[1]} cols")

    X = df.drop(columns=[label_col], errors='ignore')
    y = df[label_col]

    print(f"[INFO] X shape: {X.shape}, y shape: {y.shape}")
    return X, y

def remove_correlated_features(X, threshold=0.95):
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        print("[INFO] Not enough numeric columns for correlation removal.")
        return X

    corr_matrix = X[numeric_cols].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > threshold)]
    if to_drop:
        print(f"[INFO] Dropping correlated numeric features: {to_drop}")
        X.drop(columns=to_drop, inplace=True, errors='ignore')
    return X

def plot_mutual_info(X, y):
    X_num = X.select_dtypes(include=[np.number])
    print(f"[INFO] Computing mutual info on {X_num.shape[1]} numeric columns...")
    if X_num.empty:
        print("[WARN] No numeric columns remain. Skipping mutual info plot.")
        return None

    mi_scores = mutual_info_classif(X_num, y, discrete_features='auto', random_state=42)
    mi_df = pd.DataFrame({"feature": X_num.columns, "mi_score": mi_scores})
    mi_df.sort_values(by='mi_score', ascending=False, inplace=True)

    plt.figure(figsize=(10,5))
    plt.bar(mi_df['feature'], mi_df['mi_score'], color='skyblue')
    plt.xticks(rotation=90)
    plt.ylabel("Mutual Info vs. Label")
    plt.title("Bot-IoT: Numeric Feature 'Leakage' Score")
    plt.tight_layout()
    plt.show()

    return mi_df

def train_and_evaluate_models(
    X_train, y_train,
    X_val,   y_val,
    partial_fit_models=PARTIAL_FIT_MODELS,
    batch_size=BATCH_SIZE,
    n_epochs=N_EPOCHS
):
    models = {
        "SGD": SGDClassifier(
            loss='log_loss',    
            alpha=0.01,
            penalty='l2',
            random_state=42
        ),
        "NB" : GaussianNB(),
        "DT" : DecisionTreeClassifier(
            max_depth=3,
            min_samples_leaf=200,
            random_state=42
        ),
        "RF" : RandomForestClassifier(
            n_estimators=20,
            max_depth=3,
            min_samples_leaf=200,
            max_features='sqrt',
            random_state=42
        ),
        "XGB": xgb.XGBClassifier(
            random_state=42,
            eval_metric='mlogloss',
            max_depth=3,
            min_child_weight=500,
            gamma=15,
            subsample=0.5,
            colsample_bytree=0.5,
            n_estimators=50,
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

    classes = np.array([0,1])
    n_samples = len(X_train_sc)

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
                f1m = f1_score(y_val_np, y_val_pred, average='macro', labels=[0,1])
                print(f"[Epoch {epoch+1}] {name} => Accuracy: {acc:.4f}, Macro-F1: {f1m:.4f}")

            y_pred = clf.predict(X_val_sc)
            y_proba= clf.predict_proba(X_val_sc) if hasattr(clf, "predict_proba") else None
            compute_metrics_binary(y_val_np, y_pred, y_proba)
        else:
            clf.fit(X_train_sc, y_train_np)
            y_pred = clf.predict(X_val_sc)
            y_proba= clf.predict_proba(X_val_sc) if hasattr(clf, "predict_proba") else None
            compute_metrics_binary(y_val_np, y_pred, y_proba)


def compute_metrics_binary(y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', labels=[0,1])

    print("\nClassification Report (Labels=[0,1]):")
    print(classification_report(y_true, y_pred, zero_division=0, labels=[0,1]))
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")

    if (y_proba is not None) and (y_proba.shape[1] == 2):
        y_prob1 = y_proba[:,1]
        rocAUC  = roc_auc_score(y_true, y_prob1)
        prAUC   = average_precision_score(y_true, y_prob1)
        print(f"ROC AUC: {rocAUC:.4f}")
        print(f"PR AUC : {prAUC:.4f}")

    y_true_bin = np.where(y_true==1, 1, 0)
    y_pred_bin = np.where(y_pred==1, 1, 0)
    cm_bin = confusion_matrix(y_true_bin, y_pred_bin, labels=[0,1])
    tn, fp, fn, tp = cm_bin.ravel()

    tnr = tn/(tn+fp) if (tn+fp)>0 else 0
    far = fp/(tn+fp) if (tn+fp)>0 else 0

    print(f"TNR (Label=0 correct): {tnr:.4f}")
    print(f"FAR (False Alarm Rate): {far:.4f}")

    cm_full = confusion_matrix(y_true, y_pred, labels=[0,1])
    print("Full Confusion Matrix (0,1):\n", cm_full)
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
        print(f"[WARN] '{LABEL_COL}' not found. Skipping.")
        return

    df_hist[LABEL_COL] = pd.to_numeric(df_hist[LABEL_COL], errors='coerce')
    df_hist.dropna(subset=[LABEL_COL], inplace=True)
    if df_hist.empty:
        print("[WARN] No valid labeled rows in historical. Skipping.")
        return

    X_h = df_hist.drop(columns=[LABEL_COL], errors='ignore')
    y_h = df_hist[LABEL_COL]

    if len(X_h)==0:
        print("[WARN] 0 samples in historical. Cannot retrain.")
        return

    clf = RandomForestClassifier(
        n_estimators=30,
        max_depth=5,
        min_samples_leaf=50,
        random_state=42
    )
    clf.fit(X_h, y_h)
    print("[INFO] Batch retraining complete with updated historical data.")

def main():
    X, y = load_data(
        file_path=TRAIN_DATA_PATH,
        label_col=LABEL_COL,
        drop_leak=LEAKY_COLUMNS
    )
    if X is None or y is None:
        return

    X = remove_correlated_features(X, threshold=0.95)

    mi_df = plot_mutual_info(X, y)
    if mi_df is not None:
        print("\nTop 15 columns by MI:\n", mi_df.head(15))

    for col in DROP_MI_FEATURES:
        if col in X.columns:
            print(f"[INFO] Dropping high-MI col '{col}' from dataset.")
            X.drop(columns=[col], inplace=True, errors='ignore')

    if len(y.unique()) < 2:
        print("[ERROR] Only one class remains after cleaning. Exiting.")
        return

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    print(f"[INFO] Stratified => Train={X_train.shape[0]}, Val={X_val.shape[0]}")

    train_and_evaluate_models(X_train, y_train, X_val, y_val)

if __name__ == "__main__":
    main()
    batch_retraining_demo()

