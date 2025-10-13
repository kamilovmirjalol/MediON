"""
pc/train_rf.py
--------------
Train a compact RandomForest model for the MediON Pico runtime using logged CSV
files. Designed for desktop Python (3.9+).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)


FEATURES_ORDER = [
    "hr_mean",
    "hr_std",
    "rr_rmssd",
    "acdc_ratio",
    "ppg_sqi",
    "beat_count",
    "acc_var",
    "motion_frac",
    "hour_sin",
    "hour_cos",
    "hr_z",
    "rr_rmssd_z",
]


def load_logs(logs_dir: Path):
    frames = []
    for path in sorted(logs_dir.glob("*.csv")):
        try:
            df = pd.read_csv(path)
            df["__file"] = path.name
            frames.append(df)
        except Exception as exc:
            print(f"[WARN] Failed to load {path}: {exc}")
    if not frames:
        raise FileNotFoundError(f"No CSV logs found in {logs_dir}")
    df = pd.concat(frames, ignore_index=True)
    return df


def prepare_dataframe(df: pd.DataFrame):
    df = df.copy()
    # Drop QC failures and unlabeled rows
    df = df[(df["dropped_by_qc"] == 0) & (df["label"].isin([0, 1]))]
    if df.empty:
        raise ValueError("No rows remaining after QC/label filtering.")

    # Parse timestamps to maintain chronological order
    if "timestamp_end_iso" in df.columns:
        ts = pd.to_datetime(df["timestamp_end_iso"], errors="coerce")
        fallback = pd.to_datetime(df.index, unit="s")  # monotonic fallback
        ts = ts.fillna(fallback)
        df = df.assign(_ts=ts).sort_values("_ts").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
        df["_ts"] = pd.to_datetime(df.index, unit="s")

    for feat in FEATURES_ORDER:
        if feat not in df.columns:
            df[feat] = 0.0
    df[FEATURES_ORDER] = df[FEATURES_ORDER].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df["label"] = df["label"].astype(int)
    return df


def time_split(df: pd.DataFrame, test_fraction=0.2):
    n = len(df)
    if n < 10:
        raise ValueError("Not enough samples for train/test split.")
    split_idx = max(1, int(n * (1 - test_fraction)))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test


def train_model(train_df: pd.DataFrame):
    clf = RandomForestClassifier(
        n_estimators=15,
        max_depth=4,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
    )
    X = train_df[FEATURES_ORDER].values
    y = train_df["label"].values
    clf.fit(X, y)
    return clf


def evaluate_model(clf, test_df: pd.DataFrame):
    X_test = test_df[FEATURES_ORDER].values
    y_test = test_df["label"].values
    proba = clf.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    report = classification_report(y_test, preds, digits=3)
    cm = confusion_matrix(y_test, preds)
    avg_precision = average_precision_score(y_test, proba)
    precision, recall, thresholds = precision_recall_curve(y_test, proba)

    metrics = {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "average_precision": float(avg_precision),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
    }
    return metrics, proba


def write_metrics(metrics_path: Path, metrics: dict):
    metrics_path.write_text(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Train RandomForest model for MediON Pico runtime.")
    parser.add_argument("--logs", type=Path, default=Path("../logs"), help="Directory containing CSV logs.")
    parser.add_argument("--out", type=Path, default=Path("rf.joblib"), help="Output joblib filename.")
    parser.add_argument("--metrics", type=Path, default=Path("metrics.json"), help="Metrics JSON filename.")
    args = parser.parse_args()

    logs_dir = args.logs
    if not logs_dir.exists():
        raise FileNotFoundError(f"Logs directory not found: {logs_dir}")

    df = load_logs(logs_dir)
    df = prepare_dataframe(df)
    train_df, test_df = time_split(df)

    clf = train_model(train_df)
    metrics, proba = evaluate_model(clf, test_df)

    print("[INFO] Evaluation report\n", metrics["classification_report"])
    print("[INFO] Confusion matrix:\n", np.array(metrics["confusion_matrix"]))
    print("[INFO] Average precision (PR-AUC): {:.3f}".format(metrics["average_precision"]))

    dump(clf, args.out)
    write_metrics(args.metrics, metrics)
    print(f"[INFO] Saved model to {args.out}")
    print(f"[INFO] Saved metrics to {args.metrics}")


if __name__ == "__main__":
    main()

