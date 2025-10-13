"""
pc/export_rf.py
---------------
Convert a scikit-learn RandomForestClassifier (trained via train_rf.py) into the
compact JSON format used by the MediON Pico runtime.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import f1_score


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
            frames.append(df)
        except Exception as exc:
            print(f"[WARN] Failed to load {path}: {exc}")
    if not frames:
        raise FileNotFoundError(f"No CSV logs found in {logs_dir}")
    df = pd.concat(frames, ignore_index=True)
    return df


def prepare_dataframe(df: pd.DataFrame):
    df = df.copy()
    df = df[(df["dropped_by_qc"] == 0) & (df["label"].isin([0, 1]))]
    if df.empty:
        raise ValueError("No QC-passing labelled rows available.")

    if "timestamp_end_iso" in df.columns:
        ts = pd.to_datetime(df["timestamp_end_iso"], errors="coerce")
        fallback = pd.to_datetime(df.index, unit="s")
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
        raise ValueError("Not enough samples for validation split.")
    split_idx = max(1, int(n * (1 - test_fraction)))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test


def search_best_threshold(y_true, proba):
    best_thresh = 0.5
    best_f1 = -1.0
    for thresh in np.arange(0.2, 0.81, 0.01):
        preds = (proba >= thresh).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(np.round(thresh, 4))
    return best_thresh, best_f1


def export_tree(estimator):
    tree = estimator.tree_
    nodes = []
    for node_id in range(tree.node_count):
        left = tree.children_left[node_id]
        right = tree.children_right[node_id]
        if left == right:
            values = tree.value[node_id][0]
            total = float(np.sum(values))
            prob = 0.0 if total <= 0 else float(values[1] / total)
            nodes.append({"leaf": float(np.round(prob, 6))})
        else:
            nodes.append(
                {
                    "fi": int(tree.feature[node_id]),
                    "th": float(np.round(tree.threshold[node_id], 6)),
                    "lt": int(left),
                    "rt": int(right),
                    "leaf": None,
                }
            )
    return {"nodes": nodes, "root": 0}


def runtime_predict(model_json, feature_row):
    order = model_json["features_order"]
    vector = [float(feature_row[name]) for name in order]
    total = 0.0
    count = 0
    for tree in model_json["trees"]:
        nodes = tree["nodes"]
        idx = tree.get("root", 0)
        while True:
            node = nodes[idx]
            leaf = node.get("leaf")
            if leaf is not None:
                total += float(leaf)
                count += 1
                break
            fi = node["fi"]
            th = node["th"]
            feat = vector[fi] if fi < len(vector) else 0.0
            idx = node["lt"] if feat <= th else node["rt"]
    return total / count if count else 0.5


def main():
    parser = argparse.ArgumentParser(description="Export trained RF model to Pico JSON format.")
    parser.add_argument("--model", type=Path, default=Path("rf.joblib"), help="Path to trained RandomForest joblib.")
    parser.add_argument("--logs", type=Path, default=Path("../logs"), help="Logs directory (for validation split).")
    parser.add_argument("--out", type=Path, default=Path("../model/model_user.json"), help="Output JSON path.")
    args = parser.parse_args()

    clf = load(args.model)
    if not hasattr(clf, "estimators_"):
        raise TypeError("Loaded model is not a RandomForestClassifier.")

    df = load_logs(args.logs)
    df = prepare_dataframe(df)
    _, val_df = time_split(df)
    X_val = val_df[FEATURES_ORDER].values
    y_val = val_df["label"].values
    proba_val = clf.predict_proba(X_val)[:, 1]

    threshold, best_f1 = search_best_threshold(y_val, proba_val)
    print(f"[INFO] Best F1 threshold={threshold:.2f} (F1={best_f1:.3f}) on validation split.")

    model_json = {
        "version": 1,
        "features_order": FEATURES_ORDER,
        "threshold": threshold,
        "trees": [export_tree(est) for est in clf.estimators_],
    }

    # Round-trip validation
    mismatches = 0
    for idx in range(min(len(val_df), 10)):
        row = val_df.iloc[idx]
        runtime_prob = runtime_predict(model_json, row)
        sklearn_prob = proba_val[idx]
        if abs(runtime_prob - sklearn_prob) > 0.02:
            mismatches += 1
            print(
                "[WARN] Round-trip mismatch on row {}: runtime={:.3f}, sklearn={:.3f}".format(
                    idx, runtime_prob, sklearn_prob
                )
            )
    if mismatches:
        print(f"[WARN] Detected {mismatches} validation mismatches (tolerance 0.02).")
    else:
        print("[INFO] JSON export validated against sklearn predictions.")

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(model_json, indent=2))
    print(f"[INFO] Exported model to {out_path}")


if __name__ == "__main__":
    main()

