#!/usr/bin/env python3

import argparse
import json
import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score


FEATURES = [
    "mean_hr_bpm",
    "sdnn_ms",
    "rmssd_ms",
    "pnn20",
    "sd1_ms",
    "sd2_ms",
    "amp_mean",
    "rise_ms_mean",
    "width50_ms_mean",
]


def load_events(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(glob.glob(os.path.join(p, "*.csv")))
        else:
            files.extend(glob.glob(p))
    if not files:
        raise SystemExit("No CSV files found")
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["_src"] = f
            dfs.append(df)
        except Exception as e:
            print("[WARN] skip {}: {}".format(f, e))
    if not dfs:
        raise SystemExit("No readable CSV files")
    data = pd.concat(dfs, ignore_index=True)
    return data


def map_labels(df):
    y = None
    if "label" in df.columns and df["label"].dtype != np.number:
        # text labels calm/stress
        y = df["label"].map({"calm": 0, "stress": 1})
    elif "label" in df.columns:
        y = df["label"].astype(int)
    elif "label_id" in df.columns:
        y = df["label_id"].astype(int)
    else:
        raise SystemExit("Required label column missing: label or label_id")
    # drop rows without labels (e.g., session_summary)
    mask = y.isin([0, 1])
    return y[mask], mask


def time_split(df, frac=0.2):
    if "timestamp_ms" in df.columns:
        order = np.argsort(df["timestamp_ms"].values)
        n = len(order)
        split = int(n * (1 - frac))
        idx_train = order[:split]
        idx_val = order[split:]
        return idx_train, idx_val
    # fallback: random split
    n = len(df)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(n * (1 - frac))
    return idx[:split], idx[split:]


def export_sharded(rf, threshold, out_head, shard_dir):
    out_head = Path(out_head)
    trees_dir = Path(shard_dir)
    trees_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "type": "random_forest",
        "n_estimators": len(rf.estimators_),
        "class_weight": "balanced",
        "features": FEATURES,
        # Store both keys for threshold to support multiple loaders
        "threshold": float(threshold),
        "decision_threshold": float(threshold),
        # Shard directory info for loaders
        "trees_path": str(trees_dir.name),
        "shard_dir": str(trees_dir.as_posix()),
        "tree_pattern": "tree_{:04d}.json",
    }
    with open(out_head, "w") as f:
        json.dump(meta, f)
    for i, est in enumerate(rf.estimators_):
        t = est.tree_
        tree_obj = {
            "children_left": t.children_left.tolist(),
            "children_right": t.children_right.tolist(),
            "feature": t.feature.tolist(),
            "threshold": t.threshold.tolist(),
            # store positive-class probability at each node (leaf used in inference)
            "value_pos": (t.value[:, 0, 1] / np.maximum(t.value[:, 0, :].sum(axis=1), 1e-9)).tolist(),
        }
        with open(trees_dir / f"tree_{i:04d}.json", "w") as f:
            json.dump(tree_obj, f)


def main():
    ap = argparse.ArgumentParser()
    # Input selection: either --csv paths or convenient --events.csv flag
    ap.add_argument("--csv", nargs="+", help="CSV files or directories")
    ap.add_argument("--events.csv", dest="events_flag", action="store_true",
                    help="Use ./events.csv as the sole input")
    ap.add_argument("--n_estimators", type=int, default=200)
    ap.add_argument("--max_depth", type=int, default=None)
    ap.add_argument("--out", default=".", help="(Optional) base output directory")
    ap.add_argument("--out_head", default="model_rf.json",
                    help="Path for model head JSON (default: model_rf.json)")
    ap.add_argument("--shard_dir", default="trees_rf",
                    help="Directory for tree shards (default: trees_rf)")
    ap.add_argument("--min_samples_leaf", type=int, default=1)
    args = ap.parse_args()

    inputs = []
    if args.events_flag:
        inputs.append("events.csv")
    if args.csv:
        inputs.extend(args.csv)
    if not inputs:
        raise SystemExit("Provide --csv <paths> or --events.csv")

    df = load_events(inputs)
    y_all, mask = map_labels(df)
    df = df.loc[mask].reset_index(drop=True)
    y = y_all.reset_index(drop=True)

    # QC filters (optional)
    if "sqi" in df.columns:
        df = df[df["sqi"] >= 0.4].reset_index(drop=True)
        y = y.loc[df.index]
    if "n_beats" in df.columns:
        df = df[df["n_beats"] >= 6].reset_index(drop=True)
        y = y.loc[df.index]

    # Ensure features exist
    for c in FEATURES:
        if c not in df.columns:
            raise SystemExit(f"Missing feature: {c}")

    X = df[FEATURES].astype(float).values
    y = y.values

    idx_train, idx_val = time_split(df, frac=0.2)
    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X[idx_train], y[idx_train])

    # tune threshold by F1 on time-based validation
    prob = rf.predict_proba(X[idx_val])[:, 1]
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.2, 0.8, 61):
        pred = (prob >= thr).astype(int)
        f1 = f1_score(y[idx_val], pred)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    print("Val F1:", best_f1, "Thr:", best_thr)

    # Resolve output paths
    out_head = args.out_head
    shard_dir = args.shard_dir
    # If --out provided and out_head/shard_dir are relative, place them under --out
    if args.out and os.path.isdir(args.out):
        if not os.path.isabs(out_head):
            out_head = str(Path(args.out) / out_head)
        if not os.path.isabs(shard_dir):
            shard_dir = str(Path(args.out) / shard_dir)

    export_sharded(rf, best_thr, out_head=out_head, shard_dir=shard_dir)


if __name__ == "__main__":
    main()
