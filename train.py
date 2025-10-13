#!/usr/bin/env python3
# export_rf.py — shard RF model into tiny per-tree files for MicroPython

import argparse, json, os, numpy as np, pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.base import clone

DEFAULT_CSV = "stress_features_1.csv"
FEATURES = [
    "hr_mean_bpm",
    "sdnn_ms","rmssd_ms","pnn20",
    "hr_slope_60s","hr_var_60s",
    "sdnn30_ms","rmssd30_ms",
    "cv_ibi","amp_cv",
    "step_hz","acc_vrms","running",
]

def round_float(x, nd=3): return float(np.round(x, nd))

def tree_to_packed_nodes(sktree, thr_nd=3, leaf_nd=4):
    """
    Packed nodes to minimize size.
    Internal: [f, thr, l, r]
    Leaf    : [-1, p1]
    """
    t = sktree.tree_
    L, R, F, T, V = t.children_left, t.children_right, t.feature, t.threshold, t.value
    nodes = []
    def dfs(i):
        idx = len(nodes)
        if L[i] == -1 and R[i] == -1:
            counts = V[i][0]
            s = float(np.sum(counts)) or 1.0
            p1 = float(counts[1]/s) if counts.shape[0] > 1 else 0.0
            nodes.append([-1, round_float(p1, leaf_nd)])
        else:
            nodes.append([int(F[i]), round_float(float(T[i]), thr_nd), None, None])
            li = dfs(L[i]); ri = dfs(R[i])
            nodes[idx][2] = li; nodes[idx][3] = ri
        return idx
    dfs(0)
    return nodes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--out_head", default="model_rf.json", help="header file")
    ap.add_argument("--shard_dir", default="trees_rf", help="dir for per-tree JSONs")
    ap.add_argument("--n_estimators", type=int, default=64)
    ap.add_argument("--max_depth", type=int, default=6)
    ap.add_argument("--min_samples_leaf", type=int, default=3)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--thr_nd", type=int, default=3)
    ap.add_argument("--leaf_nd", type=int, default=4)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    y = (df["kind"] == "stress").astype(int).values
    X = df[[c for c in FEATURES if c in df.columns]].copy().replace([np.inf,-np.inf], np.nan).fillna(0.0).astype(float)
    w = df["weight"].values if "weight" in df.columns else None

    cls = np.bincount(y)
    n_splits = max(2, min(args.n_splits, int(cls.min())))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    base = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        n_jobs=-1, random_state=42
    )

    # OOF preds to tune threshold
    oof = np.zeros(len(df), float)
    f1s = []
    for tr, te in skf.split(X, y):
        m = clone(base).fit(X.iloc[tr], y[tr], sample_weight=None if w is None else w[tr])
        p = m.predict_proba(X.iloc[te])[:,1]
        oof[te] = p
        f1s.append(f1_score(y[te], (p>=0.5).astype(int), zero_division=0))
    best_t = 0.5; best_f1 = -1.0
    for t in np.linspace(0.2, 0.8, 121):
        f = f1_score(y, (oof>=t).astype(int), zero_division=0)
        if f > best_f1: best_f1, best_t = f, float(t)

    rf = clone(base).fit(X, y, sample_weight=w)

    # write header + shards
    os.makedirs(args.shard_dir, exist_ok=True)
    # write each tree
    total_bytes = 0
    for i, est in enumerate(rf.estimators_):
        nodes = tree_to_packed_nodes(est, thr_nd=args.thr_nd, leaf_nd=args.leaf_nd)
        obj = {"nodes": nodes}  # keep per-file minimal
        path = os.path.join(args.shard_dir, f"tree_{i:03d}.json")
        s = json.dumps(obj, separators=(",",":"), ensure_ascii=False)
        with open(path, "w") as f: f.write(s)
        total_bytes += len(s)

    head = {
        "model":"RandomForestPackedSharded","version":"1.0",
        "exported_at": datetime.utcnow().isoformat()+"Z",
        "n_classes":2,"class_names":["calm","stress"],
        "features":[c for c in FEATURES if c in df.columns],
        "n_estimators": int(args.n_estimators),
        "max_depth": int(args.max_depth),
        "min_samples_leaf": int(args.min_samples_leaf),
        "decision_threshold": float(best_t),
        "cv":{"n_splits": int(n_splits),"f1_fold":[float(v) for v in f1s],"f1_oof_tuned": float(best_f1)},
        "nodes_format":"packed",
        "sharded": True,
        "shard_dir": args.shard_dir,
        "tree_pattern": "tree_{:03d}.json"
    }
    with open(args.out_head, "w") as f:
        json.dump(head, f, separators=(",",":"), ensure_ascii=False)

    print(f"Header: {args.out_head}")
    print(f"Trees:  {args.n_estimators} files in {args.shard_dir} (total ~{total_bytes} bytes)")
    print(f"Tuned threshold: {best_t:.3f}, OOF F1={best_f1:.3f}")

if __name__ == "__main__":
    main()
