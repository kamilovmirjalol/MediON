# rf_infer.py — streaming/sharded RF inference for MicroPython (non-blocking)
try:
    import ujson as json  # faster on MicroPython
except:
    import json

try:
    import os as uos, gc
except:
    uos = None
    gc = None

try:
    import math
except:
    math = None

def _finite(v):
    try:
        if math and hasattr(math, "isfinite"):
            return v if math.isfinite(v) else 0.0
    except:
        pass
    try:
        if v != v:  # NaN
            return 0.0
    except:
        return 0.0
    return v

def load_model_head(path="model_rf.json", log_fn=print):
    try:
        with open(path, "r") as f:
            data = f.read()
        head = json.loads(data)
        for k in ("features", "n_estimators", "decision_threshold"):
            if k not in head:
                log_fn("model head missing key: {}".format(k))
                return None
        if "sharded" not in head or not head["sharded"]:
            log_fn("model head is not sharded; export with sharding.")
            return None
        return head
    except Exception as e:
        log_fn("load_model_head error: {}".format(e))
        return None

def make_feature_vector(model_head, feats_dict):
    names = model_head.get("features", [])
    x = []
    for name in names:
        v = feats_dict.get(name, 0.0)
        if v is None:
            v = 0.0
        x.append(float(_finite(v)))
    return x

# Packed node traversal (one tree already loaded)
# Leaf: [-1, p1]; Internal: [f, thr, l, r]
def _tree_proba_packed(nodes, x):
    idx = 0
    while True:
        n = nodes[idx]
        if n[0] == -1:
            p1 = n[1]
            if p1 < 0.0: p1 = 0.0
            if p1 > 1.0: p1 = 1.0
            return p1
        f, thr, l, r = n[0], n[1], n[2], n[3]
        xv = x[f] if 0 <= f < len(x) else 0.0
        idx = l if xv <= thr else r

def _load_tree_nodes(path):
    with open(path, "r") as f:
        data = f.read()
    obj = json.loads(data)
    return obj["nodes"]

class RFStreamer:
    """
    Non-blocking sharded RF evaluator.
    - call start(x) when you have a fresh feature vector
    - call step(k) each loop to process up to k trees
    - when done, prob() gives the averaged probability
    """
    def __init__(self, model_head, shard_dir=None, patt=None):
        self.mh = model_head
        self.n = int(model_head.get("n_estimators", 0))
        self.dir = shard_dir or model_head.get("shard_dir", "trees_rf")
        self.patt = patt or model_head.get("tree_pattern", "tree_{:03d}.json")
        self.reset()

    def reset(self):
        self._x = None
        self._i = 0
        self._acc = 0.0
        self._done = True
        self._last_prob = 0.5

    def start(self, x):
        self._x = x
        self._i = 0
        self._acc = 0.0
        self._done = (self.n <= 0)

    def step(self, max_trees=2):
        """
        Process up to max_trees trees this call. Keep it small (2–4).
        """
        if self._done or self._x is None:
            return
        end = min(self._i + max_trees, self.n)
        for j in range(self._i, end):
            try:
                path = "{}/{}".format(self.dir, self.mh.get("tree_pattern", "tree_{:03d}.json").format(j))
                nodes = _load_tree_nodes(path)
                p = _tree_proba_packed(nodes, self._x)
                self._acc += p
                # free as we go
                del nodes
                if gc: gc.collect()
            except Exception as e:
                # ignore this tree; add neutral prob
                self._acc += 0.5
        self._i = end
        if self._i >= self.n:
            self._done = True
            self._last_prob = (self._acc / float(self.n)) if self.n > 0 else 0.5

    def done(self):
        return self._done

    def prob(self):
        return self._last_prob
        

def predict_is_stress_stream(streamer, decision_threshold):
    p = streamer.prob()
    return (p >= float(decision_threshold)), p

def features_from_bfe(feats, step_hz, acc_vrms, running_bool):
    return {
        "hr_mean_bpm":  float(feats.get("hr_mean_bpm", 0.0)),
        "sdnn_ms":      float(feats.get("sdnn_ms", 0.0)),
        "rmssd_ms":     float(feats.get("rmssd_ms", 0.0)),
        "pnn20":        float(feats.get("pnn20", 0.0)),
        "hr_slope_60s": float(feats.get("hr_slope_60s", 0.0)),
        "hr_var_60s":   float(feats.get("hr_var_60s", 0.0)),
        "sdnn30_ms":    float(feats.get("sdnn30_ms", 0.0)),
        "rmssd30_ms":   float(feats.get("rmssd30_ms", 0.0)),
        "cv_ibi":       float(feats.get("cv_ibi", 0.0)),
        "amp_cv":       float(feats.get("amp_cv", 0.0)),
        "step_hz":      float(step_hz or 0.0),
        "acc_vrms":     float(acc_vrms or 0.0),
        "running":      1.0 if running_bool else 0.0,
    }

def make_x(model_head, feats_dict):
    return make_feature_vector(model_head, feats_dict)

# rf_infer.py  — inside class RFStreamer
def progress(self):
    """
    Returns (n_done, n_total). Safe to call anytime.
    """
    try:
        return int(self._i), int(self._n_trees)
    except:
        # Fallback if attribute names differ in your implementation
        n_total = getattr(self, "n_trees", 0)
        n_done  = getattr(self, "i", 0)
        return int(n_done), int(n_total)

def busy(self):
    """True while a job is in progress."""
    return not self.done()

