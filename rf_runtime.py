"""
rf_runtime.py
-------------
Minimal Random Forest runtime for the compact JSON model produced by the PC
training scripts. Designed for MicroPython.
"""

import json

MODEL_PATH = "model/model_user.json"


class RandomForestRuntime:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.features_order = []
        self.threshold = 0.6

    # ------------------------------------------------------------------ loading
    def load(self):
        try:
            with open(self.model_path) as fp:
                data = json.load(fp)
        except OSError:
            self.model = None
            return False
        except ValueError:
            self.model = None
            return False

        trees = data.get("trees")
        features = data.get("features_order")
        if not isinstance(trees, list) or not isinstance(features, list):
            self.model = None
            return False

        self.model = {
            "trees": trees,
        }
        self.features_order = features
        self.threshold = float(data.get("threshold", 0.6))
        return True

    def ensure_loaded(self):
        if self.model is None:
            self.load()
        return self.model is not None

    # ---------------------------------------------------------------- prediction
    def predict_proba(self, feature_dict):
        if not self.ensure_loaded():
            return None
        if not self.model["trees"]:
            return None

        # Build feature vector in required order
        vector = []
        for name in self.features_order:
            vector.append(float(feature_dict.get(name, 0.0)))

        total = 0.0
        count = 0
        for tree in self.model["trees"]:
            nodes = tree.get("nodes")
            if not nodes:
                continue
            idx = tree.get("root", 0)
            prob = self._traverse(nodes, idx, vector)
            total += prob
            count += 1
        if count == 0:
            return None
        return max(0.0, min(1.0, total / count))

    def _traverse(self, nodes, idx, vector):
        # Iterative traversal to avoid recursion limits
        while True:
            node = nodes[idx]
            if node.get("leaf") is not None:
                return float(node["leaf"])
            fi = node.get("fi")
            th = node.get("th")
            if fi is None or th is None:
                # malformed node, bail out
                return 0.5
            try:
                feat = vector[int(fi)]
            except IndexError:
                feat = 0.0
            if feat <= th:
                idx = node.get("lt", idx)
            else:
                idx = node.get("rt", idx)

    # ---------------------------------------------------------------- utilities
    def get_threshold(self):
        return self.threshold

    def self_test(self):
        """Basic sanity check with a dummy tree."""
        self.model = {
            "trees": [
                {
                    "nodes": [
                        {"fi": 0, "th": 0.5, "lt": 1, "rt": 2, "leaf": None},
                        {"leaf": 0.2},
                        {"leaf": 0.8},
                    ],
                    "root": 0,
                }
            ]
        }
        self.features_order = ["x"]
        prob_lo = self.predict_proba({"x": 0.4})
        prob_hi = self.predict_proba({"x": 0.6})
        return prob_lo == 0.2 and prob_hi == 0.8


__all__ = ["RandomForestRuntime", "MODEL_PATH"]

