import sys
import os
import pickle
from collections import OrderedDict
import numpy as np
import pandas as pd

#!/usr/bin/env python3
"""
Read and summarize a pickle file.

Saves a short human-readable summary to stdout. Default path is:
    /projects/midst-experiments/ensemble_attack/diabetes/10k/shadow_models_and_data/initial_model_rmia_1/shadow_workspace/pre_trained_model/rmia_shadows.pkl

Usage:
    python diabetes_eval.py [path/to/file.pkl]
"""


try:
except Exception:
        np = None

try:
except Exception:
        pd = None


def summarize(obj, name="root", max_items=10):
        t = type(obj)
        print(f"{name}: type={t.__module__}.{t.__name__}")
        if isinstance(obj, dict):
                keys = list(obj.keys())
                print(f"  dict with {len(keys)} keys: {keys[:max_items]}{' ...' if len(keys) > max_items else ''}")
        elif isinstance(obj, (list, tuple, set)):
                print(f"  {t.__name__} of length {len(obj)}")
                for i, item in enumerate(list(obj)[:max_items]):
                        summarize(item, name=f"{name}[{i}]", max_items=3)
        elif np is not None and isinstance(obj, np.ndarray):
                print(f"  ndarray shape={obj.shape} dtype={obj.dtype}")
        elif pd is not None and isinstance(obj, pd.DataFrame):
                print(f"  DataFrame shape={obj.shape}")
                print(f"  columns: {list(obj.columns[:max_items])}{' ...' if obj.shape[1] > max_items else ''}")
                print(obj.head(3).to_string())
        elif hasattr(obj, "__dict__"):
                attrs = {k: type(v).__name__ for k, v in vars(obj).items()}
                print(f"  object with attributes: {list(attrs.keys())[:max_items]}{' ...' if len(attrs) > max_items else ''}")
        else:
                # fallback: print repr summary
                r = repr(obj)
                print(f"  repr: {r[:200]}{' ...' if len(r) > 200 else ''}")


def load_pickle(path):
        with open(path, "rb") as f:
                return pickle.load(f)


def main():
        default_path = "/projects/midst-experiments/ensemble_attack/diabetes/10k/shadow_models_and_data/initial_model_rmia_1/shadow_workspace/pre_trained_model/rmia_shadows.pkl"
        path = sys.argv[1] if len(sys.argv) > 1 else default_path

        if not os.path.exists(path):
                print(f"File not found: {path}", file=sys.stderr)
                sys.exit(2)

        try:
                obj = load_pickle(path)
        except Exception as e:
                print(f"Failed to load pickle: {e}", file=sys.stderr)
                sys.exit(3)

        print(f"Loaded pickle: {path}")
        summarize(obj, name="root")


if __name__ == "__main__":
        main()