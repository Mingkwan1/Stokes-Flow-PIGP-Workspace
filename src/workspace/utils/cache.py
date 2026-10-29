"""Hyperparameter cache: config fingerprint + load/save of fitted theta.

The fingerprint hashes everything that changes the optimum, so a cache fitted
for different physics/settings is never loaded silently. Hash and file formats
are identical to the ones previously inlined in the main script, so existing
cache files stay valid.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def arr_hash(*arrays):
    """Content hash of point sets - catches a moved collocation grid that
    a shape-only check would miss."""
    h = hashlib.sha256()
    for A in arrays:
        h.update(np.ascontiguousarray(np.asarray(A), dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


def fingerprint(cfg):
    """16-hex-char hash of a JSON-serialisable config dict."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def load_theta(path, fp, expected_shape):
    """Return the cached theta as a numpy array, or None if the file is
    missing, the fingerprint differs, or the parameter count changed."""
    path = Path(path)
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=False)
    if str(d["fingerprint"]) != fp:
        print("[cache] fingerprint mismatch (config changed) -> refitting")
        return None
    th = np.asarray(d["theta"])
    if th.shape != tuple(expected_shape):
        print(f"[cache] parameter count changed ({th.shape} vs {tuple(expected_shape)}) -> refitting")
        return None
    return th


def save_theta(path, fp, theta, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path,
             theta=np.asarray(theta),
             fingerprint=np.asarray(fp),
             value=np.asarray(float(value)))
    print(f"[cache] wrote {path}")