#!/usr/bin/env python
"""Compare two experiment result directories without recomputing models."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

SUPPORTED = {".csv", ".parquet", ".json", ".npz", ".npy"}

def compare_frame(a: Path, b: Path, rtol: float, atol: float):
    left = pd.read_csv(a) if a.suffix == ".csv" else pd.read_parquet(a)
    right = pd.read_csv(b) if b.suffix == ".csv" else pd.read_parquet(b)
    if list(left.columns) != list(right.columns) or left.shape != right.shape:
        return False, f"schema/shape differs: {left.shape} vs {right.shape}"
    for col in left.columns:
        x, y = left[col], right[col]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            if not np.allclose(x.to_numpy(), y.to_numpy(), rtol=rtol, atol=atol, equal_nan=True):
                return False, f"numeric column differs: {col}"
        elif not x.fillna("<NA>").astype(str).equals(y.fillna("<NA>").astype(str)):
            return False, f"column differs: {col}"
    return True, "ok"

def compare_file(a: Path, b: Path, rtol: float, atol: float):
    if a.suffix in {".csv", ".parquet"}:
        return compare_frame(a, b, rtol, atol)
    if a.suffix == ".json":
        return (json.loads(a.read_text()) == json.loads(b.read_text()), "json differs")
    if a.suffix == ".npy":
        return (np.allclose(np.load(a), np.load(b), rtol=rtol, atol=atol, equal_nan=True), "array differs")
    if a.suffix == ".npz":
        x, y = np.load(a, allow_pickle=True), np.load(b, allow_pickle=True)
        if set(x.files) != set(y.files): return False, "npz keys differ"
        for key in x.files:
            xa, ya = x[key], y[key]
            try:
                equal = np.allclose(xa, ya, rtol=rtol, atol=atol, equal_nan=True)
            except TypeError:
                equal = np.array_equal(xa, ya)
            if not equal: return False, f"npz array differs: {key}"
        return True, "ok"
    return True, "skipped"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("reference", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("--rtol", type=float, default=0.0)
    p.add_argument("--atol", type=float, default=0.0)
    args = p.parse_args()
    ref = {p.relative_to(args.reference): p for p in args.reference.rglob("*") if p.suffix in SUPPORTED}
    cand = {p.relative_to(args.candidate): p for p in args.candidate.rglob("*") if p.suffix in SUPPORTED}
    failures = []
    for rel in sorted(set(ref) | set(cand)):
        if rel not in ref or rel not in cand:
            failures.append((rel, "missing in one directory")); continue
        ok, msg = compare_file(ref[rel], cand[rel], args.rtol, args.atol)
        if not ok: failures.append((rel, msg))
    if failures:
        for rel, msg in failures: print(f"FAIL {rel}: {msg}")
        raise SystemExit(1)
    print(f"PASS: {len(ref)} supported artifacts match")

if __name__ == "__main__": main()
