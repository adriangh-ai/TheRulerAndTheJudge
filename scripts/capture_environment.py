#!/usr/bin/env python
"""Capture the software environment used for an experiment run."""
from __future__ import annotations
import argparse, json, os, platform, subprocess, sys
from pathlib import Path

def command(args):
    try:
        return subprocess.check_output(args, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return f"unavailable: {exc}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path, nargs="?", default=Path("environment.json"))
    args = parser.parse_args()
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "pip_freeze": command([sys.executable, "-m", "pip", "freeze"]).splitlines(),
        "git_commit": command(["git", "rev-parse", "HEAD"]),
        "git_status": command(["git", "status", "--porcelain"]),
        "nvidia_smi": command(["nvidia-smi"]),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(args.output)

if __name__ == "__main__":
    main()
