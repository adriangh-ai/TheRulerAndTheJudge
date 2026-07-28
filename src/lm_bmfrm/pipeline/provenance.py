"""Run provenance and sampler diagnostics."""
from __future__ import annotations
import json, os, subprocess, sys, warnings
from datetime import datetime
import arviz as az
import numpy as np


def _git_commit_hash():
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


def _git_is_dirty():
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL)
        return bool(out.decode().strip())
    except Exception:
        return None


def _package_versions():
    out = {"python": sys.version}
    for name in ["numpy", "pandas", "pymc", "arviz", "jax", "jaxlib", "pytensor", "scipy", "sklearn", "numpyro"]:
        try:
            mod = __import__(name)
            out[name] = str(getattr(mod, "__version__", "unknown"))
        except Exception:
            out[name] = "unavailable"
    return out


def write_run_config(results_dir, args, extra=None):
    config = {
        "argv": sys.argv,
        "args": {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v)) for k, v in vars(args).items()},
        "git_commit": _git_commit_hash(),
        "git_dirty": _git_is_dirty(),
        "package_versions": _package_versions(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "written_at": datetime.now().isoformat(),
    }
    if extra:
        config.update(extra)
    try:
        with open(os.path.join(results_dir, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, sort_keys=True, default=str)
    except Exception as e:
        warnings.warn(f"Could not write run_config.json: {e}")
    return config


def _sampler_diagnostics(idata):
    """Return (n_divergences, min_ess_bulk) with defensive fallbacks."""
    n_div = np.nan
    min_ess = np.nan
    try:
        if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
            n_div = int(np.asarray(idata.sample_stats["diverging"]).sum())
    except Exception:
        pass
    try:
        min_ess = float(az.ess(idata).to_array().to_series().dropna().min())
    except Exception:
        pass
    return n_div, min_ess
