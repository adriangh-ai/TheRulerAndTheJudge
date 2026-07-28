"""Psychometric evaluation of LLM judges.

The package intentionally avoids importing JAX/PyMC at package import time.
Use :func:`lm_bmfrm.runtime.configure_runtime` before importing fitting modules.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"

from .api import ExperimentResult, load_experiment, run_experiment

__all__ = [
    "ExperimentResult",
    "load_experiment",
    "run_experiment",
]