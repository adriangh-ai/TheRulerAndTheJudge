"""Runtime configuration for the JAX/PyTensor experiment backend."""
from __future__ import annotations

import os
import warnings


_BACKEND_WARNING_PATTERNS = (
    ".*dtype float64 requested.*",
    ".*dtype int64 requested.*",
    ".*GPU interconnect information.*",
    ".*Some donated buffers were not.*",
)


def configure_runtime(
    *,
    cuda_visible_devices: str = "0",
    xla_preallocate: bool = False,
    pytensor_floatx: str = "float32",
    pytensor_mode: str = "JAX",
    suppress_backend_warnings: bool = True,
) -> None:
    """Apply the runtime settings used by the reported experiments.

    Call this before importing modules that initialize PyTensor, JAX, or PyMC.
    The defaults reproduce the original experiment backend.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = str(xla_preallocate).lower()

    if suppress_backend_warnings:
        for pattern in _BACKEND_WARNING_PATTERNS:
            warnings.filterwarnings("ignore", message=pattern)

    import pytensor

    pytensor.config.floatX = pytensor_floatx
    pytensor.config.mode = pytensor_mode
