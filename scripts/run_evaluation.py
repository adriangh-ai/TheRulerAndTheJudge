#!/usr/bin/env python
"""Run the Stage 1--3 evaluation pipeline from a checkout or installation."""
from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lm_bmfrm.runtime import configure_runtime

configure_runtime()

from lm_bmfrm.pipeline.cli import main


if __name__ == "__main__":
    main()
