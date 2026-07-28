#!/usr/bin/env python
"""Run API-based judge generation from a checkout or installation."""
from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lm_bmfrm.generation.api import main


if __name__ == "__main__":
    main()
