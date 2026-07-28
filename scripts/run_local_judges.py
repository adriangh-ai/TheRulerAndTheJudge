#!/usr/bin/env python
"""Run local judge generation using the original module CLI."""
import runpy

if __name__ == "__main__":
    runpy.run_module("lm_bmfrm.generation.local", run_name="__main__")
