"""Installed console entry points."""

def evaluate() -> None:
    from lm_bmfrm.runtime import configure_runtime
    configure_runtime()
    from lm_bmfrm.pipeline.cli import main
    main()
