"""Minimal public API for running and reading evaluation experiments.

The numerical and orchestration implementation remains in
``lm_bmfrm.pipeline.cli``. This module only provides a stable programmatic
surface for human and automated callers.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from typing import Any, Sequence

import pandas as pd


_PRIMARY_TABLES = {
    "leaderboard": "leaderboard_results.parquet",
    "predictions": "predictions.parquet",
    "human_audit": "human_audit.parquet",
    "human_item_audit": "human_item_audit.parquet",
    "llm_audit": "llm_audit.parquet",
    "llm_item_audit": "llm_item_audit.parquet",
    "benchmark_cards": "benchmark_cards.parquet",
    "benchmark_curves": "benchmark_curves.parquet",
    "rhat_records": "rhat_records.parquet",
}


@dataclass(frozen=True)
class ExperimentResult:
    """Read-only access to one completed evaluation directory."""

    run_dir: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_dir", Path(self.run_dir).resolve())

        required = [
            "resolved_config.json",
            "run_config.json",
            "leaderboard_results.parquet",
            "predictions.parquet",
        ]
        missing = [
            filename
            for filename in required
            if not (self.run_dir / filename).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{self.run_dir} is missing required experiment artifacts: "
                + ", ".join(missing)
            )

    def read_table(self, name: str) -> pd.DataFrame:
        """Read a documented primary Parquet output."""

        try:
            filename = _PRIMARY_TABLES[name]
        except KeyError as exc:
            valid = ", ".join(sorted(_PRIMARY_TABLES))
            raise KeyError(
                f"Unknown table {name!r}. Valid names: {valid}."
            ) from exc

        path = self.run_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)

        return pd.read_parquet(path)

    def read_json(self, filename: str) -> dict[str, Any]:
        """Read a JSON artifact from this run directory."""

        path = self.run_dir / filename
        if path.parent != self.run_dir:
            raise ValueError("filename must refer directly to the run directory.")
        if not path.is_file():
            raise FileNotFoundError(path)

        return json.loads(path.read_text(encoding="utf-8"))

    @property
    def leaderboard(self) -> pd.DataFrame:
        return self.read_table("leaderboard")

    @property
    def predictions(self) -> pd.DataFrame:
        return self.read_table("predictions")

    @property
    def resolved_config(self) -> dict[str, Any]:
        return self.read_json("resolved_config.json")

    @property
    def provenance(self) -> dict[str, Any]:
        return self.read_json("run_config.json")

    def machine_summary(self) -> dict[str, Any]:
        """Return a compact machine-readable completion record."""

        available_tables = {
            name: str(self.run_dir / filename)
            for name, filename in _PRIMARY_TABLES.items()
            if (self.run_dir / filename).is_file()
        }

        return {
            "status": "complete",
            "run_dir": str(self.run_dir),
            "resolved_config": str(self.run_dir / "resolved_config.json"),
            "provenance": str(self.run_dir / "run_config.json"),
            "input_manifest": str(
                self.run_dir / "judge_input_manifest.parquet"
            ),
            "tables": available_tables,
        }


def load_experiment(run_dir: str | Path) -> ExperimentResult:
    """Open an existing evaluation result directory."""

    return ExperimentResult(Path(run_dir))


def run_experiment(
    config_path: str | Path,
    *,
    results_root: str | Path | None = None,
    extra_args: Sequence[str] = (),
) -> ExperimentResult:
    """Run the existing Stage 1--3 CLI pipeline programmatically.

    Parameters
    ----------
    config_path:
        JSON configuration accepted by ``lm-bmfrm-evaluate``.
    results_root:
        Parent directory for the timestamped experiment directory. A temporary
        directory is used when omitted.
    extra_args:
        Additional CLI arguments passed unchanged to the established
        orchestrator.

    Notes
    -----
    This function does not implement or reorder any preprocessing, fitting,
    validation, metric, or serialization operation. It calls the same
    ``pipeline.cli.main`` function used by the packaged command-line entry
    point.
    """

    config_path = Path(config_path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    if results_root is None:
        root = Path(tempfile.mkdtemp(prefix="lm_bmfrm_"))
    else:
        root = Path(results_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)

    before = {
        path.resolve()
        for path in root.iterdir()
        if path.is_dir()
    }

    argv = [
        "--config",
        str(config_path),
        "--results_root",
        str(root),
        *[str(value) for value in extra_args],
    ]

    # Delayed import preserves the existing runtime/import sequence.
    from lm_bmfrm.runtime import configure_runtime

    configure_runtime()
    
    from lm_bmfrm.pipeline.cli import main

    main(argv)

    after = {
        path.resolve()
        for path in root.iterdir()
        if path.is_dir()
    }
    created = sorted(after - before)

    if len(created) != 1:
        raise RuntimeError(
            "Expected exactly one new experiment directory under "
            f"{root}, but found {len(created)}: {created}"
        )

    return ExperimentResult(created[0])