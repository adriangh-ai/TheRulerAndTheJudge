from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from lm_bmfrm.pipeline.config import load_experiment_config
from lm_bmfrm.pipeline.data_quality import (
    load_judge_outputs,
    resolve_dimensions,
    resolve_judge_files,
    validate_judgment_dataframe,
)


ROOT = Path(__file__).parents[1]


def _judgment_frame(score: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": ["i1"],
            "system_id": ["s1"],
            "rater_id": ["j1"],
            "dimension": ["coherence"],
            "score": [score],
        }
    )


def test_external_paper_config_matches_packaged_defaults():
    packaged = load_experiment_config()
    external = load_experiment_config(ROOT / "configs" / "paper_default.json")
    assert external == packaged
    assert packaged.model.use_mc is True
    assert packaged.sampling.draws == 2000
    assert packaged.evaluation.theta_grid.num == 401


def test_partial_config_merges_without_mutating_other_defaults(tmp_path):
    path = tmp_path / "override.json"
    path.write_text(json.dumps({"sampling": {"draws": 17}}), encoding="utf-8")
    config = load_experiment_config(path)
    assert config.sampling.draws == 17
    assert config.sampling.chains_audit == 4
    assert config.evaluation.tau_h == 0.8


def test_unknown_config_key_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"sampling": {"drawws": 17}}), encoding="utf-8")
    with pytest.raises(ValueError, match="drawws"):
        load_experiment_config(path)


def test_explicit_judge_file_order_is_preserved(tmp_path, monkeypatch):
    first = tmp_path / "b.parquet"
    second = tmp_path / "a.parquet"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    frames = {
        str(first): _judgment_frame(score=2),
        str(second): _judgment_frame(score=4).assign(rater_id="j2"),
    }
    monkeypatch.setattr(pd, "read_parquet", lambda path: frames[str(path)].copy())

    resolved = resolve_judge_files(
        judge_folder=str(tmp_path),
        benchmark="summeval",
        explicit_files=[str(first), str(second)],
    )
    loaded, exclusions, manifest = load_judge_outputs(resolved)

    assert resolved == [str(first), str(second)]
    assert loaded["score"].tolist() == [2, 4]
    assert manifest["filename"].tolist() == ["b.parquet", "a.parquet"]
    assert exclusions["n_dropped_nan_score"].tolist() == [0, 0]


def test_schema_validation_rejects_out_of_range_scores():
    with pytest.raises(ValueError, match="outside"):
        validate_judgment_dataframe(_judgment_frame(score=6), source="fixture")


def test_dimension_resolution_preserves_first_seen_order():
    frame = pd.concat(
        [
            _judgment_frame().assign(dimension="fluency"),
            _judgment_frame().assign(dimension="coherence"),
        ],
        ignore_index=True,
    )
    assert resolve_dimensions(frame, "summeval", None) == ["fluency", "coherence"]


def test_score_list_validation_rejects_out_of_range_values(tmp_path, monkeypatch):
    path = tmp_path / "judge.parquet"
    path.write_bytes(b"fixture")
    frame = _judgment_frame().assign(score_list=[[3, 6]])
    monkeypatch.setattr(pd, "read_parquet", lambda _: frame.copy())
    with pytest.raises(ValueError, match="score_list"):
        load_judge_outputs([str(path)])
