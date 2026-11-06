import json

import pandas as pd

from lm_bmfrm.api import load_experiment, run_experiment


def _write_fake_run(run_dir):
    run_dir.mkdir(parents=True)

    (run_dir / "resolved_config.json").write_text(
        json.dumps({"sampling": {"sampler_seed": 42}}),
        encoding="utf-8",
    )
    (run_dir / "run_config.json").write_text(
        json.dumps({"git_commit": "test"}),
        encoding="utf-8",
    )

    pd.DataFrame(
        [{"Dimension": "coherence", "Model": "judge", "NSMF": 0.5}]
    ).to_parquet(run_dir / "leaderboard_results.parquet", index=False)

    pd.DataFrame(
        [{"dimension": "coherence", "judge_model": "judge"}]
    ).to_parquet(run_dir / "predictions.parquet", index=False)


def test_load_experiment(tmp_path):
    run_dir = tmp_path / "existing_run"
    _write_fake_run(run_dir)

    result = load_experiment(run_dir)

    assert result.run_dir == run_dir.resolve()
    assert result.resolved_config["sampling"]["sampler_seed"] == 42
    assert result.leaderboard.iloc[0]["Model"] == "judge"
    assert result.machine_summary()["status"] == "complete"


def test_run_experiment_delegates_to_existing_cli(
    tmp_path,
    monkeypatch,
):
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")

    def fake_main(argv):
        assert "--config" in argv
        assert str(config_path.resolve()) in argv
        assert "--results_root" in argv

        root = tmp_path / "results"
        _write_fake_run(root / "generated_run")

    monkeypatch.setattr(
        "lm_bmfrm.pipeline.cli.main",
        fake_main,
    )

    result = run_experiment(
        config_path,
        results_root=tmp_path / "results",
    )

    assert result.run_dir.name == "generated_run"