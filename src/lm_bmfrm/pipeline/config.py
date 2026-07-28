"""Typed experiment configuration for the Stage 1--3 pipeline.

The packaged JSON file is the canonical default configuration. A user-provided
JSON file is recursively merged over it, and command-line arguments can then
override individual fields. This keeps the historical defaults unchanged while
making every paper-relevant setting explicit and serializable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.resources import files
import json
from pathlib import Path
from typing import Any


BENCHMARKS = {"summeval", "summeval_synth", "hitz_basse", "hanna", "hitz_basse_vas"}
ANCHOR_DRAW_SELECTIONS = {"medoids", "linspace"}
CV_MODES = {"nested", "loocv"}
OUTER_SPLIT_STRATEGIES = {"random", "stratified_theta", "constrained_blocked"}
DIF_VARIANTS = {"mc", "meanfield"}


@dataclass(frozen=True)
class DataConfig:
    benchmark: str | None = None
    judge_folder: str = "llm_judge_outputs"
    judge_files: tuple[str, ...] | None = None
    dimensions: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ModelConfig:
    anchor_draw_selection: str = "linspace"
    use_mc: bool = True
    add_alpha_main: bool = False
    audit_add_alpha: bool = False
    dif_variant: str = "mc"


@dataclass(frozen=True)
class SamplingConfig:
    draws: int = 2000
    chains_audit: int = 4
    chains_cv: int = 4
    target_accept: float = 0.99
    sampler_seed: int = 42
    anchor_draw_compute_n: int = 200
    anchor_draw_fit_n: int = 200
    anchor_draw_compute_seed: int = 42
    anchor_draw_fit_seed: int = 123


@dataclass(frozen=True)
class ThetaGridConfig:
    min: float = -8.0
    max: float = 8.0
    num: int = 401


@dataclass(frozen=True)
class EvaluationConfig:
    cv_mode: str = "loocv"
    theta_grid: ThetaGridConfig = ThetaGridConfig()
    tau_h: float = 0.8
    tau_l: float = 0.8
    conformal_alpha: float = 0.05
    kfold_seed: int = 42


@dataclass(frozen=True)
class NestedCVConfig:
    outer_test_size: float = 0.30
    outer_splits: int = 10
    outer_seed: int = 42
    outer_split_strategy: str = "constrained_blocked"
    outer_stratify_dimension: str = "__per_dimension__"
    outer_constraint_tau_h: float = 0.8
    outer_min_supported_pairs: int = -1
    outer_max_supported_pairs: int = -1
    outer_min_sr_h: float | None = None
    outer_max_sr_h: float | None = None
    outer_min_spread_fraction: float = 0.50
    outer_max_candidate_splits: int = 20000


@dataclass(frozen=True)
class OutputConfig:
    results_root: str = "results"
    save_artifacts: bool = True
    save_idata: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    sampling: SamplingConfig
    evaluation: EvaluationConfig
    nested_cv: NestedCVConfig
    outputs: OutputConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _reject_unknown_keys(mapping: dict[str, Any], allowed: set[str], section: str) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown configuration key(s) in '{section}': {names}")


def _as_optional_tuple(value: Any, field_name: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"'{field_name}' must be null or a non-empty-string list.")
    if len(set(value)) != len(value):
        raise ValueError(f"'{field_name}' contains duplicate entries.")
    return tuple(value)


def _require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"'{field_name}' must be a JSON boolean.")
    return value


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{field_name}' must be a non-empty string.")
    return value


def _validate_probability(value: float, field_name: str, *, open_interval: bool = False) -> float:
    value = float(value)
    valid = 0.0 < value < 1.0 if open_interval else 0.0 <= value <= 1.0
    if not valid:
        interval = "(0, 1)" if open_interval else "[0, 1]"
        raise ValueError(f"'{field_name}' must be in {interval}; got {value}.")
    return value


def _positive_int(value: Any, field_name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"'{field_name}' must be positive; got {value}.")
    return value


def _parse_config(raw: dict[str, Any]) -> ExperimentConfig:
    top_sections = {"data", "model", "sampling", "evaluation", "nested_cv", "outputs"}
    _reject_unknown_keys(raw, top_sections, "root")
    missing = top_sections - set(raw)
    if missing:
        raise ValueError(f"Missing configuration section(s): {', '.join(sorted(missing))}")

    data = raw["data"]
    model = raw["model"]
    sampling = raw["sampling"]
    evaluation = raw["evaluation"]
    nested = raw["nested_cv"]
    outputs = raw["outputs"]

    _reject_unknown_keys(data, {"benchmark", "judge_folder", "judge_files", "dimensions"}, "data")
    _reject_unknown_keys(
        model,
        {"anchor_draw_selection", "use_mc", "add_alpha_main", "audit_add_alpha", "dif_variant"},
        "model",
    )
    _reject_unknown_keys(
        sampling,
        {
            "draws",
            "chains_audit",
            "chains_cv",
            "target_accept",
            "sampler_seed",
            "anchor_draw_compute_n",
            "anchor_draw_fit_n",
            "anchor_draw_compute_seed",
            "anchor_draw_fit_seed",
        },
        "sampling",
    )
    _reject_unknown_keys(
        evaluation,
        {"cv_mode", "theta_grid", "tau_h", "tau_l", "conformal_alpha", "kfold_seed"},
        "evaluation",
    )
    _reject_unknown_keys(
        nested,
        {
            "outer_test_size",
            "outer_splits",
            "outer_seed",
            "outer_split_strategy",
            "outer_stratify_dimension",
            "outer_constraint_tau_h",
            "outer_min_supported_pairs",
            "outer_max_supported_pairs",
            "outer_min_sr_h",
            "outer_max_sr_h",
            "outer_min_spread_fraction",
            "outer_max_candidate_splits",
        },
        "nested_cv",
    )
    _reject_unknown_keys(outputs, {"results_root", "save_artifacts", "save_idata"}, "outputs")

    theta_grid = evaluation["theta_grid"]
    _reject_unknown_keys(theta_grid, {"min", "max", "num"}, "evaluation.theta_grid")

    benchmark = data["benchmark"]
    if benchmark is not None and benchmark not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark '{benchmark}'.")
    if model["anchor_draw_selection"] not in ANCHOR_DRAW_SELECTIONS:
        raise ValueError(f"Unknown anchor_draw_selection '{model['anchor_draw_selection']}'.")
    if model["dif_variant"] not in DIF_VARIANTS:
        raise ValueError(f"Unknown dif_variant '{model['dif_variant']}'.")
    if evaluation["cv_mode"] not in CV_MODES:
        raise ValueError(f"Unknown cv_mode '{evaluation['cv_mode']}'.")
    if nested["outer_split_strategy"] not in OUTER_SPLIT_STRATEGIES:
        raise ValueError(f"Unknown outer_split_strategy '{nested['outer_split_strategy']}'.")

    grid_min = float(theta_grid["min"])
    grid_max = float(theta_grid["max"])
    if not grid_min < grid_max:
        raise ValueError("evaluation.theta_grid.min must be smaller than max.")
    grid_num = _positive_int(theta_grid["num"], "evaluation.theta_grid.num")
    if grid_num <= 1:
        raise ValueError("evaluation.theta_grid.num must be greater than 1.")
    if float(nested["outer_min_spread_fraction"]) < 0.0:
        raise ValueError("nested_cv.outer_min_spread_fraction must be nonnegative.")

    outer_min_sr_h = nested["outer_min_sr_h"]
    outer_max_sr_h = nested["outer_max_sr_h"]
    if outer_min_sr_h is not None:
        outer_min_sr_h = _validate_probability(outer_min_sr_h, "nested_cv.outer_min_sr_h")
    if outer_max_sr_h is not None:
        outer_max_sr_h = _validate_probability(outer_max_sr_h, "nested_cv.outer_max_sr_h")
    if outer_min_sr_h is not None and outer_max_sr_h is not None and outer_min_sr_h > outer_max_sr_h:
        raise ValueError("nested_cv.outer_min_sr_h cannot exceed outer_max_sr_h.")

    return ExperimentConfig(
        data=DataConfig(
            benchmark=benchmark,
            judge_folder=_require_string(data["judge_folder"], "data.judge_folder"),
            judge_files=_as_optional_tuple(data["judge_files"], "data.judge_files"),
            dimensions=_as_optional_tuple(data["dimensions"], "data.dimensions"),
        ),
        model=ModelConfig(
            anchor_draw_selection=str(model["anchor_draw_selection"]),
            use_mc=_require_bool(model["use_mc"], "model.use_mc"),
            add_alpha_main=_require_bool(model["add_alpha_main"], "model.add_alpha_main"),
            audit_add_alpha=_require_bool(model["audit_add_alpha"], "model.audit_add_alpha"),
            dif_variant=str(model["dif_variant"]),
        ),
        sampling=SamplingConfig(
            draws=_positive_int(sampling["draws"], "sampling.draws"),
            chains_audit=_positive_int(sampling["chains_audit"], "sampling.chains_audit"),
            chains_cv=_positive_int(sampling["chains_cv"], "sampling.chains_cv"),
            target_accept=_validate_probability(
                sampling["target_accept"], "sampling.target_accept", open_interval=True
            ),
            sampler_seed=int(sampling["sampler_seed"]),
            anchor_draw_compute_n=_positive_int(
                sampling["anchor_draw_compute_n"], "sampling.anchor_draw_compute_n"
            ),
            anchor_draw_fit_n=_positive_int(
                sampling["anchor_draw_fit_n"], "sampling.anchor_draw_fit_n"
            ),
            anchor_draw_compute_seed=int(sampling["anchor_draw_compute_seed"]),
            anchor_draw_fit_seed=int(sampling["anchor_draw_fit_seed"]),
        ),
        evaluation=EvaluationConfig(
            cv_mode=str(evaluation["cv_mode"]),
            theta_grid=ThetaGridConfig(
                min=grid_min,
                max=grid_max,
                num=grid_num,
            ),
            tau_h=_validate_probability(evaluation["tau_h"], "evaluation.tau_h"),
            tau_l=_validate_probability(evaluation["tau_l"], "evaluation.tau_l"),
            conformal_alpha=_validate_probability(
                evaluation["conformal_alpha"], "evaluation.conformal_alpha", open_interval=True
            ),
            kfold_seed=int(evaluation["kfold_seed"]),
        ),
        nested_cv=NestedCVConfig(
            outer_test_size=_validate_probability(
                nested["outer_test_size"], "nested_cv.outer_test_size", open_interval=True
            ),
            outer_splits=_positive_int(nested["outer_splits"], "nested_cv.outer_splits"),
            outer_seed=int(nested["outer_seed"]),
            outer_split_strategy=str(nested["outer_split_strategy"]),
            outer_stratify_dimension=str(nested["outer_stratify_dimension"]),
            outer_constraint_tau_h=_validate_probability(
                nested["outer_constraint_tau_h"], "nested_cv.outer_constraint_tau_h"
            ),
            outer_min_supported_pairs=int(nested["outer_min_supported_pairs"]),
            outer_max_supported_pairs=int(nested["outer_max_supported_pairs"]),
            outer_min_sr_h=outer_min_sr_h,
            outer_max_sr_h=outer_max_sr_h,
            outer_min_spread_fraction=float(nested["outer_min_spread_fraction"]),
            outer_max_candidate_splits=_positive_int(
                nested["outer_max_candidate_splits"], "nested_cv.outer_max_candidate_splits"
            ),
        ),
        outputs=OutputConfig(
            results_root=_require_string(outputs["results_root"], "outputs.results_root"),
            save_artifacts=_require_bool(outputs["save_artifacts"], "outputs.save_artifacts"),
            save_idata=_require_bool(outputs["save_idata"], "outputs.save_idata"),
        ),
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Configuration file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON configuration at {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a JSON object: {path}")
    return raw


def default_config_dict() -> dict[str, Any]:
    resource = files("lm_bmfrm.configs").joinpath("paper_default.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def load_experiment_config(path: str | Path | None = None) -> ExperimentConfig:
    """Load packaged defaults and merge an optional user configuration over them."""
    raw = default_config_dict()
    if path is not None:
        raw = _deep_merge(raw, _read_json(Path(path)))
    return _parse_config(raw)
