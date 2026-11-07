# The Ruler and the Judge

**Benchmark-Conditional Evaluation of LLM-as-a-Judge**

This repository contains the implementation and reproducibility artifacts for the paper *The Ruler and the Judge: Benchmark-Conditional Evaluation of LLM-as-a-Judge*.

LLM judges are commonly validated using a single aggregate agreement statistic, such as correlation with human ratings. This project instead treats judge validation as a benchmark-conditional measurement problem:

1. **Stage 1 — Human-ruler audit:** determine whether the human benchmark supports the intended comparison.
2. **Stage 2 — Judge calibration:** determine whether an LLM judge behaves coherently as a linked measurement instrument.
3. **Stage 3 — Held-out evaluation:** determine whether the calibrated judge recovers human-linked measurements for unseen systems.

The resulting protocol can select a judge, warn that only restricted claims are supported, or refuse judge selection when the benchmark or available judge pool cannot sustain the intended comparison.

## Repository status

This is a frozen research artifact prepared for anonymous review. It contains:

- the complete Stage 1, Stage 2, and Stage 3 implementation;
- exact experiment configurations, seeds, and resolved run configurations;
- benchmark-specific judge outputs used in the experiments;
- machine-readable JSON, CSV, Parquet, NetCDF, and NumPy artifacts;
- notebooks reproducing the paper tables and figures;
- the additional HANNA analysis;
- the complete threshold-sensitivity and rebuttal analyses;
- documented input and output contracts;
- a configuration-driven command-line interface;
- a minimal programmatic interface and automation guidance.

The historical evaluation command remains the authoritative reproduction path. The public Python interface delegates to the same orchestration code and does not introduce an alternative numerical implementation.

---

## Pairwise measurement conventions

### Tie-symmetric discrete-grid dominance

Posterior system locations in Stage 3 are represented on a shared finite grid. When two independent posterior draws occupy the same grid point, strict dominance alone leaves positive same-grid mass unallocated.

All support-based quantities use the tie-symmetric mid-dominance convention:

```math
P_{\mathrm{mid}}(A>B)
=
P(A>B)
+
\frac{1}{2}P(A=B).
```

Equivalently, half of the same-grid posterior mass is allocated to each direction. This guarantees the expected symmetry

```math
P_{\mathrm{mid}}(A>B) + P_{\mathrm{mid}}(B>A) = 1
```

even when the two discrete posteriors assign positive mass to identical grid locations.

This convention applies to posterior-dominance quantities used for:

- supported resolution;
- human-supported and human-indistinguishable pair classification;
- judge-supported pair classification;
- false separation rate;
- unsupported separation rate;
- support precision and recall;
- support confusion counts;
- other support-based decision summaries.

Normalized signed-margin fidelity (nSMF) uses posterior-predictive pairwise probabilities,

```math
g_{ab}=\mathbb{E}_{\theta_a,\theta_b}
\left[\sigma(\theta_a-\theta_b)\right],
```

rather than finite-sample posterior-dominance probabilities.

### False Separation Rate

False Separation Rate (FSR) is calculated only over pairs that the human ruler does not distinguish at the declared human-support threshold.

Let

```math
\omega^H_{ab}=\left|2p^H_{ab}-1\right|,
\qquad
\omega^J_{ab}=\left|2p^J_{ab}-1\right|,
```

where $p^H_{ab}$ and $p^J_{ab}$ use tie-symmetric mid-dominance.

The human-indistinguishable set is

```math
\mathcal{U}_H(\tau_H)
=
\left\{(a,b):\omega^H_{ab}<\tau_H\right\}.
```

A false separation occurs only when a pair belongs to this set and the judge exceeds its own separation threshold:

```math
\omega^J_{ab}\geq\tau_L.
```

Therefore,

```math
\mathrm{FSR}
=
\frac{
\left|
\left\{
(a,b)\in\mathcal{U}_H(\tau_H):
\omega^J_{ab}\geq\tau_L
\right\}
\right|
}{
|\mathcal{U}_H(\tau_H)|
}.
```

The denominator is the number of **human-indistinguishable pairs**, not the number of all system pairs and not the number of judge-separated pairs. When the human-indistinguishable set is empty, FSR is undefined rather than artificially set by an unrelated denominator.

The notebooks and released support-based tables use this convention.

---

## Method overview

### Stage 1: auditing the human ruler

For each dataset dimension, the human ratings are modeled with a many-facet ordered Rasch model:

```math
y^H_{sij}
\sim
\mathrm{OrderedLogistic}
\left(
\theta^H_s-\beta^H_i-\rho^H_j,
c^H
\right),
```

where:

- $\theta^H_s$ is the latent location of system \(s\);
- $\beta^H_i$ is item difficulty;
- $\rho^H_j$ is human-rater severity;
- $c^H$ contains ordered human response thresholds.

Stage 1 reports:

- human-rater infit and outfit MNSQ;
- human-item infit and outfit MNSQ;
- benchmark information and conditional standard error of measurement;
- occupied system spread;
- approximate pairwise distinguishability;
- supported resolution at the declared human threshold.

These diagnostics determine whether a dimension supports fine-grained selection, only coarse or warned use, or refusal.

### Stage 2: calibrating judges as linked instruments

Each judge receives judge-specific ordered response thresholds linked to the Stage 1 human ruler. The main model marginalizes over selected posterior draws of the human anchor, propagating uncertainty in the human-linked system-item predictor.

Stage 2 reports:

- judge-level infit and outfit MNSQ;
- judge-by-item diagnostics;
- judge threshold geometry;
- optional judge-by-item differential item functioning diagnostics;
- sampling diagnostics.

Stage 2 evaluates whether a judge behaves coherently as a measurement instrument. It is distinct from held-out predictive performance.

### Stage 3: held-out system measurement

The unit of generalization is the evaluated system. In leave-one-system-out cross-validation, Stage 1 and Stage 2 are refit using all remaining systems. The held-out system then receives:

- a judge-based posterior from its judge ratings;
- a human-reference posterior from its human ratings.

Fold-specific rulers are affinely linked to the full-data human ruler before aggregating held-out results.

Stage 3 reports:

- latent-system Spearman correlation;
- raw-score Spearman correlation where available;
- normalized signed-margin fidelity (nSMF);
- magnitude and directional loss components;
- support-aware metrics including FSR;
- empirical transfer intervals and coverage;
- ordinal Earth Mover's Distance for human-score-distribution prediction.

---

## Repository layout

```text
.
├── configs/                    Experiment configuration templates
├── llm_judge_outputs/          Released judge-rating files by benchmark
├── notebooks/                  Paper, HANNA, rebuttal, and sensitivity analyses
├── results/                    Rebuttal-era result directories and derived tables
├── scripts/                    CLI wrappers and verification utilities
├── src/lm_bmfrm/               Installable implementation
├── tests/                      Regression and contract tests
├── AGENTS.md                   Operational guidance for automated callers
├── PROGRAMMATIC_USAGE.md       Python quickstart
├── INPUT_SCHEMA.md             Judge-input contract
├── OUTPUT_SCHEMA.md            Result-artifact map
└── pyproject.toml              Package and CLI definition
```

---

## Installation

Python 3.11 is recommended for reproducing the released environment.

```bash
git clone <ANONYMIZED_REPOSITORY_URL>
cd TheRulerAndTheJudge

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Verify the installation:

```bash
pytest -q
lm-bmfrm-evaluate --help
```

### JAX backend

The Bayesian models use PyMC with the NumPyro/JAX NUTS backend. A compatible GPU installation is strongly recommended for the full experiments. CPU execution is possible but substantially slower.

The reported experiments used:

- PyTensor `floatX=float32`;
- PyTensor `mode=JAX`;
- vectorized NumPyro-backed NUTS chains;
- an NVIDIA RTX 5090 GPU.

Exact package and runtime versions for each released experiment are stored in its `run_config.json`.

---

## Command-line usage

The installed command-line entry point is:

```bash
lm-bmfrm-evaluate --help
```

The script wrapper invokes the same implementation:

```bash
python scripts/run_evaluation.py --help
```

Run an experiment from a JSON configuration:

```bash
lm-bmfrm-evaluate \
  --config configs/paper_default.json
```

The configuration controls benchmark selection, judge files and their order, dimensions, sampling settings, seeds, thresholds, cross-validation, and output behavior.

### Reproduce a released experiment

Every released result directory contains:

- `resolved_config.json`: the complete effective experiment configuration;
- `run_config.json`: command, package versions, Git state, and runtime metadata;
- `judge_input_manifest.parquet`: ordered inputs and file hashes.

The closest reproduction entry point is:

```bash
lm-bmfrm-evaluate \
  --config results/<EXPERIMENT_DIRECTORY>/resolved_config.json
```

Run from the repository root so relative input paths resolve consistently. When relocating the repository, update only filesystem paths that necessarily changed; preserve judge-file order, dimensions, seeds, thresholds, model settings, and sampling settings.

Examples of released experiment directories include:

```text
results/summeval_MC_per_judge_linspace_20260726_002902/
results/hitz_basse_MC_per_judge_linspace_20260726_130449/
results/hanna_MC_per_judge_linspace_20260726_200116/
results/summeval_synth_MC_per_judge_linspace_20260726_170249/
```

The original submission-era outputs are retained separately under `results_submission/`.

---

## Programmatic and agent-driven use

The package exposes a minimal public interface:

```python
from lm_bmfrm import load_experiment, run_experiment
```

Run the established CLI pipeline from Python:

```python
import json

from lm_bmfrm import run_experiment

result = run_experiment(
    "configs/paper_default.json",
    results_root="results/programmatic",
)

print(json.dumps(result.machine_summary(), indent=2))
```

Read an existing run without refitting:

```python
from lm_bmfrm import load_experiment

result = load_experiment(
    "results/hanna_MC_per_judge_linspace_20260726_200116"
)

leaderboard = result.leaderboard
predictions = result.predictions
human_audit = result.read_table("human_audit")
llm_audit = result.read_table("llm_audit")
```

The Python interface delegates to the same `pipeline.cli.main` orchestration function used by `lm-bmfrm-evaluate`. It does not reimplement or reorder preprocessing, fitting, cross-validation, metrics, or serialization.

Automated callers should consume the JSON, CSV, and Parquet artifacts rather than parse progress bars or formatted terminal tables. See:

- [`AGENTS.md`](AGENTS.md)
- [`PROGRAMMATIC_USAGE.md`](PROGRAMMATIC_USAGE.md)
- [`INPUT_SCHEMA.md`](INPUT_SCHEMA.md)
- [`OUTPUT_SCHEMA.md`](OUTPUT_SCHEMA.md)

---

## Judge-input schema

Judge ratings are supplied as Parquet files. The required columns are:

| Column | Description |
|---|---|
| `item_id` | Identifier matching a benchmark item |
| `system_id` | Evaluated system or generation configuration |
| `rater_id` | Judge/model identifier |
| `dimension` | Rubric dimension |
| `score` | Ordinal score in `{1, 2, 3, 4, 5}` |

Optional fields include:

- `score_list`: repeated ordinal ratings for an empirical soft-label target;
- `model_output`: raw judge response;
- `prompt`: prompt sent to the judge;
- `is_llm`: accepted but normalized by the evaluator.

For reproducibility, judge files should be explicitly listed in the configuration. Their order and hashes are recorded in `judge_input_manifest.parquet`.

A minimal custom-judge example is provided under:

```text
examples/custom_judge/
```

Its reduced sampling settings demonstrate installation, validation, execution, and output access; they are not intended for scientific inference.

---

## Output artifacts

Each run creates a timestamped result directory.

### Reproducibility and provenance

| Artifact | Purpose |
|---|---|
| `resolved_config.json` | Complete effective configuration and seeds |
| `run_config.json` | Environment, package versions, Git state, and command |
| `judge_input_manifest.parquet` | Ordered judge files, hashes, sizes, and row counts |
| `judge_coverage.csv` | Coverage against the expected benchmark grid |
| `judge_exclusions.csv` | Excluded files or rows |

### Stage 1

| Artifact | Purpose |
|---|---|
| `human_audit.parquet` | Human-rater fit diagnostics |
| `human_item_audit.parquet` | Human-item fit diagnostics |
| `benchmark_cards.parquet` | Benchmark-resolution summaries |
| `benchmark_curves.parquet` | Information and precision curves |
| `stage1_params_<dimension>.json` | Stage 1 posterior summaries |

### Stage 2

| Artifact | Purpose |
|---|---|
| `llm_audit.parquet` | Judge-level fit diagnostics |
| `llm_item_audit.parquet` | Judge-by-item diagnostics |
| `Stage_2_dif_audit/` | Judge-by-item DIF summaries |
| `stage2_idata_<dimension>__<judge>.nc` | Optional full Stage 2 posterior |

### Stage 3

| Artifact | Purpose |
|---|---|
| `predictions.parquet` | Held-out system estimates and residuals |
| `leaderboard_results.parquet` | Rank, nSMF, support, interval, and EMD metrics |
| `q_raw_loocv.json` | Raw empirical transfer quantiles |
| `q_norm_loocv.json` | Normalized empirical transfer quantiles |
| `rhat_records.parquet` | Sampling diagnostics |
| `loocv_posterior_artifacts.npz` | Persisted fold/posterior arrays |
| `loocv_posterior_artifacts_manifest.parquet` | Artifact coordinates and metadata |

The primary downstream comparison table is `leaderboard_results.parquet`.

---

## Analysis notebooks

The notebooks reproduce the reported analysis layer from released machine-readable artifacts. They do not replace the Stage 1–3 implementation.

The repository includes notebooks for:

- the original paper tables and figures;
- the synthetic-judge stress test;
- the additional HANNA evaluation;
- the rebuttal supporting tables;
- strict-versus-mid-dominance comparisons;
- complete threshold sensitivity;
- Stage 2 MNSQ sensitivity;
- operational decision summaries.

Generated CSV, Parquet, and LaTeX tables are stored alongside the notebook outputs and under:

```text
results/_rebuttal_supporting_tables/
notebooks/appendix_tables/
```

Execute notebooks from the repository root so their relative paths resolve correctly.

---

## Datasets and judge outputs

The main paper evaluates:

- **SummEval:** 17 summarization systems, 100 items, three expert raters, and four dimensions;
- **HiTZ-BASSE:** 24 evaluated generation configurations, 45 items, three raters, and five dimensions.

The additional analysis evaluates:

- **HANNA:** story-generation systems across relevance, coherence, empathy, surprise, engagement, and complexity.

Nine judge configurations spanning GPT-OSS, DeepSeek, Qwen, Llama, and Gemma families are evaluated independently for each benchmark dimension.

Judge output files used in the released experiments are under `llm_judge_outputs/`. Original dataset ownership and licensing remain with the respective dataset authors. Users are responsible for complying with the terms of the underlying datasets and model providers.

---

## Reproducibility and validation

Run the test suite:

```bash
pytest -q
```

Important validation utilities include:

```bash
python scripts/verify_source_equivalence.py
python scripts/compare_results.py --help
python scripts/capture_environment.py
```

The repository separates several forms of validation:

- unit and regression tests for preprocessing and metrics;
- source-structure contracts for result-sensitive functions;
- run provenance and input hashing;
- comparison of complete output directories;
- notebook reconstruction of reported tables and figures.

`numerical_contract.json` is a structural freeze of selected function definitions. It detects source changes; it is not by itself a numerical-output test. Full experiment comparisons should use `scripts/compare_results.py` together with the released result directories.

Because JAX/XLA GPU sampling may not be bitwise identical across separate process executions, scientific comparisons should focus on posterior summaries, reported metrics, and decision stability rather than expecting raw posterior draw arrays to match element by element.

---

## Main operational thresholds

Unless a released configuration states otherwise, the paper uses:

- human support threshold: \(\tau_H=0.8\);
- judge support threshold: \(\tau_L=0.8\);
- Stage 2 productive MNSQ range: `[0.5, 1.5]`;
- illustrative risk-sensitive FSR tolerance: `0.30`;
- empirical transfer interval level: `1 - alpha = 0.95`.

These are transparent operational decision rules, not universal validity constants. The released sensitivity analysis evaluates how conclusions change under alternative thresholds.

---

## Computational requirements

The full pipeline is computationally expensive because Stage 3 refits Stage 1 and Stage 2 within each held-out-system fold.

The paper reports approximately:

- 20 GPU-minutes for one SummEval judge-dimension LOOCV run;
- 35 GPU-hours for the final main experiment suite;
- execution on one NVIDIA RTX 5090.

Runtime depends strongly on the number of systems, dimensions, judges, chains, draws, and whether optional posterior artifacts are stored.

The minimal example intentionally uses reduced sampling settings and must not be used as evidence for scientific conclusions.

---

## Scope and limitations

This implementation validates judges relative to a specific benchmark, dimension, annotation design, and downstream claim. It does not produce a universal judge ranking.

The method requires disaggregated human ratings with system, item, rater, dimension, and ordinal-score identifiers. Aggregate-only benchmarks cannot support the full ruler audit or posterior support analysis.

Each rubric dimension is modeled independently and is assumed to be approximately unidimensional. Small system counts may make cross-validation and judge differentiation unstable. These limitations are part of the motivation for auditing benchmark resolution before reporting a judge-based leaderboard.

---

## Citation

This repository accompanies the anonymous NeurIPS 2026 submission:

```bibtex
@inproceedings{anonymous2026the,
  title     = {The Ruler and the Judge: Benchmark-Conditional Evaluation of {LLM}-as-a-Judge},
  author    = {Anonymous},
  booktitle = {The Fortieth Annual Conference on Neural Information Processing Systems Evaluations and Datasets Track},
  year      = {2026},
  url       = {https://openreview.net/forum?id=NmRNfA31p3}
}
```

Replace the anonymous citation with the final bibliographic record after de-anonymization.

---

## License

See [`LICENSE`](LICENSE). Dataset annotations, model outputs, and third-party dependencies may be governed by separate licenses or terms of use. Consult the corresponding dataset and model documentation before redistribution or downstream use.
