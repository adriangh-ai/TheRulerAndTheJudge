#!/usr/bin/env python
"""Compare the refactor against active pre-refactor computational definitions.

The verifier permits only the explicitly documented changes below:
- the mid-dominance tie correction;
- configuration/seed parameterization and input-validation plumbing.
All other active pre-refactor functions must remain AST-equivalent after
comments and docstrings are removed.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import hashlib
from pathlib import Path


CORE_FILES = [
    "cutpoints.py",
    "data_preprocess.py",
    "estimate.py",
    "evaluation_metrics.py",
    "fit_analysis.py",
    "irt_model.py",
    "irt_utils.py",
    "prompt_templates.py",
    "utils.py",
]
ORIGINAL_PIPELINE = "llm_judge_eval_pipeline_nested_contrained_v2_full_log(1).py"

INTENTIONAL_CORE_DIFFERENCES = {
    ("evaluation_metrics.py", "pairwise_dominance_probability"),
}
INTENTIONAL_PIPELINE_DIFFERENCES = {
    "build_judge_coverage_table",
    "fit_fold_models_per_dim",
    "fit_human_anchor_for_dim",
    "fit_single_judge_for_dim",
    "run_benchmark_audit",
    "run_judge_benchmark",
    "run_nested_system_cv_conformal",
    "run_system_cv_conformal",
    "safe_mode",
}


class StripDocstrings(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body = node.body[1:]
        return node


def function_hashes(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    output = defaultdict(list)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            node = StripDocstrings().visit(node)
            ast.fix_missing_locations(node)
            payload = ast.dump(node, annotate_fields=True, include_attributes=False)
            output[node.name].append(hashlib.sha256(payload.encode()).hexdigest())
    return output


def active_hashes(path: Path):
    return {name: versions[-1] for name, versions in function_hashes(path).items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("original_source_dir", type=Path)
    parser.add_argument("--refactored-root", type=Path, default=Path(__file__).parents[1])
    args = parser.parse_args()

    package = args.refactored_root / "src" / "lm_bmfrm"
    failures = []

    for filename in CORE_FILES:
        before = active_hashes(args.original_source_dir / filename)
        after = active_hashes(package / filename)
        for name, before_hash in sorted(before.items()):
            if (filename, name) in INTENTIONAL_CORE_DIFFERENCES:
                continue
            if after.get(name) != before_hash:
                failures.append(f"{filename}:{name}")

    before_pipeline = active_hashes(args.original_source_dir / ORIGINAL_PIPELINE)
    after_pipeline = {}
    for path in (package / "pipeline").glob("*.py"):
        after_pipeline.update(active_hashes(path))

    before_pipeline.pop("main", None)
    for name, before_hash in sorted(before_pipeline.items()):
        if name in INTENTIONAL_PIPELINE_DIFFERENCES:
            continue
        if after_pipeline.get(name) != before_hash:
            failures.append(f"pipeline:{name}")

    if failures:
        print("Unexpected computational source differences detected:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)

    print("PASS: all non-whitelisted active computational functions match the original sources")
    print("Documented intentional differences:")
    for filename, name in sorted(INTENTIONAL_CORE_DIFFERENCES):
        print(f"  - {filename}:{name} (mid-dominance tie correction)")
    for name in sorted(INTENTIONAL_PIPELINE_DIFFERENCES):
        print(f"  - pipeline:{name} (configuration/validation plumbing)")


if __name__ == "__main__":
    main()
