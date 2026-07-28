from __future__ import annotations
import ast
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parents[1] / "src" / "lm_bmfrm"

def test_no_shadowed_top_level_definitions():
    offenders = {}
    for path in ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
        if duplicates:
            offenders[str(path.relative_to(ROOT))] = duplicates
    assert not offenders, offenders
