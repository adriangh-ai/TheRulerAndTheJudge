# Minimal custom-judge example

This example shows how to evaluate ratings supplied by a new judge without
changing the Stage 1--3 implementation.

The included Parquet file follows the schema in `INPUT_SCHEMA.md`. It is copied
from one released HANNA judge-output file and placed here only to make the input
contract and execution command self-contained.

Run:

```bash
lm-bmfrm-evaluate \
  --config examples/custom_judge/config.json
```

Or from Python:
```python
import json

from lm_bmfrm import run_experiment

result = run_experiment("examples/custom_judge/config.json")
print(json.dumps(result.machine_summary(), indent=2))
```

The reduced draws and chains are intended only to demonstrate installation,
input preparation, execution, and output access. They are not suitable for
scientific inference or comparison with the paper results.

To evaluate another judge, replace custom_judge.parquet with a file following
the same schema and update data.judge_files.