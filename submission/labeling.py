import hashlib
import json
from pathlib import Path

from prior_model import normalize_condition

_DIR = Path(__file__).parent

with open(_DIR / "lookup_tables.json") as f:
    _TABLES = json.load(f)


def _pseudo_random(input):
    key = json.dumps(
        {k: input.get(k, "") for k in
         ("benchmark", "condition", "subject_content", "item_content")},
        sort_keys=True,
    )
    h = hashlib.md5(key.encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def acquisition_function(input):
    benchmark = input.get("benchmark", "")
    condition = normalize_condition(input.get("condition", ""))

    bench_stat = _TABLES["per_benchmark"].get(benchmark)
    bc_stat = _TABLES["per_benchmark_condition"].get(f"{benchmark}|||{condition}")
    if not bench_stat:
        novelty = 2.0
    elif not bc_stat or bc_stat["count"] < 100:
        novelty = 1.0
    else:
        novelty = 0.0

    return float(novelty + _pseudo_random(input))
