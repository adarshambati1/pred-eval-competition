"""Build the shipped lookup_tables.json: full-data tables + curated cells.

- Tables from ALL training rows (train_ncf.py builds them from the 90%
  train split for honest validation; the shipped tables should use 100%).
- Inject curated leaderboard cells (curation/curated_cells.json).
- Record which benchmarks the NCF head was trained on, so model.py's
  familiarity gate opens only for those (curated benchmarks have stats
  but no item-level training data; the head must stay closed there).
"""

from __future__ import annotations

import json
from pathlib import Path

from train_ncf import SUBMISSION_DIR, aggregate, build_tables, load_data, render_subject_content


def main():
    responses, subjects, items = load_data()
    agg = aggregate(responses)
    subject_content = {
        r["subject_id"]: render_subject_content(r) for _, r in subjects.iterrows()
    }
    tables = build_tables(agg, subject_content)
    tables["ncf_benchmarks"] = sorted(agg["benchmark_id"].unique())

    n_sb, n_b = 0, 0
    for fname in ["curated_cells.json", "curated_cells_llm.json"]:
        curated_path = Path(__file__).parent / "curation" / fname
        if not curated_path.exists():
            continue
        curated = json.load(open(curated_path))
        for k, v in curated["per_subject_benchmark"].items():
            if k not in tables["per_subject_benchmark"]:
                tables["per_subject_benchmark"][k] = v
                n_sb += 1
        for k, v in curated["per_benchmark"].items():
            if k not in tables["per_benchmark"]:
                tables["per_benchmark"][k] = v
                n_b += 1
    print(f"injected {n_sb} curated (subject, benchmark) cells, {n_b} benchmark cells")

    out = SUBMISSION_DIR / "lookup_tables.json"
    with open(out, "w") as f:
        json.dump(tables, f)
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
