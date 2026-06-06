"""Curate Open LLM Leaderboard raw accuracies into curation/curated_cells_llm.json.

Validated: leaderboard MMLU-PRO Raw vs training mmlupro means for the 25
overlapping subjects gives corr 0.954, mean |diff| 0.023.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

PSEUDO_N = 100

# leaderboard raw column -> benchmark id aliases (raw = plain accuracy in [0,1]).
# Generic aliases ("math", "gpqa") dropped: a name collision with a hidden
# benchmark that has a different difficulty distribution injects WRONG stats.
COLS = {
    "IFEval Raw": ["ifeval"],
    "BBH Raw": ["bbh"],
    "MATH Lvl 5 Raw": ["math_lvl_5"],
    "GPQA Raw": ["gpqa_diamond"],
    "MUSR Raw": ["musr"],
}


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def main():
    p = hf_hub_download(
        "open-llm-leaderboard/contents", "data/train-00000-of-00001.parquet",
        repo_type="dataset",
    )
    lb = pd.read_parquet(p)
    subs = pd.read_parquet(
        hf_hub_download("aims-foundations/measurement-db", "subjects.parquet",
                        repo_type="dataset")
    )
    sub_by_norm = {}
    for _, r in subs.iterrows():
        for cand in {r["display_name"], r["display_name"].split("/")[-1]}:
            sub_by_norm.setdefault(norm(cand), r["display_name"])

    out = {"per_subject_benchmark": {}, "per_benchmark": {}}
    for col, aliases in COLS.items():
        cells, accs = {}, []
        for _, r in lb.iterrows():
            raw = r[col]
            if not isinstance(raw, (int, float)) or pd.isna(raw) or not (0 < raw < 1):
                continue
            for cand in {str(r["fullname"]), str(r["fullname"]).split("/")[-1]}:
                disp = sub_by_norm.get(norm(cand))
                if disp:
                    cells[f"Name: {disp}"] = {"mean": round(float(raw), 6), "count": PSEUDO_N}
                    accs.append(float(raw))
                    break
        for alias in aliases:
            for k, v in cells.items():
                out["per_subject_benchmark"][f"{k}|||{alias}"] = v
            if accs:
                out["per_benchmark"][alias] = {
                    "mean": round(float(np.mean(accs)), 6), "count": PSEUDO_N * 4
                }
        print(f"{col}: {len(cells)} subjects -> {aliases}")

    path = Path(__file__).parent / "curation" / "curated_cells_llm.json"
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"wrote {path} ({len(out['per_subject_benchmark'])} cells)")


if __name__ == "__main__":
    main()
