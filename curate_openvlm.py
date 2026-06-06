"""Curate OpenVLM leaderboard aggregates into lookup-table cells.

Validation mode (default): LOBO on the three VLM benchmarks shared between
the training corpus and the leaderboard (ai2d_test, mathvista_mini,
mmbench_v11). For each, build tables WITHOUT it, inject the curated
leaderboard cells for it, and measure how much of the gap to the
fully-trained prior the curation recovers, at several pseudo-counts,
with and without the 5-label offset.

Build mode (--build): emit curation cells for the ~20 leaderboard
benchmarks NOT in training, keyed under plausible hidden-pool aliases,
into curation/curated_cells.json for train_ncf.py / table injection.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

SUBMISSION_DIR = Path(__file__).parent / "submission"
sys.path.insert(0, str(SUBMISSION_DIR))
from prior_model import prior_logit  # noqa: E402

from train_ncf import aggregate, auc, build_tables, load_data, log_loss, render_subject_content  # noqa: E402

K = 5
ALPHA_OFF = 5.0
PSEUDO_COUNTS = [25, 100, 400]

# leaderboard column -> (shared training benchmark id) for validation
SHARED = {
    "AI2D": "ai2d_test",
    "MathVista": "mathvista_mini",
    "MMBench_TEST_EN_V11": "mmbench_v11",
}

# leaderboard column -> list of plausible hidden benchmark ids (lowercased
# VLMEvalKit conventions), for build mode. Accuracy-scaled columns only.
NEW_BENCHMARKS = {
    "MMStar": ["mmstar"],
    "MMVet": ["mmvet", "mm-vet"],
    "MMMU_VAL": ["mmmu_val", "mmmu_dev_val", "mmmu"],
    "HallusionBench": ["hallusionbench"],
    "RealWorldQA": ["realworldqa"],
    "SEEDBench_IMG": ["seedbench_img", "seedbench"],
    "SEEDBench2_Plus": ["seedbench2_plus"],
    "POPE": ["pope"],
    "BLINK": ["blink"],
    "QBench": ["qbench"],
    "ABench": ["abench"],
    "MTVQA": ["mtvqa"],
    "ScienceQA_TEST": ["scienceqa_test", "scienceqa"],
    "CCBench": ["ccbench"],
    "MMT-Bench_VAL": ["mmt-bench_val", "mmt-bench"],
    "MMBench_TEST_CN_V11": ["mmbench_cn_v11"],
    "MMBench_TEST_CN": ["mmbench_cn"],
    "ScienceQA_VAL": ["scienceqa_val"],
    "MME": [],            # excluded: 0-2800 scale, not accuracy
    "OCRBench": ["ocrbench"],  # special: counts out of 1000 -> /10
    "LLaVABench": [],     # excluded: GPT-judged relative score
}


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def overall(row, col):
    v = row.get(col)
    if col == "OCRBench" and isinstance(v, dict):
        total = sum(x for x in v.values() if isinstance(x, (int, float)))
        return total / 10.0 if total else None
    if isinstance(v, dict):
        o = v.get("Overall")
        return float(o) if isinstance(o, (int, float)) else None
    return float(v) if isinstance(v, (int, float)) else None


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def offset_from(sel, z, y):
    p_hat = float(np.clip(sigmoid(z[sel]).mean(), 0.02, 0.98))
    m = float(np.clip((y[sel].sum() + ALPHA_OFF * p_hat) / (len(sel) + ALPHA_OFF), 0.02, 0.98))
    return float(np.clip(math.log(m / (1 - m)) - math.log(p_hat / (1 - p_hat)), -1.5, 1.5))


def load_leaderboard_and_match():
    lb = json.load(open(Path(__file__).parent / "curation" / "OpenVLM.json"))["results"]
    subs = pd.read_parquet(
        hf_hub_download("aims-foundations/measurement-db", "subjects.parquet", repo_type="dataset")
    )
    by_norm = {}
    for _, r in subs.iterrows():
        by_norm.setdefault(norm(r["display_name"]), []).append(r["display_name"])
    matches = {}  # leaderboard model -> registry display_name
    for m in lb:
        hits = by_norm.get(norm(m))
        if hits:
            matches[m] = hits[0]
    return lb, matches


def curated_cells(lb, matches, col, pseudo_n):
    """{name_line: {mean, count}} for one leaderboard column."""
    cells = {}
    accs = []
    for m, disp in matches.items():
        acc = overall(lb[m], col)
        if acc is not None and 0 < acc < 100:
            cells[f"Name: {disp}"] = {"mean": acc / 100.0, "count": pseudo_n}
            accs.append(acc / 100.0)
    bench_mean = float(np.mean(accs)) if accs else None
    return cells, bench_mean


def validate():
    lb, matches = load_leaderboard_and_match()
    print(f"matched {len(matches)} leaderboard models to registry subjects")

    responses, subjects, items = load_data()
    agg = aggregate(responses)
    subject_content = {
        r["subject_id"]: render_subject_content(r) for _, r in subjects.iterrows()
    }

    for col, bench in SHARED.items():
        held = (agg["benchmark_id"] == bench).to_numpy()
        tr, te = agg[~held].reset_index(drop=True), agg[held].reset_index(drop=True)
        tables = build_tables(tr, subject_content)
        y = te["label"].to_numpy(dtype=np.float64)

        def priors(tbl):
            cache = {}
            out = np.empty(len(te))
            for i, (sid, b, c) in enumerate(
                zip(te["subject_id"], te["benchmark_id"], te["test_condition"])
            ):
                key = (sid, c)
                if key not in cache:
                    cache[key] = prior_logit(tbl, b, c, subject_content.get(sid, f"Name: {sid}"))
                out[i] = cache[key]
            return out

        z0 = priors(tables)
        rng = np.random.default_rng(0)
        sel = rng.choice(len(y), size=min(K, len(y)), replace=False)
        print(f"\n{bench} ({len(te)} rows)")
        print(f"  no curation        : prior={log_loss(y, sigmoid(z0)):.4f}  "
              f"+off={log_loss(y, sigmoid(z0 + offset_from(sel, z0, y))):.4f}")

        for pn in PSEUDO_COUNTS:
            cells, bench_mean = curated_cells(lb, matches, col, pn)
            tbl = json.loads(json.dumps(tables))  # deep copy
            for k, v in cells.items():
                tbl["per_subject_benchmark"][f"{k}|||{bench}"] = v
            if bench_mean is not None:
                tbl["per_benchmark"][bench] = {"mean": bench_mean, "count": pn * 4}
            z = priors(tbl)
            off = offset_from(sel, z, y)
            print(f"  curated n={pn:<4}     : prior={log_loss(y, sigmoid(z)):.4f}  "
                  f"+off={log_loss(y, sigmoid(z + off)):.4f}  "
                  f"auc={auc(y, sigmoid(z)):.4f}  ({len(cells)} subject cells)")


def build():
    lb, matches = load_leaderboard_and_match()
    out = {"per_subject_benchmark": {}, "per_benchmark": {}}
    pn = 100
    for col, aliases in NEW_BENCHMARKS.items():
        if not aliases:
            continue
        cells, bench_mean = curated_cells(lb, matches, col, pn)
        if not cells:
            continue
        for alias in aliases:
            for k, v in cells.items():
                out["per_subject_benchmark"][f"{k}|||{alias}"] = v
            if bench_mean is not None:
                out["per_benchmark"][alias] = {"mean": bench_mean, "count": pn * 4}
        print(f"{col}: {len(cells)} subjects -> aliases {aliases}")
    path = Path(__file__).parent / "curation" / "curated_cells.json"
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"\nwrote {path} ({path.stat().st_size/1e3:.0f} KB, "
          f"{len(out['per_subject_benchmark'])} subject-benchmark cells)")


if __name__ == "__main__":
    if "--build" in sys.argv:
        build()
    else:
        validate()
