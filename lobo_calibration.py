"""LOBO sweep of calibration knobs: offset strength, clip, and final clamp.

Evaluated on the unseen-benchmark (LOBO) regime with random K=5 labels,
averaged over 10 seeds, pooled across all 16 benchmarks.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

SUBMISSION_DIR = Path(__file__).parent / "submission"
sys.path.insert(0, str(SUBMISSION_DIR))
from prior_model import prior_logit  # noqa: E402

from train_ncf import aggregate, auc, build_tables, load_data, log_loss, render_subject_content  # noqa: E402

K = 5
N_SEEDS = 10
ALPHAS = [1.0, 2.0, 3.0, 5.0]
CLIPS = [1.5, 2.5, 4.0]
EPS = [0.005, 0.002]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def offset_from(sel, z, y, alpha, clip):
    p_hat = float(np.clip(sigmoid(z[sel]).mean(), 0.02, 0.98))
    m = float(np.clip((y[sel].sum() + alpha * p_hat) / (len(sel) + alpha), 0.01, 0.99))
    return float(np.clip(math.log(m / (1 - m)) - math.log(p_hat / (1 - p_hat)), -clip, clip))


def main():
    responses, subjects, items = load_data()
    agg = aggregate(responses)
    subject_content = {
        r["subject_id"]: render_subject_content(r) for _, r in subjects.iterrows()
    }

    folds = []  # (z, y) per benchmark
    for bench in sorted(agg["benchmark_id"].unique()):
        held = (agg["benchmark_id"] == bench).to_numpy()
        tr, te = agg[~held].reset_index(drop=True), agg[held].reset_index(drop=True)
        tables = build_tables(tr, subject_content)
        cache = {}
        z = np.empty(len(te))
        for i, (sid, b, c) in enumerate(
            zip(te["subject_id"], te["benchmark_id"], te["test_condition"])
        ):
            key = (sid, c)
            if key not in cache:
                cache[key] = prior_logit(tables, b, c, subject_content.get(sid, f"Name: {sid}"))
            z[i] = cache[key]
        folds.append((z, te["label"].to_numpy(dtype=np.float64)))
        print(f"fold ready: {bench}", flush=True)

    print(f"\n{'alpha':>5} {'clip':>5} {'eps':>6} {'log_loss':>9} {'auc':>7}")
    for alpha in ALPHAS:
        for clip in CLIPS:
            for eps in EPS:
                ys, ps = [], []
                for z, y in folds:
                    p_acc = np.zeros_like(z)
                    for s in range(N_SEEDS):
                        sel = np.random.default_rng(s).choice(
                            len(y), size=min(K, len(y)), replace=False)
                        off = offset_from(sel, z, y, alpha, clip)
                        p_acc += sigmoid(z + off)
                    ys.append(y)
                    ps.append(np.clip(p_acc / N_SEEDS, eps, 1 - eps))
                yy, pp = np.concatenate(ys), np.concatenate(ps)
                print(f"{alpha:>5} {clip:>5} {eps:>6} {log_loss(yy, pp):>9.4f} "
                      f"{auc(yy, pp):>7.4f}", flush=True)


if __name__ == "__main__":
    main()
