"""Pooled LOBO score for prior + per-benchmark label offset (no NCF)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SUBMISSION_DIR = Path(__file__).parent / "submission"
sys.path.insert(0, str(SUBMISSION_DIR))
from prior_model import prior_logit  # noqa: E402

from train_ncf import aggregate, auc, build_tables, load_data, log_loss, render_subject_content  # noqa: E402

SEED = 0
K_LABELS = 5
OFFSET_ALPHA = 5.0


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def main():
    rng = np.random.default_rng(SEED)
    responses, subjects, items = load_data()
    agg = aggregate(responses)
    subject_content = {
        row["subject_id"]: render_subject_content(row) for _, row in subjects.iterrows()
    }
    item_ok = set(items["item_id"])
    agg = agg[agg["item_id"].isin(item_ok)].reset_index(drop=True)

    ys, ps = [], []
    for bench in sorted(agg["benchmark_id"].unique()):
        is_held = (agg["benchmark_id"] == bench).to_numpy()
        tr, te = agg[~is_held].reset_index(drop=True), agg[is_held].reset_index(drop=True)
        tables = build_tables(tr, subject_content)

        cache = {}
        z = np.empty(len(te), dtype=np.float32)
        for i, (sid, b, c) in enumerate(
            zip(te["subject_id"], te["benchmark_id"], te["test_condition"])
        ):
            key = (sid, c)
            if key not in cache:
                cache[key] = prior_logit(tables, b, c, subject_content.get(sid, f"Name: {sid}"))
            z[i] = cache[key]
        y = te["label"].to_numpy(dtype=np.float32)

        sel = rng.choice(len(y), size=min(K_LABELS, len(y)), replace=False)
        p_hat = float(np.clip(sigmoid(z[sel]).mean(), 0.02, 0.98))
        m = float(np.clip((y[sel].sum() + OFFSET_ALPHA * p_hat) / (len(sel) + OFFSET_ALPHA), 0.02, 0.98))
        off = np.clip(math.log(m / (1 - m)) - math.log(p_hat / (1 - p_hat)), -1.5, 1.5)

        p = np.clip(sigmoid(z + off), 0.005, 0.995)
        print(f"{bench:<20} {len(te):>7}  prior+off={log_loss(y, p):.4f}")
        ys.append(y); ps.append(p)

    y, p = np.concatenate(ys), np.concatenate(ps)
    print(f"\npooled prior+off: log_loss={log_loss(y, p):.4f}  auc={auc(y, p):.4f}")


if __name__ == "__main__":
    main()
