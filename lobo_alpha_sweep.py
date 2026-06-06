"""Joint LOBO sweep: prior shrinkage ALPHA x offset (alpha', clip).

ALPHA=20 was never validated. Folds' tables are built once; priors are
recomputed per ALPHA (cheap). Offsets use 10 random K=5 label draws.
Also reports the known-benchmark item-cold-start split per ALPHA so we
don't improve one regime by wrecking the other.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

SUBMISSION_DIR = Path(__file__).parent / "submission"
sys.path.insert(0, str(SUBMISSION_DIR))
import prior_model  # noqa: E402
from prior_model import prior_logit  # noqa: E402

from train_ncf import aggregate, build_tables, load_data, log_loss, render_subject_content  # noqa: E402

K = 5
N_SEEDS = 10
ALPHAS = [8.0, 20.0, 40.0]
OFFSET_PARAMS = [(0.5, 2.5), (1.0, 2.5), (1.0, 4.0), (2.0, 2.5)]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def offset_from(sel, z, y, a_off, clip):
    p_hat = float(np.clip(sigmoid(z[sel]).mean(), 0.02, 0.98))
    m = float(np.clip((y[sel].sum() + a_off * p_hat) / (len(sel) + a_off), 0.01, 0.99))
    return float(np.clip(math.log(m / (1 - m)) - math.log(p_hat / (1 - p_hat)), -clip, clip))


def main():
    responses, subjects, items = load_data()
    agg = aggregate(responses)
    subject_content = {
        r["subject_id"]: render_subject_content(r) for _, r in subjects.iterrows()
    }

    # --- LOBO folds: tables built once each ---
    benches = sorted(agg["benchmark_id"].unique())
    folds = []
    for bench in benches:
        held = (agg["benchmark_id"] == bench).to_numpy()
        tr, te = agg[~held].reset_index(drop=True), agg[held].reset_index(drop=True)
        folds.append((build_tables(tr, subject_content), te))
        print(f"fold ready: {bench}", flush=True)

    # --- known-benchmark split (10% items per benchmark), tables once ---
    rng = np.random.default_rng(0)
    val_items = set()
    for bench, grp in agg.groupby("benchmark_id"):
        uniq = grp["item_id"].unique()
        val_items.update(rng.choice(uniq, size=max(1, int(len(uniq) * 0.1)), replace=False))
    is_val = agg["item_id"].isin(val_items).to_numpy()
    ktables = build_tables(agg[~is_val].reset_index(drop=True), subject_content)
    kval = agg[is_val].reset_index(drop=True)
    print("known-benchmark fold ready", flush=True)

    def priors(tables, rows):
        cache = {}
        out = np.empty(len(rows))
        for i, (sid, b, c) in enumerate(
            zip(rows["subject_id"], rows["benchmark_id"], rows["test_condition"])
        ):
            key = (sid, b, c)
            if key not in cache:
                cache[key] = prior_logit(tables, b, c, subject_content.get(sid, f"Name: {sid}"))
            out[i] = cache[key]
        return out

    print(f"\n{'ALPHA':>6} {'known LL':>9} | per (a_off, clip): LOBO LL")
    for alpha in ALPHAS:
        prior_model.ALPHA = alpha

        kz = priors(ktables, kval)
        k_ll = log_loss(kval["label"].to_numpy(dtype=np.float64),
                        np.clip(sigmoid(kz), 0.005, 0.995))

        fold_data = []
        for tables, te in folds:
            z = priors(tables, te)
            fold_data.append((z, te["label"].to_numpy(dtype=np.float64)))

        line = f"{alpha:>6} {k_ll:>9.4f} |"
        for a_off, clip in OFFSET_PARAMS:
            ys, ps = [], []
            for z, y in fold_data:
                acc = np.zeros_like(z)
                for s in range(N_SEEDS):
                    sel = np.random.default_rng(s).choice(
                        len(y), size=min(K, len(y)), replace=False)
                    acc += sigmoid(z + offset_from(sel, z, y, a_off, clip))
                ys.append(y)
                ps.append(np.clip(acc / N_SEEDS, 0.005, 0.995))
            ll = log_loss(np.concatenate(ys), np.concatenate(ps))
            line += f"  ({a_off},{clip})={ll:.4f}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
