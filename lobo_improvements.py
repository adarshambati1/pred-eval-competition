"""LOBO test of two proposed improvements for unseen benchmarks.

1. Label-selection policy for the per-benchmark offset:
   random vs max-entropy (deployed labeling.py) vs representative
   (pred closest to the benchmark's mean pred).
2. Similarity-blended item-side prior: instead of shrinking an unknown
   benchmark to the GLOBAL mean, blend known benchmarks' difficulty by
   cosine similarity between the item embedding and each known
   benchmark's mean item embedding (softmax with temperature tau).

All variants evaluated per held-out benchmark, pooled. Random policy
averaged over 10 seeds.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SUBMISSION_DIR = Path(__file__).parent / "submission"
sys.path.insert(0, str(SUBMISSION_DIR))
from prior_model import prior_logit, _shrink  # noqa: E402

from train_ncf import (  # noqa: E402
    aggregate,
    auc,
    build_tables,
    encode_cached,
    load_data,
    log_loss,
    render_subject_content,
)

K = 5
ALPHA_OFF = 5.0
TAUS = [0.05, 0.1, 0.2]
N_SEEDS = 10


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def offset_from(sel, z, y):
    p_hat = float(np.clip(sigmoid(z[sel]).mean(), 0.02, 0.98))
    m = float(np.clip((y[sel].sum() + ALPHA_OFF * p_hat) / (len(sel) + ALPHA_OFF), 0.02, 0.98))
    return float(np.clip(math.log(m / (1 - m)) - math.log(p_hat / (1 - p_hat)), -1.5, 1.5))


def main():
    device = "mps"
    responses, subjects, items = load_data()
    agg = aggregate(responses)
    subject_content = {
        r["subject_id"]: render_subject_content(r) for _, r in subjects.iterrows()
    }
    item_content = dict(zip(items["item_id"], items["content"].fillna("")))
    agg = agg[agg["item_id"].isin(item_content.keys())].reset_index(drop=True)

    uniq_items = sorted(agg["item_id"].unique())
    item_emb = encode_cached([item_content[i] for i in uniq_items], "items", device)
    item_idx = {iid: k for k, iid in enumerate(uniq_items)}
    item_bench = dict(zip(agg["item_id"], agg["benchmark_id"]))

    # mean (normalized) item embedding per benchmark
    bench_ids = sorted(agg["benchmark_id"].unique())
    centroids = {}
    for b in bench_ids:
        rows = [item_idx[i] for i in set(agg.loc[agg.benchmark_id == b, "item_id"])]
        c = item_emb[rows].mean(axis=0)
        centroids[b] = c / np.linalg.norm(c)

    results = {}  # name -> list of (y, p) arrays

    def add(name, y, p):
        results.setdefault(name, ([], []))
        results[name][0].append(y)
        results[name][1].append(np.clip(p, 0.005, 0.995))

    for bench in bench_ids:
        held = (agg["benchmark_id"] == bench).to_numpy()
        tr, te = agg[~held].reset_index(drop=True), agg[held].reset_index(drop=True)
        tables = build_tables(tr, subject_content)
        g = tables["global_mean"]

        cache = {}
        z = np.empty(len(te), dtype=np.float64)
        for i, (sid, b, c) in enumerate(
            zip(te["subject_id"], te["benchmark_id"], te["test_condition"])
        ):
            key = (sid, c)
            if key not in cache:
                cache[key] = prior_logit(tables, b, c, subject_content.get(sid, f"Name: {sid}"))
            z[i] = cache[key]
        y = te["label"].to_numpy(dtype=np.float64)
        add("prior", y, sigmoid(z))

        # --- blended item-side prior ---
        others = [b for b in bench_ids if b != bench]
        cent = np.stack([centroids[b] for b in others])           # (15, d)
        p_bench = np.array([
            _shrink(tables["per_benchmark"].get(b), g) for b in others
        ])
        te_emb = item_emb[te["item_id"].map(item_idx).to_numpy()]
        te_emb = te_emb / np.linalg.norm(te_emb, axis=1, keepdims=True)
        sims = te_emb @ cent.T                                    # (n, 15)
        z_blend = {}
        for tau in TAUS:
            w = np.exp(sims / tau)
            w /= w.sum(axis=1, keepdims=True)
            p_bl = sigmoid(w @ logit(p_bench))
            zb = z + logit(p_bl) - logit(np.full_like(p_bl, g))
            z_blend[tau] = zb
            add(f"blend t={tau}", y, sigmoid(zb))

        # --- label-selection policies for the offset ---
        p = sigmoid(z)
        ent_sel = np.argsort(np.abs(p - 0.5))[:K]                 # max entropy
        rep_sel = np.argsort(np.abs(p - p.mean()))[:K]            # representative
        add("off entropy", y, sigmoid(z + offset_from(ent_sel, z, y)))
        add("off represent", y, sigmoid(z + offset_from(rep_sel, z, y)))
        rnd_p = np.zeros_like(p)
        for s in range(N_SEEDS):
            sel = np.random.default_rng(s).choice(len(y), size=min(K, len(y)), replace=False)
            rnd_p += sigmoid(z + offset_from(sel, z, y))
        add("off random", y, rnd_p / N_SEEDS)

        # --- best blend + representative offset (the candidate ship) ---
        for tau in TAUS:
            zb = z_blend[tau]
            pb = sigmoid(zb)
            sel = np.argsort(np.abs(pb - pb.mean()))[:K]
            add(f"blend t={tau} + off", y, sigmoid(zb + offset_from(sel, zb, y)))

        print(f"done {bench}", flush=True)

    print("\n=== pooled over 16 held-out benchmarks ===")
    for name, (ys, ps) in results.items():
        yy, pp = np.concatenate(ys), np.concatenate(ps)
        print(f"{name:<22} log_loss={log_loss(yy, pp):.4f}  auc={auc(yy, pp):.4f}")


if __name__ == "__main__":
    main()
