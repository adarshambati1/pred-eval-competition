"""lobo_eval.py — Leave-one-benchmark-out validation.

For each held-out benchmark B:
  - build lookup tables from all rows EXCEPT B
  - train the residual head on those rows (same recipe as train_ncf.py)
  - predict B's rows cold (prior falls back to global; NCF sees only text)
  - also simulate the adaptive-labeling channel: reveal K random labels from B,
    apply the per-benchmark offset exactly as model.py does, and re-score.

This mirrors the cleaned private test set far better than held-out items from
known benchmarks. Prints per-benchmark and pooled log-loss/AUC for:
    prior / prior+NCF / prior+NCF+offset(K labels)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SUBMISSION_DIR = Path(__file__).parent / "submission"
sys.path.insert(0, str(SUBMISSION_DIR))
from prior_model import prior_logit  # noqa: E402

from train_ncf import (  # noqa: E402
    BATCH,
    LR,
    ResidualHead,
    aggregate,
    auc,
    build_tables,
    encode_cached,
    load_data,
    log_loss,
    render_subject_content,
)

SEED = 0
K_LABELS = 5          # labels revealed per benchmark (competition default)
OFFSET_ALPHA = 5.0    # same shrinkage as model.py
LOBO_EPOCHS = 2       # head overfits past epoch ~1-2; keep it cheap per fold


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def fit_head(item_t, subj_t, ii, si, pr, y, device, rng):
    model = ResidualHead(item_t.shape[1] + subj_t.shape[1] + 1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    n = len(y)
    for _ in range(LOBO_EPOCHS):
        perm = rng.permutation(n)
        for s in range(0, n, BATCH):
            b = perm[s : s + BATCH]
            x = torch.cat(
                [item_t[ii[b]], subj_t[si[b]],
                 torch.from_numpy(pr[b]).to(device).unsqueeze(1)], dim=1)
            logit = torch.from_numpy(pr[b]).to(device) + model(x)
            loss = bce(logit, torch.from_numpy(y[b]).to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def head_logits(model, item_t, subj_t, ii, si, pr, device):
    out = []
    with torch.no_grad():
        for s in range(0, len(pr), 65536):
            sl = slice(s, s + 65536)
            x = torch.cat(
                [item_t[ii[sl]], subj_t[si[sl]],
                 torch.from_numpy(pr[sl]).to(device).unsqueeze(1)], dim=1)
            out.append((torch.from_numpy(pr[sl]).to(device) + model(x)).cpu().numpy())
    return np.concatenate(out)


def main():
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    responses, subjects, items = load_data()
    agg = aggregate(responses)
    subject_content = {
        row["subject_id"]: render_subject_content(row) for _, row in subjects.iterrows()
    }
    item_content = dict(zip(items["item_id"], items["content"].fillna("")))
    agg = agg[agg["item_id"].isin(item_content.keys())].reset_index(drop=True)

    uniq_items = sorted(agg["item_id"].unique())
    uniq_subj = sorted(agg["subject_id"].unique())
    item_emb = encode_cached([item_content[i] for i in uniq_items], "items", device)
    subj_emb = encode_cached(
        [subject_content.get(s, f"Name: {s}") for s in uniq_subj], "subjects", device)
    item_idx = {iid: k for k, iid in enumerate(uniq_items)}
    subj_idx = {sid: k for k, sid in enumerate(uniq_subj)}
    item_t = torch.from_numpy(item_emb).to(device)
    subj_t = torch.from_numpy(subj_emb).to(device)

    benches = sorted(agg["benchmark_id"].unique())
    pooled = {"prior": ([], []), "ncf": ([], []), "ncf_off": ([], [])}

    print(f"{'benchmark':<20} {'rows':>7}  {'prior':>7} {'ncf':>7} {'ncf+off':>7}   auc(ncf)")
    for bench in benches:
        is_held = (agg["benchmark_id"] == bench).to_numpy()
        tr, te = agg[~is_held].reset_index(drop=True), agg[is_held].reset_index(drop=True)
        tables = build_tables(tr, subject_content)

        def priors(rows):
            cache = {}
            out = np.empty(len(rows), dtype=np.float32)
            for i, (sid, b, c) in enumerate(
                zip(rows["subject_id"], rows["benchmark_id"], rows["test_condition"])
            ):
                key = (sid, b, c)
                if key not in cache:
                    cache[key] = prior_logit(
                        tables, b, c, subject_content.get(sid, f"Name: {sid}"))
                out[i] = cache[key]
            return out

        tr_pr, te_pr = priors(tr), priors(te)

        def feats(rows, pr):
            return (rows["item_id"].map(item_idx).to_numpy(),
                    rows["subject_id"].map(subj_idx).to_numpy(),
                    pr, rows["label"].to_numpy(dtype=np.float32))

        tr_ii, tr_si, tr_p, tr_y = feats(tr, tr_pr)
        te_ii, te_si, te_p, te_y = feats(te, te_pr)

        head = fit_head(item_t, subj_t, tr_ii, tr_si, tr_p, tr_y, device, rng)
        z = head_logits(head, item_t, subj_t, te_ii, te_si, te_p, device)

        # Simulate adaptive labeling: K random labeled rows from the held-out
        # benchmark -> per-benchmark offset, exactly the model.py formula.
        sel = rng.choice(len(te_y), size=min(K_LABELS, len(te_y)), replace=False)
        p_hat = float(np.clip(sigmoid(z[sel]).mean(), 0.02, 0.98))
        m = (te_y[sel].sum() + OFFSET_ALPHA * p_hat) / (len(sel) + OFFSET_ALPHA)
        m = float(np.clip(m, 0.02, 0.98))
        off = np.clip(math.log(m / (1 - m)) - math.log(p_hat / (1 - p_hat)), -1.5, 1.5)

        p_prior = np.clip(sigmoid(te_p), 0.005, 0.995)
        p_ncf = np.clip(sigmoid(z), 0.005, 0.995)
        p_off = np.clip(sigmoid(z + off), 0.005, 0.995)

        print(f"{bench:<20} {len(te):>7}  {log_loss(te_y, p_prior):>7.4f} "
              f"{log_loss(te_y, p_ncf):>7.4f} {log_loss(te_y, p_off):>7.4f}   "
              f"{auc(te_y, p_ncf):.4f}")

        for name, p in [("prior", p_prior), ("ncf", p_ncf), ("ncf_off", p_off)]:
            pooled[name][0].append(te_y)
            pooled[name][1].append(p)

    print("\n=== pooled over all held-out benchmarks ===")
    for name, (ys, ps) in pooled.items():
        y, p = np.concatenate(ys), np.concatenate(ps)
        print(f"{name:<8} log_loss={log_loss(y, p):.4f}  auc={auc(y, p):.4f}")


if __name__ == "__main__":
    main()
