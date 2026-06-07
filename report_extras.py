"""Numbers for the report upgrade.

1. Per-draw spread: pooled LOBO log-loss per random label draw (10 seeds),
   for the untuned (a'=5, clip 1.5) and tuned (a'=1, clip 2.5) offsets.
   Reports mean +- std across draws.
2. Fitted 1PL IRT baseline: P = sigmoid(theta_subject - d_{bench,cond} + mu),
   maximum likelihood with L2, evaluated on (a) the known-benchmark
   item-cold-start split and (b) pooled LOBO (held-out benchmark uses the
   mean difficulty).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

SUBMISSION_DIR = Path(__file__).parent / "submission"
sys.path.insert(0, str(SUBMISSION_DIR))
from prior_model import prior_logit  # noqa: E402

from train_ncf import aggregate, auc, build_tables, load_data, log_loss, render_subject_content  # noqa: E402

K = 5
N_SEEDS = 10
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def offset_from(sel, z, y, a_off, clip):
    p_hat = float(np.clip(sigmoid_np(z[sel]).mean(), 0.02, 0.98))
    m = float(np.clip((y[sel].sum() + a_off * p_hat) / (len(sel) + a_off), 0.01, 0.99))
    return float(np.clip(math.log(m / (1 - m)) - math.log(p_hat / (1 - p_hat)), -clip, clip))


def fit_1pl(rows, n_subj, n_bc, subj_idx, bc_idx, iters=300):
    s = torch.tensor(rows["si"], dtype=torch.long, device=DEVICE)
    b = torch.tensor(rows["bi"], dtype=torch.long, device=DEVICE)
    y = torch.tensor(rows["y"], dtype=torch.float32, device=DEVICE)
    theta = torch.zeros(n_subj, device=DEVICE, requires_grad=True)
    d = torch.zeros(n_bc, device=DEVICE, requires_grad=True)
    mu = torch.zeros(1, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([theta, d, mu], lr=0.1)
    bce = torch.nn.BCEWithLogitsLoss()
    for _ in range(iters):
        z = theta[s] - d[b] + mu
        loss = bce(z, y) + 1e-4 * (theta.pow(2).mean() + d.pow(2).mean())
        opt.zero_grad(); loss.backward(); opt.step()
    return (theta.detach().cpu().numpy(), d.detach().cpu().numpy(),
            float(mu.detach().cpu()))


def main():
    responses, subjects, items = load_data()
    agg = aggregate(responses)
    subject_content = {
        r["subject_id"]: render_subject_content(r) for _, r in subjects.iterrows()
    }
    subj_ids = sorted(agg["subject_id"].unique())
    subj_idx = {s: i for i, s in enumerate(subj_ids)}
    agg["bc"] = agg["benchmark_id"] + "|||" + agg["test_condition"]

    # ---------------- known-benchmark split: 1PL baseline ----------------
    rng = np.random.default_rng(0)
    val_items = set()
    for bench, grp in agg.groupby("benchmark_id"):
        uniq = grp["item_id"].unique()
        val_items.update(rng.choice(uniq, size=max(1, int(len(uniq) * 0.1)), replace=False))
    is_val = agg["item_id"].isin(val_items).to_numpy()
    tr, va = agg[~is_val], agg[is_val]
    bcs = sorted(tr["bc"].unique())
    bc_idx = {b: i for i, b in enumerate(bcs)}
    tr_rows = {"si": tr["subject_id"].map(subj_idx).to_numpy(),
               "bi": tr["bc"].map(bc_idx).to_numpy(),
               "y": tr["label"].to_numpy(dtype=np.float32)}
    theta, d, mu = fit_1pl(tr_rows, len(subj_ids), len(bcs), subj_idx, bc_idx)
    d_mean = float(d.mean())
    va_si = va["subject_id"].map(subj_idx).to_numpy()
    va_bi = va["bc"].map(lambda b: bc_idx.get(b, -1)).to_numpy()
    z = theta[va_si] - np.where(va_bi >= 0, d[np.maximum(va_bi, 0)], d_mean) + mu
    y = va["label"].to_numpy(dtype=np.float64)
    p = np.clip(sigmoid_np(z), 0.005, 0.995)
    print(f"1PL known-benchmark: log_loss={log_loss(y, p):.4f}  auc={auc(y, p):.4f}",
          flush=True)

    # ---------------- LOBO: 1PL + per-draw offset spread ----------------
    benches = sorted(agg["benchmark_id"].unique())
    irt_ys, irt_ps = [], []
    seed_lls = {("5.0", "1.5"): [[] for _ in range(N_SEEDS)],
                ("1.0", "2.5"): [[] for _ in range(N_SEEDS)]}
    seed_ys = [[] for _ in range(N_SEEDS)]

    for bench in benches:
        held = (agg["benchmark_id"] == bench).to_numpy()
        tr, te = agg[~held], agg[held]

        # 1PL fit without this benchmark
        bcs = sorted(tr["bc"].unique())
        bc_idx = {b: i for i, b in enumerate(bcs)}
        tr_rows = {"si": tr["subject_id"].map(subj_idx).to_numpy(),
                   "bi": tr["bc"].map(bc_idx).to_numpy(),
                   "y": tr["label"].to_numpy(dtype=np.float32)}
        theta, d, mu = fit_1pl(tr_rows, len(subj_ids), len(bcs), subj_idx, bc_idx)
        te_si = te["subject_id"].map(subj_idx).to_numpy()
        z_irt = theta[te_si] - float(d.mean()) + mu
        y = te["label"].to_numpy(dtype=np.float64)
        irt_ys.append(y)
        irt_ps.append(np.clip(sigmoid_np(z_irt), 0.005, 0.995))

        # shrinkage prior + offsets, per seed
        tables = build_tables(tr.reset_index(drop=True), subject_content)
        cache = {}
        z = np.empty(len(te))
        for i, (sid, b, c) in enumerate(
            zip(te["subject_id"], te["benchmark_id"], te["test_condition"])
        ):
            key = (sid, c)
            if key not in cache:
                cache[key] = prior_logit(tables, b, c, subject_content.get(sid, f"Name: {sid}"))
            z[i] = cache[key]
        for s in range(N_SEEDS):
            sel = np.random.default_rng(s).choice(len(y), size=min(K, len(y)), replace=False)
            for (ao, cl) in seed_lls:
                off = offset_from(sel, z, y, float(ao), float(cl))
                seed_lls[(ao, cl)][s].append(np.clip(sigmoid_np(z + off), 0.005, 0.995))
            if (ao, cl) == ("1.0", "2.5"):
                pass
        for s in range(N_SEEDS):
            seed_ys[s].append(y)
        print(f"done {bench}", flush=True)

    yy = np.concatenate(irt_ys)
    print(f"1PL LOBO (mean difficulty fallback): "
          f"log_loss={log_loss(yy, np.concatenate(irt_ps)):.4f}  "
          f"auc={auc(yy, np.concatenate(irt_ps)):.4f}", flush=True)

    for (ao, cl), per_seed in seed_lls.items():
        lls = []
        for s in range(N_SEEDS):
            ys_ = np.concatenate(seed_ys[s])
            ps_ = np.concatenate(per_seed[s])
            lls.append(log_loss(ys_, ps_))
        lls = np.array(lls)
        print(f"offsets a'={ao} clip={cl}: per-draw pooled LL "
              f"mean={lls.mean():.4f} std={lls.std():.4f} "
              f"min={lls.min():.4f} max={lls.max():.4f}", flush=True)


if __name__ == "__main__":
    main()
