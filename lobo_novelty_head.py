"""Novelty-trained head: learn item-level corrections that transfer to
unseen benchmarks, by training ONLY on LOBO-regime examples.

For each benchmark B, compute priors for B's rows using tables built
WITHOUT B (so every training example looks like deployment on a novel
benchmark). Train the usual residual head on 12 benchmarks' LOBO rows,
evaluate pooled on the other 4 (rotated, 4 disjoint eval groups).

If prior+head beats prior on held-out groups, the head transfers.
Also evaluates head+offset (the deployment combo).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
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

K = 5
A_OFF, CLIP = 0.5, 2.5
EPOCHS = 2


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def offset_from(sel, z, y):
    p_hat = float(np.clip(sigmoid(z[sel]).mean(), 0.02, 0.98))
    m = float(np.clip((y[sel].sum() + A_OFF * p_hat) / (len(sel) + A_OFF), 0.01, 0.99))
    return float(np.clip(math.log(m / (1 - m)) - math.log(p_hat / (1 - p_hat)), -CLIP, CLIP))


def main():
    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    responses, subjects, items = load_data()
    agg = aggregate(responses)
    subject_content = {
        r["subject_id"]: render_subject_content(r) for _, r in subjects.iterrows()
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

    # LOBO-regime dataset: per benchmark, priors computed without it
    benches = sorted(agg["benchmark_id"].unique())
    fold = {}
    for bench in benches:
        held = (agg["benchmark_id"] == bench).to_numpy()
        tr, te = agg[~held].reset_index(drop=True), agg[held].reset_index(drop=True)
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
        fold[bench] = {
            "ii": te["item_id"].map(item_idx).to_numpy(),
            "si": te["subject_id"].map(subj_idx).to_numpy(),
            "z": z,
            "y": te["label"].to_numpy(dtype=np.float32),
        }
        print(f"fold ready: {bench} ({len(te)} rows)", flush=True)

    groups = [benches[i::4] for i in range(4)]  # 4 disjoint eval groups

    pooled = {"prior": ([], []), "prior+off": ([], []),
              "prior+head": ([], []), "prior+head+off": ([], [])}

    for g, eval_benches in enumerate(groups):
        train_benches = [b for b in benches if b not in eval_benches]
        ii = np.concatenate([fold[b]["ii"] for b in train_benches])
        si = np.concatenate([fold[b]["si"] for b in train_benches])
        z = np.concatenate([fold[b]["z"] for b in train_benches])
        y = np.concatenate([fold[b]["y"] for b in train_benches])

        model = ResidualHead(item_emb.shape[1] + subj_emb.shape[1] + 1).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        bce = nn.BCEWithLogitsLoss()
        rng = np.random.default_rng(g)
        n = len(y)
        for _ in range(EPOCHS):
            perm = rng.permutation(n)
            for s in range(0, n, BATCH):
                b_ = perm[s:s + BATCH]
                x = torch.cat([item_t[ii[b_]], subj_t[si[b_]],
                               torch.from_numpy(z[b_]).to(device).unsqueeze(1)], dim=1)
                logit = torch.from_numpy(z[b_]).to(device) + model(x)
                loss = bce(logit, torch.from_numpy(y[b_]).to(device))
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()

        for bench in eval_benches:
            f = fold[bench]
            zs = []
            with torch.no_grad():
                for s in range(0, len(f["y"]), 65536):
                    sl = slice(s, s + 65536)
                    x = torch.cat([item_t[f["ii"][sl]], subj_t[f["si"][sl]],
                                   torch.from_numpy(f["z"][sl]).to(device).unsqueeze(1)], dim=1)
                    zs.append((torch.from_numpy(f["z"][sl]).to(device) + model(x)).cpu().numpy())
            zh = np.concatenate(zs)
            zp = f["z"].astype(np.float64)
            y_ = f["y"].astype(np.float64)

            accs = {k: np.zeros_like(zp) for k in ("prior+off", "prior+head+off")}
            for s in range(10):
                sel = np.random.default_rng(s).choice(len(y_), size=min(K, len(y_)), replace=False)
                accs["prior+off"] += sigmoid(zp + offset_from(sel, zp, y_))
                accs["prior+head+off"] += sigmoid(zh + offset_from(sel, zh, y_))

            pooled["prior"][0].append(y_); pooled["prior"][1].append(np.clip(sigmoid(zp), .005, .995))
            pooled["prior+head"][0].append(y_); pooled["prior+head"][1].append(np.clip(sigmoid(zh), .005, .995))
            for k in accs:
                pooled[k][0].append(y_); pooled[k][1].append(np.clip(accs[k] / 10, .005, .995))
        print(f"group {g + 1}/4 evaluated: {eval_benches}", flush=True)

    print("\n=== pooled over all benchmarks (novelty-trained head) ===")
    for name, (ys, ps) in pooled.items():
        yy, pp = np.concatenate(ys), np.concatenate(ps)
        print(f"{name:<16} log_loss={log_loss(yy, pp):.4f}  auc={auc(yy, pp):.4f}")


if __name__ == "__main__":
    main()
