"""LOBO test: better subject-ability estimates for unseen benchmarks.

Variants for the subject-side mean used in the prior's Rasch combination:
  pooled    - row-weighted mean over all rows (current behavior)
  balanced  - unweighted mean of the subject's per-benchmark means
  modality  - balanced mean over benchmarks of the SAME modality as the
              held-out benchmark (vision vs text; oracle modality here,
              cheap heuristic at deploy)

Each evaluated as prior-only and prior+offset (a'=0.5, clip 2.5, 10 seeds).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

SUBMISSION_DIR = Path(__file__).parent / "submission"
sys.path.insert(0, str(SUBMISSION_DIR))
from prior_model import _logit, _shrink, _sigmoid  # noqa: E402

from train_ncf import aggregate, auc, load_data, log_loss, render_subject_content  # noqa: E402

K = 5
A_OFF, CLIP = 0.5, 2.5
ALPHA = 40.0

VISION = {"ai2d_test", "mathvista_mini", "mmbench_v11"}  # multimodal training benches


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def offset_from(sel, z, y):
    p_hat = float(np.clip(sigmoid_np(z[sel]).mean(), 0.02, 0.98))
    m = float(np.clip((y[sel].sum() + A_OFF * p_hat) / (len(sel) + A_OFF), 0.01, 0.99))
    return float(np.clip(math.log(m / (1 - m)) - math.log(p_hat / (1 - p_hat)), -CLIP, CLIP))


def main():
    responses, subjects, items = load_data()
    agg = aggregate(responses)

    results = {}

    def add(name, y, p):
        results.setdefault(name, ([], []))
        results[name][0].append(y)
        results[name][1].append(np.clip(p, 0.005, 0.995))

    benches = sorted(agg["benchmark_id"].unique())
    for bench in benches:
        held = (agg["benchmark_id"] == bench).to_numpy()
        tr, te = agg[~held].reset_index(drop=True), agg[held].reset_index(drop=True)
        g = float(tr["label"].mean())

        bench_stats = tr.groupby("benchmark_id")["label"].agg(["mean", "count"])
        # subject x benchmark means (train side)
        sb = tr.groupby(["subject_id", "benchmark_id"])["label"].agg(["mean", "count"])

        # pooled subject mean (current behavior)
        s_pooled = tr.groupby("subject_id")["label"].agg(["mean", "count"])

        # balanced: average the subject's per-benchmark means equally,
        # count = number of benchmarks * a nominal per-benchmark weight
        bal_mean = sb.groupby(level=0)["mean"].mean()
        bal_n = sb.groupby(level=0)["mean"].size()

        # modality: balanced over same-modality benchmarks only
        same_mod = (
            sb.reset_index()
        )
        same_mod["vision"] = same_mod["benchmark_id"].isin(VISION)
        bench_is_vision = bench in VISION
        mod = same_mod[same_mod["vision"] == bench_is_vision]
        mod_mean = mod.groupby("subject_id")["mean"].mean()
        mod_n = mod.groupby("subject_id")["mean"].size()

        y = te["label"].to_numpy(dtype=np.float64)
        sids = te["subject_id"].to_numpy()

        def prior_vec(kind):
            cache = {}
            out = np.empty(len(te))
            for i, sid in enumerate(sids):
                if sid not in cache:
                    if kind == "pooled":
                        st = (
                            {"mean": float(s_pooled.loc[sid, "mean"]),
                             "count": int(s_pooled.loc[sid, "count"])}
                            if sid in s_pooled.index else None
                        )
                    elif kind == "balanced":
                        st = (
                            {"mean": float(bal_mean[sid]), "count": int(bal_n[sid]) * 50}
                            if sid in bal_mean.index else None
                        )
                    else:  # modality
                        if sid in mod_mean.index:
                            st = {"mean": float(mod_mean[sid]), "count": int(mod_n[sid]) * 50}
                        elif sid in bal_mean.index:  # fall back to balanced
                            st = {"mean": float(bal_mean[sid]), "count": int(bal_n[sid]) * 25}
                        else:
                            st = None
                    p_subj = _shrink(st, g, ALPHA)
                    # unknown benchmark: item side falls back to global
                    cache[sid] = _logit(p_subj) + _logit(g) - _logit(g)
                out[i] = cache[sid]
            return out

        for kind in ("pooled", "balanced", "modality"):
            z = prior_vec(kind)
            add(f"{kind}", y, sigmoid_np(z))
            acc = np.zeros_like(z)
            for s in range(10):
                sel = np.random.default_rng(s).choice(len(y), size=min(K, len(y)), replace=False)
                acc += sigmoid_np(z + offset_from(sel, z, y))
            add(f"{kind}+off", y, acc / 10)
        print(f"done {bench}", flush=True)

    print("\n=== pooled LOBO ===")
    for name, (ys, ps) in results.items():
        yy, pp = np.concatenate(ys), np.concatenate(ps)
        print(f"{name:<14} log_loss={log_loss(yy, pp):.4f}  auc={auc(yy, pp):.4f}")


if __name__ == "__main__":
    main()
