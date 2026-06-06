import json
import math
from pathlib import Path

from prior_model import normalize_condition, prior_logit

_DIR = Path(__file__).parent

with open(_DIR / "lookup_tables.json") as f:
    _TABLES = json.load(f)

_ENCODER = None
_HEAD = None
_torch = None

try:
    import numpy as _np
    import torch as _torch_mod
    import torch.nn as _nn
    from sentence_transformers import SentenceTransformer

    _torch = _torch_mod
    _ckpt = _torch.load(_DIR / "ncf_head.pt", map_location="cpu", weights_only=False)

    class _ResidualHead(_nn.Module):
        def __init__(self, dim_in, hidden):
            super().__init__()
            self.net = _nn.Sequential(
                _nn.Linear(dim_in, hidden),
                _nn.ReLU(),
                _nn.Linear(hidden, hidden // 4),
                _nn.ReLU(),
                _nn.Linear(hidden // 4, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    _DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"
    _HEAD = _ResidualHead(_ckpt["dim_in"], _ckpt["hidden"])
    _HEAD.load_state_dict(_ckpt["state_dict"])
    _HEAD.eval()
    _HEAD.to(_DEVICE)
    _ENCODER = SentenceTransformer(_ckpt["encoder"], device=_DEVICE)
except Exception:
    _ENCODER = None
    _HEAD = None

_emb_cache = {}
_platt = None
_platt_key = None


def _sigmoid(x):
    if x > 500:
        return 1.0
    if x < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _clamp(p, eps=0.005):
    return max(eps, min(1.0 - eps, p))


def _embed(text):
    # MiniLM truncates to 256 tokens; 2000 chars is beyond that horizon, so
    # pre-truncating gives an identical embedding while bounding tokenizer
    # cost on very long item texts.
    text = text[:2000]
    if text not in _emb_cache:
        _emb_cache[text] = _ENCODER.encode(
            text, normalize_embeddings=True, show_progress_bar=False
        )
    return _emb_cache[text]


def _raw_logit(input):
    benchmark = input.get("benchmark", "")
    condition = normalize_condition(input.get("condition", ""))
    subject_content = input.get("subject_content", "")
    item_content = input.get("item_content", "")

    z = prior_logit(_TABLES, benchmark, condition, subject_content)

    ncf_benches = _TABLES.get("ncf_benchmarks")
    if ncf_benches is not None:
        known_bench = benchmark in ncf_benches
    else:
        bench_stat = _TABLES["per_benchmark"].get(benchmark)
        known_bench = bench_stat is not None and bench_stat["count"] >= 50

    if known_bench and _HEAD is not None and _ENCODER is not None and item_content:
        try:
            item_e = _embed(item_content)
            subj_e = _embed(subject_content)
            x = _torch.from_numpy(
                _np.concatenate([item_e, subj_e, [_np.float32(z)]]).astype("float32")
            ).unsqueeze(0)
            with _torch.no_grad():
                z = z + float(_HEAD(x).item())
        except Exception:
            pass
    return z


def _fit_platt(labeled):
    zs = [_raw_logit(ex) for ex in labeled]
    ys = [float(ex.get("label", 0)) for ex in labeled]
    a, b = 1.0, 0.0
    lam = 15.0 / max(len(zs), 1)
    lr = 0.1
    for _ in range(200):
        ga = lam * (a - 1.0)
        gb = lam * b
        for z, y in zip(zs, ys):
            p = _sigmoid(a * z + b)
            ga += (p - y) * z / len(zs)
            gb += (p - y) / len(zs)
        a -= lr * ga
        b -= lr * gb
    a = max(0.25, min(2.0, a))
    b = max(-1.5, min(1.5, b))

    offsets = {}
    by_bench = {}
    for z, y, ex in zip(zs, ys, labeled):
        by_bench.setdefault(ex.get("benchmark", ""), []).append((z, y))
    alpha = 0.5
    for bench, pairs in by_bench.items():
        n = len(pairs)
        p_hat = sum(_sigmoid(a * z + b) for z, _ in pairs) / n
        p_hat = max(0.02, min(0.98, p_hat))
        m = (sum(y for _, y in pairs) + alpha * p_hat) / (n + alpha)
        m = max(0.01, min(0.99, m))
        c = math.log(m / (1 - m)) - math.log(p_hat / (1 - p_hat))
        offsets[bench] = max(-2.5, min(2.5, c))

    return a, b, offsets


def predict(input, labeled=None):
    global _platt, _platt_key

    z = _raw_logit(input)

    if labeled:
        key = len(labeled)
        if _platt_key != key:
            try:
                _platt = _fit_platt(labeled)
            except Exception:
                _platt = (1.0, 0.0, {})
            _platt_key = key
        a, b, offsets = _platt
        z = a * z + b + offsets.get(input.get("benchmark", ""), 0.0)

    return float(_clamp(_sigmoid(z)))
