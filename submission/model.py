import json
import math
from pathlib import Path

from prior_model import normalize_condition, prior_logit

_DIR = Path(__file__).parent

with open(_DIR / "lookup_tables.json") as f:
    _TABLES = json.load(f)

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


def _raw_logit(input):
    benchmark = input.get("benchmark", "")
    condition = normalize_condition(input.get("condition", ""))
    subject_content = input.get("subject_content", "")
    return prior_logit(_TABLES, benchmark, condition, subject_content)


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
    alpha = 1.0
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
