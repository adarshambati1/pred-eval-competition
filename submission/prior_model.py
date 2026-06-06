import math

ALPHA = 40.0


def _logit(p):
    p = max(1e-9, min(1.0 - 1e-9, p))
    return math.log(p / (1.0 - p))


def _sigmoid(x):
    if x > 500:
        return 1.0
    if x < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _shrink(stat, parent, alpha=None):
    if alpha is None:
        alpha = ALPHA
    if not stat:
        return parent
    n = stat["count"]
    return (stat["mean"] * n + parent * alpha) / (n + alpha)


def parse_name_line(subject_content):
    for line in subject_content.split("\n"):
        line = line.strip()
        if line.startswith("Name:"):
            return line
    return f"Name: {subject_content.split(chr(10))[0].strip()}"


def normalize_condition(condition):
    cond = (condition or "").strip()
    return cond if cond else "none"


def prior_probability(tables, benchmark, condition, subject_content, use_condition=True):
    g = tables["global_mean"]
    cond = normalize_condition(condition)
    name_line = parse_name_line(subject_content)

    per_subject = tables["per_subject"]
    subj_stat = per_subject.get(subject_content) or per_subject.get(name_line)

    p_bench = _shrink(tables["per_benchmark"].get(benchmark), g)
    if use_condition:
        bc_stat = tables["per_benchmark_condition"].get(f"{benchmark}|||{cond}")
        p_bc = _shrink(bc_stat, p_bench)
    else:
        p_bc = p_bench

    p_subj = _shrink(subj_stat, g)
    combo = _sigmoid(_logit(p_subj) + _logit(p_bc) - _logit(g))

    sb_stat = tables["per_subject_benchmark"].get(f"{name_line}|||{benchmark}")
    p_sb = _shrink(sb_stat, combo)

    if use_condition:
        sbc_stat = tables["per_subject_benchmark_condition"].get(
            f"{name_line}|||{benchmark}|||{cond}"
        )
        return _shrink(sbc_stat, p_sb)
    return p_sb


def prior_logit(tables, benchmark, condition, subject_content, use_condition=True):
    return _logit(
        prior_probability(tables, benchmark, condition, subject_content, use_condition)
    )
