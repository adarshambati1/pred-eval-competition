"""train_ncf.py — Build condition-aware lookup tables and train the NCF residual head.

Pipeline:
  1. Load all response tables + registries from HuggingFace (cached locally).
  2. Aggregate to (subject, item, condition) with soft labels (mean over trials).
  3. Cold-start split: hold out 10% of ITEMS per benchmark (mirrors the test regime
     where every test item is unseen).
  4. Build hierarchical lookup tables from TRAIN rows only.
  5. Encode item/subject text with all-MiniLM-L6-v2 (cached to .npy).
  6. Train an MLP residual head: final_logit = prior_logit + MLP(item_emb, subj_emb, prior_logit).
  7. Report ablations (global / no-condition prior / full prior / prior+NCF) on the
     held-out cold-start items, then save artifacts into submission/.

Usage:
    conda activate pred-eval
    python train_ncf.py
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from huggingface_hub import HfApi, hf_hub_download

SUBMISSION_DIR = Path(__file__).parent / "submission"
sys.path.insert(0, str(SUBMISSION_DIR))
from prior_model import normalize_condition, parse_name_line, prior_logit  # noqa: E402

REPO_ID = "aims-foundations/measurement-db"
REGISTRY_FILES = {"subjects.parquet", "items.parquet", "benchmarks.parquet"}
ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = Path(__file__).parent / "emb_cache"
CACHE_DIR.mkdir(exist_ok=True)

SEED = 0
VAL_ITEM_FRAC = 0.10
EPOCHS = 6
BATCH = 4096
LR = 1e-3
HIDDEN = 256


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def render_subject_content(row) -> str:
    """Same rendering as the platform / original train.py."""
    display_name = row.get("display_name") or row.get("subject_id", "unknown")
    lines = [f"Name: {display_name}"]
    for key, label in [
        ("provider", "Organization"),
        ("params", "Parameters"),
        ("release_date", "Released"),
        ("family", "Family"),
    ]:
        value = row.get(key)
        if pd.notna(value) and str(value).strip():
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def load_data():
    api = HfApi()
    repo_files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    response_files = sorted(
        f for f in repo_files
        if f.endswith(".parquet")
        and f not in REGISTRY_FILES
        and not f.endswith("_traces.parquet")
    )
    print(f"Loading {len(response_files)} response files...")
    dfs = []
    for rf in response_files:
        path = hf_hub_download(REPO_ID, rf, repo_type="dataset")
        df = pd.read_parquet(
            path, columns=["subject_id", "item_id", "benchmark_id", "test_condition", "response"]
        )
        dfs.append(df)
    responses = pd.concat(dfs, ignore_index=True)

    subjects = pd.read_parquet(hf_hub_download(REPO_ID, "subjects.parquet", repo_type="dataset"))
    items = pd.read_parquet(
        hf_hub_download(REPO_ID, "items.parquet", repo_type="dataset"),
        columns=["item_id", "benchmark_id", "content"],
    )
    return responses, subjects, items


def aggregate(responses: pd.DataFrame) -> pd.DataFrame:
    """Binary-filter and aggregate trials to (subject, item, condition) soft labels."""
    df = responses[responses["response"].isin([0.0, 1.0])].copy()
    df["test_condition"] = df["test_condition"].fillna("").map(normalize_condition)
    agg = (
        df.groupby(["subject_id", "item_id", "benchmark_id", "test_condition"])["response"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "label", "count": "n_trials"})
    )
    print(f"Aggregated {len(df)} responses -> {len(agg)} (subject, item, condition) cells")
    return agg


# ---------------------------------------------------------------------------
# Lookup tables (train rows only — honest cold-start validation)
# ---------------------------------------------------------------------------

def build_tables(rows: pd.DataFrame, subject_content: dict[str, str]) -> dict:
    def stats(group_cols, key_fn):
        g = rows.groupby(group_cols).agg(
            mean=("label", "mean"), count=("label", "size")
        )
        out = {}
        for idx, r in g.iterrows():
            for key in key_fn(idx):
                out[key] = {"mean": round(float(r["mean"]), 6), "count": int(r["count"])}
        return out

    name_line = {sid: parse_name_line(c) for sid, c in subject_content.items()}

    def subj_keys(sid):
        return [name_line[sid], subject_content[sid]]

    tables = {
        "global_mean": round(float(rows["label"].mean()), 6),
        "per_subject": stats("subject_id", subj_keys),
        "per_benchmark": stats("benchmark_id", lambda b: [b]),
        "per_benchmark_condition": stats(
            ["benchmark_id", "test_condition"], lambda k: [f"{k[0]}|||{k[1]}"]
        ),
        "per_subject_benchmark": stats(
            ["subject_id", "benchmark_id"], lambda k: [f"{name_line[k[0]]}|||{k[1]}"]
        ),
        "per_subject_benchmark_condition": stats(
            ["subject_id", "benchmark_id", "test_condition"],
            lambda k: [f"{name_line[k[0]]}|||{k[1]}|||{k[2]}"],
        ),
    }
    return tables


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def encode_cached(texts: list[str], tag: str, device: str) -> np.ndarray:
    """Encode texts with MiniLM, caching by content hash."""
    digest = hashlib.sha256("\x00".join(texts).encode()).hexdigest()[:16]
    cache_path = CACHE_DIR / f"{tag}_{digest}.npy"
    if cache_path.exists():
        print(f"  embeddings cache hit: {cache_path.name}")
        return np.load(cache_path)
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(ENCODER_NAME, device=device)
    emb = encoder.encode(
        texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True
    ).astype(np.float32)
    np.save(cache_path, emb)
    return emb


# ---------------------------------------------------------------------------
# Residual MLP
# ---------------------------------------------------------------------------

class ResidualHead(nn.Module):
    def __init__(self, dim_in: int, hidden: int = HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 4),
            nn.ReLU(),
            nn.Linear(hidden // 4, 1),
        )
        # Start at the prior: zero-init the last layer so delta == 0 at step 0.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """AUC for soft labels: binarize at 0.5 (ties dropped)."""
    yb = (y > 0.5).astype(int)
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    n_pos, n_neg = yb.sum(), (1 - yb).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[yb == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main():
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    responses, subjects, items = load_data()
    agg = aggregate(responses)

    subject_content = {
        row["subject_id"]: render_subject_content(row) for _, row in subjects.iterrows()
    }
    item_content = dict(zip(items["item_id"], items["content"].fillna("")))

    # Drop rows whose item has no registered content (cannot embed).
    agg = agg[agg["item_id"].isin(item_content.keys())].reset_index(drop=True)

    # --- Cold-start item split: hold out 10% of items per benchmark ---
    val_items: set[str] = set()
    for bench, grp in agg.groupby("benchmark_id"):
        uniq = grp["item_id"].unique()
        n_val = max(1, int(len(uniq) * VAL_ITEM_FRAC))
        val_items.update(rng.choice(uniq, size=n_val, replace=False))
    is_val = agg["item_id"].isin(val_items).to_numpy()
    train_rows, val_rows = agg[~is_val].reset_index(drop=True), agg[is_val].reset_index(drop=True)
    print(f"Split: {len(train_rows)} train rows, {len(val_rows)} val rows "
          f"({len(val_items)} held-out items)")

    # --- Lookup tables from train rows only ---
    tables = build_tables(train_rows, subject_content)
    print(f"Tables: {len(tables['per_subject_benchmark_condition'])} (s,b,c) cells, "
          f"{len(tables['per_benchmark_condition'])} (b,c) cells")

    # --- Priors for every row (computed exactly as at test time) ---
    def compute_priors(rows: pd.DataFrame, use_condition: bool) -> np.ndarray:
        cache: dict[tuple, float] = {}
        out = np.empty(len(rows), dtype=np.float32)
        cols = zip(rows["subject_id"], rows["benchmark_id"], rows["test_condition"])
        for i, (sid, bench, cond) in enumerate(cols):
            key = (sid, bench, cond, use_condition)
            if key not in cache:
                cache[key] = prior_logit(
                    tables, bench, cond, subject_content.get(sid, f"Name: {sid}"),
                    use_condition=use_condition,
                )
            out[i] = cache[key]
        return out

    print("Computing priors...")
    train_prior = compute_priors(train_rows, use_condition=True)
    val_prior = compute_priors(val_rows, use_condition=True)
    val_prior_nocond = compute_priors(val_rows, use_condition=False)

    # --- Embeddings ---
    print("Encoding item text...")
    uniq_items = sorted({*train_rows["item_id"], *val_rows["item_id"]})
    item_emb = encode_cached([item_content[i] for i in uniq_items], "items", device)
    item_idx = {iid: k for k, iid in enumerate(uniq_items)}

    print("Encoding subject text...")
    uniq_subj = sorted({*train_rows["subject_id"], *val_rows["subject_id"]})
    subj_emb = encode_cached(
        [subject_content.get(s, f"Name: {s}") for s in uniq_subj], "subjects", device
    )
    subj_idx = {sid: k for k, sid in enumerate(uniq_subj)}

    def features(rows: pd.DataFrame, prior: np.ndarray):
        ii = rows["item_id"].map(item_idx).to_numpy()
        si = rows["subject_id"].map(subj_idx).to_numpy()
        return ii, si, prior, rows["label"].to_numpy(dtype=np.float32)

    tr_ii, tr_si, tr_pr, tr_y = features(train_rows, train_prior)
    va_ii, va_si, va_pr, va_y = features(val_rows, val_prior)

    item_t = torch.from_numpy(item_emb).to(device)
    subj_t = torch.from_numpy(subj_emb).to(device)

    dim_in = item_emb.shape[1] + subj_emb.shape[1] + 1
    model = ResidualHead(dim_in).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()

    def eval_val() -> np.ndarray:
        model.eval()
        preds = []
        with torch.no_grad():
            for s in range(0, len(va_y), 65536):
                sl = slice(s, s + 65536)
                x = torch.cat(
                    [
                        item_t[va_ii[sl]],
                        subj_t[va_si[sl]],
                        torch.from_numpy(va_pr[sl]).to(device).unsqueeze(1),
                    ],
                    dim=1,
                )
                logit = torch.from_numpy(va_pr[sl]).to(device) + model(x)
                preds.append(torch.sigmoid(logit).cpu().numpy())
        return np.concatenate(preds)

    # --- Baselines on the cold-start val split ---
    print("\n=== Cold-start validation ablations ===")
    p_global = np.full_like(va_y, tables["global_mean"])
    print(f"global-only        : log_loss={log_loss(va_y, p_global):.4f}")
    p_nocond = 1 / (1 + np.exp(-val_prior_nocond))
    print(f"prior (no cond)    : log_loss={log_loss(va_y, p_nocond):.4f}  auc={auc(va_y, p_nocond):.4f}")
    p_prior = 1 / (1 + np.exp(-va_pr))
    print(f"prior (full)       : log_loss={log_loss(va_y, p_prior):.4f}  auc={auc(va_y, p_prior):.4f}")

    # --- Train ---
    best_ll, best_state = float("inf"), None
    n = len(tr_y)
    for epoch in range(EPOCHS):
        model.train()
        perm = rng.permutation(n)
        total = 0.0
        for s in range(0, n, BATCH):
            b = perm[s : s + BATCH]
            x = torch.cat(
                [
                    item_t[tr_ii[b]],
                    subj_t[tr_si[b]],
                    torch.from_numpy(tr_pr[b]).to(device).unsqueeze(1),
                ],
                dim=1,
            )
            logit = torch.from_numpy(tr_pr[b]).to(device) + model(x)
            loss = bce(logit, torch.from_numpy(tr_y[b]).to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        p_val = eval_val()
        ll = log_loss(va_y, p_val)
        marker = ""
        if ll < best_ll:
            best_ll = ll
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            marker = "  *"
        print(f"epoch {epoch + 1}: train_bce={total / n:.4f}  val_log_loss={ll:.4f}  "
              f"val_auc={auc(va_y, p_val):.4f}{marker}")

    model.load_state_dict(best_state)
    p_final = eval_val()
    print(f"\nprior + NCF (best) : log_loss={log_loss(va_y, p_final):.4f}  "
          f"auc={auc(va_y, p_final):.4f}")

    # --- Save artifacts ---
    with open(SUBMISSION_DIR / "lookup_tables.json", "w") as f:
        json.dump(tables, f)
    torch.save(
        {"state_dict": model.state_dict(), "dim_in": dim_in, "hidden": HIDDEN,
         "encoder": ENCODER_NAME},
        SUBMISSION_DIR / "ncf_head.pt",
    )
    (SUBMISSION_DIR / "models.txt").write_text(ENCODER_NAME + "\n")
    size_mb = (SUBMISSION_DIR / "lookup_tables.json").stat().st_size / 1e6
    print(f"\nSaved lookup_tables.json ({size_mb:.1f} MB), ncf_head.pt, models.txt "
          f"to {SUBMISSION_DIR}/")
    print("Now run: python build_zip.py")


if __name__ == "__main__":
    main()
