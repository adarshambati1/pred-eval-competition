# Predictive Evaluation Competition — Final Submission

Code for my entry in the CS 321M Predictive Evaluation Competition: predicting
the probability that an AI subject answers a benchmark item correctly, without
running the subject on the item (item cold-start).

**Best leaderboard score: -0.60 negative log-loss** (organizer baseline: -0.79;
reproduced across three runs). The final model contains **no neural network at
test time**: a hierarchical shrinkage prior over accuracy statistics, 3,663
(subject, benchmark) cells curated from the public OpenVLM leaderboard, and a
per-benchmark calibration estimated online from the competition's five
revealed labels per data category.

## Final submission (`submission/`)

The exact code behind the best-scoring Codabench run:

| File | Role |
|---|---|
| `model.py` | `predict()`: hierarchical prior + curated tables + online label calibration (global Platt rescaling + per-benchmark logit offsets). Pure Python. |
| `prior_model.py` | The shrinkage cascade (global → benchmark → benchmark+condition, crossed with subject ability), identical code at train and test time. |
| `labeling.py` | `acquisition_function()`: novelty bonus + deterministic pseudo-random tiebreak. Random-within-benchmark beats uncertainty sampling here (it preserves subject diversity, giving unbiased base-rate estimates). |
| `lookup_tables.json` | Accuracy statistics from the 4.4M-response training corpus plus curated leaderboard cells. |

Rebuild the Codabench ZIP with `python build_zip.py`.

## Method summary

1. **Hierarchical prior** — mean/count tables for five cell families, combined
   by empirical-Bayes shrinkage (`(m·n + p·α)/(n + α)`, α=40) down the
   hierarchy, with subject ability and item-side difficulty fused additively
   in logit space (Rasch-style).
2. **External curation** — 173 OpenVLM-leaderboard models matched to training
   subjects by normalized name; per-benchmark accuracies converted into
   lookup cells for ~19 benchmarks absent from training. Validated by
   deletion on shared benchmarks (e.g. mathvista 1.17 → 0.75 log-loss).
3. **Adaptive labels** — a regularized Platt rescaling plus a per-benchmark
   offset (shrinkage applied online, α′=0.5, clip ±2.5). On unseen
   benchmarks, five labels are worth more than any offline modeling change
   we tested (0.660 → 0.586 pooled leave-one-benchmark-out log-loss).

A content-based NCF head (sentence embeddings → MLP logit correction,
`variants/model_with_head.py` + `variants/ncf_head.pt`) improves held-out
items of *known* benchmarks (0.519 → 0.462) but transfers negatively to
unseen benchmarks (0.660 → 0.765 pooled LOBO; worse on 14/16) and scored
worse on the live leaderboard twice — including when retrained exclusively
on LOBO-regime examples (0.85). The shipped model therefore omits it.

## Experiments (repo root)

| Script | What it does |
|---|---|
| `train_ncf.py` | Trains the NCF head; known-benchmark cold-start ablations. |
| `lobo_eval.py` | Leave-one-benchmark-out evaluation of prior / head / offsets. |
| `lobo_prior_off.py` | LOBO for prior + label offsets (no head). |
| `lobo_improvements.py` | Label-selection policies (random/entropy/representative) and similarity-blended priors. Both "clever" ideas lose to the simple ones. |
| `lobo_calibration.py` | Sweep of offset strength / clip / clamp. |
| `lobo_alpha_sweep.py` | Joint sweep of prior shrinkage α and offset params. |
| `lobo_novelty_head.py` | Head trained exclusively on LOBO-regime examples (negative result). |
| `lobo_subject_ability.py` | Balanced / modality-split subject ability (no effect). |
| `curate_openvlm.py` | OpenVLM leaderboard curation, with validation-by-deletion (`--build` to emit cells). |
| `curate_openllm.py` | Open LLM Leaderboard curation (validated offline; no live gain). |
| `build_final_tables.py` | Full-data tables + curated cell injection → `submission/lookup_tables.json`. |

## Reproducing

```bash
conda create -n pred-eval python=3.12 pandas numpy torch sentence-transformers huggingface_hub
conda activate pred-eval

./run_experiments.sh                 # everything: curation -> submission ZIP -> all ablations
```

`run_experiments.sh` runs the full pipeline in order (13 stages, ~3 h total;
each stage's output is tee'd to `logs/`). Stages 1--5 (~20 min) produce the
final submission artifacts; stages 6--13 are the experiments behind every
number in the report. Individual stages can be run directly, e.g.:

```bash
python build_final_tables.py         # submission/lookup_tables.json
python train_ncf.py                  # head + known-benchmark ablations (~10 min)
python lobo_eval.py                  # LOBO ablations
```

Training data: `aims-foundations/measurement-db` on HuggingFace (downloaded
automatically). All experiments are seeded (seed 0). Item/subject embeddings
are cached to `emb_cache/` on first run. The OpenVLM leaderboard snapshot is
fetched from `http://opencompass.openxlab.space/assets/OpenVLM.json` into
`curation/`.
