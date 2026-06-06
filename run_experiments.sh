#!/usr/bin/env bash
# Reproduces every experiment in the report, in order.
#
# Prereqs:   conda env with python>=3.11, pandas, numpy, torch,
#            sentence-transformers, huggingface_hub (see README)
# Runtime:   ~2.5-3.5 h total on Apple M-series / comparable GPU box.
#            Stages 1-3 (data + final submission tables) are ~20 min;
#            everything after is ablations and can be run selectively.
# Output:    submission/lookup_tables.json, my_submission.zip,
#            and per-experiment results printed to stdout (tee'd to logs/).

set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
mkdir -p logs curation

run () {
    echo
    echo "=== [$1] $2 ==="
    shift 2
    "$@" 2>&1 | tee "logs/$1.log"
}

# ---------------------------------------------------------------- pipeline --
# 1. External data: OpenVLM leaderboard snapshot + curated cells
if [ ! -f curation/OpenVLM.json ]; then
    echo "=== [0] fetching OpenVLM leaderboard snapshot ==="
    curl -sL "http://opencompass.openxlab.space/assets/OpenVLM.json" \
        -o curation/OpenVLM.json
fi
run 01-curate-vlm    "OpenVLM curation: validation-by-deletion"  $PY curate_openvlm.py
run 02-curate-vlm-b  "OpenVLM curation: build cells"             $PY curate_openvlm.py --build
run 03-curate-llm    "Open LLM Leaderboard curation"             $PY curate_openllm.py

# 2. Final submission artifacts (tables include curated cells)
run 04-tables        "full-data lookup tables"                   $PY build_final_tables.py
run 05-zip           "package Codabench ZIP"                     $PY build_zip.py

# ------------------------------------------------------------- experiments --
# 3. NCF head + known-benchmark cold-start ablations (Table 1, left; Fig 1b)
run 06-train-ncf     "head training + known-benchmark ablations" $PY train_ncf.py

# 4. LOBO ablations (Table 1, right; Fig 1a; Appendix Fig 3)
run 07-lobo          "LOBO: prior / head / offsets"              $PY lobo_eval.py
run 08-lobo-off      "LOBO: prior + offsets (no head)"           $PY lobo_prior_off.py

# 5. Negative results and parameter sweeps (Section 4)
run 09-policies      "label-selection policies + blended priors" $PY lobo_improvements.py
run 10-calibration   "offset strength / clip / clamp sweep"      $PY lobo_calibration.py
run 11-alpha         "prior alpha x offset joint sweep"          $PY lobo_alpha_sweep.py
run 12-novelty-head  "novelty-trained head (negative result)"    $PY lobo_novelty_head.py
run 13-subject       "subject-ability variants (no effect)"      $PY lobo_subject_ability.py

echo
echo "All experiments complete. Results in logs/, submission ZIP at my_submission.zip"
