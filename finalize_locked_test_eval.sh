#!/usr/bin/env bash
# Locked-test confirmatory evaluation (2nd revision instruction 2.1).
#
# Run this ONCE, after the validation sweep (run_both_experiments.sh) is
# done and analyzed (10_aggregate_seed_metrics.py / 11 or
# 19_paired_significance_hierarchical.py / 13_factorial_anova.py or
# 20_query_level_logistic_variance.py), to pick the single winning
# condition. This script does NOT pick the winner for you -- fill in
# WINNING_RUN_ID_PREFIX below first.
#
# What it does: temporarily swaps data/Golf/jsons/user (currently the
# validation set) for data/Golf/jsons/user_locked_test (never touched by
# the validation sweep), rebuilds only the tensor/user cache to match,
# re-runs inference+evaluation (NOT training -- reuses the winning
# condition's already-trained checkpoints) for that one condition across
# all 5 seeds, then restores the validation state so nothing else is
# disturbed.
#
# Usage:
#   ./finalize_locked_test_eval.sh
#
set -euo pipefail

# --- EDIT THIS after picking the winner from validation results ---
# Must match a run_id already trained during the validation sweep, e.g.
# "ATCN_Golf-D" (05-08 will look for {WINNING_RUN_ID_PREFIX}-s{SEED}).
WINNING_RUN_ID_PREFIX="ATCN_Golf-D"
SEEDS="42 43 44 45 46"
PY="${PY:-python}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

JSON_USER="data/Golf/jsons/user"
JSON_LOCKED="data/Golf/jsons/user_locked_test"
JSON_USER_BAK="data/Golf/jsons/user_validation_bak"
TENSOR_USER="data/Golf/tensor/user"
TENSOR_USER_BAK="data/Golf/tensor/user_validation_bak"

if [ ! -d "$JSON_LOCKED" ]; then
  echo "ERROR: $JSON_LOCKED not found. Run 01-3c_validation_locked_test_split.py first." >&2
  exit 1
fi
if [ -e "$JSON_USER_BAK" ] || [ -e "$TENSOR_USER_BAK" ]; then
  echo "ERROR: backup dirs already exist ($JSON_USER_BAK / $TENSOR_USER_BAK)." \
       "A previous run may have been interrupted before restoring -- check manually." >&2
  exit 1
fi

restore() {
  echo "Restoring validation state..."
  if [ -d "$JSON_USER_BAK" ]; then
    rm -rf "$JSON_USER"
    mv "$JSON_USER_BAK" "$JSON_USER"
  fi
  if [ -d "$TENSOR_USER_BAK" ]; then
    rm -rf "$TENSOR_USER"
    mv "$TENSOR_USER_BAK" "$TENSOR_USER"
  fi
  echo "Restored."
}
trap restore EXIT

echo "Swapping in locked-test set..."
mv "$JSON_USER" "$JSON_USER_BAK"
cp -r "$JSON_LOCKED" "$JSON_USER"
mv "$TENSOR_USER" "$TENSOR_USER_BAK"
"$PY" 02_json_to_tensor.py --sport Golf --subject-type user

BASE_EXPERT_FEATURE_DIR="data/Golf/experiments/${WINNING_RUN_ID_PREFIX}/features/expert"
if [ ! -d "$BASE_EXPERT_FEATURE_DIR" ]; then
  echo "ERROR: $BASE_EXPERT_FEATURE_DIR not found -- 07_evaluate.py needs it for the" >&2
  echo "expert-side retrieval metrics (shared base features, same convention as" >&2
  echo "09_run_golf_experiments.py). Point BASE_EXPERT_FEATURE_DIR at wherever the" >&2
  echo "winning condition's expert features actually live and rerun." >&2
  exit 1
fi

echo "Running locked-test inference + evaluation for $WINNING_RUN_ID_PREFIX (seeds: $SEEDS)..."
for SEED in $SEEDS; do
  RUN_ID="${WINNING_RUN_ID_PREFIX}-s${SEED}"
  OUT_JSON="data/Golf/experiments/${RUN_ID}/results/evaluation/pro_holdout_user_metrics_LOCKED_TEST.json"
  echo "--- $RUN_ID ---"
  # This run_id's results/user_similarity/ almost certainly still has this
  # seed's VALIDATION-set query results in it (different sample_ids than the
  # locked-test set, so 06 below would never overwrite them by filename --
  # they'd just sit there and get counted alongside the fresh locked-test
  # ones). Same staleness class as repair_stale_user_holdout.py -- wipe it
  # first so only this run's locked-test results exist.
  rm -rf "data/Golf/experiments/${RUN_ID}/results/user_similarity"
  rm -rf "data/Golf/experiments/${RUN_ID}/results/evaluation"
  "$PY" 03_preprocess_and_build_features.py --sport Golf --preprocess-method "${WINNING_RUN_ID_PREFIX#*_}" \
    --run-id "$RUN_ID" --subject-type user --overwrite
  "$PY" 06_infer_user_similarity.py --sport Golf --run-id "$RUN_ID"
  "$PY" 07_evaluate.py --sport Golf --run-id "$RUN_ID" --require-pro-holdout \
    --feature-dir "$BASE_EXPERT_FEATURE_DIR" \
    --pro-holdout-output "$OUT_JSON"
  # 07_evaluate.py can hit an early `return` (exit 0, NOT an error) on a
  # missing/misconfigured input without ever reaching the pro-holdout step --
  # set -e will not catch that. Verify the file actually landed.
  if [ ! -f "$OUT_JSON" ]; then
    echo "ERROR: $OUT_JSON was not created -- 07_evaluate.py exited early without" >&2
    echo "running the pro-holdout evaluation (check its printed output above for why)." >&2
    exit 1
  fi
  # Verify the CONTENT is right, not just that a file exists: found_reports
  # must match the locked-test set exactly (52 as of writing), never the
  # stale 91-ish validation-set count from a leftover-file mixup.
  EXPECTED_COUNT=$(find "$JSON_LOCKED" -iname "*.json" | wc -l)
  ACTUAL_COUNT=$("$PY" -c "import json,sys; print(int(json.load(open(sys.argv[1]))['summary']['found_reports']))" "$OUT_JSON")
  if [ "$ACTUAL_COUNT" != "$EXPECTED_COUNT" ]; then
    echo "ERROR: $OUT_JSON has count=$ACTUAL_COUNT but the locked-test set has" >&2
    echo "$EXPECTED_COUNT files -- results/user_similarity for this run_id is" >&2
    echo "contaminated with leftover files from a different split. Investigate" >&2
    echo "before trusting this number." >&2
    exit 1
  fi
done

echo ""
echo "Done. Locked-test results saved as pro_holdout_user_metrics_LOCKED_TEST.json"
echo "under each seed's results/evaluation/ dir (validation results, if you want to"
echo "keep comparing, are still pro_holdout_user_metrics.json from the earlier sweep)."
echo "Aggregate with: python 10_aggregate_seed_metrics.py ... (point at the LOCKED_TEST files)"
