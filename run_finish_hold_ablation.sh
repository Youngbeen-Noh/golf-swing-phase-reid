#!/usr/bin/env bash
# Finish-hold ablation (paper 7.2 / revision_log.md item 3): isolates whether
# the sequence-level branch's synthetic finish-hold tail segment (see
# 03_preprocess_and_build_features.py's phase_synchronized_resample,
# include_tail) contributes to Seq-only's consistently-lowest performance
# (6.3), separately from phase information itself being useful.
#
# Scope is deliberately narrow -- NOT the full 45-condition grid. Only the
# one condition/option pair that isolates the hypothesis: A-TCN / Golf-D /
# Seq-only, 5 seeds, --no-finish-hold. Compare its Top-1 in the resulting
# pro_holdout_user_metrics.json against the existing ATCN_Golf-D seq-only
# rows already in results_seed_aggregate.csv (the with-hold baseline, mean
# 81.76%, no code changes needed there).
#
# IMPORTANT run_id note: 09_run_golf_experiments.py always names the shared
# base run "ATCN_Golf-D" (derived only from --ATCN + --methods Golf-D --
# --no-finish-hold does NOT change the name, it only changes what's inside
# the features). This script therefore requires data/Golf/experiments to be
# EMPTY before it starts (same precondition run_both_experiments.sh already
# enforces) -- otherwise this run's --no-finish-hold features could silently
# land in / collide with an unrelated existing "ATCN_Golf-D" folder. If you
# still have a normal (with-hold) ATCN_Golf-D run sitting in
# data/Golf/experiments from earlier work, archive or remove it first.
#
# Usage:
#   git pull origin main
#   chmod +x run_finish_hold_ablation.sh
#   nohup ./run_finish_hold_ablation.sh > run_finish_hold_ablation.log 2>&1 &
#   SEEDS="42 43" ./run_finish_hold_ablation.sh    # override the seed list
#
# Step 03 (features) runs once under the shared base run_id; each seed then
# trains against those same features via --run-id-suffix (run_both_experiments.sh's
# shared-features pattern), so seeds differ only in training stochasticity,
# not in data. No step 08 (visualization) -- this is a quantitative-only
# comparison, and result run_ids look like ATCN_Golf-D-seqonly-s42, ...

set -euo pipefail

ARCHIVE_ROOT="${ARCHIVE_ROOT:-$HOME/experiments_archive}"
SEEDS="${SEEDS:-42 43 44 45 46}"

log() {
    printf '\n=== [%s] %s ===\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

if [ -d data/Golf/experiments ] && [ -n "$(ls -A data/Golf/experiments 2>/dev/null)" ]; then
    echo "data/Golf/experiments already has content in it -- clear it before running this" >&2
    echo "script (this run will build a shared base run_id literally named" >&2
    echo "'ATCN_Golf-D', same as a normal non-ablation run, so stale content here" >&2
    echo "would silently mix in). Either:" >&2
    echo "  rm -rf data/Golf/experiments        # if it's stale/leftover, or" >&2
    echo "  mv data/Golf/experiments ~/experiments_archive/<name>   # if it's worth keeping" >&2
    exit 1
fi

log "Step 1/2: building --no-finish-hold features once (A-TCN / Golf-D, cap=16/expert) -> run_id=ATCN_Golf-D"
python 09_run_golf_experiments.py --ATCN --methods Golf-D \
    --max-samples-per-expert 16 --no-finish-hold \
    --start-at 03 --end-at 03

for seed in $SEEDS; do
    log "Step 2/2: seed ${seed} / Seq-only (lambda_step=0, alpha=1.0) -> run_id=ATCN_Golf-D-seqonly-s${seed}"
    python 09_run_golf_experiments.py --ATCN --methods Golf-D \
        --max-samples-per-expert 16 \
        --run-id-suffix="-seqonly-s${seed}" \
        --train-extra "--lambda-step 0 --seed ${seed}" \
        --alpha 1.0 \
        --end-at 07
done

dest="$ARCHIVE_ROOT/golf_finish_hold_ablation"
if [ -e "$dest" ]; then
    echo "Refusing to overwrite existing archive: $dest" >&2
    echo "Results are still under data/Golf/experiments/ -- move them manually." >&2
    exit 1
fi
mkdir -p "$ARCHIVE_ROOT"
mv data/Golf/experiments "$dest"
log "Done. Archived to $dest -- copy back ATCN_Golf-D-seqonly-s{42..46}/results/evaluation/pro_holdout_user_metrics.json."
