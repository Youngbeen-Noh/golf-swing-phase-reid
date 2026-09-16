#!/usr/bin/env bash
# Runs both Golf experiments (Experiment 2: 25 experts, then Experiment 1: 7-expert
# subset) end to end over multiple seeds (P0-2), archiving results between them so
# run_ids don't collide.
#
# Usage:
#   git pull origin main            # make sure the code is current first
#   chmod +x run_both_experiments.sh
#   nohup ./run_both_experiments.sh > run_both_experiments.log 2>&1 &
#   # or inside a tmux/screen session: ./run_both_experiments.sh
#   SEEDS="42 43" ./run_both_experiments.sh    # override the seed list
#
# Per experiment: features (step 03) are built once per model x method under the
# base run_id, then every (training option x seed) trains against those shared
# features via --run-id-suffix, which skips step 03 and reuses the base features.
# Seeds therefore share identical inputs/splits and differ only in training
# stochasticity (init/batching), which is what P0-2 asks to measure.
# run_ids look like ATCN_Golf-A-s42, ATCN_Golf-A-seqonly-s42, ...
#
# Step 08 visualization runs only for the first seed (best method per model);
# later seeds stop at step 07 to save wall-clock time.
#
# Stops immediately on the first failing command (set -e) rather than silently
# continuing with a broken/partial pipeline.

set -euo pipefail

ARCHIVE_ROOT="${ARCHIVE_ROOT:-$HOME/experiments_archive}"
SEEDS="${SEEDS:-42 43 44 45 46}"
# The 7-expert subset roster (comma-separated data-folder names) is withheld
# from the public release under the anonymization policy of the paper; it is
# available from the corresponding author together with the pose data.
EXPERT7="${EXPERT7:?set EXPERT7 to the 7-expert roster (available on request)}"

log() {
    printf '\n=== [%s] %s ===\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

if [ -d data/Golf/experiments ] && [ -n "$(ls -A data/Golf/experiments 2>/dev/null)" ]; then
    echo "data/Golf/experiments already has content in it -- clear it before running this" >&2
    echo "script, or its files will mix with this run's output (stale expert folders" >&2
    echo "under results/user_similarity can silently corrupt the known-pro holdout counts" >&2
    echo "since both experiments reuse the same run_id names). Either:" >&2
    echo "  rm -rf data/Golf/experiments        # if it's stale/leftover, or" >&2
    echo "  mv data/Golf/experiments ~/experiments_archive/<name>   # if it's worth keeping" >&2
    exit 1
fi

archive() {
    # $1 = destination name under ARCHIVE_ROOT
    local dest="$ARCHIVE_ROOT/$1"
    if [ -e "$dest" ]; then
        echo "Refusing to overwrite existing archive: $dest" >&2
        exit 1
    fi
    mkdir -p "$ARCHIVE_ROOT"
    mv data/Golf/experiments "$dest"
    log "Archived results to $dest"
}

run_experiment() {
    # $1 = human-readable label; remaining args are forwarded to every
    # 09_run_golf_experiments.py call (e.g. --max-samples-per-expert 16,
    # --expert-ids ...).
    local label="$1"
    shift
    local first_seed=1
    local seed

    log "$label: building features (step 03) once for all model x method combos"
    python 09_run_golf_experiments.py --all-models --start-at 03 --end-at 03 "$@"

    for seed in $SEEDS; do
        # Visualize (step 08, best method per model) only for the first seed;
        # later seeds stop after evaluation (step 07).
        local viz_args=(--end-at 07)
        if [ "$first_seed" -eq 1 ]; then
            viz_args=(--visualize-best-only)
        fi

        log "$label: seed ${seed} / Seq+Step"
        python 09_run_golf_experiments.py --all-models \
            --run-id-suffix="-s${seed}" \
            --train-extra "--seed ${seed}" \
            "${viz_args[@]}" "$@"

        log "$label: seed ${seed} / Seq-only"
        python 09_run_golf_experiments.py --all-models \
            --run-id-suffix="-seqonly-s${seed}" \
            --train-extra "--lambda-step 0 --seed ${seed}" \
            --alpha 1.0 \
            "${viz_args[@]}" "$@"

        log "$label: seed ${seed} / Step-only"
        python 09_run_golf_experiments.py --all-models \
            --run-id-suffix="-steponly-s${seed}" \
            --train-extra "--lambda-sequence 0 --seed ${seed}" \
            --alpha 0.0 \
            "${viz_args[@]}" "$@"

        first_seed=0
    done
}

log "Phase 1/4: Experiment 2 (25 experts, max-samples-per-expert=16), seeds: ${SEEDS}"
run_experiment "Experiment 2" --max-samples-per-expert 16

log "Phase 2/4: Archiving Experiment 2 results"
archive golf_experiment_2

log "Phase 3/4: Experiment 1 (7-expert subset, no cap), seeds: ${SEEDS}"
run_experiment "Experiment 1" --expert-ids "$EXPERT7"

log "Phase 4/4: Archiving Experiment 1 results"
archive golf_experiment_1

log "Done. Results under $ARCHIVE_ROOT/golf_experiment_1 and $ARCHIVE_ROOT/golf_experiment_2"
