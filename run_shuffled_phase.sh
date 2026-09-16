#!/usr/bin/env bash
set -euo pipefail

cd /home/aisl/yb/thesis/yb_thesis
git pull origin main

LIVE_ROOT=experiments_archive/golf_experiment_2
SHUFFLE_ROOT=${LIVE_ROOT}/shuffled_phase_root

# Fixed permutation for the WHOLE ablation (one shuffle-seed shared by every
# training/query sample and every one of the 5 model seeds below) -- per
# --shuffle-phase-order's own contract: "deterministic per --shuffle-seed,
# not per sample" (03_preprocess_and_build_features.py). Compares directly
# against the already-reported Sequence-only-140 (91.3 +- 2.1%): same 140
# frames, same per-segment resampling, same no-finish-hold -- only the order
# of the 7 concatenated segments differs.
SHUFFLE_SEED=7

echo "=== building shuffled-phase features (expert + user, shuffle_seed=${SHUFFLE_SEED}) ==="
# --output-root, when given explicitly, is used AS-IS as the direct parent of
# features/{expert,user}/... (03_preprocess_and_build_features.py's
# `base_output = Path(args.output_root) if args.output_root else
# experiment_dir(...)` -- NOT auto-nested under data/Golf/experiments/{run_id}
# the way the default resolution is). An earlier version of this script
# assumed the nested form and every downstream step referenced the wrong
# path, so training/prototypes/inference all failed with
# "No expert feature files found" / "Checkpoint not found" against paths that
# were never written. Safe to re-run from scratch: already-built feature
# files are skipped (not overwritten) since --overwrite is not passed.
python 03_preprocess_and_build_features.py --sport Golf --subject-type all \
  --preprocess-method Golf-D \
  --sequence-mode phase-concat --sequence-length-golf 140 --step-length-golf 20 \
  --no-finish-hold --shuffle-phase-order --shuffle-seed ${SHUFFLE_SEED} \
  --run-id shuffledphase --output-root ${SHUFFLE_ROOT}

for SEED in 42 43 44 45 46; do
  RUN_TAG=shuffledphase
  OUT_RID=ATCN_Golf-D-${RUN_TAG}-s${SEED}

  echo "=== seed=${SEED}: training (sequence-only, lambda_step=0) ==="
  # --output-dir, when given explicitly, is used AS-IS as the checkpoint
  # directory (train_common.experiment_dir()'s "exp_dir/models" auto-append
  # only applies when --output-dir is omitted) -- so "models" must be
  # appended here explicitly to match every other script's expectation of
  # <run>/models/atcn_step_sequence_best.pt.
  python 04-1_train_atcn.py --sport Golf --seed ${SEED} --run-id ${OUT_RID} \
    --feature-dir ${SHUFFLE_ROOT}/features/expert \
    --output-dir ${LIVE_ROOT}/${OUT_RID}/models \
    --lambda-sequence 1.0 --lambda-step 0.0 --lambda-classification 0.0

  echo "=== seed=${SEED}: building expert prototypes ==="
  python 05_build_expert_prototypes.py --sport Golf --run-id ${OUT_RID} \
    --checkpoint ${LIVE_ROOT}/${OUT_RID}/models/atcn_step_sequence_best.pt \
    --feature-dir ${SHUFFLE_ROOT}/features/expert \
    --output ${LIVE_ROOT}/${OUT_RID}/prototypes/expert_prototypes.pkl

  echo "=== seed=${SEED}: inference + evaluation ==="
  python 06_infer_user_similarity.py --sport Golf --run-id ${OUT_RID} \
    --checkpoint ${LIVE_ROOT}/${OUT_RID}/models/atcn_step_sequence_best.pt \
    --prototype-store ${LIVE_ROOT}/${OUT_RID}/prototypes/expert_prototypes.pkl \
    --feature-dir ${SHUFFLE_ROOT}/features/user \
    --output-dir ${LIVE_ROOT}/${OUT_RID}/results/user_similarity \
    --alpha 1.0

  python 07_evaluate.py --sport Golf --run-id ${OUT_RID} \
    --checkpoint ${LIVE_ROOT}/${OUT_RID}/models/atcn_step_sequence_best.pt \
    --prototype-store ${LIVE_ROOT}/${OUT_RID}/prototypes/expert_prototypes.pkl \
    --feature-dir ${SHUFFLE_ROOT}/features/expert \
    --pro-holdout-result-dir ${LIVE_ROOT}/${OUT_RID}/results/user_similarity \
    --pro-holdout-output ${LIVE_ROOT}/${OUT_RID}/results/evaluation/pro_holdout_user_metrics.json
done

echo "=== aggregating ==="
python 10_aggregate_seed_metrics.py ${LIVE_ROOT} --out-prefix results_shuffledphase

echo "=== significance vs Sequence-only-140 and vs the full baseline ==="
python 19_paired_significance_hierarchical.py ${LIVE_ROOT} \
  --compare "ATCN_Golf-D_seq+step vs ATCN_Golf-D-shuffledphase_seq+step" \
  --compare "ATCN_Golf-D-seq140_seq+step vs ATCN_Golf-D-shuffledphase_seq+step"

echo "Done. See results_shuffledphase_*.csv and results_paired_significance_hierarchical.csv"
