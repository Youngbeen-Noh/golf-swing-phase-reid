#!/usr/bin/env bash
set -euo pipefail

cd /home/aisl/yb/thesis/yb_thesis
git pull origin main

LIVE_ROOT=experiments_archive/golf_experiment_2
GALLERY_FEATURES=${LIVE_ROOT}/ATCN_Golf-D/features/expert
QUERY_FEATURES=${LIVE_ROOT}/ATCN_Golf-D/features/user
VARIANT_ROOT=${LIVE_ROOT}/prototype_variant_root/data/Golf/experiments

VARIANTS="mean medoid trimmed multiproto"

for SEED in 42 43 44 45 46; do
  SRC_RID=ATCN_Golf-D-s${SEED}
  CHECKPOINT=${LIVE_ROOT}/${SRC_RID}/models/atcn_step_sequence_best.pt
  PROTO_DIR=${LIVE_ROOT}/${SRC_RID}/prototypes/variants

  echo "=== seed=${SEED}: building 4 prototype variants ==="
  python 27_build_prototype_variants.py --sport Golf \
    --checkpoint ${CHECKPOINT} \
    --feature-dir ${GALLERY_FEATURES} \
    --output-dir ${PROTO_DIR}

  for VARIANT in ${VARIANTS}; do
    RUN_TAG=protovariant-${VARIANT}
    OUT_RID=ATCN_Golf-D-${RUN_TAG}-s${SEED}

    echo "=== seed=${SEED} variant=${VARIANT} ==="

    python 06_infer_user_similarity.py --sport Golf --run-id ${OUT_RID} \
      --checkpoint ${CHECKPOINT} \
      --prototype-store ${PROTO_DIR}/expert_prototypes_${VARIANT}.pkl \
      --feature-dir ${QUERY_FEATURES} \
      --output-dir ${VARIANT_ROOT}/${OUT_RID}/results/user_similarity

    python 07_evaluate.py --sport Golf --run-id ${OUT_RID} \
      --checkpoint ${CHECKPOINT} \
      --prototype-store ${PROTO_DIR}/expert_prototypes_${VARIANT}.pkl \
      --feature-dir ${GALLERY_FEATURES} \
      --pro-holdout-result-dir ${VARIANT_ROOT}/${OUT_RID}/results/user_similarity \
      --pro-holdout-output ${VARIANT_ROOT}/${OUT_RID}/results/evaluation/pro_holdout_user_metrics.json
  done
done

python 10_aggregate_seed_metrics.py ${VARIANT_ROOT} --out-prefix results_protovariants
echo "Done building/evaluating. See results_protovariants_*.csv"

echo "=== copying results into live tree for significance testing ==="
for SEED in 42 43 44 45 46; do
  for VARIANT in ${VARIANTS}; do
    OUT_RID=ATCN_Golf-D-protovariant-${VARIANT}-s${SEED}
    mkdir -p ${LIVE_ROOT}/${OUT_RID}/results/evaluation
    cp ${VARIANT_ROOT}/${OUT_RID}/results/evaluation/pro_holdout_user_metrics.json \
       ${LIVE_ROOT}/${OUT_RID}/results/evaluation/pro_holdout_user_metrics.json
  done
done

python 19_paired_significance_hierarchical.py ${LIVE_ROOT} \
  --compare "ATCN_Golf-D_seq+step vs ATCN_Golf-D-protovariant-medoid_seq+step" \
  --compare "ATCN_Golf-D_seq+step vs ATCN_Golf-D-protovariant-trimmed_seq+step" \
  --compare "ATCN_Golf-D_seq+step vs ATCN_Golf-D-protovariant-multiproto_seq+step"
echo "Done. See results_paired_significance_hierarchical.csv"
