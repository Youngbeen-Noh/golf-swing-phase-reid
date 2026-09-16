#!/usr/bin/env bash
set -euo pipefail

cd /home/aisl/yb/thesis/yb_thesis
git pull origin main

LIVE_ROOT=experiments_archive/golf_experiment_2
GALLERY_FEATURES=${LIVE_ROOT}/ATCN_Golf-D/features/expert
QUERY_FEATURES=${LIVE_ROOT}/ATCN_Golf-D/features/user

# Capacity-matched hyperparameters, found by a local (CPU-only, no data
# needed) grid search over each architecture's own parameter-counting
# constructor, targeting A-TCN's retrieval_params_total (848,896, Table S4,
# Experiment 2) -- see 28_search_capacity_matched_hparams.py. Both land
# within 0.2% of the target:
#   Transformer: model_dim=160, num_layers=2, num_heads=2, feedforward_dim=160 -> 847,016 (+0.2%)
#   ST-GCN:      hidden_dim=240, num_blocks=2                                  -> 849,248 (+0.0%)
# A-TCN itself is unchanged (it already IS the capacity target); only
# Transformer and ST-GCN are retrained here.

for SEED in 42 43 44 45 46; do
  echo "=== seed=${SEED}: Transformer (equal-capacity) ==="
  T_RID=Transformer_Golf-D-equalcap-s${SEED}
  python 04-2_train_transformer.py --sport Golf --seed ${SEED} --run-id ${T_RID} \
    --feature-dir ${GALLERY_FEATURES} \
    --output-dir ${LIVE_ROOT}/${T_RID}/models \
    --model-dim 160 --num-layers 2 --num-heads 2 --feedforward-dim 160 \
    --lambda-sequence 1.0 --lambda-step 1.0 --lambda-classification 0.0

  python 05_build_expert_prototypes.py --sport Golf --run-id ${T_RID} \
    --checkpoint ${LIVE_ROOT}/${T_RID}/models/transformer_step_sequence_best.pt \
    --feature-dir ${GALLERY_FEATURES} \
    --output ${LIVE_ROOT}/${T_RID}/prototypes/expert_prototypes.pkl

  python 06_infer_user_similarity.py --sport Golf --run-id ${T_RID} \
    --checkpoint ${LIVE_ROOT}/${T_RID}/models/transformer_step_sequence_best.pt \
    --prototype-store ${LIVE_ROOT}/${T_RID}/prototypes/expert_prototypes.pkl \
    --feature-dir ${QUERY_FEATURES} \
    --output-dir ${LIVE_ROOT}/${T_RID}/results/user_similarity \
    --alpha 0.5

  python 07_evaluate.py --sport Golf --run-id ${T_RID} \
    --checkpoint ${LIVE_ROOT}/${T_RID}/models/transformer_step_sequence_best.pt \
    --prototype-store ${LIVE_ROOT}/${T_RID}/prototypes/expert_prototypes.pkl \
    --feature-dir ${GALLERY_FEATURES} \
    --pro-holdout-result-dir ${LIVE_ROOT}/${T_RID}/results/user_similarity \
    --pro-holdout-output ${LIVE_ROOT}/${T_RID}/results/evaluation/pro_holdout_user_metrics.json

  echo "=== seed=${SEED}: ST-GCN (equal-capacity) ==="
  S_RID=STGCN_Golf-D-equalcap-s${SEED}
  python 04-3_train_stgcn.py --sport Golf --seed ${SEED} --run-id ${S_RID} \
    --feature-dir ${GALLERY_FEATURES} \
    --output-dir ${LIVE_ROOT}/${S_RID}/models \
    --hidden-dim 240 --num-blocks 2 \
    --lambda-sequence 1.0 --lambda-step 1.0 --lambda-classification 0.0

  python 05_build_expert_prototypes.py --sport Golf --run-id ${S_RID} \
    --checkpoint ${LIVE_ROOT}/${S_RID}/models/stgcn_step_sequence_best.pt \
    --feature-dir ${GALLERY_FEATURES} \
    --output ${LIVE_ROOT}/${S_RID}/prototypes/expert_prototypes.pkl

  python 06_infer_user_similarity.py --sport Golf --run-id ${S_RID} \
    --checkpoint ${LIVE_ROOT}/${S_RID}/models/stgcn_step_sequence_best.pt \
    --prototype-store ${LIVE_ROOT}/${S_RID}/prototypes/expert_prototypes.pkl \
    --feature-dir ${QUERY_FEATURES} \
    --output-dir ${LIVE_ROOT}/${S_RID}/results/user_similarity \
    --alpha 0.5

  python 07_evaluate.py --sport Golf --run-id ${S_RID} \
    --checkpoint ${LIVE_ROOT}/${S_RID}/models/stgcn_step_sequence_best.pt \
    --prototype-store ${LIVE_ROOT}/${S_RID}/prototypes/expert_prototypes.pkl \
    --feature-dir ${GALLERY_FEATURES} \
    --pro-holdout-result-dir ${LIVE_ROOT}/${S_RID}/results/user_similarity \
    --pro-holdout-output ${LIVE_ROOT}/${S_RID}/results/evaluation/pro_holdout_user_metrics.json
done

echo "=== aggregating ==="
python 10_aggregate_seed_metrics.py ${LIVE_ROOT} --out-prefix results_equalcapacity

echo "=== significance vs each architecture's own original (smaller-capacity) result, and vs A-TCN ==="
python 19_paired_significance_hierarchical.py ${LIVE_ROOT} \
  --compare "Transformer_Golf-D_seq+step vs Transformer_Golf-D-equalcap_seq+step" \
  --compare "STGCN_Golf-D_seq+step vs STGCN_Golf-D-equalcap_seq+step" \
  --compare "ATCN_Golf-D_seq+step vs Transformer_Golf-D-equalcap_seq+step" \
  --compare "ATCN_Golf-D_seq+step vs STGCN_Golf-D-equalcap_seq+step"

echo "Done. See results_equalcapacity_*.csv and results_paired_significance_hierarchical.csv"
