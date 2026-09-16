#!/usr/bin/env bash
set -euo pipefail

cd /home/aisl/yb/thesis/yb_thesis
git pull origin main

PERTURB_ROOT=/home/aisl/experiments_archive/golf_experiment_2/event_perturbation_root

for MAG in 5 10 20; do
  for SEED in 42 43 44 45 46; do
    COMBO_ROOT=${PERTURB_ROOT}/mag${MAG}_s${SEED}
    RUN_TAG=eventperturb-mag${MAG}
    SRC_RID=ATCN_Golf-D-s${SEED}
    OUT_RID=ATCN_Golf-D-${RUN_TAG}-s${SEED}

    echo "=== magnitude=${MAG} seed=${SEED} ==="

    python 26_apply_event_perturbation.py \
      --json-dir data/Golf/jsons/user \
      --magnitude ${MAG} --seed ${SEED} --out-root ${COMBO_ROOT}

    python 02_json_to_tensor.py --root ${COMBO_ROOT} --sport Golf --subject-type user

    python 03_preprocess_and_build_features.py --root ${COMBO_ROOT} --sport Golf \
      --preprocess-method Golf-D --subject-type user --run-id ${RUN_TAG} \
      --output-root ${COMBO_ROOT}/data/Golf/experiments/${RUN_TAG}

    python 06_infer_user_similarity.py --sport Golf --run-id ${OUT_RID} \
      --checkpoint /home/aisl/experiments_archive/golf_experiment_2/${SRC_RID}/models/atcn_step_sequence_best.pt \
      --prototype-store /home/aisl/experiments_archive/golf_experiment_2/${SRC_RID}/prototypes/expert_prototypes.pkl \
      --feature-dir ${COMBO_ROOT}/data/Golf/experiments/${RUN_TAG}/features/user \
      --output-dir ${PERTURB_ROOT}/data/Golf/experiments/${OUT_RID}/results/user_similarity

    python 07_evaluate.py --sport Golf --run-id ${OUT_RID} \
      --checkpoint /home/aisl/experiments_archive/golf_experiment_2/${SRC_RID}/models/atcn_step_sequence_best.pt \
      --prototype-store /home/aisl/experiments_archive/golf_experiment_2/${SRC_RID}/prototypes/expert_prototypes.pkl \
      --feature-dir /home/aisl/experiments_archive/golf_experiment_2/ATCN_Golf-D/features/expert \
      --pro-holdout-result-dir ${PERTURB_ROOT}/data/Golf/experiments/${OUT_RID}/results/user_similarity \
      --pro-holdout-output ${PERTURB_ROOT}/data/Golf/experiments/${OUT_RID}/results/evaluation/pro_holdout_user_metrics.json
  done
done

python 10_aggregate_seed_metrics.py ${PERTURB_ROOT}/data/Golf/experiments --out-prefix results_eventperturb
echo "Done. See results_eventperturb_*.csv"
