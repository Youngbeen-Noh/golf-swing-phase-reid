"""Direct classification-head evaluation for the CE classifier baseline
(2nd revision instruction 4, "Cross-entropy classifier" / "metric learning
자체의 효과 분리").

A checkpoint trained with --lambda-classification 1.0 --lambda-sequence 0
--lambda-step 0 (see train_common.py / 04-1..3_train_*.py) has a working
softmax classifier head over the closed expert set. This script evaluates
that head DIRECTLY (argmax over logits on each holdout query), bypassing
the nearest-prototype retrieval pipeline entirely -- this is what isolates
"does metric learning / prototype retrieval add anything over a plain
classifier" (the CE+nearest-prototype baseline, by contrast, is just the
SAME checkpoint run through the existing 05_build_expert_prototypes.py ->
06_infer_user_similarity.py -> 07_evaluate.py pipeline unchanged, since that
pipeline only ever looks at the embedding, never the classifier head).

Output is pro_holdout_user_metrics.json-compatible ({"summary":...,
"rows": [{"expert_id", "sample_id", "final_rank"}, ...]}), so it plugs
directly into 10_aggregate_seed_metrics.py, 11_paired_significance.py,
19_paired_significance_hierarchical.py, and 20_query_level_logistic_variance.py
without any changes to those scripts. final_rank here means "rank of the
true expert in the softmax output, sorted by probability descending" --
1 if the argmax prediction is correct.

Usage:
  python 21_evaluate_ce_classifier.py --checkpoint data/Golf/experiments/ATCN_Golf-D-CE-s42/models/atcn_step_sequence_best.pt \
      --user-feature-dir data/Golf/experiments/ATCN_Golf-D-CE-s42/features/user \
      --out data/Golf/experiments/ATCN_Golf-D-CE-s42/results/evaluation/pro_holdout_user_metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from encoder_loader import build_model_from_checkpoint
from train_common import infer_expert_id, scalar


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_feature(path: Path) -> tuple[torch.Tensor, torch.Tensor, str]:
    with np.load(path, allow_pickle=False) as data:
        sequence = torch.from_numpy(data["sequence"].astype(np.float32))
        steps = torch.from_numpy(data["steps"].astype(np.float32))
        sample_id = str(scalar(data["sample_id"]))
    return sequence, steps, sample_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--user-feature-dir", required=True, help="data/{Sport}/experiments/{run_id}/features/user")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(Path(args.checkpoint))
    label_map = checkpoint.get("label_map")
    if not label_map:
        raise SystemExit(
            "Checkpoint has no label_map -- was it trained with a classifier head "
            "(--lambda-classification > 0, num_classes > 0)?"
        )
    index_to_expert = {idx: expert_id for expert_id, idx in label_map.items()}

    model = build_model_from_checkpoint(checkpoint, device)
    model.eval()

    user_dir = Path(args.user_feature_dir)
    feature_paths = sorted(user_dir.rglob("*.npz"))
    if not feature_paths:
        raise SystemExit(f"No feature files found under {user_dir}")

    rows: list[dict[str, Any]] = []
    correct = 0
    for feature_path in feature_paths:
        expert_id = feature_path.parent.name
        sequence, steps, sample_id = load_feature(feature_path)
        with torch.no_grad():
            outputs = model(sequence.unsqueeze(0).to(device), steps.unsqueeze(0).to(device))
        if "logits" not in outputs:
            raise SystemExit("Model has no classifier head (outputs['logits'] missing).")
        probs = torch.softmax(outputs["logits"][0], dim=0).cpu().numpy()
        ranked_indices = np.argsort(-probs)
        ranked_experts = [index_to_expert.get(int(i), f"<unknown:{i}>") for i in ranked_indices]
        if expert_id in ranked_experts:
            final_rank = ranked_experts.index(expert_id) + 1
        else:
            # true expert wasn't in this checkpoint's training label set (open-set query) --
            # not expected for the closed-set known-pro holdout, but guard against it anyway.
            final_rank = len(ranked_experts) + 1
        if final_rank == 1:
            correct += 1
        rows.append(
            {
                "expert_id": expert_id,
                "sample_id": sample_id,
                "final_rank": final_rank,
                "final_top1": ranked_experts[0] if ranked_experts else "",
            }
        )

    top1_accuracy = correct / len(rows)
    mrr = float(np.mean([1.0 / r["final_rank"] for r in rows]))
    # Nest under "final_ranking" to match 07_evaluate.py's pro_holdout schema --
    # 10_aggregate_seed_metrics.py's load_run_metrics() reads
    # summary["final_ranking"]["count"/"top1_accuracy"/"mrr"/"mean_rank"]
    # specifically, not top-level summary fields, despite this script's
    # docstring claiming direct compatibility. A flat summary here silently
    # produces "No usable metrics json" for every run (final_ranking
    # missing -> final.get("count") is None) with no other symptom.
    final_ranking_block = {
        "count": len(rows),
        "top1_accuracy": top1_accuracy,
        "mrr": mrr,
        "mean_rank": float(np.mean([r["final_rank"] for r in rows])),
    }
    summary = {
        **final_ranking_block,
        "final_ranking": final_ranking_block,
        "evaluation_mode": "ce_classifier_direct_argmax",
    }
    payload = {"summary": summary, "rows": rows}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
