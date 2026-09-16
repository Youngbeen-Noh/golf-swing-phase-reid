"""Count trainable parameters per encoder (P1-1 efficiency, 6.4 Results).

Instantiates each of the three StepSequence* models with the exact
hyperparameters documented in the paper's Table 4 (5.4 Implementation
Details) -- no checkpoint needed, since parameter count only depends on
architecture shape, not learned weights. Reports encoder-only params
(sequence_encoder + step_encoder, what retrieval actually uses) separately
from the auxiliary classifier (lambda_cls=0, unused at inference/retrieval
time but structurally present and counted in the checkpoint).

Usage:
  python 14_count_parameters.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from encoder_loader import load_training_module

JOINT_NAMES = [
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_shoulder", "right_shoulder", "left_knee", "right_knee",
    "left_heel", "right_heel", "left_toe", "right_toe",
    "left_eye", "right_eye", "left_elbow", "right_elbow",
    "left_ear", "right_ear",
]
J = len(JOINT_NAMES)
C = 5  # x, y, score, dx, dy

CONFIGS = {
    "A-TCN": {
        "script": "04-1_train_atcn.py",
        "build": lambda mod, num_classes: mod.build_model_from_config({
            "input_dim": J * C, "hidden_dim": 128, "embedding_dim": 128,
            "num_blocks": 4, "dropout": 0.2, "num_classes": num_classes,
        }),
    },
    "Transformer": {
        "script": "04-2_train_transformer.py",
        "build": lambda mod, num_classes: mod.build_model_from_config({
            "input_dim": J * C, "model_dim": 64, "embedding_dim": 128,
            "num_layers": 1, "num_heads": 2, "feedforward_dim": 128,
            "dropout": 0.1, "num_classes": num_classes,
        }),
    },
    "ST-GCN": {
        "script": "04-3_train_stgcn.py",
        "build": lambda mod, num_classes: mod.build_model_from_config({
            "input_channels": C, "joint_names": JOINT_NAMES, "hidden_dim": 64,
            "embedding_dim": 64, "num_blocks": 2, "dropout": 0.1,
            "num_classes": num_classes,
        }),
    },
}


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def main() -> None:
    rows = []
    for name, cfg in CONFIGS.items():
        mod = load_training_module(cfg["script"])
        for experiment, num_classes in [("Experiment 1 (7 experts)", 7), ("Experiment 2 (25 experts)", 25)]:
            model = cfg["build"](mod, num_classes)
            seq_params = count_params(model.sequence_encoder)
            step_params = count_params(model.step_encoder)
            classifier_params = count_params(model.classifier) if model.classifier is not None else 0
            total = count_params(model)
            rows.append({
                "model": name,
                "experiment": experiment,
                "sequence_encoder_params": seq_params,
                "step_encoder_params": step_params,
                "retrieval_params_total": seq_params + step_params,
                "aux_classifier_params_unused_at_retrieval": classifier_params,
                "checkpoint_total_params": total,
            })
            print(f"{name:<12} {experiment:<26} retrieval(seq+step)={seq_params+step_params:>9,}  "
                  f"(seq={seq_params:,}, step={step_params:,})  classifier(unused)={classifier_params:,}  "
                  f"checkpoint_total={total:,}")

    out_path = Path("results_param_counts.csv")
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
