"""Equal-capacity hyperparameter search (2nd revision instructions, "recommended"
tier, "Architecture와 capacity 분리" -- separate architecture effect from
parameter-count effect): this paper's three encoders have very different
parameter counts under their default hyperparameters (Table S4, retrieval_
params_total, Experiment 2): A-TCN 848,896, Transformer 120,936, ST-GCN
68,864 -- roughly a 7-12x spread. Any accuracy gap between them could
therefore be capacity, not architecture.

This is a pure hyperparameter-counting search -- CPU-only, no data or GPU
needed, no training -- that instantiates each architecture's own model class
directly (importing 04-2_train_transformer.py / 04-3_train_stgcn.py as
modules) across a hyperparameter grid and reports the closest matches to
A-TCN's parameter count, so Transformer/ST-GCN can be retrained once at a
capacity-matched configuration and compared against both their own original
(smaller-capacity) result and A-TCN's.

Usage:
  python 28_search_capacity_matched_hparams.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent

TARGET = 848_896  # A-TCN retrieval_params_total, Experiment 2 (Table S4)
J, C = 18, 5
INPUT_DIM = J * C

JOINT_NAMES = [
    "left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_elbow", "left_wrist",
    "right_elbow", "right_wrist", "left_knee", "left_heel", "left_toe", "right_knee",
    "right_heel", "right_toe", "left_eye", "right_eye", "left_ear", "right_ear",
]
assert len(JOINT_NAMES) == J


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # needed for dataclass field resolution under `from __future__ import annotations`
    spec.loader.exec_module(module)
    return module


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def transformer_params(
    train_transformer, model_dim: int, num_layers: int, num_heads: int, feedforward_dim: int,
    embedding_dim: int = 128, dropout: float = 0.07,
) -> int | None:
    if model_dim % num_heads != 0:
        return None
    model = train_transformer.StepSequenceTransformer(
        input_dim=INPUT_DIM, model_dim=model_dim, embedding_dim=embedding_dim,
        num_layers=num_layers, num_heads=num_heads, feedforward_dim=feedforward_dim,
        dropout=dropout, num_classes=0,
    )
    return count_params(model)


def stgcn_params(train_stgcn, hidden_dim: int, num_blocks: int, embedding_dim: int = 64, dropout: float = 0.2) -> int:
    model = train_stgcn.StepSequenceSTGCN(
        input_channels=C, joint_names=JOINT_NAMES, hidden_dim=hidden_dim,
        embedding_dim=embedding_dim, num_blocks=num_blocks, dropout=dropout, num_classes=0,
    )
    return count_params(model)


def main() -> None:
    train_transformer = _load_module("train_transformer_search", "04-2_train_transformer.py")
    train_stgcn = _load_module("train_stgcn_search", "04-3_train_stgcn.py")

    print(f"Target (A-TCN retrieval_params_total, Experiment 2): {TARGET:,}")
    print(
        "Current Transformer (model_dim=64, layers=1, heads=2, ff=128): "
        f"{transformer_params(train_transformer, 64, 1, 2, 128):,}"
    )
    print(f"Current ST-GCN (hidden_dim=64, blocks=2): {stgcn_params(train_stgcn, 64, 2):,}\n")

    print("=== Transformer capacity search (model_dim, num_layers, num_heads, feedforward_dim) ===")
    results_t = []
    for model_dim in range(64, 385, 8):
        for num_heads in (1, 2, 4, 8):
            if model_dim % num_heads != 0:
                continue
            for num_layers in (1, 2, 3, 4):
                for feedforward_dim in range(model_dim, model_dim * 4 + 1, 32):
                    p = transformer_params(train_transformer, model_dim, num_layers, num_heads, feedforward_dim)
                    if p is None:
                        continue
                    err = abs(p - TARGET) / TARGET
                    results_t.append((err, p, model_dim, num_layers, num_heads, feedforward_dim))
    results_t.sort(key=lambda r: r[0])
    print("Top 5 closest matches:")
    for err, p, model_dim, num_layers, num_heads, feedforward_dim in results_t[:5]:
        print(
            f"  model_dim={model_dim}, num_layers={num_layers}, num_heads={num_heads}, "
            f"feedforward_dim={feedforward_dim} -> {p:,} params ({err:+.1%} vs target)"
        )

    print("\n=== ST-GCN capacity search (hidden_dim, num_blocks) ===")
    results_s = []
    for hidden_dim in range(48, 513, 8):
        for num_blocks in (2, 3, 4, 5, 6):
            p = stgcn_params(train_stgcn, hidden_dim, num_blocks)
            err = abs(p - TARGET) / TARGET
            results_s.append((err, p, hidden_dim, num_blocks))
    results_s.sort(key=lambda r: r[0])
    print("Top 5 closest matches:")
    for err, p, hidden_dim, num_blocks in results_s[:5]:
        print(f"  hidden_dim={hidden_dim}, num_blocks={num_blocks} -> {p:,} params ({err:+.1%} vs target)")


if __name__ == "__main__":
    main()
