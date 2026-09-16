from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from train_common import (
    SPORTS,
    MotionFeatureDataset,
    build_samples,
    collect_feature_files,
    experiment_dir,
    load_shape_info,
    resolve_run_id,
    set_seed,
    train_model,
    write_latest_run_id,
)


MODEL_TYPE = "stgcn"
MODEL_NAME = "ST-GCN"
MODEL_SCRIPT = "04-3_train_stgcn.py"
CHECKPOINT_PREFIX = "stgcn_step_sequence"

SKELETON_EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_heel"),
    ("left_heel", "left_toe"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_heel"),
    ("right_heel", "right_toe"),
    ("left_eye", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
]


def build_adjacency(joint_names: list[str]) -> torch.Tensor:
    """Build normalized skeleton adjacency with self loops."""
    joint_index = {name: idx for idx, name in enumerate(joint_names)}
    joint_count = len(joint_names)
    adjacency = torch.eye(joint_count, dtype=torch.float32)
    for left, right in SKELETON_EDGES:
        if left not in joint_index or right not in joint_index:
            continue
        i = joint_index[left]
        j = joint_index[right]
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0

    degree = adjacency.sum(dim=1).clamp_min(1.0)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    return degree_inv_sqrt[:, None] * adjacency * degree_inv_sqrt[None, :]


class GraphTemporalBlock(nn.Module):
    """Simple spatial graph convolution followed by temporal convolution."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.graph_projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.temporal = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1), nn.BatchNorm2d(out_channels))
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        graph = torch.einsum("bctv,vw->bctw", x, adjacency)
        graph = self.graph_projection(graph)
        residual = self.residual(x)
        return F.relu(self.temporal(graph) + residual, inplace=True)


class STGCNEncoder(nn.Module):
    """Spatial-temporal graph convolution encoder."""

    def __init__(
        self,
        input_channels: int,
        joint_names: list[str],
        hidden_dim: int,
        embedding_dim: int,
        num_blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.register_buffer("adjacency", build_adjacency(joint_names), persistent=False)
        channels = [input_channels] + [hidden_dim] * num_blocks
        self.blocks = nn.ModuleList(
            GraphTemporalBlock(channels[idx], channels[idx + 1], dropout=dropout)
            for idx in range(num_blocks)
        )
        self.projection = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2).contiguous()
        adjacency = self.adjacency.to(dtype=x.dtype, device=x.device)
        for block in self.blocks:
            x = block(x, adjacency)
        x = x.mean(dim=(2, 3))
        x = self.projection(x)
        return F.normalize(x, p=2, dim=-1)


class StepSequenceSTGCN(nn.Module):
    """Step-sequence embedding model backed by ST-GCN encoders."""

    def __init__(
        self,
        input_channels: int,
        joint_names: list[str],
        hidden_dim: int,
        embedding_dim: int,
        num_blocks: int,
        dropout: float,
        num_classes: int = 0,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.sequence_encoder = STGCNEncoder(
            input_channels=input_channels,
            joint_names=joint_names,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_blocks=num_blocks,
            dropout=dropout,
        )
        self.step_encoder = STGCNEncoder(
            input_channels=input_channels,
            joint_names=joint_names,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_blocks=num_blocks,
            dropout=dropout,
        )
        self.classifier = nn.Linear(embedding_dim, num_classes) if num_classes > 0 else None

    def encode_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.sequence_encoder(sequence)

    def encode_steps(self, steps: torch.Tensor) -> torch.Tensor:
        batch, step_count, length, joints, channels = steps.shape
        embeddings = self.step_encoder(steps.reshape(batch * step_count, length, joints, channels))
        return embeddings.reshape(batch, step_count, self.embedding_dim)

    def forward(self, sequence: torch.Tensor, steps: torch.Tensor) -> dict[str, torch.Tensor]:
        sequence_embedding = self.encode_sequence(sequence)
        step_embeddings = self.encode_steps(steps)
        step_mean_embedding = F.normalize(step_embeddings.mean(dim=1), p=2, dim=-1)
        sample_embedding = F.normalize((sequence_embedding + step_mean_embedding) / 2.0, p=2, dim=-1)
        outputs = {
            "sequence_embedding": sequence_embedding,
            "step_embeddings": step_embeddings,
            "step_mean_embedding": step_mean_embedding,
            "sample_embedding": sample_embedding,
        }
        if self.classifier is not None:
            outputs["logits"] = self.classifier(sample_embedding)
        return outputs


def model_config_from_args(args: argparse.Namespace, shape_info: dict[str, Any], num_classes: int) -> dict[str, Any]:
    return {
        "input_channels": int(shape_info["sequence_shape"][2]),
        "joint_names": list(shape_info["joint_names"]),
        "hidden_dim": int(args.hidden_dim),
        "embedding_dim": int(args.embedding_dim),
        "num_blocks": int(args.num_blocks),
        "dropout": float(args.dropout),
        "num_classes": int(num_classes),
    }


def build_model_from_config(model_config: dict[str, Any]) -> StepSequenceSTGCN:
    return StepSequenceSTGCN(
        input_channels=int(model_config["input_channels"]),
        joint_names=[str(item) for item in model_config["joint_names"]],
        hidden_dim=int(model_config["hidden_dim"]),
        embedding_dim=int(model_config["embedding_dim"]),
        num_blocks=int(model_config["num_blocks"]),
        dropout=float(model_config["dropout"]),
        num_classes=int(model_config.get("num_classes", 0)),
    )


def build_model_from_checkpoint(checkpoint: dict[str, Any], device: torch.device) -> StepSequenceSTGCN:
    model = build_model_from_config(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ST-GCN step_sequence embedding encoder.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--sport", choices=SPORTS, required=True)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument(
        "--pk-k",
        type=int,
        default=2,
        help=(
            "P*K SupCon batch sampler: K samples per expert per batch (P=batch_size/pk_k experts "
            "per batch), guaranteeing every batch has real positive pairs. Set to 0 to fall back to "
            "plain random shuffling (not recommended)."
        ),
    )
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lambda-sequence", type=float, default=1.0)
    parser.add_argument("--lambda-step", type=float, default=1.0)
    parser.add_argument("--lambda-classification", type=float, default=0.0)
    parser.add_argument(
        "--metric-loss",
        choices=["supcon", "triplet"],
        default="supcon",
        help="Metric-learning objective for sequence/step embeddings (2nd revision instruction 4, Triplet baseline).",
    )
    parser.add_argument("--triplet-margin", type=float, default=0.3, help="Only used when --metric-loss triplet.")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--early-stop-patience", type=int, default=100)
    parser.add_argument(
        "--monitor-metric",
        default="val_total",
        help=(
            "Metric used for best checkpoint selection. "
            "Examples: val_accuracy, val_total, val_classification."
        ),
    )
    parser.add_argument("--monitor-mode", choices=["auto", "min", "max"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", default=None, help="Experiment run folder name. Defaults to YB_RUN_ID or latest experiment.")
    parser.add_argument("--feature-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--no-class-weights",
        action="store_false",
        dest="use_class_weights",
        help="Disable inverse-frequency class weights for ST-GCN classification loss.",
    )
    parser.set_defaults(use_class_weights=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    root = Path(args.root)
    run_id = resolve_run_id(root, args.sport, args.run_id)
    args.run_id = run_id
    write_latest_run_id(root, args.sport, run_id)
    exp_dir = experiment_dir(root, args.sport, run_id)
    output_dir = Path(args.output_dir) if args.output_dir else exp_dir / "models"
    feature_dir = Path(args.feature_dir) if args.feature_dir else exp_dir / "features" / "expert"
    feature_files = collect_feature_files(root, args.sport, feature_dir)

    if not feature_files:
        print(f"No expert feature files found: {feature_dir}")
        print("Run 03_preprocess_and_build_features.py first.")
        return

    samples, label_map = build_samples(feature_files)
    if len(label_map) < 2:
        warnings.warn(
            "Metric learning usually needs at least two experts. "
            "Checkpoint can still be created, but the contrastive loss may be zero.",
            stacklevel=2,
        )

    dataset = MotionFeatureDataset(samples)
    shape_info = load_shape_info(samples[0].path)
    model_config = model_config_from_args(args, shape_info, num_classes=len(label_map))
    model = build_model_from_config(model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_model(
        args=args,
        model=model,
        label_map=label_map,
        shape_info=shape_info,
        dataset=dataset,
        model_type=MODEL_TYPE,
        model_name=MODEL_NAME,
        model_script=MODEL_SCRIPT,
        model_config=model_config,
        checkpoint_prefix=CHECKPOINT_PREFIX,
        curve_title="ST-GCN Training Loss",
        output_dir=output_dir,
        device=device,
    )


if __name__ == "__main__":
    main()
