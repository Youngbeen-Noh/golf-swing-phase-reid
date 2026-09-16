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


MODEL_TYPE = "atcn"
MODEL_NAME = "A-TCN"
MODEL_SCRIPT = "04-1_train_atcn.py"
CHECKPOINT_PREFIX = "atcn_step_sequence"


class DilatedConvBlock(nn.Module):
    """Residual dilated temporal convolution block.

    Input and output shape: (B, C, T)
    """

    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run residual block."""
        return self.activation(x + self.net(x))


class ATCNEncoder(nn.Module):
    """A-TCN encoder.

    Input:
        x: shape (B, T, D)

    Output:
        L2-normalized embedding shape (B, E)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        num_blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [DilatedConvBlock(hidden_dim, dilation=2**idx, dropout=dropout) for idx in range(num_blocks)]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode temporal features."""
        x = self.input_projection(x)
        x = x.transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = self.pool(x).squeeze(-1)
        x = self.projection(x)
        return F.normalize(x, p=2, dim=-1)


class StepSequenceATCN(nn.Module):
    """Step-sequence embedding model with independent sequence and step encoders."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        num_blocks: int,
        dropout: float,
        num_classes: int = 0,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.sequence_encoder = ATCNEncoder(input_dim, hidden_dim, embedding_dim, num_blocks, dropout)
        self.step_encoder = ATCNEncoder(input_dim, hidden_dim, embedding_dim, num_blocks, dropout)
        self.classifier = nn.Linear(embedding_dim, num_classes) if num_classes > 0 else None

    def encode_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        """Encode sequence shape (B, L, J, C) into (B, E)."""
        batch, length, joints, channels = sequence.shape
        return self.sequence_encoder(sequence.reshape(batch, length, joints * channels))

    def encode_steps(self, steps: torch.Tensor) -> torch.Tensor:
        """Encode steps shape (B, S, Ls, J, C) into (B, S, E)."""
        batch, step_count, length, joints, channels = steps.shape
        embeddings = self.step_encoder(steps.reshape(batch * step_count, length, joints * channels))
        return embeddings.reshape(batch, step_count, self.embedding_dim)

    def forward(self, sequence: torch.Tensor, steps: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return sequence, step, and combined sample embeddings."""
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
        "input_dim": int(shape_info["sequence_shape"][1] * shape_info["sequence_shape"][2]),
        "hidden_dim": int(args.hidden_dim),
        "embedding_dim": int(args.embedding_dim),
        "num_blocks": int(args.num_blocks),
        "dropout": float(args.dropout),
        "num_classes": int(num_classes),
    }


def build_model_from_config(model_config: dict[str, Any]) -> StepSequenceATCN:
    return StepSequenceATCN(
        input_dim=int(model_config["input_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        embedding_dim=int(model_config["embedding_dim"]),
        num_blocks=int(model_config["num_blocks"]),
        dropout=float(model_config["dropout"]),
        num_classes=int(model_config.get("num_classes", 0)),
    )


def build_model_from_checkpoint(checkpoint: dict[str, Any], device: torch.device) -> StepSequenceATCN:
    """Recreate an A-TCN model from checkpoint metadata."""
    model = build_model_from_config(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train A-TCN step_sequence embedding encoder.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--sport", choices=SPORTS, required=True)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
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
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
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
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--early-stop-patience", type=int, default=100)
    parser.add_argument(
        "--monitor-metric",
        default="val_total",
        help=(
            "Metric used for best checkpoint selection. "
            "Examples: val_total, val_accuracy, val_classification."
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
        help="Disable inverse-frequency class weights for the auxiliary classification loss.",
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
        curve_title="A-TCN Training Loss",
        output_dir=output_dir,
        device=device,
    )


if __name__ == "__main__":
    main()
