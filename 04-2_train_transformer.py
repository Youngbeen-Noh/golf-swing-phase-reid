from __future__ import annotations

import argparse
import json
import math
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


MODEL_TYPE = "transformer"
MODEL_NAME = "Transformer Encoder"
MODEL_SCRIPT = "04-2_train_transformer.py"
CHECKPOINT_PREFIX = "transformer_step_sequence"


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal temporal positional encoding."""

    def __init__(self, model_dim: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / model_dim))
        pe = torch.zeros(max_len, model_dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        return x + self.pe[:, :length, :].to(dtype=x.dtype, device=x.device)


class TemporalTransformerEncoder(nn.Module):
    """Temporal Transformer encoder over flattened joint-channel features."""

    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.position = SinusoidalPositionalEncoding(model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(model_dim)
        self.projection = nn.Sequential(
            nn.Linear(model_dim * 3, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = self.input_projection(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.position(x)
        x = self.encoder(x)
        x = self.norm(x)
        cls = x[:, 0, :]
        tokens = x[:, 1:, :]
        pooled = torch.cat([cls, tokens.mean(dim=1), tokens.std(dim=1, unbiased=False)], dim=-1)
        x = self.projection(pooled)
        return F.normalize(x, p=2, dim=-1)


class StepSequenceTransformer(nn.Module):
    """Step-sequence embedding model backed by Transformer encoders."""

    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
        num_classes: int = 0,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.sequence_encoder = TemporalTransformerEncoder(
            input_dim=input_dim,
            model_dim=model_dim,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )
        self.classifier = nn.Linear(embedding_dim, num_classes) if num_classes > 0 else None
        self.step_encoder = TemporalTransformerEncoder(
            input_dim=input_dim,
            model_dim=model_dim,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )

    def encode_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        batch, length, joints, channels = sequence.shape
        return self.sequence_encoder(sequence.reshape(batch, length, joints * channels))

    def encode_steps(self, steps: torch.Tensor) -> torch.Tensor:
        batch, step_count, length, joints, channels = steps.shape
        embeddings = self.step_encoder(steps.reshape(batch * step_count, length, joints * channels))
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
        "input_dim": int(shape_info["sequence_shape"][1] * shape_info["sequence_shape"][2]),
        "model_dim": int(args.model_dim),
        "embedding_dim": int(args.embedding_dim),
        "num_layers": int(args.num_layers),
        "num_heads": int(args.num_heads),
        "feedforward_dim": int(args.feedforward_dim),
        "dropout": float(args.dropout),
        "num_classes": int(num_classes),
    }


def build_model_from_config(model_config: dict[str, Any]) -> StepSequenceTransformer:
    return StepSequenceTransformer(
        input_dim=int(model_config["input_dim"]),
        model_dim=int(model_config["model_dim"]),
        embedding_dim=int(model_config["embedding_dim"]),
        num_layers=int(model_config["num_layers"]),
        num_heads=int(model_config["num_heads"]),
        feedforward_dim=int(model_config["feedforward_dim"]),
        dropout=float(model_config["dropout"]),
        num_classes=int(model_config.get("num_classes", 0)),
    )


def build_model_from_checkpoint(checkpoint: dict[str, Any], device: torch.device) -> StepSequenceTransformer:
    model = build_model_from_config(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Transformer Encoder step_sequence embedding encoder.")
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
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--feedforward-dim", type=int, default=128)
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
        help="Disable inverse-frequency class weights for Transformer classification loss.",
    )
    parser.set_defaults(use_class_weights=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model_dim % args.num_heads != 0:
        raise ValueError("--model-dim must be divisible by --num-heads")

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
        curve_title="Transformer Training Loss",
        output_dir=output_dir,
        device=device,
    )


if __name__ == "__main__":
    main()
