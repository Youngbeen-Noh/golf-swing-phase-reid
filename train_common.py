from __future__ import annotations

import argparse
import csv
import json
import os
import random
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler, Subset


SPORTS = ["Golf", "Archery"]


@dataclass(frozen=True)
class FeatureSample:
    """One feature npz sample."""

    path: Path
    sample_id: str
    expert_id: str
    label: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def scalar(value: np.ndarray | Any) -> Any:
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def infer_expert_id(sample_id: str) -> str:
    return sample_id.rsplit("_", 1)[0]


def collect_feature_files(root: Path, sport: str, feature_dir: Path | None = None) -> list[Path]:
    feature_dir = feature_dir or root / "data" / sport / "features" / "expert"
    if not feature_dir.exists():
        return []
    return sorted(path for path in feature_dir.rglob("*.npz") if path.name != ".gitkeep")


def latest_run_id(root: Path, sport: str) -> str | None:
    experiments_dir = root / "data" / sport / "experiments"
    if not experiments_dir.exists():
        return None
    marker_path = experiments_dir / ".latest_run_id"
    if marker_path.exists():
        marker = marker_path.read_text(encoding="utf-8-sig").strip()
        if marker:
            return marker
    run_dirs = [path for path in experiments_dir.iterdir() if path.is_dir() and (path / "features").exists()]
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda path: path.stat().st_mtime).name


def resolve_run_id(root: Path, sport: str, run_id: str | None) -> str:
    if run_id:
        return run_id
    env_run_id = os.environ.get("YB_RUN_ID")
    if env_run_id:
        return env_run_id
    latest = latest_run_id(root, sport)
    if latest:
        return latest
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def experiment_dir(root: Path, sport: str, run_id: str) -> Path:
    return root / "data" / sport / "experiments" / run_id


def write_latest_run_id(root: Path, sport: str, run_id: str) -> None:
    experiments_dir = root / "data" / sport / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    (experiments_dir / ".latest_run_id").write_text(run_id, encoding="utf-8")


def build_samples(feature_files: list[Path]) -> tuple[list[FeatureSample], dict[str, int]]:
    expert_ids = sorted({infer_expert_id(path.stem) for path in feature_files})
    label_map = {expert_id: idx for idx, expert_id in enumerate(expert_ids)}
    samples = [
        FeatureSample(
            path=path,
            sample_id=path.stem,
            expert_id=infer_expert_id(path.stem),
            label=label_map[infer_expert_id(path.stem)],
        )
        for path in feature_files
    ]
    return samples, label_map


class MotionFeatureDataset(Dataset):
    """Dataset for preprocessed feature npz files."""

    def __init__(self, samples: list[FeatureSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        with np.load(sample.path, allow_pickle=False) as data:
            sequence = data["sequence"].astype(np.float32)
            steps = data["steps"].astype(np.float32)
            step_names = [str(item) for item in data["step_names"].tolist()]
            feature_channels = [str(item) for item in data["feature_channels"].tolist()]
            joint_names = [str(item) for item in data["joint_names"].tolist()]

        return {
            "sequence": torch.from_numpy(sequence),
            "steps": torch.from_numpy(steps),
            "label": torch.tensor(sample.label, dtype=torch.long),
            "sample_id": sample.sample_id,
            "expert_id": sample.expert_id,
            "step_names": step_names,
            "feature_channels": feature_channels,
            "joint_names": joint_names,
        }


class PKBatchSampler(Sampler[list[int]]):
    """Yields batches of P classes x K samples per class (local indices into
    whatever Subset/Dataset this wraps), so every class present in a batch is
    guaranteed to have >= K members. Without this, SupCon's positive_mask can
    end up empty for anchors whose class happens to appear only once in a
    plain-shuffled batch, silently dropping that sample's gradient signal
    (see supervised_contrastive_loss's valid_anchor masking).

    Classes with fewer than K available samples are sampled with replacement
    for that batch (duplicate-sampling policy, documented for P0-5).

    dynamic=True (training): a fresh random P*K assignment is drawn every
    epoch (every call to __iter__), maximizing data exposure/regularization.
    dynamic=False (validation): the batch assignment is fixed once at
    construction time and repeated identically every epoch, and covers every
    sample at least once, so the monitored val metric is comparable
    epoch-to-epoch (only the model changes, not which pairs happen to land
    together in a batch).
    """

    def __init__(
        self,
        labels: list[int],
        k: int,
        batch_size: int,
        seed: int,
        dynamic: bool = True,
        steps_per_epoch: int | None = None,
    ) -> None:
        if k < 2:
            raise ValueError(f"k must be >= 2 to guarantee SupCon positives, got {k}")
        if batch_size % k != 0:
            raise ValueError(f"batch_size ({batch_size}) must be a multiple of k ({k})")
        self.k = k
        self.p = batch_size // k
        self.seed = seed
        self.dynamic = dynamic

        self.by_label: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            self.by_label.setdefault(label, []).append(idx)
        self.classes = sorted(self.by_label)
        if len(self.classes) < self.p:
            # Fewer experts than the requested batch_size/k would need (e.g. a
            # 7-expert subset run with hyperparameters tuned for 25 experts).
            # Use every available class each batch instead of failing; the
            # realized batch size is then len(self.classes)*k < batch_size.
            warnings.warn(
                f"P*K sampler: only {len(self.classes)} classes available, less than "
                f"P={self.p} (batch_size={batch_size}, k={k}). Using all {len(self.classes)} "
                f"classes per batch instead (effective batch size {len(self.classes) * k}).",
                stacklevel=2,
            )
            self.p = len(self.classes)

        self.steps_per_epoch = steps_per_epoch or max(1, -(-len(labels) // batch_size))  # ceil
        self._epoch_counter = 0
        self._fixed_batches: list[list[int]] | None = None
        if not self.dynamic:
            self._fixed_batches = self._cover_all_samples()

    def _draw_batch(self, rng: random.Random) -> list[int]:
        classes = rng.sample(self.classes, self.p)
        batch: list[int] = []
        for label in classes:
            pool = self.by_label[label]
            if len(pool) >= self.k:
                batch.extend(rng.sample(pool, self.k))
            else:
                batch.extend(rng.choices(pool, k=self.k))
        rng.shuffle(batch)
        return batch

    def _cover_all_samples(self) -> list[list[int]]:
        """Deterministically build enough P*K batches to touch every sample at
        least once (used for the static validation pass). Each class's queue
        is consumed k-at-a-time; a leftover fragment smaller than k is always
        included as-is and padded with extra (repeated) draws to reach k, so
        no sample is ever silently dropped from the coverage guarantee."""
        rng = random.Random(f"{self.seed}:pk_static")
        queues: dict[int, list[int]] = {label: list(idxs) for label, idxs in self.by_label.items()}
        for pool in queues.values():
            rng.shuffle(pool)
        done = {label: False for label in self.classes}

        batches: list[list[int]] = []
        while not all(done.values()):
            pending = [c for c in self.classes if not done[c]]
            chosen = pending[: self.p]
            if len(chosen) < self.p:
                # Pad with already-covered classes just to keep batch shape
                # P*K; re-sampling a finished class is harmless (coverage was
                # already achieved for it).
                filler = [c for c in self.classes if c not in chosen]
                chosen = chosen + rng.sample(filler, min(self.p - len(chosen), len(filler)))

            batch: list[int] = []
            for label in chosen:
                queue = queues[label]
                if len(queue) >= self.k:
                    take = queue[: self.k]
                    queues[label] = queue[self.k :]
                    if not queues[label]:
                        done[label] = True
                elif queue:
                    extras = rng.choices(self.by_label[label], k=self.k - len(queue))
                    take = queue + extras
                    queues[label] = []
                    done[label] = True
                else:
                    take = rng.choices(self.by_label[label], k=self.k)
                batch.extend(take)
            rng.shuffle(batch)
            batches.append(batch)
        return batches

    def __iter__(self):
        if not self.dynamic:
            assert self._fixed_batches is not None
            yield from self._fixed_batches
            return
        rng = random.Random(f"{self.seed}:pk:{self._epoch_counter}")
        self._epoch_counter += 1
        for _ in range(self.steps_per_epoch):
            yield self._draw_batch(rng)

    def __len__(self) -> int:
        return len(self._fixed_batches) if self._fixed_batches is not None else self.steps_per_epoch


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if embeddings.shape[0] <= 1:
        return embeddings.sum() * 0.0

    embeddings = F.normalize(embeddings, dim=-1)
    logits = embeddings @ embeddings.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask

    if not positive_mask.any():
        return embeddings.sum() * 0.0

    exp_logits = torch.exp(logits) * (~self_mask)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positive_count = positive_mask.sum(dim=1).clamp_min(1)
    mean_log_prob = (positive_mask * log_prob).sum(dim=1) / positive_count
    valid_anchor = positive_mask.sum(dim=1) > 0
    return -mean_log_prob[valid_anchor].mean()


def triplet_batch_hard_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Batch-hard triplet loss (FaceNet/"In Defense of the Triplet Loss"
    formulation): per anchor, the hardest positive (furthest same-label
    embedding) and hardest negative (closest different-label embedding)
    within the batch. Requires each label to appear >=2 times in the batch
    -- true by construction under PKBatchSampler (P classes x K samples)."""
    if embeddings.shape[0] <= 1:
        return embeddings.sum() * 0.0

    embeddings = F.normalize(embeddings, dim=-1)
    dist = torch.cdist(embeddings, embeddings, p=2)
    same = labels[:, None].eq(labels[None, :])
    self_mask = torch.eye(labels.shape[0], dtype=torch.bool, device=labels.device)
    positive_mask = same & ~self_mask
    negative_mask = ~same

    if not positive_mask.any() or not negative_mask.any():
        return embeddings.sum() * 0.0

    hardest_positive = torch.where(positive_mask, dist, torch.full_like(dist, -float("inf"))).max(dim=1).values
    hardest_negative = torch.where(negative_mask, dist, torch.full_like(dist, float("inf"))).min(dim=1).values
    valid_anchor = positive_mask.any(dim=1) & negative_mask.any(dim=1)
    losses = F.relu(hardest_positive - hardest_negative + margin)
    return losses[valid_anchor].mean() if valid_anchor.any() else embeddings.sum() * 0.0


def compute_step_sequence_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    lambda_sequence: float,
    lambda_step: float,
    temperature: float,
    lambda_classification: float = 0.0,
    class_weights: torch.Tensor | None = None,
    metric_loss: str = "supcon",
    triplet_margin: float = 0.3,
) -> dict[str, torch.Tensor]:
    if metric_loss == "triplet":
        sequence_loss = triplet_batch_hard_loss(outputs["sequence_embedding"], labels, triplet_margin)
        step_losses = [
            triplet_batch_hard_loss(outputs["step_embeddings"][:, step_idx, :], labels, triplet_margin)
            for step_idx in range(outputs["step_embeddings"].shape[1])
        ]
    elif metric_loss == "supcon":
        sequence_loss = supervised_contrastive_loss(outputs["sequence_embedding"], labels, temperature)
        step_losses = [
            supervised_contrastive_loss(outputs["step_embeddings"][:, step_idx, :], labels, temperature)
            for step_idx in range(outputs["step_embeddings"].shape[1])
        ]
    else:
        raise ValueError(f"Unknown metric_loss: {metric_loss!r} (expected 'supcon' or 'triplet')")
    step_loss = torch.stack(step_losses).mean()
    classification_loss = sequence_loss.new_tensor(0.0)
    if lambda_classification > 0.0 and "logits" in outputs:
        classification_loss = F.cross_entropy(outputs["logits"], labels, weight=class_weights)
    total_loss = lambda_sequence * sequence_loss + lambda_step * step_loss + lambda_classification * classification_loss
    return {
        "total_loss": total_loss,
        "sequence_loss": sequence_loss,
        "step_loss": step_loss,
        "classification_loss": classification_loss,
    }


def split_indices(dataset: Dataset, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    samples = getattr(dataset, "samples", None)
    if samples is not None and val_ratio > 0:
        grouped: dict[int, list[int]] = {}
        for idx, sample in enumerate(samples):
            label = int(getattr(sample, "label"))
            grouped.setdefault(label, []).append(idx)

        rng = random.Random(seed)
        train_indices: list[int] = []
        val_indices: list[int] = []
        for indices in grouped.values():
            indices = list(indices)
            rng.shuffle(indices)
            if len(indices) <= 1:
                train_indices.extend(indices)
                continue
            val_count = int(round(len(indices) * val_ratio))
            val_count = max(1, min(val_count, len(indices) - 1))
            val_indices.extend(indices[:val_count])
            train_indices.extend(indices[val_count:])

        if train_indices and val_indices:
            rng.shuffle(train_indices)
            rng.shuffle(val_indices)
            return train_indices, val_indices

    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_count = int(round(len(indices) * val_ratio))
    if len(indices) > 1 and val_ratio > 0:
        val_count = max(1, min(val_count, len(indices) - 1))
    else:
        val_count = 0
    return indices[val_count:], indices[:val_count]


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    lambda_step: float,
    temperature: float,
    lambda_classification: float = 0.0,
    lambda_sequence: float = 1.0,
    class_weights: torch.Tensor | None = None,
    metric_loss: str = "supcon",
    triplet_margin: float = 0.3,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals = {"total_loss": 0.0, "sequence_loss": 0.0, "step_loss": 0.0, "classification_loss": 0.0}
    count = 0
    classification_correct = 0
    classification_count = 0

    for batch in loader:
        sequence = batch["sequence"].float().to(device)
        steps = batch["steps"].float().to(device)
        labels = batch["label"].long().to(device)

        with torch.set_grad_enabled(is_train):
            outputs = model(sequence, steps)
            losses = compute_step_sequence_loss(
                outputs,
                labels,
                lambda_sequence,
                lambda_step,
                temperature,
                lambda_classification,
                class_weights,
                metric_loss,
                triplet_margin,
            )
            if is_train:
                optimizer.zero_grad()
                losses["total_loss"].backward()
                optimizer.step()

        batch_size = sequence.shape[0]
        for key in totals:
            totals[key] += float(losses[key].detach().cpu()) * batch_size
        if "logits" in outputs:
            predictions = outputs["logits"].detach().argmax(dim=1)
            classification_correct += int(predictions.eq(labels).sum().item())
            classification_count += int(labels.numel())
        count += batch_size

    metrics = {key: value / max(count, 1) for key, value in totals.items()}
    metrics["classification_accuracy"] = classification_correct / max(classification_count, 1)
    return metrics


def load_shape_info(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        sequence = data["sequence"]
        steps = data["steps"]
        return {
            "sequence_shape": tuple(sequence.shape),
            "steps_shape": tuple(steps.shape),
            "joint_names": [str(item) for item in data["joint_names"].tolist()],
            "feature_channels": [str(item) for item in data["feature_channels"].tolist()],
            "step_names": [str(item) for item in data["step_names"].tolist()],
        }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    label_map: dict[str, int],
    args: argparse.Namespace,
    shape_info: dict[str, Any],
    epoch: int,
    val_loss: float,
    model_type: str,
    model_name: str,
    model_script: str,
    model_config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_map": label_map,
            "sport": args.sport,
            "epoch": epoch,
            "val_loss": val_loss,
            "model_type": model_type,
            "model_name": model_name,
            "model_script": model_script,
            "model_config": model_config,
            "shape_info": shape_info,
            "train_args": vars(args),
        },
        path,
    )


def save_training_history(history: list[dict[str, float]], output_dir: Path, title: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "training_history.json"
    csv_path = output_dir / "training_history.csv"
    curve_path = output_dir / "training_loss_curve.png"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)

    fieldnames = [
        "epoch",
        "train_total",
        "train_sequence",
        "train_step",
        "val_total",
        "val_sequence",
        "val_step",
        "train_classification",
        "val_classification",
        "train_accuracy",
        "val_accuracy",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    save_loss_curve(history, curve_path, title)


def save_loss_curve(history: list[dict[str, float]], output_path: Path, title: str) -> None:
    if not history:
        return

    width, height = 1200, 720
    margin_left, margin_right = 90, 40
    margin_top, margin_bottom = 70, 85
    image = np.full((height, width, 3), 250, dtype=np.uint8)
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    epochs = np.array([row["epoch"] for row in history], dtype=np.float32)
    train = np.array([row["train_total"] for row in history], dtype=np.float32)
    val = np.array([row["val_total"] for row in history], dtype=np.float32)
    min_loss = float(min(train.min(), val.min()))
    max_loss = float(max(train.max(), val.max()))
    if abs(max_loss - min_loss) < 1e-6:
        max_loss = min_loss + 1.0

    def to_point(epoch: float, loss: float) -> tuple[int, int]:
        x = margin_left + int(round((epoch - epochs.min()) / max(epochs.max() - epochs.min(), 1.0) * plot_w))
        y = margin_top + int(round((max_loss - loss) / (max_loss - min_loss) * plot_h))
        return x, y

    cv2.rectangle(
        image,
        (margin_left, margin_top),
        (margin_left + plot_w, margin_top + plot_h),
        (210, 210, 210),
        1,
    )
    cv2.putText(image, title, (40, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 55, 105), 2, cv2.LINE_AA)
    cv2.putText(image, "train_total", (820, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 150, 80), 2, cv2.LINE_AA)
    cv2.putText(image, "val_total", (1000, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (45, 95, 220), 2, cv2.LINE_AA)

    train_points = [to_point(float(epoch), float(loss)) for epoch, loss in zip(epochs, train)]
    val_points = [to_point(float(epoch), float(loss)) for epoch, loss in zip(epochs, val)]
    for points, color in [(train_points, (50, 150, 80)), (val_points, (45, 95, 220))]:
        for start, end in zip(points[:-1], points[1:]):
            cv2.line(image, start, end, color, 2, cv2.LINE_AA)

    best_idx = int(np.argmin(val))
    best_point = val_points[best_idx]
    cv2.circle(image, best_point, 6, (20, 45, 200), -1, cv2.LINE_AA)
    cv2.putText(
        image,
        f"best epoch={int(epochs[best_idx])}, val={float(val[best_idx]):.4f}",
        (margin_left, height - 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (40, 40, 40),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(image, f"loss {max_loss:.3f}", (10, margin_top + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
    cv2.putText(image, f"loss {min_loss:.3f}", (10, margin_top + plot_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
    cv2.putText(image, "epoch", (width // 2 - 25, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1)
    cv2.imwrite(str(output_path), image)


def metric_from_epoch_metrics(
    metric_name: str,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
) -> float:
    """Return a train/val metric by the names stored in training_history.csv."""
    metric_map = {
        "train_total": train_metrics["total_loss"],
        "train_sequence": train_metrics["sequence_loss"],
        "train_step": train_metrics["step_loss"],
        "train_classification": train_metrics["classification_loss"],
        "train_accuracy": train_metrics["classification_accuracy"],
        "val_total": val_metrics["total_loss"],
        "val_sequence": val_metrics["sequence_loss"],
        "val_step": val_metrics["step_loss"],
        "val_classification": val_metrics["classification_loss"],
        "val_accuracy": val_metrics["classification_accuracy"],
    }
    if metric_name not in metric_map:
        valid = ", ".join(sorted(metric_map))
        raise ValueError(f"Unknown monitor metric {metric_name!r}. Valid values: {valid}")
    return float(metric_map[metric_name])


def monitor_mode_for_metric(metric_name: str, requested_mode: str) -> str:
    """Resolve auto/min/max monitor mode."""
    if requested_mode != "auto":
        if requested_mode not in {"min", "max"}:
            raise ValueError("--monitor-mode must be one of auto, min, max")
        return requested_mode
    return "max" if "accuracy" in metric_name else "min"


def train_model(
    args: argparse.Namespace,
    model: nn.Module,
    label_map: dict[str, int],
    shape_info: dict[str, Any],
    dataset: MotionFeatureDataset,
    model_type: str,
    model_name: str,
    model_script: str,
    model_config: dict[str, Any],
    checkpoint_prefix: str,
    curve_title: str,
    output_dir: Path,
    device: torch.device,
) -> None:
    train_idx, val_idx = split_indices(dataset, args.val_ratio, args.seed)
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx) if val_idx else None

    pk_k = int(getattr(args, "pk_k", 0) or 0)
    if pk_k > 0:
        train_labels = [int(dataset.samples[i].label) for i in train_idx]
        train_sampler = PKBatchSampler(train_labels, k=pk_k, batch_size=args.batch_size, seed=args.seed, dynamic=True)
        train_loader = DataLoader(train_subset, batch_sampler=train_sampler)
        print(f"P*K sampler: P={train_sampler.p} classes x K={pk_k} samples/class = batch_size {args.batch_size}")

        val_loader = None
        if val_subset is not None:
            val_labels = [int(dataset.samples[i].label) for i in val_idx]
            try:
                val_sampler = PKBatchSampler(val_labels, k=pk_k, batch_size=args.batch_size, seed=args.seed, dynamic=False)
                val_loader = DataLoader(val_subset, batch_sampler=val_sampler)
            except ValueError as exc:
                warnings.warn(
                    f"P*K sampler unavailable for validation split ({exc}); falling back to plain shuffle=False batching.",
                    stacklevel=2,
                )
                val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
    else:
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False) if val_subset is not None else None

    model.to(device)
    class_weights = None
    if getattr(args, "use_class_weights", False) and label_map:
        counts = torch.zeros(len(label_map), dtype=torch.float32)
        for sample in dataset.samples:
            counts[int(sample.label)] += 1.0
        class_weights = counts.sum() / (len(label_map) * counts.clamp_min(1.0))
        class_weights = class_weights.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    monitor_metric = str(getattr(args, "monitor_metric", "val_total"))
    monitor_mode = monitor_mode_for_metric(monitor_metric, str(getattr(args, "monitor_mode", "auto")))
    best_score = -float("inf") if monitor_mode == "max" else float("inf")
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_path = output_dir / f"{checkpoint_prefix}_best.pt"
    latest_path = output_dir / f"{checkpoint_prefix}_latest.pt"
    history: list[dict[str, float]] = []

    print(
        f"Training model={model_name}, sport={args.sport}, run_id={args.run_id}, "
        f"samples={len(dataset)}, experts={len(label_map)}, device={device}"
    )
    print(f"sequence_shape={shape_info['sequence_shape']}, steps_shape={shape_info['steps_shape']}")
    print(f"split=train:{len(train_idx)}, val:{len(val_idx)}")
    print(
        "loss_weights="
        f"sequence:{getattr(args, 'lambda_sequence', 1.0)}, "
        f"step:{args.lambda_step}, "
        f"classification:{getattr(args, 'lambda_classification', 0.0)}"
    )
    print(f"monitor_metric={monitor_metric}, monitor_mode={monitor_mode}")
    if class_weights is not None:
        print(f"class_weights={class_weights.detach().cpu().numpy().round(4).tolist()}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.lambda_step,
            args.temperature,
            getattr(args, "lambda_classification", 0.0),
            getattr(args, "lambda_sequence", 1.0),
            class_weights,
            getattr(args, "metric_loss", "supcon"),
            getattr(args, "triplet_margin", 0.3),
        )
        if val_loader is not None:
            val_metrics = run_epoch(
                model,
                val_loader,
                None,
                device,
                args.lambda_step,
                args.temperature,
                getattr(args, "lambda_classification", 0.0),
                getattr(args, "lambda_sequence", 1.0),
                class_weights,
                getattr(args, "metric_loss", "supcon"),
                getattr(args, "triplet_margin", 0.3),
            )
        else:
            val_metrics = train_metrics

        print(
            f"epoch={epoch:03d} "
            f"train_total={train_metrics['total_loss']:.5f} "
            f"val_total={val_metrics['total_loss']:.5f} "
            f"val_seq={val_metrics['sequence_loss']:.5f} "
            f"val_step={val_metrics['step_loss']:.5f} "
            f"val_cls={val_metrics['classification_loss']:.5f} "
            f"val_acc={val_metrics['classification_accuracy']:.3f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_total": float(train_metrics["total_loss"]),
                "train_sequence": float(train_metrics["sequence_loss"]),
                "train_step": float(train_metrics["step_loss"]),
                "val_total": float(val_metrics["total_loss"]),
                "val_sequence": float(val_metrics["sequence_loss"]),
                "val_step": float(val_metrics["step_loss"]),
                "train_classification": float(train_metrics["classification_loss"]),
                "val_classification": float(val_metrics["classification_loss"]),
                "train_accuracy": float(train_metrics["classification_accuracy"]),
                "val_accuracy": float(val_metrics["classification_accuracy"]),
            }
        )
        monitor_score = metric_from_epoch_metrics(monitor_metric, train_metrics, val_metrics)
        save_training_history(history, output_dir, curve_title)
        save_checkpoint(
            latest_path,
            model,
            label_map,
            args,
            shape_info,
            epoch,
            val_metrics["total_loss"],
            model_type,
            model_name,
            model_script,
            model_config,
        )
        improved = monitor_score >= best_score if monitor_mode == "max" else monitor_score <= best_score
        if improved:
            best_score = monitor_score
            best_loss = val_metrics["total_loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                best_path,
                model,
                label_map,
                args,
                shape_info,
                epoch,
                best_loss,
                model_type,
                model_name,
                model_script,
                model_config,
            )
        else:
            epochs_without_improvement += 1

        patience = int(getattr(args, "early_stop_patience", 0) or 0)
        if patience > 0 and epochs_without_improvement >= patience:
            print(
                f"Early stopping triggered at epoch={epoch}. "
                f"best_epoch={best_epoch}, best_{monitor_metric}={best_score:.5f}, "
                f"best_val_loss={best_loss:.5f}"
            )
            break

    print(f"Saved best checkpoint: {best_path}")
    print(f"Saved latest checkpoint: {latest_path}")
    print(f"Best epoch={best_epoch}, best_{monitor_metric}={best_score:.5f}, best_val_loss={best_loss:.5f}")
    print(f"Label map: {json.dumps(label_map, ensure_ascii=False)}")
