from __future__ import annotations

import argparse
import os
import pickle
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from encoder_loader import build_model_from_checkpoint, checkpoint_model_type, default_checkpoint_path
from train_common import MotionFeatureDataset, build_samples, collect_feature_files


def latest_run_id(root: Path, sport: str) -> str | None:
    experiments_dir = root / "data" / sport / "experiments"
    if not experiments_dir.exists():
        return None
    marker_path = experiments_dir / ".latest_run_id"
    if marker_path.exists():
        marker = marker_path.read_text(encoding="utf-8-sig").strip()
        if marker:
            return marker
    run_dirs = [path for path in experiments_dir.iterdir() if path.is_dir() and (path / "models").exists()]
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


def normalize(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize one vector."""
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def mean_normalized(vectors: list[np.ndarray]) -> np.ndarray:
    """Average vectors and L2-normalize the mean."""
    return normalize(np.mean(np.stack(vectors, axis=0), axis=0))


def load_model(checkpoint: dict[str, Any], device: torch.device) -> Any:
    """Recreate the model from checkpoint metadata."""
    return build_model_from_checkpoint(checkpoint, device)


def build_prototype_store(
    model: Any,
    dataset: Any,
    checkpoint: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Build file-based Expert Prototype Store from expert samples."""
    sequence_vectors: dict[str, list[np.ndarray]] = defaultdict(list)
    step_vectors: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    sample_records = []
    step_names = list(checkpoint["shape_info"]["step_names"])

    with torch.no_grad():
        for index, sample_info in enumerate(dataset.samples):
            sample = dataset[index]
            sequence = sample["sequence"].unsqueeze(0).float().to(device)
            steps = sample["steps"].unsqueeze(0).float().to(device)
            outputs = model(sequence, steps)

            sequence_embedding = outputs["sequence_embedding"][0].detach().cpu().numpy()
            step_embeddings = outputs["step_embeddings"][0].detach().cpu().numpy()

            expert_id = str(sample_info.expert_id)
            sequence_vectors[expert_id].append(sequence_embedding)
            for step_idx, step_name in enumerate(step_names):
                step_vectors[expert_id][step_name].append(step_embeddings[step_idx])

            sample_records.append(
                {
                    "sample_id": sample_info.sample_id,
                    "expert_id": expert_id,
                    "path": str(sample_info.path),
                }
            )

    sequence_prototypes = {
        expert_id: mean_normalized(vectors)
        for expert_id, vectors in sequence_vectors.items()
    }
    step_prototypes = {
        expert_id: {
            step_name: mean_normalized(vectors)
            for step_name, vectors in step_dict.items()
        }
        for expert_id, step_dict in step_vectors.items()
    }

    return {
        "sport": checkpoint["sport"],
        "model_type": checkpoint_model_type(checkpoint),
        "embedding_dim": int(checkpoint["model_config"]["embedding_dim"]),
        "step_names": step_names,
        "feature_channels": list(checkpoint["shape_info"]["feature_channels"]),
        "joint_names": list(checkpoint["shape_info"]["joint_names"]),
        "label_map": checkpoint["label_map"],
        "sequence_prototypes": sequence_prototypes,
        "step_prototypes": step_prototypes,
        "samples": sample_records,
    }


def save_store(store: dict[str, Any], output_path: Path) -> None:
    """Save prototype store as pickle."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(store, file)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build expert sequence/step prototypes from a trained encoder.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--sport", choices=["Golf", "Archery"], required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--feature-dir", default=None)
    parser.add_argument("--run-id", default=None, help="Experiment run folder name. Defaults to YB_RUN_ID or latest experiment.")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    run_id = resolve_run_id(root, args.sport, args.run_id)
    write_latest_run_id(root, args.sport, run_id)
    exp_dir = experiment_dir(root, args.sport, run_id)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else default_checkpoint_path(exp_dir / "models")
    feature_dir = Path(args.feature_dir) if args.feature_dir else exp_dir / "features" / "expert"
    output_path = Path(args.output) if args.output else exp_dir / "prototypes" / "expert_prototypes.pkl"

    if checkpoint_path is None or not checkpoint_path.exists():
        # Report the path actually checked (an explicit --checkpoint, if
        # given) rather than always the default exp_dir/models location --
        # otherwise a missing checkpoint at a custom --checkpoint path prints
        # a misleading "not found" location that doesn't match what was
        # actually passed, wasting time diagnosing the real failure.
        print(f"Checkpoint not found: {checkpoint_path if checkpoint_path else exp_dir / 'models'}")
        print("Run one of 04-1_train_atcn.py, 04-2_train_transformer.py, or 04-3_train_stgcn.py first.")
        return

    feature_files = collect_feature_files(root, args.sport, feature_dir)
    if not feature_files:
        print(f"No expert feature files found: {feature_dir}")
        return

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint["sport"] != args.sport:
        raise ValueError(f"Checkpoint sport={checkpoint['sport']} does not match --sport {args.sport}")

    samples, _ = build_samples(feature_files)
    dataset = MotionFeatureDataset(samples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint, device)
    store = build_prototype_store(model, dataset, checkpoint, device)
    save_store(store, output_path)

    print(f"Saved prototype store: {output_path}")
    print(f"run_id={run_id}")
    print(f"experts={len(store['sequence_prototypes'])}, samples={len(store['samples'])}")


if __name__ == "__main__":
    main()
