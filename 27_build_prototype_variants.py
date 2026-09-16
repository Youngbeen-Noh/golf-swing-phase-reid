"""Prototype robustness variants (2nd revision instructions, "recommended"
tier): does the CHOICE of expert-prototype aggregation method matter for
retrieval accuracy? 05_build_expert_prototypes.py always uses a simple
per-expert mean of L2-normalized embeddings (mean_normalized) -- this paper's
own Limitations (Section 6.1) already flag this as "sensitive to outliers and
class imbalance" and note that "alternatives (medoid, trimmed mean,
multi-prototype) were not compared."

This script builds four gallery-side prototype-store variants from the SAME
already-trained checkpoint's embeddings, with a SINGLE forward pass over the
training pool (no retraining):
  - mean:       the existing default (per-expert mean, L2-normalized) --
                included here for a self-contained side-by-side comparison,
                identical to 05_build_expert_prototypes.py's output.
  - medoid:     the actual training-sample embedding closest (by total
                cosine distance) to every other same-expert embedding -- a
                real data point, immune to being pulled toward an outlier
                the way a synthetic mean can be.
  - trimmed:    drop the furthest TRIM_FRACTION of an expert's embeddings (by
                cosine distance from that expert's own initial mean), then
                mean + L2-normalize the rest -- a robust-statistics
                compromise between mean and medoid. Falls back to the plain
                mean when there are too few samples to trim anything.
  - multiproto: k-means (k=3, cosine-flavored via L2-normalized Euclidean
                k-means) per expert over that expert's embeddings, storing
                all k' <= k normalized centroids (k' = min(k, n_samples), so
                an expert with fewer training samples than k never produces
                empty clusters). 06_infer_user_similarity.py's
                best_cosine_similarity() takes the MAX similarity over an
                expert's sub-prototypes at query time -- a small,
                backward-compatible extension (a prototype may now be a (D,)
                vector, unchanged behavior, or a (K,D) stack).

Query-side and encoder are completely unaffected: only the gallery-side
prototype-store file differs per variant, so 06_infer_user_similarity.py /
07_evaluate.py are reused unchanged, pointed at each variant's .pkl in turn.

Usage:
  python 27_build_prototype_variants.py --sport Golf \
      --checkpoint data/Golf/experiments/ATCN_Golf-D-s42/models/atcn_step_sequence_best.pt \
      --feature-dir data/Golf/experiments/ATCN_Golf-D/features/expert \
      --output-dir data/Golf/experiments/ATCN_Golf-D-s42/prototypes
  # writes expert_prototypes_mean.pkl, _medoid.pkl, _trimmed.pkl, _multiproto.pkl
  # then, per variant, reuse 06_infer_user_similarity.py / 07_evaluate.py with
  # --prototype-store pointed at that variant's file (same checkpoint/query
  # features as the real run).
"""

from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from encoder_loader import build_model_from_checkpoint, checkpoint_model_type, default_checkpoint_path
from train_common import MotionFeatureDataset, build_samples, collect_feature_files

TRIM_FRACTION = 0.2
MULTIPROTO_K = 3
KMEANS_SEED = 42
KMEANS_MAX_ITER = 100


def normalize(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize one vector."""
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def mean_normalized(vectors: list[np.ndarray]) -> np.ndarray:
    """Average vectors and L2-normalize the mean (05_build_expert_prototypes.py's
    existing default aggregation, reused here unchanged for comparison)."""
    return normalize(np.mean(np.stack(vectors, axis=0), axis=0))


def cosine_distance_matrix(vectors: np.ndarray) -> np.ndarray:
    """vectors: (N, D). Returns (N, N) pairwise cosine distance (1 - cosine similarity)."""
    normed = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
    return 1.0 - (normed @ normed.T)


def medoid(vectors: list[np.ndarray]) -> np.ndarray:
    """The real embedding minimizing total cosine distance to every other
    vector in the same expert's set (ties broken by array order, so this is
    deterministic)."""
    stacked = np.stack(vectors, axis=0)
    if len(vectors) == 1:
        return normalize(stacked[0])
    total_dist = cosine_distance_matrix(stacked).sum(axis=1)
    return normalize(stacked[int(np.argmin(total_dist))])


def trimmed_mean(vectors: list[np.ndarray], trim_fraction: float = TRIM_FRACTION) -> np.ndarray:
    """Drop the furthest trim_fraction of vectors (by cosine distance from
    the expert's own initial mean), then mean + L2-normalize the rest."""
    stacked = np.stack(vectors, axis=0)
    n = len(vectors)
    n_drop = int(np.floor(n * trim_fraction))
    if n_drop == 0 or n - n_drop < 1:
        return mean_normalized(vectors)
    initial_mean = normalize(np.mean(stacked, axis=0))
    normed = stacked / (np.linalg.norm(stacked, axis=1, keepdims=True) + 1e-12)
    sim_to_mean = normed @ initial_mean
    keep_idx = np.argsort(sim_to_mean)[n_drop:]  # drop the n_drop LOWEST-similarity (furthest) vectors
    return normalize(np.mean(stacked[keep_idx], axis=0))


def kmeans_prototypes(
    vectors: list[np.ndarray], k: int = MULTIPROTO_K, seed: int = KMEANS_SEED, max_iter: int = KMEANS_MAX_ITER
) -> np.ndarray:
    """Cosine-flavored k-means (L2-normalized Euclidean k-means, equivalent to
    cosine k-means on unit vectors): returns (k', D) normalized centroids,
    k' = min(k, n_samples) so an expert with fewer training samples than k
    never produces an empty cluster."""
    stacked = np.stack(vectors, axis=0).astype(np.float64)
    normed = stacked / (np.linalg.norm(stacked, axis=1, keepdims=True) + 1e-12)
    n = normed.shape[0]
    k_eff = min(k, n)
    rng = np.random.default_rng(seed)
    centroids = normed[rng.choice(n, size=k_eff, replace=False)].copy()

    for _ in range(max_iter):
        assignments = np.argmax(normed @ centroids.T, axis=1)
        new_centroids = np.empty_like(centroids)
        moved = False
        for c in range(k_eff):
            members = normed[assignments == c]
            if len(members) == 0:
                new_centroids[c] = centroids[c]  # keep the stale centroid rather than crash on an empty cluster
                continue
            new_c = normalize(members.mean(axis=0))
            if not np.allclose(new_c, centroids[c]):
                moved = True
            new_centroids[c] = new_c
        centroids = new_centroids
        if not moved:
            break
    return centroids.astype(np.float32)


VARIANTS: dict[str, Callable[[list[np.ndarray]], np.ndarray]] = {
    "mean": mean_normalized,
    "medoid": medoid,
    "trimmed": trimmed_mean,
    "multiproto": kmeans_prototypes,
}


def collect_raw_embeddings(
    model: Any, dataset: Any, checkpoint: dict[str, Any], device: torch.device
) -> tuple[dict[str, list[np.ndarray]], dict[str, dict[str, list[np.ndarray]]], list[str], list[dict[str, str]]]:
    """One forward pass over every training sample, returning RAW per-sample
    embeddings grouped by expert (and by expert+step) -- i.e. everything
    05_build_expert_prototypes.py's build_prototype_store computes just
    before collapsing to a single mean per expert."""
    sequence_vectors: dict[str, list[np.ndarray]] = defaultdict(list)
    step_vectors: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    step_names = list(checkpoint["shape_info"]["step_names"])
    sample_records: list[dict[str, str]] = []

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
                {"sample_id": sample_info.sample_id, "expert_id": expert_id, "path": str(sample_info.path)}
            )

    return sequence_vectors, step_vectors, step_names, sample_records


def build_variant_store(
    agg_fn: Callable[[list[np.ndarray]], np.ndarray],
    sequence_vectors: dict[str, list[np.ndarray]],
    step_vectors: dict[str, dict[str, list[np.ndarray]]],
    step_names: list[str],
    checkpoint: dict[str, Any],
    sample_records: list[dict[str, str]],
) -> dict[str, Any]:
    sequence_prototypes = {expert_id: agg_fn(vectors) for expert_id, vectors in sequence_vectors.items()}
    step_prototypes = {
        expert_id: {step_name: agg_fn(vectors) for step_name, vectors in step_dict.items()}
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(store, file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--sport", choices=["Golf", "Archery"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    checkpoint_path = Path(args.checkpoint)
    feature_dir = Path(args.feature_dir)
    output_dir = Path(args.output_dir)

    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    feature_files = collect_feature_files(root, args.sport, feature_dir)
    if not feature_files:
        raise SystemExit(f"No expert feature files found: {feature_dir}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint["sport"] != args.sport:
        raise ValueError(f"Checkpoint sport={checkpoint['sport']} does not match --sport {args.sport}")

    samples, _ = build_samples(feature_files)
    dataset = MotionFeatureDataset(samples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_checkpoint(checkpoint, device)

    print(f"Collecting raw per-sample embeddings: {len(dataset.samples)} training samples, device={device}")
    sequence_vectors, step_vectors, step_names, sample_records = collect_raw_embeddings(
        model, dataset, checkpoint, device
    )
    n_experts = len(sequence_vectors)
    print(f"experts={n_experts}, samples={len(sample_records)}")

    for variant_name, agg_fn in VARIANTS.items():
        store = build_variant_store(agg_fn, sequence_vectors, step_vectors, step_names, checkpoint, sample_records)
        output_path = output_dir / f"expert_prototypes_{variant_name}.pkl"
        save_store(store, output_path)
        print(f"[saved] {variant_name}: {output_path}")


if __name__ == "__main__":
    main()
