from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from train_common import collect_feature_files, infer_expert_id


def load_sequence(path: Path) -> np.ndarray:
    """Load the sequence tensor (L, J, C) from a preprocessed feature npz."""
    with np.load(path, allow_pickle=False) as data:
        return data["sequence"].astype(np.float64)


def flatten_frames(sequence: np.ndarray) -> np.ndarray:
    """(L, J, C) -> (L, J*C)."""
    length = sequence.shape[0]
    return sequence.reshape(length, -1)


def normalize(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        return vector
    return vector / norm


def cosine_descriptor(sequence: np.ndarray) -> np.ndarray:
    """Mean-pool a sequence over time into one L2-normalized pose descriptor.

    This is the 'no learning' baseline from the review (C5 / P0-3): the exact
    same preprocessed input the learned encoders receive, collapsed to a
    single vector with plain averaging instead of a trained encoder.
    """
    pooled = flatten_frames(sequence).mean(axis=0)
    return normalize(pooled)


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Multivariate DTW distance between two (T, D) sequences.

    Anti-diagonal vectorized DP: O(T1*T2) total work, but only O(T1+T2)
    Python-level iterations (each processes a whole anti-diagonal with numpy
    ops), which is what makes this fast enough to run pairwise over a full
    holdout x train grid without a compiled dependency. Local cost is squared
    Euclidean distance, accumulated through the DP, with a single sqrt at the
    end -- the standard DTW convention (matches e.g. dtaidistance).
    """
    t1, t2 = a.shape[0], b.shape[0]
    diff = a[:, None, :] - b[None, :, :]
    cost = (diff**2).sum(axis=-1)

    inf = np.inf
    acc = np.full((t1, t2), inf, dtype=np.float64)
    acc[0, 0] = cost[0, 0]

    for k in range(1, t1 + t2 - 1):
        i_min = max(0, k - (t2 - 1))
        i_max = min(t1 - 1, k)
        if i_min > i_max:
            continue
        i_idx = np.arange(i_min, i_max + 1)
        j_idx = k - i_idx

        has_up = i_idx > 0
        has_left = j_idx > 0
        has_diag = has_up & has_left

        up = np.where(has_up, acc[np.clip(i_idx - 1, 0, None), j_idx], inf)
        left = np.where(has_left, acc[i_idx, np.clip(j_idx - 1, 0, None)], inf)
        diagp = np.where(has_diag, acc[np.clip(i_idx - 1, 0, None), np.clip(j_idx - 1, 0, None)], inf)

        acc[i_idx, j_idx] = cost[i_idx, j_idx] + np.minimum(np.minimum(up, left), diagp)

    return float(np.sqrt(acc[t1 - 1, t2 - 1]))


def metrics_from_ranks(ranks: list[int]) -> dict[str, float]:
    """Mirrors 07_evaluate.py's metrics_from_ranks so baseline and learned-model
    numbers are computed identically and directly comparable."""
    if not ranks:
        return {"count": 0.0, "top1_accuracy": 0.0, "mrr": 0.0, "mean_rank": 0.0}
    return {
        "count": float(len(ranks)),
        "top1_accuracy": sum(1 for rank in ranks if rank == 1) / len(ranks),
        "mrr": sum(1.0 / rank for rank in ranks) / len(ranks),
        "mean_rank": sum(ranks) / len(ranks),
    }


def rank_of_expert(ranking: list[tuple[str, float]], expert_id: str) -> int:
    for rank, (candidate_id, _score) in enumerate(ranking, start=1):
        if candidate_id == expert_id:
            return rank
    return len(ranking) + 1  # correct expert absent from gallery -> worst possible rank


def run_cosine_baseline(train_files: list[Path], holdout_files: list[Path]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train_by_expert: dict[str, list[np.ndarray]] = {}
    for path in train_files:
        expert_id = infer_expert_id(path.stem)
        train_by_expert.setdefault(expert_id, []).append(cosine_descriptor(load_sequence(path)))

    prototypes = {
        expert_id: normalize(np.mean(np.stack(vectors, axis=0), axis=0)) for expert_id, vectors in train_by_expert.items()
    }
    expert_ids = sorted(prototypes)

    rows: list[dict[str, Any]] = []
    ranks: list[int] = []
    for path in holdout_files:
        true_expert = infer_expert_id(path.stem)
        query = cosine_descriptor(load_sequence(path))
        scores = [(eid, float(query @ prototypes[eid])) for eid in expert_ids]
        ranking = sorted(scores, key=lambda item: -item[1])
        rank = rank_of_expert(ranking, true_expert)
        ranks.append(rank)
        rows.append({"sample_id": path.stem, "expert_id": true_expert, "rank": rank, "top1_expert_id": ranking[0][0]})
        print(f"[cosine] {path.stem}: true={true_expert} rank={rank} top1={ranking[0][0]}")

    return {"method": "cosine_raw_pose", "gallery_experts": expert_ids, **metrics_from_ranks(ranks)}, rows


def run_dtw_baseline(
    train_files: list[Path], holdout_files: list[Path], max_train_per_expert: int | None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train_by_expert: dict[str, list[np.ndarray]] = {}
    for path in train_files:
        expert_id = infer_expert_id(path.stem)
        train_by_expert.setdefault(expert_id, []).append(flatten_frames(load_sequence(path)))
    if max_train_per_expert:
        train_by_expert = {eid: seqs[:max_train_per_expert] for eid, seqs in train_by_expert.items()}
    expert_ids = sorted(train_by_expert)

    rows: list[dict[str, Any]] = []
    ranks: list[int] = []
    total = len(holdout_files)
    for qi, path in enumerate(holdout_files, start=1):
        true_expert = infer_expert_id(path.stem)
        query = flatten_frames(load_sequence(path))
        scores = []
        for eid in expert_ids:
            min_dist = min(dtw_distance(query, train_seq) for train_seq in train_by_expert[eid])
            scores.append((eid, min_dist))
        ranking = sorted(scores, key=lambda item: item[1])  # ascending distance = most similar first
        rank = rank_of_expert(ranking, true_expert)
        ranks.append(rank)
        rows.append({"sample_id": path.stem, "expert_id": true_expert, "rank": rank, "top1_expert_id": ranking[0][0]})
        print(f"[dtw {qi}/{total}] {path.stem}: true={true_expert} rank={rank} top1={ranking[0][0]} dist={ranking[0][1]:.2f}")

    return {"method": "dtw_nearest_neighbor", "gallery_experts": expert_ids, **metrics_from_ranks(ranks)}, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="No-learning retrieval baselines (cosine on raw pose, DTW nearest-neighbor) for the "
        "known-pro holdout protocol, computed identically to 07_evaluate.py's metrics so numbers are "
        "directly comparable to the learned-encoder results (review P0-3 / C5)."
    )
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--sport", choices=["Golf", "Archery"], default="Golf")
    parser.add_argument("--run-id", required=True, help="Experiment run folder name under data/<sport>/experiments/.")
    parser.add_argument("--method", choices=["cosine", "dtw", "both"], default="both")
    parser.add_argument(
        "--dtw-max-train-per-expert",
        type=int,
        default=None,
        help="Cap train samples compared per expert for DTW (speed/memory control). Default: no cap.",
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    exp_dir = root / "data" / args.sport / "experiments" / args.run_id
    train_files = collect_feature_files(root, args.sport, exp_dir / "features" / "expert")
    holdout_files = collect_feature_files(root, args.sport, exp_dir / "features" / "user")

    if not train_files:
        print(f"No expert (train) feature files found under {exp_dir / 'features' / 'expert'}")
        return
    if not holdout_files:
        print(f"No user (holdout) feature files found under {exp_dir / 'features' / 'user'}")
        return

    print(f"run_id={args.run_id}, train={len(train_files)}, holdout={len(holdout_files)}")

    output_path = Path(args.output) if args.output else exp_dir / "results" / "evaluation" / "baseline_retrieval_metrics.json"
    payload: dict[str, Any] = {"sport": args.sport, "run_id": args.run_id, "results": {}}

    if args.method in ("cosine", "both"):
        summary, rows = run_cosine_baseline(train_files, holdout_files)
        payload["results"]["cosine_raw_pose"] = {"summary": summary, "rows": rows}
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.method in ("dtw", "both"):
        summary, rows = run_dtw_baseline(train_files, holdout_files, args.dtw_max_train_per_expert)
        payload["results"]["dtw_nearest_neighbor"] = {"summary": summary, "rows": rows}
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
