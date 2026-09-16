from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any
from datetime import datetime

import numpy as np
import torch

from encoder_loader import build_model_from_checkpoint, checkpoint_model_type, default_checkpoint_path


def scalar(value: np.ndarray | Any) -> Any:
    """Convert 0-d numpy arrays to Python values."""
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def cosine_similarity(left: np.ndarray, right: np.ndarray, eps: float = 1e-12) -> float:
    """Compute cosine similarity between two vectors."""
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom < eps:
        return 0.0
    return float(np.dot(left, right) / denom)


def best_cosine_similarity(query_embedding: np.ndarray, prototype: np.ndarray) -> float:
    """A prototype is normally a single (D,) vector (per-expert mean, the
    05_build_expert_prototypes.py default) -- returns its plain cosine
    similarity, identical to prior behavior. A "multi-prototype" robustness
    variant (27_build_prototype_variants.py, 2nd revision instructions,
    "recommended" tier) instead stores a (K, D) stack of per-expert
    sub-prototypes (e.g. k-means clusters over that expert's training
    embeddings, capturing more than one style within one expert); in that
    case the MAX similarity over the K sub-prototypes is returned, so the
    query is matched against whichever sub-style it is closest to."""
    prototype = np.asarray(prototype, dtype=np.float32)
    if prototype.ndim == 1:
        return cosine_similarity(query_embedding, prototype)
    return max(cosine_similarity(query_embedding, row) for row in prototype)


def rank_sequence(
    query_embedding: np.ndarray,
    sequence_prototypes: dict[str, np.ndarray],
) -> list[dict[str, float | str]]:
    """Rank experts by sequence prototype similarity."""
    ranking = [
        {
            "expert_id": expert_id,
            "score": best_cosine_similarity(query_embedding, prototype),
        }
        for expert_id, prototype in sequence_prototypes.items()
    ]
    return sorted(ranking, key=lambda item: float(item["score"]), reverse=True)


def rank_steps(
    query_step_embeddings: np.ndarray,
    step_prototypes: dict[str, dict[str, np.ndarray]],
    step_names: list[str],
) -> dict[str, list[dict[str, float | str]]]:
    """Rank experts per step."""
    result: dict[str, list[dict[str, float | str]]] = {}
    for step_idx, step_name in enumerate(step_names):
        ranking = []
        for expert_id, expert_steps in step_prototypes.items():
            if step_name not in expert_steps:
                continue
            ranking.append(
                {
                    "expert_id": expert_id,
                    "score": best_cosine_similarity(query_step_embeddings[step_idx], expert_steps[step_name]),
                }
            )
        result[step_name] = sorted(ranking, key=lambda item: float(item["score"]), reverse=True)
    return result


def final_ranking(
    sequence_ranking: list[dict[str, float | str]],
    step_similarity: dict[str, list[dict[str, float | str]]],
    alpha: float,
) -> list[dict[str, float | str]]:
    """Combine sequence similarity and mean step similarity."""
    sequence_scores = {str(item["expert_id"]): float(item["score"]) for item in sequence_ranking}
    step_scores: dict[str, list[float]] = {expert_id: [] for expert_id in sequence_scores}
    for ranking in step_similarity.values():
        for item in ranking:
            expert_id = str(item["expert_id"])
            if expert_id in step_scores:
                step_scores[expert_id].append(float(item["score"]))

    combined = []
    for expert_id, sequence_score in sequence_scores.items():
        mean_step_score = float(np.mean(step_scores[expert_id])) if step_scores[expert_id] else 0.0
        score = float(alpha) * sequence_score + (1.0 - float(alpha)) * mean_step_score
        combined.append(
            {
                "expert_id": expert_id,
                "score": score,
                "sequence_score": sequence_score,
                "mean_step_score": mean_step_score,
            }
        )
    return sorted(combined, key=lambda item: float(item["score"]), reverse=True)


def low_similarity_steps(
    step_similarity: dict[str, list[dict[str, float | str]]],
) -> list[dict[str, float | str]]:
    """Return steps sorted by their best expert similarity ascending."""
    rows = []
    for step_name, ranking in step_similarity.items():
        best_score = float(ranking[0]["score"]) if ranking else 0.0
        best_expert = str(ranking[0]["expert_id"]) if ranking else ""
        rows.append(
            {
                "step": step_name,
                "best_score": best_score,
                "best_expert_id": best_expert,
            }
        )
    return sorted(rows, key=lambda item: float(item["best_score"]))


def load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    """Load a PyTorch checkpoint."""
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def load_model(checkpoint: dict[str, Any], device: torch.device) -> Any:
    """Recreate model from checkpoint metadata."""
    return build_model_from_checkpoint(checkpoint, device)


def load_prototype_store(path: Path) -> dict[str, Any]:
    """Load Expert Prototype Store."""
    with path.open("rb") as file:
        return pickle.load(file)


def load_feature_sample(path: Path) -> dict[str, Any]:
    """Load one user feature npz sample."""
    with np.load(path, allow_pickle=False) as data:
        return {
            "sequence": torch.from_numpy(data["sequence"].astype(np.float32)),
            "steps": torch.from_numpy(data["steps"].astype(np.float32)),
            "sample_id": str(scalar(data["sample_id"])),
            "sport": str(scalar(data["sport"])),
            "subject_type": str(scalar(data["subject_type"])),
            "step_names": [str(item) for item in data["step_names"].tolist()],
            "source_json": str(scalar(data["source_json"])),
        }


def infer_one(
    feature_path: Path,
    model: Any,
    prototype_store: dict[str, Any],
    device: torch.device,
    alpha: float,
) -> dict[str, Any]:
    """Infer similarity report for one user feature file."""
    sample = load_feature_sample(feature_path)
    sequence = sample["sequence"].unsqueeze(0).float().to(device)
    steps = sample["steps"].unsqueeze(0).float().to(device)

    with torch.no_grad():
        outputs = model(sequence, steps)

    sequence_embedding = outputs["sequence_embedding"][0].detach().cpu().numpy()
    step_embeddings = outputs["step_embeddings"][0].detach().cpu().numpy()
    step_names = list(prototype_store["step_names"])

    sequence_ranking = rank_sequence(sequence_embedding, prototype_store["sequence_prototypes"])
    step_similarity = rank_steps(step_embeddings, prototype_store["step_prototypes"], step_names)
    final = final_ranking(sequence_ranking, step_similarity, alpha)

    return {
        "sample_id": sample["sample_id"],
        "sport": sample["sport"],
        "source_json": sample["source_json"],
        "feature_path": str(feature_path),
        "alpha": alpha,
        "sequence_ranking": sequence_ranking,
        "step_similarity": step_similarity,
        "final_ranking": final,
        "low_similarity_steps": low_similarity_steps(step_similarity),
    }


def collect_user_features(root: Path, sport: str, feature_dir: Path | None = None) -> list[Path]:
    """Collect user feature files for one sport."""
    feature_dir = feature_dir or root / "data" / sport / "features" / "user"
    if not feature_dir.exists():
        return []
    return sorted(path for path in feature_dir.rglob("*.npz") if path.name != ".gitkeep")


def save_report(report: dict[str, Any], output_dir: Path) -> Path:
    """Save one similarity report JSON."""
    sample_id = str(report["sample_id"])
    person_id = sample_id.rsplit("_", 1)[0]
    output_path = output_dir / person_id / f"{sample_id}_similarity.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    return output_path


def resolve_run_id(run_id: str | None) -> str:
    if run_id:
        return run_id
    env_run_id = os.environ.get("YB_RUN_ID")
    if env_run_id:
        return env_run_id
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def latest_run_id(root: Path, sport: str) -> str | None:
    experiments_dir = root / "data" / sport / "experiments"
    if not experiments_dir.exists():
        return None
    marker_path = experiments_dir / ".latest_run_id"
    if marker_path.exists():
        marker = marker_path.read_text(encoding="utf-8-sig").strip()
        if marker:
            return marker
    run_dirs = [path for path in experiments_dir.iterdir() if path.is_dir() and (path / "prototypes").exists()]
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda path: path.stat().st_mtime).name


def resolve_experiment_run_id(root: Path, sport: str, run_id: str | None) -> str:
    if run_id:
        return run_id
    env_run_id = os.environ.get("YB_RUN_ID")
    if env_run_id:
        return env_run_id
    latest = latest_run_id(root, sport)
    if latest:
        return latest
    return resolve_run_id(None)


def experiment_dir(root: Path, sport: str, run_id: str) -> Path:
    return root / "data" / sport / "experiments" / run_id


def write_latest_run_id(root: Path, sport: str, run_id: str) -> None:
    experiments_dir = root / "data" / sport / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    (experiments_dir / ".latest_run_id").write_text(run_id, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compare user embedding with expert prototypes.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--sport", choices=["Golf", "Archery"], required=True)
    parser.add_argument("--feature-path", default=None, help="Optional single user feature npz.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--prototype-store", default=None)
    parser.add_argument("--feature-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-id", default=None, help="Experiment run folder name. Defaults to YB_RUN_ID or latest experiment.")
    parser.add_argument("--alpha", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    run_id = resolve_experiment_run_id(root, args.sport, args.run_id)
    write_latest_run_id(root, args.sport, run_id)
    exp_dir = experiment_dir(root, args.sport, run_id)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else default_checkpoint_path(exp_dir / "models")
    prototype_path = (
        Path(args.prototype_store)
        if args.prototype_store
        else exp_dir / "prototypes" / "expert_prototypes.pkl"
    )
    feature_dir = Path(args.feature_dir) if args.feature_dir else exp_dir / "features" / "user"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else exp_dir / "results" / "user_similarity"
    )

    if checkpoint_path is None or not checkpoint_path.exists():
        # Report the path actually checked (an explicit --checkpoint, if
        # given), not always the default exp_dir/models location -- same fix
        # as 05_build_expert_prototypes.py's identical pattern.
        print(f"Checkpoint not found: {checkpoint_path if checkpoint_path else exp_dir / 'models'}")
        print("Run one of 04-1_train_atcn.py, 04-2_train_transformer.py, or 04-3_train_stgcn.py first.")
        return
    if not prototype_path.exists():
        print(f"Prototype store not found: {prototype_path}")
        print("Run 05_build_expert_prototypes.py first.")
        return

    feature_paths = [Path(args.feature_path)] if args.feature_path else collect_user_features(root, args.sport, feature_dir)
    if not feature_paths:
        print(f"No user feature files found: {feature_dir}")
        print("Run 03_preprocess_and_build_features.py for user data first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint["sport"] != args.sport:
        raise ValueError(f"Checkpoint sport={checkpoint['sport']} does not match --sport {args.sport}")
    prototype_store = load_prototype_store(prototype_path)
    if prototype_store["sport"] != args.sport:
        raise ValueError(f"Prototype sport={prototype_store['sport']} does not match --sport {args.sport}")
    prototype_model_type = prototype_store.get("model_type")
    if prototype_model_type and str(prototype_model_type).lower() != checkpoint_model_type(checkpoint):
        raise ValueError(
            f"Prototype model_type={prototype_model_type} does not match "
            f"checkpoint model_type={checkpoint_model_type(checkpoint)}"
        )

    model = load_model(checkpoint, device)
    print(f"Inference sport={args.sport}, user_samples={len(feature_paths)}, device={device}, run_id={run_id}")

    for feature_path in feature_paths:
        report = infer_one(feature_path, model, prototype_store, device, alpha=args.alpha)
        output_path = save_report(report, output_dir)
        best = report["final_ranking"][0] if report["final_ranking"] else {"expert_id": "", "score": 0.0}
        print(f"[save] {output_path} | best={best['expert_id']} score={float(best['score']):.4f}")


if __name__ == "__main__":
    main()
