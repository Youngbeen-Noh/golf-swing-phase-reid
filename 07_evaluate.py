from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any
from datetime import datetime

import numpy as np
import cv2
import torch


def load_inference_module() -> Any:
    """Load 06_infer_user_similarity.py even though its filename starts with a number."""
    module_path = Path(__file__).resolve().with_name("06_infer_user_similarity.py")
    spec = importlib.util.spec_from_file_location("infer_user_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load inference module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


INFER_MODULE = load_inference_module()
load_checkpoint = INFER_MODULE.load_checkpoint
load_model = INFER_MODULE.load_model
load_prototype_store = INFER_MODULE.load_prototype_store
infer_one = INFER_MODULE.infer_one
default_checkpoint_path = INFER_MODULE.default_checkpoint_path
checkpoint_model_type = INFER_MODULE.checkpoint_model_type


def infer_expert_id(sample_id: str) -> str:
    """Infer expert id from sample_id naming convention: some-expert_001 -> some-expert."""
    return sample_id.rsplit("_", 1)[0]


def collect_expert_features(root: Path, sport: str, feature_dir: Path | None = None) -> list[Path]:
    """Collect expert feature npz files for evaluation queries."""
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
    run_dirs = [path for path in experiments_dir.iterdir() if path.is_dir() and (path / "prototypes").exists()]
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


def rank_of_correct_expert(ranking: list[dict[str, Any]], expert_id: str) -> int | None:
    """Return 1-based rank of the correct expert, or None when absent."""
    for rank, item in enumerate(ranking, start=1):
        if str(item["expert_id"]) == expert_id:
            return rank
    return None


def compute_metrics(results: list[dict[str, Any]], recall_k: int) -> dict[str, float]:
    """Compute retrieval metrics for expert-query evaluation."""
    if not results:
        return {
            "top1_accuracy": 0.0,
            f"recall@{recall_k}": 0.0,
            "mAP": 0.0,
            "MRR": 0.0,
        }

    top1_hits = 0
    recall_hits = 0
    reciprocal_ranks = []

    for result in results:
        correct_rank = result["correct_rank"]
        if correct_rank == 1:
            top1_hits += 1
        if correct_rank is not None and correct_rank <= recall_k:
            recall_hits += 1
        reciprocal_ranks.append(0.0 if correct_rank is None else 1.0 / float(correct_rank))

    query_count = len(results)
    mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    return {
        "top1_accuracy": top1_hits / query_count,
        f"recall@{recall_k}": recall_hits / query_count,
        "mAP": mrr,
        "MRR": mrr,
    }


def compute_per_expert_metrics(results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Compute retrieval metrics grouped by the true expert."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(str(result["expert_id"]), []).append(result)

    output: dict[str, dict[str, float]] = {}
    for expert_id, rows in sorted(grouped.items()):
        ranks = [row["correct_rank"] for row in rows]
        top1 = sum(1 for rank in ranks if rank == 1)
        valid_ranks = [float(rank) for rank in ranks if rank is not None]
        output[expert_id] = {
            "sample_count": float(len(rows)),
            "top1_accuracy": top1 / max(len(rows), 1),
            "mean_rank": float(np.mean(valid_ranks)) if valid_ranks else 0.0,
            "mrr": float(np.mean([0.0 if rank is None else 1.0 / float(rank) for rank in ranks])),
        }
    return output


def build_confusion_matrix(results: list[dict[str, Any]]) -> tuple[list[str], np.ndarray]:
    """Build true-expert by predicted-top1 confusion matrix."""
    labels = sorted({str(row["expert_id"]) for row in results} | {str(row["top1_expert_id"]) for row in results})
    index = {label: idx for idx, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int32)
    for row in results:
        matrix[index[str(row["expert_id"])]][index[str(row["top1_expert_id"])]] += 1
    return labels, matrix


def save_confusion_csv(labels: list[str], matrix: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["true\\pred", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *[int(value) for value in row]])


def save_confusion_png(labels: list[str], matrix: np.ndarray, output_path: Path) -> None:
    """Draw a compact confusion matrix image with OpenCV."""
    cell = 92
    left = 260
    top = 150
    width = left + cell * len(labels) + 80
    height = top + cell * len(labels) + 110
    image = np.full((height, width, 3), 250, dtype=np.uint8)
    max_value = int(matrix.max()) if matrix.size else 1
    max_value = max(max_value, 1)

    cv2.putText(image, "Expert Retrieval Confusion Matrix", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 55, 105), 2, cv2.LINE_AA)
    cv2.putText(image, "Predicted top-1 expert", (left, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(image, "True expert", (35, top - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1, cv2.LINE_AA)

    for col, label in enumerate(labels):
        x = left + col * cell + 8
        cv2.putText(image, label[:12], (x, top - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (45, 45, 45), 1, cv2.LINE_AA)
    for row, label in enumerate(labels):
        y = top + row * cell + cell // 2
        cv2.putText(image, label[:20], (35, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (45, 45, 45), 1, cv2.LINE_AA)

    for row in range(len(labels)):
        for col in range(len(labels)):
            value = int(matrix[row, col])
            intensity = int(235 - 170 * (value / max_value))
            color = (245, intensity, intensity) if row == col else (intensity, intensity, 245)
            x0 = left + col * cell
            y0 = top + row * cell
            cv2.rectangle(image, (x0, y0), (x0 + cell - 4, y0 + cell - 4), color, -1)
            cv2.rectangle(image, (x0, y0), (x0 + cell - 4, y0 + cell - 4), (220, 220, 220), 1)
            cv2.putText(image, str(value), (x0 + 32, y0 + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (25, 25, 25), 2, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def save_per_expert_metrics(metrics: dict[str, dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)


def save_evaluation(payload: dict[str, Any], output_path: Path) -> None:
    """Save evaluation JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def load_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_id_list(value: str | None) -> set[str] | None:
    if value is None:
        return None
    items = {item.strip() for item in value.split(",") if item.strip()}
    return items or None


def rank_of(ranking: list[dict[str, Any]], expert_id: str) -> int | None:
    for idx, item in enumerate(ranking, start=1):
        if str(item.get("expert_id", "")) == expert_id:
            return idx
    return None


def metrics_from_ranks(ranks: list[int]) -> dict[str, float]:
    if not ranks:
        return {"count": 0.0, "top1_accuracy": 0.0, "mrr": 0.0, "mean_rank": 0.0}
    return {
        "count": float(len(ranks)),
        "top1_accuracy": sum(1 for rank in ranks if rank == 1) / len(ranks),
        "mrr": sum(1.0 / rank for rank in ranks) / len(ranks),
        "mean_rank": sum(ranks) / len(ranks),
    }


def expert_ids_from_reports(result_dir: Path) -> set[str]:
    """Infer prototype expert ids from ranking entries in user similarity reports."""
    expert_ids: set[str] = set()
    for report_path in result_dir.rglob("*_similarity.json"):
        report = load_json_file(report_path)
        for ranking_key in ("sequence_ranking", "final_ranking"):
            for item in report.get(ranking_key, []):
                if "expert_id" in item:
                    expert_ids.add(str(item["expert_id"]))
    return expert_ids


def collect_known_pro_reports(result_dir: Path, expert_ids: set[str]) -> list[tuple[str, Path]]:
    """Collect user reports whose user folder name is also a known expert id."""
    reports: list[tuple[str, Path]] = []
    for expert_id in sorted(expert_ids):
        expert_dir = result_dir / expert_id
        if expert_dir.exists():
            reports.extend((expert_id, path) for path in sorted(expert_dir.glob("*_similarity.json")))
    return reports


def evaluate_pro_holdout_reports(result_dir: Path, expert_ids: set[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    final_ranks: list[int] = []
    sequence_ranks: list[int] = []

    reports = collect_known_pro_reports(result_dir, expert_ids)
    for expert_id, report_path in reports:
        report = load_json_file(report_path)
        sample_id = str(report.get("sample_id", report_path.stem.replace("_similarity", "")))
        sequence_ranking = report.get("sequence_ranking", [])
        final_ranking = report.get("final_ranking", [])
        sequence_rank = rank_of(sequence_ranking, expert_id)
        final_rank = rank_of(final_ranking, expert_id)
        if sequence_rank is not None:
            sequence_ranks.append(sequence_rank)
        if final_rank is not None:
            final_ranks.append(final_rank)

        rows.append(
            {
                "expert_id": expert_id,
                "sample_id": sample_id,
                "report_path": str(report_path),
                "status": "ok",
                "sequence_rank": sequence_rank if sequence_rank is not None else "",
                "final_rank": final_rank if final_rank is not None else "",
                "sequence_top1": sequence_ranking[0]["expert_id"] if sequence_ranking else "",
                "final_top1": final_ranking[0]["expert_id"] if final_ranking else "",
                "alpha": report.get("alpha", ""),
            }
        )

    summary = {
        "result_dir": str(result_dir),
        "expert_ids": sorted(expert_ids),
        "found_reports": len(rows),
        "sequence_only": metrics_from_ranks(sequence_ranks),
        "final_ranking": metrics_from_ranks(final_ranks),
    }
    return summary, rows


def save_pro_holdout_outputs(summary: dict[str, Any], rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "expert_id",
            "sample_id",
            "report_path",
            "status",
            "sequence_rank",
            "final_rank",
            "sequence_top1",
            "final_top1",
            "alpha",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved pro holdout metrics: {output_path}")
    print(f"Saved pro holdout CSV: {csv_path}")


def run_pro_holdout_evaluation(
    root: Path,
    sport: str,
    run_id: str,
    result_dir_arg: str | None,
    expert_ids_arg: str | None,
    output_arg: str | None,
    require: bool,
) -> None:
    """Evaluate known-pro user holdout reports when they are available."""
    if sport != "Golf":
        return

    exp_dir = experiment_dir(root, sport, run_id)
    result_dir = Path(result_dir_arg) if result_dir_arg else exp_dir / "results" / "user_similarity"
    output_path = Path(output_arg) if output_arg else exp_dir / "results" / "evaluation" / "pro_holdout_user_metrics.json"

    if not result_dir.exists():
        message = f"User similarity result directory not found for pro holdout evaluation: {result_dir}"
        if require:
            raise FileNotFoundError(message)
        print(f"[skip] {message}")
        return

    expert_ids = parse_id_list(expert_ids_arg) or expert_ids_from_reports(result_dir)
    if not expert_ids:
        message = f"No expert ids found in user similarity rankings: {result_dir}"
        if require:
            raise ValueError(message)
        print(f"[skip] {message}")
        return

    summary, rows = evaluate_pro_holdout_reports(result_dir, expert_ids)
    if not rows:
        message = f"No known-pro user holdout reports found under: {result_dir}"
        if require:
            raise ValueError(message)
        print(f"[skip] {message}")
        return

    save_pro_holdout_outputs(summary, rows, output_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate expert retrieval and Golf known-pro holdout reports.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--sport", choices=["Golf", "Archery"], required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--prototype-store", default=None)
    parser.add_argument("--feature-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--run-id", default=None, help="Experiment run folder name. Defaults to YB_RUN_ID or latest experiment.")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--recall-k", type=int, default=5)
    parser.add_argument("--skip-pro-holdout", action="store_true", help="Skip Golf known-pro user holdout evaluation.")
    parser.add_argument(
        "--require-pro-holdout",
        action="store_true",
        help="Fail instead of skipping when Golf known-pro holdout reports are missing.",
    )
    parser.add_argument(
        "--pro-holdout-result-dir",
        default=None,
        help="User similarity result directory for known-pro holdout evaluation.",
    )
    parser.add_argument(
        "--pro-holdout-output",
        default=None,
        help="Output path for known-pro holdout metrics.",
    )
    parser.add_argument(
        "--pro-holdout-expert-ids",
        default=None,
        help="Comma-separated known expert ids to evaluate as user holdouts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    run_id = resolve_run_id(root, args.sport, args.run_id)
    write_latest_run_id(root, args.sport, run_id)
    exp_dir = experiment_dir(root, args.sport, run_id)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else default_checkpoint_path(exp_dir / "models")
    prototype_path = (
        Path(args.prototype_store)
        if args.prototype_store
        else exp_dir / "prototypes" / "expert_prototypes.pkl"
    )
    feature_dir = Path(args.feature_dir) if args.feature_dir else exp_dir / "features" / "expert"
    output_path = (
        Path(args.output)
        if args.output
        else exp_dir / "results" / "evaluation" / "expert_retrieval_metrics.json"
    )

    if checkpoint_path is None or not checkpoint_path.exists():
        print(f"Checkpoint not found in: {exp_dir / 'models'}")
        print("Run one of 04-1_train_atcn.py, 04-2_train_transformer.py, or 04-3_train_stgcn.py first.")
        return
    if not prototype_path.exists():
        print(f"Prototype store not found: {prototype_path}")
        print("Run 05_build_expert_prototypes.py first.")
        return

    feature_paths = collect_expert_features(root, args.sport, feature_dir)
    if not feature_paths:
        print(f"No expert feature files found: {feature_dir}")
        print("Run 03_preprocess_and_build_features.py first.")
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
    results = []

    print(f"Evaluation sport={args.sport}, queries={len(feature_paths)}, device={device}, run_id={run_id}")
    for feature_path in feature_paths:
        report = infer_one(feature_path, model, prototype_store, device, alpha=args.alpha)
        expert_id = infer_expert_id(str(report["sample_id"]))
        correct_rank = rank_of_correct_expert(report["final_ranking"], expert_id)
        top1 = report["final_ranking"][0]["expert_id"] if report["final_ranking"] else ""
        row = {
            "sample_id": report["sample_id"],
            "expert_id": expert_id,
            "correct_rank": correct_rank,
            "top1_expert_id": top1,
            "final_ranking": report["final_ranking"],
            "low_similarity_steps": report["low_similarity_steps"],
        }
        results.append(row)
        print(f"[query] {report['sample_id']} correct={expert_id} rank={correct_rank} top1={top1}")

    metrics = compute_metrics(results, recall_k=args.recall_k)
    per_expert_metrics = compute_per_expert_metrics(results)
    labels, confusion_matrix = build_confusion_matrix(results)
    payload = {
        "sport": args.sport,
        "run_id": run_id,
        "alpha": args.alpha,
        "recall_k": args.recall_k,
        "query_count": len(results),
        "metrics": metrics,
        "per_expert_metrics": per_expert_metrics,
        "confusion_labels": labels,
        "confusion_matrix": confusion_matrix.tolist(),
        "results": results,
    }
    save_evaluation(payload, output_path)
    save_per_expert_metrics(per_expert_metrics, output_path.parent / "per_expert_metrics.json")
    save_confusion_csv(labels, confusion_matrix, output_path.parent / "confusion_matrix.csv")
    save_confusion_png(labels, confusion_matrix, output_path.parent / "confusion_matrix.png")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved evaluation: {output_path}")
    print(f"Saved per-expert metrics: {output_path.parent / 'per_expert_metrics.json'}")
    print(f"Saved confusion matrix: {output_path.parent / 'confusion_matrix.png'}")

    if not args.skip_pro_holdout:
        run_pro_holdout_evaluation(
            root=root,
            sport=args.sport,
            run_id=run_id,
            result_dir_arg=args.pro_holdout_result_dir,
            expert_ids_arg=args.pro_holdout_expert_ids,
            output_arg=args.pro_holdout_output,
            require=args.require_pro_holdout,
        )


if __name__ == "__main__":
    main()
