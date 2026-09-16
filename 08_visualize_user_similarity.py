from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from encoder_loader import build_model_from_checkpoint, default_checkpoint_path as find_default_checkpoint


DEFAULT_SPORT = "Golf"
SPORTS = ["Golf", "Archery"]
USER_COLOR = (215, 80, 35)
EXPERT_COLORS = [
    (45, 130, 245),
    (30, 165, 95),
    (165, 90, 220),
    (25, 185, 220),
    (220, 125, 30),
    (120, 120, 120),
]

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


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def scalar(value: np.ndarray | Any) -> Any:
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def load_feature(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "path": path,
            "sequence": data["sequence"].astype(np.float32),
            "steps": data["steps"].astype(np.float32),
            "sample_id": str(scalar(data["sample_id"])),
            "joint_names": [str(item) for item in data["joint_names"].tolist()],
            "step_names": [str(item) for item in data["step_names"].tolist()],
        }


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to build TSNE embedding plots. Install torch first.") from exc

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_embedding_model(checkpoint_path: Path, device: Any) -> tuple[Any, dict[str, Any]]:
    checkpoint = load_checkpoint(checkpoint_path)
    model = build_model_from_checkpoint(checkpoint, device)
    return model, checkpoint


def resolve_user_feature_path(root: Path, sport: str, report: dict[str, Any], run_id: str | None) -> Path:
    feature_path = Path(report["feature_path"])
    if feature_path.exists():
        return feature_path

    sample_id = str(report["sample_id"])
    user_id = person_id_from_sample(sample_id)
    candidates = []
    if run_id:
        candidates.append(root / "data" / sport / "experiments" / run_id / "features" / "user" / user_id / f"{sample_id}.npz")
    candidates.append(root / "data" / sport / "features" / "user" / user_id / f"{sample_id}.npz")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return feature_path


def person_id_from_sample(sample_id: str) -> str:
    return sample_id.rsplit("_", 1)[0]


def collect_result_files(root: Path, sport: str, result_path: str | None) -> list[Path]:
    if result_path:
        path = Path(result_path)
        return sorted(path.rglob("*.json")) if path.is_dir() else [path]
    result_dir = root / "data" / sport / "results" / "user_similarity"
    return sorted(result_dir.rglob("*.json")) if result_dir.exists() else []


def latest_run_id(root: Path, sport: str) -> str | None:
    experiments_dir = root / "data" / sport / "experiments"
    if not experiments_dir.exists():
        return None
    marker_path = experiments_dir / ".latest_run_id"
    if marker_path.exists():
        marker = marker_path.read_text(encoding="utf-8-sig").strip()
        if marker:
            return marker
    run_dirs = [
        path
        for path in experiments_dir.iterdir()
        if path.is_dir() and (path / "results" / "user_similarity").exists()
    ]
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda path: path.stat().st_mtime).name


def resolve_run_id(root: Path, sport: str, run_id: str | None, result_path: str | None) -> str | None:
    if result_path:
        return None
    if run_id and run_id != "latest":
        return run_id
    env_run_id = os.environ.get("YB_RUN_ID")
    if env_run_id and not run_id:
        return env_run_id
    return latest_run_id(root, sport)


def collect_run_result_files(root: Path, sport: str, run_id: str | None, result_path: str | None) -> list[Path]:
    if result_path:
        return collect_result_files(root, sport, result_path)
    if run_id:
        result_dir = root / "data" / sport / "experiments" / run_id / "results" / "user_similarity"
        return sorted(result_dir.rglob("*.json")) if result_dir.exists() else []
    return collect_result_files(root, sport, None)


def collect_expert_features(root: Path, sport: str, expert_id: str, run_id: str | None) -> list[Path]:
    base_dir = (
        root / "data" / sport / "experiments" / run_id / "features" / "expert"
        if run_id
        else root / "data" / sport / "features" / "expert"
    )
    feature_dir = base_dir / expert_id
    if feature_dir.exists():
        return sorted(feature_dir.rglob("*.npz"))
    return sorted(path for path in base_dir.rglob("*.npz") if path.stem.startswith(f"{expert_id}_"))


def collect_subject_feature_files(root: Path, sport: str, subject_type: str, run_id: str | None) -> list[Path]:
    if run_id:
        experiment_dir = root / "data" / sport / "experiments" / run_id / "features" / subject_type
        if experiment_dir.exists():
            return sorted(path for path in experiment_dir.rglob("*.npz") if path.name != ".gitkeep")

    base_dir = root / "data" / sport / "features" / subject_type
    if base_dir.exists():
        return sorted(path for path in base_dir.rglob("*.npz") if path.name != ".gitkeep")
    return []


def feature_distance(a: np.ndarray, b: np.ndarray) -> float:
    length = min(a.shape[0], b.shape[0])
    if length <= 0:
        return float("inf")
    diff = a[:length, :, :2] - b[:length, :, :2]
    return float(np.mean(diff * diff))


def choose_best_expert_sample(
    root: Path,
    sport: str,
    expert_id: str,
    user_feature: dict[str, Any],
    mode: str,
    run_id: str | None,
    step_index: int | None = None,
) -> dict[str, Any] | None:
    candidates = collect_expert_features(root, sport, expert_id, run_id)
    if not candidates:
        return None

    best_feature: dict[str, Any] | None = None
    best_score = float("inf")
    for path in candidates:
        feature = load_feature(path)
        if mode == "sequence":
            score = feature_distance(user_feature["sequence"], feature["sequence"])
        else:
            assert step_index is not None
            score = feature_distance(user_feature["steps"][step_index], feature["steps"][step_index])
        if score < best_score:
            best_score = score
            best_feature = feature
    return best_feature


def joint_lookup(joint_names: list[str]) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(joint_names)}


def pose_bounds(sequences: list[np.ndarray]) -> tuple[float, float, float, float]:
    xy = np.concatenate([sequence[:, :, :2].reshape(-1, 2) for sequence in sequences], axis=0)
    finite = np.isfinite(xy).all(axis=1)
    if not np.any(finite):
        return 0.0, 1.0, 0.0, 1.0
    xy = xy[finite]
    min_x, min_y = np.min(xy, axis=0)
    max_x, max_y = np.max(xy, axis=0)
    pad_x = max((max_x - min_x) * 0.12, 0.25)
    pad_y = max((max_y - min_y) * 0.12, 0.25)
    return float(min_x - pad_x), float(max_x + pad_x), float(min_y - pad_y), float(max_y + pad_y)


def point_to_pixel(
    point: np.ndarray,
    bounds: tuple[float, float, float, float],
    width: int,
    height: int,
    margin: int,
) -> tuple[int, int]:
    min_x, max_x, min_y, max_y = bounds
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    x = margin + (float(point[0]) - min_x) / span_x * (width - 2 * margin)
    y = margin + (float(point[1]) - min_y) / span_y * (height - 2 * margin)
    return int(round(x)), int(round(y))


def draw_text(
    image: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float = 0.55,
    color: tuple[int, int, int] = (35, 35, 35),
    thickness: int = 1,
) -> None:
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_skeleton(
    image: np.ndarray,
    pose: np.ndarray,
    joint_names: list[str],
    bounds: tuple[float, float, float, float],
    color: tuple[int, int, int],
    label: str,
    label_pos: tuple[int, int],
    min_score: float = 0.05,
) -> None:
    width = image.shape[1]
    height = image.shape[0]
    margin = 55
    lookup = joint_lookup(joint_names)

    for start, end in SKELETON_EDGES:
        if start not in lookup or end not in lookup:
            continue
        a = pose[lookup[start]]
        b = pose[lookup[end]]
        if pose.shape[1] >= 3 and (a[2] < min_score or b[2] < min_score):
            continue
        pt_a = point_to_pixel(a[:2], bounds, width, height, margin)
        pt_b = point_to_pixel(b[:2], bounds, width, height, margin)
        cv2.line(image, pt_a, pt_b, color, 2, cv2.LINE_AA)

    for joint in pose:
        if pose.shape[1] >= 3 and joint[2] < min_score:
            continue
        cv2.circle(image, point_to_pixel(joint[:2], bounds, width, height, margin), 4, color, -1, cv2.LINE_AA)

    cv2.rectangle(image, (label_pos[0] - 8, label_pos[1] - 18), (label_pos[0] + 18, label_pos[1] + 2), color, -1)
    draw_text(image, label, (label_pos[0] + 26, label_pos[1]), color=(25, 25, 25))


def write_comparison_video(
    output_path: Path,
    user_sequence: np.ndarray,
    expert_sequence: np.ndarray,
    joint_names: list[str],
    title: str,
    expert_label: str,
    expert_color: tuple[int, int, int],
    fps: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 960, 720
    bounds = pose_bounds([user_sequence, expert_sequence])
    frame_count = min(user_sequence.shape[0], expert_sequence.shape[0])
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    for frame_idx in range(frame_count):
        image = np.full((height, width, 3), 248, dtype=np.uint8)
        draw_text(image, title, (35, 42), scale=0.8, color=(15, 50, 100), thickness=2)
        draw_text(image, f"frame {frame_idx + 1}/{frame_count}", (35, 72), scale=0.55)
        draw_skeleton(image, user_sequence[frame_idx], joint_names, bounds, USER_COLOR, "user", (700, 42))
        draw_skeleton(image, expert_sequence[frame_idx], joint_names, bounds, expert_color, expert_label, (700, 72))
        writer.write(image)

    writer.release()


def draw_bar(
    image: np.ndarray,
    label: str,
    score: float,
    x: int,
    y: int,
    width: int,
    color: tuple[int, int, int],
) -> None:
    draw_text(image, label[:28], (x, y), scale=0.48)
    cv2.rectangle(image, (x + 205, y - 14), (x + 205 + width, y), (225, 225, 225), -1)
    fill = int(np.clip(score, 0.0, 1.0) * width)
    cv2.rectangle(image, (x + 205, y - 14), (x + 205 + fill, y), color, -1)
    draw_text(image, f"{score:.4f}", (x + 215 + width, y), scale=0.45)


def save_summary_image(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((1080, 1680, 3), 250, dtype=np.uint8)
    sample_id = report["sample_id"]
    final = report.get("final_ranking", [])
    sequence = report.get("sequence_ranking", [])
    step_similarity = report.get("step_similarity", {})
    alpha = float(report.get("alpha", 0.0))

    draw_text(image, f"User Similarity Summary: {sample_id}", (40, 55), scale=1.0, color=(10, 45, 95), thickness=2)
    draw_text(image, f"Sport: {report.get('sport', '')}    alpha: {alpha:.2f}", (40, 95), scale=0.65)

    y = 150
    draw_text(image, "Final Ranking (sequence-step score)", (40, y), scale=0.72, color=(15, 50, 100), thickness=2)
    for idx, item in enumerate(final[:6], start=1):
        score = float(item["score"])
        label = f"{idx}. {item['expert_id']}"
        draw_bar(image, label, score, 40, y + 42 * idx, 360, (55, 145, 235))
        draw_text(
            image,
            f"seq {float(item.get('sequence_score', 0.0)):.4f} / step {float(item.get('mean_step_score', 0.0)):.4f}",
            (735, y + 42 * idx),
            scale=0.45,
        )

    y = 475
    draw_text(image, "Sequence Ranking", (40, y), scale=0.72, color=(15, 50, 100), thickness=2)
    for idx, item in enumerate(sequence[:6], start=1):
        draw_bar(image, f"{idx}. {item['expert_id']}", float(item["score"]), 40, y + 38 * idx, 360, (45, 170, 105))

    x2 = 980
    y2 = 150
    draw_text(image, "Step Top-1 Ranking", (x2, y2), scale=0.72, color=(15, 50, 100), thickness=2)
    for idx, (step_name, ranking) in enumerate(step_similarity.items(), start=1):
        if not ranking:
            continue
        top = ranking[0]
        yy = y2 + 38 * idx
        draw_text(image, step_name, (x2, yy), scale=0.50)
        draw_text(image, str(top["expert_id"]), (x2 + 215, yy), scale=0.50)
        draw_text(image, f"{float(top['score']):.4f}", (x2 + 510, yy), scale=0.50)

    y3 = 515
    draw_text(image, "Low Similarity Steps", (x2, y3), scale=0.72, color=(15, 50, 100), thickness=2)
    low_steps = report.get("low_similarity_steps", [])
    if low_steps:
        for idx, item in enumerate(low_steps[:8], start=1):
            draw_text(
                image,
                f"{idx}. {item.get('step', '')}: {item.get('best_expert_id', '')} ({float(item.get('best_score', 0.0)):.4f})",
                (x2, y3 + 38 * idx),
                scale=0.50,
            )
    else:
        draw_text(image, "None", (x2, y3 + 38), scale=0.55)

    draw_text(image, "Generated by 08_visualize_user_similarity.py", (40, 1035), scale=0.45, color=(120, 120, 120))
    cv2.imwrite(str(output_path), image)


def expert_color_map(expert_ids: list[str]) -> dict[str, tuple[int, int, int]]:
    mapping: dict[str, tuple[int, int, int]] = {}
    for idx, expert_id in enumerate(expert_ids):
        mapping[expert_id] = EXPERT_COLORS[idx % len(EXPERT_COLORS)]
    return mapping


def safe_filename(text: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in text)
    return cleaned.strip("_") or "step"


def sample_sequence_tsne_path(out_dir: Path) -> Path:
    return out_dir / "tsne_sequence.png"


def sample_step_tsne_path(out_dir: Path, step_idx: int, step_name: str) -> Path:
    return out_dir / f"tsne_step_{step_idx + 1:02d}_{safe_filename(step_name)}.png"


def infer_run_id_from_path(root: Path, sport: str, path: Path) -> str | None:
    experiments_dir = root / "data" / sport / "experiments"
    try:
        relative = path.resolve().relative_to(experiments_dir.resolve())
    except (OSError, ValueError):
        return None
    return relative.parts[0] if relative.parts else None


def infer_run_id_for_report(
    root: Path,
    sport: str,
    report_path: Path,
    output_root: Path,
    report: dict[str, Any],
) -> str | None:
    output_run_id = infer_run_id_from_output_root(root, sport, output_root)
    if output_run_id:
        return output_run_id

    report_run_id = infer_run_id_from_path(root, sport, report_path)
    if report_run_id:
        return report_run_id

    feature_path = Path(str(report.get("feature_path", "")))
    for candidate in (feature_path, root / feature_path):
        feature_run_id = infer_run_id_from_path(root, sport, candidate)
        if feature_run_id:
            return feature_run_id
    return None


def infer_run_id_from_reports(root: Path, sport: str, report_paths: list[Path]) -> str | None:
    for report_path in report_paths:
        run_id = infer_run_id_from_path(root, sport, report_path)
        if run_id:
            return run_id
        try:
            report = load_json(report_path)
        except (OSError, json.JSONDecodeError):
            continue
        feature_path = Path(str(report.get("feature_path", "")))
        for candidate in (feature_path, root / feature_path):
            run_id = infer_run_id_from_path(root, sport, candidate)
            if run_id:
                return run_id
    return None


def default_checkpoint_path(root: Path, sport: str, run_id: str | None) -> Path | None:
    if not run_id:
        return None
    return find_default_checkpoint(root / "data" / sport / "experiments" / run_id / "models")


def compute_tsne(points: np.ndarray, perplexity: float, seed: int) -> tuple[np.ndarray, float]:
    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for TSNE plots. Install scikit-learn first.") from exc

    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] < 2:
        raise ValueError("TSNE needs at least two embeddings.")

    actual_perplexity = min(max(float(perplexity), 1.0), float(points.shape[0] - 1))
    init = "pca" if points.shape[0] >= 4 and points.shape[1] >= 2 else "random"
    kwargs = {
        "n_components": 2,
        "perplexity": actual_perplexity,
        "init": init,
        "learning_rate": "auto",
        "metric": "cosine",
        "random_state": seed,
    }
    try:
        coords = TSNE(max_iter=1000, **kwargs).fit_transform(points)
    except TypeError:
        coords = TSNE(n_iter=1000, **kwargs).fit_transform(points)
    return coords.astype(np.float32), actual_perplexity


def tsne_point_positions(
    coords: np.ndarray,
    left: int,
    top: int,
    width: int,
    height: int,
) -> np.ndarray:
    finite = np.isfinite(coords).all(axis=1)
    if not np.any(finite):
        return np.column_stack(
            [
                np.full(coords.shape[0], left + width // 2, dtype=np.int32),
                np.full(coords.shape[0], top + height // 2, dtype=np.int32),
            ]
        )

    valid = coords[finite]
    min_xy = valid.min(axis=0)
    max_xy = valid.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)
    scaled = (coords - min_xy) / span
    x = left + np.clip(scaled[:, 0], 0.0, 1.0) * width
    y = top + (1.0 - np.clip(scaled[:, 1], 0.0, 1.0)) * height
    return np.column_stack([np.round(x).astype(np.int32), np.round(y).astype(np.int32)])


def draw_tsne_plot(
    output_path: Path,
    coords: np.ndarray,
    records: list[dict[str, Any]],
    title: str,
    subtitle: str,
    perplexity: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1400, 1000
    plot_left, plot_top = 80, 115
    plot_width, plot_height = 940, 800
    image = np.full((height, width, 3), 250, dtype=np.uint8)

    expert_ids = sorted({str(record["person_id"]) for record in records if record["subject_type"] == "expert"})
    colors = expert_color_map(expert_ids)
    counts = {
        "user": sum(1 for record in records if record["subject_type"] == "user"),
        "expert": sum(1 for record in records if record["subject_type"] == "expert"),
    }

    draw_text(image, title, (40, 52), scale=0.95, color=(10, 45, 95), thickness=2)
    draw_text(
        image,
        f"{subtitle}  |  user={counts['user']} expert={counts['expert']} perplexity={perplexity:.1f}",
        (40, 88),
        scale=0.55,
        color=(70, 70, 70),
    )
    cv2.rectangle(
        image,
        (plot_left, plot_top),
        (plot_left + plot_width, plot_top + plot_height),
        (215, 215, 215),
        1,
        cv2.LINE_AA,
    )

    positions = tsne_point_positions(coords, plot_left + 24, plot_top + 24, plot_width - 48, plot_height - 48)
    draw_labels = len(records) <= 80
    for position, record in zip(positions, records):
        x, y = int(position[0]), int(position[1])
        subject_type = str(record["subject_type"])
        if subject_type == "user":
            cv2.drawMarker(image, (x, y), USER_COLOR, cv2.MARKER_TRIANGLE_UP, 14, 2, cv2.LINE_AA)
        else:
            color = colors.get(str(record["person_id"]), (80, 80, 80))
            cv2.circle(image, (x, y), 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(image, (x, y), 5, color, -1, cv2.LINE_AA)
            cv2.circle(image, (x, y), 6, (40, 40, 40), 1, cv2.LINE_AA)
        if draw_labels:
            draw_text(image, str(record["sample_id"])[:24], (x + 8, y - 4), scale=0.34, color=(55, 55, 55))

    legend_x = 1065
    draw_text(image, "Legend", (legend_x, 135), scale=0.72, color=(15, 50, 100), thickness=2)
    cv2.drawMarker(image, (legend_x + 10, 175), USER_COLOR, cv2.MARKER_TRIANGLE_UP, 14, 2, cv2.LINE_AA)
    draw_text(image, f"user samples ({counts['user']})", (legend_x + 32, 181), scale=0.50)

    y = 220
    draw_text(image, "expert samples", (legend_x, y), scale=0.55, color=(65, 65, 65), thickness=1)
    max_legend = 20
    for idx, expert_id in enumerate(expert_ids[:max_legend], start=1):
        yy = y + idx * 30
        cv2.circle(image, (legend_x + 10, yy - 5), 6, colors[expert_id], -1, cv2.LINE_AA)
        draw_text(image, expert_id[:28], (legend_x + 32, yy), scale=0.48)
    if len(expert_ids) > max_legend:
        draw_text(image, f"+ {len(expert_ids) - max_legend} more experts", (legend_x, y + (max_legend + 1) * 30), scale=0.48)

    cv2.imwrite(str(output_path), image)


def embed_feature_records(
    model: Any,
    torch_module: Any,
    device: Any,
    feature_paths: list[tuple[str, Path]],
) -> list[dict[str, Any]]:
    return [
        embed_feature_record(model, torch_module, device, subject_type, path)
        for subject_type, path in feature_paths
    ]


def embed_feature_record(
    model: Any,
    torch_module: Any,
    device: Any,
    subject_type: str,
    path: Path,
    feature: dict[str, Any] | None = None,
) -> dict[str, Any]:
    feature = feature or load_feature(path)
    with torch_module.no_grad():
        sequence = torch_module.from_numpy(feature["sequence"]).unsqueeze(0).float().to(device)
        steps = torch_module.from_numpy(feature["steps"]).unsqueeze(0).float().to(device)
        outputs = model(sequence, steps)
    return {
        "subject_type": subject_type,
        "person_id": person_id_from_sample(str(feature["sample_id"])),
        "sample_id": str(feature["sample_id"]),
        "path": str(path),
        "step_names": list(feature["step_names"]),
        "sequence_embedding": outputs["sequence_embedding"][0].detach().cpu().numpy().astype(np.float32),
        "step_embeddings": outputs["step_embeddings"][0].detach().cpu().numpy().astype(np.float32),
    }


def prepare_tsne_context(
    root: Path,
    sport: str,
    run_id: str | None,
    checkpoint_path: Path | None,
) -> dict[str, Any] | None:
    if checkpoint_path is None:
        print("[warn] TSNE skipped: run_id or --checkpoint is required to locate an encoder checkpoint.")
        return None
    if not checkpoint_path.exists():
        print(f"[warn] TSNE skipped: checkpoint not found: {checkpoint_path}")
        return None

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to build TSNE embedding plots. Install torch first.") from exc

    expert_paths = collect_subject_feature_files(root, sport, "expert", run_id)
    if not expert_paths:
        print("[warn] TSNE skipped: no expert feature files found.")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_embedding_model(checkpoint_path, device)
    if checkpoint["sport"] != sport:
        raise ValueError(f"Checkpoint sport={checkpoint['sport']} does not match --sport {sport}")
    step_names = [str(item) for item in checkpoint["shape_info"]["step_names"]]
    expert_records = embed_feature_records(
        model,
        torch,
        device,
        [("expert", path) for path in expert_paths],
    )
    print(f"TSNE context ready: expert_embeddings={len(expert_records)}, device={device}")
    return {
        "model": model,
        "torch": torch,
        "device": device,
        "step_names": step_names,
        "expert_records": expert_records,
    }


def sample_tsne_paths(out_dir: Path, step_names: list[str]) -> list[Path]:
    return [sample_sequence_tsne_path(out_dir)] + [
        sample_step_tsne_path(out_dir, step_idx, step_name)
        for step_idx, step_name in enumerate(step_names)
    ]


def save_sample_tsne_visualizations(
    out_dir: Path,
    sport: str,
    report: dict[str, Any],
    user_feature: dict[str, Any],
    tsne_context: dict[str, Any] | None,
    overwrite: bool,
    perplexity: float,
    seed: int,
) -> None:
    if tsne_context is None:
        return

    step_names = list(user_feature["step_names"])
    expected_paths = sample_tsne_paths(out_dir, step_names)
    if not overwrite and all(path.exists() for path in expected_paths):
        return

    user_record = embed_feature_record(
        tsne_context["model"],
        tsne_context["torch"],
        tsne_context["device"],
        "user",
        Path(user_feature["path"]),
        user_feature,
    )
    records = list(tsne_context["expert_records"]) + [user_record]
    if len(records) < 2:
        print(f"[warn] TSNE skipped for {report.get('sample_id', '')}: need at least one expert and one user embedding.")
        return

    sample_id = str(report["sample_id"])
    sequence_top = ""
    if report.get("sequence_ranking"):
        top = report["sequence_ranking"][0]
        sequence_top = f" / top={top.get('expert_id', '')} score={float(top.get('score', 0.0)):.4f}"

    sequence_path = sample_sequence_tsne_path(out_dir)
    if overwrite or not sequence_path.exists():
        sequence_points = np.stack([record["sequence_embedding"] for record in records], axis=0)
        sequence_coords, actual_perplexity = compute_tsne(sequence_points, perplexity, seed)
        draw_tsne_plot(
            sequence_path,
            sequence_coords,
            records,
            "t-SNE: Sequence Embeddings",
            f"{sport} / {sample_id} vs all expert embeddings{sequence_top}",
            actual_perplexity,
        )
        print(f"[save] {sequence_path}")

    for step_idx, step_name in enumerate(step_names):
        output_path = sample_step_tsne_path(out_dir, step_idx, step_name)
        if overwrite or not output_path.exists():
            step_points = np.stack([record["step_embeddings"][step_idx] for record in records], axis=0)
            step_coords, actual_perplexity = compute_tsne(step_points, perplexity, seed)
            step_top = ""
            ranking = report.get("step_similarity", {}).get(step_name, [])
            if ranking:
                top = ranking[0]
                step_top = f" / top={top.get('expert_id', '')} score={float(top.get('score', 0.0)):.4f}"
            draw_tsne_plot(
                output_path,
                step_coords,
                records,
                f"t-SNE: Step {step_idx + 1} Embeddings",
                f"{sport} / {sample_id} / {step_name}{step_top}",
                actual_perplexity,
            )
            print(f"[save] {output_path}")


def process_report(
    root: Path,
    sport: str,
    report_path: Path,
    output_root: Path,
    fps: int,
    overwrite: bool,
    tsne_context: dict[str, Any] | None,
    tsne_perplexity: float,
    tsne_seed: int,
) -> None:
    report = load_json(report_path)
    sample_id = str(report["sample_id"])
    user_id = person_id_from_sample(sample_id)
    out_dir = output_root / user_id / sample_id
    summary_path = out_dir / "summary.png"
    sequence_video_path = out_dir / "sequence_compare.mp4"
    step_video_path = out_dir / "step_sequence_compare.mp4"

    run_id = infer_run_id_for_report(root, sport, report_path, output_root, report)
    user_feature_path = resolve_user_feature_path(root, sport, report, run_id)
    user_feature = load_feature(user_feature_path)
    joint_names = user_feature["joint_names"]
    step_names = user_feature["step_names"]
    tsne_paths = sample_tsne_paths(out_dir, step_names) if tsne_context is not None else []
    existing_outputs = [summary_path, sequence_video_path, step_video_path] + tsne_paths
    if not overwrite and existing_outputs and all(path.exists() for path in existing_outputs):
        print(f"[skip] {out_dir}")
        return

    if overwrite or not summary_path.exists():
        save_summary_image(report, summary_path)
        print(f"[save] {summary_path}")

    try:
        save_sample_tsne_visualizations(
            out_dir=out_dir,
            sport=sport,
            report=report,
            user_feature=user_feature,
            tsne_context=tsne_context,
            overwrite=overwrite,
            perplexity=tsne_perplexity,
            seed=tsne_seed,
        )
    except Exception as exc:
        print(f"[warn] Failed to save TSNE for {sample_id}: {type(exc).__name__}: {exc}")

    sequence_top = report["sequence_ranking"][0]["expert_id"]
    sequence_expert = choose_best_expert_sample(root, sport, sequence_top, user_feature, "sequence", run_id)
    if not overwrite and sequence_video_path.exists():
        pass
    elif sequence_expert is not None:
        write_comparison_video(
            sequence_video_path,
            user_feature["sequence"],
            sequence_expert["sequence"],
            joint_names,
            f"Sequence comparison: {sample_id}",
            f"{sequence_top} / {sequence_expert['sample_id']}",
            (45, 130, 245),
            fps,
        )
        print(f"[save] {sequence_video_path}")
    else:
        print(f"[warn] No expert feature found for sequence top: {sequence_top}")

    step_winners: list[str] = []
    for step_name in step_names:
        ranking = report["step_similarity"].get(step_name, [])
        if ranking:
            step_winners.append(str(ranking[0]["expert_id"]))
    colors = expert_color_map(sorted(set(step_winners)))

    step_user_parts = []
    step_expert_parts = []
    step_titles = []
    step_labels = []
    step_colors = []
    for step_idx, step_name in enumerate(step_names):
        ranking = report["step_similarity"].get(step_name, [])
        if not ranking:
            continue
        expert_id = str(ranking[0]["expert_id"])
        expert_feature = choose_best_expert_sample(root, sport, expert_id, user_feature, "step", run_id, step_idx)
        if expert_feature is None:
            print(f"[warn] No expert feature found for step={step_name}, expert={expert_id}")
            continue
        step_user_parts.append(user_feature["steps"][step_idx])
        step_expert_parts.append(expert_feature["steps"][step_idx])
        step_titles.append(f"{step_name}: top {expert_id} score={float(ranking[0]['score']):.4f}")
        step_labels.append(f"{expert_id} / {expert_feature['sample_id']}")
        step_colors.append(colors[expert_id])

    if step_user_parts and (overwrite or not step_video_path.exists()):
        write_step_sequence_video(
            step_video_path,
            step_user_parts,
            step_expert_parts,
            joint_names,
            step_titles,
            step_labels,
            step_colors,
            fps,
        )
        print(f"[save] {step_video_path}")


def write_step_sequence_video(
    output_path: Path,
    user_steps: list[np.ndarray],
    expert_steps: list[np.ndarray],
    joint_names: list[str],
    titles: list[str],
    expert_labels: list[str],
    expert_colors: list[tuple[int, int, int]],
    fps: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1120, 760
    bounds = pose_bounds(user_steps + expert_steps)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    legend_items: dict[str, tuple[int, int, int]] = {}
    for label, color in zip(expert_labels, expert_colors):
        expert_id = label.split(" / ", 1)[0]
        legend_items[expert_id] = color

    for step_idx, (user_step, expert_step) in enumerate(zip(user_steps, expert_steps)):
        frame_count = min(user_step.shape[0], expert_step.shape[0])
        for frame_idx in range(frame_count):
            image = np.full((height, width, 3), 248, dtype=np.uint8)
            draw_text(image, "Step-sequence comparison", (35, 42), scale=0.8, color=(15, 50, 100), thickness=2)
            draw_text(image, titles[step_idx], (35, 74), scale=0.58)
            draw_text(image, f"step {step_idx + 1}/{len(user_steps)}  frame {frame_idx + 1}/{frame_count}", (35, 104), scale=0.52)
            draw_skeleton(image, user_step[frame_idx], joint_names, bounds, USER_COLOR, "user", (760, 42))
            draw_skeleton(
                image,
                expert_step[frame_idx],
                joint_names,
                bounds,
                expert_colors[step_idx],
                expert_labels[step_idx],
                (760, 72),
            )

            y = 120
            draw_text(image, "Legend", (760, y), scale=0.55, color=(15, 50, 100), thickness=2)
            for idx, (expert_id, color) in enumerate(legend_items.items(), start=1):
                yy = y + idx * 30
                cv2.rectangle(image, (760, yy - 15), (782, yy + 3), color, -1)
                draw_text(image, expert_id, (792, yy), scale=0.48)
            writer.write(image)

    writer.release()


def infer_run_id_from_output_root(root: Path, sport: str, output_root: Path) -> str | None:
    experiments_dir = root / "data" / sport / "experiments"
    try:
        relative = output_root.resolve().relative_to(experiments_dir.resolve())
    except ValueError:
        return None
    return relative.parts[0] if relative.parts else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize user similarity JSON results.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--sport", default=DEFAULT_SPORT, choices=SPORTS)
    parser.add_argument("--result-path", default=None, help="One result JSON or a directory of result JSON files.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-id", default=None, help="Results run folder to visualize. Defaults to YB_RUN_ID or latest.")
    parser.add_argument("--checkpoint", default=None, help="Optional encoder checkpoint for TSNE embedding plots.")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-seed", type=int, default=42)
    parser.add_argument("--skip-tsne", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    run_id = resolve_run_id(root, args.sport, args.run_id, args.result_path)
    output_root = (
        Path(args.output_dir)
        if args.output_dir
        else (
            root / "data" / args.sport / "experiments" / run_id / "results" / "visualization"
            if run_id
            else root / "data" / args.sport / "results" / "visualization"
        )
    )
    reports = collect_run_result_files(root, args.sport, run_id, args.result_path)
    if not reports:
        print("No user similarity JSON files found. Run 06_infer_user_similarity.py first.")
        return
    print(f"Visualizing sport={args.sport}, reports={len(reports)}, run_id={run_id or 'custom'}")

    tsne_run_id = run_id or infer_run_id_from_reports(root, args.sport, reports)
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else default_checkpoint_path(root, args.sport, tsne_run_id)
    )
    tsne_context = None
    if not args.skip_tsne:
        try:
            tsne_context = prepare_tsne_context(
                root=root,
                sport=args.sport,
                run_id=tsne_run_id,
                checkpoint_path=checkpoint_path,
            )
        except Exception as exc:
            print(f"[warn] Failed to prepare TSNE embeddings: {type(exc).__name__}: {exc}")

    for report_path in reports:
        try:
            process_report(
                root,
                args.sport,
                report_path,
                output_root,
                args.fps,
                args.overwrite,
                tsne_context,
                args.tsne_perplexity,
                args.tsne_seed,
            )
        except Exception as exc:
            print(f"[warn] Failed to visualize {report_path}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
