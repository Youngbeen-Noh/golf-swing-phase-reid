from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np


GOLF_JOINT_NAMES = [
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_shoulder",
    "right_shoulder",
    "left_knee",
    "right_knee",
    "left_heel",
    "right_heel",
    "left_toe",
    "right_toe",
    "left_eye",
    "right_eye",
    "left_elbow",
    "right_elbow",
    "left_ear",
    "right_ear",
]

ARCHERY_JOINT_NAMES = [
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_shoulder",
    "right_shoulder",
    "left_knee",
    "right_knee",
    "left_heel",
    "right_heel",
    "left_eye",
    "right_eye",
    "left_elbow",
    "right_elbow",
    "left_ear",
    "right_ear",
]

SPORT_JOINT_NAMES = {
    "Golf": GOLF_JOINT_NAMES,
    "Archery": ARCHERY_JOINT_NAMES,
}

SUBJECT_TYPES = ["expert", "user"]
DEFAULT_CHANNELS = ["x", "y", "score"]
PIXEL_CHANNELS = ["pixel_x", "pixel_y", "score"]


def load_json(json_path: Path) -> dict[str, Any]:
    """Load one extraction JSON file."""
    with json_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def coordinates_to_tensor(
    data: dict[str, Any],
    joint_names: list[str],
    channels: list[str],
    json_path: Path,
) -> np.ndarray:
    """Convert frame-wise JSON coordinates to a pose tensor.

    Args:
        data: Extraction JSON dictionary.
        joint_names: Fixed sport-specific joint order.
        channels: Channel names for C dimension, e.g. x/y/score.
        json_path: Source JSON path used for warnings.

    Returns:
        Pose tensor with shape (T, J, C).
    """
    coordinates = data.get("coordinates", [])
    if not isinstance(coordinates, list):
        raise ValueError(f"coordinates must be a list: {json_path}")

    frame_count = int(data.get("frame_count", len(coordinates)))
    pose = np.zeros((frame_count, len(joint_names), len(channels)), dtype=np.float32)
    missing_joints: set[str] = set()

    for frame_idx in range(frame_count):
        frame_data = coordinates[frame_idx] if frame_idx < len(coordinates) else {}
        landmarks = frame_data.get("landmarks", {}) if isinstance(frame_data, dict) else {}
        if not isinstance(landmarks, dict):
            landmarks = {}

        for joint_idx, joint_name in enumerate(joint_names):
            values = landmarks.get(joint_name)
            if values is None:
                missing_joints.add(joint_name)
                continue
            for channel_idx, channel_name in enumerate(channels):
                pose[frame_idx, joint_idx, channel_idx] = float(values.get(channel_name, 0.0) or 0.0)

    if missing_joints:
        warnings.warn(
            f"{json_path.name}: missing joints filled with 0 -> {', '.join(sorted(missing_joints))}",
            stacklevel=2,
        )
    return pose


def build_output_payload(
    data: dict[str, Any],
    pose: np.ndarray,
    sport: str,
    subject_type: str,
    json_path: Path,
    joint_names: list[str],
    channels: list[str],
) -> dict[str, Any]:
    """Build arrays and metadata saved into one npz file."""
    video_name = str(data.get("video_name", json_path.with_suffix(".mp4").name))
    return {
        "pose": pose,
        "sport": np.array(sport),
        "subject_type": np.array(subject_type),
        "sample_id": np.array(json_path.stem),
        "video_name": np.array(video_name),
        "frame_count": np.array(int(data.get("frame_count", pose.shape[0])), dtype=np.int32),
        "fps": np.array(float(data.get("fps", 0.0)), dtype=np.float32),
        "width": np.array(int(data.get("width", 0)), dtype=np.int32),
        "height": np.array(int(data.get("height", 0)), dtype=np.int32),
        "joint_names": np.array(joint_names),
        "channels": np.array(channels),
        "steps_json": np.array(json.dumps(data.get("steps", {}), ensure_ascii=False)),
        "source_json": np.array(str(json_path)),
    }


def convert_one_json(
    json_path: Path,
    output_path: Path,
    sport: str,
    subject_type: str,
    channels: list[str],
) -> None:
    """Convert one extraction JSON into one compressed npz tensor file."""
    data = load_json(json_path)
    joint_names = SPORT_JOINT_NAMES[sport]
    pose = coordinates_to_tensor(data, joint_names, channels, json_path)
    payload = build_output_payload(data, pose, sport, subject_type, json_path, joint_names, channels)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)


def iter_json_files(root: Path, sports: list[str], subjects: list[str]) -> list[tuple[Path, str, str, Path]]:
    """Collect JSON files from data/{Sport}/jsons/{expert,user}."""
    items: list[tuple[Path, str, str, Path]] = []
    for sport in sports:
        for subject_type in subjects:
            json_dir = root / "data" / sport / "jsons" / subject_type
            if not json_dir.exists():
                continue
            for json_path in sorted(json_dir.rglob("*.json")):
                items.append((json_path, sport, subject_type, json_path.relative_to(json_dir)))
    return items


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Convert extraction JSON files to pose tensor npz files.")
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parent),
        help="Project root. Default: directory containing this script.",
    )
    parser.add_argument(
        "--sport",
        choices=["all", "Golf", "Archery"],
        default="all",
        help="Sport to convert.",
    )
    parser.add_argument(
        "--subject-type",
        choices=["all", "expert", "user"],
        default="all",
        help="Subject group to convert.",
    )
    parser.add_argument(
        "--channels",
        choices=["normalized", "pixel"],
        default="normalized",
        help="normalized saves x/y/score. pixel saves pixel_x/pixel_y/score.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing tensor files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    sports = list(SPORT_JOINT_NAMES) if args.sport == "all" else [args.sport]
    subjects = SUBJECT_TYPES if args.subject_type == "all" else [args.subject_type]
    channels = PIXEL_CHANNELS if args.channels == "pixel" else DEFAULT_CHANNELS

    json_files = iter_json_files(root, sports, subjects)
    if not json_files:
        print("No JSON files found.")
        return

    converted = 0
    skipped = 0
    for json_path, sport, subject_type, relative_path in json_files:
        output_path = root / "data" / sport / "tensor" / subject_type / relative_path.with_suffix(".npz")
        if output_path.exists() and not args.overwrite:
            skipped += 1
            print(f"[skip] {output_path}")
            continue

        convert_one_json(json_path, output_path, sport, subject_type, channels)
        converted += 1
        print(f"[save] {output_path}")

    print(f"Done. converted={converted}, skipped={skipped}")


if __name__ == "__main__":
    main()
