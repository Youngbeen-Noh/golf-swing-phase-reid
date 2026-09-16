from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


SPORTS = ["Golf", "Archery"]
SUBJECT_TYPES = ["expert", "user"]

GOLF_KEYFRAME_NAMES = [
    "address",
    "take_away",
    "half",
    "top",
    "down_half",
    "impact",
    "follow_through",
    "finish",
]

ARCHERY_KEYFRAME_NAMES = [
    "address",
    "setup",
    "drawing",
    "holding",
    "release",
]

DEFAULT_SEQUENCE_LENGTH = {
    "Golf": 160,
    "Archery": 100,
}

DEFAULT_STEP_LENGTH = {
    "Golf": 20,
    "Archery": 20,
}

GOLF_LEGACY_STEP_TARGET = {
    "address": 6,
    "take_away": 8,
    "half": 10,
    "top": 12,
    "down_half": 12,
    "impact": 10,
    "follow_through": 12,
    "finish": 10,
}

KEYFRAME_NAMES = {
    "Golf": GOLF_KEYFRAME_NAMES,
    "Archery": ARCHERY_KEYFRAME_NAMES,
}


@dataclass(frozen=True)
class PreprocessConfig:
    """Ablation preset for one preprocessing experiment."""

    name: str
    sport: str
    operations: tuple[str, ...]
    smoothing_mode: str = "moving_average"


PREPROCESS_CONFIGS: dict[str, PreprocessConfig] = {
    "Golf-A": PreprocessConfig("Golf-A", "Golf", ()),
    "Golf-B": PreprocessConfig("Golf-B", "Golf", ("SPAR",)),
    "Golf-C": PreprocessConfig("Golf-C", "Golf", ("DPAR",)),
    "Golf-D": PreprocessConfig("Golf-D", "Golf", ("SPAR", "PAA")),
    "Golf-E": PreprocessConfig("Golf-E", "Golf", ("DPAR", "PAA")),
    "Archery-A": PreprocessConfig("Archery-A", "Archery", ()),
    "Archery-B": PreprocessConfig("Archery-B", "Archery", ("TORSO_CENTERING",)),
    "Archery-C": PreprocessConfig("Archery-C", "Archery", ("SHOULDER_CENTERING",)),
    "Archery-D": PreprocessConfig("Archery-D", "Archery", ("SHOULDER_LINE_ALIGNMENT",)),
    "Archery-E": PreprocessConfig("Archery-E", "Archery", ("BOW_ARM_ALIGNMENT",)),
    "Archery-F": PreprocessConfig("Archery-F", "Archery", (), smoothing_mode="release_preserving"),
}


DEFAULT_PREPROCESS_METHOD = {
    "Golf": "Golf-A",
    "Archery": "Archery-A",
}


def interval_name(start_name: str, end_name: str) -> str:
    """Return a stable name for one keyframe-to-keyframe interval."""
    return f"{start_name}_to_{end_name}"


def scalar(value: np.ndarray | Any) -> Any:
    """Convert a 0-d numpy array loaded from npz into a Python scalar."""
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def moving_average(sequence: np.ndarray, window: int, smooth_score: bool = False) -> np.ndarray:
    """Apply temporal moving average to pose tensor shape (T, J, C).

    x/y channels are smoothed by default. score is preserved unless
    smooth_score is true.
    """
    if window <= 1 or sequence.shape[0] <= 1:
        return sequence.astype(np.float32).copy()

    output = sequence.astype(np.float32).copy()
    pad = window // 2
    kernel = np.ones(window, dtype=np.float32) / float(window)
    smooth_channels = range(sequence.shape[2]) if smooth_score else [0, 1]

    for joint_idx in range(sequence.shape[1]):
        for channel_idx in smooth_channels:
            values = sequence[:, joint_idx, channel_idx]
            padded = np.pad(values, (pad, pad), mode="edge")
            output[:, joint_idx, channel_idx] = np.convolve(padded, kernel, mode="valid")[: sequence.shape[0]]
    return output


def release_preserving_moving_average(
    sequence: np.ndarray,
    step_dict: dict[str, int],
    window: int,
    preserve_radius: int | None = None,
) -> np.ndarray:
    """Smooth pose coordinates while keeping release-adjacent frames unchanged."""
    smoothed = moving_average(sequence, window)
    if window <= 1 or sequence.shape[0] <= 1 or "release" not in step_dict:
        return smoothed

    radius = max(1, window // 2) if preserve_radius is None else max(0, preserve_radius)
    release_frame = int(np.clip(step_dict["release"], 0, sequence.shape[0] - 1))
    start = max(0, release_frame - radius)
    end = min(sequence.shape[0], release_frame + radius + 1)
    smoothed[start:end, :, :2] = sequence[start:end, :, :2]
    return smoothed


def joint_index(joint_names: list[str], name: str) -> int | None:
    """Return joint index or None when the joint does not exist."""
    return joint_names.index(name) if name in joint_names else None


def joint_pair_center(sequence: np.ndarray, joint_names: list[str], left_name: str, right_name: str) -> np.ndarray | None:
    """Return per-frame center of a left/right joint pair."""
    left_idx = joint_index(joint_names, left_name)
    right_idx = joint_index(joint_names, right_name)
    if left_idx is None or right_idx is None:
        return None
    return (sequence[:, left_idx, :2] + sequence[:, right_idx, :2]) / 2.0


def shoulder_center(sequence: np.ndarray, joint_names: list[str]) -> np.ndarray | None:
    """Return per-frame shoulder center."""
    return joint_pair_center(sequence, joint_names, "left_shoulder", "right_shoulder")


def pelvis_center(sequence: np.ndarray, joint_names: list[str]) -> np.ndarray | None:
    """Return per-frame pelvis center."""
    return joint_pair_center(sequence, joint_names, "left_hip", "right_hip")


def torso_center(sequence: np.ndarray, joint_names: list[str]) -> np.ndarray | None:
    """Return per-frame torso center from shoulder and pelvis centers."""
    shoulders = shoulder_center(sequence, joint_names)
    pelvis = pelvis_center(sequence, joint_names)
    if shoulders is None or pelvis is None:
        return None
    return (shoulders + pelvis) / 2.0


def subtract_reference_center(sequence: np.ndarray, centers: np.ndarray | None, reference_frame: int) -> np.ndarray:
    """Translate all frames by one reference center."""
    if centers is None or sequence.shape[0] == 0:
        return sequence.astype(np.float32).copy()
    ref = int(np.clip(reference_frame, 0, sequence.shape[0] - 1))
    output = sequence.astype(np.float32).copy()
    output[:, :, :2] = output[:, :, :2] - centers[ref][None, None, :]
    return output


def subtract_dynamic_center(sequence: np.ndarray, centers: np.ndarray | None) -> np.ndarray:
    """Translate each frame by its own center."""
    if centers is None:
        return sequence.astype(np.float32).copy()
    output = sequence.astype(np.float32).copy()
    output[:, :, :2] = output[:, :, :2] - centers[:, None, :]
    return output


def rotate_sequence_xy(sequence: np.ndarray, angle_radians: float, origin: np.ndarray) -> np.ndarray:
    """Rotate x/y coordinates around one origin."""
    output = sequence.astype(np.float32).copy()
    cos_a = float(np.cos(angle_radians))
    sin_a = float(np.sin(angle_radians))
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    xy = output[:, :, :2] - origin.reshape(1, 1, 2)
    output[:, :, :2] = xy @ rotation.T + origin.reshape(1, 1, 2)
    return output


def align_joint_line(
    sequence: np.ndarray,
    joint_names: list[str],
    left_name: str,
    right_name: str,
    reference_frame: int,
    eps: float = 1e-6,
) -> np.ndarray:
    """Rotate pose so a left/right joint line is horizontal at reference frame."""
    left_idx = joint_index(joint_names, left_name)
    right_idx = joint_index(joint_names, right_name)
    if left_idx is None or right_idx is None or sequence.shape[0] == 0:
        return sequence.astype(np.float32).copy()

    ref = int(np.clip(reference_frame, 0, sequence.shape[0] - 1))
    left = sequence[ref, left_idx, :2]
    right = sequence[ref, right_idx, :2]
    vector = right - left
    if float(np.linalg.norm(vector)) < eps:
        return sequence.astype(np.float32).copy()

    origin = (left + right) / 2.0
    return rotate_sequence_xy(sequence, -float(np.arctan2(vector[1], vector[0])), origin)


def align_bow_arm(
    sequence: np.ndarray,
    joint_names: list[str],
    reference_frame: int,
    eps: float = 1e-6,
) -> np.ndarray:
    """Rotate pose so the longer shoulder-to-wrist arm vector is horizontal."""
    candidates = [
        ("left_shoulder", "left_wrist"),
        ("right_shoulder", "right_wrist"),
    ]
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    if sequence.shape[0] == 0:
        return sequence.astype(np.float32).copy()
    ref = int(np.clip(reference_frame, 0, sequence.shape[0] - 1))

    for shoulder_name, wrist_name in candidates:
        shoulder_idx = joint_index(joint_names, shoulder_name)
        wrist_idx = joint_index(joint_names, wrist_name)
        if shoulder_idx is None or wrist_idx is None:
            continue
        shoulder = sequence[ref, shoulder_idx, :2]
        wrist = sequence[ref, wrist_idx, :2]
        vector = wrist - shoulder
        length = float(np.linalg.norm(vector))
        if length >= eps and (best is None or length > best[0]):
            best = (length, shoulder, vector)

    if best is None:
        return sequence.astype(np.float32).copy()
    _, origin, vector = best
    return rotate_sequence_xy(sequence, -float(np.arctan2(vector[1], vector[0])), origin)


def torso_scale_normalize(
    sequence: np.ndarray,
    joint_names: list[str],
    reference_frame: int,
    eps: float = 1e-6,
) -> np.ndarray:
    """Normalize x/y channels by torso length at reference_frame.

    Torso length is shoulder-center to hip-center distance. This keeps the
    original coordinate origin and only scales x/y. score is unchanged.
    """
    if sequence.shape[0] == 0:
        return sequence.astype(np.float32).copy()

    left_shoulder = joint_index(joint_names, "left_shoulder")
    right_shoulder = joint_index(joint_names, "right_shoulder")
    left_hip = joint_index(joint_names, "left_hip")
    right_hip = joint_index(joint_names, "right_hip")
    if None in {left_shoulder, right_shoulder, left_hip, right_hip}:
        return sequence.astype(np.float32).copy()

    ref = int(np.clip(reference_frame, 0, sequence.shape[0] - 1))
    shoulder_center = (sequence[ref, left_shoulder, :2] + sequence[ref, right_shoulder, :2]) / 2.0
    hip_center = (sequence[ref, left_hip, :2] + sequence[ref, right_hip, :2]) / 2.0
    torso_length = float(np.linalg.norm(shoulder_center - hip_center))
    if torso_length < eps:
        torso_length = 1.0

    output = sequence.astype(np.float32).copy()
    output[:, :, :2] = output[:, :, :2] / torso_length
    return output


def sequence_xy_to_pixels(
    sequence: np.ndarray,
    channels: list[str],
    width: int,
    height: int,
) -> np.ndarray:
    """Return a copy whose x/y channels are in pixel coordinates."""
    output = sequence.astype(np.float32).copy()
    if output.shape[2] < 2:
        return output
    if len(channels) >= 2 and channels[0] == "pixel_x" and channels[1] == "pixel_y":
        return output
    if width > 0:
        output[:, :, 0] *= float(width)
    if height > 0:
        output[:, :, 1] *= float(height)
    return output


def legacy_address_refs(
    sequence_px: np.ndarray,
    joint_names: list[str],
    reference_frame: int,
    eps: float = 1e-6,
) -> tuple[np.ndarray | None, float, float]:
    """Return address pelvis center, address torso length, and torso angle."""
    shoulders = shoulder_center(sequence_px, joint_names)
    pelvis = pelvis_center(sequence_px, joint_names)
    if shoulders is None or pelvis is None or sequence_px.shape[0] == 0:
        return None, 1.0, 0.0

    ref = int(np.clip(reference_frame, 0, sequence_px.shape[0] - 1))
    shoulder_ref = shoulders[ref]
    pelvis_ref = pelvis[ref]
    vector = shoulder_ref - pelvis_ref
    torso_length = float(np.linalg.norm(vector))
    if not np.isfinite(torso_length) or torso_length < eps:
        torso_length = 1.0
    torso_angle = float(np.arctan2(vector[1], vector[0]))
    return pelvis_ref.astype(np.float32), torso_length, torso_angle


def apply_legacy_golf_transform(
    sequence: np.ndarray,
    joint_names: list[str],
    channels: list[str],
    width: int,
    height: int,
    operations: tuple[str, ...],
    reference_frame: int,
) -> np.ndarray:
    """Apply the senior-study golf coordinate normalization.

    Golf-A keeps the image origin and scales by address torso length.
    SPAR uses the address pelvis as a fixed origin. DPAR uses the per-frame
    pelvis as a dynamic origin. PAA rotates by the address torso axis.
    """
    output = sequence_xy_to_pixels(sequence, channels, width, height)
    if output.shape[0] == 0:
        return output.astype(np.float32)

    address_pelvis, torso_length, torso_angle = legacy_address_refs(output, joint_names, reference_frame)
    xy = output[:, :, :2]

    if "DPAR" in operations:
        centers = pelvis_center(output, joint_names)
        if centers is not None:
            xy = xy - centers[:, None, :]
        elif address_pelvis is not None:
            xy = xy - address_pelvis.reshape(1, 1, 2)
    elif "SPAR" in operations and address_pelvis is not None:
        xy = xy - address_pelvis.reshape(1, 1, 2)

    xy = xy / float(torso_length)

    if "PAA" in operations:
        angle = -torso_angle
        cos_a = float(np.cos(angle))
        sin_a = float(np.sin(angle))
        rotated = xy.copy()
        rotated[:, :, 0] = xy[:, :, 0] * cos_a - xy[:, :, 1] * sin_a
        rotated[:, :, 1] = xy[:, :, 0] * sin_a + xy[:, :, 1] * cos_a
        xy = rotated

    output[:, :, :2] = xy
    return output.astype(np.float32)


def recompute_steps_from_targets(step_target: dict[str, int], keyframe_names: list[str]) -> dict[str, int]:
    """Rebuild keyframe starts after phase-synchronized segment resampling."""
    starts: dict[str, int] = {}
    current = 0
    for name in keyframe_names:
        starts[name] = current
        current += int(step_target.get(name, 0))
    return starts


def phase_synchronized_resample(
    sequence: np.ndarray,
    step_dict: dict[str, int],
    keyframe_names: list[str],
    step_target: dict[str, int],
    include_tail: bool = True,
) -> tuple[np.ndarray, dict[str, int]]:
    """Resample each keyframe segment to the senior-study PSR template.

    include_tail=True (default, unchanged behavior): the last keyframe name
    additionally gets a synthetic "keyframe -> end of sequence" segment
    (e.g. a finish-hold), matching every prior experiment in this repo.

    include_tail=False (P0-4b / 7.2 finish-hold ablation): only the real
    N-1 keyframe-to-keyframe segments are built -- the same 7 segments the
    phase-level branch already uses -- so the sequence-level branch no
    longer carries frames the phase-level branch never sees.
    """
    if sequence.shape[0] == 0:
        return sequence.astype(np.float32), step_dict.copy()

    segments: list[np.ndarray] = []
    max_frame = sequence.shape[0] - 1
    names = keyframe_names if include_tail else keyframe_names[:-1]
    for index, name in enumerate(names):
        start = int(np.clip(step_dict.get(name, 0), 0, max_frame))
        if index + 1 < len(keyframe_names):
            next_name = keyframe_names[index + 1]
            end = int(np.clip(step_dict.get(next_name, start), start, max_frame))
        else:
            end = max_frame
        target = int(step_target.get(name, max(1, end - start + 1)))
        if target <= 0:
            continue
        segments.append(resample_sequence(slice_inclusive(sequence, start, end), target))

    if not segments:
        return sequence.astype(np.float32), step_dict.copy()
    resampled = np.concatenate(segments, axis=0).astype(np.float32)
    # Always recompute over the FULL keyframe_names (not `names`): this only
    # sums prior targets to get each keyframe's start offset, so the excluded
    # last keyframe's start still lands correctly at the end of the resampled
    # array (sum of the 7 real segments) -- the phase-level branch's
    # step_intervals() needs that key present even when include_tail=False.
    return resampled, recompute_steps_from_targets(step_target, keyframe_names)


def phase_concat_sequence(
    preprocessed: np.ndarray,
    feature_step_dict: dict[str, int],
    keyframe_names: list[str],
    final_end: int,
    step_length: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Build a sequence-branch input by resampling each of the 7 REAL
    keyframe-to-keyframe segments to `step_length` frames and concatenating
    them -- i.e. the exact same per-segment resampling the phase/step branch
    already applies, just flattened into one sequence instead of stacked.

    This exists for the frame-budget-controlled ablation (2nd revision
    instruction 2.4): the legacy sequence branch resamples the WHOLE
    preprocessed span (including, by default, a synthetic finish-hold tail
    the phase branch never sees) in one global interpolation to
    `sequence_length` frames, which confounds "phase decomposition helps"
    with "different frame count / different resampling granularity than the
    phase branch". Calling this with step_length=20 yields Sequence-only-140
    (7 x 20, tail already excluded upstream since `preprocessed` here must be
    built with include_tail=False); resampling that result again to 160
    yields Sequence-only-160, controlling for the original 160-frame length
    while keeping identical real content and resampling granularity.

    rng, if given, shuffles the 7 segments' order before concatenating
    (Shuffled-phase condition) -- same permutation applies to every sample
    processed with the same rng state at the call site, so pass a
    freshly-seeded Generator per run, not per sample.
    """
    intervals = step_intervals(feature_step_dict, keyframe_names, preprocessed.shape[0], final_end=final_end)
    segments = [resample_sequence(slice_inclusive(preprocessed, start, end), step_length) for _, start, end in intervals]
    if rng is not None:
        order = rng.permutation(len(segments))
        segments = [segments[i] for i in order]
    if not segments:
        return preprocessed.astype(np.float32)
    return np.concatenate(segments, axis=0).astype(np.float32)


def operation_reference_frame(step_dict: dict[str, int], operation: str, fallback: int) -> int:
    """Choose a keyframe for an operation-specific reference pose."""
    if operation == "BOW_ARM_ALIGNMENT":
        return int(step_dict.get("holding", step_dict.get("setup", fallback)))
    if operation == "SHOULDER_LINE_ALIGNMENT":
        return int(step_dict.get("setup", fallback))
    return fallback


def apply_preprocessing(
    sequence: np.ndarray,
    joint_names: list[str],
    step_dict: dict[str, int],
    config: PreprocessConfig,
    smoothing_window: int,
    reference_frame: int,
) -> np.ndarray:
    """Apply common preprocessing plus experiment-specific ablation operations."""
    if config.smoothing_mode == "release_preserving":
        output = release_preserving_moving_average(sequence, step_dict, smoothing_window)
    else:
        output = moving_average(sequence, smoothing_window)

    output = torso_scale_normalize(output, joint_names, reference_frame)

    for operation in config.operations:
        op_ref = operation_reference_frame(step_dict, operation, reference_frame)
        if operation == "PAA":
            output = align_joint_line(output, joint_names, "left_hip", "right_hip", op_ref)
        elif operation == "SPAR":
            output = subtract_reference_center(output, pelvis_center(output, joint_names), op_ref)
        elif operation == "DPAR":
            output = subtract_dynamic_center(output, pelvis_center(output, joint_names))
        elif operation == "TORSO_CENTERING":
            output = subtract_reference_center(output, torso_center(output, joint_names), op_ref)
        elif operation == "SHOULDER_CENTERING":
            output = subtract_reference_center(output, shoulder_center(output, joint_names), op_ref)
        elif operation == "SHOULDER_LINE_ALIGNMENT":
            output = align_joint_line(output, joint_names, "left_shoulder", "right_shoulder", op_ref)
        elif operation == "BOW_ARM_ALIGNMENT":
            output = align_bow_arm(output, joint_names, op_ref)
        else:
            raise ValueError(f"Unknown preprocessing operation: {operation}")
    return output.astype(np.float32)


def resample_sequence(sequence: np.ndarray, output_length: int) -> np.ndarray:
    """Linearly resample temporal pose tensor from (T, J, C) to (L, J, C)."""
    if output_length <= 0:
        raise ValueError("output_length must be positive")
    if sequence.shape[0] == 0:
        return np.zeros((output_length, sequence.shape[1], sequence.shape[2]), dtype=np.float32)
    if sequence.shape[0] == 1:
        return np.repeat(sequence.astype(np.float32), output_length, axis=0)
    if sequence.shape[0] == output_length:
        return sequence.astype(np.float32).copy()

    source_x = np.linspace(0.0, 1.0, sequence.shape[0], dtype=np.float32)
    target_x = np.linspace(0.0, 1.0, output_length, dtype=np.float32)
    flat = sequence.reshape(sequence.shape[0], -1)
    resampled = np.vstack([np.interp(target_x, source_x, flat[:, idx]) for idx in range(flat.shape[1])]).T
    return resampled.reshape(output_length, sequence.shape[1], sequence.shape[2]).astype(np.float32)


def compute_velocity(sequence: np.ndarray) -> np.ndarray:
    """Compute dx/dy velocity with shape (T, J, 2). First frame is 0."""
    velocity = np.zeros((sequence.shape[0], sequence.shape[1], 2), dtype=np.float32)
    if sequence.shape[0] > 1:
        velocity[1:] = sequence[1:, :, :2] - sequence[:-1, :, :2]
    return velocity


def build_features(sequence: np.ndarray) -> np.ndarray:
    """Build feature tensor shape (T, J, 5): x, y, score, dx, dy."""
    score = sequence[:, :, 2:3] if sequence.shape[2] >= 3 else np.zeros((*sequence.shape[:2], 1), dtype=np.float32)
    velocity = compute_velocity(sequence)
    return np.concatenate([sequence[:, :, :2], score, velocity], axis=-1).astype(np.float32)


def parse_steps(steps_json: str) -> dict[str, int] | list[dict[str, int]]:
    """Parse step JSON string stored by 02_json_to_tensor.py."""
    raw = json.loads(steps_json)
    if isinstance(raw, list):
        return [{key: int(value) for key, value in step.items()} for step in raw]
    if isinstance(raw, dict):
        return {key: int(value) for key, value in raw.items()}
    raise ValueError("steps_json must decode to dict or list[dict]")


def step_intervals(
    step_dict: dict[str, int],
    keyframe_names: list[str],
    frame_count: int,
    final_end: int | None = None,
) -> list[tuple[str, int, int]]:
    """Convert ordered keyframes to inclusive transition intervals.

    For N keyframes this returns N-1 intervals, e.g. Golf's 8 keyframes become
    7 intervals from address_to_take_away through follow_through_to_finish.
    """
    max_frame = max(0, frame_count - 1)
    last_end = max_frame if final_end is None else int(np.clip(final_end, 0, max_frame))
    intervals: list[tuple[str, int, int]] = []

    for start_name, end_name in zip(keyframe_names[:-1], keyframe_names[1:]):
        start = int(np.clip(step_dict.get(start_name, 0), 0, last_end))
        end = int(np.clip(step_dict.get(end_name, start), 0, last_end))
        intervals.append((interval_name(start_name, end_name), start, max(start, end)))
    return intervals


def slice_inclusive(sequence: np.ndarray, start: int, end: int) -> np.ndarray:
    """Slice sequence using inclusive temporal bounds."""
    if sequence.shape[0] == 0:
        return sequence
    safe_start = int(np.clip(start, 0, sequence.shape[0] - 1))
    safe_end = int(np.clip(end, safe_start, sequence.shape[0] - 1))
    return sequence[safe_start : safe_end + 1]


def archery_arrow_specs(
    steps: list[dict[str, int]],
    frame_count: int,
) -> list[tuple[int, dict[str, int], int, int]]:
    """Return (arrow_index, step_dict, start_frame, end_frame) for each arrow."""
    specs = []
    for arrow_idx, step_dict in enumerate(steps):
        start_frame = int(step_dict.get("address", 0))
        if arrow_idx + 1 < len(steps):
            next_address = int(steps[arrow_idx + 1].get("address", frame_count))
            end_frame = max(start_frame, next_address - 1)
        else:
            end_frame = frame_count - 1
        specs.append((arrow_idx, step_dict, start_frame, end_frame))
    return specs


def sample_specs(
    sport: str,
    steps: dict[str, int] | list[dict[str, int]],
    frame_count: int,
) -> list[tuple[str, dict[str, int], int, int]]:
    """Build sample specs for one tensor file.

    Returns:
        List of (sample_suffix, step_dict, sequence_start, sequence_end).
    """
    if sport == "Archery":
        if isinstance(steps, dict):
            return [("", steps, 0, frame_count - 1)]
        if not isinstance(steps, list):
            raise ValueError("Archery steps must be dict or list[dict]")
        arrows = archery_arrow_specs(steps, frame_count)
        if len(arrows) == 1:
            return [("", arrows[0][1], arrows[0][2], arrows[0][3])]
        return [(f"_arrow{arrow_idx + 1:03d}", step_dict, start, end) for arrow_idx, step_dict, start, end in arrows]

    if not isinstance(steps, dict):
        raise ValueError("Golf steps must be dict")
    return [("", steps, 0, frame_count - 1)]


def process_one_tensor(
    tensor_path: Path,
    output_dir: Path,
    sequence_length: int,
    step_length: int,
    smoothing_window: int,
    preprocess_config: PreprocessConfig,
    overwrite: bool,
    no_finish_hold: bool = False,
    sequence_mode: str = "legacy",
    shuffle_phase_order: bool = False,
    shuffle_seed: int = 0,
) -> tuple[int, int]:
    """Convert one raw pose tensor npz into one or more feature npz files."""
    with np.load(tensor_path, allow_pickle=False) as data:
        pose = data["pose"].astype(np.float32)
        sport = str(scalar(data["sport"]))
        subject_type = str(scalar(data["subject_type"]))
        sample_id = str(scalar(data["sample_id"]))
        frame_count = int(scalar(data["frame_count"]))
        fps = float(scalar(data["fps"]))
        width = int(scalar(data["width"]))
        height = int(scalar(data["height"]))
        joint_names = [str(item) for item in data["joint_names"].tolist()]
        channels = [str(item) for item in data["channels"].tolist()] if "channels" in data.files else ["x", "y", "score"]
        source_json = str(scalar(data["source_json"]))
        steps = parse_steps(str(scalar(data["steps_json"])))

    keyframe_names = KEYFRAME_NAMES[sport]
    converted = 0
    skipped = 0

    for suffix, step_dict, sequence_start, sequence_end in sample_specs(sport, steps, frame_count):
        out_path = output_dir / f"{sample_id}{suffix}.npz"
        if out_path.exists() and not overwrite:
            skipped += 1
            print(f"[skip] {out_path}")
            continue

        reference_frame = int(step_dict.get("address", sequence_start))
        if sport == "Golf":
            smoothed = moving_average(pose, smoothing_window)
            transformed = apply_legacy_golf_transform(
                sequence=smoothed,
                joint_names=joint_names,
                channels=channels,
                width=width,
                height=height,
                operations=preprocess_config.operations,
                reference_frame=reference_frame,
            )
            preprocessed, feature_step_dict = phase_synchronized_resample(
                sequence=transformed,
                step_dict=step_dict,
                keyframe_names=keyframe_names,
                step_target=GOLF_LEGACY_STEP_TARGET,
                include_tail=not no_finish_hold,
            )
            feature_sequence_start = 0
            feature_sequence_end = max(0, preprocessed.shape[0] - 1)
        else:
            preprocessed = apply_preprocessing(
                sequence=pose,
                joint_names=joint_names,
                step_dict=step_dict,
                config=preprocess_config,
                smoothing_window=smoothing_window,
                reference_frame=reference_frame,
            )
            feature_step_dict = step_dict
            feature_sequence_start = sequence_start
            feature_sequence_end = sequence_end

        if sequence_mode == "phase-concat" and sport == "Golf":
            # Seeded from shuffle_seed ALONE (not sample_id/suffix): the CLI help
            # and phase_concat_sequence()'s own docstring both promise "the same
            # permutation applies to every sample" for a given --shuffle-seed --
            # keying the seed on sample_id as this line previously did silently
            # gave every sample its OWN independent random order instead, which
            # would test something quite different (no cross-sample positional
            # consistency at all, rather than "does the TRUE order matter versus
            # some OTHER fixed, consistent order").
            shuffle_rng = np.random.default_rng(shuffle_seed) if shuffle_phase_order else None
            phase_concat = phase_concat_sequence(
                preprocessed, feature_step_dict, keyframe_names, feature_sequence_end, step_length, rng=shuffle_rng
            )
            sequence_pose = resample_sequence(phase_concat, sequence_length) if sequence_length != phase_concat.shape[0] else phase_concat
        else:
            sequence_raw = slice_inclusive(preprocessed, feature_sequence_start, feature_sequence_end)
            sequence_pose = resample_sequence(sequence_raw, sequence_length)
        sequence_features = build_features(sequence_pose)

        intervals = step_intervals(
            feature_step_dict,
            keyframe_names,
            preprocessed.shape[0],
            final_end=feature_sequence_end,
        )
        step_names = [name for name, _, _ in intervals]
        step_feature_list = []
        for _, start, end in intervals:
            step_pose = resample_sequence(slice_inclusive(preprocessed, start, end), step_length)
            step_feature_list.append(build_features(step_pose))
        step_features = np.stack(step_feature_list, axis=0).astype(np.float32)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            sequence=sequence_features,
            steps=step_features,
            sport=np.array(sport),
            subject_type=np.array(subject_type),
            sample_id=np.array(f"{sample_id}{suffix}"),
            source_tensor=np.array(str(tensor_path)),
            source_json=np.array(source_json),
            fps=np.array(fps, dtype=np.float32),
            width=np.array(width, dtype=np.int32),
            height=np.array(height, dtype=np.int32),
            joint_names=np.array(joint_names),
            feature_channels=np.array(["x", "y", "score", "dx", "dy"]),
            source_channels=np.array(channels),
            keyframe_names=np.array(keyframe_names),
            step_names=np.array(step_names),
            step_intervals_json=np.array(json.dumps(intervals, ensure_ascii=False)),
            step_starts_json=np.array(json.dumps(feature_step_dict, ensure_ascii=False)),
            source_step_starts_json=np.array(json.dumps(step_dict, ensure_ascii=False)),
            sequence_length=np.array(sequence_length, dtype=np.int32),
            step_length=np.array(step_length, dtype=np.int32),
            preprocess_method=np.array(preprocess_config.name),
            preprocess_operations=np.array(list(preprocess_config.operations)),
            smoothing_mode=np.array(preprocess_config.smoothing_mode),
            smoothing_window=np.array(smoothing_window, dtype=np.int32),
            golf_legacy_step_target_json=np.array(json.dumps(GOLF_LEGACY_STEP_TARGET, ensure_ascii=False) if sport == "Golf" else ""),
            golf_legacy_preprocessing=np.array(sport == "Golf"),
        )
        converted += 1
        print(f"[save] {out_path}")

    return converted, skipped


def iter_tensor_files(
    root: Path,
    sports: list[str],
    subjects: list[str],
    expert_ids: set[str] | None = None,
) -> list[tuple[Path, str, str, Path]]:
    """Collect raw tensor files from data/{Sport}/tensor/{expert,user}.

    When expert_ids is given, only files whose top-level person folder
    (relative_path.parts[0]) is in that set are kept, for both expert
    and user subject types -- lets one run_id restrict the whole
    dataset (train + holdout) to a subset of experts, e.g. to
    reproduce a smaller-roster experiment from the shared tensor pool.
    """
    items: list[tuple[Path, str, str, Path]] = []
    for sport in sports:
        for subject_type in subjects:
            tensor_dir = root / "data" / sport / "tensor" / subject_type
            if not tensor_dir.exists():
                continue
            for tensor_path in sorted(tensor_dir.rglob("*.npz")):
                relative_path = tensor_path.relative_to(tensor_dir)
                if expert_ids is not None and relative_path.parts[0] not in expert_ids:
                    continue
                items.append((tensor_path, sport, subject_type, relative_path))
    return items


def parse_expert_ids(value: str | None) -> set[str] | None:
    if value is None:
        return None
    expert_ids = {item.strip() for item in value.split(",") if item.strip()}
    return expert_ids or None


def resolve_run_id(run_id: str | None) -> str:
    if run_id:
        return run_id
    env_run_id = os.environ.get("YB_RUN_ID")
    if env_run_id:
        return env_run_id
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def experiment_dir(root: Path, sport: str, run_id: str) -> Path:
    return root / "data" / sport / "experiments" / run_id


def write_latest_run_id(root: Path, sport: str, run_id: str) -> None:
    experiments_dir = root / "data" / sport / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    (experiments_dir / ".latest_run_id").write_text(run_id, encoding="utf-8")


def normalize_preprocess_method(method: str) -> str:
    """Normalize method aliases such as golf-b to Golf-B."""
    lowered = method.lower()
    if lowered == "auto":
        return "auto"
    for name in PREPROCESS_CONFIGS:
        if lowered == name.lower() or lowered == name.lower().replace("-", ""):
            return name
    valid = ", ".join(["auto", *PREPROCESS_CONFIGS])
    raise ValueError(f"Unknown --preprocess-method {method!r}. Valid values: {valid}")


def resolve_preprocess_config(method: str, sport: str) -> PreprocessConfig:
    """Resolve a sport-compatible preprocessing preset."""
    normalized = normalize_preprocess_method(method)
    if normalized == "auto":
        normalized = DEFAULT_PREPROCESS_METHOD[sport]
    config = PREPROCESS_CONFIGS[normalized]
    if config.sport != sport:
        raise ValueError(f"--preprocess-method {config.name} is for {config.sport}, but current sport is {sport}")
    return config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess pose tensors and build fixed-length feature tensors.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--sport", choices=["all", "Golf", "Archery"], default="all")
    parser.add_argument("--subject-type", choices=["all", "expert", "user"], default="all")
    parser.add_argument("--sequence-length-golf", type=int, default=DEFAULT_SEQUENCE_LENGTH["Golf"])
    parser.add_argument("--sequence-length-archery", type=int, default=DEFAULT_SEQUENCE_LENGTH["Archery"])
    parser.add_argument("--step-length-golf", type=int, default=DEFAULT_STEP_LENGTH["Golf"])
    parser.add_argument("--step-length-archery", type=int, default=DEFAULT_STEP_LENGTH["Archery"])
    parser.add_argument("--smoothing-window", type=int, default=5)
    parser.add_argument(
        "--preprocess-method",
        default="auto",
        help=(
            "Ablation preset: auto, Golf-A, Golf-B, Golf-C, Golf-D, Golf-E, "
            "Archery-A, Archery-B, Archery-C, Archery-D, Archery-E, Archery-F."
        ),
    )
    parser.add_argument("--run-id", default=None, help="Experiment run folder name. Defaults to YB_RUN_ID or current timestamp.")
    parser.add_argument("--output-root", default=None, help="Optional root for generated feature folders.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max-samples-per-expert",
        type=int,
        default=0,
        help="Cap the number of tensor files processed per expert (0 = no limit). Only applies to subject_type=expert.",
    )
    parser.add_argument(
        "--expert-ids",
        default=None,
        help=(
            "Optional comma-separated expert ids to restrict this run to (both expert and "
            "holdout/user tensors). Default: all experts found under data/{Sport}/tensor. "
            "Use this to derive a smaller-roster experiment from the same shared tensor pool."
        ),
    )
    parser.add_argument(
        "--no-finish-hold",
        action="store_true",
        help=(
            "Golf only (P0-4b / paper 7.2 ablation): drop the synthetic finish-hold tail "
            "segment from the sequence-level PSR template, so the sequence-level input "
            "(previously 160 frames incl. ~20 frames of held finish pose) covers only the "
            "same 7 real phases the phase-level branch already sees. Does not change "
            "phase-level input at all. Use a distinct --run-id from prior runs -- this is a "
            "different feature definition, not a bugfix to overwrite existing runs with."
        ),
    )
    parser.add_argument(
        "--sequence-mode",
        choices=["legacy", "phase-concat"],
        default="legacy",
        help=(
            "Golf only. 'legacy' (default): one global resample of the whole "
            "PSR-normalized span (optionally incl. synthetic finish-hold tail) to "
            "--sequence-length-golf. 'phase-concat' (2nd revision instruction 2.4, "
            "frame-budget-controlled ablation): resample each of the 7 REAL "
            "keyframe-to-keyframe segments to --step-length-golf frames and "
            "concatenate them (informationally identical to the phase branch, just "
            "flattened) -- use --sequence-length-golf 140 for Sequence-only-140 "
            "(no extra resampling, since 7x20=140), or 160 for Sequence-only-160 "
            "(one extra holistic resample of the 140-frame result, isolating the "
            "effect of matching the legacy 160-frame length). Combine with "
            "--no-finish-hold for a clean run-id (the tail is unused by this mode "
            "regardless, since it slices only the 7 real intervals)."
        ),
    )
    parser.add_argument(
        "--shuffle-phase-order",
        action="store_true",
        help=(
            "Golf only, requires --sequence-mode phase-concat: randomly permute the "
            "7 segments' order before concatenating (Shuffled-phase condition, "
            "2nd revision instruction 2.4) -- tests whether the sequence encoder's "
            "result depends on temporal phase order, not just phase content. "
            "Deterministic per --shuffle-seed, not per sample."
        ),
    )
    parser.add_argument("--shuffle-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.shuffle_phase_order and args.sequence_mode != "phase-concat":
        raise SystemExit("--shuffle-phase-order requires --sequence-mode phase-concat")
    root = Path(args.root)
    sports = SPORTS if args.sport == "all" else [args.sport]
    subjects = SUBJECT_TYPES if args.subject_type == "all" else [args.subject_type]
    preprocess_configs = {
        sport: resolve_preprocess_config(args.preprocess_method, sport)
        for sport in sports
    }
    run_id = resolve_run_id(args.run_id)
    for sport in sports:
        write_latest_run_id(root, sport, run_id)
    expert_ids = parse_expert_ids(args.expert_ids)
    tensor_files = iter_tensor_files(root, sports, subjects, expert_ids)

    if args.max_samples_per_expert > 0:
        from collections import defaultdict
        import random

        grouped: dict[tuple[str, str], list] = defaultdict(list)
        non_expert: list = []
        for item in tensor_files:
            _, sport, subject_type, relative_path = item
            if subject_type == "expert":
                expert_id = relative_path.parent.name
                grouped[(sport, expert_id)].append(item)
            else:
                non_expert.append(item)
        filtered: list = list(non_expert)
        for key, items in sorted(grouped.items()):
            sport, expert_id = key
            capped = items[: args.max_samples_per_expert]
            print(f"[cap] {sport}/{expert_id}: {len(items)} -> {len(capped)} samples")
            filtered.extend(capped)
        tensor_files = filtered

    if not tensor_files:
        print("No tensor files found. Run 02_json_to_tensor.py first.")
        return

    converted = 0
    skipped = 0
    for tensor_path, sport, subject_type, relative_path in tensor_files:
        sequence_length = args.sequence_length_golf if sport == "Golf" else args.sequence_length_archery
        step_length = args.step_length_golf if sport == "Golf" else args.step_length_archery
        preprocess_config = preprocess_configs[sport]
        base_output = Path(args.output_root) if args.output_root else experiment_dir(root, sport, run_id)
        output_dir = base_output / "features" / subject_type / relative_path.parent
        made, ignored = process_one_tensor(
            tensor_path=tensor_path,
            output_dir=output_dir,
            sequence_length=sequence_length,
            step_length=step_length,
            smoothing_window=args.smoothing_window,
            preprocess_config=preprocess_config,
            overwrite=args.overwrite,
            no_finish_hold=args.no_finish_hold,
            sequence_mode=args.sequence_mode,
            shuffle_phase_order=args.shuffle_phase_order,
            shuffle_seed=args.shuffle_seed,
        )
        converted += made
        skipped += ignored

    method_summary = ", ".join(f"{sport}:{config.name}" for sport, config in preprocess_configs.items())
    print(f"Done. run_id={run_id}, preprocess_method={method_summary}, converted={converted}, skipped={skipped}")


if __name__ == "__main__":
    main()
