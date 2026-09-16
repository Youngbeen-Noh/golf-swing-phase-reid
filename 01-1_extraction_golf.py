from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO


KEYPOINTS = [
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

# YOLO pose COCO keypoint order.
YOLO_INDEX = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_heel": 15,
    "right_heel": 16,
    # COCO has ankles but no toe landmarks. Keep toe names for compatibility.
    "left_toe": 15,
    "right_toe": 16,
}

EIGHT_STEP = [
    "address",
    "take_away",
    "half",
    "top",
    "down_half",
    "impact",
    "follow_through",
    "finish",
]

FINISH_TAIL_BUFFER = 0


def get_index_list(joint_name: str) -> int:
    return KEYPOINTS.index(joint_name)


def get_index_eight(step_name: str) -> int:
    return EIGHT_STEP.index(step_name)


def line_2d(one_point: list[float], another_point: list[float]) -> list[float]:
    dx = another_point[0] - one_point[0]
    slope = (another_point[1] - one_point[1]) / (dx if dx != 0 else 0.000001)
    intercept = one_point[1] - slope * one_point[0]
    return [slope, intercept]


def get_largest_person_idx(result: Any) -> int:
    if result.boxes is None or len(result.boxes) == 0:
        return 0
    if len(result.boxes) == 1:
        return 0

    max_area = -1.0
    max_idx = 0
    for idx, box in enumerate(result.boxes):
        x1, y1, x2, y2 = box.xyxy[0].detach().cpu().tolist()
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area > max_area:
            max_area = area
            max_idx = idx
    return max_idx


def moving_average(values: list[float], window_len: int) -> list[float]:
    if not values:
        return []
    if len(values) < window_len:
        return [float(v) for v in values]

    pad = window_len // 2
    padded = np.pad(np.asarray(values, dtype=float), (pad, pad), mode="edge")
    kernel = np.ones(window_len, dtype=float) / window_len
    return np.convolve(padded, kernel, mode="valid").tolist()


def smooth_landmarks(
    cv_pose: list[list[list[float]]],
    empty_frames: set[int],
    width: int,
    height: int,
    window_len: int = 3,
) -> list[list[list[float]]]:
    for joint in range(len(KEYPOINTS)):
        for axis in range(5):
            cv_pose[joint][axis] = moving_average(cv_pose[joint][axis], window_len)

    pose = [[[], [], [], [], []] for _ in KEYPOINTS]
    frame_count = len(cv_pose[0][0]) if cv_pose else 0

    for frame_num in range(frame_count):
        for joint in range(len(KEYPOINTS)):
            if frame_num in empty_frames:
                for axis in range(5):
                    pose[joint][axis].append(0.0)
                continue

            pose[joint][0].append(float(cv_pose[joint][0][frame_num]) / width)
            pose[joint][1].append(float(cv_pose[joint][1][frame_num]) / height)
            pose[joint][2].append(float(cv_pose[joint][2][frame_num]))
            pose[joint][3].append(float(cv_pose[joint][3][frame_num]))
            pose[joint][4].append(float(cv_pose[joint][4][frame_num]))
    return pose


def get_step(video_length: int, pose_landmark: list[list[list[float]]], force_address: int | None = None, verbose: bool = False) -> list[int]:
    step_frame = [0] * len(EIGHT_STEP)
    half_flag = 0
    min_y_value = 0.0

    if force_address is not None:
        state = 1
        step_frame[get_index_eight("address")] = force_address
        start_frame = force_address + 1
        if verbose:
            print(f"[Golf]   address={force_address} (forced)")
    else:
        state = -1
        start_frame = 2

    for frame in range(start_frame, video_length):
        if pose_landmark[0][0][frame] == 0:
            continue

        if state == -1:
            state += 1

        elif state == 0:
            diff_right_wrist = [
                max(pose_landmark[get_index_list("right_wrist")][0][frame - 2 : frame + 1])
                - min(pose_landmark[get_index_list("right_wrist")][0][frame - 2 : frame + 1]),
                max(pose_landmark[get_index_list("right_wrist")][1][frame - 2 : frame + 1])
                - min(pose_landmark[get_index_list("right_wrist")][1][frame - 2 : frame + 1]),
            ]
            diff_right_shoulder = [
                max(pose_landmark[get_index_list("right_shoulder")][0][frame - 2 : frame + 1])
                - min(pose_landmark[get_index_list("right_shoulder")][0][frame - 2 : frame + 1]),
                max(pose_landmark[get_index_list("right_shoulder")][1][frame - 2 : frame + 1])
                - min(pose_landmark[get_index_list("right_shoulder")][1][frame - 2 : frame + 1]),
            ]

            if all(x <= 0.0005 for x in diff_right_wrist[:2] + diff_right_shoulder[:2]):
                step_frame[get_index_eight("address")] = frame
                if verbose:
                    print(f"[Golf]   address={frame}")
                state += 1

        elif state == 1:
            if (
                pose_landmark[get_index_list("left_hip")][1][frame]
                >= pose_landmark[get_index_list("right_wrist")][1][frame]
                and pose_landmark[get_index_list("right_hip")][1][frame]
                >= pose_landmark[get_index_list("right_wrist")][1][frame]
            ):
                step_frame[get_index_eight("take_away")] = frame
                if verbose:
                    print(f"[Golf]   take_away={frame}")
                state += 1

        elif state == 2:
            step_frame[get_index_eight("half")] = frame
            if (
                pose_landmark[get_index_list("right_shoulder")][1][step_frame[get_index_eight("address")]]
                >= pose_landmark[get_index_list("right_wrist")][1][frame]
            ):
                min_y_value = max(pose_landmark[get_index_list("right_wrist")][1][frame:video_length])
                if verbose:
                    print(f"[Golf]   half={frame}")
                state += 1

        elif state == 3:
            half_value = (
                min(pose_landmark[get_index_list("right_wrist")][1][step_frame[get_index_eight("address")] : video_length])
                + pose_landmark[get_index_list("right_wrist")][1][get_index_eight("address")]
            ) / 2

            if pose_landmark[get_index_list("right_wrist")][1][frame] <= min_y_value:
                min_y_value = pose_landmark[get_index_list("right_wrist")][1][frame]
                step_frame[get_index_eight("top")] = frame

            ys = pose_landmark[get_index_list("right_wrist")][1][frame : frame + 3]
            if half_flag == 0:
                if sum(y <= half_value for y in ys) >= 3:
                    half_flag = 1
            elif half_flag == 1:
                if sum(y >= half_value for y in ys) >= 3:
                    if verbose:
                        print(f"[Golf]   top={step_frame[get_index_eight('top')]}")
                    state += 1

        elif state == 4:
            takeaway_to_top_length = step_frame[get_index_eight("top")] - step_frame[get_index_eight("take_away")]
            wrist_y = pose_landmark[get_index_list("right_wrist")][1][frame : frame + takeaway_to_top_length]
            if wrist_y and wrist_y[0] == max(wrist_y):
                step_frame[get_index_eight("impact")] = frame
                if verbose:
                    print(f"[Golf]   impact={frame}")
                state += 1

                y_vals = pose_landmark[get_index_list("right_wrist")][1]
                deltas = [
                    abs(y_vals[i + 1] - y_vals[i])
                    for i in range(step_frame[get_index_eight("address")], step_frame[get_index_eight("take_away")])
                ]
                if deltas:
                    avg_delta = sum(deltas) / len(deltas)
                    address_flag = 0
                    for f in range(step_frame[get_index_eight("take_away")], step_frame[get_index_eight("address")], -1):
                        delta_f = abs(y_vals[f] - y_vals[f - 1])
                        if delta_f < avg_delta:
                            address_flag += 1
                            if address_flag == 7:
                                step_frame[get_index_eight("address")] = max(f, 0)
                                if verbose:
                                    print(f"[Golf]   address refined={max(f, 0)}")
                                break

        elif state == 5:
            shoulder = (
                pose_landmark[get_index_list("left_shoulder")][1][frame]
                + pose_landmark[get_index_list("right_shoulder")][1][frame]
            ) / 2
            hip = (
                pose_landmark[get_index_list("left_hip")][1][frame]
                + pose_landmark[get_index_list("right_hip")][1][frame]
            ) / 2
            base_y = hip - abs(shoulder - hip) / 3
            if base_y > pose_landmark[get_index_list("right_wrist")][1][frame]:
                step_frame[get_index_eight("follow_through")] = frame
                if verbose:
                    print(f"[Golf]   follow_through={frame}")
                state += 1

        elif state == 6:
            follow = step_frame[get_index_eight("follow_through")]
            top = step_frame[get_index_eight("top")]
            search_end = min(follow + ((follow - top) * 2), video_length)
            wrist_y = [pose_landmark[get_index_list("right_wrist")][1][f] for f in range(follow, search_end)]
            if wrist_y:
                finish_idx = wrist_y.index(min(wrist_y)) + follow
                step_frame[get_index_eight("finish")] = finish_idx

            for idx_frame in range(step_frame[get_index_eight("finish")], video_length):
                wrist_y_value = pose_landmark[get_index_list("left_wrist")][1][idx_frame]
                threshold = (
                    pose_landmark[get_index_list("left_shoulder")][1][step_frame[get_index_eight("finish")]]
                    + pose_landmark[get_index_list("left_wrist")][1][step_frame[get_index_eight("finish")]]
                ) / 2
                if threshold >= wrist_y_value:
                    step_frame[get_index_eight("finish")] = idx_frame
                    break

            down_wrist_y = pose_landmark[get_index_list("right_wrist")][1][
                step_frame[get_index_eight("top")] : step_frame[get_index_eight("impact")]
            ]
            down_shoulder_y = pose_landmark[get_index_list("left_shoulder")][1][
                step_frame[get_index_eight("top")] : step_frame[get_index_eight("impact")]
            ]
            for down_frame in range(len(down_wrist_y)):
                if down_wrist_y[down_frame] >= down_shoulder_y[down_frame]:
                    step_frame[get_index_eight("down_half")] = down_frame + step_frame[get_index_eight("top")]
                    break
            if verbose:
                print(f"[Golf]   finish={step_frame[get_index_eight('finish')]}  down_half={step_frame[get_index_eight('down_half')]}")
            break

    return step_frame


def wrist_smoothing(
    step_frame: list[int],
    pose_landmark: list[list[list[float]]],
    cv_pose_landmark: list[list[list[float]]],
    width: int,
    height: int,
) -> None:
    take_away = step_frame[get_index_eight("take_away")]
    top = step_frame[get_index_eight("top")]
    impact = step_frame[get_index_eight("impact")]
    finish = step_frame[get_index_eight("finish")]

    if 0 in [take_away, top, impact, finish] or top <= take_away or finish <= impact:
        return

    step_min = min(pose_landmark[get_index_list("right_wrist")][0][take_away:top])
    min_index = pose_landmark[get_index_list("right_wrist")][0][take_away:top].index(step_min) + take_away
    xslope, xintercept = line_2d([take_away, pose_landmark[get_index_list("right_wrist")][0][take_away]], [min_index, step_min])
    for frame in range(take_away, min_index):
        pose_landmark[get_index_list("right_wrist")][0][frame] = xslope * frame + xintercept
        cv_pose_landmark[get_index_list("right_wrist")][0][frame] = pose_landmark[get_index_list("right_wrist")][0][frame] * width

    step_min = min(pose_landmark[get_index_list("left_wrist")][1][impact:finish])
    min_index = pose_landmark[get_index_list("left_wrist")][1][impact:finish].index(step_min) + impact
    if min_index - 2 <= impact:
        return

    xslope, xintercept = line_2d(
        [impact, pose_landmark[get_index_list("left_wrist")][0][impact]],
        [min_index - 2, pose_landmark[get_index_list("left_wrist")][0][min_index - 2]],
    )
    yslope, yintercept = line_2d(
        [impact, pose_landmark[get_index_list("left_wrist")][1][impact]],
        [min_index - 2, pose_landmark[get_index_list("left_wrist")][1][min_index - 2]],
    )
    start = impact + (min_index - impact) // 3
    for frame in range(start, min_index + 1):
        pose_landmark[get_index_list("right_wrist")][0][frame] = xslope * frame + xintercept + 10 / width
        pose_landmark[get_index_list("right_wrist")][1][frame] = yslope * frame + yintercept
        cv_pose_landmark[get_index_list("right_wrist")][0][frame] = pose_landmark[get_index_list("right_wrist")][0][frame] * width
        cv_pose_landmark[get_index_list("right_wrist")][1][frame] = pose_landmark[get_index_list("right_wrist")][1][frame] * height

    xslope, xintercept = line_2d(
        [impact, pose_landmark[get_index_list("right_wrist")][0][impact]],
        [start, pose_landmark[get_index_list("right_wrist")][0][start]],
    )
    yslope, yintercept = line_2d(
        [impact, pose_landmark[get_index_list("right_wrist")][1][impact]],
        [start, pose_landmark[get_index_list("right_wrist")][1][start]],
    )
    for frame in range(impact, start):
        pose_landmark[get_index_list("right_wrist")][0][frame] = xslope * frame + xintercept
        pose_landmark[get_index_list("right_wrist")][1][frame] = yslope * frame + yintercept
        cv_pose_landmark[get_index_list("right_wrist")][0][frame] = pose_landmark[get_index_list("right_wrist")][0][frame] * width
        cv_pose_landmark[get_index_list("right_wrist")][1][frame] = pose_landmark[get_index_list("right_wrist")][1][frame] * height


def extract_pose(video_path: Path, model: YOLO, device: str) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cv_pose = [[[], [], [], [], []] for _ in KEYPOINTS]
    empty_frames: set[int] = set()
    width = 0
    height = 0
    frame = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        if frame == 0:
            height, width = image.shape[:2]

        results = model.predict(image, device=device, verbose=False)
        result = results[0] if results else None
        if (
            result is None
            or result.keypoints is None
            or result.keypoints.xy is None
            or len(result.keypoints.xy) == 0
        ):
            for joint in range(len(KEYPOINTS)):
                for axis in range(5):
                    cv_pose[joint][axis].append(0.0)
            empty_frames.add(frame)
            frame += 1
            continue

        person_idx = get_largest_person_idx(result)
        keypoints_xy = result.keypoints.xy[person_idx].detach().cpu().numpy()
        keypoints_xyn = result.keypoints.xyn[person_idx].detach().cpu().numpy()
        keypoints_conf = None
        if getattr(result.keypoints, "conf", None) is not None:
            keypoints_conf = result.keypoints.conf[person_idx].detach().cpu().numpy()

        for joint, name in enumerate(KEYPOINTS):
            yolo_idx = YOLO_INDEX[name]
            if yolo_idx < len(keypoints_xy):
                cv_pose[joint][0].append(float(keypoints_xy[yolo_idx][0]))
                cv_pose[joint][1].append(float(keypoints_xy[yolo_idx][1]))
                cv_pose[joint][2].append(float(keypoints_xyn[yolo_idx][0]))
                cv_pose[joint][3].append(float(keypoints_xyn[yolo_idx][1]))
                conf = float(keypoints_conf[yolo_idx]) if keypoints_conf is not None else 0.0
                cv_pose[joint][4].append(conf)
            else:
                for axis in range(5):
                    cv_pose[joint][axis].append(0.0)
        frame += 1

    cap.release()
    if frame == 0 or fps == 0 or width == 0 or height == 0:
        raise RuntimeError(f"Video read failed: {video_path}")

    pose = smooth_landmarks(cv_pose, empty_frames, width, height, window_len=3)
    step_frames = get_step(frame, pose)
    if (
        0 in step_frames[1:]
        or len(step_frames) != len(set(step_frames))
        or abs(step_frames[get_index_eight("top")] - step_frames[get_index_eight("impact")]) < 2
    ):
        print(f"[Golf] Step detection failed, retrying with address=0: {step_frames}")
        step_frames = get_step(frame, pose, force_address=0, verbose=True)
    if 0 in step_frames[1:]:
        missing = EIGHT_STEP[step_frames.index(0)]
        raise RuntimeError(f"Golf step not found: {missing}, raw={step_frames}")
    if len(step_frames) != len(set(step_frames)):
        raise RuntimeError(f"Golf step frame duplicated: {step_frames}")
    if abs(step_frames[get_index_eight("top")] - step_frames[get_index_eight("impact")]) < 2:
        raise RuntimeError(f"Golf impact step not found: {step_frames}")

    step_dict_log = {step: frame_num for step, frame_num in zip(EIGHT_STEP, step_frames)}
    print(f"[Golf] Steps: {step_dict_log}")

    wrist_smoothing(step_frames, pose, cv_pose, width, height)
    step_dict = {step: int(frame_num) for step, frame_num in zip(EIGHT_STEP, step_frames)}

    # Drop frames well past the finish keyframe: the swing motion of interest
    # is over by then, and keeping the rest would bleed unrelated tail
    # footage (e.g. the golfer walking off) into the finish phase segment.
    saved_frame_count = min(frame, step_dict["finish"] + FINISH_TAIL_BUFFER + 1)
    saved_empty_frames = {f for f in empty_frames if f < saved_frame_count}

    return {
        "sport": "Golf",
        "video": str(video_path),
        "video_name": video_path.name,
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": saved_frame_count,
        "empty_frames": sorted(saved_empty_frames),
        "steps": step_dict,
        "coordinates": build_coordinates(pose, cv_pose, saved_frame_count),
    }


def build_coordinates(
    pose: list[list[list[float]]],
    cv_pose: list[list[list[float]]],
    frame_count: int,
) -> list[dict[str, Any]]:
    frames = []
    for frame_num in range(frame_count):
        landmarks = {}
        for joint, name in enumerate(KEYPOINTS):
            landmarks[name] = {
                "x": round(float(pose[joint][0][frame_num]), 6),
                "y": round(float(pose[joint][1][frame_num]), 6),
                "pixel_x": round(float(cv_pose[joint][0][frame_num]), 3),
                "pixel_y": round(float(cv_pose[joint][1][frame_num]), 3),
                "score": round(float(pose[joint][4][frame_num]), 6),
            }
        frames.append({"frame": frame_num, "landmarks": landmarks})
    return frames


def resolve_model_path(model_arg: str) -> str:
    candidate = Path(model_arg)
    script_dir = Path(__file__).resolve().parent
    for path in [candidate, script_dir / model_arg, script_dir.parent / model_arg]:
        if path.exists():
            return str(path)
    return model_arg


def collect_videos(inputs: list[str]) -> list[Path]:
    videos: list[Path] = []
    extensions = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            videos.extend(sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in extensions))
        else:
            matches = [Path(p) for p in glob.glob(item)]
            videos.extend(matches if matches else [path])
    return videos


def sample_stem_from_video_path(video_path: Path) -> str:
    """Build sample id from optional person folder and video filename."""
    stem = video_path.stem
    person_id = video_path.parent.name
    if person_id.lower() in {"expert", "user", "videos"}:
        return stem
    if stem.startswith(f"{person_id}_"):
        return stem
    return f"{person_id}_{stem}"


def person_id_from_sample_id(sample_id: str) -> str:
    return sample_id.rsplit("_", 1)[0]


def save_json(data: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_id = str(data.get("sample_id") or Path(data["video_name"]).stem)
    person_id = person_id_from_sample_id(sample_id)
    output_path = output_dir / person_id / f"{sample_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    return output_path


def infer_subject_type(video_path: Path, subject_type: str) -> str:
    if subject_type != "auto":
        return subject_type
    for part in reversed(video_path.parts):
        lowered = part.lower()
        if lowered in {"expert", "user"}:
            return lowered
    return "user"


def resolve_output_dir(video_path: Path, output_dir: str | None, subject_type: str) -> Path:
    if output_dir:
        return Path(output_dir)
    subject = infer_subject_type(video_path, subject_type)
    return Path(__file__).resolve().parent / "data" / "Golf" / "jsons" / subject


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO pose extraction and golf 8-step detection.")
    parser.add_argument("inputs", nargs="+", help="Video file, directory, or glob pattern.")
    parser.add_argument("--model", default="yolo26l-pose.pt", help="YOLO pose model path.")
    parser.add_argument("--device", default="auto", help="Inference device: auto, cpu, or cuda:0.")
    parser.add_argument(
        "--subject-type",
        choices=["auto", "expert", "user"],
        default="auto",
        help="JSON output group. auto infers expert/user from the input video path.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output JSON files.",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    model = YOLO(resolve_model_path(args.model), task="pose")
    model.to(device)
    print(f"[Golf] Device: {device}")

    processed = 0
    failed = 0
    skipped = 0
    for video_path in collect_videos(args.inputs):
        sample_stem = sample_stem_from_video_path(video_path)
        person_id = person_id_from_sample_id(sample_stem)
        output_dir = resolve_output_dir(video_path, args.output_dir, args.subject_type)
        expected_output_path = output_dir / person_id / f"{sample_stem}.json"
        if expected_output_path.exists():
            skipped += 1
            print(f"[Golf] Skip existing: {expected_output_path}")
            continue

        print(f"[Golf] Processing {video_path}")
        try:
            data = extract_pose(video_path, model, device)
            data["video_name"] = f"{sample_stem}{video_path.suffix}"
            data["sample_id"] = sample_stem
            data["model"] = args.model
            data["device"] = device
            data["subject_type"] = infer_subject_type(video_path, args.subject_type)
            output_path = save_json(data, output_dir)
            processed += 1
            print(f"[Golf] Saved {output_path}")
        except Exception as exc:
            failed += 1
            print(f"[Golf] Skip failed video: {video_path} | {type(exc).__name__}: {exc}")

    print(f"[Golf] Done. processed={processed}, failed={failed}, skipped={skipped}")


if __name__ == "__main__":
    main()
