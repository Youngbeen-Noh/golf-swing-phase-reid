"""Build a blank ground-truth labeling template for golf event detection (P0-4).

Scans the holdout pose JSONs (data/Golf/jsons/user/*/*.json — the same 91
samples used for known-pro holdout retrieval) and writes one CSV row per
sample with the video path, fps, and frame_count, plus 8 empty columns (one
per EIGHT_STEP event) for a human to fill in by watching the video.

The auto-detected `steps` values already stored in each JSON are deliberately
NOT included here, so the labeler is not anchored on the automatic detector's
answer while labeling.

Usage:
  python 12a_make_event_gt_template.py
  python 12a_make_event_gt_template.py --json-dir data/Golf/jsons/user --out event_detection_gt_template.csv

After filling in the CSV:
  - Enter the 0-indexed frame number (matching the video's own frame count)
    for each of the 8 events you can confidently identify.
  - If an event cannot be confidently identified in a given clip (occlusion,
    ambiguous motion, etc.), leave its cell blank AND write why in
    `exclude_reason` for that row — the evaluation script (12) will exclude
    the whole sample from metrics and report it separately, rather than
    silently treating a blank as frame 0.
  - Leave `exclude_reason` blank for normally-labeled rows.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

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


def load_json(path: Path) -> dict:
    import json

    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def build_rows(json_dir: Path) -> list[dict]:
    paths = sorted(json_dir.glob("*/*.json"))
    rows = []
    for path in paths:
        data = load_json(path)
        row = {
            "sample_id": data.get("sample_id", path.stem),
            "video": data.get("video", ""),
            "fps": data.get("fps", ""),
            "frame_count": data.get("frame_count", ""),
        }
        for event in EIGHT_STEP:
            row[f"{event}_gt"] = ""
        row["exclude_reason"] = ""
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json-dir", default="data/Golf/jsons/user", help="Holdout pose JSON directory.")
    parser.add_argument("--out", default="event_detection_gt_template.csv", help="Output CSV path.")
    args = parser.parse_args()

    json_dir = Path(args.json_dir)
    rows = build_rows(json_dir)
    if not rows:
        raise SystemExit(f"No JSONs found under {json_dir}")

    fieldnames = ["sample_id", "video", "fps", "frame_count"] + [f"{e}_gt" for e in EIGHT_STEP] + ["exclude_reason"]
    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {out_path}  ({len(rows)} rows, all GT columns blank)")
    print("Fill in the *_gt columns by watching each `video`, or set exclude_reason to skip a sample.")


if __name__ == "__main__":
    main()
