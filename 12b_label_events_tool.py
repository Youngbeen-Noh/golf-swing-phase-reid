"""Interactive frame-by-frame GT labeling helper for P0-4 (event detection eval).

Steps through event_detection_gt_template.csv one holdout sample at a time,
opens its video, and lets you mark each of the 8 EIGHT_STEP frames with the
keyboard. Progress is saved back into the same CSV after every sample, so you
can quit anytime (q) and resume later -- already-labeled or excluded rows are
skipped automatically on the next run.

Usage:
  python 12b_label_events_tool.py
  python 12b_label_events_tool.py --gt-csv event_detection_gt_template.csv
  python 12b_label_events_tool.py --only some-expert_004        # relabel one sample
  python 12b_label_events_tool.py --redo                       # revisit everything

Controls (video window must be focused):
  d / a        step +1 / -1 frame
  c / z        step +10 / -10 frames
  v / x        step +1s / -1s (fps frames)
  g            type an exact frame number in the terminal and jump to it
  SPACE        mark current frame as GT for the current event, advance to next
  u            undo: go back one event (re-mark it)
  n            skip this whole sample -- type a reason in the terminal
                (written to exclude_reason, no GT recorded)
  q            save progress and quit
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

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

WINDOW_NAME = "event GT labeler"


def load_rows(path: Path) -> tuple[list[dict], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return rows, fieldnames


def save_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def is_done(row: dict) -> bool:
    if (row.get("exclude_reason") or "").strip():
        return True
    return all((row.get(f"{e}_gt") or "").strip() != "" for e in EIGHT_STEP)


def draw_overlay(frame, sample_id: str, sample_idx: int, n_samples: int,
                  frame_idx: int, frame_count: int, fps: float,
                  event_idx: int, marks: dict[str, int]) -> None:
    lines = [
        f"[{sample_idx + 1}/{n_samples}] {sample_id}",
        f"event: {EIGHT_STEP[event_idx]} ({event_idx + 1}/8)   frame {frame_idx}/{frame_count - 1}   fps {fps:g}",
        "marked: " + ", ".join(f"{e}={marks[e]}" for e in EIGHT_STEP if e in marks),
        "a/d:-+1  z/c:-+10  x/v:-+1s  g:goto  SPACE:mark  u:undo  n:skip-sample  q:quit",
    ]
    h = 22 * len(lines) + 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (8, 20 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def label_sample(row: dict, video_root: Path, sample_idx: int, n_samples: int) -> bool:
    """Returns True if the sample was labeled/skipped (row mutated in place), False if aborted (quit)."""
    video_path = video_root / row["video"]
    if not video_path.exists():
        print(f"[warn] video not found, skipping: {video_path}")
        row["exclude_reason"] = "video file not found"
        return True

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[warn] could not open video, skipping: {video_path}")
        row["exclude_reason"] = "video failed to open"
        return True

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(row["fps"]) if row.get("fps") else cap.get(cv2.CAP_PROP_FPS)
    frame_count_meta = int(row["frame_count"]) if row.get("frame_count") else total_frames
    step_1s = max(1, round(fps))

    frame_idx = 0
    event_idx = 0
    marks: dict[str, int] = {}
    sample_id = row["sample_id"]

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    upper_hint = total_frames - 1 if total_frames > 0 else None

    def read_frame(idx: int, previous):
        idx = max(0, idx if upper_hint is None else min(idx, upper_hint))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            return idx, frame
        # cv2's reported frame count is unreliable for some codecs; a failed
        # seek near the true end shouldn't abort the whole sample.
        print(f"[warn] could not read frame {idx}, staying put.")
        return previous

    frame_idx, frame = read_frame(0, (0, None))

    if frame is None:
        print(f"[warn] failed to read frame 0 of {video_path}, skipping.")
        cap.release()
        row["exclude_reason"] = "could not read video frames"
        return True

    result = "continue"

    while True:

        display = frame.copy()
        draw_overlay(display, sample_id, sample_idx, n_samples, frame_idx, frame_count_meta, fps, event_idx, marks)
        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            result = "quit"
            break
        elif key == ord("d"):
            frame_idx, frame = read_frame(frame_idx + 1, (frame_idx, frame))
        elif key == ord("a"):
            frame_idx, frame = read_frame(frame_idx - 1, (frame_idx, frame))
        elif key == ord("c"):
            frame_idx, frame = read_frame(frame_idx + 10, (frame_idx, frame))
        elif key == ord("z"):
            frame_idx, frame = read_frame(frame_idx - 10, (frame_idx, frame))
        elif key == ord("v"):
            frame_idx, frame = read_frame(frame_idx + step_1s, (frame_idx, frame))
        elif key == ord("x"):
            frame_idx, frame = read_frame(frame_idx - step_1s, (frame_idx, frame))
        elif key == ord("g"):
            try:
                target = int(input(f"  go to frame (0-{total_frames - 1}): ").strip())
            except ValueError:
                print("  invalid frame number, ignored.")
            else:
                frame_idx, frame = read_frame(target, (frame_idx, frame))
        elif key == ord("u"):
            if event_idx > 0:
                event_idx -= 1
                marks.pop(EIGHT_STEP[event_idx], None)
                print(f"  undo -> re-marking '{EIGHT_STEP[event_idx]}'")
            else:
                print("  already at the first event, nothing to undo.")
        elif key == ord("n"):
            reason = input(f"  skip {sample_id} -- reason: ").strip()
            row["exclude_reason"] = reason or "skipped by labeler"
            result = "skipped"
            break
        elif key == 32:  # SPACE
            event = EIGHT_STEP[event_idx]
            marks[event] = frame_idx
            print(f"  {event} = {frame_idx}")
            event_idx += 1
            if event_idx >= len(EIGHT_STEP):
                result = "done"
                break

    cv2.destroyWindow(WINDOW_NAME)
    cap.release()

    if result == "quit":
        return False
    if result == "done":
        for event, f in marks.items():
            row[f"{event}_gt"] = str(f)
        print(f"  saved {sample_id}: {marks}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gt-csv", default="event_detection_gt_template.csv")
    parser.add_argument("--video-root", default=".", help="Root the CSV's `video` column is relative to.")
    parser.add_argument("--only", default=None, help="Label just this one sample_id (ignores done/skip state).")
    parser.add_argument("--redo", action="store_true", help="Revisit already-labeled/skipped rows too.")
    args = parser.parse_args()

    gt_csv = Path(args.gt_csv)
    if not gt_csv.exists():
        raise SystemExit(f"GT CSV not found: {gt_csv}. Run 12a_make_event_gt_template.py first.")

    rows, fieldnames = load_rows(gt_csv)
    video_root = Path(args.video_root)

    if args.only:
        targets = [i for i, r in enumerate(rows) if r["sample_id"] == args.only]
        if not targets:
            raise SystemExit(f"sample_id not found: {args.only}")
    else:
        targets = [i for i, r in enumerate(rows) if args.redo or not is_done(r)]

    if not targets:
        print("Nothing left to label (all rows done). Pass --redo to revisit.")
        return

    print(f"{len(targets)} sample(s) to label. Press 'q' anytime to save and quit.")
    n_samples = len(targets)
    for progress_idx, row_idx in enumerate(targets):
        row = rows[row_idx]
        keep_going = label_sample(row, video_root, progress_idx, n_samples)
        save_rows(gt_csv, rows, fieldnames)
        if not keep_going:
            print(f"Saved progress to {gt_csv}. Resume anytime by re-running this script.")
            return

    print(f"All {n_samples} sample(s) labeled. Saved to {gt_csv}.")


if __name__ == "__main__":
    main()
