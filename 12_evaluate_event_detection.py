"""Evaluate automatic golf event detection against human-labeled GT (P0-4).

Compares the auto-detected `steps` already stored in each holdout pose JSON
(data/Golf/jsons/user/*/*.json) against a human-labeled ground-truth CSV
produced from the 12a_make_event_gt_template.py template. None of these
holdout samples' events were manually corrected before being used for
retrieval evaluation elsewhere in the pipeline -- this script is the first
place that GT is compared against the automatic detector.

Exclusion handling (three disjoint buckets over the full template):
  - excluded_flagged   : `exclude_reason` is non-empty (researcher judged the
                          sample unusable for GT), OR the labeled GT frames
                          are not strictly increasing address < ... < finish
                          (labeling mistake -- the sample is dropped rather
                          than silently evaluated against invalid GT).
  - excluded_unlabeled : `exclude_reason` is empty but at least one *_gt cell
                          is still blank or unparseable (row not fully
                          labeled yet).
  - usable              : all 8 *_gt cells are filled, strictly increasing,
                          and exclude_reason is empty. Only these rows enter
                          the metrics below.

Two parallel error units are reported, because the 91 holdout clips are NOT
comparable on either raw frames or wall-clock time alone: their reported fps
spans ~20-60 (some near-round values look like true capture fps, others like
42.3/45.5 look like variable-frame-rate sources cv2 is mis-reporting), and on
top of that some clips are slow-motion captures where the same physical swing
is spread over more frames/more wall-clock seconds than a normal-speed clip.
Neither "N frames off" nor "N ms off" is then a fair unit across clips.

  1. frames   : auto_frame - gt_frame, exactly what the instruction doc's P0-4
                spec asks for (--tol, default 3/5 frames). Meaningful within
                a single clip or across clips that happen to share fps/speed,
                but not safely comparable across the full heterogeneous set.
  2. pct      : (auto_frame - gt_frame) / (gt_finish - gt_address) * 100 --
                error as a percentage of that SAME clip's own GT swing
                duration (address..finish). This is scale-invariant to fps
                and to slow-motion stretching, since both the error and the
                yardstick it's measured against come from the same clip.
                (--tol-pct, default 2/5 percent of swing length.) Treat this
                as the primary cross-clip metric; report frame-based
                alongside it because the instruction doc asks for frames
                explicitly.

"success rate" / sample-level failure are reported under both units the same
way: success at --success-tol frames, and separately at --success-tol-pct
percent of swing length. A usable sample "fails" (pct) if any of its 8 events
exceeds --success-tol-pct.

Note: this only evaluates detection quality on the 91 samples that already
survived automatic extraction (extraction failures upstream, in
01-1_extraction_golf.py, are raised and the video is skipped entirely rather
than saved as JSON -- no persistent log of those exists, so that separate,
upstream failure rate is NOT covered here; see required_information.md).

Usage:
  python 12a_make_event_gt_template.py
  # ... hand-label event_detection_gt_template.csv (12b_label_events_tool.py) ...
  python 12_evaluate_event_detection.py --gt-csv event_detection_gt_template.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

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


def load_json_steps(json_dir: Path, sample_id: str) -> dict[str, int] | None:
    matches = list(json_dir.glob(f"*/{sample_id}.json"))
    if not matches:
        return None
    with matches[0].open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    return data.get("steps")


def parse_frame(value: str) -> int | None:
    value = (value or "").strip()
    if value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def load_gt_rows(gt_csv: Path) -> list[dict[str, Any]]:
    with gt_csv.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def classify_rows(
    rows: list[dict[str, Any]], json_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    usable: list[dict[str, Any]] = []
    excluded_flagged: list[dict[str, Any]] = []
    excluded_unlabeled: list[dict[str, Any]] = []

    for row in rows:
        sample_id = row["sample_id"]
        reason = (row.get("exclude_reason") or "").strip()
        if reason:
            excluded_flagged.append({"sample_id": sample_id, "reason": reason})
            continue

        gt = {event: parse_frame(row.get(f"{event}_gt", "")) for event in EIGHT_STEP}
        if any(v is None for v in gt.values()):
            missing = [e for e, v in gt.items() if v is None]
            excluded_unlabeled.append({"sample_id": sample_id, "missing_events": ",".join(missing)})
            continue

        ordered = [gt[e] for e in EIGHT_STEP]
        if not all(ordered[i] < ordered[i + 1] for i in range(len(ordered) - 1)):
            excluded_flagged.append(
                {"sample_id": sample_id, "reason": f"GT frames not strictly increasing: {ordered}"}
            )
            continue

        auto = load_json_steps(json_dir, sample_id)
        if auto is None:
            excluded_flagged.append({"sample_id": sample_id, "reason": "no matching pose JSON found"})
            continue

        usable.append({"sample_id": sample_id, "gt": gt, "auto": auto})

    return usable, excluded_flagged, excluded_unlabeled


def summarize(
    values: list[float], tolerances: list[float], success_tol: float, unit: str
) -> dict[str, Any]:
    n = len(values)
    abs_values = sorted(abs(v) for v in values)
    mae = sum(abs_values) / n if n else float("nan")
    median = (
        abs_values[n // 2]
        if n % 2 == 1
        else (abs_values[n // 2 - 1] + abs_values[n // 2]) / 2
        if n
        else float("nan")
    )
    out: dict[str, Any] = {"n": n, f"mae_{unit}": mae, f"median_abs_diff_{unit}": median}
    for tol in tolerances:
        tol_key = str(tol).rstrip("0").rstrip(".") if isinstance(tol, float) else str(tol)
        out[f"acc_within_{tol_key}{unit[0]}"] = sum(1 for v in abs_values if v <= tol) / n if n else float("nan")
    out[f"success_rate_{unit}"] = sum(1 for v in abs_values if v <= success_tol) / n if n else float("nan")
    return out


def compute_metrics(
    usable: list[dict[str, Any]],
    tol_frames: list[int],
    success_tol_frames: int,
    tol_pct: list[float],
    success_tol_pct: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    per_sample_rows: list[dict[str, Any]] = []
    frame_diffs_by_event: dict[str, list[int]] = {e: [] for e in EIGHT_STEP}
    pct_diffs_by_event: dict[str, list[float]] = {e: [] for e in EIGHT_STEP}

    for sample in usable:
        swing_span = sample["gt"]["finish"] - sample["gt"]["address"]
        for event in EIGHT_STEP:
            gt_frame = sample["gt"][event]
            auto_frame = int(sample["auto"][event])
            diff = auto_frame - gt_frame
            diff_pct = diff / swing_span * 100.0
            frame_diffs_by_event[event].append(diff)
            pct_diffs_by_event[event].append(diff_pct)
            per_sample_rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "event": event,
                    "gt_frame": gt_frame,
                    "auto_frame": auto_frame,
                    "swing_span_frames": swing_span,
                    "diff_frames": diff,
                    "abs_diff_frames": abs(diff),
                    "diff_pct_of_swing": round(diff_pct, 3),
                    "abs_diff_pct_of_swing": round(abs(diff_pct), 3),
                }
            )

    per_event: list[dict[str, Any]] = []
    for event in EIGHT_STEP:
        row = {"event": event}
        row.update(summarize(frame_diffs_by_event[event], tol_frames, success_tol_frames, "frames"))
        row.update(summarize(pct_diffs_by_event[event], tol_pct, success_tol_pct, "pct"))
        per_event.append(row)

    all_frame_diffs = [d for diffs in frame_diffs_by_event.values() for d in diffs]
    all_pct_diffs = [d for diffs in pct_diffs_by_event.values() for d in diffs]
    overall = {"event": "ALL"}
    overall.update(summarize(all_frame_diffs, tol_frames, success_tol_frames, "frames"))
    overall.update(summarize(all_pct_diffs, tol_pct, success_tol_pct, "pct"))
    per_event.append(overall)

    n_samples = len(usable)
    failed_frames = 0
    failed_pct = 0
    for sample in usable:
        swing_span = sample["gt"]["finish"] - sample["gt"]["address"]
        worst_frames = max(abs(int(sample["auto"][e]) - sample["gt"][e]) for e in EIGHT_STEP)
        worst_pct = max(abs(int(sample["auto"][e]) - sample["gt"][e]) / swing_span * 100.0 for e in EIGHT_STEP)
        if worst_frames > success_tol_frames:
            failed_frames += 1
        if worst_pct > success_tol_pct:
            failed_pct += 1
    sample_summary = {
        "n_usable_samples": n_samples,
        "n_failed_samples_frames": failed_frames,
        "failure_rate_frames": failed_frames / n_samples if n_samples else float("nan"),
        "success_tol_frames": success_tol_frames,
        "n_failed_samples_pct": failed_pct,
        "failure_rate_pct": failed_pct / n_samples if n_samples else float("nan"),
        "success_tol_pct": success_tol_pct,
    }

    return per_event, sample_summary, per_sample_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        print(f"[warn] Nothing to write for {path}", file=sys.stderr)
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}  ({len(rows)} rows)")


def print_report(
    total_rows: int,
    excluded_flagged: list[dict[str, Any]],
    excluded_unlabeled: list[dict[str, Any]],
    per_event: list[dict[str, Any]],
    sample_summary: dict[str, Any],
) -> None:
    n_usable = total_rows - len(excluded_flagged) - len(excluded_unlabeled)
    print("\n== Event detection evaluation (P0-4) ==")
    print(f"template rows (before exclusion): {total_rows}")
    print(f"excluded (flagged unusable):       {len(excluded_flagged)}")
    print(f"excluded (not yet labeled):        {len(excluded_unlabeled)}")
    print(f"usable (evaluated):                {n_usable}")
    if excluded_flagged:
        print("  flagged reasons:")
        for row in excluded_flagged:
            print(f"    {row['sample_id']}: {row['reason']}")
    if excluded_unlabeled:
        print(f"  [warn] {len(excluded_unlabeled)} rows still missing GT values -- re-run after labeling for a complete P0-4 report.")

    print(f"\nsample-level failure, frame-based (>{sample_summary['success_tol_frames']}f on any event): "
          f"{sample_summary['n_failed_samples_frames']} ({sample_summary['failure_rate_frames']*100:.1f}%) "
          f"of {sample_summary['n_usable_samples']}")
    print(f"sample-level failure, pct-of-swing-based (>{sample_summary['success_tol_pct']:g}% on any event): "
          f"{sample_summary['n_failed_samples_pct']} ({sample_summary['failure_rate_pct']*100:.1f}%) "
          f"of {sample_summary['n_usable_samples']}")

    frame_tol_keys = [k for k in per_event[0] if k.startswith("acc_within") and k.endswith("f")]
    pct_tol_keys = [k for k in per_event[0] if k.startswith("acc_within") and k.endswith("p")]

    header = (
        f"{'event':<16}{'n':>4}  {'MAE(f)':>7}  {'MAE(%)':>7}  "
        + "  ".join(k.replace("acc_within_", "acc@") for k in frame_tol_keys)
        + "  "
        + "  ".join(k.replace("acc_within_", "acc@") for k in pct_tol_keys)
        + f"  {'succ(f)':>8}  {'succ(%)':>8}"
    )
    print(f"\n{header}")
    for row in per_event:
        frame_cols = "  ".join(f"{row[k]*100:6.1f}%" for k in frame_tol_keys)
        pct_cols = "  ".join(f"{row[k]*100:6.1f}%" for k in pct_tol_keys)
        print(
            f"{row['event']:<16}{row['n']:>4}  {row['mae_frames']:>7.2f}  {row['mae_pct']:>7.2f}  "
            f"{frame_cols}  {pct_cols}  {row['success_rate_frames']*100:7.1f}%  {row['success_rate_pct']*100:7.1f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gt-csv", default="event_detection_gt_template.csv", help="Hand-labeled GT CSV.")
    parser.add_argument("--json-dir", default="data/Golf/jsons/user", help="Holdout pose JSON directory.")
    parser.add_argument("--tol", type=int, nargs="+", default=[3, 5], help="Tolerance-accuracy thresholds in frames.")
    parser.add_argument("--success-tol", type=int, default=5, help="Frame tolerance defining per-event/per-sample success.")
    parser.add_argument(
        "--tol-pct", type=float, nargs="+", default=[2, 5],
        help="Tolerance-accuracy thresholds as percent of that clip's own GT swing duration (address..finish). "
        "Scale-invariant to fps/slow-motion, unlike --tol; treat as the primary cross-clip metric.",
    )
    parser.add_argument(
        "--success-tol-pct", type=float, default=5.0,
        help="Percent-of-swing tolerance defining per-event/per-sample success.",
    )
    parser.add_argument("--out-prefix", default="results_event_detection", help="Prefix for output CSVs.")
    args = parser.parse_args()

    gt_csv = Path(args.gt_csv)
    json_dir = Path(args.json_dir)
    if not gt_csv.exists():
        raise SystemExit(f"GT CSV not found: {gt_csv}. Run 12a_make_event_gt_template.py first.")

    rows = load_gt_rows(gt_csv)
    usable, excluded_flagged, excluded_unlabeled = classify_rows(rows, json_dir)

    per_event, sample_summary, per_sample_rows = compute_metrics(
        usable, args.tol, args.success_tol, args.tol_pct, args.success_tol_pct
    )

    print_report(len(rows), excluded_flagged, excluded_unlabeled, per_event, sample_summary)

    write_csv(Path(f"{args.out_prefix}_per_event.csv"), per_event)
    write_csv(Path(f"{args.out_prefix}_per_sample.csv"), per_sample_rows)
    exclusions = (
        [{"sample_id": r["sample_id"], "status": "excluded_flagged", "detail": r["reason"]} for r in excluded_flagged]
        + [{"sample_id": r["sample_id"], "status": "excluded_unlabeled", "detail": r["missing_events"]} for r in excluded_unlabeled]
        + [{"sample_id": s["sample_id"], "status": "usable", "detail": ""} for s in usable]
    )
    write_csv(Path(f"{args.out_prefix}_exclusions.csv"), exclusions)

    summary_row = {**sample_summary, "template_rows": len(rows),
                    "excluded_flagged": len(excluded_flagged), "excluded_unlabeled": len(excluded_unlabeled)}
    write_csv(Path(f"{args.out_prefix}_sample_summary.csv"), [summary_row])


if __name__ == "__main__":
    main()
