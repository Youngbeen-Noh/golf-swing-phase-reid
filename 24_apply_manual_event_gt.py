"""Event robustness (2nd revision instruction 4, "Manual events vs. automatic
events", required): rebuild the QUERY side only using hand-labeled event
positions instead of the automatic detector's, so the downstream retrieval
task can be re-evaluated with the automatic detector's error removed.

Scope: only the current 39-clip VALIDATION subset is touched -- the locked
test subset is deliberately left alone (it is evaluated exactly once,
Section 3.12, and this event-robustness check is not that one evaluation).
The gallery/expert-prototype side is unaffected either way, since it is
built once from the training pool with automatic events and this script
never touches it; only the query tensors change, so the SAME already-
trained checkpoint and prototype store can be reused directly (no
retraining needed).

Writes a self-contained data/ tree under --out-root (default:
manual_event_root/), structured exactly like the real repo root
(data/Golf/jsons/user/{expert}/{sample_id}.json) so 02_json_to_tensor.py
and 03_preprocess_and_build_features.py can run against it with --root
pointed there, unchanged.

Usage:
  python 24_apply_manual_event_gt.py --gt-csv event_detection_gt_template.csv \
      --json-dir data/Golf/jsons/user --out-root manual_event_root
  python 02_json_to_tensor.py --root manual_event_root --sport Golf --subject-type user
  python 03_preprocess_and_build_features.py --root manual_event_root --sport Golf \
      --preprocess-method Golf-D --subject-type user --run-id manualevents \
      --output-root manual_event_root/data/Golf/experiments/manualevents
  # then 06/07 with --checkpoint/--prototype-store pointing at the REAL,
  # already-trained run and --feature-dir pointing at the manual-event
  # features just built (see the printed command at the end of this script).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

STEP_NAMES = [
    "address", "take_away", "half", "top", "down_half", "impact", "follow_through", "finish",
]


def load_gt(gt_csv: Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    with gt_csv.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("exclude_reason"):
                continue
            steps: dict[str, int] = {}
            ok = True
            for name in STEP_NAMES:
                val = row.get(f"{name}_gt", "")
                if val in (None, ""):
                    ok = False
                    break
                steps[name] = int(val)
            if ok:
                out[row["sample_id"]] = steps
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gt-csv", default="event_detection_gt_template.csv")
    parser.add_argument("--json-dir", default="data/Golf/jsons/user", help="Current (validation-subset) holdout JSON directory.")
    parser.add_argument("--out-root", default="manual_event_root")
    args = parser.parse_args()

    gt = load_gt(Path(args.gt_csv))
    print(f"Loaded {len(gt)} fully-labeled, non-excluded manual GT samples from {args.gt_csv}")

    json_dir = Path(args.json_dir)
    out_json_dir = Path(args.out_root) / "data" / "Golf" / "jsons" / "user"

    matched = 0
    skipped_no_gt = 0
    for src_path in sorted(json_dir.rglob("*.json")):
        sample_id = src_path.stem
        if sample_id not in gt:
            skipped_no_gt += 1
            continue
        data = json.loads(src_path.read_text(encoding="utf-8"))
        old_steps = data.get("steps", {})
        data["steps"] = gt[sample_id]
        dest_path = out_json_dir / src_path.parent.name / src_path.name
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        matched += 1
        deltas = {k: gt[sample_id][k] - old_steps.get(k, gt[sample_id][k]) for k in STEP_NAMES}
        nonzero = {k: v for k, v in deltas.items() if v != 0}
        print(f"[patched] {sample_id}: {nonzero if nonzero else '(automatic detector was already exact)'}")

    print(f"\nDone. {matched} samples in the current validation subset have manual GT and were patched "
          f"into {out_json_dir} ({skipped_no_gt} validation-subset samples had no usable manual GT and were left out).")
    if matched == 0:
        raise SystemExit("No samples matched -- check --json-dir points at the CURRENT validation subset "
                          "(data/Golf/jsons/user) and --gt-csv covers those sample_ids.")
    print(
        "\nNext steps:\n"
        f"  python 02_json_to_tensor.py --root {args.out_root} --sport Golf --subject-type user\n"
        f"  python 03_preprocess_and_build_features.py --root {args.out_root} --sport Golf "
        f"--preprocess-method Golf-D --subject-type user --run-id manualevents "
        f"--output-root {args.out_root}/data/Golf/experiments/manualevents\n"
        "  # then, for each seed, reuse the REAL trained checkpoint against the manual-event features:\n"
        "  for SEED in 42 43 44 45 46; do\n"
        "    RID=ATCN_Golf-D-s$SEED\n"
        "    python 06_infer_user_similarity.py --sport Golf --run-id $RID \\\n"
        "      --checkpoint data/Golf/experiments/$RID/models/atcn_step_sequence_best.pt \\\n"
        "      --prototype-store data/Golf/experiments/$RID/prototypes/expert_prototypes.pkl \\\n"
        f"      --feature-dir {args.out_root}/data/Golf/experiments/manualevents/features/user \\\n"
        f"      --output-dir {args.out_root}/data/Golf/experiments/$RID/results/user_similarity\n"
        "    python 07_evaluate.py --sport Golf --run-id $RID \\\n"
        "      --checkpoint data/Golf/experiments/$RID/models/atcn_step_sequence_best.pt \\\n"
        "      --prototype-store data/Golf/experiments/$RID/prototypes/expert_prototypes.pkl \\\n"
        "      --feature-dir data/Golf/experiments/ATCN_Golf-D/features/expert \\\n"
        f"      --pro-holdout-result-dir {args.out_root}/data/Golf/experiments/$RID/results/user_similarity \\\n"
        f"      --pro-holdout-output {args.out_root}/data/Golf/experiments/$RID/results/evaluation/pro_holdout_user_metrics.json\n"
        "  done\n"
        f"  python 10_aggregate_seed_metrics.py {args.out_root}/data/Golf/experiments --out-prefix results_manualevents\n"
    )


if __name__ == "__main__":
    main()
