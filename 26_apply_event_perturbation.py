"""Event perturbation robustness check (2nd revision instructions, "recommended"
tier): synthetically perturb the QUERY side's automatic-detector event
positions by a controlled random magnitude (+-5, +-10, +-20 frames) and
re-evaluate retrieval accuracy with the SAME already-trained checkpoint (no
retraining needed) -- extends the existing manual-vs-automatic event
comparison (Section 4.5, 24_apply_manual_event_gt.py) from "how far off is
the real automatic detector" to "how much event-position noise, of a known
controlled magnitude, can the model tolerate before retrieval degrades".

Scope: only the 39-clip VALIDATION subset is touched (never locked-test,
Section 3.12, evaluated exactly once). The gallery/expert-prototype side is
built once from the training pool with automatic events and is never
touched here; only query tensors change, so the SAME already-trained
checkpoint and prototype store can be reused directly.

Perturbation design: each of the 8 event frame indices is shifted
independently by an integer offset drawn uniformly from [-magnitude,
+magnitude] (magnitude in {5, 10, 20}), using a deterministic RNG seeded by
(seed, sample_id, magnitude) so every magnitude level can be run under the
same 5 seeds (42-46) already used throughout this paper for a real mean+-SD
robustness curve, rather than a single perturbed draw per magnitude.
Perturbed values are clipped to [event_(i-1)+1, event_(i+1)-1] using the
ORIGINAL (un-perturbed) neighbor positions as the clip bound -- a simple,
order-preserving approximation, not an exact joint constraint over all 8
events simultaneously -- and to [0, frame_count-1], so the strict
e_1 < e_2 < ... < e_8 ordering (Section 3.1) is never violated.

Usage (repeat per magnitude x seed; each writes an independent tree):
  python 26_apply_event_perturbation.py --json-dir data/Golf/jsons/user \
      --magnitude 10 --seed 42 --out-root event_perturbation_root/mag10_s42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

STEP_NAMES = [
    "address", "take_away", "half", "top", "down_half", "impact", "follow_through", "finish",
]


def perturb_events(steps: dict, frame_count: int, magnitude: int, rng: random.Random) -> tuple[dict, dict]:
    """Independently perturb each event by a random offset in
    [-magnitude, +magnitude], clipped to stay strictly between its two
    ORIGINAL neighbors and within [0, frame_count-1]. Returns
    (perturbed_steps, raw_offset_by_event) -- the latter for diagnostics
    (how often clipping actually kicked in)."""
    original = [steps[name] for name in STEP_NAMES]
    perturbed: list[int] = []
    raw_offsets: dict[str, int] = {}
    for i, val in enumerate(original):
        offset = rng.randint(-magnitude, magnitude)
        raw_offsets[STEP_NAMES[i]] = offset
        new_val = val + offset
        lo = perturbed[i - 1] + 1 if i > 0 else 0
        hi = original[i + 1] - 1 if i + 1 < len(original) else frame_count - 1
        new_val = max(lo, min(hi, new_val)) if lo <= hi else val
        perturbed.append(new_val)
    return dict(zip(STEP_NAMES, perturbed)), raw_offsets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json-dir", default="data/Golf/jsons/user", help="Current (validation-subset) holdout JSON directory.")
    parser.add_argument("--magnitude", type=int, required=True, choices=[5, 10, 20])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()

    json_dir = Path(args.json_dir)
    out_json_dir = Path(args.out_root) / "data" / "Golf" / "jsons" / "user"

    clipped_count = 0
    total_events = 0
    n_samples = 0
    for src_path in sorted(json_dir.rglob("*.json")):
        sample_id = src_path.stem
        data = json.loads(src_path.read_text(encoding="utf-8"))
        steps = data.get("steps", {})
        frame_count = int(data["frame_count"])
        rng = random.Random(f"{args.seed}:{sample_id}:eventperturb:{args.magnitude}")
        perturbed, raw_offsets = perturb_events(steps, frame_count, args.magnitude, rng)

        for name in STEP_NAMES:
            total_events += 1
            intended = steps[name] + raw_offsets[name]
            if perturbed[name] != intended:
                clipped_count += 1

        data["steps"] = perturbed
        dest_path = out_json_dir / src_path.parent.name / src_path.name
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        n_samples += 1

    print(
        f"Perturbed {n_samples} validation-subset samples at magnitude=+-{args.magnitude} frames, "
        f"seed={args.seed}, written to {out_json_dir}."
    )
    print(f"Clipping (order/bounds safety) affected {clipped_count}/{total_events} event positions.")
    if n_samples == 0:
        raise SystemExit(f"No samples found under {json_dir} -- check --json-dir points at the current validation subset.")

    run_tag = f"eventperturb-mag{args.magnitude}"
    # Seed goes LAST in the output run_id (ATCN_Golf-D-eventperturb-mag{N}-s{seed}),
    # matching 10_aggregate_seed_metrics.py's RUN_ID_PATTERN (model_method<extra>-s<seed>)
    # -- the same convention already used for the manual-events comparison
    # (ATCN_Golf-D-manualevents-s{seed}) -- so "Golf-D-eventperturb-mag{N}" is
    # aggregated as its own method bucket across the 5 seeds. This differs
    # from the SOURCE checkpoint's own run_id (ATCN_Golf-D-s{seed}, no extra
    # tag), which is the original, unperturbed training run being reused.
    out_rid = f"ATCN_Golf-D-{run_tag}-s{args.seed}"
    print(
        "\nNext steps (identical pattern to 24_apply_manual_event_gt.py):\n"
        f"  python 02_json_to_tensor.py --root {args.out_root} --sport Golf --subject-type user\n"
        f"  python 03_preprocess_and_build_features.py --root {args.out_root} --sport Golf "
        f"--preprocess-method Golf-D --subject-type user --run-id {run_tag} "
        f"--output-root {args.out_root}/data/Golf/experiments/{run_tag}\n"
        "  # then, reusing the REAL trained checkpoint for this seed against the perturbed features\n"
        "  # (SRC_RID = original training run this seed's checkpoint lives under;\n"
        "  #  OUT_RID = new label the perturbed-eval results get aggregated under):\n"
        f"  SRC_RID=ATCN_Golf-D-s{args.seed}\n"
        f"  OUT_RID={out_rid}\n"
        "  python 06_infer_user_similarity.py --sport Golf --run-id $OUT_RID \\\n"
        "    --checkpoint data/Golf/experiments/$SRC_RID/models/atcn_step_sequence_best.pt \\\n"
        "    --prototype-store data/Golf/experiments/$SRC_RID/prototypes/expert_prototypes.pkl \\\n"
        f"    --feature-dir {args.out_root}/data/Golf/experiments/{run_tag}/features/user \\\n"
        f"    --output-dir {args.out_root}/data/Golf/experiments/$OUT_RID/results/user_similarity\n"
        "  python 07_evaluate.py --sport Golf --run-id $OUT_RID \\\n"
        "    --checkpoint data/Golf/experiments/$SRC_RID/models/atcn_step_sequence_best.pt \\\n"
        "    --prototype-store data/Golf/experiments/$SRC_RID/prototypes/expert_prototypes.pkl \\\n"
        "    --feature-dir data/Golf/experiments/ATCN_Golf-D/features/expert \\\n"
        f"    --pro-holdout-result-dir {args.out_root}/data/Golf/experiments/$OUT_RID/results/user_similarity \\\n"
        f"    --pro-holdout-output {args.out_root}/data/Golf/experiments/$OUT_RID/results/evaluation/pro_holdout_user_metrics.json\n"
        "  # repeat for all 5 seeds (42-46) at this magnitude, then all 3 magnitudes (5, 10, 20), then:\n"
        f"  python 10_aggregate_seed_metrics.py {args.out_root}/data/Golf/experiments --out-prefix results_eventperturb_mag{args.magnitude}\n"
    )


if __name__ == "__main__":
    main()
