"""DMSM (Dynamic Motion Similarity Measurement) baseline reimplementation
(2nd revision instruction 4, "Prior work: DMSM 동일 데이터 재구현").

Faithful to Lee, S.-S.; Choi, J.-H.; Byun, J.; Hwang, K.-I. "Dynamic Golf
Swing Analysis Framework Based on Efficient Similarity Assessment." Sensors
2025, 25, 7073 (this paper's own reference [3]/[24], NOT the RL-correction
paper Sensors 2026, 26, 392, which is a different work by an overlapping
author set).

Algorithm (paper Section 3.4, Eq. 1-2, Figure 8):
  1. Segment the swing into seven canonical phases (this repo's existing
     phase_synchronized_resample / step feature array already does this --
     DMSM's own "phase-wise temporal alignment" is the same idea as this
     repo's GOLF_LEGACY_STEP_TARGET-based per-segment resampling, so no new
     preprocessing is needed; the existing "steps" feature array (7, 20, J,
     5) IS the phase-aligned representation DMSM operates on).
  2. Address-phase alignment (Figure 8, "offset@address"): every joint
     trajectory is shifted so the address-frame position is the origin,
     removing camera-frame translation before comparison.
  3. Per phase, per joint, per coordinate (x, y), the dissimilarity is the
     integral of the absolute difference between the two trajectories
     (Eq. 1), approximated via the trapezoidal rule on the phase-aligned
     20-frame grid (Eq. 2). This repo sums |dx| + |dy| per (phase, joint)
     via np.trapz and totals across all 7 phases and 18 joints -- the
     paper's core (unweighted) variant; it mentions an "optional" joint-
     weighted variant (hips/spine/hands emphasized) that is NOT implemented
     here since the paper does not give explicit weights.

Two things the paper does NOT specify in closed form, so are documented
choices made here rather than paper-derived facts:
  - Converting the raw dissimilarity integral D_total into a bounded
    "similarity" score (paper Table 3 reports "mean similarity", but Eq. 1-2
    only define a dissimilarity/distance). This script uses
    similarity = 1 / (1 + D_total), a standard, monotonic, paper-orthogonal
    choice -- Top-1 retrieval only depends on the ranking, which
    1/(1+D_total) preserves exactly (it is a strictly decreasing function of
    D_total), so this choice cannot change *which* expert is predicted,
    only the reported "score" magnitude.
  - The paper evaluates pairwise clip-vs-clip dissimilarity directly (its
    own same-/different-player experiment), not clip-vs-gallery retrieval.
    To plug DMSM into this paper's retrieval evaluation protocol (matching
    every other baseline/condition), each holdout query is compared against
    every training clip of every expert, and an expert's dissimilarity is
    the MINIMUM (nearest-neighbor) over that expert's clips -- consistent
    with the "gallery control" and "prototype robustness" items in
    BASELINES.md, which are still open.

Output is pro_holdout_user_metrics.json-compatible, so it plugs directly
into 10_aggregate_seed_metrics.py, 11/19_paired_significance*.py, and
20_query_level_logistic_variance.py unchanged.

Usage:
  python 22_evaluate_dmsm_baseline.py \
      --expert-feature-dir data/Golf/experiments/ATCN_Golf-D/features/expert \
      --user-feature-dir data/Golf/experiments/ATCN_Golf-D/features/user \
      --out data/Golf/experiments/ATCN_Golf-D/results/evaluation/dmsm_pro_holdout_metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

# np.trapz was removed in NumPy 2.0 (renamed np.trapezoid). A ternary
# short-circuits correctly here (np.trapz is only accessed if the "if" branch
# is false); `getattr(np, "trapezoid", np.trapz)` does NOT, since Python
# evaluates a function call's arguments -- including the default-value
# argument np.trapz -- before the call, so the attribute access happens
# unconditionally and raises AttributeError under NumPy >= 2.0 even though
# "trapezoid" exists and getattr would never have needed the fallback.
_TRAPEZOID = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def load_steps_xy(path: Path) -> np.ndarray:
    """(7, 20, J, 2) phase-aligned x,y trajectories, address-frame-offset."""
    with np.load(path, allow_pickle=False) as data:
        steps = data["steps"][:, :, :, :2].astype(np.float64)  # drop score/dx/dy
    address_position = steps[0, 0]  # phase 0 ("address_to_take_away"), frame 0
    return steps - address_position[None, None, :, :]


def dmsm_dissimilarity(a: np.ndarray, b: np.ndarray) -> float:
    """Eq. 1-2: sum over phases and joints of the trapezoidal integral of
    |x_a(t)-x_b(t)| + |y_a(t)-y_b(t)| along each phase's 20-frame grid."""
    diff = np.abs(a - b).sum(axis=-1)  # (7, 20, J): |dx|+|dy| per frame/joint
    t = np.linspace(0.0, 1.0, a.shape[1])
    per_phase_joint = _TRAPEZOID(diff, x=t, axis=1)  # (7, J)
    return float(per_phase_joint.sum())


def collect_expert_gallery(expert_dir: Path) -> dict[str, list[np.ndarray]]:
    gallery: dict[str, list[np.ndarray]] = {}
    for path in sorted(expert_dir.rglob("*.npz")):
        expert_id = path.parent.name
        gallery.setdefault(expert_id, []).append(load_steps_xy(path))
    return gallery


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--expert-feature-dir", required=True, help="data/{Sport}/experiments/{run_id}/features/expert")
    parser.add_argument("--user-feature-dir", required=True, help="data/{Sport}/experiments/{run_id}/features/user")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    gallery = collect_expert_gallery(Path(args.expert_feature_dir))
    if not gallery:
        raise SystemExit(f"No expert feature files found under {args.expert_feature_dir}")

    user_paths = sorted(Path(args.user_feature_dir).rglob("*.npz"))
    if not user_paths:
        raise SystemExit(f"No user feature files found under {args.user_feature_dir}")

    rows: list[dict[str, Any]] = []
    correct = 0
    for query_path in user_paths:
        true_expert = query_path.parent.name
        sample_id = query_path.stem
        query = load_steps_xy(query_path)

        expert_dissimilarity = {
            expert_id: min(dmsm_dissimilarity(query, candidate) for candidate in candidates)
            for expert_id, candidates in gallery.items()
        }
        ranked = sorted(expert_dissimilarity, key=lambda e: expert_dissimilarity[e])
        final_rank = ranked.index(true_expert) + 1 if true_expert in ranked else len(ranked) + 1
        if final_rank == 1:
            correct += 1
        rows.append(
            {
                "expert_id": true_expert,
                "sample_id": sample_id,
                "final_rank": final_rank,
                "final_top1": ranked[0] if ranked else "",
                "dmsm_dissimilarity_to_true_expert": round(expert_dissimilarity.get(true_expert, float("nan")), 4),
            }
        )

    # Nest under "final_ranking" to match 07_evaluate.py's pro_holdout schema --
    # see the identical fix/comment in 21_evaluate_ce_classifier.py. Without
    # this, 10_aggregate_seed_metrics.py silently drops every DMSM run with
    # "No usable metrics json" (final_ranking missing).
    final_ranking_block = {
        "count": len(rows),
        "top1_accuracy": correct / len(rows),
        "mrr": float(np.mean([1.0 / r["final_rank"] for r in rows])),
        "mean_rank": float(np.mean([r["final_rank"] for r in rows])),
    }
    summary = {
        **final_ranking_block,
        "final_ranking": final_ranking_block,
        "evaluation_mode": "dmsm_nearest_neighbor_gallery",
        "method": "DMSM (Lee, S.-S. et al., Sensors 2025, 25, 7073), unweighted core variant, reimplemented",
    }
    payload = {"summary": summary, "rows": rows}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
