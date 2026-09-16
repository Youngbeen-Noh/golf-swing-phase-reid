from __future__ import annotations

"""Re-verify near_duplicate_review_list.csv candidate pairs with an independent,
automated signal: DTW alignment distance + mean-pose embedding cosine distance
on the actual 2D pose sequences (not the perceptual-hash-only signal the
original candidate list was built from). Revision instruction item 2: only 24
of 1,088 candidate pairs had been reviewed; this replaces ad hoc/manual review
with a reproducible, data-driven judgment for every pair.

Method:
  1. Load each clip's pose tensor (T, 18, 2) [x, y], drop the confidence channel.
  2. Per-frame normalize: center on mid-hip, scale by shoulder width (removes
     camera position/zoom/subject-scale differences, keeps swing shape/timing).
  3. DTW-align the two sequences (Euclidean per-frame cost over 36-dim flattened
     joints), normalized by the alignment path length -> dtw_distance.
  4. Also compute cosine distance between each clip's time-averaged pose vector
     -> embedding_cosine_dist (a coarse, alignment-free secondary signal).
  5. Calibrate a same-session decision threshold on the 24 already-hand-judged
     rows (22 same_session, 2 false_positive), then apply it to every
     previously-unjudged row. Pairs that fall in the ambiguous band between the
     confidently-same and confidently-different clusters are left for manual
     review rather than guessed.

Output: near_duplicate_review_list_dtw.csv (adds dtw_distance,
embedding_cosine_dist, auto_judgment, auto_confidence columns; leaves the
original judgment column untouched for rows that were already hand-reviewed).
"""

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
TENSOR_DIRS = [ROOT / "data" / "Golf" / "tensor" / "expert", ROOT / "data" / "Golf" / "tensor" / "user"]

LEFT_HIP, RIGHT_HIP = 2, 3
LEFT_SHOULDER, RIGHT_SHOULDER = 4, 5


def find_tensor(sample_id: str) -> Path | None:
    # sample_id like "some-expert_001" -> player = "some-expert"
    player = sample_id.rsplit("_", 1)[0]
    for base in TENSOR_DIRS:
        p = base / player / f"{sample_id}.npz"
        if p.exists():
            return p
    return None


def load_normalized_pose(path: Path) -> np.ndarray:
    d = np.load(path, allow_pickle=True)
    pose = d["pose"][:, :, :2].astype(np.float64)  # (T, 18, 2), drop score channel
    mid_hip = (pose[:, LEFT_HIP, :] + pose[:, RIGHT_HIP, :]) / 2.0
    shoulder_width = np.linalg.norm(pose[:, LEFT_SHOULDER, :] - pose[:, RIGHT_SHOULDER, :], axis=1)
    shoulder_width = np.clip(shoulder_width, 1e-4, None)
    centered = pose - mid_hip[:, None, :]
    normalized = centered / shoulder_width[:, None, None]
    return normalized.reshape(normalized.shape[0], -1)  # (T, 36)


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Standard DTW with Euclidean per-frame cost, path-length-normalized."""
    n, m = len(a), len(b)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    diffs = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)  # (n, m)
    for i in range(1, n + 1):
        row_diff = diffs[i - 1]
        for j in range(1, m + 1):
            cost[i, j] = row_diff[j - 1] + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    # count path length via backtrack for normalization
    i, j = n, m
    path_len = 0
    while i > 0 or j > 0:
        path_len += 1
        choices = []
        if i > 0 and j > 0:
            choices.append((cost[i - 1, j - 1], i - 1, j - 1))
        if i > 0:
            choices.append((cost[i - 1, j], i - 1, j))
        if j > 0:
            choices.append((cost[i, j - 1], i, j - 1))
        _, i, j = min(choices, key=lambda c: c[0])
    return float(cost[n, m] / max(path_len, 1))


def embedding_cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    va, vb = a.mean(axis=0), b.mean(axis=0)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return float(1.0 - np.dot(va, vb) / (na * nb))


def main() -> None:
    src = ROOT / "near_duplicate_review_list.csv"
    dst = ROOT / "near_duplicate_review_list_dtw.csv"

    rows = []
    with src.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames + ["dtw_distance", "embedding_cosine_dist", "auto_judgment", "auto_confidence"]
        for row in reader:
            row["auto_judgment"] = ""
            row["auto_confidence"] = ""
            rows.append(row)

    pose_cache: dict[str, np.ndarray] = {}
    missing = 0

    def get_pose(sample_id: str) -> np.ndarray | None:
        if sample_id in pose_cache:
            return pose_cache[sample_id]
        path = find_tensor(sample_id)
        if path is None:
            return None
        arr = load_normalized_pose(path)
        pose_cache[sample_id] = arr
        return arr

    calib_dtw = {"same_session": [], "false_positive": []}
    calib_emb = {"same_session": [], "false_positive": []}
    for idx, row in enumerate(rows):
        a_id, b_id = row["sample_a"], row["sample_b"]
        pa, pb = get_pose(a_id), get_pose(b_id)
        if pa is None or pb is None:
            row["dtw_distance"] = ""
            row["embedding_cosine_dist"] = ""
            row["auto_judgment"] = "tensor_not_found"
            row["auto_confidence"] = ""
            missing += 1
            continue
        dtw_d = dtw_distance(pa, pb)
        emb_d = embedding_cosine_dist(pa, pb)
        row["dtw_distance"] = f"{dtw_d:.6f}"
        row["embedding_cosine_dist"] = f"{emb_d:.6f}"
        existing = (row.get("judgment") or "").strip().lower()
        if existing in ("same_session", "false_positive"):
            calib_dtw[existing].append(dtw_d)
            calib_emb[existing].append(emb_d)
        print(f"[{idx + 1}/{len(rows)}] {a_id} vs {b_id}: dtw={dtw_d:.4f} emb_cos={emb_d:.4f}", file=sys.stderr)

    # Checkpoint: write raw distances now so an error in classification below
    # never loses the (expensive) DTW computation.
    raw_dst = ROOT / "near_duplicate_review_list_dtw_raw.csv"
    with raw_dst.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCheckpoint written: {raw_dst}", file=sys.stderr)

    same = np.array(calib_dtw["same_session"])
    diff = np.array(calib_dtw["false_positive"])
    same_e = np.array(calib_emb["same_session"])
    diff_e = np.array(calib_emb["false_positive"])
    print(f"\nCalibration set (dtw_distance): same_session n={len(same)} range=[{same.min():.4f},{same.max():.4f}] "
          f"mean={same.mean():.4f} | false_positive n={len(diff)} values={list(diff)}", file=sys.stderr)
    print(f"Calibration set (embedding_cosine_dist): same_session n={len(same_e)} range=[{same_e.min():.4f},{same_e.max():.4f}] "
          f"mean={same_e.mean():.4f} | false_positive n={len(diff_e)} values={list(diff_e)}", file=sys.stderr)

    # Decision bands from the labeled calibration subset: confidently-same is at
    # or below the same_session max observed; confidently-different is at or
    # above the false_positive min observed. Anything in between is ambiguous.
    same_hi = float(same.max()) if len(same) else 0.0
    diff_lo = float(diff.min()) if len(diff) else float("inf")
    if diff_lo <= same_hi:
        # calibration bands overlap -- no safe automatic separation exists;
        # everything outside a hand-judged row must go to manual review.
        same_hi_safe, diff_lo_safe = -1.0, float("inf")
    else:
        same_hi_safe, diff_lo_safe = same_hi, diff_lo

    n_auto_same = n_auto_diff = n_ambiguous = 0
    for row in rows:
        if row["auto_judgment"] == "tensor_not_found":
            continue
        existing = (row.get("judgment") or "").strip().lower()
        if existing in ("same_session", "false_positive"):
            row["auto_judgment"] = f"prior_manual:{existing}"
            row["auto_confidence"] = "manual"
            continue
        d = float(row["dtw_distance"])
        if d <= same_hi_safe:
            row["auto_judgment"] = "same_session"
            row["auto_confidence"] = "high"
            n_auto_same += 1
        elif d >= diff_lo_safe:
            row["auto_judgment"] = "false_positive"
            row["auto_confidence"] = "high"
            n_auto_diff += 1
        else:
            row["auto_judgment"] = "needs_manual_review"
            row["auto_confidence"] = "low"
            n_ambiguous += 1

    with dst.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"\nWrote {dst}\n"
        f"auto_same_session={n_auto_same}, auto_false_positive={n_auto_diff}, "
        f"needs_manual_review={n_ambiguous}, tensor_not_found={missing}, "
        f"decision_band=(<= {same_hi_safe:.4f} same, >= {diff_lo_safe:.4f} different)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
