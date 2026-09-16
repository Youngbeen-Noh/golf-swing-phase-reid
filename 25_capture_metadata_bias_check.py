"""Bias control check (2nd revision instruction 2.3): does capture-condition
metadata alone -- fps, resolution, clip duration -- predict expert identity
above chance, independent of genuine swing-style pose content?

Motivation: the model's input is normalized 2D pose coordinates only (never
raw pixels, so clothing/background/body-shape-in-pixels are architecturally
excluded already, Section 3.4). But fps and clip framing are NOT fully
erased by that normalization (temporal resampling assumes a given frame
rate, and per-expert clips may share a consistent capture setup by
coincidence of source), so a genuine risk remains: if fps/resolution/clip
length happen to be near-unique per expert in this dataset, the model could
in principle be exploiting that as an identity shortcut rather than swing
style, and no pose-level ablation would catch it.

This script builds the simplest possible classifier that CANNOT see pose
content at all -- 1-nearest-centroid on standardized [fps, width, height,
frame_count] computed from the training pool -- and evaluates it on the
validation queries under the identical protocol (25-expert closed set,
chance level 4%). If this metadata-only baseline scores far above chance,
capture-condition confounding is a real concern needing further
investigation. If it is near chance, capture metadata alone cannot explain
this paper's reported accuracy, which bounds (without proving zero) how much
of the model's real performance such confounds could plausibly explain.

Usage:
  python 25_capture_metadata_bias_check.py --train-json-dir data/Golf/jsons/expert \
      --query-json-dir data/Golf/jsons/user
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FEATURES = ["fps", "width", "height", "frame_count"]


def load_metadata(json_dir: Path) -> tuple[list[str], np.ndarray]:
    expert_ids: list[str] = []
    rows: list[list[float]] = []
    for path in sorted(json_dir.rglob("*.json")):
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        expert_ids.append(path.parent.name)
        rows.append([float(data[k]) for k in FEATURES])
    return expert_ids, np.array(rows, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-json-dir", default="data/Golf/jsons/expert")
    parser.add_argument("--query-json-dir", default="data/Golf/jsons/user")
    args = parser.parse_args()

    train_experts, train_feats = load_metadata(Path(args.train_json_dir))
    query_experts, query_feats = load_metadata(Path(args.query_json_dir))
    print(f"Train (gallery) samples: {len(train_experts)}, unique experts: {len(set(train_experts))}")
    print(f"Query (validation) samples: {len(query_experts)}")

    mean = train_feats.mean(axis=0)
    std = train_feats.std(axis=0)
    std[std == 0] = 1.0
    train_norm = (train_feats - mean) / std
    query_norm = (query_feats - mean) / std

    unique_experts = sorted(set(train_experts))
    centroids = np.array(
        [train_norm[[i for i, e in enumerate(train_experts) if e == expert]].mean(axis=0) for expert in unique_experts]
    )

    correct = 0
    ranks = []
    rows = []
    for true_expert, feat in zip(query_experts, query_norm):
        dists = np.linalg.norm(centroids - feat[None, :], axis=1)
        order = np.argsort(dists)
        ranked_experts = [unique_experts[i] for i in order]
        final_rank = ranked_experts.index(true_expert) + 1 if true_expert in ranked_experts else len(ranked_experts) + 1
        ranks.append(final_rank)
        if final_rank == 1:
            correct += 1
        rows.append({"expert_id": true_expert, "final_rank": final_rank, "top1": ranked_experts[0]})

    n = len(query_experts)
    top1 = correct / n
    mrr = float(np.mean([1.0 / r for r in ranks]))
    chance = 1.0 / len(unique_experts)
    print(f"\n=== Metadata-only (fps/width/height/frame_count) nearest-centroid baseline ===")
    print(f"n_queries={n}, n_experts={len(unique_experts)}, chance_level={chance:.4f}")
    print(f"Top-1 accuracy: {top1:.4f}  (vs. chance {chance:.4f})")
    print(f"MRR: {mrr:.4f}")
    print(f"Mean rank: {np.mean(ranks):.4f}")

    # Per-feature spread check: is any single feature suspiciously
    # near-deterministic per expert (e.g. every clip of one expert sharing
    # the exact same fps found nowhere else)?
    print("\n=== Per-feature diagnostic: fraction of experts with near-zero within-expert variance ===")
    for i, feat_name in enumerate(FEATURES):
        near_constant = 0
        for expert in unique_experts:
            vals = train_feats[[j for j, e in enumerate(train_experts) if e == expert], i]
            if len(vals) > 1 and np.std(vals) < 1e-6:
                near_constant += 1
        print(f"  {feat_name}: {near_constant}/{len(unique_experts)} experts have zero within-expert variance "
              f"(all training clips share the exact same value)")


if __name__ == "__main__":
    main()
