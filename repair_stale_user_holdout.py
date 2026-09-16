"""Repair stale known-pro holdout results after the validation/locked-test
split was applied to tensor/jsons AFTER a sweep had already archived
results built against the old (91-query) tensor state.

Root cause: per-seed run_ids (e.g. ATCN_Golf-D-s42) don't rebuild their own
features/user -- they share the BASE run's features/user
(ATCN_Golf-D/features/user) via --feature-dir (see 09_run_golf_experiments.py
build_commands()). If that base features/user was built while tensor/user
still had all 91 clips, every seed sharing it inherited 91-query results
-- and simply overwriting won't remove the 52 now-stale per-seed result
files that no longer have a matching feature (06/07 only ever add/overwrite
by filename, never delete orphans).

This script, for each experiment archive root:
  1. Finds every "base" run_id (no -s<seed> suffix) and rebuilds its
     features/user from scratch (rm -rf + 03_preprocess_and_build_features.py
     --subject-type user), using whatever is CURRENTLY in data/Golf/tensor/user
     (must already be the corrected 39-clip validation set -- check first).
  2. Finds every seed/option run_id belonging to that base, deletes its
     results/user_similarity and results/evaluation directories outright
     (removing stale 91-query leftovers), and reruns 06_infer_user_similarity.py
     + 07_evaluate.py against the corrected base features -- reusing each
     run's already-trained checkpoint, no retraining.

Usage:
  python repair_stale_user_holdout.py ~/experiments_archive/golf_experiment_2 ~/experiments_archive/golf_experiment_1
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

from encoder_loader import default_checkpoint_path

RUN_ID_PATTERN = re.compile(
    r"^(?P<base>(?:ATCN|Transformer|STGCN)_Golf-[A-E])"
    r"(?P<option>-seqonly|-steponly)?-s(?P<seed>\d+)$"
)


def find_python() -> str:
    return sys.executable


def rebuild_base_user_features(root: Path, base_run_id: str) -> None:
    """--output-root makes 03 write to {output_root}/features/{subject_type}/...,
    so pointing it at the base run's own directory reproduces the same layout
    09_run_golf_experiments.py originally built (root/data/Golf/experiments/{base_run_id}
    at sweep time; here root IS already that base run's own directory).

    Restricts to the SAME expert roster the base run was originally built
    with (e.g. Experiment 1's 7-expert subset), inferred from the existing
    features/expert subdirectory names -- that side was never touched by the
    validation/locked-test split bug, so it is still a reliable source of
    truth. Without this, Experiment 1's user-side rebuild would silently pull
    in all 25 experts from the current tensor/user pool instead of its own
    7-expert closed set."""
    base_dir = root / base_run_id
    method = base_run_id.split("_", 1)[1]  # "Golf-D" from "ATCN_Golf-D"
    user_feature_dir = base_dir / "features" / "user"
    if user_feature_dir.exists():
        shutil.rmtree(user_feature_dir)

    expert_feature_dir = base_dir / "features" / "expert"
    expert_ids = sorted(p.name for p in expert_feature_dir.iterdir() if p.is_dir()) if expert_feature_dir.exists() else []
    if not expert_ids:
        print(f"[warn] no features/expert subdirs found for {base_run_id}; "
              f"rebuilding user features against the FULL current tensor/user pool "
              f"-- verify this base run is meant to use all experts (e.g. Experiment 2).",
              file=sys.stderr)

    cmd = [
        find_python(), "03_preprocess_and_build_features.py",
        "--sport", "Golf", "--preprocess-method", method,
        "--subject-type", "user", "--overwrite",
        "--run-id", base_run_id,
        "--output-root", str(base_dir),
    ]
    if expert_ids:
        cmd.extend(["--expert-ids", ",".join(expert_ids)])
    print(f"[rebuild-features] {base_run_id}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def refresh_run(root: Path, run_id: str, base_run_id: str) -> None:
    run_dir = root / run_id
    base_user_feature_dir = root / base_run_id / "features" / "user"
    base_expert_feature_dir = root / base_run_id / "features" / "expert"
    for stale in ["results/user_similarity", "results/evaluation"]:
        stale_path = run_dir / stale
        if stale_path.exists():
            shutil.rmtree(stale_path)

    checkpoint = default_checkpoint_path(run_dir / "models")
    if checkpoint is None:
        print(f"[warn] no checkpoint found for {run_id}, skipping", file=sys.stderr)
        return
    prototype_store = run_dir / "prototypes" / "expert_prototypes.pkl"
    if not prototype_store.exists():
        print(f"[warn] no prototype store for {run_id}, skipping", file=sys.stderr)
        return

    output_dir = run_dir / "results" / "user_similarity"
    infer_cmd = [
        find_python(), "06_infer_user_similarity.py",
        "--sport", "Golf", "--run-id", run_id,
        "--checkpoint", str(checkpoint),
        "--prototype-store", str(prototype_store),
        "--feature-dir", str(base_user_feature_dir),
        "--output-dir", str(output_dir),
    ]
    print(f"[infer] {run_id}")
    subprocess.run(infer_cmd, check=True)

    eval_cmd = [
        find_python(), "07_evaluate.py",
        "--sport", "Golf", "--run-id", run_id,
        "--checkpoint", str(checkpoint),
        "--prototype-store", str(prototype_store),
        "--feature-dir", str(base_expert_feature_dir),
        "--output", str(run_dir / "results" / "evaluation" / "expert_retrieval_metrics.json"),
        "--require-pro-holdout",
        "--pro-holdout-result-dir", str(output_dir),
        "--pro-holdout-output", str(run_dir / "results" / "evaluation" / "pro_holdout_user_metrics.json"),
    ]
    print(f"[evaluate] {run_id}")
    subprocess.run(eval_cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for root_str in args.roots:
        root = Path(root_str).expanduser()
        if not root.is_dir():
            print(f"[warn] not a directory: {root}", file=sys.stderr)
            continue

        run_ids = sorted(p.name for p in root.iterdir() if p.is_dir())
        by_base: dict[str, list[str]] = {}
        for run_id in run_ids:
            m = RUN_ID_PATTERN.match(run_id)
            if m:
                by_base.setdefault(m.group("base"), []).append(run_id)

        print(f"\n=== {root} : {len(by_base)} base conditions, {sum(len(v) for v in by_base.values())} seed/option runs ===")
        if args.dry_run:
            for base, members in sorted(by_base.items()):
                print(f"  {base}: {members}")
            continue

        for base_run_id, members in sorted(by_base.items()):
            rebuild_base_user_features(root, base_run_id)
            for run_id in members:
                refresh_run(root, run_id, base_run_id)

    print("\nDone. Now rerun: 10_aggregate_seed_metrics.py, 19_paired_significance_hierarchical.py, "
          "20_query_level_logistic_variance.py, 13_factorial_anova.py")


if __name__ == "__main__":
    main()
