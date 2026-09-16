from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

"""Split the existing known-pro holdout pool into a validation subset (used to
select the best model x preprocessing x training-option condition) and a
separately locked test subset (touched exactly once, to report the final
number). This exists because the prior single-holdout design used the same
clips both to pick the winning condition and to report its accuracy, which is
optimistic (test-set-tuning bias) -- see revision instruction item 1.

Session-aware: reuses the same reviewed near-duplicate CSV as
01-3b_group_aware_holdout_split.py so that clips from the same recording
session never split across validation/test (only across train/holdout,
which 01-3b already guarantees). A session group is therefore atomic across
all three pools now.

Usage (run once, after 01-3b has produced the current train/holdout split):
    python 01-3c_validation_locked_test_split.py --session-review-csv near_duplicate_review_list.csv --dry-run
    python 01-3c_validation_locked_test_split.py --session-review-csv near_duplicate_review_list.csv
"""


def load_session_groups(csv_path: Path) -> dict[str, list[set[str]]]:
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    members_by_player: dict[str, set[str]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            player = row["player"]
            a, b = row["sample_a"], row["sample_b"]
            judgment = (row.get("judgment") or row.get("same_session") or "").strip().lower()
            members_by_player.setdefault(player, set()).update({a, b})
            if judgment in {"same", "same_session", "yes", "y", "1", "true"}:
                union(a, b)

    groups_by_player: dict[str, list[set[str]]] = {}
    for player, members in members_by_player.items():
        clusters: dict[str, set[str]] = {}
        for m in members:
            clusters.setdefault(find(m), set()).add(m)
        groups_by_player[player] = [g for g in clusters.values() if len(g) > 1]
    return groups_by_player


def collect_jsons(json_dir: Path, expert_ids: set[str] | None) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    if not json_dir.exists():
        return groups
    for expert_dir in sorted(p for p in json_dir.iterdir() if p.is_dir()):
        expert_id = expert_dir.name
        if expert_ids is not None and expert_id not in expert_ids:
            continue
        json_paths = sorted(expert_dir.glob("*.json"))
        if json_paths:
            groups[expert_id] = json_paths
    return groups


def build_units(json_paths: list[Path], session_groups: list[set[str]]) -> list[list[Path]]:
    by_stem = {p.stem: p for p in json_paths}
    grouped_stems: set[str] = set()
    units: list[list[Path]] = []
    for group in session_groups:
        paths = [by_stem[stem] for stem in group if stem in by_stem]
        if len(paths) > 1:
            units.append(sorted(paths, key=lambda p: p.name))
            grouped_stems.update(p.stem for p in paths)
    for p in json_paths:
        if p.stem not in grouped_stems:
            units.append([p])
    return units


def plan_test_units(
    units: list[list[Path]],
    target_count: int,
    seed: int,
    expert_id: str,
    restarts: int = 200,
) -> list[list[Path]]:
    """Pick units (whole session groups stay together) totalling as close to
    target_count as possible, leaving at least one unit for validation."""
    total = sum(len(u) for u in units)
    max_test = total - 1 if len(units) > 1 else 0
    if target_count <= 0 or max_test <= 0:
        return []

    best: list[list[Path]] | None = None
    best_gap = None
    for attempt in range(restarts):
        rng = random.Random(f"{seed}:{expert_id}:test:{attempt}")
        order = units[:]
        rng.shuffle(order)
        chosen: list[list[Path]] = []
        count = 0
        # never take every unit -- always leave >=1 unit for validation
        for unit in order[:-1]:
            if count + len(unit) <= min(target_count, max_test):
                chosen.append(unit)
                count += len(unit)
        gap = abs(count - target_count)
        if best_gap is None or gap < best_gap:
            best, best_gap = chosen, gap
            if gap == 0:
                break
    return best or []


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def move_json(source_path: Path, dest_path: Path, subject_type: str, overwrite: bool) -> None:
    if dest_path.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists. Use --overwrite to replace it: {dest_path}")
    data = load_json(source_path)
    data["subject_type"] = subject_type
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with dest_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    source_path.unlink()


def parse_expert_ids(value: str | None) -> set[str] | None:
    if value is None:
        return None
    ids = {item.strip() for item in value.split(",") if item.strip()}
    return ids or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split the current known-pro holdout pool (data/Golf/jsons/user) into a "
            "validation subset (condition selection) and a locked test subset "
            "(single confirmatory report), session-group-aware."
        )
    )
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--session-review-csv", required=True, help="Reviewed near-duplicate CSV with a judgment column.")
    parser.add_argument("--test-ratio", type=float, default=0.5, help="Fraction of the current holdout to lock away as test. Default: 0.5.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expert-ids", default=None)
    parser.add_argument("--holdout-json-dir", default=None, help="Default: data/Golf/jsons/user (the current single holdout).")
    parser.add_argument("--test-json-dir", default=None, help="Default: data/Golf/jsons/user_locked_test.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.test_ratio < 1.0:
        raise ValueError("--test-ratio must be in the range (0, 1).")

    root = Path(args.root)
    holdout_dir = Path(args.holdout_json_dir) if args.holdout_json_dir else root / "data" / "Golf" / "jsons" / "user"
    test_dir = Path(args.test_json_dir) if args.test_json_dir else root / "data" / "Golf" / "jsons" / "user_locked_test"
    expert_ids = parse_expert_ids(args.expert_ids)

    if test_dir.exists() and any(test_dir.rglob("*.json")) and not args.overwrite:
        raise RuntimeError(
            f"{test_dir} already has JSON files. This script should run once. "
            "Pass --overwrite only if you intentionally want to re-split."
        )

    session_groups_by_player = load_session_groups(Path(args.session_review_csv))
    holdout_groups = collect_jsons(holdout_dir, expert_ids)
    if not holdout_groups:
        print(f"No holdout JSON files found: {holdout_dir}")
        return

    total_moved = 0
    total_kept_val = 0
    for expert_id, json_paths in sorted(holdout_groups.items()):
        session_groups = session_groups_by_player.get(expert_id, [])
        units = build_units(json_paths, session_groups)
        target = max(1, round(len(json_paths) * args.test_ratio))
        test_units = plan_test_units(units, target, args.seed, expert_id)
        test_paths = {p for unit in test_units for p in unit}

        val_count = len(json_paths) - len(test_paths)
        total_kept_val += val_count
        total_moved += len(test_paths)

        group_note = ""
        multi = [u for u in test_units if len(u) > 1]
        if multi:
            group_note = "  [grouped: " + "; ".join("+".join(p.stem for p in u) for u in multi) + "]"
        print(
            f"[plan] {expert_id}: holdout_total={len(json_paths)}, "
            f"validation={val_count}, locked_test={len(test_paths)}{group_note}"
        )

        for p in sorted(test_paths, key=lambda x: x.name):
            dest = test_dir / expert_id / p.name
            print(f"  [-> locked_test] {p} -> {dest}")
            if not args.dry_run:
                move_json(p, dest, "user", args.overwrite)
                total_moved += 0  # already counted above

    action = "planned" if args.dry_run else "done"
    print(f"\n{action}. validation={total_kept_val}, locked_test={total_moved}")
    if args.dry_run:
        print(
            "\nNote: unreviewed near-duplicate pairs (judgment blank in the CSV) are NOT "
            "treated as same-session here -- they fall through as independent singleton "
            "units and could still split across validation/locked_test. Complete the "
            "near-duplicate review (revision item 2) before treating this split as final."
        )


if __name__ == "__main__":
    main()
