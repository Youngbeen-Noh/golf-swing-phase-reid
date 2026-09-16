from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any


def load_session_groups(csv_path: Path) -> dict[str, list[set[str]]]:
    """Read a reviewed near-duplicate CSV and union same-session pairs per player.

    Expects columns: player, sample_a, sample_b, judgment (or same_session).
    Rows judged as same-session are merged via union-find; everything else
    (false positive / pending / blank) is ignored.
    """
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


def collect_expert_jsons(expert_json_dir: Path, expert_ids: set[str] | None) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    if not expert_json_dir.exists():
        return groups
    for expert_dir in sorted(p for p in expert_json_dir.iterdir() if p.is_dir()):
        expert_id = expert_dir.name
        if expert_ids is not None and expert_id not in expert_ids:
            continue
        json_paths = sorted(expert_dir.glob("*.json"))
        if json_paths:
            groups[expert_id] = json_paths
    return groups


def holdout_count(total: int, ratio: float) -> int:
    if total <= 1 or ratio <= 0.0:
        return 0
    return min(total - 1, max(1, round(total * ratio)))


def build_units(json_paths: list[Path], session_groups: list[set[str]]) -> list[list[Path]]:
    """Partition an expert's JSON files into split-atomic units: session groups
    stay together as one unit, everything else is its own singleton unit."""
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


def plan_holdout_units(
    units: list[list[Path]],
    target_count: int,
    seed: int,
    expert_id: str,
    restarts: int = 200,
) -> list[list[Path]]:
    """Pick a subset of units (whole session groups stay together) whose total
    file count is as close as possible to target_count, without exceeding
    total-1 (at least one file must remain for training)."""
    total = sum(len(u) for u in units)
    max_holdout = total - 1
    if target_count <= 0 or max_holdout <= 0:
        return []

    best: list[list[Path]] | None = None
    best_gap = None

    for attempt in range(restarts):
        rng = random.Random(f"{seed}:{expert_id}:{attempt}")
        order = units[:]
        rng.shuffle(order)
        chosen: list[list[Path]] = []
        count = 0
        for unit in order:
            if count + len(unit) <= min(target_count, max_holdout):
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


def move_json_to_user(source_path: Path, dest_path: Path, overwrite: bool) -> None:
    if dest_path.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists. Use --overwrite to replace it: {dest_path}")
    data = load_json(source_path)
    data["subject_type"] = "user"
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with dest_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    source_path.unlink()


def move_json_to_expert(source_path: Path, dest_path: Path, overwrite: bool) -> None:
    if dest_path.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists. Use --overwrite to replace it: {dest_path}")
    data = load_json(source_path)
    data["subject_type"] = "expert"
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
            "Rebuild the Golf known-pro holdout split so confirmed same-session "
            "video groups (from a reviewed near-duplicate CSV) always stay "
            "entirely on one side of the split (train or holdout)."
        )
    )
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--session-review-csv", required=True, help="Reviewed near-duplicate CSV with a judgment column.")
    parser.add_argument("--user-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expert-ids", default=None)
    parser.add_argument("--expert-json-dir", default=None)
    parser.add_argument("--user-json-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    expert_json_dir = Path(args.expert_json_dir) if args.expert_json_dir else root / "data" / "Golf" / "jsons" / "expert"
    user_json_dir = Path(args.user_json_dir) if args.user_json_dir else root / "data" / "Golf" / "jsons" / "user"
    expert_ids = parse_expert_ids(args.expert_ids)

    session_groups_by_player = load_session_groups(Path(args.session_review_csv))

    # Pull every JSON currently in either expert/ or user/ back into one pool per
    # player so regrouping can move files freely across the former split line.
    expert_groups = collect_expert_jsons(expert_json_dir, expert_ids)
    user_groups = collect_expert_jsons(user_json_dir, expert_ids)

    all_experts = sorted(set(expert_groups) | set(user_groups))
    for expert_id in all_experts:
        json_paths = expert_groups.get(expert_id, []) + user_groups.get(expert_id, [])
        json_paths.sort(key=lambda p: p.name)
        total = len(json_paths)
        target = holdout_count(total, args.user_ratio)

        session_groups = session_groups_by_player.get(expert_id, [])
        units = build_units(json_paths, session_groups)
        holdout_units = plan_holdout_units(units, target, args.seed, expert_id)
        holdout_paths = {p for unit in holdout_units for p in unit}

        holdout_stems = {p.stem for p in holdout_paths}
        group_note = ""
        multi_unit_in_holdout = [u for u in holdout_units if len(u) > 1]
        if multi_unit_in_holdout:
            group_note = "  [grouped: " + "; ".join("+".join(p.stem for p in u) for u in multi_unit_in_holdout) + "]"

        print(
            f"[plan] {expert_id}: total={total}, target_holdout={target}, "
            f"actual_holdout={len(holdout_paths)}, units={len(units)}{group_note}"
        )

        for path in json_paths:
            currently_holdout = path.parent.parent.name == "user" if path.parent.parent.exists() else False
            should_be_holdout = path.stem in holdout_stems

            if should_be_holdout and not currently_holdout:
                dest = user_json_dir / expert_id / path.name
                print(f"  [-> holdout] {path} -> {dest}")
                if not args.dry_run:
                    move_json_to_user(path, dest, args.overwrite)
            elif not should_be_holdout and currently_holdout:
                dest = expert_json_dir / expert_id / path.name
                print(f"  [-> train]   {path} -> {dest}")
                if not args.dry_run:
                    move_json_to_expert(path, dest, args.overwrite)

    action = "planned" if args.dry_run else "done"
    print(f"\n{action}.")


if __name__ == "__main__":
    main()
