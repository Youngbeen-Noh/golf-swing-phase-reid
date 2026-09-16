from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def collect_expert_jsons(expert_json_dir: Path, expert_ids: set[str] | None) -> dict[str, list[Path]]:
    """Collect successfully extracted expert JSON files by expert id."""
    groups: dict[str, list[Path]] = {}
    if not expert_json_dir.exists():
        return groups

    for expert_dir in sorted(path for path in expert_json_dir.iterdir() if path.is_dir()):
        expert_id = expert_dir.name
        if expert_ids is not None and expert_id not in expert_ids:
            continue
        json_paths = sorted(expert_dir.glob("*.json"))
        if json_paths:
            groups[expert_id] = json_paths
    return groups


def collect_existing_user_jsons(user_json_dir: Path, expert_ids: set[str]) -> dict[str, list[Path]]:
    """Collect existing known-pro user JSONs that would indicate a prior split."""
    groups: dict[str, list[Path]] = {}
    for expert_id in sorted(expert_ids):
        expert_user_dir = user_json_dir / expert_id
        if not expert_user_dir.exists():
            continue
        json_paths = sorted(expert_user_dir.glob("*.json"))
        if json_paths:
            groups[expert_id] = json_paths
    return groups


def holdout_count(total: int, ratio: float) -> int:
    """Return the per-expert holdout count while leaving at least one train JSON."""
    if total <= 1 or ratio <= 0.0:
        return 0
    return min(total - 1, max(1, round(total * ratio)))


def build_split_plan(
    groups: dict[str, list[Path]],
    ratio: float,
    seed: int,
) -> dict[str, list[Path]]:
    """Build a deterministic JSON holdout split plan."""
    plan: dict[str, list[Path]] = {}
    for expert_id, json_paths in sorted(groups.items()):
        count = holdout_count(len(json_paths), ratio)
        if count == 0:
            plan[expert_id] = []
            continue
        rng = random.Random(f"{seed}:{expert_id}")
        selected = rng.sample(json_paths, count)
        plan[expert_id] = sorted(selected, key=lambda path: path.name)
    return plan


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def move_json_to_user(source_path: Path, dest_path: Path, overwrite: bool) -> None:
    """Move one expert JSON into the user folder and mark its subject type as user."""
    if dest_path.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists. Use --overwrite to replace it: {dest_path}")

    data = load_json(source_path)
    data["subject_type"] = "user"

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with dest_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    source_path.unlink()


def parse_expert_ids(value: str | None) -> set[str] | None:
    if value is None:
        return None
    expert_ids = {item.strip() for item in value.split(",") if item.strip()}
    return expert_ids or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split successfully extracted Golf expert JSON files into expert train JSONs "
            "and known-pro user holdout JSONs."
        )
    )
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--sport", choices=["Golf"], default="Golf")
    parser.add_argument("--user-ratio", type=float, default=0.2, help="Per-expert holdout ratio. Default: 0.2.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--expert-ids",
        default=None,
        help="Optional comma-separated expert ids to split. Default: all expert folders.",
    )
    parser.add_argument(
        "--expert-json-dir",
        default=None,
        help="Source JSON directory. Default: data/Golf/jsons/expert.",
    )
    parser.add_argument(
        "--user-json-dir",
        default=None,
        help="Destination JSON directory. Default: data/Golf/jsons/user.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite colliding user holdout JSON files.")
    parser.add_argument("--dry-run", action="store_true", help="Print the split plan without moving files.")
    parser.add_argument(
        "--allow-existing-user-holdout",
        action="store_true",
        help="Allow splitting even when user JSONs already exist for the same expert ids.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.user_ratio < 1.0:
        raise ValueError("--user-ratio must be in the range [0, 1).")

    root = Path(args.root)
    expert_json_dir = Path(args.expert_json_dir) if args.expert_json_dir else root / "data" / args.sport / "jsons" / "expert"
    user_json_dir = Path(args.user_json_dir) if args.user_json_dir else root / "data" / args.sport / "jsons" / "user"
    expert_ids = parse_expert_ids(args.expert_ids)

    groups = collect_expert_jsons(expert_json_dir, expert_ids)
    if not groups:
        print(f"No expert JSON files found: {expert_json_dir}")
        return

    existing_user_jsons = collect_existing_user_jsons(user_json_dir, set(groups))
    if existing_user_jsons and not args.allow_existing_user_holdout:
        existing_summary = ", ".join(f"{expert_id}={len(paths)}" for expert_id, paths in sorted(existing_user_jsons.items()))
        raise RuntimeError(
            "Known-pro user holdout JSONs already exist. "
            "This script should normally be run only once after extraction and before 02_json_to_tensor.py. "
            f"Existing holdouts: {existing_summary}. "
            "Regenerate/reset jsons first, or pass --allow-existing-user-holdout only if you intentionally want an additional split."
        )

    plan = build_split_plan(groups, args.user_ratio, args.seed)
    moved = 0
    kept = 0

    for expert_id, json_paths in sorted(groups.items()):
        selected = set(plan[expert_id])
        kept += len(json_paths) - len(selected)
        print(
            f"[plan] {expert_id}: total={len(json_paths)}, "
            f"expert_train={len(json_paths) - len(selected)}, user_holdout={len(selected)}"
        )

        for source_path in plan[expert_id]:
            dest_path = user_json_dir / expert_id / source_path.name
            print(f"  [holdout] {source_path} -> {dest_path}")
            if not args.dry_run:
                move_json_to_user(source_path, dest_path, args.overwrite)
                moved += 1

    action = "planned" if args.dry_run else "done"
    print(f"{action}. expert_train={kept}, user_holdout={sum(len(paths) for paths in plan.values())}, moved={moved}")


if __name__ == "__main__":
    main()
