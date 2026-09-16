"""Macro recall / per-class recall / confusion matrix for the holdout eval (6.1).

5.5's evaluation metrics list (Top-1/Top-3/Top-5/MRR/mean rank) doesn't include
macro recall, per-class recall, or a confusion matrix, though the instruction
doc asks for them. Each run's pro_holdout_user_metrics.json already carries
per-query ground truth (expert_id) and prediction (final_top1) across all
5 seeds, so both can be derived without any new evaluation run -- this just
aggregates what already exists in the experiment archive.

Usage:
  python 15_holdout_confusion_and_recall.py ^
      --archive-root "D:/AISL/experiments_archive/golf_experiment_2" ^
      --run-prefix ATCN_Golf-D --seeds 42 43 44 45 46
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def load_run(archive_root: Path, run_prefix: str, seed: int) -> dict:
    path = archive_root / f"{run_prefix}-s{seed}" / "results" / "evaluation" / "pro_holdout_user_metrics.json"
    if not path.exists():
        raise SystemExit(f"Missing metrics json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archive-root", required=True, help="e.g. D:/AISL/experiments_archive/golf_experiment_2")
    parser.add_argument("--run-prefix", default="ATCN_Golf-D", help="Run id without the -sSEED suffix.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 45, 44, 46])
    parser.add_argument("--out-prefix", default="results_holdout")
    args = parser.parse_args()

    root = Path(args.archive_root)
    experts: list[str] | None = None
    per_seed_recall: dict[str, list[float]] = defaultdict(list)
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for seed in args.seeds:
        data = load_run(root, args.run_prefix, seed)
        if experts is None:
            experts = sorted(data["summary"]["expert_ids"])
        correct = defaultdict(int)
        total = defaultdict(int)
        for row in data["rows"]:
            true_e, pred_e = row["expert_id"], row["final_top1"]
            total[true_e] += 1
            correct[true_e] += int(pred_e == true_e)
            confusion[true_e][pred_e] += 1
        for e in experts:
            if total[e] > 0:
                per_seed_recall[e].append(correct[e] / total[e])

    macro_per_seed = [
        mean(per_seed_recall[e][i] for e in experts if len(per_seed_recall[e]) > i)
        for i in range(len(args.seeds))
    ]
    print(f"macro recall per seed: {[round(v, 4) for v in macro_per_seed]}")
    print(f"macro recall mean+-sd: {mean(macro_per_seed):.4f} +- {stdev(macro_per_seed):.4f}")

    recall_rows = []
    for e in sorted(experts):
        vals = per_seed_recall[e]
        recall_rows.append({
            "expert_id": e,
            "recall_mean": round(mean(vals), 4),
            "recall_sd": round(stdev(vals), 4) if len(vals) > 1 else 0.0,
            "n_seeds": len(vals),
        })

    conf_path = Path(f"{args.out_prefix}_confusion_matrix.csv")
    with conf_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + experts)
        for te in experts:
            w.writerow([te] + [confusion[te].get(pe, 0) for pe in experts])
    print(f"Saved: {conf_path} (summed over {len(args.seeds)} seeds)")

    recall_path = Path(f"{args.out_prefix}_per_expert_recall.csv")
    with recall_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recall_rows[0].keys()))
        w.writeheader()
        w.writerows(recall_rows)
    print(f"Saved: {recall_path}")


if __name__ == "__main__":
    main()
