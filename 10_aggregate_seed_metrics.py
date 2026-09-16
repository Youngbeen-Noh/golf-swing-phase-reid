"""Aggregate multi-seed run results into mean +- SD and bootstrap 95% CIs (P0-2).

Scans one or more experiment result roots (each containing run_id folders like
ATCN_Golf-A-s42, Transformer_Golf-C-steponly-s44, ...), reads each run's
results/evaluation/pro_holdout_user_metrics.json, and writes:

  1. <out-prefix>_seed_raw.csv       one row per (run_id, seed): the per-seed
                                     raw metrics the paper must publish alongside
                                     the aggregates (top1/top3/top5/mrr/mean_rank)
  2. <out-prefix>_aggregate.csv      one row per (experiment, model, preprocess,
                                     training option): n_seeds, mean, SD, and
                                     bootstrap 95% CI for every metric

Usage (after run_both_experiments.sh finishes):

  python 10_aggregate_seed_metrics.py \
      ~/experiments_archive/golf_experiment_2 \
      ~/experiments_archive/golf_experiment_1

Notes:
  - Top-1/MRR/mean_rank come from summary.final_ranking; Top-3/Top-5 are
    recomputed from the per-query rows (final_rank <= 3 / <= 5).
  - The bootstrap CI resamples the per-seed metric values with replacement
    (percentile method, deterministic via --bootstrap-seed). With few seeds this
    is a coarse interval; the per-query rows in seed_raw's source jsons remain
    available for paired tests (McNemar / paired bootstrap) between conditions.
  - Legacy run_ids without a -s<seed> suffix are skipped with a warning so old
    single-seed folders don't silently mix into the aggregates.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import zlib
from pathlib import Path
from typing import Any

import numpy as np

RUN_ID_PATTERN = re.compile(
    r"^(?P<model>ATCN|Transformer|STGCN)_(?P<method>Golf-[A-E])"
    # Generic run-id-suffix segment (e.g. -seq140, -CE, -triplet) between the
    # preprocessing method and the seq/step option: matches any run of
    # -<tag> components a new ablation/baseline convention introduces, so a
    # new naming scheme (this one caught -seq140/-seq160, -CE, and -triplet
    # each on a *separate* occasion, silently dropping every run of that
    # kind until each was hand-added here) doesn't need another one-off fix.
    r"(?P<extra>(?:-[a-zA-Z0-9]+)*?)"
    r"(?P<option>-seqonly|-steponly|-seqphase)?-s(?P<seed>\d+)$"
)

OPTION_LABELS = {
    None: "seq+step",
    "-seqonly": "seq-only",
    "-steponly": "step-only",
    "-seqphase": "seq+step-fbmatched",
}

METRICS = ["top1_accuracy", "top3_accuracy", "top5_accuracy", "mrr", "mean_rank"]


def load_run_metrics(run_dir: Path, metrics_filename: str) -> dict[str, float] | None:
    metrics_path = run_dir / "results" / "evaluation" / metrics_filename
    if not metrics_path.exists():
        return None
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    final = data.get("summary", {}).get("final_ranking", {})
    if not final.get("count"):
        return None

    ranks = [
        int(row["final_rank"])
        for row in data.get("rows", [])
        if isinstance(row.get("final_rank"), (int, float)) and row.get("final_rank") != ""
    ]
    count = len(ranks) if ranks else int(final["count"])
    top3 = sum(1 for r in ranks if r <= 3) / count if ranks else float("nan")
    top5 = sum(1 for r in ranks if r <= 5) / count if ranks else float("nan")

    return {
        "count": float(count),
        "top1_accuracy": float(final["top1_accuracy"]),
        "top3_accuracy": top3,
        "top5_accuracy": top5,
        "mrr": float(final["mrr"]),
        "mean_rank": float(final["mean_rank"]),
    }


def collect_rows(roots: list[Path], metrics_filename: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in roots:
        if not root.is_dir():
            print(f"[warn] Not a directory, skipping: {root}", file=sys.stderr)
            continue
        experiment = root.name
        for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            match = RUN_ID_PATTERN.match(run_dir.name)
            if match is None:
                print(f"[warn] run_id without seed suffix, skipping: {run_dir.name}", file=sys.stderr)
                continue
            metrics = load_run_metrics(run_dir, metrics_filename)
            if metrics is None:
                print(f"[warn] No usable metrics json, skipping: {run_dir.name}", file=sys.stderr)
                continue
            # Fold any extra run-id-suffix tag (e.g. -seq140, -CE, -triplet) into the
            # grouping key so a differently-constructed run never gets pooled with the
            # plain Golf-{A-E} condition it's meant to be compared against.
            method = f"{match.group('method')}{match.group('extra')}"
            rows.append(
                {
                    "experiment": experiment,
                    "model": match.group("model"),
                    "method": method,
                    "option": OPTION_LABELS[match.group("option")],
                    "seed": int(match.group("seed")),
                    "run_id": run_dir.name,
                    **metrics,
                }
            )
    return rows


def bootstrap_ci(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def aggregate(rows: list[dict[str, Any]], n_boot: int, bootstrap_seed: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["experiment"], row["model"], row["method"], row["option"])
        groups.setdefault(key, []).append(row)

    aggregated: list[dict[str, Any]] = []
    for key in sorted(groups):
        experiment, model, method, option = key
        group = groups[key]
        seeds = sorted(row["seed"] for row in group)
        if len(seeds) != len(set(seeds)):
            print(f"[warn] Duplicate seeds in {key}: {seeds}", file=sys.stderr)
        out: dict[str, Any] = {
            "experiment": experiment,
            "model": model,
            "method": method,
            "option": option,
            "n_seeds": len(group),
            "seeds": " ".join(str(s) for s in seeds),
            "query_count": int(group[0]["count"]),
        }
        for metric in METRICS:
            values = np.array([row[metric] for row in group], dtype=np.float64)
            # A dedicated RNG per (group, metric) keeps every CI reproducible
            # regardless of how many groups/metrics were scanned before it.
            # crc32 (not hash()) so the seed is stable across interpreter runs.
            group_key = f"{bootstrap_seed}|{experiment}|{model}|{method}|{option}|{metric}"
            rng = np.random.default_rng(zlib.crc32(group_key.encode("utf-8")))
            lo, hi = bootstrap_ci(values, n_boot, rng)
            out[f"{metric}_mean"] = float(np.mean(values))
            out[f"{metric}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
            out[f"{metric}_ci95_lo"] = lo
            out[f"{metric}_ci95_hi"] = hi
        aggregated.append(out)
    return aggregated


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        print(f"[warn] Nothing to write for {path}", file=sys.stderr)
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}  ({len(rows)} rows)")


def print_leaderboard(aggregated: list[dict[str, Any]], top_n: int) -> None:
    by_experiment: dict[str, list[dict[str, Any]]] = {}
    for row in aggregated:
        by_experiment.setdefault(row["experiment"], []).append(row)
    for experiment, rows in sorted(by_experiment.items()):
        rows = sorted(rows, key=lambda r: r["top1_accuracy_mean"], reverse=True)
        print(f"\n== {experiment}: top {min(top_n, len(rows))} by mean Top-1 ==")
        header = f"{'model':<12}{'method':<9}{'option':<11}{'n':>2}  {'Top-1 mean+-SD':<20}{'95% CI':<18}{'MRR':>7}{'meanRk':>8}"
        print(header)
        for row in rows[:top_n]:
            ci = f"[{row['top1_accuracy_ci95_lo']:.3f},{row['top1_accuracy_ci95_hi']:.3f}]"
            print(
                f"{row['model']:<12}{row['method']:<9}{row['option']:<11}{row['n_seeds']:>2}  "
                f"{row['top1_accuracy_mean']:.3f} +- {row['top1_accuracy_sd']:.3f}     "
                f"{ci:<18}{row['mrr_mean']:>7.3f}{row['mean_rank_mean']:>8.3f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "roots",
        nargs="+",
        help="Experiment result roots containing run_id folders "
        "(e.g. ~/experiments_archive/golf_experiment_2).",
    )
    parser.add_argument("--out-prefix", default="results_seed", help="Prefix for output CSVs.")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=123)
    parser.add_argument("--top-n", type=int, default=10, help="Leaderboard rows to print per experiment.")
    parser.add_argument(
        "--metrics-filename",
        default="pro_holdout_user_metrics.json",
        help="Filename under each run's results/evaluation/ to read "
        "(e.g. pro_holdout_user_metrics_LOCKED_TEST.json for the locked-test confirmatory pass).",
    )
    args = parser.parse_args()

    roots = [Path(r).expanduser() for r in args.roots]
    rows = collect_rows(roots, args.metrics_filename)
    if not rows:
        print("No runs with seed suffix found. Did the multi-seed script finish?", file=sys.stderr)
        raise SystemExit(1)

    expected = {}
    for row in rows:
        expected.setdefault((row["experiment"], row["model"], row["method"], row["option"]), 0)
        expected[(row["experiment"], row["model"], row["method"], row["option"])] += 1
    n_seeds_seen = {v for v in expected.values()}
    if len(n_seeds_seen) > 1:
        print(f"[warn] Uneven seed counts across conditions: {sorted(n_seeds_seen)} "
              "(some runs missing or still in progress?)", file=sys.stderr)

    aggregated = aggregate(rows, args.n_boot, args.bootstrap_seed)
    rows_sorted = sorted(rows, key=lambda r: (r["experiment"], r["model"], r["method"], r["option"], r["seed"]))
    write_csv(Path(f"{args.out_prefix}_raw.csv"), rows_sorted)
    write_csv(Path(f"{args.out_prefix}_aggregate.csv"), aggregated)
    print_leaderboard(aggregated, args.top_n)


if __name__ == "__main__":
    main()
