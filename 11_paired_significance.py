"""Paired significance tests between retrieval conditions (P0-2).

Complements 10_aggregate_seed_metrics.py. Where step 10 reports each condition's
mean +- SD / CI in isolation, this script asks whether two conditions differ
*on the same held-out queries*, which is what P0-2 requires before any
"A beats B" claim.

For each condition (model, preprocess, training option) it loads the per-query
final ranks from every seed's pro_holdout_user_metrics.json and builds, per test
query (keyed by expert_id + sample_id), two seed-averaged statistics:
  - correct rate  = mean over seeds of 1[final_rank == 1]
  - reciprocal rank = mean over seeds of 1 / final_rank
Learning-free baselines (baseline_retrieval_metrics.json: cosine_raw_pose,
dtw_nearest_neighbor) are treated as a single "seed".

Per comparison it reports:
  - McNemar exact test on binarized per-query correctness (seed-avg correct
    rate >= 0.5), using the exact binomial p on discordant pairs
  - paired bootstrap on the mean per-query difference in reciprocal rank
    (percentile 95% CI + two-sided p), deterministic via --bootstrap-seed
  - Holm-Bonferroni adjusted p across all comparisons in the run

Default comparison set per experiment (override with --compare):
  best condition (by seed-mean Top-1)  vs  each of:
    its seq-only counterpart, its step-only/seq+step counterpart,
    the best condition of every other model, and the DTW + cosine baselines.

Usage:
  python 11_paired_significance.py \
      ~/experiments_archive/golf_experiment_2 \
      ~/experiments_archive/golf_experiment_1

  # explicit pairs (run_id stems without the -sSEED suffix; 'baseline:dtw' /
  # 'baseline:cosine' refer to baseline_retrieval_metrics.json):
  python 11_paired_significance.py ~/experiments_archive/golf_experiment_2 \
      --compare "ATCN_Golf-D_seq+step vs ATCN_Golf-D_step-only" \
      --compare "ATCN_Golf-D_seq+step vs baseline:dtw"
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import zlib
from itertools import combinations
from math import comb
from pathlib import Path
from typing import Any

import numpy as np

RUN_ID_PATTERN = re.compile(
    r"^(?P<model>ATCN|Transformer|STGCN)_(?P<method>Golf-[A-E])"
    r"(?P<option>-seqonly|-steponly)?-s(?P<seed>\d+)$"
)
OPTION_LABELS = {None: "seq+step", "-seqonly": "seq-only", "-steponly": "step-only"}
OPTION_TO_SUFFIX = {"seq+step": "", "seq-only": "-seqonly", "step-only": "-steponly"}


def condition_key(model: str, method: str, option: str) -> str:
    """Stable, seed-free label, e.g. ATCN_Golf-D_seq+step."""
    return f"{model}_{method}_{option}"


def load_condition_perquery(root: Path) -> dict[str, dict[tuple[str, str], dict[str, float]]]:
    """Return {condition_key: {(expert_id, sample_id): {correct, rr, n_seeds}}}."""
    per_seed: dict[str, dict[tuple[str, str], list[int]]] = {}
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        m = RUN_ID_PATTERN.match(run_dir.name)
        if m is None:
            continue
        key = condition_key(m.group("model"), m.group("method"), OPTION_LABELS[m.group("option")])
        metrics_path = run_dir / "results" / "evaluation" / "pro_holdout_user_metrics.json"
        if not metrics_path.exists():
            print(f"[warn] missing metrics json: {run_dir.name}", file=sys.stderr)
            continue
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        bucket = per_seed.setdefault(key, {})
        for row in data.get("rows", []):
            fr = row.get("final_rank", "")
            if fr == "" or fr is None:
                continue
            qkey = (str(row.get("expert_id", "")), str(row.get("sample_id", "")))
            bucket.setdefault(qkey, []).append(int(fr))

    out: dict[str, dict[tuple[str, str], dict[str, float]]] = {}
    for key, queries in per_seed.items():
        out[key] = {
            qkey: {
                "correct": float(np.mean([1.0 if r == 1 else 0.0 for r in ranks])),
                "rr": float(np.mean([1.0 / r for r in ranks])),
                "n_seeds": float(len(ranks)),
            }
            for qkey, ranks in queries.items()
        }
    return out


def load_baseline_perquery(root: Path) -> dict[str, dict[tuple[str, str], dict[str, float]]]:
    """Baselines live in any run's baseline_retrieval_metrics.json (single seed)."""
    out: dict[str, dict[tuple[str, str], dict[str, float]]] = {}
    candidates = sorted(root.glob("*/results/evaluation/baseline_retrieval_metrics.json"))
    if not candidates:
        return out
    data = json.loads(candidates[0].read_text(encoding="utf-8"))
    name_map = {"cosine_raw_pose": "baseline:cosine", "dtw_nearest_neighbor": "baseline:dtw"}
    for json_name, label in name_map.items():
        block = data.get("results", {}).get(json_name)
        if not block:
            continue
        q: dict[tuple[str, str], dict[str, float]] = {}
        for row in block.get("rows", []):
            rank = row.get("rank")
            if rank is None:
                continue
            qkey = (str(row.get("expert_id", "")), str(row.get("sample_id", "")))
            q[qkey] = {"correct": 1.0 if int(rank) == 1 else 0.0, "rr": 1.0 / int(rank), "n_seeds": 1.0}
        if q:
            out[label] = q
    return out


def seed_mean_top1(perq: dict[tuple[str, str], dict[str, float]]) -> float:
    if not perq:
        return float("nan")
    return float(np.mean([v["correct"] for v in perq.values()]))


def mcnemar_exact(a_correct: np.ndarray, b_correct: np.ndarray) -> tuple[int, int, float]:
    """Exact (binomial) McNemar on binary correctness. Returns (b01, b10, p)."""
    a = a_correct >= 0.5
    b = b_correct >= 0.5
    b01 = int(np.sum(~a & b))  # A wrong, B right
    b10 = int(np.sum(a & ~b))  # A right, B wrong
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    k = min(b01, b10)
    tail = sum(comb(n, i) for i in range(0, k + 1)) * (0.5**n)
    p = min(1.0, 2.0 * tail)
    return b01, b10, p


def paired_bootstrap(diff: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float, float, float]:
    """Mean paired diff, 95% CI, two-sided p (via CI-symmetric bootstrap)."""
    mean = float(np.mean(diff))
    if len(diff) < 2:
        return mean, float("nan"), float("nan"), float("nan")
    samples = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    lo, hi = float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))
    # two-sided bootstrap p: proportion of resamples on the other side of 0, x2
    prop = float(np.mean(samples <= 0)) if mean > 0 else float(np.mean(samples >= 0))
    p = min(1.0, 2.0 * prop)
    return mean, lo, hi, p


def align(a: dict[tuple[str, str], dict[str, float]], b: dict[tuple[str, str], dict[str, float]]):
    keys = sorted(set(a) & set(b))
    a_corr = np.array([a[k]["correct"] for k in keys])
    b_corr = np.array([b[k]["correct"] for k in keys])
    rr_diff = np.array([a[k]["rr"] - b[k]["rr"] for k in keys])
    return keys, a_corr, b_corr, rr_diff


def holm(pvals: list[float]) -> list[float]:
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj = [0.0] * len(pvals)
    running = 0.0
    m = len(pvals)
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def default_comparisons(conditions: dict[str, dict], baselines: dict[str, dict]) -> list[tuple[str, str]]:
    top1 = {k: seed_mean_top1(v) for k, v in conditions.items()}
    if not top1:
        return []
    best = max(top1, key=lambda k: top1[k])
    m = re.match(r"^(ATCN|Transformer|STGCN)_(Golf-[A-E])_(seq\+step|seq-only|step-only)$", best)
    pairs: list[tuple[str, str]] = []
    if m:
        model, method, option = m.group(1), m.group(2), m.group(3)
        for other_opt in ("seq+step", "seq-only", "step-only"):
            if other_opt != option:
                cand = condition_key(model, method, other_opt)
                if cand in conditions:
                    pairs.append((best, cand))
        for other_model in ("ATCN", "Transformer", "STGCN"):
            if other_model != model:
                same_model = {k: v for k, v in top1.items() if k.startswith(other_model + "_")}
                if same_model:
                    pairs.append((best, max(same_model, key=lambda k: same_model[k])))
    for b in ("baseline:dtw", "baseline:cosine"):
        if b in baselines:
            pairs.append((best, b))
    return pairs


def run_experiment(root: Path, explicit: list[tuple[str, str]], n_boot: int, boot_seed: int) -> list[dict[str, Any]]:
    conditions = load_condition_perquery(root)
    baselines = load_baseline_perquery(root)
    pool = {**conditions, **baselines}
    pairs = explicit or default_comparisons(conditions, baselines)

    results: list[dict[str, Any]] = []
    for a_key, b_key in pairs:
        if a_key not in pool or b_key not in pool:
            print(f"[warn] unknown condition in pair: {a_key} vs {b_key}", file=sys.stderr)
            continue
        keys, a_corr, b_corr, rr_diff = align(pool[a_key], pool[b_key])
        if not keys:
            print(f"[warn] no shared queries: {a_key} vs {b_key}", file=sys.stderr)
            continue
        b01, b10, mc_p = mcnemar_exact(a_corr, b_corr)
        rng = np.random.default_rng(zlib.crc32(f"{boot_seed}|{root.name}|{a_key}|{b_key}".encode()))
        mean_d, lo, hi, bp = paired_bootstrap(rr_diff, n_boot, rng)
        results.append(
            {
                "experiment": root.name, "A": a_key, "B": b_key, "n_queries": len(keys),
                "top1_A": round(float(np.mean(a_corr)), 4), "top1_B": round(float(np.mean(b_corr)), 4),
                "top1_diff": round(float(np.mean(a_corr) - np.mean(b_corr)), 4),
                "mcnemar_b01_Awrong": b01, "mcnemar_b10_Aright": b10, "mcnemar_p": round(mc_p, 5),
                "rr_diff_mean": round(mean_d, 4), "rr_diff_ci95_lo": round(lo, 4), "rr_diff_ci95_hi": round(hi, 4),
                "bootstrap_p": round(bp, 5),
            }
        )
    for r, adj in zip(results, holm([x["mcnemar_p"] for x in results] or [])):
        r["mcnemar_p_holm"] = round(adj, 5)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--compare", action="append", default=[],
                        help='Explicit "A vs B" pair; run_id stems or baseline:dtw / baseline:cosine. Applied to every root.')
    parser.add_argument("--out", default="results_paired_significance.csv")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=123)
    args = parser.parse_args()

    explicit: list[tuple[str, str]] = []
    for spec in args.compare:
        if " vs " not in spec:
            parser.error(f"--compare must look like 'A vs B', got: {spec}")
        a, b = spec.split(" vs ", 1)
        explicit.append((a.strip(), b.strip()))

    all_results: list[dict[str, Any]] = []
    for root_str in args.roots:
        root = Path(root_str).expanduser()
        if not root.is_dir():
            print(f"[warn] not a directory: {root}", file=sys.stderr)
            continue
        res = run_experiment(root, explicit, args.n_boot, args.bootstrap_seed)
        all_results.extend(res)
        print(f"\n== {root.name}: {len(res)} comparisons ==")
        for r in res:
            sig = "*" if r["mcnemar_p_holm"] < 0.05 else " "
            print(f" {sig} {r['A']:<26} vs {r['B']:<22} "
                  f"dTop1={r['top1_diff']:+.3f}  McNemar p={r['mcnemar_p']:.3f} "
                  f"(Holm {r['mcnemar_p_holm']:.3f})  dRR={r['rr_diff_mean']:+.3f} "
                  f"[{r['rr_diff_ci95_lo']:+.3f},{r['rr_diff_ci95_hi']:+.3f}]")

    if all_results:
        with open(args.out, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            w.writeheader()
            w.writerows(all_results)
        print(f"\nSaved: {args.out}  ({len(all_results)} comparisons)")
    else:
        print("No comparisons produced.", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
