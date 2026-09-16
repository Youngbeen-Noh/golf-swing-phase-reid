"""Paired significance tests with a hierarchical (query + seed) bootstrap
(2nd revision instruction 3.1/3.2).

11_paired_significance.py averaged each query's correctness across the 5
seeds BEFORE testing ("seed-mean per-query correctness >= 0.5"), then ran an
exact McNemar test on that binarized average -- the instruction singles this
out as a vague, invalid analysis unit: seeds are not independent replicates
of the query set (same 91 queries, same split, only training-stochasticity
differs), so collapsing them first understates uncertainty and the
"binarized average" isn't a real paired binary observation.

This script keeps every (query, seed) cell as its own paired observation and
estimates uncertainty with a two-level hierarchical bootstrap: resample
queries (with replacement, capturing "a different sample of test clips"),
and for each resampled query independently resample ONE seed (capturing
"a different training run") -- using the SAME resampled seed index for both
conditions being compared, since seed 42 trained on condition A and seed 42
trained on condition B share the same weight-init/data-shuffle draw and are
therefore legitimately paired, not just the queries.

Reports, per comparison:
  - top1_A, top1_B, top1_diff: pooled means/difference over all (query, seed)
    cells actually observed (point estimate; not bootstrap-resampled)
  - odds_ratio: pooled McNemar-style b01/b10 from all (query, seed) cells
  - hierarchical bootstrap 95% CI and two-sided p for both top1_diff and
    odds_ratio (this is the significance verdict; the naive pooled McNemar
    exact p is reported alongside only as an uncorrelated-observations
    reference, not as the verdict)
  - Holm-Bonferroni correction across the predefined comparison family

Usage:
  python 19_paired_significance_hierarchical.py ~/experiments_archive/golf_experiment_2 ~/experiments_archive/golf_experiment_1
  python 19_paired_significance_hierarchical.py ~/experiments_archive/golf_experiment_2 \
      --compare "ATCN_Golf-D_seq+step vs ATCN_Golf-D_step-only"
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import zlib
from math import comb
from pathlib import Path
from typing import Any

import numpy as np

RUN_ID_PATTERN = re.compile(
    r"^(?P<model>ATCN|Transformer|STGCN)_(?P<method>Golf-[A-E])"
    # Generic run-id-suffix segment (e.g. -seq140, -CE, -triplet) -- see the
    # matching comment in 10_aggregate_seed_metrics.py's RUN_ID_PATTERN.
    r"(?P<extra>(?:-[a-zA-Z0-9]+)*?)"
    r"(?P<option>-seqonly|-steponly|-seqphase)?-s(?P<seed>\d+)$"
)
OPTION_LABELS = {
    None: "seq+step",
    "-seqonly": "seq-only",
    "-steponly": "step-only",
    "-seqphase": "seq+step-fbmatched",
}


def condition_key(model: str, method: str, option: str) -> str:
    return f"{model}_{method}_{option}"


def load_condition_perquery_perseed(root: Path) -> dict[str, dict[tuple[str, str], dict[int, int]]]:
    """{condition_key: {(expert_id, sample_id): {seed: correct[0/1]}}} -- seed level preserved."""
    out: dict[str, dict[tuple[str, str], dict[int, int]]] = {}
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        m = RUN_ID_PATTERN.match(run_dir.name)
        if m is None:
            continue
        # Same fold as 10_aggregate_seed_metrics.py: a differently-constructed
        # run (e.g. frame-budget ablation, CE/triplet baseline) shares a
        # preprocessing method (e.g. Golf-D) with the legacy sweep, so fold
        # the extra tag into the method to avoid condition_key colliding with
        # the legacy Golf-D condition it's meant to be compared against.
        method = f"{m.group('method')}{m.group('extra')}"
        key = condition_key(m.group("model"), method, OPTION_LABELS[m.group("option")])
        seed = int(m.group("seed"))
        metrics_path = run_dir / "results" / "evaluation" / "pro_holdout_user_metrics.json"
        if not metrics_path.exists():
            print(f"[warn] missing metrics json: {run_dir.name}", file=sys.stderr)
            continue
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        bucket = out.setdefault(key, {})
        for row in data.get("rows", []):
            fr = row.get("final_rank", "")
            if fr == "" or fr is None:
                continue
            qkey = (str(row.get("expert_id", "")), str(row.get("sample_id", "")))
            bucket.setdefault(qkey, {})[seed] = 1 if int(fr) == 1 else 0
    return out


BASELINE_METHOD_PATTERN = re.compile(r"(Golf-[A-E])")


def _baseline_method_from_dir(run_dir_name: str) -> str:
    """Baseline run directories (e.g. "ATCN_Golf-D", written by
    baseline_retrieval.py / 22_evaluate_dmsm_baseline.py) have no "-sNN" seed
    suffix -- RUN_ID_PATTERN's required trailing seed group never matches
    them -- so the preprocessing condition is extracted directly instead."""
    m = BASELINE_METHOD_PATTERN.search(run_dir_name)
    return m.group(1) if m else run_dir_name


def load_baseline_perquery_perseed(root: Path) -> dict[str, dict[tuple[str, str], dict[int, int]]]:
    """Baselines have a single deterministic run per preprocessing condition
    -- modeled as seed=0 (n_seeds=1) -- and are keyed "baseline:{cosine,dtw,
    dmsm}:{method}" (e.g. "baseline:dtw:Golf-D") so the SPECIFIC condition
    being compared against can be selected explicitly.

    Section 3.10 states cosine/DTW were "computed for all five preprocessing
    conditions," so baseline_retrieval_metrics.json exists once PER
    condition, under that condition's own run directory. The previous
    version of this function took only sorted(...)[0] -- the alphabetically
    FIRST matching file, regardless of which condition it belonged to --
    which for run directories named "ATCN_Golf-A"/"...-Golf-B"/etc. always
    resolved to Golf-A (since "A" sorts first), silently using the WEAKEST
    baseline condition's numbers for every comparison regardless of which
    condition the trained model being compared actually used. Confirmed
    against Table 8: the old code's output for a Golf-D comparison
    (cosine=2.56%, dtw=10.26%) exactly matches Table 8's Golf-A row (2.6%,
    10.3%), not its Golf-D row (35.9%, 53.8%)."""
    out: dict[str, dict[tuple[str, str], dict[int, int]]] = {}
    name_map = {"cosine_raw_pose": "cosine", "dtw_nearest_neighbor": "dtw"}
    for candidate in sorted(root.glob("*/results/evaluation/baseline_retrieval_metrics.json")):
        method = _baseline_method_from_dir(candidate.parents[2].name)
        data = json.loads(candidate.read_text(encoding="utf-8"))
        for json_name, short_label in name_map.items():
            block = data.get("results", {}).get(json_name)
            if not block:
                continue
            q: dict[tuple[str, str], dict[int, int]] = {}
            for row in block.get("rows", []):
                rank = row.get("rank")
                if rank is None:
                    continue
                qkey = (str(row.get("expert_id", "")), str(row.get("sample_id", "")))
                q[qkey] = {0: 1 if int(rank) == 1 else 0}
            if q:
                out.setdefault(f"baseline:{short_label}:{method}", q)

    # DMSM (22_evaluate_dmsm_baseline.py) is a separate deterministic,
    # no-learning baseline written to its own pro_holdout-style file (top-level
    # "rows"/"final_rank", the same schema as a trained condition's metrics
    # json) rather than nested under "results" like cosine/dtw above -- loaded
    # from its own glob accordingly, same per-condition keying as above
    # (though DMSM was in practice only ever run under one condition, Golf-D
    # -- Section 3.10).
    for candidate in sorted(root.glob("*/results/evaluation/dmsm_pro_holdout_metrics.json")):
        method = _baseline_method_from_dir(candidate.parents[2].name)
        data = json.loads(candidate.read_text(encoding="utf-8"))
        q = {}
        for row in data.get("rows", []):
            rank = row.get("final_rank")
            if rank is None:
                continue
            qkey = (str(row.get("expert_id", "")), str(row.get("sample_id", "")))
            q[qkey] = {0: 1 if int(rank) == 1 else 0}
        if q:
            out.setdefault(f"baseline:dmsm:{method}", q)

    return out


def pooled_top1(perq: dict[tuple[str, str], dict[int, int]]) -> float:
    vals = [c for seeds in perq.values() for c in seeds.values()]
    return float(np.mean(vals)) if vals else float("nan")


def pooled_mcnemar(
    a: dict[tuple[str, str], dict[int, int]], b: dict[tuple[str, str], dict[int, int]], keys: list[tuple[str, str]]
) -> tuple[int, int, float]:
    """Naive reference stat only -- pools every shared (query, seed) cell as if
    independent, which understates correlation. Not the reported verdict."""
    b01 = b10 = 0
    for qk in keys:
        a_seeds, b_seeds = set(a[qk]), set(b[qk])
        if a_seeds == {0} and b_seeds != {0}:
            pair_seeds = b_seeds
        elif b_seeds == {0} and a_seeds != {0}:
            pair_seeds = a_seeds
        else:
            pair_seeds = a_seeds & b_seeds
        for s in pair_seeds:
            av = a[qk][s] if s in a[qk] else a[qk][0]
            bv = b[qk][s] if s in b[qk] else b[qk][0]
            if av == 0 and bv == 1:
                b01 += 1
            elif av == 1 and bv == 0:
                b10 += 1
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    k = min(b01, b10)
    tail = sum(comb(n, i) for i in range(0, k + 1)) * (0.5**n)
    p = min(1.0, 2.0 * tail)
    return b01, b10, p


def pooled_odds_ratio(b01: int, b10: int) -> float:
    if b10 == 0:
        return float("inf") if b01 > 0 else float("nan")
    return b01 / b10


def hierarchical_bootstrap(
    a: dict[tuple[str, str], dict[int, int]],
    b: dict[tuple[str, str], dict[int, int]],
    keys: list[tuple[str, str]],
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Resample queries, then one shared seed per resampled query (query and
    seed are the two nesting levels; the seed draw is shared between A and B
    since seed 42 on A and seed 42 on B are the same weight-init/data-shuffle
    draw -- a real pairing, not an independent nuisance)."""
    n = len(keys)
    diffs = np.empty(n_boot)
    log_ors = np.empty(n_boot)
    # A deterministic no-learning baseline is modeled with a single "seed" key
    # 0 (load_baseline_perquery_perseed), while a trained condition has real
    # seed keys 42-46 -- their intersection is always empty, so intersecting
    # seed sets here would silently drop every query from the resample below
    # (leaving diffs/log_ors at their zero-effect default and reporting a
    # spurious p=1.0 for every vs-baseline comparison, regardless of the true
    # effect size). When one side is a single-key deterministic baseline, we
    # instead resample the OTHER side's seeds and pair each draw with the
    # baseline's one fixed value.
    key_seed_lists = []
    for k in keys:
        a_seeds, b_seeds = set(a[k]), set(b[k])
        if a_seeds == {0} and b_seeds != {0}:
            key_seed_lists.append(sorted(b_seeds))
        elif b_seeds == {0} and a_seeds != {0}:
            key_seed_lists.append(sorted(a_seeds))
        else:
            key_seed_lists.append(sorted(a_seeds & b_seeds))

    for i in range(n_boot):
        q_idx = rng.integers(0, n, size=n)
        b01 = b10 = 0
        total_diff = 0.0
        for qi in q_idx:
            seeds = key_seed_lists[qi]
            if not seeds:
                continue
            s = seeds[rng.integers(0, len(seeds))]
            qk = keys[qi]
            av = a[qk][s] if s in a[qk] else a[qk][0]
            bv = b[qk][s] if s in b[qk] else b[qk][0]
            total_diff += av - bv
            if av == 0 and bv == 1:
                b01 += 1
            elif av == 1 and bv == 0:
                b10 += 1
        diffs[i] = total_diff / n
        # Haldane-Anscombe correction (+0.5) so log-OR stays finite when a
        # resample has zero discordant pairs on one side.
        log_ors[i] = np.log((b01 + 0.5) / (b10 + 0.5))

    diff_mean = float(np.mean(diffs))
    diff_lo, diff_hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    diff_prop = float(np.mean(diffs <= 0)) if diff_mean > 0 else float(np.mean(diffs >= 0))
    diff_p = min(1.0, 2.0 * diff_prop)

    or_mean = float(np.exp(np.mean(log_ors)))
    or_lo, or_hi = float(np.exp(np.percentile(log_ors, 2.5))), float(np.exp(np.percentile(log_ors, 97.5)))
    or_prop = float(np.mean(log_ors <= 0)) if np.mean(log_ors) > 0 else float(np.mean(log_ors >= 0))
    or_p = min(1.0, 2.0 * or_prop)

    return {
        "diff_mean": diff_mean, "diff_ci95_lo": diff_lo, "diff_ci95_hi": diff_hi, "diff_p": diff_p,
        "or_mean": or_mean, "or_ci95_lo": or_lo, "or_ci95_hi": or_hi, "or_p": or_p,
    }


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


CLEAN_CONDITION_PATTERN = re.compile(r"^(ATCN|Transformer|STGCN)_(Golf-[A-E])_(seq\+step|seq-only|step-only)$")


def default_comparisons(conditions: dict[str, Any], baselines: dict[str, Any]) -> list[tuple[str, str]]:
    top1 = {k: pooled_top1(v) for k, v in conditions.items()}
    # "best" must come from the clean, headline 45-condition sweep only (no
    # extra tag folded into the method) -- root now also holds many derived
    # ablation/variant conditions (protovariant-*, eventperturb-*, equalcap,
    # shuffledphase, ...) that can tie or nearly tie the headline result
    # (e.g. protovariant-mean reproduces it exactly). If max() over the FULL
    # top1 dict picks one of those on a tie -- ties break toward whichever
    # key was inserted first, which follows sorted(root.iterdir()) order, so
    # an alphabetically-earlier extra-tagged directory can win -- the regex
    # match below fails and this function silently returns zero comparisons
    # (confirmed: this is what produced "0 comparisons" on a real run once
    # protovariant-mean's directory existed in the same root).
    clean_top1 = {k: v for k, v in top1.items() if CLEAN_CONDITION_PATTERN.match(k)}
    if not clean_top1:
        return []
    best = max(clean_top1, key=lambda k: clean_top1[k])
    m = CLEAN_CONDITION_PATTERN.match(best)
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
        for short_label in ("dtw", "cosine", "dmsm"):
            b = f"baseline:{short_label}:{method}"
            if b in baselines:
                pairs.append((best, b))
    return pairs


def run_experiment(
    root: Path, explicit: list[tuple[str, str]], n_boot: int, boot_seed: int
) -> list[dict[str, Any]]:
    conditions = load_condition_perquery_perseed(root)
    baselines = load_baseline_perquery_perseed(root)
    pool = {**conditions, **baselines}
    pairs = explicit or default_comparisons(conditions, baselines)

    results: list[dict[str, Any]] = []
    for a_key, b_key in pairs:
        if a_key not in pool or b_key not in pool:
            print(f"[warn] unknown condition in pair: {a_key} vs {b_key}", file=sys.stderr)
            continue
        a, b = pool[a_key], pool[b_key]
        keys = sorted(set(a) & set(b))
        if not keys:
            print(f"[warn] no shared queries: {a_key} vs {b_key}", file=sys.stderr)
            continue
        b01, b10, naive_mcnemar_p = pooled_mcnemar(a, b, keys)
        rng = np.random.default_rng(zlib.crc32(f"{boot_seed}|{root.name}|{a_key}|{b_key}".encode()))
        hb = hierarchical_bootstrap(a, b, keys, n_boot, rng)
        results.append(
            {
                "experiment": root.name, "A": a_key, "B": b_key, "n_queries": len(keys),
                "top1_A": round(pooled_top1(a), 4), "top1_B": round(pooled_top1(b), 4),
                "top1_diff_pooled": round(pooled_top1(a) - pooled_top1(b), 4),
                "odds_ratio_pooled": round(pooled_odds_ratio(b01, b10), 4) if np.isfinite(pooled_odds_ratio(b01, b10)) else "inf",
                "hbootstrap_diff_mean": round(hb["diff_mean"], 4),
                "hbootstrap_diff_ci95_lo": round(hb["diff_ci95_lo"], 4),
                "hbootstrap_diff_ci95_hi": round(hb["diff_ci95_hi"], 4),
                "hbootstrap_diff_p": round(hb["diff_p"], 5),
                "hbootstrap_or_mean": round(hb["or_mean"], 4),
                "hbootstrap_or_ci95_lo": round(hb["or_ci95_lo"], 4),
                "hbootstrap_or_ci95_hi": round(hb["or_ci95_hi"], 4),
                "hbootstrap_or_p": round(hb["or_p"], 5),
                "naive_mcnemar_p_pooled_cells": round(naive_mcnemar_p, 5),
            }
        )
    for r, adj in zip(results, holm([x["hbootstrap_diff_p"] for x in results] or [])):
        r["hbootstrap_diff_p_holm"] = round(adj, 5)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--compare", action="append", default=[])
    parser.add_argument("--out", default="results_paired_significance_hierarchical.csv")
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
        print(f"\n== {root.name}: {len(res)} comparisons (hierarchical query+seed bootstrap, n_boot={args.n_boot}) ==")
        for r in res:
            sig = "*" if r["hbootstrap_diff_p_holm"] < 0.05 else " "
            print(
                f" {sig} {r['A']:<26} vs {r['B']:<22} "
                f"dTop1={r['top1_diff_pooled']:+.3f} "
                f"[{r['hbootstrap_diff_ci95_lo']:+.3f},{r['hbootstrap_diff_ci95_hi']:+.3f}] "
                f"p={r['hbootstrap_diff_p']:.4f} (Holm {r['hbootstrap_diff_p_holm']:.4f})  "
                f"OR={r['odds_ratio_pooled']} [{r['hbootstrap_or_ci95_lo']:.2f},{r['hbootstrap_or_ci95_hi']:.2f}]"
            )

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
