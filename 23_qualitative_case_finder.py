"""Find fresh qualitative case-study examples (Section 4.6) on the corrected
39-clip validation set, replacing the pre-split examples (expert_17_023,
expert_03_025) which no longer fully exist in the current split.

Scans results/user_similarity/*/*_similarity.json (06_infer_user_similarity.py
output) for each of the 5 seeds of a run_id prefix, computes per-query
sequence_rank / final_rank / rank_delta (= final_rank - sequence_rank), and
reports:
  - the 5-seed-mean rank_delta distribution (improved / degraded / unchanged
    counts), matching Section 4.6's original methodology
  - the best "success" case: 5-seed-mean final_rank == 1 with the largest
    5-seed-mean improvement over sequence_rank
  - the best per-seed "failure" case: a single seed where sequence_rank == 1
    but final_rank > 1, picking the one with the largest degradation

For the two picked cases, also prints the full similarity JSON (sequence
ranking top-3, all 7 per-phase top-1 scores for the true expert, final
ranking top-3, mean_phase_score for the true expert and its top rivals) so
the paper's per-phase narrative can be rewritten from real numbers.

Usage:
  python 23_qualitative_case_finder.py --archive-root ~/experiments_archive/golf_experiment_2 \
      --run-prefix ATCN_Golf-D --seeds 42 43 44 45 46
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def rank_of(true_expert: str, ranking: list[dict[str, Any]]) -> int:
    ids = [str(r["expert_id"]) for r in ranking]
    return ids.index(true_expert) + 1 if true_expert in ids else len(ids) + 1


def load_seed_reports(archive_root: Path, run_id: str) -> dict[tuple[str, str], dict[str, Any]]:
    base = archive_root / run_id / "results" / "user_similarity"
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(base.rglob("*_similarity.json")):
        true_expert = path.parent.name
        report = json.loads(path.read_text(encoding="utf-8"))
        sample_id = str(report["sample_id"])
        out[(true_expert, sample_id)] = report
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archive-root", required=True)
    parser.add_argument("--run-prefix", default="ATCN_Golf-D")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = parser.parse_args()

    archive_root = Path(args.archive_root).expanduser()
    per_seed: dict[int, dict[tuple[str, str], dict[str, Any]]] = {}
    for seed in args.seeds:
        run_id = f"{args.run_prefix}-s{seed}"
        per_seed[seed] = load_seed_reports(archive_root, run_id)

    all_keys = sorted(set().union(*[set(d) for d in per_seed.values()]))

    # 5-seed-mean rank_delta distribution
    mean_deltas = []
    improved = degraded = unchanged = 0
    per_query_mean_delta = {}
    for key in all_keys:
        deltas = []
        for seed in args.seeds:
            report = per_seed[seed].get(key)
            if report is None:
                continue
            true_expert = key[0]
            seq_rank = rank_of(true_expert, report["sequence_ranking"])
            final_rank = rank_of(true_expert, report["final_ranking"])
            deltas.append(final_rank - seq_rank)
        if not deltas:
            continue
        mean_delta = sum(deltas) / len(deltas)
        per_query_mean_delta[key] = mean_delta
        mean_deltas.append(mean_delta)
        if mean_delta < 0:
            improved += 1
        elif mean_delta > 0:
            degraded += 1
        else:
            unchanged += 1

    n = len(mean_deltas)
    print(f"=== Rank-delta distribution ({n} queries, {args.run_prefix}, 5-seed mean) ===")
    print(f"improved: {improved} ({100*improved/n:.1f}%), degraded: {degraded} ({100*degraded/n:.1f}%), "
          f"unchanged: {unchanged} ({100*unchanged/n:.1f}%)")
    print(f"mean rank_delta: {sum(mean_deltas)/n:.3f}")

    # Best success case: mean final_rank == 1 (every seed correct) with
    # largest mean improvement over sequence_rank.
    success_candidates = []
    for key in all_keys:
        final_ranks = []
        seq_ranks = []
        for seed in args.seeds:
            report = per_seed[seed].get(key)
            if report is None:
                continue
            true_expert = key[0]
            final_ranks.append(rank_of(true_expert, report["final_ranking"]))
            seq_ranks.append(rank_of(true_expert, report["sequence_ranking"]))
        if final_ranks and all(r == 1 for r in final_ranks) and any(r > 1 for r in seq_ranks):
            mean_seq_rank = sum(seq_ranks) / len(seq_ranks)
            success_candidates.append((mean_seq_rank, key))
    success_candidates.sort(reverse=True)
    if success_candidates:
        mean_seq_rank, best_key = success_candidates[0]
        print(f"\n=== SUCCESS case candidate: {best_key[1]} (true expert: {best_key[0]}) "
              f"(mean sequence_rank={mean_seq_rank:.2f}, mean final_rank=1.0) ===")
        seed = args.seeds[0]
        print(f"-- seed {seed} full report --")
        print(json.dumps(per_seed[seed][best_key], ensure_ascii=False, indent=2))
    else:
        print("\nNo clean success case found (final_rank==1 every seed with sequence_rank>1 some seed).")

    # Best failure case: a single seed where sequence_rank == 1 but
    # final_rank > 1, picking the largest degradation.
    failure_candidates = []
    for key in all_keys:
        true_expert = key[0]
        for seed in args.seeds:
            report = per_seed[seed].get(key)
            if report is None:
                continue
            seq_rank = rank_of(true_expert, report["sequence_ranking"])
            final_rank = rank_of(true_expert, report["final_ranking"])
            if seq_rank == 1 and final_rank > 1:
                failure_candidates.append((final_rank - seq_rank, seed, key))
    failure_candidates.sort(reverse=True)
    if failure_candidates:
        delta, seed, key = failure_candidates[0]
        print(f"\n=== FAILURE case candidate: {key[1]} (true expert: {key[0]}) (seed {seed}, "
              f"sequence_rank=1 -> final_rank degraded by {delta}) ===")
        print(json.dumps(per_seed[seed][key], ensure_ascii=False, indent=2))
    else:
        print("\nNo failure case found (no query with sequence_rank==1 degrading to final_rank>1 in any seed).")


if __name__ == "__main__":
    main()
