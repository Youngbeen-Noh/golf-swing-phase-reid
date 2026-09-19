# -*- coding: utf-8 -*-
"""Crossed (seed x query) bootstrap used for the manuscript's Tables 7, 8 and 11.

Scheme (as stated in Section 3.11): in each of 10,000 replicates, seeds are
resampled with replacement (5 from 5) and queries are resampled with
replacement (Q from Q); the replicate accuracy is the mean of the correctness
matrix over the resampled query x seed grid. Both conditions in a comparison
share the same resampled seed and query indices (deterministic baselines use
the same query resample; their value does not depend on the seed draw).
Two-sided p = 2 * min(P(diff <= 0), P(diff >= 0)), capped at 1; Holm within
each reported family. Bootstrap RNG seed 123.

Usage:
  python 30_crossed_bootstrap_stats.py --archive-root experiments_archive/golf_experiment_2 \
      --live-root data/Golf/experiments
"""
import argparse
import json
from pathlib import Path

import numpy as np

SEEDS = [42, 43, 44, 45, 46]
NBOOT = 10000
RNGSEED = 123


def load_rows(prefix, root, metrics="pro_holdout_user_metrics.json"):
    """(Q,S) matrices: correct 0/1 and rank int, plus ordered query keys."""
    per_seed_c, per_seed_r = [], []
    qkeys = None
    for s in SEEDS:
        p = Path(root) / f"{prefix}-s{s}" / "results" / "evaluation" / metrics
        d = json.loads(p.read_text(encoding="utf-8"))
        c, r = {}, {}
        for row in d.get("rows", []):
            fr = row.get("final_rank")
            if fr in ("", None):
                continue
            k = (str(row.get("expert_id")), str(row.get("sample_id")))
            r[k] = int(fr)
            c[k] = 1 if int(fr) == 1 else 0
        if qkeys is None:
            qkeys = sorted(c)
        assert sorted(c) == qkeys, f"query mismatch {prefix} s{s}"
        per_seed_c.append([c[k] for k in qkeys])
        per_seed_r.append([r[k] for k in qkeys])
    return (np.array(per_seed_c, float).T, np.array(per_seed_r, float).T, qkeys)


def crossed_ci(mat, nboot=NBOOT, rngseed=RNGSEED):
    rng = np.random.default_rng(rngseed)
    Q, S = mat.shape
    accs = np.empty(nboot)
    for b in range(nboot):
        qi = rng.integers(0, Q, Q)
        si = rng.integers(0, S, S)
        accs[b] = mat[np.ix_(qi, si)].mean()
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def crossed_compare(mA, mB, nboot=NBOOT, rngseed=RNGSEED):
    """Shared seed+query indices. mB may be (Q,1) for deterministic baselines."""
    rng = np.random.default_rng(rngseed)
    Q, S = mA.shape
    diffs = np.empty(nboot)
    for b in range(nboot):
        qi = rng.integers(0, Q, Q)
        si = rng.integers(0, S, S)
        a = mA[np.ix_(qi, si)].mean()
        bv = mB[np.ix_(qi, si)].mean() if mB.shape[1] == S else mB[qi, 0].mean()
        diffs[b] = a - bv
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {
        "diff": float(mA.mean() - (mB.mean())),
        "ci_lo": float(np.percentile(diffs, 2.5)),
        "ci_hi": float(np.percentile(diffs, 97.5)),
        "p": float(min(1.0, p)),
    }


def holm(ps):
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    out = [None] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * ps[idx])
        out[idx] = min(1.0, running)
    return out


def fmt(mat):
    seed_means = mat.mean(axis=0)
    return f"{mat.mean()*100:5.1f} +- {seed_means.std(ddof=1)*100:4.1f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--archive-root", default="experiments_archive/golf_experiment_2",
                    help="canonical archived run dirs (Table 7/11 configurations)")
    ap.add_argument("--live-root", default="data/Golf/experiments",
                    help="live run dirs (held-out test confirmatory evaluation)")
    args = ap.parse_args()
    ARCH, LIVE = Path(args.archive_root), Path(args.live_root)

    print("=" * 78)
    print("A. Table 7 configs -- crossed bootstrap 95% CI (validation, 39 queries)")
    print("=" * 78)
    t7 = ["ATCN_Golf-D", "ATCN_Golf-D-steponly", "ATCN_Golf-C",
          "Transformer_Golf-D-steponly", "Transformer_Golf-C-steponly", "Transformer_Golf-C"]
    mats = {}
    for pfx in t7:
        m, r, q = load_rows(pfx, ARCH)
        mats[pfx] = (m, r, q)
        lo, hi = crossed_ci(m)
        print(f"  {pfx:30s} {fmt(m)}  CI=[{lo*100:.1f}, {hi*100:.1f}]")

    print()
    print("=" * 78)
    print("B. Held-out test (52 queries) -- selected configuration")
    print("=" * 78)
    mL, rL, qL = load_rows("ATCN_Golf-D", root=LIVE,
                           metrics="pro_holdout_user_metrics_LOCKED_TEST.json")
    lo, hi = crossed_ci(mL)
    print(f"  Top-1 {fmt(mL)}  CI=[{lo*100:.1f}, {hi*100:.1f}]")
    for k, name in [(3, "Top-3"), (5, "Top-5")]:
        mk = (rL <= k).astype(float)
        lo2, hi2 = crossed_ci(mk)
        print(f"  {name} {fmt(mk)}  CI=[{lo2*100:.1f}, {hi2*100:.1f}]")
    mrr = 1.0 / rL
    seed_means = mrr.mean(axis=0)
    rng = np.random.default_rng(RNGSEED)
    accs = np.empty(NBOOT)
    Q, S = mrr.shape
    for b in range(NBOOT):
        qi = rng.integers(0, Q, Q)
        si = rng.integers(0, S, S)
        accs[b] = mrr[np.ix_(qi, si)].mean()
    print(f"  MRR   {mrr.mean():.3f} +- {seed_means.std(ddof=1):.3f}  "
          f"CI=[{np.percentile(accs,2.5):.3f}, {np.percentile(accs,97.5):.3f}]")
    print(f"  Mean rank {rL.mean():.3f} +- {rL.mean(axis=0).std(ddof=1):.3f}")
    # sensitivity: exclude the one query whose session pair crosses the
    # train/held-out boundary (anonymized id expert_20_005 in the released data)
    keep = [i for i, k in enumerate(qL) if k[1] != "expert_20_005"]
    if len(keep) < len(qL):
        mL2 = mL[keep]
        lo3, hi3 = crossed_ci(mL2)
        print(f"  [sensitivity] excl boundary-pair query ({len(keep)}q): "
              f"{fmt(mL2)}  CI=[{lo3*100:.1f}, {hi3*100:.1f}]")

    print()
    print("=" * 78)
    print("C. Table 11 family -- crossed bootstrap, Holm over 7")
    print("=" * 78)
    t11 = {}
    for pfx in ["ATCN_Golf-D-seqonly", "ATCN_Golf-D-seq140-seqonly", "ATCN_Golf-D-seq160-seqonly",
                "ATCN_Golf-D-shuffledphase", "ATCN_Golf-D-seq140-seqphase"]:
        m, r, q = load_rows(pfx, ARCH)
        t11[pfx] = m
    t11["ATCN_Golf-D"] = mats["ATCN_Golf-D"][0]
    t11["ATCN_Golf-D-steponly"] = mats["ATCN_Golf-D-steponly"][0]
    fam = [
        ("legacy Seq-only  vs Phase-only", "ATCN_Golf-D-seqonly", "ATCN_Golf-D-steponly"),
        ("Seq-only-140     vs Phase-only", "ATCN_Golf-D-seq140-seqonly", "ATCN_Golf-D-steponly"),
        ("Seq-only-160     vs Phase-only", "ATCN_Golf-D-seq160-seqonly", "ATCN_Golf-D-steponly"),
        ("Shuffled         vs Seq+Phase ", "ATCN_Golf-D-shuffledphase", "ATCN_Golf-D"),
        ("Seq+Phase-140    vs Seq-only-140", "ATCN_Golf-D-seq140-seqphase", "ATCN_Golf-D-seq140-seqonly"),
        ("Seq+Phase        vs Phase-only", "ATCN_Golf-D", "ATCN_Golf-D-steponly"),
        ("Seq-only-140     vs legacy Seq-only", "ATCN_Golf-D-seq140-seqonly", "ATCN_Golf-D-seqonly"),
    ]
    res = [crossed_compare(t11[a], t11[b]) for _, a, b in fam]
    hs = holm([r["p"] for r in res])
    for (label, _, _), r, h in zip(fam, res, hs):
        print(f"  {label:38s} d={r['diff']*100:+5.1f} "
              f"[{r['ci_lo']*100:+.1f},{r['ci_hi']*100:+.1f}] p={r['p']:.4f} Holm={h:.4f}")


if __name__ == "__main__":
    main()
