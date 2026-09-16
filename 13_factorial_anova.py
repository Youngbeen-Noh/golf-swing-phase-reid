"""3-way ANOVA effect-size decomposition: model x preprocess x option (P1-1/6.2).

Results 6.2 (Preprocessing Ablation) only supported "preprocessing matters more
than encoder choice" with a descriptive range comparison (Golf-A vs Golf-B~E
spread vs best-per-model spread), which the instruction doc's P1-1/10.2 gate
explicitly disallows as a causal claim without a real factorial analysis. This
script runs that analysis: a fixed-effects ANOVA on results_seed_raw.csv
(model x preprocess x option, with the 5 seeds as replicates within each
condition) and reports eta-squared (share of total variance) per factor and
per two-way interaction.

Observation unit and eta-squared type (2nd revision instruction 3.3):
  - Each observation is one seed's CONDITION-LEVEL mean accuracy (aggregated
    over the fixed 91-query holdout set), not a per-query observation. The 5
    seeds per cell are repeated measurements over the SAME query set and the
    SAME train/holdout split -- they are not independent samples of "queries"
    or "splits", only of training-stochasticity. This ANOVA therefore cannot
    speak to query-level or split-level uncertainty; see
    19_paired_significance_hierarchical.py (hierarchical query+seed bootstrap)
    and 20_query_level_logistic_variance.py (query-level logistic GEE, which
    treats accuracy's bounded 0/1 nature directly) for that.
  - eta_sq reported below is CLASSICAL (non-partial) eta-squared:
    SS_effect / SS_total. Partial eta-squared (SS_effect / (SS_effect +
    SS_residual)) is also reported as eta_sq_partial for comparison; the two
    diverge because of the large residual (seed-noise) pool at this sample
    size (5 seeds x 5 methods x 3 options x up to 3 models per experiment).

Usage:
  python 13_factorial_anova.py
  python 13_factorial_anova.py --metric mrr
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols


def load_raw(path: Path, metric: str) -> pd.DataFrame:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    df = pd.DataFrame(rows)
    df["response"] = df[metric].astype(float) * (100.0 if metric == "top1_accuracy" else 1.0)
    df = df.rename(columns={"method": "preprocess"})
    return df


def run_anova(df: pd.DataFrame) -> pd.DataFrame:
    formula = (
        "response ~ C(model) + C(preprocess) + C(option) "
        "+ C(model):C(preprocess) + C(model):C(option) + C(preprocess):C(option)"
    )
    model = ols(formula, data=df).fit()
    aov = sm.stats.anova_lm(model, typ=2)
    ss_total = aov["sum_sq"].sum()
    ss_residual = aov.loc["Residual", "sum_sq"]
    aov["eta_sq"] = aov["sum_sq"] / ss_total
    aov["eta_sq_partial"] = aov["sum_sq"] / (aov["sum_sq"] + ss_residual)
    aov = aov.reset_index().rename(columns={"index": "term"})
    return aov


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-csv", default="results_seed_raw.csv")
    parser.add_argument("--metric", default="top1_accuracy", choices=["top1_accuracy", "mrr", "mean_rank"])
    parser.add_argument("--out-prefix", default="results_factorial_anova")
    args = parser.parse_args()

    df = load_raw(Path(args.raw_csv), args.metric)

    all_rows = []
    for experiment in sorted(df["experiment"].unique()):
        sub = df[df["experiment"] == experiment]
        aov = run_anova(sub)
        aov.insert(0, "experiment", experiment)
        all_rows.append(aov)
        print(
            f"\n=== {experiment} (n={len(sub)} seed-level condition means, metric={args.metric}) ==="
            f"\n(observation = one seed's aggregate accuracy over the fixed holdout query set; "
            f"eta_sq = classical/non-partial SS_effect/SS_total; eta_sq_partial = SS_effect/(SS_effect+SS_residual))"
        )
        print(aov[["term", "sum_sq", "df", "F", "PR(>F)", "eta_sq", "eta_sq_partial"]].round(4).to_string(index=False))

    combined = pd.concat(all_rows, ignore_index=True)
    out_path = Path(f"{args.out_prefix}_{args.metric}.csv")
    combined.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
