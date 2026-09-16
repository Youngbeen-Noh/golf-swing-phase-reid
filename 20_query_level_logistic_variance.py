"""Query-level logistic variance decomposition (2nd revision instruction 3.3).

13_factorial_anova.py's 3-way ANOVA treats one seed's CONDITION-LEVEL mean
accuracy as the observation (n=5 seeds/cell) -- defensible for "does the
condition mean differ", but accuracy is a bounded [0,1] aggregate of a binary
outcome, and the instruction asks for a mixed model / query-level logistic
model as a check that doesn't first average away the query dimension.

This script keeps every (model, preprocess, option, seed, query) cell as its
own binary observation (correct = 1[final_rank == 1]) and fits:
  1. An ordinary logistic GLM (correct ~ model + preprocess + option + all
     2-way interactions) -- used only for a deviance-based, eta-squared-style
     "share of explained deviance" per term (a binomial analog to the
     ANOVA's SS-based eta_sq). Its naive p-values are NOT reported as the
     final significance verdict, because...
  2. ...the same formula is refit as a GEE (Generalized Estimating Equations,
     Binomial family, exchangeable working correlation) with query
     (expert_id+sample_id) as the cluster/group variable, since the same
     query is measured repeatedly across every seed and every condition.
     GEE's robust ("sandwich") standard errors are valid under that
     within-query correlation, unlike the GLM's naive SEs. This is the
     query-level logistic alternative the instruction asks to consider given
     accuracy's bounded distribution, and it is the significance verdict this
     script actually reports.

Usage:
  python 20_query_level_logistic_variance.py ~/experiments_archive/golf_experiment_2 ~/experiments_archive/golf_experiment_1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial

RUN_ID_PATTERN = re.compile(
    r"^(?P<model>ATCN|Transformer|STGCN)_(?P<method>Golf-[A-E])"
    r"(?P<option>-seqonly|-steponly)?-s(?P<seed>\d+)$"
)
OPTION_LABELS = {None: "seq+step", "-seqonly": "seq-only", "-steponly": "step-only"}


def load_long(root: Path) -> pd.DataFrame:
    """One row per (model, preprocess, option, seed, expert_id, sample_id)."""
    records: list[dict[str, Any]] = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        m = RUN_ID_PATTERN.match(run_dir.name)
        if m is None:
            continue
        model, method, seed = m.group("model"), m.group("method"), m.group("seed")
        option = OPTION_LABELS[m.group("option")]
        metrics_path = run_dir / "results" / "evaluation" / "pro_holdout_user_metrics.json"
        if not metrics_path.exists():
            print(f"[warn] missing metrics json: {run_dir.name}", file=sys.stderr)
            continue
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        for row in data.get("rows", []):
            fr = row.get("final_rank", "")
            if fr == "" or fr is None:
                continue
            query_id = f"{row.get('expert_id', '')}__{row.get('sample_id', '')}"
            records.append(
                {
                    "model": model, "preprocess": method, "option": option, "seed": int(seed),
                    "query_id": query_id, "correct": 1 if int(fr) == 1 else 0,
                }
            )
    return pd.DataFrame.from_records(records)


TERMS = [
    "C(model)", "C(preprocess)", "C(option)",
    "C(model):C(preprocess)", "C(model):C(option)", "C(preprocess):C(option)",
]


def glm_deviance_shares(df: pd.DataFrame, formula: str) -> pd.DataFrame:
    """Type-II-style deviance decomposition: for each term, compare deviance
    of (full model) vs (full model minus that term), as a share of the null
    deviance -- a binomial analog to the ANOVA's SS_effect/SS_total eta_sq.

    Builds each reduced formula from the explicit TERMS list (excluding the
    target term) rather than via substring removal on the formula string --
    a naive `formula.replace(f" + {term}", "")` is unsafe here because
    "C(preprocess)" is a literal prefix-substring of the interaction term
    "C(preprocess):C(option)" (and "C(model)" of both interactions
    involving it): removing "C(model)" that way also strips the
    " + C(model)" prefix out of " + C(model):C(preprocess)"/":C(option)",
    and removing "C(preprocess)" collides with "C(preprocess):C(option)"
    the same way. Confirmed on a real run: this previously produced
    C(option)'s deviance share as exactly 0 and made
    C(preprocess):C(option)'s reported share identical to C(preprocess)'s."""
    full = smf.glm(formula, data=df, family=Binomial()).fit()
    null = smf.glm("correct ~ 1", data=df, family=Binomial()).fit()
    null_dev = null.deviance

    rows = []
    for term in TERMS:
        reduced_terms = [t for t in TERMS if t != term]
        reduced_formula = "correct ~ " + " + ".join(reduced_terms) if reduced_terms else "correct ~ 1"
        reduced = smf.glm(reduced_formula, data=df, family=Binomial()).fit()
        dev_explained_by_term = reduced.deviance - full.deviance
        rows.append(
            {
                "term": term,
                "deviance_explained": dev_explained_by_term,
                "pseudo_eta_sq": dev_explained_by_term / null_dev,
                "df": full.df_model - reduced.df_model,
            }
        )
    out = pd.DataFrame(rows)
    out["mcfadden_pseudo_r2_full_model"] = 1.0 - full.deviance / null_dev
    return out


def gee_significance(df: pd.DataFrame, formula: str) -> pd.DataFrame:
    """Same formula, GEE with exchangeable correlation clustered by query_id
    (the same query is repeated across every seed and every condition), so
    the robust SEs -- and therefore the p-values below -- are valid under
    that non-independence, unlike a naive GLM fit on the same long table."""
    model = smf.gee(
        formula, groups="query_id", data=df, cov_struct=Exchangeable(), family=Binomial()
    )
    result = model.fit()
    summary_df = pd.DataFrame(
        {
            "coef": result.params,
            "robust_se": result.bse,
            "z": result.tvalues,
            "p_robust": result.pvalues,
        }
    )
    summary_df["odds_ratio"] = np.exp(summary_df["coef"])
    return summary_df.reset_index().rename(columns={"index": "term_level"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--out-prefix", default="results_query_level_logistic")
    args = parser.parse_args()

    formula = (
        "correct ~ C(model) + C(preprocess) + C(option) "
        "+ C(model):C(preprocess) + C(model):C(option) + C(preprocess):C(option)"
    )

    all_deviance, all_gee = [], []
    for root_str in args.roots:
        root = Path(root_str).expanduser()
        if not root.is_dir():
            print(f"[warn] not a directory: {root}", file=sys.stderr)
            continue
        df = load_long(root)
        if df.empty:
            print(f"[warn] no per-query rows found under {root}", file=sys.stderr)
            continue
        n_queries = df["query_id"].nunique()
        n_seeds = df["seed"].nunique()
        print(
            f"\n=== {root.name}: {len(df)} (query x seed x condition) binary observations, "
            f"{n_queries} unique queries, {n_seeds} seeds ==="
        )

        dev = glm_deviance_shares(df, formula)
        dev.insert(0, "experiment", root.name)
        all_deviance.append(dev)
        print("\n-- GLM deviance decomposition (pseudo eta_sq, descriptive only) --")
        print(dev.round(4).to_string(index=False))

        gee = gee_significance(df, formula)
        gee.insert(0, "experiment", root.name)
        all_gee.append(gee)
        print(
            "\n-- GEE (Binomial, exchangeable, clustered by query_id) -- "
            "robust p-values are the significance verdict --"
        )
        print(gee.round(4).to_string(index=False))

    if all_deviance:
        pd.concat(all_deviance, ignore_index=True).to_csv(f"{args.out_prefix}_deviance.csv", index=False)
        print(f"\nSaved: {args.out_prefix}_deviance.csv")
    if all_gee:
        pd.concat(all_gee, ignore_index=True).to_csv(f"{args.out_prefix}_gee.csv", index=False)
        print(f"Saved: {args.out_prefix}_gee.csv")


if __name__ == "__main__":
    main()
