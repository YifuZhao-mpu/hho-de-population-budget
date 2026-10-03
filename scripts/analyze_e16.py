"""Analysis of stage E16, exactly as fixed in its pre-registration.

For each of the five engineering problems: feasibility of L-SHADE at
N_init = 6D and at 18D (Fisher's exact test), Cliff's delta between the two arms
on the Deb-rule fitness that the runs minimise (negative favours 6D) with a
percentile bootstrap interval over runs, a two-sided Mann-Whitney test with Holm
correction across the five problems, and the best and median objective among
feasible runs.  The directional expectation registered in advance is checked
against evaluations per variable.  Reported whatever it shows.

Run:  python3 scripts/analyze_e16.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sehho.stats import cliffs_delta, holm

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
SRC = os.path.join(RES, "E16_engineering_population.csv")
PRE = os.path.join(RES, "analysis", "PREREGISTRATION_E16_engineering_population.json")
BOOT = 20000
SEED = 20261002
NEGLIGIBLE = 0.147


def boot_ci(a, b, rng, level=0.95):
    """Percentile interval for Cliff's delta, resampling runs within each arm."""
    ia = rng.integers(0, len(a), (BOOT, len(a)))
    ib = rng.integers(0, len(b), (BOOT, len(b)))
    d = np.array([cliffs_delta(a[i], b[j]) for i, j in zip(ia, ib)])
    return (float(np.quantile(d, (1 - level) / 2)), float(np.quantile(d, 1 - (1 - level) / 2)))


def main():
    pre = json.load(open(PRE))
    df = pd.read_csv(SRC)
    rng = np.random.default_rng(SEED)
    rows = []
    for prob in pre["design"]["problems"]:
        sub = df[df.problem == prob]
        a = sub[sub.algo == "LSHADE-N6"]
        b = sub[sub.algo == "LSHADE-N18"]
        if len(a) != 30 or len(b) != 30:
            raise SystemExit(f"{prob}: expected 30 runs per arm, got {len(a)} and {len(b)}")
        fa, fb = int(a.feasible.sum()), int(b.feasible.sum())
        fisher_p = float(sps.fisher_exact([[fa, 30 - fa], [fb, 30 - fb]])[1])
        xa, xb = a.best_f.to_numpy(), b.best_f.to_numpy()
        d = cliffs_delta(xa, xb)
        lo, hi = boot_ci(xa, xb, rng)
        mw = float(sps.mannwhitneyu(xa, xb, alternative="two-sided")[1])
        feas = lambda s: s[s.feasible.astype(bool)].objective
        dim = int(a.dim.iloc[0])
        rows.append(dict(problem=prob, dim=dim, evals_per_variable=pre["design"]["budget"] / dim,
                         feas_6D=fa, feas_18D=fb, fisher_p=fisher_p,
                         delta=d, ci_lo=lo, ci_hi=hi, mw_p=mw,
                         best_6D=feas(a).min() if fa else np.nan, best_18D=feas(b).min() if fb else np.nan,
                         median_6D=feas(a).median() if fa else np.nan,
                         median_18D=feas(b).median() if fb else np.nan))
    out = pd.DataFrame(rows)
    out["mw_p_holm"] = holm(out.mw_p.to_numpy())
    out["verdict"] = ["6D better" if r.ci_hi < 0 and r.mw_p_holm < 0.05 else
                      "18D better" if r.ci_lo > 0 and r.mw_p_holm < 0.05 else
                      "negligible" if -NEGLIGIBLE < r.ci_lo and r.ci_hi < NEGLIGIBLE else
                      "no detectable difference" for r in out.itertuples()]
    pd.set_option("display.width", 220)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    os.makedirs(os.path.join(RES, "analysis"), exist_ok=True)
    out.to_csv(os.path.join(RES, "analysis", "e16_engineering_population.csv"), index=False)
    # registered directional expectation: 6D is not worse on the problem with the
    # fewest evaluations per variable, and 18D is not worse on those with the most
    low = out.sort_values("evals_per_variable").iloc[0]
    high = out[out.evals_per_variable >= 3000]
    check = {"lowest_evals_per_variable_problem": low.problem,
             "6D_not_worse_there": bool(low.ci_lo <= 0 or low.mw_p_holm >= 0.05),
             "18D_not_worse_where_evals_per_variable_ge_3000": bool(((high.ci_hi >= 0) | (high.mw_p_holm >= 0.05)).all())}
    print("\nRegistered directional expectation:", check)
    json.dump({"rows": out.to_dict("records"), "directional_expectation": check},
              open(os.path.join(RES, "analysis", "e16_verdict.json"), "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
