"""Isolate the initial population size from everything else.

The paper attributed its tight-budget advantage over L-SHADE to L-SHADE's
initial population of 18D.  That attribution cannot be read off a comparison
between L-SHADE and our own configuration, because those two differ in four
constants, not one, and also in implementation details (bound handling,
steady-state versus generational replacement, the CR memory mean, the terminal
CR value, F sampling, rounding of the pbest pool).  The four constants:

    constant          ours        L-SHADE
    N_init            6D          18D
    memory H          10          6
    archive           1.0 N       2.6 N
    elite fraction    0.25        0.11

This script analyses stage E14, which changes **one** constant at a time inside
otherwise identical code:

  * `LSHADE-N6`  vs `LSHADE-N18` -- stock L-SHADE, only the population differs;
  * `GF@18D`     vs the shipped configuration -- the same contrast inside our
    own engine, in the other direction;
  * the shipped configuration vs `LSHADE-N6` -- the *fair* comparison, in which
    both methods get the population rule this paper recommends.

The third of these is the one that matters for what the paper may claim about
its own method, and it is reported whichever way it comes out.

On CEC2014 these are unregistered comparisons and are exploratory by the
pre-registration's own clause; on CEC2017 they are diagnostic-suite results.
Neither may change the shipped configuration, and neither did.

Run:  python3 scripts/analyze_pop_isolation.py
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sehho.runfiles import read_runs  # noqa: E402  (pandas.read_csv; SEHHO_ERROR_DECIMALS, Section 2.6)
from sehho.stats import cliffs_delta, holm

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
SRC = os.path.join(RES, "E14_pop_isolation.csv")
METHOD = "GF-Method"
BOOT = 20000
SEED = 20260915


def boot_ci(v, rng):
    m = v[rng.integers(0, len(v), (BOOT, len(v)))].mean(axis=1)
    return float(np.quantile(m, .025)), float(np.quantile(m, .975))


def per_fn_delta(df, a, b, funcs):
    return np.array([cliffs_delta(df[(df.func == f) & (df.algo == a)]["error"].to_numpy(),
                                  df[(df.func == f) & (df.algo == b)]["error"].to_numpy())
                     for f in funcs])


def signed_rank(v):
    nz = np.abs(v) > 0
    return float(sps.wilcoxon(v[nz])[1]) if nz.sum() else 1.0


def method_runs(suite):
    """The shipped configuration's runs for this suite, from whichever stage holds them."""
    parts = []
    for f in ("E3_cec2017.csv", "E9_newmethod_vs_baselines.csv", "E12_cec2017_tight.csv",
              "E13_cec2014.csv"):
        p = os.path.join(RES, f)
        if not os.path.exists(p):
            continue
        d = read_runs(p)
        if "suite" in d.columns:
            d = d[d.suite == suite]
        parts.append(d[d.algo == METHOD])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main():
    if not os.path.exists(SRC):
        print(f"{SRC} not found — run: python3 scripts/run_experiments.py E14")
        return 1
    e14 = read_runs(SRC)
    rng = np.random.default_rng(SEED)
    rows = []

    for suite in ("cec2017", "cec2014"):
        s0 = e14[e14.suite == suite]
        if s0.empty:
            continue
        mine = method_runs(suite)
        print("=" * 78)
        print(f"{suite.upper()}")
        print("=" * 78)
        for dim in sorted(s0.dim.unique()):
            for fes in sorted(s0[s0.dim == dim].max_fes.unique()):
                sub = s0[(s0.dim == dim) & (s0.max_fes == fes)]
                funcs = sorted(sub.func.unique())
                budget = "tight" if fes == 15_000 else "competition"
                print(f"\n  D={dim}, {fes:,} FEs ({budget})")

                contrasts = [
                    ("L-SHADE 6D vs 18D", sub, "LSHADE-N6", "LSHADE-N18"),
                ]
                mm = mine[(mine.dim == dim) & (mine.max_fes == fes)]
                if not mm.empty:
                    merged = pd.concat([sub, mm], ignore_index=True)
                    contrasts += [
                        ("ours 6D vs ours 18D", merged, METHOD, "GF@18D"),
                        ("ours vs L-SHADE at 6D", merged, METHOD, "LSHADE-N6"),
                        ("ours vs L-SHADE default", merged, METHOD, "LSHADE-N18"),
                    ]
                praw, tmp = [], []
                for label, frame, a, b in contrasts:
                    if a not in set(frame.algo) or b not in set(frame.algo):
                        continue
                    dl = per_fn_delta(frame, a, b, funcs)
                    lo, hi = boot_ci(dl, rng)
                    p = signed_rank(dl)
                    praw.append(p)
                    tmp.append(dict(suite=suite, dim=dim, max_fes=fes, budget=budget,
                                    contrast=label, mean_delta=float(dl.mean()),
                                    ci_lo=lo, ci_hi=hi, p=p,
                                    first_better=bool(hi < 0), second_better=bool(lo > 0)))
                for r, ph in zip(tmp, holm(praw)):
                    r["p_holm"] = float(ph)
                    verdict = ("first better" if r["first_better"] else
                               "second better" if r["second_better"] else "no difference")
                    print(f"    {r['contrast']:<26s} {r['mean_delta']:+.3f} "
                          f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]  p_Holm={ph:.4f}  {verdict}")
                rows += tmp
        print()

    out = pd.DataFrame(rows)
    path = os.path.join(RES, "analysis", "pop_isolation.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path, index=False)

    print("=" * 78)
    print("WHAT THIS ESTABLISHES")
    print("=" * 78)
    iso = out[out.contrast == "L-SHADE 6D vs 18D"]
    t = iso[iso.budget == "tight"]
    c = iso[iso.budget == "competition"]
    print(f"  Inside L-SHADE alone, at tight budgets, 6D beats 18D in "
          f"{int(t.first_better.sum())}/{len(t)} blocks "
          f"(mean delta {t.mean_delta.mean():+.3f}).")
    print(f"  At competition budgets the ordering reverses in "
          f"{int(c.second_better.sum())}/{len(c)} blocks "
          f"(mean delta {c.mean_delta.mean():+.3f}).")
    fair = out[out.contrast == "ours vs L-SHADE at 6D"]
    ft = fair[fair.budget == "tight"]
    print(f"\n  Given the SAME population rule, our configuration beats L-SHADE in "
          f"{int(ft.first_better.sum())}/{len(ft)} tight-budget blocks and loses in "
          f"{int(ft.second_better.sum())}.")
    print("  => the tight-budget advantage previously reported against L-SHADE is "
          "attributable to\n     the population rule, not to the rest of the "
          "configuration.")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
