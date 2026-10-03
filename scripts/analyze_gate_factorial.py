"""Disentangle the escape-energy gate from the Levy perturbation it also gates.

Deleting the gate (`AlwaysPbest`) does two things at once: it widens the pbest
pool on every update, and — because the same predicate gates the Levy
perturbation — it raises Levy activation from qbar*0.3 = 4.6% to 30% of updates
and extends it past t = T/2, where the proposed gate is identically zero. A
comparison of `Full` against `AlwaysPbest` therefore cannot attribute the
difference to elite breadth.

This script analyses the 2x2 factorial that separates them:

                       Levy gated (default)    Levy off
    gate on            Full                    NoLevy
    gate off           AlwaysPbest             AlwaysPbest+NoLevy

The gate effect is estimated twice, once at each Levy level. If the two agree,
the gate effect is not attributable to the Levy change; if they differ, the two
interact and neither main effect is interpretable alone.

The same 2x2 is run at two budgets on the *same* suite and dimension
(CEC2017, D=30, 15,000 and 300,000 evaluations), which is what makes the
budget claim identifiable: suite and dimensionality are held fixed.

Run:  python3 scripts/analyze_gate_factorial.py
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sehho.stats import friedman, cliffs_delta, cliffs_magnitude, holm

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
CELLS = {("on", "gated"): "SEHHO:Full",
         ("on", "off"): "SEHHO:NoLevy",
         ("off", "gated"): "SEHHO:AlwaysPbest",
         ("off", "off"): "SEHHO:AlwaysPbest+NoLevy"}
BLOCKS = [("15,000 FEs", "E4c_ablation_cec2017_30D_15k.csv"),
          ("300,000 FEs", "E4b_ablation_cec2017_30D.csv")]


def mean_matrix(df, algos, funcs, value="error"):
    return np.array([[df[(df.func == f) & (df.algo == a)][value].mean() for a in algos]
                     for f in funcs])


def per_function_delta(df, a, b, funcs, value="error"):
    """Mean per-function Cliff's delta of `a` against `b` (negative favours a)."""
    ds = []
    for f in funcs:
        x = df[(df.func == f) & (df.algo == a)][value].to_numpy()
        y = df[(df.func == f) & (df.algo == b)][value].to_numpy()
        ds.append(cliffs_delta(x, y))
    return float(np.mean(ds))


def analyse(df, label):
    algos = [CELLS[k] for k in CELLS]
    missing = [a for a in algos if a not in set(df.algo)]
    if missing:
        print(f"  [{label}] missing cells: {missing} — skipped")
        return None
    funcs = sorted(df.func.unique())
    M = mean_matrix(df, algos, funcs)
    fr = friedman(M)
    rank = dict(zip(algos, fr["avg_ranks"]))

    print(f"\n=== CEC2017, D=30, {label} — 2x2 factorial over {len(funcs)} functions ===")
    print(f"    Friedman within the 2x2: F={fr['iman_davenport_F']:.2f}, p={fr['p']:.2e}")
    print(f"\n    {'':<12}{'Levy gated':>14}{'Levy off':>14}")
    for g in ("on", "off"):
        print(f"    gate {g:<7}" + "".join(
            f"{rank[CELLS[(g, l)]]:>14.2f}" for l in ("gated", "off")))

    rows, praw = [], []
    for lev in ("gated", "off"):
        a, b = CELLS[("off", lev)], CELLS[("on", lev)]   # gate removed vs kept
        x = M[:, algos.index(b)]      # gate ON  (baseline)
        y = M[:, algos.index(a)]      # gate OFF
        nz = np.abs(y - x) > 0
        stat, p = sps.wilcoxon(x[nz], y[nz]) if nz.sum() else (np.nan, 1.0)
        rows.append(dict(effect=f"remove gate, Levy {lev}",
                         rank_gate_on=rank[b], rank_gate_off=rank[a],
                         rank_change=rank[a] - rank[b],
                         wins_for_gate_off=int((y < x).sum()),
                         losses=int((y > x).sum()), p=float(p),
                         delta=per_function_delta(df, a, b, funcs)))
        praw.append(p)
    for lev in ("on", "off"):
        a, b = CELLS[(lev, "off")], CELLS[(lev, "gated")]  # Levy removed vs kept
        x = M[:, algos.index(b)]
        y = M[:, algos.index(a)]
        nz = np.abs(y - x) > 0
        stat, p = sps.wilcoxon(x[nz], y[nz]) if nz.sum() else (np.nan, 1.0)
        rows.append(dict(effect=f"remove Levy, gate {lev}",
                         rank_gate_on=rank[b], rank_gate_off=rank[a],
                         rank_change=rank[a] - rank[b],
                         wins_for_gate_off=int((y < x).sum()),
                         losses=int((y > x).sum()), p=float(p),
                         delta=per_function_delta(df, a, b, funcs)))
        praw.append(p)
    for r, pa in zip(rows, holm(praw)):
        r["p_holm"] = float(pa)
        r["significant"] = bool(pa < 0.05)

    print(f"\n    {'effect':<28}{'rank change':>12}{'W/L':>8}{'p_Holm':>10}"
          f"{'delta':>9}  sig")
    for r in rows:
        print(f"    {r['effect']:<28}{r['rank_change']:>+12.2f}"
              f"{r['wins_for_gate_off']:>5d}/{r['losses']:<3d}{r['p_holm']:>10.3f}"
              f"{r['delta']:>+9.3f}  {'YES' if r['significant'] else 'no'}")

    g_gated = rows[0]["rank_change"]
    g_off = rows[1]["rank_change"]
    print(f"\n    gate effect with Levy gated : {g_gated:+.2f} rank")
    print(f"    gate effect with Levy off   : {g_off:+.2f} rank")
    print(f"    interaction (difference)    : {g_gated - g_off:+.2f} rank")
    out = pd.DataFrame(rows)
    out.insert(0, "block", label)
    return out


def main():
    parts = []
    for label, fname in BLOCKS:
        p = os.path.join(RES, fname)
        if not os.path.exists(p):
            print(f"  [{label}] {fname} not found — skipped")
            continue
        df = pd.read_csv(p)
        if "dim" in df.columns:
            df = df[df.dim == 30]
        r = analyse(df, label)
        if r is not None:
            parts.append(r)
    if parts:
        out = os.path.join(RES, "analysis", "gate_levy_factorial.csv")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        pd.concat(parts).to_csv(out, index=False)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
