"""Inferential tests for the gate ablation: interactions and equivalence.

Three obligations that a point estimate plus a pair of p-values does not
discharge, and that are met here by re-analysing data already collected:

1. **Interaction with Levy.**  "The gate effect is the same with and without
   Levy" is a claim about a *difference of effects*.  Reporting one effect as
   significant and the other as not does not establish it (Gelman & Stern,
   2006).  We form a per-function difference-in-differences and test it.

2. **Interaction with budget.**  Same error, same remedy: the claim that the
   gate matters more at 300,000 evaluations than at 15,000 needs a test of the
   difference between the two gate effects, not two separate tests.

3. **Equivalence for Levy.**  p = 0.958 is not evidence of no effect.  To say
   the Levy perturbation cannot account for the gate result we need an
   equivalence procedure: a bootstrap confidence interval on the effect size
   that lies entirely inside a pre-declared negligible margin.

The effect measure throughout is **Cliff's delta computed per function** over
the 30 runs, which is scale-free and therefore commensurable across functions
and across budgets whose error magnitudes differ by orders of magnitude.
Sign convention: delta(A, B) < 0 means A tends to produce the smaller error.

Margin for equivalence: |delta| < 0.147, the Romano et al. (2006) threshold
below which an effect is conventionally called negligible.  Declared here
rather than chosen after seeing the interval, and applied to *every* contrast
on the same footing -- gate, Levy and interaction alike -- so that the reading
"no effect worth having" is available or refused by one uniform rule.

The interval reported is a 95% percentile bootstrap interval.  A two-one-sided
test at the 5% level would use the 90% interval, so requiring the 95% interval
to lie inside the margin is deliberately conservative: it implies TOST
significance at 2.5% on each side.

Run:  python3 scripts/analyze_gate_inference.py
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sehho.stats import cliffs_delta, holm

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
NEGLIGIBLE = 0.147
BOOT = 20000
RNG = np.random.default_rng(20260913)

CELLS = {("on", "gated"): "SEHHO:Full",
         ("on", "off"): "SEHHO:NoLevy",
         ("off", "gated"): "SEHHO:AlwaysPbest",
         ("off", "off"): "SEHHO:AlwaysPbest+NoLevy"}
BLOCKS = [("15,000 FEs", "E4c_ablation_cec2017_30D_15k.csv"),
          ("300,000 FEs", "E4b_ablation_cec2017_30D.csv")]


def load(fname):
    df = pd.read_csv(os.path.join(RES, fname))
    if "dim" in df.columns:
        df = df[df.dim == 30]
    return df


def per_function_delta(df, a, b, funcs):
    """delta_f(a vs b) for each function; negative means `a` is better."""
    out = []
    for f in funcs:
        x = df[(df.func == f) & (df.algo == a)]["error"].to_numpy()
        y = df[(df.func == f) & (df.algo == b)]["error"].to_numpy()
        out.append(cliffs_delta(x, y))
    return np.asarray(out)


def boot_ci(v, level=0.95):
    idx = RNG.integers(0, len(v), (BOOT, len(v)))
    means = v[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def signed_rank(v):
    nz = np.abs(v) > 0
    if nz.sum() == 0:
        return 1.0
    return float(sps.wilcoxon(v[nz])[1])


def report(name, v, equivalence=True):
    lo, hi = boot_ci(v)
    p = signed_rank(v)
    line = (f"  {name:<44s} mean {v.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
            f"  p={p:.4f}")
    if equivalence:
        inside = (lo > -NEGLIGIBLE) and (hi < NEGLIGIBLE)
        line += f"   equivalent at |d|<{NEGLIGIBLE}: {'YES' if inside else 'NO'}"
    print(line)
    return dict(quantity=name, mean=float(v.mean()), ci_lo=lo, ci_hi=hi, p=p,
                n=len(v),
                equivalent=(bool((lo > -NEGLIGIBLE) and (hi < NEGLIGIBLE))
                            if equivalence else None))


def main():
    rows = []
    gate_delta = {}

    for label, fname in BLOCKS:
        df = load(fname)
        if not all(a in set(df.algo) for a in CELLS.values()):
            print(f"[{label}] missing cells — skipped")
            continue
        funcs = sorted(df.func.unique())
        print(f"\n=== CEC2017 D=30, {label} ({len(funcs)} functions) ===")

        # gate effect at each Levy setting: delta(gate-off vs gate-on)
        g_gated = per_function_delta(df, CELLS[("off", "gated")], CELLS[("on", "gated")], funcs)
        g_off = per_function_delta(df, CELLS[("off", "off")], CELLS[("on", "off")], funcs)
        gate_delta[label] = dict(gated=g_gated, off=g_off, funcs=funcs)

        print("  -- gate effect (negative favours removing the gate)")
        r = report("gate effect, Levy gated", g_gated, True); r["block"] = label; rows.append(r)
        r = report("gate effect, Levy off", g_off, True); r["block"] = label; rows.append(r)

        # Levy effect at each gate setting, with an equivalence verdict
        l_on = per_function_delta(df, CELLS[("on", "off")], CELLS[("on", "gated")], funcs)
        l_off = per_function_delta(df, CELLS[("off", "off")], CELLS[("off", "gated")], funcs)
        print("  -- Levy effect (equivalence tested against the negligible margin)")
        r = report("Levy effect, gate on", l_on, True); r["block"] = label; rows.append(r)
        r = report("Levy effect, gate off", l_off, True); r["block"] = label; rows.append(r)

        # interaction: does the gate effect depend on the Levy setting?
        print("  -- gate x Levy interaction (difference of the two gate effects)")
        r = report("interaction gate x Levy", g_gated - g_off, True)
        r["block"] = label
        rows.append(r)

    # interaction with budget: does the gate effect differ between budgets?
    if len(gate_delta) == 2:
        a, b = BLOCKS[1][0], BLOCKS[0][0]        # 300,000 minus 15,000
        fa, fb = gate_delta[a]["funcs"], gate_delta[b]["funcs"]
        assert fa == fb, "function sets differ between budgets"
        print(f"\n=== gate x budget interaction ({a} minus {b}) ===")
        for lev in ("gated", "off"):
            d = gate_delta[a][lev] - gate_delta[b][lev]
            r = report(f"interaction gate x budget, Levy {lev}", d, True)
            r["block"] = f"{a} - {b}"
            rows.append(r)

    out = pd.DataFrame(rows)
    # Holm within each coherent family of tests, not across families
    out["family"] = np.where(out.quantity.str.startswith("interaction"), "interaction",
                    np.where(out.quantity.str.startswith("gate effect"), "gate effect",
                             "Levy effect"))
    out["p_holm"] = np.nan
    for fam, idx in out.groupby("family").groups.items():
        out.loc[idx, "p_holm"] = holm(out.loc[idx, "p"].to_numpy())
    print("\n=== Holm-corrected within family ===")
    for fam in ("gate effect", "Levy effect", "interaction"):
        sub = out[out.family == fam]
        print(f"  {fam} (n={len(sub)})")
        for _, r in sub.iterrows():
            eq = ("" if r.equivalent is None or (isinstance(r.equivalent, float) and np.isnan(r.equivalent))
                  else ("  [equivalent]" if r.equivalent else "  [NOT equivalent]"))
            print(f"    {r.block:<22s} {r.quantity:<40s} p={r.p:.4f}  "
                  f"p_Holm={r.p_holm:.4f}{eq}")
    path = os.path.join(RES, "analysis", "gate_inference.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path, index=False)
    print(f"\nwrote {path}")
    print("\nNOTE: a non-significant interaction is not by itself evidence of no "
          "interaction; the equivalence column is what licenses that reading, and "
          "only where it says YES.")


if __name__ == "__main__":
    main()
