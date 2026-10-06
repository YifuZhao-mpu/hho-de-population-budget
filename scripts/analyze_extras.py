"""Numbers the paper quotes in its text that no table, and no other script, computes.

Each value is recomputed from the raw per-run results in results/*.csv and
written, one row per quantity, to results/analysis/extras.csv:

  (a) The suite gap of the primary endpoint.  The pre-registered contrast,
      R-SHADE against SEHHO-COBL at the competition budget, is estimated on
      CEC2014 (stage E13) and on CEC2017, the suite on which the repair was
      diagnosed (E3 + E9).  The difference of the two mean per-function
      deltas, CEC2017 minus CEC2014, gets a 95% percentile bootstrap interval
      in which the functions of each suite are resampled independently (the
      two suites have different functions, so there is nothing to pair).
  (b) RC01 feasibility at 15,000 evaluations: feasible runs of R-SHADE (E10)
      against SEHHO-COBL (E5), 30 runs each, Fisher's exact test, two-sided.
  (c) The schedule-basis control of the population-rule search (E7, the
      table "Choosing the population rule on the design suite"): fixed N=120
      with the budget-based schedule (N120b) against fixed N=120 (N120), in
      each of the four blocks, and whether the whole interval lies inside the
      negligible band |delta| < 0.147.
  (d) The population effect on CEC2017 at D=30 with 300,000 evaluations (E6):
      the published N=30 against N=120.  For context, the gate effect of the
      same block is copied from results/analysis/gate_inference.csv (written
      by analyze_gate_inference.py) with its sign turned to the same
      convention, together with the ratio of the two effects.
  (e) The configuration the original composition rule gives (E8 "-pAnneal",
      the configuration of stage E15) against R-SHADE (E11 "Stripped") on the
      design suite, CEC2022, in the four blocks of analyze_composition.py.
  (f) HHO on the CEC2014 composition functions F23-F30 (E13, both dimensions
      and both budgets): the share of runs whose error is within 0.01 of 200,
      the error of the origin, which the all-zero shift of the third
      component makes reachable.

Conventions, those of analyze_cec2014.py, analyze_poprule.py,
analyze_composition.py and sehho/stats.py:
  * cliffs_delta(a, b) per function over the 30 runs of each arm (negative
    means `a` has the smaller errors), averaged over the functions;
  * 95% percentile bootstrap over functions, 20,000 resamples drawn as
    rng.integers(0, n, (20000, n)), numpy's default quantile;
  * negligible = the whole interval lies inside |delta| < 0.147;
  * each section draws from one generator seeded like the script whose
    analysis it extends, and consumes it in that script's block order:
    analyze_cec2014.py (20260915) for (a), analyze_poprule.py (20260914) for
    (c), analyze_composition.py (20260914) for (e).  (d) extends no script
    and uses 20260915.  Every section reproduces on its own.

Monte Carlo check: every interval is recomputed with 20 further generators
spawned from the section's seed.  The ranges of the two bounds over these and
the reported interval are written to the CSV, and `mc_stable` says whether the
interval's side of zero (and, in (c) and (e), its negligible-band verdict) is
the same for all of them.  An unstable verdict is a property of the data, not
of the seed: no seed should be chosen to settle it.

Signs: (a) negative = the CEC2017 estimate is the more favourable to R-SHADE;
(c) negative favours N120b; (d) positive favours N=120 (delta of the published
setting against the change, the convention of the ablation table), and the
gate effect is stated the same way (positive favours deleting the gate);
(e) positive favours R-SHADE (delta of the changed configuration against the
reference, the convention of analyze_composition.py).

make_tables.py calls suite_gap() and poprule_basis_control() for the notes of
the primary-endpoint and population-rule tables, so the tables and this file
cannot disagree.

Run:  python3 scripts/analyze_extras.py [--out DIR]
      (default DIR: results/analysis)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sehho.stats import cliffs_delta

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
ERROR_DECIMALS = int(os.environ["SEHHO_ERROR_DECIMALS"]) if os.environ.get("SEHHO_ERROR_DECIMALS") else None
METHOD = "GF-Method"          # R-SHADE in the paper
PUBLISHED = "SEHHO-COBL"
BOOT = 20000
SEED_CEC2014 = 20260915       # analyze_cec2014.py
SEED_POPRULE = 20260914       # analyze_poprule.py
SEED_COMPOSITION = 20260914   # analyze_composition.py
SEED_DEFAULT = 20260915       # (d), which extends no existing script
MC_REPLICATES = 20
NEGLIGIBLE = 0.147
RC01 = "P3_heat_exchanger_RC01"
ZERO_SHIFT_ERROR, ZERO_SHIFT_TOL = 200.0, 0.01
COLUMNS = ["item", "quantity", "block", "dim", "max_fes", "estimate", "ci_lo", "ci_hi",
           "p", "negligible", "n", "seed", "ci_lo_min", "ci_lo_max", "ci_hi_min",
           "ci_hi_max", "mc_stable", "detail"]


def _read(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found")
    df = pd.read_csv(path)
    # SEHHO_ERROR_DECIMALS is set only by scripts/resolution_check.py, which re-runs this
    # script on a copy of the package with every per-run error rounded (main text
    # Section 2.6); when it is unset nothing changes.
    if ERROR_DECIMALS is not None and not name.startswith("analysis") and "error" in df.columns:
        df["error"] = np.round(df["error"].to_numpy(dtype=float), ERROR_DECIMALS)
    return df


def per_fn_delta(sub, a, b, col="algo"):
    """delta(a, b) per function, functions in ascending order; `col` names the arm."""
    out = []
    for f in sorted(sub.func.unique()):
        x = sub[(sub.func == f) & (sub[col] == a)]["error"].to_numpy()
        y = sub[(sub.func == f) & (sub[col] == b)]["error"].to_numpy()
        if not len(x) or not len(y):
            raise SystemExit(f"function {f}: no runs for {a if not len(x) else b}")
        out.append(cliffs_delta(x, y))
    return np.array(out)


def boot_means(v, rng):
    return v[rng.integers(0, len(v), (BOOT, len(v)))].mean(axis=1)


def pct_ci(m, level=0.95):
    return (float(np.quantile(m, (1 - level) / 2)),
            float(np.quantile(m, 1 - (1 - level) / 2)))


def _mc(row, draw, seed, band=False):
    """Add the Monte Carlo spread of the row's interval (see the module docstring).

    `draw(rng)` returns the bootstrap distribution; the replicates use their
    own generators, so the reported interval is unaffected by this check.
    """
    reps = [pct_ci(draw(np.random.default_rng(s)))
            for s in np.random.SeedSequence(seed).spawn(MC_REPLICATES)]
    reps.append((row["ci_lo"], row["ci_hi"]))
    side = {"above" if lo > 0 else "below" if hi < 0 else "spans" for lo, hi in reps}
    inside = {bool(lo > -NEGLIGIBLE and hi < NEGLIGIBLE) for lo, hi in reps}
    row.update(ci_lo_min=min(r[0] for r in reps), ci_lo_max=max(r[0] for r in reps),
               ci_hi_min=min(r[1] for r in reps), ci_hi_max=max(r[1] for r in reps),
               mc_stable=bool(len(side) == 1 and (not band or len(inside) == 1)))
    return row


# ------------------------------------------------------------------ (a)
def suite_gap():
    """CEC2017 minus CEC2014 mean delta of R-SHADE against SEHHO-COBL, competition budget."""
    c17 = pd.concat([_read("E3_cec2017.csv"),
                     _read("E9_newmethod_vs_baselines.csv").query("tag == 'E3_cec2017'")],
                    ignore_index=True)
    c14 = _read("E13_cec2014.csv")
    rng = np.random.default_rng(SEED_CEC2014)
    rows = []
    for dim in (30, 50):
        fes = 10_000 * dim
        v17 = per_fn_delta(c17[(c17.dim == dim) & (c17.max_fes == fes)], METHOD, PUBLISHED)
        v14 = per_fn_delta(c14[(c14.dim == dim) & (c14.max_fes == fes)], METHOD, PUBLISHED)
        m17 = boot_means(v17, rng)          # each suite resampled on its own
        m14 = boot_means(v14, rng)
        lo, hi = pct_ci(m17 - m14)
        rows.append(_mc(dict(
            item="a", quantity="suite_gap_cec2017_minus_cec2014",
            block=f"D={dim}, {fes:,} FEs", dim=dim, max_fes=fes,
            estimate=float(v17.mean() - v14.mean()), ci_lo=lo, ci_hi=hi,
            includes_zero=bool(lo <= 0 <= hi), cec2017=float(v17.mean()),
            cec2014=float(v14.mean()),       # (these three are not CSV columns)
            n=f"{len(v17)}+{len(v14)}", seed=SEED_CEC2014,
            detail=(f"R-SHADE vs SEHHO-COBL, mean per-function Cliff's delta: CEC2017 "
                    f"{v17.mean():+.4f} ({len(v17)} functions, E3+E9), CEC2014 "
                    f"{v14.mean():+.4f} ({len(v14)} functions, E13); difference CEC2017 "
                    f"minus CEC2014, negative = the CEC2017 estimate is the more "
                    f"favourable to R-SHADE; the two suites resampled independently")),
            lambda g, a=v17, b=v14: boot_means(a, g) - boot_means(b, g), SEED_CEC2014))
    return rows


# ------------------------------------------------------------------ (b)
def rc01_fisher():
    """Feasible runs on RC01 at 15,000 evaluations, R-SHADE against SEHHO-COBL."""
    e5, e10 = _read("E5_engineering.csv"), _read("E10_newmethod_engineering.csv")
    a = e10[(e10.problem == RC01) & (e10.algo == METHOD)].feasible.astype(bool)
    b = e5[(e5.problem == RC01) & (e5.algo == PUBLISHED)].feasible.astype(bool)
    table = [[int(a.sum()), int((~a).sum())], [int(b.sum()), int((~b).sum())]]
    odds, p = sps.fisher_exact(table, alternative="two-sided")
    fes = int(e10[e10.problem == RC01].max_fes.iloc[0])
    return [dict(
        item="b", quantity="rc01_feasibility_fisher", block=f"RC01, {fes:,} FEs",
        max_fes=fes, estimate=float(a.mean() - b.mean()), p=float(p),
        n=f"{len(a)}+{len(b)}", counts=table,   # (counts is not a CSV column)
        detail=(f"feasible runs: R-SHADE {table[0][0]}/{len(a)} (E10), SEHHO-COBL "
                f"{table[1][0]}/{len(b)} (E5); estimate = difference in feasibility rate, "
                f"R-SHADE minus SEHHO-COBL; odds ratio {odds:.3f}; Fisher's exact test, "
                f"two-sided"))]


# ------------------------------------------------------------------ (c)
def poprule_basis_control():
    """N120b against N120 in each block of the population-rule search (E7, CEC2022)."""
    d = _read("E7_poprule_cec2022.csv")
    # candidate and block are parsed exactly as analyze_poprule.py parses them
    d["cand"] = d.algo.str.slice(4).str.split("@").str[0]
    d["block"] = d.algo.str.split("@").str[1]
    rng = np.random.default_rng(SEED_POPRULE)
    rows = []
    for dim in sorted(d.dim.unique()):
        for blk in ("matched", "competition"):
            sub = d[(d.dim == dim) & (d.block == blk)]
            if sub.empty:
                continue
            fes = int(sub.max_fes.iloc[0])
            v = per_fn_delta(sub, "N120b", "N120", col="cand")
            lo, hi = pct_ci(boot_means(v, rng))
            rows.append(_mc(dict(
                item="c", quantity="n120b_vs_n120", block=f"D={dim}, {fes:,} FEs",
                dim=int(dim), max_fes=fes, estimate=float(v.mean()), ci_lo=lo, ci_hi=hi,
                negligible=bool(lo > -NEGLIGIBLE and hi < NEGLIGIBLE), n=str(len(v)),
                seed=SEED_POPRULE,
                detail=(f"Cliff's delta of fixed N=120 with the budget-based schedule "
                        f"(N120b) against fixed N=120 (N120), {len(v)} CEC2022 functions; "
                        f"negative favours N120b; negligible = the whole interval lies "
                        f"inside |delta|<{NEGLIGIBLE}")),
                lambda g, v=v: boot_means(v, g), SEED_POPRULE, band=True))
    return rows


# ------------------------------------------------------------------ (d)
def population_effect():
    """N=30 against N=120 on CEC2017, D=30, 300,000 evaluations (E6)."""
    e6 = _read("E6_popsize_cec2017_30D.csv")
    e6 = e6[(e6.dim == 30) & (e6.max_fes == 300_000)]
    v = per_fn_delta(e6, "E6_popsize_N30", "E6_popsize_N120", col="tag")
    rng = np.random.default_rng(SEED_DEFAULT)
    lo, hi = pct_ci(boot_means(v, rng))
    blk = "CEC2017, D=30, 300,000 FEs"
    rows = [_mc(dict(
        item="d", quantity="e6_n30_vs_n120", block=blk, dim=30, max_fes=300_000,
        estimate=float(v.mean()), ci_lo=lo, ci_hi=hi, n=str(len(v)), seed=SEED_DEFAULT,
        detail=(f"Cliff's delta of the published N=30 against N=120 (SEHHO-COBL, E6), "
                f"{len(v)} functions; positive favours N=120")),
        lambda g: boot_means(v, g), SEED_DEFAULT)]
    gi = os.path.join(RES, "analysis", "gate_inference.csv")
    if os.path.exists(gi):
        g = pd.read_csv(gi)
        g = g[(g.quantity == "gate effect, Levy gated") & (g.block == "300,000 FEs")]
        if len(g):
            g = g.iloc[0]
            gate = -float(g["mean"])
            rows.append(dict(
                item="d", quantity="gate_effect_reference", block=blk, dim=30,
                max_fes=300_000, estimate=gate, ci_lo=-float(g.ci_hi), ci_hi=-float(g.ci_lo),
                p=float(g.p), n=str(int(g.n)),
                detail=("gate effect with the Levy perturbation gated, copied from "
                        "results/analysis/gate_inference.csv (analyze_gate_inference.py) "
                        "with its sign reversed: positive favours deleting the gate")))
            rows.append(dict(
                item="d", quantity="population_over_gate_ratio", block=blk, dim=30,
                max_fes=300_000, estimate=float(v.mean()) / gate,
                detail="e6_n30_vs_n120 divided by gate_effect_reference"))
    return rows


# ------------------------------------------------------------------ (e)
def original_rule_vs_rshade_cec2022():
    """The E15 configuration (E8 "-pAnneal") against R-SHADE (E11 "Stripped") on CEC2022."""
    d = pd.concat([_read("E8_composition_cec2022.csv"), _read("E11_stripped_cec2022.csv")],
                  ignore_index=True)
    # candidate and block are parsed exactly as analyze_composition.py parses them
    d["cand"] = d.algo.str.slice(3).str.split("@").str[0]
    d["block"] = d.algo.str.split("@").str[1]
    rng = np.random.default_rng(SEED_COMPOSITION)
    rows = []
    for dim in sorted(d.dim.unique()):
        for blk in ("matched", "competition"):
            sub = d[(d.dim == dim) & (d.block == blk)]
            if sub.empty:
                continue
            fes = int(sub.max_fes.iloc[0])
            v = per_fn_delta(sub, "-pAnneal", "Stripped", col="cand")
            lo, hi = pct_ci(boot_means(v, rng))
            rows.append(_mc(dict(
                item="e", quantity="original_rule_config_vs_rshade",
                block=f"CEC2022, D={dim}, {fes:,} FEs", dim=int(dim), max_fes=fes,
                estimate=float(v.mean()), ci_lo=lo, ci_hi=hi,
                negligible=bool(lo > -NEGLIGIBLE and hi < NEGLIGIBLE), n=str(len(v)),
                seed=SEED_COMPOSITION,
                detail=(f"Cliff's delta of the configuration the original composition "
                        f"rule gives (E8 'GF:-pAnneal', the E15 configuration) against "
                        f"R-SHADE (E11 'GF:Stripped'), {len(v)} CEC2022 functions; positive "
                        f"favours R-SHADE; delta direction, seed and block order of "
                        f"analyze_composition.py")),
                lambda g, v=v: boot_means(v, g), SEED_COMPOSITION, band=True))
    return rows


# ------------------------------------------------------------------ (f)
def hho_zero_shift():
    """HHO runs on CEC2014 F23-F30 that end within 0.01 of the error of the origin (200)."""
    e13 = _read("E13_cec2014.csv")
    h = e13[(e13.algo == "HHO") & e13.func.between(23, 30)].copy()
    h["near"] = (h["error"] - ZERO_SHIFT_ERROR).abs() <= ZERO_SHIFT_TOL
    rows = []
    for f in sorted(h.func.unique()):
        s = h[h.func == f]
        cells = s.groupby(["dim", "max_fes"]).near.mean()
        rows.append(dict(
            item="f", quantity="hho_share_within_0.01_of_200",
            block=f"CEC2014 F{f}, D=30 and 50, both budgets",
            estimate=float(s.near.mean()), n=str(len(s)),
            cells=cells,                         # (not a CSV column)
            detail=(f"share of HHO runs (E13) with |error - {ZERO_SHIFT_ERROR:g}| <= "
                    f"{ZERO_SHIFT_TOL:g}, pooled over the four cells; per cell: "
                    + ", ".join(f"D={dm}/{fes:,} {v:.3f}" for (dm, fes), v in cells.items()))))
    return rows


# ------------------------------------------------------------------ output
def _fmt_ci(r):
    return f"{r['estimate']:+.4f} [{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]"


def _mc_note(r):
    """A warning line when the interval's verdict depends on the bootstrap seed."""
    if "mc_stable" not in r or r["mc_stable"]:
        return ""
    return (f"\n    {'':<20s} ** the verdict changes with the bootstrap seed: over "
            f"{MC_REPLICATES + 1} seeds the lower bound runs {r['ci_lo_min']:+.4f} to "
            f"{r['ci_lo_max']:+.4f} and the upper {r['ci_hi_min']:+.4f} to "
            f"{r['ci_hi_max']:+.4f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Numbers quoted in the text that no table holds.")
    ap.add_argument("--out", default=os.path.join(RES, "analysis"),
                    help="directory for extras.csv (default: results/analysis)")
    args = ap.parse_args(argv)

    a, b, c = suite_gap(), rc01_fisher(), poprule_basis_control()
    d, e, f = population_effect(), original_rule_vs_rshade_cec2022(), hho_zero_shift()

    print("(a) Primary endpoint, CEC2017 minus CEC2014 (R-SHADE vs SEHHO-COBL, "
          "competition budget)")
    for r in a:
        print(f"    {r['block']:<20s} CEC2017 {r['cec2017']:+.4f}, CEC2014 {r['cec2014']:+.4f}"
              f"  difference {_fmt_ci(r)}  interval "
              f"{'includes' if r['includes_zero'] else 'EXCLUDES'} zero" + _mc_note(r))
    r = b[0]
    (fa, ia), (fb, ib) = r["counts"]
    print(f"\n(b) RC01 feasibility, {r['block'].split(', ')[1]}")
    print(f"    R-SHADE {fa}/{fa + ia} feasible, SEHHO-COBL {fb}/{fb + ib}; "
          f"Fisher's exact test, two-sided: p = {r['p']:.4f}")
    print("\n(c) Schedule-basis control, N120b vs N120 (E7, CEC2022; negative favours N120b)")
    for r in c:
        print(f"    {r['block']:<20s} {_fmt_ci(r)}  "
              f"{'inside' if r['negligible'] else 'NOT inside'} the negligible band"
              + _mc_note(r))
    print(f"    inside |delta|<{NEGLIGIBLE} in {sum(r['negligible'] for r in c)} "
          f"of {len(c)} blocks")
    print("\n(d) Population effect, CEC2017 D=30, 300,000 FEs (E6; positive favours N=120)")
    print(f"    N=30 vs N=120        {_fmt_ci(d[0])}" + _mc_note(d[0]))
    if len(d) > 1:
        print(f"    gate effect (ref.)   {_fmt_ci(d[1])}   (gate_inference.csv, sign reversed)")
        print(f"    ratio                {d[2]['estimate']:.2f}")
    else:
        print("    (results/analysis/gate_inference.csv not found: run "
              "analyze_gate_inference.py for the gate comparison)")
    print("\n(e) Original-rule configuration (E8 -pAnneal = E15) vs R-SHADE (E11 Stripped), "
          "CEC2022; positive favours R-SHADE")
    for r in e:
        side = ("excludes zero" if r["ci_lo"] > 0 or r["ci_hi"] < 0 else "includes zero")
        print(f"    {r['block'][9:]:<20s} {_fmt_ci(r)}  {side}; "
              f"{'inside' if r['negligible'] else 'NOT inside'} the negligible band"
              + _mc_note(r))
    print(f"\n(f) HHO on CEC2014 F23-F30 (E13): runs ending within {ZERO_SHIFT_TOL:g} of "
          f"{ZERO_SHIFT_ERROR:g}, pooled over D=30/50 and both budgets")
    for r in f:
        cells = r["cells"]
        print(f"    {r['block'].split(',')[0][8:]:<4s} {100 * r['estimate']:5.1f}% of "
              f"{r['n']} runs  (per cell {100 * cells.min():.1f}-{100 * cells.max():.1f}%)")

    out = pd.DataFrame(a + b + c + d + e + f, columns=COLUMNS)
    for col in ("dim", "max_fes", "seed"):
        out[col] = out[col].astype("Int64")
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "extras.csv")
    out.to_csv(path, index=False)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
