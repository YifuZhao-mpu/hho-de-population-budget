"""Re-analyses requested in an internal pre-submission review of the manuscript
(its roadmap items REV-xx and the revision plan's work orders W1-W11; neither
refers to a journal's review -- see the README, "The word 'review' in file
comments").

Nothing here runs an optimiser: every quantity is recomputed from the per-run
CSVs in results/ and written to results/analysis/revision_*.csv, one file per
section, and every number a writer may quote is printed.

Sections (run all, or name some:  python3 scripts/analyze_revision.py gate pop)

  gate      W1 + REV-24  gate x Levy factorial (E4b, E4c): percentile, BCa and
                         Student-t intervals, Bonferroni-adjusted versions within
                         each Holm family of four, p_Holm on both scales, and
                         equivalence verdicts at |delta| < 0.05 / 0.10 / 0.147.
  pop       W1           one-constant population control (E14 + the shipped
                         configuration's runs): all intervals, adjusted within
                         the four blocks of one row and budget.
  cec2014   W1, W3, REV-38, REV-29
                         the registered primary endpoint with BCa / Student-t /
                         two-block Bonferroni intervals; the same endpoint
                         without F18, F22, F23, F24 (post hoc, requested in
                         the internal pre-submission review); the CEC2017 counterpart and the suite gap;
                         Holm-corrected signed-rank tests on per-function delta
                         for every pairwise comparison; per-function standard
                         errors of delta at 30 runs and the two-stage bootstrap.
  suites    W2, REV-29   the same pairwise statistics for CEC2017 (E3+E9, E12)
                         and CEC2022 (E1+E9, E2+E9).
  comp      W1, REV-26   composition of the repaired configuration (E8, E11) and
                         the re-inserted gate, with BCa / Student-t / adjusted
                         intervals and the margin sensitivity of each decision.
  poprule   W1           the population-rule search (E7): intervals against the
                         selected rule.
  popsize   W9, REV-32   delta(N vs N=30) for N = 60, 120, 270, 540 (E6).
  e16       W4, REV-09   E16: success rate against the best-known value, mean and
                         median among feasible runs, with the registered delta.
  eng       W5, REV-53   success rates of the seven algorithms (E5, E10) by the
                         same rule; runs that sit at the feasibility threshold.
  guide     W11, REV-08  rows of the budget-indexed guide (Table 10, Figure 3).
  hho       REV-57       SEHHO-COBL against HHO in every block.
  e15                    E15's registered secondary comparison with BCa, Student-t and
                         Bonferroni-adjusted intervals over its four blocks (Table S-e15).
  contrast               the direct contrast of the population effect (E6, N=120 vs
                         N=30) with the gate effect (E4b), main text Sections 3.2 and 5.
  resolution             the precision convention of main text Section 2.6 (post hoc,
                         added at a final integrity check): the headline contrasts with
                         delta computed at full double precision and with errors rounded
                         to 1e-8, and whether any interval's side or tie flag changes.

Conventions (those of sehho/stats.py and the existing analysis scripts):
  * delta is reported for arm A against arm B, A the changed, new or
    first-named arm; negative favours A (REV-13).  Where a source file uses
    the other order, the sign is reversed here and the row says so.
  * per-function Cliff's delta over the runs, averaged over functions;
    20,000 bootstrap resamples over functions drawn as
    rng.integers(0, n, (20000, n)).
  * Every section that re-analyses an existing analysis draws from a generator
    seeded and consumed exactly as that script's, so the percentile intervals
    it prints reproduce the stored ones; the BCa interval of a contrast is read
    from the same resamples.  The script checks this and prints any mismatch.
    Seeds: 20260913 (analyze_gate_inference.py), 20260914 (analyze_poprule.py,
    analyze_composition.py), 20260915 (analyze_cec2014.py,
    analyze_pop_isolation.py, analyze_extras.py), 20261003 (contrasts that no
    earlier script computed).
  * Inference is conditional on a suite's functions treated as exchangeable
    instances of the class the suite represents.

Run:  python3 scripts/analyze_revision.py [section ...]
"""

import contextlib
import importlib.util
import io
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from sehho.stats import cliffs_delta, holm, friedman, friedman_holm_vs_control  # noqa: E402
from sehho.intervals import (BOOT, boot_indices, boot_means, percentile_ci, bca_ci,  # noqa: E402
                             t_ci, all_intervals, margin_verdict, side, near_zero_bound,
                             signed_rank, cliff_var_unbiased, two_stage_bootstrap,
                             adjusted_level)

RES = os.path.join(HERE, "..", "results")
ERROR_DECIMALS = int(os.environ["SEHHO_ERROR_DECIMALS"]) if os.environ.get("SEHHO_ERROR_DECIMALS") else None
OUT = os.path.join(RES, "analysis")
METHOD = "GF-Method"            # SEHHO-COBL-R in the paper (seeds hash this name)
PUBLISHED = "SEHHO-COBL"
MARGINS = (0.05, 0.10, 0.147)
SEED_GATE = 20260913
SEED_POPRULE = 20260914
SEED_CEC2014 = 20260915
SEED_REVISION = 20261003
EXCLUDED_CEC2014 = (18, 22, 23, 24)   # share recipes with CEC2022 F6, F8, F9, F10 (REV-38)
SUCCESS_TOL = 1e-4                    # success: feasible and (f - f*)/|f*| <= 1e-4;
                                      # the suite's own rule (f - f* <= 1e-8) is reported too
COMPETITORS = ["jSO", "LSHADE", "SHADE", "DE", "SEHHO-COBL", "HHO"]
DISPLAY = {METHOD: "SEHHO-COBL-R", "LSHADE": "L-SHADE"}
CHECKS = []                           # (what, ok, detail) of every reproduction check


def _read(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found")
    df = pd.read_csv(path)
    # SEHHO_ERROR_DECIMALS is set only by scripts/resolution_check.py, which re-runs this
    # script on a copy of the package with every per-run error rounded (main text
    # Section 2.6); when it is unset nothing changes.
    if ERROR_DECIMALS is not None and not name.startswith("analysis/") and "error" in df.columns:
        df["error"] = np.round(df["error"].to_numpy(dtype=float), ERROR_DECIMALS)
    return df


def _write(df, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    df.to_csv(path, index=False)
    print(f"  -> wrote {os.path.relpath(path, os.path.join(HERE, '..'))}")
    return path


def per_fn_delta(sub, a, b, funcs=None, col="algo", decimals=None):
    """delta(a, b) per function in ascending function order; negative favours a.
    `decimals` rounds the errors first (used only by the `resolution` section)."""
    funcs = sorted(sub.func.unique()) if funcs is None else funcs
    out = []
    for f in funcs:
        x = sub[(sub.func == f) & (sub[col] == a)]["error"].to_numpy()
        y = sub[(sub.func == f) & (sub[col] == b)]["error"].to_numpy()
        if not len(x) or not len(y):
            raise SystemExit(f"function {f}: no runs for {a if not len(x) else b}")
        if decimals is not None:
            x, y = np.round(x, decimals), np.round(y, decimals)
        out.append(cliffs_delta(x, y))
    return np.asarray(out)


def _check(what, got, want, tol=1e-12):
    ok = bool(np.all(np.abs(np.asarray(got, float) - np.asarray(want, float)) <= tol))
    CHECKS.append((what, ok, f"got {np.round(got, 6)} stored {np.round(want, 6)}"))
    return ok


def _verdicts(r, prefix=""):
    """Equivalence verdicts at every margin, for the percentile, BCa and t intervals."""
    out = {}
    for kind in ("pct", "bca", "t"):
        lo, hi = r[f"{kind}_lo"], r[f"{kind}_hi"]
        for m in MARGINS:
            out[f"{prefix}verdict_{kind}_{m:g}"] = margin_verdict(lo, hi, m)
        out[f"{prefix}side_{kind}"] = side(lo, hi)
        adj_lo, adj_hi = r[f"{kind}_adj_lo"], r[f"{kind}_adj_hi"]
        out[f"{prefix}side_{kind}_adj"] = side(adj_lo, adj_hi)
    # the verdict that holds under all three intervals (conservative reading)
    for m in MARGINS:
        vs = {out[f"{prefix}verdict_{k}_{m:g}"] for k in ("pct", "bca", "t")}
        out[f"{prefix}verdict_all_{m:g}"] = vs.pop() if len(vs) == 1 else "inconclusive"
    sides = {out[f"{prefix}side_{k}"] for k in ("pct", "bca", "t")}
    out[f"{prefix}side_all"] = sides.pop() if len(sides) == 1 else "spans"
    adj = {out[f"{prefix}side_{k}_adj"] for k in ("pct", "bca", "t")}
    out[f"{prefix}side_all_adj"] = adj.pop() if len(adj) == 1 else "spans"
    out[f"{prefix}tie_002"] = any(near_zero_bound(r[f"{k}_lo"], r[f"{k}_hi"])
                                  for k in ("pct", "bca", "t"))
    return out


def excludes_zero_tie(r, kinds=("pct", "bca", "t"), tol=0.02):
    """The tie rule of Section 2.6: some unadjusted interval (percentile, BCa or Student-t)
    excludes zero with its bound nearest zero within `tol` of it.  (`tie_002` in the
    output files is broader: it also flags an interval that spans zero with a bound
    within `tol` of zero.)"""
    for k in kinds:
        lo, hi = r[f"{k}_lo"], r[f"{k}_hi"]
        if (0 < lo < tol) or (-tol < hi < 0):
            return True
    return False


def _fmt(r, kind="pct"):
    return f"{r['mean']:+.3f} [{r[f'{kind}_lo']:+.3f}, {r[f'{kind}_hi']:+.3f}]"


def _line(label, r, extra=""):
    print(f"    {label:<46s} {r['mean']:+.4f}  pct [{r['pct_lo']:+.4f},{r['pct_hi']:+.4f}]"
          f"  BCa [{r['bca_lo']:+.4f},{r['bca_hi']:+.4f}]  t [{r['t_lo']:+.4f},{r['t_hi']:+.4f}]"
          f"  adj{100 * r['adj_level']:.2f}% t [{r['t_adj_lo']:+.4f},{r['t_adj_hi']:+.4f}]"
          f" pct [{r['pct_adj_lo']:+.4f},{r['pct_adj_hi']:+.4f}]" + extra)


def _load_module(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ================================================================== gate (W1, REV-24)
GATE_CELLS = {("on", "gated"): "SEHHO:Full", ("on", "off"): "SEHHO:NoLevy",
              ("off", "gated"): "SEHHO:AlwaysPbest", ("off", "off"): "SEHHO:AlwaysPbest+NoLevy"}
GATE_BLOCKS = [("15,000 FEs", "E4c_ablation_cec2017_30D_15k.csv"),
               ("300,000 FEs", "E4b_ablation_cec2017_30D.csv")]
MEAN_ERROR_EFFECT = {"gate effect, Levy gated": "remove gate, Levy gated",
                     "gate effect, Levy off": "remove gate, Levy off",
                     "Levy effect, gate on": "remove Levy, gate on",
                     "Levy effect, gate off": "remove Levy, gate off"}


def gate():
    """Table 3 and Table S3.3.  Replicates analyze_gate_inference.py's draws."""
    print("\n" + "=" * 100)
    print("GATE x LEVY FACTORIAL, CEC2017 D=30 (Table 3; W1, REV-24)")
    print("delta of the changed arm against the unchanged one; negative favours the change")
    print("=" * 100)
    rng = np.random.default_rng(SEED_GATE)
    gf = _load_module("analyze_gate_factorial")
    rows, gate_delta, mean_err = [], {}, {}

    def rec(name, v, block):
        idx = boot_indices(rng, len(v))
        r = all_intervals(v, idx, k=4)
        r.update(quantity=name, block=block, p=signed_rank(v))
        rows.append(r)

    for label, fname in GATE_BLOCKS:
        df = _read(fname)
        df = df[df.dim == 30]
        funcs = sorted(df.func.unique())
        g_gated = per_fn_delta(df, GATE_CELLS[("off", "gated")], GATE_CELLS[("on", "gated")], funcs)
        g_off = per_fn_delta(df, GATE_CELLS[("off", "off")], GATE_CELLS[("on", "off")], funcs)
        gate_delta[label] = dict(gated=g_gated, off=g_off, funcs=funcs)
        rec("gate effect, Levy gated", g_gated, label)
        rec("gate effect, Levy off", g_off, label)
        l_on = per_fn_delta(df, GATE_CELLS[("on", "off")], GATE_CELLS[("on", "gated")], funcs)
        l_off = per_fn_delta(df, GATE_CELLS[("off", "off")], GATE_CELLS[("off", "gated")], funcs)
        rec("Levy effect, gate on", l_on, label)
        rec("Levy effect, gate off", l_off, label)
        rec("interaction gate x Levy", g_gated - g_off, label)
        with contextlib.redirect_stdout(io.StringIO()):
            me = gf.analyse(df, label)
        for _, r in me.iterrows():
            mean_err[(label, r.effect)] = (float(r.p), float(r.p_holm), float(r.delta))
    a, b = GATE_BLOCKS[1][0], GATE_BLOCKS[0][0]
    for lev in ("gated", "off"):
        rec(f"interaction gate x budget, Levy {lev}",
            gate_delta[a][lev] - gate_delta[b][lev], f"{a} - {b}")
    out = pd.DataFrame(rows)
    out["family"] = np.where(out.quantity.str.startswith("interaction"), "interaction",
                             np.where(out.quantity.str.startswith("gate effect"),
                                      "gate effect", "Levy effect"))
    out["p_holm"] = np.nan
    for fam, idx in out.groupby("family").groups.items():
        out.loc[idx, "p_holm"] = holm(out.loc[idx, "p"].to_numpy())
    out["p_mean_error"] = [mean_err.get((r.block, MEAN_ERROR_EFFECT.get(r.quantity, "")),
                                        (np.nan,) * 3)[0] for r in out.itertuples()]
    out["p_holm_mean_error"] = [mean_err.get((r.block, MEAN_ERROR_EFFECT.get(r.quantity, "")),
                                             (np.nan,) * 3)[1] for r in out.itertuples()]
    ver = pd.DataFrame([_verdicts(r) for r in out.to_dict("records")])
    out = pd.concat([out, ver], axis=1)
    # reproduction check against the stored analysis
    st = _read("analysis/gate_inference.csv")
    m = out.merge(st, on=["quantity", "block"], suffixes=("", "_stored"))
    _check("gate: percentile intervals reproduce gate_inference.csv",
           m[["pct_lo", "pct_hi", "mean", "p_holm"]].to_numpy(),
           m[["ci_lo", "ci_hi", "mean_stored", "p_holm_stored"]].to_numpy())
    st2 = _read("analysis/gate_levy_factorial.csv")
    for (blk, eff), (p, ph, dl) in mean_err.items():
        s = st2[(st2.block == blk) & (st2.effect == eff)]
        _check(f"gate: mean-error p_Holm {blk} {eff}", [ph, dl], s[["p_holm", "delta"]].to_numpy()[0])
    for fam in ("gate effect", "Levy effect", "interaction"):
        print(f"  {fam} (Holm and Bonferroni family of 4)")
        for r in out[out.family == fam].to_dict("records"):
            _line(f"{r['quantity']} @ {r['block']}", r,
                  f"  p={r['p']:.4f} pHolm={r['p_holm']:.4f}"
                  + (f" pHolm(mean err)={r['p_holm_mean_error']:.3f}"
                     if np.isfinite(r['p_holm_mean_error']) else ""))
            print(f"    {'':<46s} verdicts pct 0.147/0.10/0.05: {r['verdict_pct_0.147']}/"
                  f"{r['verdict_pct_0.1']}/{r['verdict_pct_0.05']}; all three intervals: "
                  f"{r['verdict_all_0.147']}/{r['verdict_all_0.1']}/{r['verdict_all_0.05']};"
                  f" side all={r['side_all']} adjusted={r['side_all_adj']}")
    _write(out, "revision_gate_inference.csv")
    return out


# ================================================================== pop isolation (W1)
POP_CONTRASTS = ["L-SHADE 6D vs 18D", "ours 6D vs ours 18D", "ours vs L-SHADE at 6D",
                 "ours vs L-SHADE default"]


def _method_runs(suite):
    parts = []
    for f in ("E3_cec2017.csv", "E9_newmethod_vs_baselines.csv", "E12_cec2017_tight.csv",
              "E13_cec2014.csv"):
        d = _read(f)
        if "suite" in d.columns:
            d = d[d.suite == suite]
        parts.append(d[d.algo == METHOD])
    return pd.concat(parts, ignore_index=True)


def pop():
    """Table 5 and its supplementary version.  Replicates analyze_pop_isolation.py."""
    print("\n" + "=" * 100)
    print("ONE-CONSTANT POPULATION CONTROL (Table 5; W1). delta of the first-named arm; "
          "negative favours it")
    print("=" * 100)
    e14 = _read("E14_pop_isolation.csv")
    rng = np.random.default_rng(SEED_CEC2014)
    rows = []
    for suite in ("cec2017", "cec2014"):
        s0 = e14[e14.suite == suite]
        mine = _method_runs(suite)
        for dim in sorted(s0.dim.unique()):
            for fes in sorted(s0[s0.dim == dim].max_fes.unique()):
                sub = s0[(s0.dim == dim) & (s0.max_fes == fes)]
                funcs = sorted(sub.func.unique())
                mm = mine[(mine.dim == dim) & (mine.max_fes == fes)]
                merged = pd.concat([sub, mm], ignore_index=True)
                for label, frame, a, b in (
                        ("L-SHADE 6D vs 18D", sub, "LSHADE-N6", "LSHADE-N18"),
                        ("ours 6D vs ours 18D", merged, METHOD, "GF@18D"),
                        ("ours vs L-SHADE at 6D", merged, METHOD, "LSHADE-N6"),
                        ("ours vs L-SHADE default", merged, METHOD, "LSHADE-N18")):
                    v = per_fn_delta(frame, a, b, funcs)
                    r = all_intervals(v, boot_indices(rng, len(v)), k=4)
                    r.update(suite=suite, dim=int(dim), max_fes=int(fes),
                             budget="tight" if fes == 15_000 else "competition",
                             contrast=label, arm_a=a, arm_b=b, p=signed_rank(v),
                             evals_per_variable=fes / dim)
                    rows.append(r)
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for _, idx in out.groupby(["suite", "dim", "max_fes"]).groups.items():
        out.loc[idx, "p_holm"] = holm(out.loc[idx, "p"].to_numpy())
    out = pd.concat([out, pd.DataFrame([_verdicts(r) for r in out.to_dict("records")])], axis=1)
    st = _read("analysis/pop_isolation.csv")
    m = out.merge(st, on=["suite", "dim", "max_fes", "contrast"])
    _check("pop: percentile intervals reproduce pop_isolation.csv",
           m[["pct_lo", "pct_hi", "mean", "p_holm_x"]].to_numpy(),
           m[["ci_lo", "ci_hi", "mean_delta", "p_holm_y"]].to_numpy())
    for c in POP_CONTRASTS:
        print(f"  {c}  (Bonferroni family: the four blocks of this row and budget)")
        for r in out[out.contrast == c].to_dict("records"):
            _line(f"{r['suite']} D={r['dim']} {r['budget']} (B/D={r['evals_per_variable']:g})", r,
                  f"  pHolm={r['p_holm']:.2g}  side all={r['side_all']} adj={r['side_all_adj']}")
    _write(out, "revision_pop_isolation.csv")
    return out


# ================================================================== CEC2014 (W1, W3, REV-38, REV-29)
def _cec17_comp():
    return pd.concat([_read("E3_cec2017.csv"),
                      _read("E9_newmethod_vs_baselines.csv").query("tag == 'E3_cec2017'")],
                     ignore_index=True)


def cec2014():
    """Table 6 (both panels), Table S5.1 and the precision argument of S2.6."""
    print("\n" + "=" * 100)
    print("PRE-REGISTERED CEC2014 EVALUATION (Table 6; W1, W3, REV-38, REV-29)")
    print("=" * 100)
    d = _read("E13_cec2014.csv")
    funcs = sorted(d.func.unique())
    rng = np.random.default_rng(SEED_CEC2014)
    comp = [(dim, 10_000 * dim) for dim in (30, 50)]
    tight = [(dim, 15_000) for dim in (30, 50)]
    prim, sec = [], []
    for dim, fes in comp:                                  # registered primary endpoint
        sub = d[(d.dim == dim) & (d.max_fes == fes)]
        v = per_fn_delta(sub, METHOD, PUBLISHED, funcs)
        r = all_intervals(v, boot_indices(rng, len(v)), k=2)
        r.update(analysis="registered primary endpoint", dim=dim, max_fes=fes,
                 functions="all 30", p=signed_rank(v), wins=int((v < 0).sum()),
                 losses=int((v > 0).sum()))
        prim.append(r)
    for label, blocks in (("competition", comp), ("tight", tight)):   # registered secondary
        for dim, fes in blocks:
            sub = d[(d.dim == dim) & (d.max_fes == fes)]
            algos = sorted(sub.algo.unique(), key=lambda a: (a != METHOD, a))
            M = np.array([[sub[(sub.func == f) & (sub.algo == a)]["error"].mean()
                           for a in algos] for f in funcs])
            fr, post = friedman_holm_vs_control(M, algos, METHOD)
            for a, _ in sorted(zip(algos, fr["avg_ranks"]), key=lambda t: t[1]):
                if a == METHOD:
                    continue
                row = next(x for x in post if x["algorithm"] == a)
                v = per_fn_delta(sub, METHOD, a, funcs)
                r = all_intervals(v, boot_indices(rng, len(v)), k=6)
                r.update(suite="cec2014", dim=dim, max_fes=fes, budget=label, opponent=a,
                         p=signed_rank(v), p_holm_friedman=float(row["p_holm"]),
                         wins=int((v < 0).sum()), losses=int((v > 0).sum()))
                sec.append(r)
    prim = pd.DataFrame(prim)
    prim["p_holm"] = holm(prim.p.to_numpy())
    sec = pd.DataFrame(sec)
    sec["p_holm"] = np.nan
    for _, idx in sec.groupby(["dim", "max_fes"]).groups.items():
        sec.loc[idx, "p_holm"] = holm(sec.loc[idx, "p"].to_numpy())
    st = _read("analysis/cec2014_confirmatory.csv")
    sp = st[st.opponent.isna()]
    _check("cec2014: primary intervals reproduce cec2014_confirmatory.csv",
           prim[["mean", "pct_lo", "pct_hi", "p_holm"]].to_numpy(),
           sp[["mean_delta", "ci_lo", "ci_hi", "p_holm"]].to_numpy())
    ss = st[st.opponent.notna()]
    m = sec.merge(ss, on=["dim", "max_fes", "opponent"], suffixes=("", "_stored"))
    _check("cec2014: secondary intervals reproduce cec2014_confirmatory.csv",
           m[["mean", "pct_lo", "pct_hi", "p_holm_friedman"]].to_numpy(),
           m[["mean_delta", "ci_lo", "ci_hi", "p_holm_stored"]].to_numpy())

    # ---- REV-38: the endpoint without the four shared-recipe functions (post hoc)
    keep = [f for f in funcs if f not in EXCLUDED_CEC2014]
    rng38 = np.random.default_rng(SEED_CEC2014)
    ex = []
    for dim, fes in comp:
        sub = d[(d.dim == dim) & (d.max_fes == fes)]
        v = per_fn_delta(sub, METHOD, PUBLISHED, keep)
        r = all_intervals(v, boot_indices(rng38, len(v)), k=2)
        r.update(analysis="post hoc: without F18, F22, F23, F24", dim=dim, max_fes=fes,
                 functions="26 (F18, F22, F23, F24 excluded)", p=signed_rank(v),
                 wins=int((v < 0).sum()), losses=int((v > 0).sum()))
        ex.append(r)
    ex = pd.DataFrame(ex)
    ex["p_holm"] = holm(ex.p.to_numpy())
    ex = pd.concat([ex, pd.DataFrame([_verdicts(r) for r in ex.to_dict("records")])], axis=1)
    prim = pd.concat([prim, pd.DataFrame([_verdicts(r) for r in prim.to_dict("records")])], axis=1)
    # the four excluded functions on their own (descriptive)
    four = []
    for dim, fes in comp:
        sub = d[(d.dim == dim) & (d.max_fes == fes)]
        v = per_fn_delta(sub, METHOD, PUBLISHED, list(EXCLUDED_CEC2014))
        four.append(dict(dim=dim, max_fes=fes, **{f"F{f}": float(x)
                                                   for f, x in zip(EXCLUDED_CEC2014, v)},
                         mean_four=float(v.mean())))

    # ---- the CEC2017 counterpart and the suite gap (replicates analyze_extras.suite_gap)
    c17 = _cec17_comp()
    rngg = np.random.default_rng(SEED_CEC2014)
    gap = []
    for dim in (30, 50):
        fes = 10_000 * dim
        v17 = per_fn_delta(c17[(c17.dim == dim) & (c17.max_fes == fes)], METHOD, PUBLISHED)
        v14 = per_fn_delta(d[(d.dim == dim) & (d.max_fes == fes)], METHOD, PUBLISHED)
        m17 = boot_means(v17, boot_indices(rngg, len(v17)))
        m14 = boot_means(v14, boot_indices(rngg, len(v14)))
        lo17, hi17 = percentile_ci(m17)
        blo, bhi = bca_ci(v17, m17)
        tlo, thi = t_ci(v17)
        glo, ghi = percentile_ci(m17 - m14)
        gap.append(dict(dim=dim, max_fes=fes, cec2017_mean=float(v17.mean()), cec2017_pct_lo=lo17,
                        cec2017_pct_hi=hi17, cec2017_bca_lo=blo, cec2017_bca_hi=bhi,
                        cec2017_t_lo=tlo, cec2017_t_hi=thi, cec2017_p=signed_rank(v17),
                        cec2014_mean=float(v14.mean()),
                        gap_cec2017_minus_cec2014=float(v17.mean() - v14.mean()),
                        gap_pct_lo=glo, gap_pct_hi=ghi, n17=len(v17), n14=len(v14)))
    gap = pd.DataFrame(gap)
    ext = _read("analysis/extras.csv")
    eg = ext[ext.quantity == "suite_gap_cec2017_minus_cec2014"].sort_values("dim")
    _check("cec2014: suite gap reproduces extras.csv (a)",
           gap[["gap_cec2017_minus_cec2014", "gap_pct_lo", "gap_pct_hi"]].to_numpy(),
           eg[["estimate", "ci_lo", "ci_hi"]].to_numpy())

    # ---- W3: per-function standard error of delta at 30 runs; two-stage bootstrap
    prec = []
    rng3 = np.random.default_rng(SEED_REVISION)
    for dim, fes in comp:
        sub = d[(d.dim == dim) & (d.max_fes == fes)]
        ra = [sub[(sub.func == f) & (sub.algo == METHOD)]["error"].to_numpy() for f in funcs]
        rb = [sub[(sub.func == f) & (sub.algo == PUBLISHED)]["error"].to_numpy() for f in funcs]
        se = np.sqrt([cliff_var_unbiased(a, b) for a, b in zip(ra, rb)])
        v = np.array([cliffs_delta(a, b) for a, b in zip(ra, rb)])
        mean, (lo2, hi2), _ = two_stage_bootstrap(ra, rb, rng3)
        one = prim[prim.dim == dim].iloc[0]
        prec.append(dict(dim=dim, max_fes=fes, contrast="SEHHO-COBL-R vs SEHHO-COBL",
                         n_functions=len(funcs), runs_per_arm=len(ra[0]),
                         se_run_median=float(np.median(se)), se_run_q1=float(np.quantile(se, .25)),
                         se_run_q3=float(np.quantile(se, .75)), se_run_min=float(se.min()),
                         se_run_max=float(se.max()), sd_between_functions=float(v.std(ddof=1)),
                         se_of_mean_functions=float(v.std(ddof=1) / np.sqrt(len(v))),
                         se_of_mean_runs_only=float(np.sqrt(np.sum(se ** 2)) / len(v)),
                         one_stage_lo=float(one.pct_lo), one_stage_hi=float(one.pct_hi),
                         two_stage_lo=lo2, two_stage_hi=hi2, mean=mean,
                         n_functions_se_below_0_1=int((se < 0.1).sum())))
    # every pairwise contrast of E13 at 30 runs, for the distribution over functions
    allse = []
    for dim in (30, 50):
        for fes in (15_000, 10_000 * dim):
            sub = d[(d.dim == dim) & (d.max_fes == fes)]
            for opp in COMPETITORS:
                for f in funcs:
                    a = sub[(sub.func == f) & (sub.algo == METHOD)]["error"].to_numpy()
                    b = sub[(sub.func == f) & (sub.algo == opp)]["error"].to_numpy()
                    allse.append(np.sqrt(cliff_var_unbiased(a, b)))
    allse = np.array(allse)
    prec.append(dict(dim=0, max_fes=0, contrast="all E13 pairwise contrasts (SEHHO-COBL-R vs "
                     "each of six, four blocks)", n_functions=len(allse), runs_per_arm=30,
                     se_run_median=float(np.median(allse)), se_run_q1=float(np.quantile(allse, .25)),
                     se_run_q3=float(np.quantile(allse, .75)), se_run_min=float(allse.min()),
                     se_run_max=float(allse.max())))
    prec = pd.DataFrame(prec)

    print("  Registered primary endpoint (delta of SEHHO-COBL-R against SEHHO-COBL; Bonferroni "
          "family = the two blocks)")
    for r in prim.to_dict("records"):
        _line(f"D={r['dim']}, {r['max_fes']:,} FEs, 30 functions", r,
              f"  p={r['p']:.2e} pHolm={r['p_holm']:.2e} W/L {r['wins']}/{r['losses']}")
    print("  REV-38 (post hoc, requested in the internal pre-submission review): without F18, F22, F23, F24")
    for r in ex.to_dict("records"):
        _line(f"D={r['dim']}, {r['max_fes']:,} FEs, 26 functions", r,
              f"  p={r['p']:.2e} pHolm={r['p_holm']:.2e} W/L {r['wins']}/{r['losses']}"
              f"  -> pct {r['side_pct']} zero; all three: {r['side_all']}; adjusted: {r['side_all_adj']}")
    for r in four:
        print(f"    the four excluded functions at D={r['dim']}: " + ", ".join(
            f"F{f} {r[f'F{f}']:+.3f}" for f in EXCLUDED_CEC2014) + f"; their mean {r['mean_four']:+.3f}")
    print("  CEC2017 counterpart and suite gap (CEC2017 minus CEC2014; replicates extras (a))")
    for r in gap.to_dict("records"):
        print(f"    D={r['dim']}: CEC2017 {r['cec2017_mean']:+.4f} pct [{r['cec2017_pct_lo']:+.4f},"
              f"{r['cec2017_pct_hi']:+.4f}] BCa [{r['cec2017_bca_lo']:+.4f},{r['cec2017_bca_hi']:+.4f}]"
              f" t [{r['cec2017_t_lo']:+.4f},{r['cec2017_t_hi']:+.4f}];  CEC2014 {r['cec2014_mean']:+.4f};"
              f"  gap {r['gap_cec2017_minus_cec2014']:+.4f} [{r['gap_pct_lo']:+.4f},{r['gap_pct_hi']:+.4f}]")
    print("  Pairwise on CEC2014 (REV-29): signed-rank on per-function delta, Holm over the six "
          "competitors of each block; registered Friedman post hoc p_Holm alongside")
    for r in sec.to_dict("records"):
        _line(f"vs {DISPLAY.get(r['opponent'], r['opponent'])} D={r['dim']} {r['budget']}", r,
              f"  pHolm(signed-rank)={r['p_holm']:.4f} pHolm(Friedman)={r['p_holm_friedman']:.4f}"
              f" W/L {r['wins']}/{r['losses']}")
    print("  W3: precision at 30 runs (Cliff's unbiased run-level SE of each per-function delta)")
    for r in prec.to_dict("records"):
        if r["dim"]:
            print(f"    D={r['dim']}: per-function SE median {r['se_run_median']:.3f} (IQR "
                  f"{r['se_run_q1']:.3f}-{r['se_run_q3']:.3f}, range {r['se_run_min']:.3f}-"
                  f"{r['se_run_max']:.3f}); SD of delta between functions {r['sd_between_functions']:.3f}"
                  f"; SE of the mean from functions {r['se_of_mean_functions']:.4f}, from runs only "
                  f"{r['se_of_mean_runs_only']:.4f}; one-stage [{r['one_stage_lo']:+.4f},"
                  f"{r['one_stage_hi']:+.4f}] two-stage [{r['two_stage_lo']:+.4f},{r['two_stage_hi']:+.4f}]")
        else:
            print(f"    {r['contrast']}: {r['n_functions']} per-function SEs, median "
                  f"{r['se_run_median']:.3f} (IQR {r['se_run_q1']:.3f}-{r['se_run_q3']:.3f}, max "
                  f"{r['se_run_max']:.3f})")
    allp = pd.concat([prim, ex], ignore_index=True)
    _write(allp, "revision_cec2014_primary.csv")
    _write(gap, "revision_cec2014_suitegap.csv")
    _write(pd.DataFrame(four), "revision_cec2014_excluded_functions.csv")
    sec = pd.concat([sec, pd.DataFrame([_verdicts(r) for r in sec.to_dict("records")])], axis=1)
    _write(sec, "revision_cec2014_pairwise.csv")
    _write(prec, "revision_precision.csv")
    return allp, sec, gap, prec


# ================================================================== suites (W2, REV-29)
SUITE_BLOCKS = [
    # (table, source files + E9 tag, dims, budget label)
    ("cec2017_competition", ("E3_cec2017.csv", "E3_cec2017"), (30, 50, 100), "competition"),
    ("cec2017_tight", ("E12_cec2017_tight.csv", None), (30, 50), "tight"),
    ("cec2022_matched", ("E1_cec2022_paper_budget.csv", "E1_cec2022_paper_budget"), (10, 20), "tight"),
    ("cec2022_competition", ("E2_cec2022_competition.csv", "E2_cec2022_competition"), (10, 20),
     "competition"),
]


def suites():
    """Tables S5.2 and S5.3: delta intervals and signed-rank tests for CEC2017 and CEC2022."""
    print("\n" + "=" * 100)
    print("PAIRWISE COMPARISONS ON PER-FUNCTION DELTA, CEC2017 AND CEC2022 (W2, REV-29)")
    print("delta of SEHHO-COBL-R against the competitor; negative favours SEHHO-COBL-R;"
          " Holm over the six competitors of each block")
    print("=" * 100)
    rng = np.random.default_rng(SEED_REVISION)
    e9 = _read("E9_newmethod_vs_baselines.csv")
    rows = []
    for table, (fname, tag), dims, budget in SUITE_BLOCKS:
        base = _read(fname)
        df = pd.concat([base, e9[e9.tag == tag]], ignore_index=True) if tag else base
        for dim in dims:
            sub = df[df.dim == dim]
            funcs = sorted(sub.func.unique())
            fes = int(sub.max_fes.iloc[0])
            for opp in COMPETITORS:
                v = per_fn_delta(sub, METHOD, opp, funcs)
                r = all_intervals(v, boot_indices(rng, len(v)), k=6)
                r.update(table=table, suite=table.split("_")[0], dim=dim, max_fes=fes,
                         budget=budget, opponent=opp, p=signed_rank(v),
                         wins=int((v < 0).sum()), losses=int((v > 0).sum()),
                         ties=int((v == 0).sum()))
                rows.append(r)
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for _, idx in out.groupby(["table", "dim"]).groups.items():
        out.loc[idx, "p_holm"] = holm(out.loc[idx, "p"].to_numpy())
    out = pd.concat([out, pd.DataFrame([_verdicts(r) for r in out.to_dict("records")])], axis=1)
    for (table, dim), s in out.groupby(["table", "dim"], sort=False):
        print(f"  {table}, D={dim}")
        for r in s.to_dict("records"):
            _line(f"vs {DISPLAY.get(r['opponent'], r['opponent'])}", r,
                  f"  pHolm={r['p_holm']:.4f}  per-function W/T/L {r['wins']}/{r['ties']}/{r['losses']}")
    _write(out, "revision_suite_pairwise.csv")
    return out


# ================================================================== composition (W1, REV-26, 33, 58)
def comp():
    """Table S4.2 and S4.3.  Replicates analyze_composition.py's draws."""
    print("\n" + "=" * 100)
    print("COMPOSITION OF THE REPAIRED CONFIGURATION, CEC2022 (S4.2, S4.3; W1)")
    print("delta of the changed configuration against the default (A = changed); positive "
          "favours the default, i.e. the part earns its place")
    print("=" * 100)
    d = pd.concat([_read("E8_composition_cec2022.csv"), _read("E11_stripped_cec2022.csv")],
                  ignore_index=True)
    d["cand"] = d.algo.str.slice(3).str.split("@").str[0]
    d["block"] = d.algo.str.split("@").str[1]
    cands = [c for c in d.cand.unique() if c != "Default"]
    funcs = sorted(d.func.unique())
    rng = np.random.default_rng(SEED_POPRULE)
    rows = []
    for dim in sorted(d.dim.unique()):
        for blk in ("matched", "competition"):
            sub = d[(d.dim == dim) & (d.block == blk)]
            fes = int(sub.max_fes.iloc[0])
            for c in cands:
                v = per_fn_delta(sub, c, "Default", funcs, col="cand")
                r = all_intervals(v, boot_indices(rng, len(v)), k=4)
                r.update(dim=int(dim), block=blk, max_fes=fes, change=c, p=signed_rank(v))
                rows.append(r)
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for _, idx in out.groupby(["dim", "max_fes"]).groups.items():
        out.loc[idx, "p_holm"] = holm(out.loc[idx, "p"].to_numpy())
    out = pd.concat([out, pd.DataFrame([_verdicts(r) for r in out.to_dict("records")])], axis=1)
    st = _read("analysis/composition.csv")
    m = out.merge(st, on=["dim", "max_fes", "change"], suffixes=("", "_stored"))
    _check("comp: percentile intervals reproduce composition.csv",
           m[["mean", "pct_lo", "pct_hi", "p_holm"]].to_numpy(),
           m[["mean_delta", "ci_lo", "ci_hi", "p_holm_stored"]].to_numpy())
    # decisions of the amended rule under each interval type and margin
    dec = []
    for c in cands:
        s = out[out.change == c]
        for kind in ("pct", "bca", "t", "pct_adj", "bca_adj", "t_adj"):
            for mg in MARGINS:
                earns = bool((s[f"{kind}_lo"] > 0).any())
                helps = bool((s[f"{kind}_hi"] < 0).any())
                negl = bool(((s[f"{kind}_lo"] > -mg) & (s[f"{kind}_hi"] < mg)).all())
                decision = ("KEEP (earns its place)" if earns else
                            "DROP (removal helps somewhere)" if helps else
                            "DROP (negligible everywhere)" if negl else "KEEP (by default)")
                dec.append(dict(change=c, interval=kind, margin=mg, decision=decision,
                                earns=earns, helps=helps, negligible_everywhere=negl))
    dec = pd.DataFrame(dec)
    for c in cands:
        print(f"  {c}  (Bonferroni family: the four blocks of this change)")
        for r in out[out.change == c].to_dict("records"):
            _line(f"D={r['dim']} {r['max_fes']:,} FEs", r,
                  f"  pHolm={r['p_holm']:.3f} side all={r['side_all']} adj={r['side_all_adj']}"
                  f"{'  [bound within 0.02 of zero]' if r['tie_002'] else ''}")
        base = dec[(dec.change == c) & (dec.interval == "pct") & (dec.margin == 0.147)].decision.iloc[0]
        alt = dec[(dec.change == c) & (dec.decision != base)]
        print(f"    amended-rule decision (pct, 0.147): {base}"
              + ("" if alt.empty else "; differs under: " + ", ".join(
                  f"{r.interval}@{r.margin:g} -> {r.decision}" for r in alt.itertuples())))
    _write(out, "revision_composition.csv")
    _write(dec, "revision_composition_decisions.csv")
    return out, dec


# ================================================================== poprule (W1, REV-56, REV-59)
def poprule():
    """Table S4.1.  Replicates analyze_poprule.py's draws."""
    print("\n" + "=" * 100)
    print("POPULATION-RULE SEARCH, CEC2022 (S4.1; W1). delta of each candidate against the "
          "selected LPSR-6D; positive favours LPSR-6D")
    print("=" * 100)
    d = _read("E7_poprule_cec2022.csv")
    d["cand"] = d.algo.str.slice(4).str.split("@").str[0]
    d["block"] = d.algo.str.split("@").str[1]
    cands = sorted(d.cand.unique())
    funcs = sorted(d.func.unique())
    winner = json.load(open(os.path.join(OUT, "selected_config.json")))["winner"]
    rng = np.random.default_rng(SEED_POPRULE)
    rows, ranks = [], []
    for dim in sorted(d.dim.unique()):
        for blk in ("matched", "competition"):
            sub = d[(d.dim == dim) & (d.block == blk)]
            fes = int(sub.max_fes.iloc[0])
            M = np.array([[sub[(sub.func == f) & (sub.cand == c)]["error"].mean() for c in cands]
                          for f in funcs])
            rk = friedman(M)["avg_ranks"]
            order = np.argsort(np.argsort(rk, kind="stable"), kind="stable") + 1
            for c, r_, o in zip(cands, rk, order):
                ranks.append(dict(dim=int(dim), block=blk, max_fes=fes, candidate=c,
                                  avg_rank=float(r_), position=int(o), of=len(cands)))
            for c in cands:
                if c == winner:
                    continue
                v = per_fn_delta(sub, c, winner, funcs, col="cand")
                r = all_intervals(v, boot_indices(rng, len(v)), k=len(cands) - 1)
                r.update(dim=int(dim), block=blk, max_fes=fes, candidate=c, versus=winner,
                         p=signed_rank(v))
                rows.append(r)
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for _, idx in out.groupby(["dim", "max_fes"]).groups.items():
        out.loc[idx, "p_holm"] = holm(out.loc[idx, "p"].to_numpy())
    out = pd.concat([out, pd.DataFrame([_verdicts(r) for r in out.to_dict("records")])], axis=1)
    st = _read("analysis/poprule.csv")
    m = out.merge(st, on=["dim", "max_fes", "candidate"], suffixes=("", "_stored"))
    _check("poprule: percentile intervals reproduce poprule.csv",
           m[["mean", "pct_lo", "pct_hi", "p_holm"]].to_numpy(),
           m[["mean_delta", "ci_lo", "ci_hi", "p_holm_stored"]].to_numpy())
    ranks = pd.DataFrame(ranks)
    for (dim, fes), s in ranks.groupby(["dim", "max_fes"]):
        s = s.sort_values("avg_rank")
        print(f"  D={dim}, {fes:,} FEs ranks: " + ", ".join(
            f"{r.position}. {r.candidate} {r.avg_rank:.2f}" for r in s.itertuples()))
    for r in out.to_dict("records"):
        _line(f"{r['candidate']} vs {winner} D={r['dim']} {r['max_fes']:,}", r,
              f"  pHolm={r['p_holm']:.3f}")
    _write(out, "revision_poprule.csv")
    _write(ranks, "revision_poprule_ranks.csv")
    return out, ranks


# ================================================================== popsize (W9, REV-32)
def popsize():
    """Table 4: delta(N vs N=30) in SEHHO-COBL, CEC2017 D=30, 300,000 evaluations (E6)."""
    print("\n" + "=" * 100)
    print("POPULATION SWEEP IN SEHHO-COBL, CEC2017 D=30, 300,000 FEs (Table 4; W9)")
    print("delta of the population N against the published N=30; negative favours N")
    print("=" * 100)
    e6 = _read("E6_popsize_cec2017_30D.csv")
    e6 = e6[(e6.dim == 30) & (e6.max_fes == 300_000)]
    rows = []
    for N in (60, 120, 270, 540):
        v = per_fn_delta(e6, f"E6_popsize_N{N}", "E6_popsize_N30", col="tag")
        # one generator per contrast, seeded as analyze_extras.py (d), so the N=120
        # row is that interval with its sign reversed
        r = all_intervals(v, boot_indices(np.random.default_rng(SEED_CEC2014), len(v)), k=4)
        r.update(N=N, contrast=f"N={N} vs N=30", p=signed_rank(v),
                 wins=int((v < 0).sum()), losses=int((v > 0).sum()))
        rows.append(r)
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p.to_numpy())
    out = pd.concat([out, pd.DataFrame([_verdicts(r) for r in out.to_dict("records")])], axis=1)
    ext = _read("analysis/extras.csv")
    e = ext[ext.quantity == "e6_n30_vs_n120"].iloc[0]
    r120 = out[out.N == 120].iloc[0]
    _check("popsize: N=120 is extras.csv (d) with the sign reversed",
           [r120["mean"], r120.pct_lo, r120.pct_hi], [-e.estimate, -e.ci_hi, -e.ci_lo])
    for r in out.to_dict("records"):
        _line(r["contrast"], r, f"  pHolm={r['p_holm']:.2e} W/L {r['wins']}/{r['losses']}")
    _write(out, "revision_popsize.csv")
    return out


# ================================================================== E16 (W4, REV-09)
_BEST_KNOWN = {}


def best_known():
    """Best-known value per problem: classical references (classical_references.py) for P1
    and P2; for RC01, RC06 and RC14 the values listed in the CEC2020 guidelines (Kumar et
    al.), which the classical references reproduce."""
    if _BEST_KNOWN:
        return _BEST_KNOWN
    ref = _read("analysis/revision_classical_refs.csv")
    out = _BEST_KNOWN
    for prob, s in ref.groupby("problem"):
        out[prob] = float(s.objective.min())
    listed = {"P3_heat_exchanger_RC01": 189.31162966, "P4_blending_pooling_RC06": 1.8638304088,
              "P5_batch_plant_RC14": 53638.942722}
    for k, v in listed.items():
        # the classical value and the listed one agree to 1e-8 relative
        if k in out:
            CHECKS.append((f"best-known {k}: classical {out[k]:.10g} vs listed {v:.10g}",
                           abs(out[k] - v) / v < 1e-8, ""))
        out[k] = v
    return out


def _success(objective, feasible, fstar, tol=SUCCESS_TOL):
    """Our success rule: feasible and (f - f*)/|f*| <= tol (relative)."""
    obj = np.asarray(objective, float)
    feas = np.asarray(feasible, bool)
    return feas & ((obj - fstar) / abs(fstar) <= tol)


def _success_suite(objective, feasible, fstar):
    """The CEC2020 suite's success rule (Kumar et al. 2020, Problem-Definitions.pdf):
    a feasible solution with f - f* <= 1e-8 (absolute) within MaxFEs."""
    obj = np.asarray(objective, float)
    feas = np.asarray(feasible, bool)
    return feas & (obj - fstar <= 1e-8)


def e16():
    print("\n" + "=" * 100)
    print("E16: L-SHADE 6D vs 18D ON THE FIVE DESIGN PROBLEMS, 15,000 FEs (Table 8; W4)")
    print(f"success = feasible and (f - f*)/|f*| <= {SUCCESS_TOL:g}; the suite's own success rate "
          "(Kumar et al. 2020) uses f - f* <= 1e-8 (absolute) and is reported beside it")
    print("=" * 100)
    df = _read("E16_engineering_population.csv")
    reg = _read("analysis/e16_engineering_population.csv").set_index("problem")
    fstar = best_known()
    rows = []
    for prob in reg.index:
        s = df[df.problem == prob]
        row = dict(problem=prob, dim=int(reg.loc[prob, "dim"]),
                   evals_per_variable=float(reg.loc[prob, "evals_per_variable"]),
                   f_star=fstar[prob])
        for arm, lab in (("LSHADE-N6", "6D"), ("LSHADE-N18", "18D")):
            a = s[s.algo == arm]
            feas = a.feasible.astype(bool)
            succ = _success(a.objective, feas, fstar[prob])
            fo = a[feas].objective
            row.update({f"runs_{lab}": len(a), f"feasible_{lab}": int(feas.sum()),
                        f"success_{lab}": int(succ.sum()),
                        f"mean_feasible_{lab}": float(fo.mean()) if len(fo) else np.nan,
                        f"median_feasible_{lab}": float(fo.median()) if len(fo) else np.nan,
                        f"best_feasible_{lab}": float(fo.min()) if len(fo) else np.nan,
                        f"sd_feasible_{lab}": float(fo.std(ddof=1)) if len(fo) > 1 else np.nan,
                        f"mean_violation_{lab}": float(a.total_violation.mean()),
                        f"median_violation_{lab}": float(a.total_violation.median()),
                        f"excess_median_{lab}": (float((fo.median() - fstar[prob]) / abs(fstar[prob]))
                                                 if len(fo) else np.nan)})
            for tol in (1e-6, 1e-3, 1e-2):
                row[f"success_{lab}_tol{tol:g}"] = int(_success(a.objective, feas, fstar[prob], tol).sum())
            row[f"success_{lab}_suite_abs1e-8"] = int(_success_suite(a.objective, feas, fstar[prob]).sum())
        for k in ("delta", "ci_lo", "ci_hi", "mw_p", "mw_p_holm", "fisher_p", "verdict"):
            row[k] = reg.loc[prob, k]
        _check(f"e16: feasible counts {prob}", [row["feasible_6D"], row["feasible_18D"]],
               [reg.loc[prob, "feas_6D"], reg.loc[prob, "feas_18D"]])
        rows.append(row)
    out = pd.DataFrame(rows)
    for r in out.to_dict("records"):
        print(f"  {r['problem']:<26s} D={r['dim']:<3d} B/D={r['evals_per_variable']:7.1f}  "
              f"feasible {r['feasible_6D']}/30 vs {r['feasible_18D']}/30  success "
              f"{r['success_6D']}/30 vs {r['success_18D']}/30 (f*={r['f_star']:.10g})")
        print(f"  {'':<26s} mean feasible {r['mean_feasible_6D']:.6g} vs {r['mean_feasible_18D']:.6g};"
              f" median {r['median_feasible_6D']:.6g} vs {r['median_feasible_18D']:.6g}; best "
              f"{r['best_feasible_6D']:.6g} vs {r['best_feasible_18D']:.6g}")
        print(f"  {'':<26s} delta(6D vs 18D) {r['delta']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"
              f" pHolm(MW)={r['mw_p_holm']:.2e} Fisher p={r['fisher_p']:.3f} -> {r['verdict']};"
              f" median violation {r['median_violation_6D']:.3g} vs {r['median_violation_18D']:.3g};"
              f" success at 1e-2: {r['success_6D_tol0.01']} vs {r['success_18D_tol0.01']};"
              f" suite rule (absolute 1e-8): {r['success_6D_suite_abs1e-8']} vs {r['success_18D_suite_abs1e-8']}")
    _write(out, "revision_e16.csv")
    return out


# ================================================================== engineering success (W5)
def eng():
    print("\n" + "=" * 100)
    print("SEVEN ALGORITHMS ON THE FIVE DESIGN PROBLEMS: SUCCESS RATES (Table 7; W5)")
    print("=" * 100)
    df = pd.concat([_read("E5_engineering.csv"), _read("E10_newmethod_engineering.csv")],
                   ignore_index=True)
    fstar = best_known()
    rows = []
    for (prob, algo), s in df.groupby(["problem", "algo"]):
        feas = s.feasible.astype(bool)
        succ = _success(s.objective, feas, fstar[prob])
        at_thr = int(((s.total_violation > 0) & (s.total_violation <= 1e-8)).sum())
        rows.append(dict(problem=prob, algo=algo, display=DISPLAY.get(algo, algo), runs=len(s),
                         feasible=int(feas.sum()), success=int(succ.sum()), f_star=fstar[prob],
                         success_tol1e_2=int(_success(s.objective, feas, fstar[prob], 1e-2).sum()),
                         success_suite_abs1e_8=int(_success_suite(s.objective, feas, fstar[prob]).sum()),
                         runs_at_threshold=at_thr,
                         runs_violation_exactly_1e_8=int((s.total_violation == 1e-8).sum()),
                         mean_violation=float(s.total_violation.mean()),
                         max_violation_feasible=float(s[feas].total_violation.max()) if feas.any() else np.nan))
    out = pd.DataFrame(rows)
    for prob, s in out.groupby("problem"):
        print(f"  {prob} (f* = {s.f_star.iloc[0]:.10g}): " + "; ".join(
            f"{r.display} {r.success}/{r.runs} (suite rule {r.success_suite_abs1e_8}; feasible "
            f"{r.feasible}, at threshold {r.runs_at_threshold})" for r in s.itertuples()))
    _write(out, "revision_engineering_success.csv")
    return out


# ================================================================== guide (W11, REV-08)
def guide():
    """Rows of the budget-indexed guide (Table 10) and the points of Figure 3.

    Every row is delta(6D vs 18D), negative favouring 6D, read from (or, where the
    source has the other arm order, sign-reversed from) the file named in `source`.
    """
    print("\n" + "=" * 100)
    print("BUDGET-INDEXED GUIDE: delta(6D vs 18D) AGAINST EVALUATIONS PER VARIABLE (Table 10, "
          "Figure 3; W11)")
    print("=" * 100)
    rows = []
    pop = pd.read_csv(os.path.join(OUT, "revision_pop_isolation.csv"))
    for r in pop[pop.contrast.isin(["L-SHADE 6D vs 18D", "ours 6D vs ours 18D"])].to_dict("records"):
        lshade = r["contrast"] == "L-SHADE 6D vs 18D"
        status = ("measured" if r["suite"] == "cec2017" else "exploratory")
        rows.append(dict(evals_per_variable=r["evals_per_variable"], dim=r["dim"],
                         budget=r["max_fes"], problems=r["suite"].upper(),
                         engine="L-SHADE" if lshade else "repaired engine (SEHHO-COBL-R)",
                         source_marker=("CEC2017 L-SHADE" if lshade and status == "measured" else
                                        "CEC2014 L-SHADE (exploratory)" if lshade else
                                        "repaired engine"),
                         delta=r["mean"], ci_lo=r["pct_lo"], ci_hi=r["pct_hi"],
                         t_lo=r["t_lo"], t_hi=r["t_hi"], status=status,
                         source="results/analysis/pop_isolation.csv (E14; " + r["contrast"] + ")",
                         sign_reversed=False, unit="functions",
                         # tie rule of Section 2.6 on the percentile, BCa and t intervals
                         tie=excludes_zero_tie(r)))
    pr = _read("analysis/poprule.csv")
    for r in pr[pr.candidate == "LPSR-18D"].to_dict("records"):
        rows.append(dict(evals_per_variable=r["max_fes"] / r["dim"], dim=r["dim"],
                         budget=r["max_fes"], problems="CEC2022", engine="repaired engine (selection runs)",
                         source_marker="CEC2022 selection data (in-sample)",
                         delta=-r["mean_delta"], ci_lo=-r["ci_hi"], ci_hi=-r["ci_lo"],
                         status="in-sample", source="results/analysis/poprule.csv (E7; LPSR-18D vs "
                         "LPSR-6D, sign reversed)", sign_reversed=True, unit="functions"))
    e = _read("analysis/e16_engineering_population.csv")
    for r in e.to_dict("records"):
        rows.append(dict(evals_per_variable=r["evals_per_variable"], dim=r["dim"], budget=15000,
                         problems=r["problem"].split("_")[0] + (" " + r["problem"].split("_")[-1]
                                                                if "RC" in r["problem"] else ""),
                         engine="L-SHADE", source_marker="E16 design problems",
                         delta=r["delta"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], status="E16",
                         source="results/analysis/e16_engineering_population.csv (E16)",
                         sign_reversed=False, unit="runs"))
    out = pd.DataFrame(rows)
    # rows without BCa / t intervals (E7, E16): the rule on the percentile interval
    miss = out.tie.isna()
    out.loc[miss, "tie"] = [excludes_zero_tie(dict(pct_lo=lo, pct_hi=hi), kinds=("pct",))
                            for lo, hi in zip(out.loc[miss, "ci_lo"], out.loc[miss, "ci_hi"])]
    out["tie"] = out.tie.astype(bool)
    out["favours"] = np.where(out.ci_hi < 0, "6D", np.where(out.ci_lo > 0, "18D", "neither"))
    # an interval that excludes zero with a bound within 0.02 of it is a tie (Section 2.6)
    out.loc[(out.favours != "neither") & out.tie, "favours"] = "tie"
    out["inside_0.147"] = (out.ci_lo > -0.147) & (out.ci_hi < 0.147)
    out = out.sort_values(["evals_per_variable", "engine", "problems"]).reset_index(drop=True)
    for r in out.to_dict("records"):
        print(f"  B/D={r['evals_per_variable']:>8.1f}  D={r['dim']:<3d} {r['problems']:<12s} "
              f"{r['engine']:<34s} {r['delta']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  "
              f"{r['status']:<11s} favours {r['favours']}")
    # the measured bands the guide may state
    meas = out[out.status.isin(["measured", "exploratory", "E16", "in-sample"])]
    low = meas[(meas.evals_per_variable >= 300) & (meas.evals_per_variable <= 1700)]
    high = meas[meas.evals_per_variable >= 10_000]
    print(f"  300-1,700 per variable: {len(low)} rows, delta from {low.delta.min():+.3f} to "
          f"{low.delta.max():+.3f}; favours 6D in {int((low.favours == '6D').sum())}")
    l10k = high[high.engine == "L-SHADE"]
    print(f"  10,000 per variable, L-SHADE: delta from {l10k.delta.min():+.3f} to {l10k.delta.max():+.3f};"
          f" favours 18D in {int((l10k.favours == '18D').sum())} of {len(l10k)}")
    _write(out, "revision_budget_guide.csv")
    return out


# ================================================================== hho (REV-57)
def hho():
    """SEHHO-COBL against HHO in every suite block (REV-57).  SEHHO-COBL-R against HHO
    is in revision_suite_pairwise.csv and revision_cec2014_pairwise.csv (one interval
    per contrast), so it is not recomputed here."""
    print("\n" + "=" * 100)
    print("SEHHO-COBL AGAINST HHO (REV-57); negative favours SEHHO-COBL")
    print("=" * 100)
    rng = np.random.default_rng(SEED_REVISION + 57)
    e9 = _read("E9_newmethod_vs_baselines.csv")
    blocks = [(t, f, tag, dims) for t, (f, tag), dims, _ in SUITE_BLOCKS]
    blocks.append(("cec2014", "E13_cec2014.csv", None, (30, 50)))
    rows = []
    for table, fname, tag, dims in blocks:
        base = _read(fname)
        df = pd.concat([base, e9[e9.tag == tag]], ignore_index=True) if tag else base
        for dim in dims:
            for fes in sorted(df[df.dim == dim].max_fes.unique()):
                sub = df[(df.dim == dim) & (df.max_fes == fes)]
                funcs = sorted(sub.func.unique())
                for a in (PUBLISHED,):
                    v = per_fn_delta(sub, a, "HHO", funcs)
                    r = all_intervals(v, boot_indices(rng, len(v)))
                    r.update(table=table, dim=int(dim), max_fes=int(fes), a=a, b="HHO",
                             p=signed_rank(v), wins=int((v < 0).sum()), losses=int((v > 0).sum()))
                    rows.append(r)
    out = pd.DataFrame(rows)
    for r in out.to_dict("records"):
        _line(f"{DISPLAY.get(r['a'], r['a'])} vs HHO, {r['table']} D={r['dim']} {r['max_fes']:,}", r,
              f"  W/L {r['wins']}/{r['losses']}")
    _write(out, "revision_hho.csv")
    return out


# ================================================================== E15 (secondary family)
def e15():
    """Table S-e15: the registered secondary comparison of E15 (original-rule configuration
    vs the repaired configuration, four blocks) with BCa, Student-t and Bonferroni-adjusted
    intervals over its family of four blocks.  Replicates analyze_e15.py's draws, so the
    percentile intervals are the registered ones; the adjusted intervals are added here
    (they are not part of the registration, whose rule uses the 95% percentile interval)."""
    print("\n" + "=" * 100)
    print("E15 SECONDARY COMPARISON: original-rule configuration vs SEHHO-COBL-R (Table S-e15)")
    print("negative favours the original-rule configuration; Bonferroni family: the four blocks")
    print("=" * 100)
    df = pd.concat([_read("E15_original_rule_cec2014.csv"), _read("E13_cec2014.csv")],
                   ignore_index=True)
    orig = "GF:OriginalRule"
    funcs = sorted(df[df.algo == orig].func.unique())
    rng = np.random.default_rng(SEED_CEC2014)          # analyze_e15.py's generator
    rows = []
    for d in (30, 50):
        for fes in (10_000 * d, 15_000):
            sub = df[(df.dim == d) & (df.max_fes == fes)]
            for a, b in ((orig, PUBLISHED), (METHOD, PUBLISHED), (orig, METHOD)):
                v = per_fn_delta(sub, a, b, funcs)
                idx = boot_indices(rng, len(v))       # consumed for every contrast, in order
                if (a, b) != (orig, METHOD):
                    continue
                r = all_intervals(v, idx, k=4)
                r.update(dim=d, max_fes=fes, budget="competition" if fes > 15_000 else "tight",
                         a=a, b=b, p=signed_rank(v))
                rows.append(r)
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p.to_numpy())
    out = pd.concat([out, pd.DataFrame([_verdicts(r) for r in out.to_dict("records")])], axis=1)
    for kind in ("pct", "bca", "t"):
        out[f"verdict_{kind}_adj_0.147"] = [margin_verdict(r[f"{kind}_adj_lo"], r[f"{kind}_adj_hi"],
                                                           0.147) for r in out.to_dict("records")]
    st = _read("analysis/e15_original_rule.csv")
    st = st[st.family == "secondary"].sort_values(["dim", "max_fes"], ascending=[True, False])
    _check("e15: percentile intervals and p_Holm reproduce e15_original_rule.csv (secondary)",
           out[["mean", "pct_lo", "pct_hi", "p_holm"]].to_numpy(),
           st[["mean_delta", "ci_lo", "ci_hi", "p_holm"]].to_numpy())
    for r in out.to_dict("records"):
        _line(f"D={r['dim']} {r['budget']}", r,
              f"  0.147 unadjusted {r['verdict_pct_0.147']}; adjusted pct "
              f"{r['verdict_pct_adj_0.147']}, t {r['verdict_t_adj_0.147']}")
    _write(out, "revision_e15.csv")
    return out


# ================================================================== direct contrast (Sec. 2.6)
def contrast():
    """The difference of effects that the abstract and Sections 3.2 and 5 state: the
    population effect (N=120 vs N=30, E6) against the gate effect (gate removed vs kept,
    E4b), both at 300,000 evaluations on CEC2017 at D=30, as a direct per-function contrast
    over the 29 functions the two experiments share (Gelman & Stern, 2006).  Seed 20261003."""
    print("\n" + "=" * 100)
    print("DIRECT CONTRAST: population effect minus gate effect, CEC2017 D=30, 300,000 FEs")
    print("negative: raising N from 30 to 120 helps more than removing the gate")
    print("=" * 100)
    e6 = _read("E6_popsize_cec2017_30D.csv")
    e6 = e6[(e6.dim == 30) & (e6.max_fes == 300_000)]
    e4b = _read("E4b_ablation_cec2017_30D.csv")
    funcs = sorted(set(e6.func) & set(e4b.func))
    pop_eff = per_fn_delta(e6, "E6_popsize_N120", "E6_popsize_N30", funcs, col="tag")
    rng = np.random.default_rng(SEED_REVISION)
    rows = []
    for lab, a, b in (("gate removed, Levy off (AlwaysPbest+NoLevy vs NoLevy)",
                       "SEHHO:AlwaysPbest+NoLevy", "SEHHO:NoLevy"),
                      ("gate removed, Levy gated (AlwaysPbest vs Full)",
                       "SEHHO:AlwaysPbest", "SEHHO:Full")):
        gate_eff = per_fn_delta(e4b, a, b, funcs)
        v = pop_eff - gate_eff
        r = all_intervals(v, boot_indices(rng, len(v)), k=2)
        r.update(contrast="delta(N=120 vs N=30) - delta(" + lab + ")", n_functions=len(funcs),
                 population_effect=float(pop_eff.mean()), gate_effect=float(gate_eff.mean()),
                 p=signed_rank(v))
        rows.append(r)
        _line(lab, r, f"  population {pop_eff.mean():+.3f}, gate {gate_eff.mean():+.3f}")
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p.to_numpy())
    _write(out, "revision_effect_contrast.csv")
    return out


def resolution():
    """Precision convention (main text Section 2.6; post hoc, added at a final integrity
    check).  Cliff's delta compares the recorded errors at full double precision, errors
    below 1e-8 being recorded as 0, so on plateau functions runs that agree to about twelve
    significant digits are still ordered.  Each headline contrast is computed both ways --
    as reported, and with every error rounded to 1e-8 first -- from the same resamples
    (seed 20261003, one index matrix per contrast), with the percentile, BCa and Student-t
    intervals and the tie rule of Section 2.6.  The family sizes are those of the reported
    analyses (two blocks for the primary endpoint, four for each row of Table 5)."""
    print("\n" + "=" * 100)
    print("PRECISION CONVENTION: delta at full precision (as reported) and at a 1e-8 resolution")
    print("=" * 100)
    rng = np.random.default_rng(SEED_REVISION)
    cases = []
    e13 = _read("E13_cec2014.csv")
    for dim in (30, 50):
        sub = e13[(e13.tag == "E13_cec2014_competition") & (e13.dim == dim)]
        cases.append((f"primary endpoint, CEC2014 D={dim}", sub, METHOD, PUBLISHED, "algo", 2))
    e14 = _read("E14_pop_isolation.csv")
    for suite in ("cec2017", "cec2014"):
        s0 = e14[e14.suite == suite]
        mine = _method_runs(suite)
        for dim in sorted(s0.dim.unique()):
            for fes in sorted(s0[s0.dim == dim].max_fes.unique()):
                sub = s0[(s0.dim == dim) & (s0.max_fes == fes)]
                merged = pd.concat([sub, mine[(mine.dim == dim) & (mine.max_fes == fes)]], ignore_index=True)
                block = f"{suite} D={dim} {'tight' if fes == 15_000 else 'competition'}"
                for label, frame, a, b in (("L-SHADE 6D vs 18D", sub, "LSHADE-N6", "LSHADE-N18"),
                                           ("ours 6D vs ours 18D", merged, METHOD, "GF@18D"),
                                           ("ours vs L-SHADE at 6D", merged, METHOD, "LSHADE-N6"),
                                           ("ours vs L-SHADE default", merged, METHOD, "LSHADE-N18")):
                    cases.append((f"Table 5: {label}, {block}", frame, a, b, "algo", 4))
    e6 = _read("E6_popsize_cec2017_30D.csv")
    e6 = e6[(e6.dim == 30) & (e6.max_fes == 300_000)]
    cases.append(("E6: N=120 vs N=30, CEC2017 D=30, 300,000", e6, "E6_popsize_N120", "E6_popsize_N30", "tag", 1))
    e4b = _read("E4b_ablation_cec2017_30D.csv")
    cases.append(("E4b: gate removed, Levy off, 300,000", e4b, "SEHHO:AlwaysPbest+NoLevy", "SEHHO:NoLevy", "algo", 1))
    cases.append(("E4b: gate removed, Levy gated, 300,000", e4b, "SEHHO:AlwaysPbest", "SEHHO:Full", "algo", 1))
    rows = []
    for label, frame, a, b, col, k in cases:
        funcs = sorted(frame.func.unique())
        full = per_fn_delta(frame, a, b, funcs, col=col)
        rnd = per_fn_delta(frame, a, b, funcs, col=col, decimals=8)
        idx = boot_indices(rng, len(full))
        rf, rr = all_intervals(full, idx, k=k), all_intervals(rnd, idx, k=k)
        row = dict(contrast=label, arm_a=a, arm_b=b, n_functions=len(funcs),
                   functions_changed=int(np.sum(np.abs(full - rnd) > 0)))
        for tag, r in (("full", rf), ("1e8", rr)):
            row.update({f"{tag}_mean": r["mean"], f"{tag}_pct_lo": r["pct_lo"], f"{tag}_pct_hi": r["pct_hi"],
                        f"{tag}_bca_lo": r["bca_lo"], f"{tag}_bca_hi": r["bca_hi"],
                        f"{tag}_t_lo": r["t_lo"], f"{tag}_t_hi": r["t_hi"],
                        f"{tag}_t_adj_lo": r["t_adj_lo"], f"{tag}_t_adj_hi": r["t_adj_hi"],
                        f"{tag}_sides": "/".join(side(r[f"{q}_lo"], r[f"{q}_hi"]) for q in ("pct", "bca", "t")),
                        f"{tag}_side_adj_t": side(r["t_adj_lo"], r["t_adj_hi"]),
                        f"{tag}_tie": excludes_zero_tie(r)})
        row["mean_shift"] = row["1e8_mean"] - row["full_mean"]
        row["classification_changed"] = (row["full_sides"] != row["1e8_sides"] or row["full_tie"] != row["1e8_tie"]
                                         or row["full_side_adj_t"] != row["1e8_side_adj_t"])
        rows.append(row)
        print(f"    {label:<58s} {row['full_mean']:+.3f} -> {row['1e8_mean']:+.3f} "
              f"[{row['1e8_pct_lo']:+.3f}, {row['1e8_pct_hi']:+.3f}]  sides {row['full_sides']} -> {row['1e8_sides']}"
              f"  tie {row['full_tie']} -> {row['1e8_tie']}{'  CHANGED' if row['classification_changed'] else ''}")
    out = pd.DataFrame(rows)
    print(f"  largest |shift| {out.mean_shift.abs().max():.3f}; classifications changed: "
          f"{int(out.classification_changed.sum())} of {len(out)}")
    _write(out, "revision_resolution.csv")
    return out


SECTIONS = dict(gate=gate, pop=pop, cec2014=cec2014, suites=suites, comp=comp, poprule=poprule,
                popsize=popsize, e16=e16, eng=eng, guide=guide, hho=hho, e15=e15,
                contrast=contrast, resolution=resolution)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    names = argv or list(SECTIONS)
    bad = [n for n in names if n not in SECTIONS]
    if bad:
        raise SystemExit(f"unknown section(s) {bad}; choose from {list(SECTIONS)}")
    for n in names:
        SECTIONS[n]()
    print("\n" + "=" * 100)
    print("REPRODUCTION CHECKS")
    for what, ok, detail in CHECKS:
        print(f"  [{'OK  ' if ok else 'FAIL'}] {what}" + ("" if ok else f"\n         {detail}"))
    return 0 if all(ok for _, ok, _ in CHECKS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
