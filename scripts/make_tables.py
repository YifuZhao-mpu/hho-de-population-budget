"""Emit the manuscript's LaTeX tables directly from the raw per-run results.

Nothing is transcribed by hand: every table in the paper and its supplement is
\\input{} from a file written here, so the text cannot drift from the data.

Revision conventions (revision plan after an internal pre-submission review;
REV-xx are its items, see the README, "The word 'review' in file comments"):

* Display name.  The configuration obtained by dismantling SEHHO-COBL is
  "GF-Method" in every raw CSV, because the seed of every run is a hash that
  includes that string; renaming it in the data would make the released seeds
  irreproducible.  Tables show it as SEHHO-COBL-R (REV-42).  L-SHADE is
  "LSHADE" in the data for the same reason.
* Sign convention (REV-13).  delta is Cliff's delta of arm A against arm B, A
  being the changed, new or first-named arm; negative favours A.  Every table
  note states it.  Where a source file stores the other order (the ablation
  tables, and the sources of the budget guide), the value is reversed here and
  the delta ledger says so.
* Delta ledger.  Every delta a table prints is recorded, with its arm order and
  source file, in results/analysis/revision_delta_ledger.csv.
* Cross-references.  Main text and supplement are separate documents linked by
  xr (\\externaldocument[S-]{supplementary} in the main text,
  \\externaldocument[M-]{main} in the supplement); a table that refers to a
  float in the other document uses the prefixed label.

Inputs: the raw results/*.csv, the stored analyses in results/analysis/, and
the revision analyses written by analyze_revision.py, classical_references.py
and check_cec_data.py (run those first; see the README).

Run:  python3 scripts/make_tables.py
"""

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import compare_block, paired_vs_full  # noqa: E402
from sehho.runfiles import read_runs  # noqa: E402  (pandas.read_csv; SEHHO_ERROR_DECIMALS, Section 2.6)
from sehho.stats import friedman, cliffs_delta  # noqa: E402
from sehho.intervals import margin_verdict  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results")
ANA = os.path.join(RES, "analysis")
TEX = os.path.join(RES, "tex")
os.makedirs(TEX, exist_ok=True)
INTERNAL_NAME = "GF-Method"
METHOD = "SEHHO-COBL-R"
DISPLAY = {INTERNAL_NAME: METHOD, "LSHADE": "L-SHADE"}
CONTROL = METHOD
ORDER = [METHOD, "jSO", "L-SHADE", "SHADE", "DE", "SEHHO-COBL", "HHO"]
COMPETITORS_INTERNAL = ["jSO", "LSHADE", "SHADE", "DE", "SEHHO-COBL", "HHO"]
MARGINS = (0.147, 0.10, 0.05)
MARGIN_TXT = {0.147: "0.147", 0.10: "0.10", 0.05: "0.05"}
RC01_ANALYTIC = 35.0 * (50.0 / 3.0) ** 0.6
NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
                7: "seven", 8: "eight"}

CONV = (r"$\delta$ is Cliff's delta of arm A against arm B, A being the changed, new or "
        r"first-named arm; negative values favour A.")
CONV_FN = (r"$\delta$ is Cliff's delta of arm A against arm B (A the changed, new or "
           r"first-named arm), computed per function over the 30 runs of each arm and "
           r"averaged over the functions; negative values favour A.")

# Where each float lives after the revision: "main" or "supp".  A reference
# from a table to a float in the other document gets the xr prefix.
LOCATION = {
    "tab:gateinference": "main", "tab:popsize": "main", "tab:popisolation": "main",
    "tab:cec2014primary": "main", "tab:engineering": "main", "tab:e16": "main",
    "tab:engbudget": "main", "tab:solverbaseline": "main", "tab:guide": "main",
    "eq:rc01star_main": "main", "fig:budgetguide": "main", "fig:ablation": "main",
    "tab:ablation": "supp", "tab:gatefactorial": "supp", "tab:supp_gate_intervals": "supp",
    "tab:supp_ablation_cec2022": "supp", "tab:poprule": "supp", "tab:composition": "supp",
    "tab:supp_composition_intervals": "supp", "tab:supp_e15": "supp", "tab:cec2014": "supp",
    "tab:cec2017": "supp", "tab:cec2017_tight": "supp", "tab:cec2022_matched": "supp",
    "tab:cec2022_competition": "supp", "tab:supp_popisolation": "supp",
    "tab:supp_classical_refs": "supp", "tab:supp_e16": "supp", "tab:supp_singular": "supp",
    "tab:supp_lineage": "supp", "tab:supp_precision": "supp", "tab:supp_validation": "supp",
    "tab:supp_hho": "supp",
    # sections a main-text table note points to (protocol provenance lives in S2.5)
    "subsec:stats": "main", "sec:eng_population": "main", "sec:S2_prereg": "supp",
}
WRITTEN = []          # (file, label, home) of every table written
LEDGER = []           # every delta printed


# ------------------------------------------------------------------ helpers
def load_comparison(main_csv, extra_csv, tag):
    """Baselines from the original stage, plus SEHHO-COBL-R's runs from E9/E12."""
    frames = []
    if os.path.exists(main_csv):
        frames.append(read_runs(main_csv))
    if os.path.exists(extra_csv):
        e = read_runs(extra_csv)
        frames.append(e[e.tag == tag] if "tag" in e.columns else e)
    if not frames:
        return None
    d = pd.concat(frames, ignore_index=True)
    d["algo"] = d.algo.replace(DISPLAY)
    return d


def _ana(name):
    p = os.path.join(ANA, name)
    if not os.path.exists(p):
        raise SystemExit(f"{p} not found: run scripts/analyze_revision.py (and "
                         f"classical_references.py, check_cec_data.py) first")
    return read_runs(p)


def sci(v, prec=2):
    if v is None or not np.isfinite(v):
        return "--"
    if v == 0:
        return "0.00E+00"
    return f"{v:.{prec}E}"


def texp(v, prec=2):
    if v == 0:
        return "0"
    m, e = f"{v:.{prec}E}".split("E")
    return f"{m}\\times10^{{{int(e)}}}"


def ptex(p):
    """p-value cell: three decimals, or "<0.001" (exact values are in the CSVs)."""
    if not np.isfinite(p):
        return "--"
    if p < 0.001:
        return r"$<$0.001"
    return f"{p:.3f}"


def stack(top, bottom):
    """A cell with a second, smaller line (an interval under its estimate)."""
    return (r"\begin{tabular}[c]{@{}c@{}}" + top + r"\\[-2pt]{\scriptsize " + bottom +
            r"}\end{tabular}")


def write(name, body, label, home):
    p = os.path.join(TEX, name)
    with open(p, "w") as fh:
        fh.write(body)
    WRITTEN.append((name, label, home))
    print("wrote", p)


def esc(s):
    return str(s).replace("_", r"\_").replace("&", r"\&")


def ref(label, home):
    """\\ref to `label` from a table living in `home`, with the xr prefix if needed."""
    where = LOCATION.get(label, home)
    if where == home:
        return f"\\ref{{{label}}}"
    return f"\\ref{{{'S-' if where == 'supp' else 'M-'}{label}}}"


def fnum(n):
    return f"{int(n):,}".replace(",", "{,}")


def dfmt(v, prec=3):
    return f"{v:+.{prec}f}"


def cfmt(lo, hi, prec=3):
    return f"[{lo:+.{prec}f},\\,{hi:+.{prec}f}]"


def led(table, row, col, value, a, b, source, reversed_=False, note=""):
    """Record one printed delta in the ledger and return it formatted."""
    LEDGER.append(dict(table_file=table, label=_current_label(table), row=row, column=col,
                       value=float(value), printed=dfmt(value), arm_a=a, arm_b=b,
                       negative_favours=a, source=source,
                       sign_reversed_relative_to_source=bool(reversed_), note=note))
    return dfmt(value)


_LABELS = {}


def _current_label(table):
    return _LABELS.get(table, "")


def robust_mark(r, kind="all"):
    """'' if the zero-exclusion verdict of the 95% percentile interval also holds
    under the BCa and Student-t intervals and under the family adjustment;
    a dagger-like mark otherwise (only for intervals that exclude zero)."""
    if r["side_pct"] == "spans":
        return ""
    if r["side_all"] != r["side_pct"]:
        return r"$^{\S}$"
    if r["side_all_adj"] != r["side_pct"]:
        return r"$^{\ddagger}$"
    return ""


def excludes_zero_tie(r, kinds=("pct", "bca", "t"), tol=0.02):
    """Tie rule of main text Section 2.6: some unadjusted interval (percentile, BCa or
    Student-t) excludes zero with its bound nearest zero within `tol` of it.  The
    `tie_002` column of the analysis files is broader (it also flags an interval that
    spans zero with a bound near zero); only intervals that exclude zero are ties."""
    for k in kinds:
        lo, hi = r[f"{k}_lo"], r[f"{k}_hi"]
        if (0 < lo < tol) or (-tol < hi < 0):
            return True
    return False


def tied_best(means):
    """Indices of every algorithm whose mean equals the lowest mean of the row exactly.

    Exact ties for the lowest mean (all 0, or the same plateau value) are credited to
    every tied algorithm, in the "#1" counts and in the bold marks alike."""
    m = np.asarray(means, dtype=float)
    return set(np.flatnonzero(m == m.min()).tolist())


def tied_best_counts(summary, algos):
    """"#1" counts from compare_block's per-function means, ties credited to all tied."""
    M = summary[[f"{a}_mean" for a in algos]].to_numpy(float)
    best = M == M.min(axis=1, keepdims=True)
    return {a: int(best[:, j].sum()) for j, a in enumerate(algos)}


TIE_NOTE = (r" Exact ties for the lowest mean are credited to every tied algorithm, so a column "
            r"can sum to more than the number of functions.")
WTL_NOTE = (r" A function on which the two samples agree run by run to within a relative "
            r"$10^{-5}$ (\texttt{numpy.allclose}) is counted as not different without a test.")


def verdict_code(v):
    return {"negligible": "negl.", "inconclusive": "incon.", "nonzero": r"$\neq0$",
            "tie": "tie"}[v]


# ================================================================== S5.2, S5.3: rank tables
def table_ranks(df, label, caption, fname, table_key, home="supp", dims=None):
    """Friedman ranks, delta with its interval and Holm-corrected signed-rank tests on
    per-function delta (REV-29) against SEHHO-COBL-R, one block per dimension."""
    _LABELS[fname] = f"tab:{label}"
    pw = _ana("revision_suite_pairwise.csv")
    pw = pw[pw.table == table_key]
    dims = dims or sorted(df.dim.unique())
    blocks = {d: compare_block(df[df.dim == d], control=CONTROL, label=f"{label} D={d}")
              for d in dims}
    algos = [a for a in ORDER if a in set(df.algo)]
    n1 = {d: tied_best_counts(blocks[d]["summary"], list(blocks[d]["ranks"].algorithm))
          for d in dims}
    lines = [r"\begin{table}[htbp]", r"\centering", f"\\caption{{{caption}}}",
             f"\\label{{tab:{label}}}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{l" + "ccccc" * len(dims) + "}", r"\toprule"]
    lines.append(" & " + " & ".join(f"\\multicolumn{{5}}{{c}}{{$D={d}$, "
                                    f"{fnum(df[df.dim == d].max_fes.iloc[0])} FEs}}"
                                    for d in dims) + r" \\")
    lines.append(" ".join(f"\\cmidrule(lr){{{2 + 5 * i}-{6 + 5 * i}}}" for i in range(len(dims))))
    lines.append("Algorithm & " + " & ".join(
        [r"Rank & \#1 & $\bar\delta$ [95\% CI] & $p_{\text{Holm}}$ & W/T/L" for _ in dims]) + r" \\")
    lines.append(r"\midrule")
    for a in algos:
        cells = []
        for d in dims:
            res = blocks[d]
            r = res["ranks"][res["ranks"].algorithm == a].iloc[0]
            if a == CONTROL:
                cells += [f"\\textbf{{{r.avg_rank:.2f}}}", f"\\textbf{{{n1[d][a]}}}",
                          "--", "--", "--"]
                continue
            ph = res["posthoc"][res["posthoc"].algorithm == a].iloc[0]
            inner = {v: k for k, v in DISPLAY.items()}.get(a, a)
            q = pw[(pw.dim == d) & (pw.opponent == inner)].iloc[0]
            star = "$^{*}$" if q.p_holm < 0.05 else ""
            dv = led(fname, a, f"D={d}", q["mean"], METHOD, a,
                     "results/analysis/revision_suite_pairwise.csv")
            cells += [f"{r.avg_rank:.2f}", f"{n1[d][a]}",
                      f"${dv}$ ${cfmt(q.pct_lo, q.pct_hi)}$", ptex(q.p_holm) + star,
                      ph.win_tie_loss]
        lines.append(f"{esc(a)} & " + " & ".join(cells) + r" \\")
    stats = "; ".join(
        f"$D={d}$: Iman--Davenport $F={blocks[d]['meta']['iman_davenport_F']:.2f}$, "
        f"$p={texp(blocks[d]['meta']['friedman_p'])}$" for d in dims)
    lines += [r"\bottomrule", r"\end{tabular}}", r"\begin{flushleft}\footnotesize",
              f"Friedman test over {blocks[dims[0]]['meta']['n_problems']} functions "
              f"({stats}). ``Rank'' is the Friedman average rank (lower is better) and "
              r"``\#1'' the number of functions on which the algorithm has the lowest mean "
              r"error." + TIE_NOTE + r" The bold row is the reference, " + METHOD + r", not the "
              r"best algorithm. " + CONV_FN + r" Here A is " + METHOD + r" and B the competitor "
              r"in the row. The interval is a 95\% percentile bootstrap interval over the "
              r"functions (20{,}000 resamples), and $p_{\text{Holm}}$ is a two-sided "
              r"signed-rank test of the per-function deltas against zero, Holm-corrected "
              r"over the six competitors of the block; $^{*}$ marks $p_{\text{Holm}}<0.05$. "
              r"Signed-rank tests on per-function deltas are used rather than Friedman "
              r"post-hoc tests against a control, whose outcome depends on which other "
              r"algorithms are in the set. "
              r"W/T/L counts functions on which " + METHOD + r" is significantly better / "
              r"not significantly different / significantly worse under two-sided "
              r"rank-sum tests on the runs, Holm-corrected across all functions $\times$ "
              r"competitors." + WTL_NOTE,
              r"\end{flushleft}", r"\end{table}", ""]
    write(fname, "\n".join(lines), f"tab:{label}", home)
    return blocks


# ================================================================== S3.1, S3.4: ablation
def table_ablation(df, control, caption, fname, label, blocks, home="supp", note_extra=""):
    """Twelve-variant ablation; delta of the variant (changed arm) against the full method."""
    _LABELS[fname] = f"tab:{label}"
    parts = []
    for dim_label, sub in blocks:
        fr, out = paired_vs_full(sub, control, label=dim_label)
        parts.append((dim_label, fr, out))
    lines = [r"\begin{table}[htbp]", r"\centering", f"\\caption{{{caption}}}",
             f"\\label{{tab:{label}}}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{l" + "ccccc" * len(parts) + "}", r"\toprule"]
    lines.append(" & " + " & ".join(
        [f"\\multicolumn{{5}}{{c}}{{{dl}}}" for dl, _, _ in parts]) + r" \\")
    lines.append(" ".join([f"\\cmidrule(lr){{{2 + 5 * i}-{6 + 5 * i}}}" for i in range(len(parts))]))
    lines.append("Variant & " + " & ".join(
        ["Rank & W/L & $p_{\\text{Holm}}$ & $\\delta$ & Mag." for _ in parts]) + r" \\")
    lines.append(r"\midrule")
    full_ranks = [p[2]["full_rank"].iloc[0] for p in parts]
    lines.append(r"\textbf{Full method} & " + " & ".join(
        [f"\\textbf{{{fr:.2f}}} & -- & -- & -- & --" for fr in full_ranks]) + r" \\")
    lines.append(r"\midrule")
    order = list(parts[-1][2]["variant"])
    for v in order:
        cells = []
        for dl_, _, out in parts:
            r = out[out.variant == v]
            if not len(r):
                cells += ["--"] * 5
                continue
            r = r.iloc[0]
            star = "$^{*}$" if r.significant_holm else ""
            # paired_vs_full returns delta(full, variant); the variant is the changed arm
            dv = led(fname, v.replace("SEHHO:", ""), dl_, -r.cliffs_delta,
                     v.replace("SEHHO:", ""), "Full",
                     "analyze.paired_vs_full on " + ("E4c/E4b" if "CEC2017" in dl_ else "E4"),
                     reversed_=True)
            # W/L from the variant's side: functions on which it has the lower / higher mean
            cells += [f"{r.avg_rank:.2f}", f"{int(r.losses)}/{int(r.wins)}",
                      f"{r.p_holm:.3f}{star}", f"${dv}$", r.magnitude]
        lines.append(f"{esc(v.replace('SEHHO:', ''))} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\begin{flushleft}\footnotesize",
              r"Friedman average rank over all variants (lower is better); the bold row is "
              r"the reference (the full method), not the best variant. " + CONV_FN +
              r" Here A is the variant in the row and B the full method, so a positive "
              r"$\delta$ means the change hurts and a negative one that it helps. W/L counts "
              r"functions on which the variant has the lower / higher mean error. "
              r"$p_{\text{Holm}}$ is a two-sided Wilcoxon signed-rank test of the variant "
              r"against the full method across matched per-function mean errors, "
              r"Holm-corrected over all variants of the block; $^{*}$ marks $p<0.05$. "
              r"Magnitudes follow \citet{romano2006appropriate}. Rows are ordered by the rightmost "
              r"block." + note_extra,
              r"\end{flushleft}", r"\end{table}", ""]
    write(fname, "\n".join(lines), f"tab:{label}", home)
    return parts


# ================================================================== S3.2: factorial ranks
def table_gate_factorial(caption, fname, home="supp"):
    """The 2x2 that separates the escape-energy gate from the Levy perturbation."""
    import importlib.util
    import io
    import contextlib
    _LABELS[fname] = "tab:gatefactorial"
    spec = importlib.util.spec_from_file_location(
        "gf", os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze_gate_factorial.py"))
    gf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gf)
    blocks = []
    for label, fn in gf.BLOCKS:
        d = read_runs(os.path.join(RES, fn))
        d = d[d.dim == 30]
        with contextlib.redirect_stdout(io.StringIO()):
            rows = gf.analyse(d, label)
        funcs = sorted(d.func.unique())
        fr = friedman(gf.mean_matrix(d, list(gf.CELLS.values()), funcs))
        blocks.append((label, dict(zip(gf.CELLS.values(), fr["avg_ranks"])), rows))
    lines = [r"\begin{table}[htbp]", r"\centering", f"\\caption{{{caption}}}",
             r"\label{tab:gatefactorial}", r"\begin{tabular}{ll" + "cc" * len(blocks) + "}",
             r"\toprule"]
    lines.append(" & & " + " & ".join(
        f"\\multicolumn{{2}}{{c}}{{{esc(l).replace(',', '{,}')}}}" for l, _, _ in blocks) + r" \\")
    lines.append(" ".join(f"\\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(blocks))))
    lines.append("Gate & L\\'evy & " + " & ".join("Rank & $\\Delta$ rank" for _ in blocks) + r" \\")
    lines.append(r"\midrule")
    names = {("on", "gated"): ("escape energy", "gated (4.6\\%)"),
             ("on", "off"): ("escape energy", "off"),
             ("off", "gated"): ("none ($q\\equiv1$)", "always (30\\%)"),
             ("off", "off"): ("none ($q\\equiv1$)", "off")}
    for key in (("on", "gated"), ("on", "off"), ("off", "gated"), ("off", "off")):
        g, lv = names[key]
        cells = []
        for _, rank, _ in blocks:
            r = rank[gf.CELLS[key]]
            base = rank[gf.CELLS[("on", key[1])]]
            cells += [f"{r:.2f}", ("--" if key[0] == "on" else f"${r - base:+.2f}$")]
        lines.append(f"{g} & {lv} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{2}{l}{\emph{effect of a change}} & "
                 + " & ".join(r"$p_{\text{Holm}}$ & $\delta$" for _ in blocks) + r" \\")
    for key, tag in (("remove gate, Levy gated", r"remove gate, L\'evy gated"),
                     ("remove gate, Levy off", r"remove gate, L\'evy off"),
                     ("remove Levy, gate on", r"remove L\'evy, gate on"),
                     ("remove Levy, gate off", r"remove L\'evy, gate off")):
        cells = []
        for blabel, _, rows in blocks:
            rr = rows[rows.effect == key].iloc[0]
            star = "$^{*}$" if rr.significant else ""
            dv = led(fname, key, blabel, rr.delta, key.split(",")[0].replace("remove ", "") +
                     " removed", "kept", "results/analysis/gate_levy_factorial.csv")
            cells += [f"{rr.p_holm:.3f}{star}", f"${dv}$"]
        lines.append(f"\\multicolumn{{2}}{{l}}{{{tag}}} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    cells = []
    for _, _, rows in blocks:
        a = rows[rows.effect == "remove gate, Levy gated"].iloc[0].rank_change
        b = rows[rows.effect == "remove gate, Levy off"].iloc[0].rank_change
        cells.append(f"\\multicolumn{{2}}{{c}}{{${a - b:+.2f}$ rank}}")
    lines.append(r"\multicolumn{2}{l}{\emph{gate}$\times$\emph{L\'evy interaction}} & "
                 + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\begin{flushleft}\footnotesize",
              r"CEC2017 at $D=30$, 29 functions, 30 runs; suite, dimension, population "
              r"size and seed protocol are identical across all cells, so budget is the "
              r"only difference between the two blocks. Ranks are Friedman averages within "
              r"the $2\times2$ (lower is better); $\Delta$ rank is the change from removing "
              r"the gate at that L\'evy setting. $p_{\text{Holm}}$ is a two-sided signed-rank "
              r"test over matched per-function mean errors, corrected across the four "
              r"effects in each block; $^{*}$ marks $p<0.05$. " + CONV_FN + r" Here A is the "
              r"configuration with the component removed. The interaction row is the "
              r"difference between the two gate effects in rank, a point estimate only. "
              r"Interval estimates for every contrast, the interactions with L\'evy and with "
              r"budget, and the equivalence verdicts are in Table~"
              + ref("tab:gateinference", home) + r".",
              r"\end{flushleft}", r"\end{table}", ""]
    write(fname, "\n".join(lines), "tab:gatefactorial", home)


# ================================================================== Table 3
GATE_LAB = {"gate effect, Levy gated": (r"with L\'evy gated", 0),
            "gate effect, Levy off": (r"with L\'evy off", 0),
            "Levy effect, gate on": (r"with the gate on", 1),
            "Levy effect, gate off": (r"with the gate off", 1),
            "interaction gate x Levy": (r"gate $\times$ L\'evy", 2),
            "interaction gate x budget, Levy gated": (r"gate $\times$ budget, L\'evy gated", 2),
            "interaction gate x budget, Levy off": (r"gate $\times$ budget, L\'evy off", 2)}
GATE_HEADS = {0: r"\emph{Removing the gate}, at a fixed L\'evy setting",
              1: r"\emph{Removing the L\'evy perturbation}, at a fixed gate setting",
              2: r"\emph{Interactions}: does one effect depend on the other factor?"}
GATE_BLK = {"15,000 FEs": "15{,}000", "300,000 FEs": "300{,}000",
            "300,000 FEs - 15,000 FEs": "difference"}
# full names, for the notes that list contrasts (REV-26 decision rule)
GATE_NAME = {"gate effect, Levy gated": r"the gate with L\'evy gated",
             "gate effect, Levy off": r"the gate with L\'evy off",
             "Levy effect, gate on": r"the L\'evy perturbation with the gate on",
             "Levy effect, gate off": r"the L\'evy perturbation with the gate off",
             "interaction gate x Levy": r"the gate $\times$ L\'evy interaction",
             "interaction gate x budget, Levy gated": r"the budget interaction with L\'evy gated",
             "interaction gate x budget, Levy off": r"the budget interaction with L\'evy off"}
GATE_AT = {"15,000 FEs": r" at 15{,}000 evaluations", "300,000 FEs": r" at 300{,}000",
           "300,000 FEs - 15,000 FEs": ""}


def _adj_t_verdict(r, m):
    """Margin verdict of the Bonferroni-adjusted (family) Student-t interval."""
    return margin_verdict(r["t_adj_lo"], r["t_adj_hi"], m)


def _gate_adjustment_note(rows, home, flips):
    """The sentence that lists every verdict of Table 3 that the family adjustment
    overturns (REV-26: a claim that holds only under some intervals is reported as
    such and listed).  `flips` holds (quantity, block, margin, verdict, adjusted)."""
    if not flips:
        return (r" Every verdict also holds under the Bonferroni-adjusted (98.75\%) "
                r"Student-$t$ interval of its family of four (Table~"
                + ref("tab:supp_gate_intervals", home) + r").")
    zero = sorted({(q, b) for q, b, m, v, va in flips if v == "nonzero"})
    negl = [(q, b, m) for q, b, m, v, va in flips if v == "negligible"]
    parts = []
    for q, b in zero:
        r = rows.loc[(q, b)]
        parts.append(f"{GATE_NAME[q]}{GATE_AT[b]}, whose adjusted intervals include zero "
                     f"(Student-$t$ ${cfmt(r.t_adj_lo, r.t_adj_hi)}$, percentile "
                     f"${cfmt(r.pct_adj_lo, r.pct_adj_hi)}$)")
    by_m = {}
    for q, b, m in negl:
        r = rows.loc[(q, b)]
        by_m.setdefault(m, []).append(f"{GATE_NAME[q]}{GATE_AT[b]} "
                                      f"(${cfmt(r.t_adj_lo, r.t_adj_hi)}$)")
    if by_m:
        n = len(negl)
        txt = (NUMBER_WORDS.get(n, str(n)) + " verdict" + ("s" if n > 1 else "")
               + r" of negligibility, " + "; ".join(
                   f"at {MARGIN_TXT[m]} " + ", ".join(v[:-1]) + (" and " if len(v) > 1 else "")
                   + v[-1]
                   for m, v in sorted(by_m.items(), key=lambda kv: -kv[0])))
        parts.append(txt)
    return (r" $^{\dagger}$: the verdict does not survive the family adjustment; under the "
            r"Bonferroni-adjusted (98.75\%) Student-$t$ interval of its family of four "
            r"(Table~" + ref("tab:supp_gate_intervals", home) + r") it is inconclusive. "
            r"This applies to " + "; and to ".join(parts) + r". Every other verdict holds "
            r"under the adjusted intervals as well.")


def _gate_arms(q):
    if q.startswith("gate effect"):
        return "gate removed", "gate kept"
    if q.startswith("Levy effect"):
        return "Levy removed", "Levy kept"
    if "budget" in q:
        return "gate effect at 300,000", "gate effect at 15,000"
    return "gate effect, Levy gated", "gate effect, Levy off"


def table_gate_inference(caption, fname, home="main"):
    """Table 3: gate x Levy factorial with intervals, both scales and three margins."""
    _LABELS[fname] = "tab:gateinference"
    d = _ana("revision_gate_inference.csv")
    lines = [r"\begin{table}[htbp]", r"\centering", f"\\caption{{{caption}}}",
             r"\label{tab:gateinference}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{llrccccccc}", r"\toprule",
             r" & & & \multicolumn{2}{c}{95\% interval} & \multicolumn{2}{c}{$p_{\text{Holm}}$} "
             r"& \multicolumn{3}{c}{verdict at $|\delta|<$} \\",
             r"\cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-10}",
             r"Contrast & Budget & $\bar\delta$ & percentile & Student-$t$ & $\delta$ scale & "
             r"mean errors & 0.147 & 0.10 & 0.05 \\", r"\midrule"]
    differs = []
    flips = []          # verdicts the Bonferroni-adjusted Student-t interval overturns
    ties = []           # zero-excluding intervals with a bound within 0.02 of zero (REV-26)
    for grp in (0, 1, 2):
        lines.append(f"\\multicolumn{{10}}{{l}}{{{GATE_HEADS[grp]}}} \\\\")
        for r in d.to_dict("records"):
            if GATE_LAB[r["quantity"]][1] != grp:
                continue
            a, b = _gate_arms(r["quantity"])
            dv = led(fname, r["quantity"], r["block"], r["mean"], a, b,
                     "results/analysis/revision_gate_inference.csv")
            star = "$^{*}$" if r["p_holm"] < 0.05 else ""
            pme = (ptex(r["p_holm_mean_error"]) + ("$^{*}$" if r["p_holm_mean_error"] < 0.05 else "")
                   if np.isfinite(r["p_holm_mean_error"]) else "--")
            # main text Section 2.6: a bound within 0.02 of zero is treated as a tie, so an
            # interval that excludes zero only by that much is reported as a tie, not as nonzero
            tie = bool(r["tie_002"]) and r["side_pct"] != "spans"
            if tie:
                ties.append((r["quantity"], r["block"]))
            ver = []
            for m in MARGINS:
                v_all = r[f"verdict_all_{m:g}"]
                if v_all != r[f"verdict_pct_{m:g}"]:
                    differs.append((r["quantity"], r["block"], m))
                if tie and v_all == "nonzero":
                    ver.append(verdict_code("tie"))
                    continue
                v_adj = _adj_t_verdict(r, m)
                mark = ""
                if v_adj != v_all:
                    flips.append((r["quantity"], r["block"], m, v_all, v_adj))
                    mark = r"$^{\dagger}$"
                ver.append(verdict_code(v_all) + mark)
            lines.append(f"\\quad {GATE_LAB[r['quantity']][0]} & {GATE_BLK[r['block']]} & "
                         f"${dv}$ & ${cfmt(r['pct_lo'], r['pct_hi'])}$ & "
                         f"${cfmt(r['t_lo'], r['t_hi'])}$ & {ptex(r['p_holm'])}{star} & {pme} & "
                         + " & ".join(ver) + r" \\")
        if grp < 2:
            lines.append(r"\midrule")
    rows = d.set_index(["quantity", "block"])
    # the verdicts the family adjustment overturns, listed from the data (REV-26)
    adj = _gate_adjustment_note(rows, home, flips)
    tie_txt = ""
    if ties:
        tl = []
        for q, b in ties:
            r = rows.loc[(q, b)]
            tl.append(f"{GATE_NAME[q]}{GATE_AT[b]} (percentile ${cfmt(r.pct_lo, r.pct_hi)}$, "
                      f"Student-$t$ ${cfmt(r.t_lo, r.t_hi)}$), whose Bonferroni-adjusted "
                      f"intervals also include zero (Student-$t$ ${cfmt(r.t_adj_lo, r.t_adj_hi)}$, "
                      f"percentile ${cfmt(r.pct_adj_lo, r.pct_adj_hi)}$)")
        only_budget_off = all(q == "interaction gate x budget, Levy off" for q, _ in ties)
        tie_txt = (r" A tie is reported for " + "; ".join(tl) + r": the unadjusted intervals "
                   r"exclude zero, but the bound nearest zero lies within 0.02 of it, which "
                   r"Section~" + ref("subsec:stats", home) + r" treats as a tie"
                   + (r", so the budget dependence is not established with L\'evy off"
                      if only_budget_off else "") + ".")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\begin{flushleft}\footnotesize",
              r"CEC2017 at $D=30$, 29 functions, 30 runs per cell; suite, dimension, "
              r"population size and seed protocol are identical everywhere, so the two budget "
              r"blocks differ only in budget. " + CONV_FN + r" Here A is the configuration "
              r"with the component removed (for the interactions: the first-named gate "
              r"effect minus the second). $\bar\delta$ is the mean over the 29 functions; the "
              r"intervals are a 95\% percentile bootstrap interval over functions (20{,}000 "
              r"resamples) and a Student-$t$ interval ($t_{28}$). $p_{\text{Holm}}$ on the "
              r"$\delta$ scale is a two-sided signed-rank test of the 29 per-function deltas, "
              r"Holm-corrected within each of the three families of four; on mean errors it "
              r"is the signed-rank test of Table~" + ref("tab:gatefactorial", home) + r" on "
              r"per-function mean errors, Holm-corrected over the four effects of a budget "
              r"(not defined for the interactions); $^{*}$ marks $p_{\text{Holm}}<0.05$. The "
              r"two scales disagree on one contrast, the gate with L\'evy off at 300{,}000 "
              f"evaluations ({ptex(rows.loc[('gate effect, Levy off', '300,000 FEs')].p_holm)} "
              f"against {ptex(rows.loc[('gate effect, Levy off', '300,000 FEs')].p_holm_mean_error)}). "
              r"Verdicts: negl.\ = the whole interval lies inside $(-m,+m)$; $\neq0$ = the "
              r"interval excludes zero; incon.\ = it includes zero and reaches outside the "
              r"band" + (r"; tie = it excludes zero with a bound within 0.02 of zero" if ties
                         else "") + r". Each verdict holds for the unadjusted percentile, BCa "
              r"and Student-$t$ intervals alike" + ("" if not differs else
                                                  " except where marked") + r". The margin "
              r"0.147 is the "
              r"threshold below which \citet{romano2006appropriate} call an effect negligible in "
              r"survey data; it is a convention, not a property of optimiser error, which is "
              r"why the verdicts at 0.10 and 0.05 are given beside it. The equivalence "
              r"margin and the $\delta$-scale analysis were added after the mean-error "
              r"results existed." + tie_txt + adj,
              r"\end{flushleft}", r"\end{table}", ""]
    write(fname, "\n".join(lines), "tab:gateinference", home)


def table_supp_gate_intervals(caption, fname, home="supp"):
    """S3.3: every interval type, plain and Bonferroni-adjusted, with the margin verdicts."""
    _LABELS[fname] = "tab:supp_gate_intervals"
    d = _ana("revision_gate_inference.csv")
    lines = [r"\begin{table}[htbp]", r"\centering", f"\\caption{{{caption}}}",
             r"\label{tab:supp_gate_intervals}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{llrccccccccc}", r"\toprule",
             r" & & & \multicolumn{3}{c}{95\%} & \multicolumn{2}{c}{98.75\% (Bonferroni, $k=4$)}"
             r" & & \multicolumn{3}{c}{verdicts at 0.147 / 0.10 / 0.05} \\",
             r"\cmidrule(lr){4-6} \cmidrule(lr){7-8} \cmidrule(lr){10-12}",
             r"Contrast & Budget & $\bar\delta$ & percentile & BCa & Student-$t$ & percentile & "
             r"Student-$t$ & $p$ & percentile & BCa / Student-$t$ & Student-$t$, 98.75\% \\",
             r"\midrule"]
    for grp in (0, 1, 2):
        lines.append(f"\\multicolumn{{12}}{{l}}{{{GATE_HEADS[grp]}}} \\\\")
        for r in d.to_dict("records"):
            if GATE_LAB[r["quantity"]][1] != grp:
                continue
            a, b = _gate_arms(r["quantity"])
            dv = led(fname, r["quantity"], r["block"], r["mean"], a, b,
                     "results/analysis/revision_gate_inference.csv")
            vp = "/".join(verdict_code(r[f"verdict_pct_{m:g}"]) for m in MARGINS)
            vb = "/".join(verdict_code(r[f"verdict_bca_{m:g}"]) for m in MARGINS)
            vt = "/".join(verdict_code(r[f"verdict_t_{m:g}"]) for m in MARGINS)
            va = "/".join(verdict_code(_adj_t_verdict(r, m)) for m in MARGINS)
            lines.append(f"\\quad {GATE_LAB[r['quantity']][0]} & {GATE_BLK[r['block']]} & ${dv}$ & "
                         f"${cfmt(r['pct_lo'], r['pct_hi'])}$ & ${cfmt(r['bca_lo'], r['bca_hi'])}$ & "
                         f"${cfmt(r['t_lo'], r['t_hi'])}$ & "
                         f"${cfmt(r['pct_adj_lo'], r['pct_adj_hi'])}$ & "
                         f"${cfmt(r['t_adj_lo'], r['t_adj_hi'])}$ & {r['p']:.4f} & {vp} & "
                         f"{vb if vb == vt else vb + ' ; ' + vt} & {va} \\\\")
        if grp < 2:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\begin{flushleft}\footnotesize",
              r"The contrasts of Table~" + ref("tab:gateinference", home) + r" with every "
              r"interval computed. " + CONV_FN + r" Inference is conditional on the 29 "
              r"CEC2017 functions treated as exchangeable instances of the class the suite "
              r"represents. Percentile and BCa intervals are read from the same 20{,}000 "
              r"bootstrap resamples of the functions (seed 20260913, drawn in the order of "
              r"\texttt{analyze\_gate\_inference.py}, so the percentile intervals are the ones "
              r"it reports); the Student-$t$ interval uses $t_{28}$ and is truncated to "
              r"$[-1,1]$. The adjusted intervals are at $1-0.05/4$, the Bonferroni level for "
              r"each family of four contrasts within which a claim is decided. $p$ is the "
              r"unadjusted two-sided signed-rank test of the 29 per-function deltas (its "
              r"Holm-adjusted value is in Table~" + ref("tab:gateinference", home) + r"). Verdicts as "
              r"in Table~" + ref("tab:gateinference", home) + r"; the penultimate column gives the "
              r"BCa verdicts, followed by the Student-$t$ ones where they differ, and the last "
              r"column the verdicts of the Bonferroni-adjusted Student-$t$ interval. Where the "
              r"last differ from the unadjusted verdicts, the verdict does not survive the "
              r"family adjustment, and Table~" + ref("tab:gateinference", home) + r" marks it "
              r"with $^{\dagger}$." + "".join(
                  r" The interval of " + GATE_NAME[r["quantity"]] + GATE_AT[r["block"]]
                  + r" excludes zero with a bound within 0.02 of it, and Table~"
                  + ref("tab:gateinference", home) + r" reports it as a tie (main text, "
                  r"Section~2.6)."
                  for r in d.to_dict("records") if bool(r["tie_002"]) and r["side_pct"] != "spans"),
              r"\end{flushleft}", r"\end{table}", ""]
    write(fname, "\n".join(lines), "tab:supp_gate_intervals", home)


# ================================================================== Table 4
def table_popsize(df, caption, fname, home="main"):
    """SEHHO-COBL across population sizes at a fixed budget, with delta(N vs N=30)."""
    _LABELS[fname] = "tab:popsize"
    ps = _ana("revision_popsize.csv").set_index("N")
    tags = sorted(df.tag.unique(), key=lambda t: int(t.split("N")[-1]))
    funcs = sorted(df.func.unique())
    M = np.array([[df[(df.func == f) & (df.tag == t)]["error"].mean() for t in tags] for f in funcs])
    fr = friedman(M)
    best = np.argmin(fr["avg_ranks"])
    lines = [r"\begin{table}[htbp]", r"\centering", f"\\caption{{{caption}}}",
             r"\label{tab:popsize}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lcccll}",
             r"\toprule",
             r"Population $N$ & Generations & Friedman rank & \#1 functions & "
             r"$\delta$($N$ vs $N{=}30$) [95\% CI] & $p_{\text{Holm}}$ \\", r"\midrule"]
    partial = []
    for i, t in enumerate(tags):
        N = int(t.split("N")[-1])
        gens = -(-(300000 - 4 * N) // N)
        if (300000 - 4 * N) % N:
            partial.append(N)
        r = fr["avg_ranks"][i]
        n1 = int(sum(i in tied_best(M[k]) for k in range(len(funcs))))
        cell = f"\\textbf{{{r:.2f}}}" if i == best else f"{r:.2f}"
        if N == 30:
            dcell, pcell, name = "--", "--", r"30 (as in SEHHO-COBL)"
        else:
            q = ps.loc[N]
            dv = led(fname, f"N={N}", "delta", q["mean"], f"N={N}", "N=30",
                     "results/analysis/revision_popsize.csv (E6)")
            dcell = f"${dv}$ ${cfmt(q.pct_lo, q.pct_hi)}$"
            pcell = ptex(q.p_holm) + ("$^{*}$" if q.p_holm < 0.05 else "")
            name = str(N)
        lines.append(f"{name} & {fnum(gens)} & {cell} & {n1} & {dcell} & {pcell} \\\\")
    gen_note = (r" Generations are $T=\lceil(3\times10^{5}-4N)/N\rceil$, the count the "
                r"algorithm computes after its $4N$-evaluation initialisation"
                + ("; for " + " and ".join(f"$N={n}$" for n in partial)
                   + " the last generation is cut short by the budget." if partial else "."))
    lines += [r"\bottomrule", r"\end{tabular}}", r"\begin{flushleft}\footnotesize",
              r"CEC2017 at $D=30$, 29 functions, 30 runs, $3\times10^{5}$ evaluations for "
              r"every row; only the population size differs. ``\#1 functions'' counts the "
              r"functions on which the row has the lowest mean error; exact ties are credited "
              r"to every tied row. Iman--Davenport "
              f"$F={fr['iman_davenport_F']:.2f}$, $p={texp(fr['p'])}$." + gen_note + " "
              + CONV_FN + r" Here A is the population in the row and B the $N=30$ of "
              r"SEHHO-COBL, so a negative value favours the larger population. Intervals are "
              r"95\% percentile bootstrap intervals over the functions (20{,}000 resamples); "
              r"$p_{\text{Holm}}$ is a two-sided signed-rank test of the per-function deltas, "
              r"Holm-corrected over the four contrasts; $^{*}$ marks $p_{\text{Holm}}<0.05$. "
              r"$N=120$ is the best of the five sizes on this suite, so its contrast is the "
              r"largest of the four by selection.",
              r"\end{flushleft}", r"\end{table}", ""]
    write(fname, "\n".join(lines), "tab:popsize", home)


# ================================================================== Table 5 and S5.7
POP_ORDER = ["L-SHADE 6D vs 18D", "ours 6D vs ours 18D", "ours vs L-SHADE at 6D",
             "ours vs L-SHADE default"]
POP_ARMS = {"L-SHADE 6D vs 18D": ("L-SHADE at 6D", "L-SHADE at 18D"),
            "ours 6D vs ours 18D": (METHOD + " at 6D", METHOD + " at 18D"),
            "ours vs L-SHADE at 6D": (METHOD, "L-SHADE at 6D"),
            "ours vs L-SHADE default": (METHOD, "L-SHADE at 18D")}


def _pop_blocks(d):
    blocks = []
    for suite in ("cec2017", "cec2014"):
        for budget in ("tight", "competition"):
            for dim in (30, 50):
                if not d[(d.suite == suite) & (d.budget == budget) & (d.dim == dim)].empty:
                    blocks.append((suite, budget, dim))
    tight = [b for b in blocks if b[1] == "tight"]
    comp = [b for b in blocks if b[1] == "competition"]
    return tight + comp, tight, comp


def _rerun_gap(d, blocks):
    """Largest |difference| between row 4 and the same contrast from the main tables'
    own L-SHADE runs (the 18D arm of E14 is an independent re-run)."""
    paths = [os.path.join(RES, f) for f in ("E3_cec2017.csv", "E9_newmethod_vs_baselines.csv",
                                            "E12_cec2017_tight.csv", "E13_cec2014.csv")]
    if not all(os.path.exists(p) for p in paths):
        return None
    e3, e9, e12, e13 = (read_runs(p) for p in paths)
    main_runs = {("cec2017", "competition"): pd.concat([e3, e9[e9.tag == "E3_cec2017"]], ignore_index=True),
                 ("cec2017", "tight"): e12,
                 ("cec2014", "competition"): e13[e13.max_fes > 15_000],
                 ("cec2014", "tight"): e13[e13.max_fes == 15_000]}
    gaps = []
    for su, bu, dm in blocks:
        r = d[(d.suite == su) & (d.budget == bu) & (d.dim == dm) & (d.contrast == "ours vs L-SHADE default")]
        m = main_runs[(su, bu)]
        m = m[m.dim == dm]
        if r.empty or m.empty:
            continue
        dl = np.mean([cliffs_delta(m[(m.func == f) & (m.algo == INTERNAL_NAME)]["error"].to_numpy(),
                                   m[(m.func == f) & (m.algo == "LSHADE")]["error"].to_numpy())
                      for f in sorted(m.func.unique())])
        gaps.append(abs(dl - float(r.iloc[0]["mean"])))
    return max(gaps) if gaps else None


def _pop_head(tight, comp):
    """Column heads shared by both panels of Table 5: suite over dimension."""
    cols = [(su, dm) for su, _, dm in tight]
    assert cols == [(su, dm) for su, _, dm in comp], "both panels need the same blocks"
    suites = []
    for su, _ in cols:
        if not suites or suites[-1][0] != su:
            suites.append([su, 0])
        suites[-1][1] += 1
    name = {"cec2017": "CEC2017 (diagnostic)", "cec2014": r"CEC2014$^{\dagger}$ (exploratory)"}
    head = " & " + " & ".join(f"\\multicolumn{{{n}}}{{c}}{{{name[su]}}}" for su, n in suites) + r" \\"
    rules, c0 = [], 2
    for su, n in suites:
        rules.append(f"\\cmidrule(lr){{{c0}-{c0 + n - 1}}}")
        c0 += n
    return [head, " ".join(rules),
            "Contrast & " + " & ".join(f"$D{{=}}{dm}$" for _, dm in cols) + r" \\", r"\midrule"]


def _pop_panel_head(budget, blocks, ncol):
    if budget == "tight":
        bd = ", ".join(f"{fnum(15000 / dm)} at $D={dm}$" for dm in sorted({dm for _, _, dm in blocks}))
        txt = r"Panel A. 15{,}000 evaluations ($B/D=$ " + bd + ")"
    else:
        txt = r"Panel B. Competition budget, $10^{4}D$ evaluations ($B/D=10{,}000$)"
    return f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\emph{{{txt}}}}} \\\\"


def table_pop_isolation(caption, fname, home="main"):
    """Table 5: the one-constant population control, in two panels (15,000 evaluations;
    competition budget), with intervals under rows 1 and 3.  No \\resizebox (NEW-1): the
    body is set in \\footnotesize (10 pt) and the intervals in \\scriptsize (8 pt), so no
    text prints below 8 pt."""
    _LABELS[fname] = "tab:popisolation"
    d = _ana("revision_pop_isolation.csv")
    blocks, tight, comp = _pop_blocks(d)
    pretty = {"L-SHADE 6D vs 18D": r"\textbf{1. L-SHADE at $6D$ vs L-SHADE at $18D$}",
              "ours 6D vs ours 18D": r"2. " + METHOD + r" at $6D$ vs " + METHOD + r" at $18D$",
              "ours vs L-SHADE at 6D": r"\textbf{3. " + METHOD + r" vs L-SHADE, both at $6D$}",
              "ours vs L-SHADE default": r"4. " + METHOD + r" vs L-SHADE at its published $18D$"}
    ncol = 1 + len(tight)
    lines = [r"\begin{table}[htbp]", r"\centering", f"\\caption{{{caption}}}",
             r"\label{tab:popisolation}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{5.4cm}" + "c" * len(tight) + "@{}}",
             r"\toprule"] + _pop_head(tight, comp)
    gap = _rerun_gap(d, blocks)
    marks = set()
    for pi, panel in enumerate((tight, comp)):
        if pi:
            lines.append(r"\midrule")
        lines.append(_pop_panel_head(panel[0][1], panel, ncol))
        for c in POP_ORDER:
            cells = []
            for su, bu, dm in panel:
                r = d[(d.suite == su) & (d.budget == bu) & (d.dim == dm) & (d.contrast == c)].iloc[0]
                a, b = POP_ARMS[c]
                dv = led(fname, c, f"{su} D={dm} {bu}", r["mean"], a, b,
                         "results/analysis/revision_pop_isolation.csv (E14)")
                mk = robust_mark(r)
                excl = r.pct_lo > 0 or r.pct_hi < 0
                if excl and excludes_zero_tie(r):
                    # a bound within 0.02 of zero: a tie by the rule of Section 2.6, not bold
                    mk = r"$^{\circ}$" + mk
                if mk:
                    marks.add(mk)
                txt = f"${dv}${mk}"
                if excl and not excludes_zero_tie(r):
                    txt = r"\textbf{\boldmath " + txt + "}"
                if c in ("L-SHADE 6D vs 18D", "ours vs L-SHADE at 6D"):
                    # the interval under its estimate, on the second line of the row label
                    txt = (r"\begin{tabular}[t]{@{}c@{}}" + txt + r"\\{\scriptsize $"
                           + cfmt(r.pct_lo, r.pct_hi) + r"$}\end{tabular}")
                cells.append(txt)
            lines.append(f"{pretty[c]} & " + " & ".join(cells) + r" \\")
    rerun = f" (at most ${gap:.3f}$ in $\\delta$)" if gap is not None else ""
    mark_txt = ""
    if any(m.startswith(r"$^{\circ}$") for m in marks):
        mark_txt += (r" $^{\circ}$: the 95\% interval excludes zero, but a bound of the "
                     r"percentile, BCa or Student-$t$ interval lies within 0.02 of zero, a tie by "
                     r"the rule of Section~" + ref("subsec:stats", home) + r" (not bold).")
    if any(r"$^{\ddagger}$" in m for m in marks):
        mark_txt += (r" $^{\ddagger}$: the 95\% interval excludes zero but the Bonferroni-"
                     r"adjusted interval over the four blocks of the row and budget (98.75\%) "
                     r"does not.")
    if any(r"$^{\S}$" in m for m in marks):
        mark_txt += r" $^{\S}$: the BCa or Student-$t$ interval includes zero (and so does its adjusted version)."
    lines += [r"\bottomrule", r"\end{tabular}", r"\begin{flushleft}\footnotesize",
              r"Mean per-function Cliff's $\delta$ with 30 runs per cell; " + CONV + r" Here "
              r"A is the first-named arm of the row. \textbf{Bold} values mark a contrast "
              r"whose 95\% percentile bootstrap interval over the functions (shown under "
              r"rows 1 and 3; all rows with BCa and Student-$t$ intervals in Table~"
              + ref("tab:supp_popisolation", home) + r") excludes zero and is not a tie." + mark_txt +
              r" CEC2017 is the diagnostic suite. The CEC2014$^{\dagger}$ columns are "
              r"\emph{exploratory}: these comparisons are not among those the pre-registration "
              r"lists, they were run after the confirmatory analysis, and the pre-registration "
              r"requires any such comparison to be labelled exploratory. Rows 1 and 2 each "
              r"change one constant, "
              r"$N_{\text{init}}$, inside one algorithm; the L-SHADE arms keep every other "
              r"setting as in our port of the published algorithm ($H=6$, "
              r"$|\mathcal{A}|=2.6N$, $p=0.11$, bound handling, memory update). Rows 3 and 4 "
              r"compare two different algorithms: besides $N_{\text{init}}$ (row 4 only), "
              + METHOD + r" and L-SHADE differ in $H$, archive rate and $p$, and in "
              r"implementation details (bound handling, steady-state versus generational "
              r"replacement, the CR memory mean, L-SHADE's terminal CR value, $F$ sampling, "
              r"rounding of the pbest pool). The L-SHADE $18D$ arm of rows 1 and 4 is an "
              r"independent re-run of stock L-SHADE with its own seed stream, not "
              r"the L-SHADE runs of Tables~" + ref("tab:cec2014", home) + r", "
              + ref("tab:cec2017", home) + r" and~" + ref("tab:cec2017_tight", home) + r", so "
              r"row 4 differs from the matching contrasts there by sampling error" + rerun +
              r". Row 1 is what licenses a statement about population sizing; row 3 is what "
              r"licenses a statement about " + METHOD + r". They point in different "
              r"directions, and both are reported.",
              r"\end{flushleft}", r"\end{table}", ""]
    write(fname, "\n".join(lines), "tab:popisolation", home)


def table_supp_popisolation(caption, fname, home="supp"):
    """S5.7: every row of Table 5 with percentile, BCa, Student-t and adjusted intervals."""
    _LABELS[fname] = "tab:supp_popisolation"
    d = _ana("revision_pop_isolation.csv")
    lines = [r"\begin{table}[htbp]", r"\centering", f"\\caption{{{caption}}}",
             r"\label{tab:supp_popisolation}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{llrrccccc}", r"\toprule",
             r"Contrast & Block & $B/D$ & $\bar\delta$ & percentile & BCa & Student-$t$ & "
             r"Student-$t$, 98.75\% & $p_{\text{Holm}}$ \\", r"\midrule"]
    for c in POP_ORDER:
        lines.append(f"\\multicolumn{{9}}{{l}}{{\\emph{{{POP_ARMS[c][0]} vs {POP_ARMS[c][1]}}}}} \\\\")
        for su in ("cec2017", "cec2014"):
            for bu in ("tight", "competition"):
                for dm in (30, 50):
                    r = d[(d.suite == su) & (d.budget == bu) & (d.dim == dm) & (d.contrast == c)]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    a, b = POP_ARMS[c]
                    dv = led(fname, c, f"{su} D={dm} {bu}", r["mean"], a, b,
                             "results/analysis/revision_pop_isolation.csv (E14)")
                    blk = (f"CEC{su[3:]}{'$^{\\dagger}$' if su == 'cec2014' else ''}, $D{{=}}{dm}$, "
                           + ("15{,}000" if bu == "tight" else fnum(10000 * dm)))
                    lines.append(f"\\quad & {blk} & {fnum(r.evals_per_variable)} & ${dv}$ & "
                                 f"${cfmt(r.pct_lo, r.pct_hi)}$ & ${cfmt(r.bca_lo, r.bca_hi)}$ & "
                                 f"${cfmt(r.t_lo, r.t_hi)}$ & ${cfmt(r.t_adj_lo, r.t_adj_hi)}$ & "
                                 f"{ptex(r.p_holm)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\begin{flushleft}\footnotesize",
              r"All rows of Table~" + ref("tab:popisolation", home) + r". " + CONV_FN +
              r" Here A is the first-named arm. Percentile and BCa intervals come from the same "
              r"20{,}000 bootstrap resamples of the functions (seed 20260915, in the order of "
              r"\texttt{analyze\_pop\_isolation.py}); Student-$t$ intervals are truncated to "
              r"$[-1,1]$; the adjusted interval is at the Bonferroni level for the four blocks "
              r"of one row and budget. $p_{\text{Holm}}$: signed-rank test of the per-function "
              r"deltas, Holm-corrected over the four contrasts of a block. $^{\dagger}$ "
              r"CEC2014 blocks are exploratory (stage E14, run after the confirmatory "
              r"analysis).",
              r"\end{flushleft}", r"\end{table}", ""]
    write(fname, "\n".join(lines), "tab:supp_popisolation", home)


# ================================================================== Table 6 and S5.1
def table_cec2014_primary(caption, fname, home="main"):
    """Table 6: Panel A the registered endpoint, its post hoc sensitivity analysis, the
    CEC2017 counterpart and the gap; Panel B pairwise delta and signed-rank tests."""
    _LABELS[fname] = "tab:cec2014primary"
    # the registration's file, time and hash scope are reported in Supplementary
    # Section S2.5, not in this main-text note (REV-10)
    prim = _ana("revision_cec2014_primary.csv")
    gap = _ana("revision_cec2014_suitegap.csv").set_index("dim")
    pw = _ana("revision_cec2014_pairwise.csv")
    reg = prim[prim.analysis == "registered primary endpoint"].set_index("dim")
    ex = prim[prim.analysis != "registered primary endpoint"].set_index("dim")
    src = "results/analysis/revision_cec2014_primary.csv"
    gsrc = "results/analysis/revision_cec2014_suitegap.csv"

    def cell(v, lo, hi, row, col, a, b, source, bold=False):
        dv = led(fname, row, col, v, a, b, source)
        top = f"${dv}$"
        if bold:
            top = r"\textbf{\boldmath " + top + "}"
        return stack(top, f"${cfmt(lo, hi)}$")

    r30, r50 = reg.loc[30], reg.loc[50]
    e30, e50 = ex.loc[30], ex.loc[50]
    g30, g50 = gap.loc[30], gap.loc[50]
    lines = [r"\begin{table}[htbp]", r"\centering", f"\\caption{{{caption}}}",
             r"\label{tab:cec2014primary}",
             r"\begin{tabular}{lcccc}", r"\toprule",
             r"\multicolumn{5}{l}{\emph{Panel A. Primary endpoint: " + METHOD + r" vs "
             r"SEHHO-COBL, competition budget}} \\", r"\midrule",
             r" & \multicolumn{2}{c}{$D=30$, 300{,}000 FEs} & \multicolumn{2}{c}{$D=50$, 500{,}000 FEs} \\",
             r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
             r"Analysis & $\bar\delta$ [95\% CI] & W/L & $\bar\delta$ [95\% CI] & W/L \\", r"\midrule",
             r"registered (30 functions) & "
             + cell(r30["mean"], r30.pct_lo, r30.pct_hi, "registered", "D=30", METHOD, "SEHHO-COBL",
                    src, bold=True) + f" & {int(r30.wins)}/{int(r30.losses)} & "
             + cell(r50["mean"], r50.pct_lo, r50.pct_hi, "registered", "D=50", METHOD, "SEHHO-COBL",
                    src, bold=True) + f" & {int(r50.wins)}/{int(r50.losses)} \\\\",
             r"post hoc: without F18, F22--F24 (26) & "
             + cell(e30["mean"], e30.pct_lo, e30.pct_hi, "post hoc without F18,F22,F23,F24", "D=30",
                    METHOD, "SEHHO-COBL", src) + f" & {int(e30.wins)}/{int(e30.losses)} & "
             + cell(e50["mean"], e50.pct_lo, e50.pct_hi, "post hoc without F18,F22,F23,F24", "D=50",
                    METHOD, "SEHHO-COBL", src) + f" & {int(e50.wins)}/{int(e50.losses)} \\\\",
             r"CEC2017, the same contrast (29) & "
             + cell(g30.cec2017_mean, g30.cec2017_pct_lo, g30.cec2017_pct_hi, "CEC2017 counterpart",
                    "D=30", METHOD, "SEHHO-COBL", gsrc) + " & -- & "
             + cell(g50.cec2017_mean, g50.cec2017_pct_lo, g50.cec2017_pct_hi, "CEC2017 counterpart",
                    "D=50", METHOD, "SEHHO-COBL", gsrc) + r" & -- \\",
             r"gap, CEC2017 minus CEC2014 & "
             + cell(g30.gap_cec2017_minus_cec2014, g30.gap_pct_lo, g30.gap_pct_hi, "suite gap", "D=30",
                    "CEC2017 estimate", "CEC2014 estimate", gsrc) + " & -- & "
             + cell(g50.gap_cec2017_minus_cec2014, g50.gap_pct_lo, g50.gap_pct_hi, "suite gap", "D=50",
                    "CEC2017 estimate", "CEC2014 estimate", gsrc) + r" & -- \\",
             r"\bottomrule", r"\end{tabular}", "", r"\vspace{0.8em}"]
    blocks = [(30, 300_000, "competition"), (50, 500_000, "competition"), (30, 15_000, "tight"),
              (50, 15_000, "tight")]
    lines += [r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lcccccccc}", r"\toprule",
              r"\multicolumn{9}{l}{\emph{Panel B. " + METHOD + r" against the SHADE-family "
              r"baselines and SEHHO-COBL, both budgets}} \\", r"\midrule",
              " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{$D={dm}$, {fnum(fes)} FEs}}"
                                 for dm, fes, _ in blocks) + r" \\",
              r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}",
              "Competitor & " + " & ".join(r"$\bar\delta$ [95\% CI] & $p_{\text{Holm}}$"
                                           for _ in blocks) + r" \\", r"\midrule"]
    for opp in ("jSO", "LSHADE", "SHADE", "SEHHO-COBL"):
        cells = []
        for dm, fes, bu in blocks:
            q = pw[(pw.dim == dm) & (pw.max_fes == fes) & (pw.opponent == opp)].iloc[0]
            lo, hi = q.pct_lo, q.pct_hi
            source = "results/analysis/revision_cec2014_pairwise.csv"
            if opp == "SEHHO-COBL" and bu == "competition":
                # one interval per contrast: the registered one of Panel A
                lo, hi = reg.loc[dm].pct_lo, reg.loc[dm].pct_hi
                source = src + " (registered interval)"
            dv = led(fname, DISPLAY.get(opp, opp), f"D={dm} {fnum(fes)}", q["mean"], METHOD,
                     DISPLAY.get(opp, opp), source)
            cells += [stack(f"${dv}$", f"${cfmt(lo, hi)}$"),
                      ptex(q.p_holm) + ("$^{*}$" if q.p_holm < 0.05 else "")]
        lines.append(f"{esc(DISPLAY.get(opp, opp))} & " + " & ".join(cells) + r" \\")
    rob = (f"BCa intervals (${cfmt(r30.bca_lo, r30.bca_hi)}$ and ${cfmt(r50.bca_lo, r50.bca_hi)}$), "
           f"Student-$t$ intervals (${cfmt(r30.t_lo, r30.t_hi)}$, ${cfmt(r50.t_lo, r50.t_hi)}$) and "
           f"two-block Bonferroni (97.5\\%) intervals (${cfmt(r30.pct_adj_lo, r30.pct_adj_hi)}$, "
           f"${cfmt(r50.pct_adj_lo, r50.pct_adj_hi)}$)")
    rob_ex = (f" (BCa ${cfmt(e30.bca_lo, e30.bca_hi)}$ and ${cfmt(e50.bca_lo, e50.bca_hi)}$)")
    ex_side = all(ex.loc[d_].side_all == "below" and ex.loc[d_].side_all_adj == "below" for d_ in (30, 50))
    gap_txt = ("includes zero in both blocks, so the gap cannot be distinguished from "
               "differences between the suites" if all(gap.loc[d_].gap_pct_lo <= 0 <= gap.loc[d_].gap_pct_hi
                                                         for d_ in (30, 50)) else
               "excludes zero in at least one block")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\begin{flushleft}\footnotesize",
              r"CEC2014, 30 functions, 30 runs per cell. No component was chosen, no constant "
              r"fitted and no diagnosis performed on these functions before these runs. The "
              r"configuration, the primary endpoint and its test, three secondary predictions "
              r"stated in words and the success criterion were fixed in an internal "
              r"pre-registration written before the first CEC2014 "
              r"run (not an external registry entry; Supplementary Section~"
              + ref("sec:S2_prereg", home) + r"). " + CONV_FN +
              r" Here A is " + METHOD + r". Each cell gives $\bar\delta$ and, below it, a 95\% "
              r"percentile bootstrap interval over the functions (20{,}000 resamples); W/L counts "
              r"functions with negative / positive $\delta$. \textbf{Panel A}: the registered "
              r"success criterion was that the interval exclude zero in favour of " + METHOD +
              r" in both blocks (bold); it does, and it does so also for " + rob + r". The second "
              r"row is a post hoc sensitivity analysis: it removes the four "
              r"functions whose recipes CEC2022 F6, F8, F9 and F10 follow (CEC2022 is the suite "
              r"the configuration was selected on), and is reported beside the registered "
              r"result whatever it shows; its intervals " + ("exclude zero under every interval "
              r"type and the Bonferroni adjustment as well" + rob_ex if ex_side else
              r"do not all exclude zero") + r". The third row is the same contrast on CEC2017, "
              r"the suite on which the changes were diagnosed (its intervals come from the "
              r"bootstrap of the gap and differ in the third decimal from those of Table~"
              + ref("tab:cec2017", home) + r", an independent bootstrap stream); the gap "
              r"(CEC2017 minus CEC2014, the two suites resampled independently) " + gap_txt +
              r". \textbf{Panel B}: "
              r"$p_{\text{Holm}}$ is a two-sided signed-rank test of the per-function deltas, "
              r"Holm-corrected over the six competitors of a block; DE and HHO, and the "
              r"Friedman post-hoc tests of the analysis script, which the pre-registration "
              r"does not name, are in Table~" + ref("tab:cec2014", home) + r". $^{*}$ marks "
              r"$p_{\text{Holm}}<0.05$. For SEHHO-COBL at the competition budget the interval is "
              r"the registered one of Panel A.",
              r"\end{flushleft}", r"\end{table}", ""]
    write(fname, "\n".join(lines), "tab:cec2014primary", home)


def table_cec2014(caption, fname, home="supp"):
    """S5.1: the full CEC2014 ranking, with both the registered and the REV-29 tests."""
    _LABELS[fname] = "tab:cec2014"
    d = read_runs(os.path.join(RES, "E13_cec2014.csv"))
    d["algo"] = d.algo.replace(DISPLAY)
    pw = _ana("revision_cec2014_pairwise.csv")
    prim = _ana("revision_cec2014_primary.csv")
    reg = prim[prim.analysis == "registered primary endpoint"].set_index("dim")
    blocks = [(30, 300_000), (50, 500_000), (30, 15_000), (50, 15_000)]
    algos = [a for a in ORDER if a in set(d.algo)]
    inv = {v: k for k, v in DISPLAY.items()}
    lines = [r"\begin{table}[htbp]", r"\centering", f"\\caption{{{caption}}}",
             r"\label{tab:cec2014}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{l" + "cccc" * len(blocks) + "}", r"\toprule"]
    lines.append(" & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{$D={dim}$, {fnum(fes)} FEs}}"
                                    for dim, fes in blocks) + r" \\")
    lines.append(" ".join(f"\\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(blocks))))
    lines.append("Algorithm & " + " & ".join(
        r"Rank & $\bar\delta$ [95\% CI] & $p^{\text{SR}}_{\text{Holm}}$ & $p^{\text{F}}_{\text{Holm}}$"
        for _ in blocks) + r" \\")
    lines.append(r"\midrule")
    res = {b: compare_block(d[(d.dim == b[0]) & (d.max_fes == b[1])], control=CONTROL,
                            label=f"cec2014 {b}") for b in blocks}
    for a in algos:
        cells = []
        for dim, fes in blocks:
            r = res[(dim, fes)]["ranks"]
            r = r[r.algorithm == a].iloc[0]
            if a == CONTROL:
                cells += [f"\\textbf{{{r.avg_rank:.2f}}}", "--", "--", "--"]
                continue
            q = pw[(pw.dim == dim) & (pw.max_fes == fes) & (pw.opponent == inv.get(a, a))].iloc[0]
            lo, hi = q.pct_lo, q.pct_hi
            source = "results/analysis/revision_cec2014_pairwise.csv"
            if a == "SEHHO-COBL" and fes > 15_000:
                lo, hi = reg.loc[dim].pct_lo, reg.loc[dim].pct_hi
                source = "results/analysis/revision_cec2014_primary.csv (registered interval)"
            dv = led(fname, a, f"D={dim} {fnum(fes)}", q["mean"], METHOD, a, source)
            cells += [f"{r.avg_rank:.2f}", f"${dv}$ ${cfmt(lo, hi)}$",
                      ptex(q.p_holm) + ("$^{*}$" if q.p_holm < 0.05 else ""),
                      ptex(q.p_holm_friedman) + ("$^{*}$" if q.p_holm_friedman < 0.05 else "")]
        lines.append(f"{esc(a)} & " + " & ".join(cells) + r" \\")
    pre = json.load(open(os.path.join(ANA, "PREREGISTRATION_cec2014.json")))
    written = (datetime.fromisoformat(pre["written_at"]).astimezone(timezone.utc)
               .strftime("%Y-%m-%d %H:%M UTC"))
    lines += [r"\bottomrule", r"\end{tabular}}", r"\begin{flushleft}\footnotesize",
              r"CEC2014, 30 functions, 30 runs per cell: the confirmatory suite, on whose "
              r"results no design decision was taken (the selection suite reuses data of "
              r"four of its functions; Table~" + ref("tab:supp_lineage", home) + r"), "
              r"evaluated under an internal pre-registration timestamped " + written +
              r" (see Table~" + ref("tab:cec2014primary", home) + r"). ``Rank'' is the "
              r"Friedman average rank (lower is better); bold marks the reference row, " +
              METHOD + r", not the best algorithm. " + CONV_FN + r" Here A is " + METHOD +
              r" and B the competitor in the row; the interval is a 95\% percentile bootstrap "
              r"interval over the functions, from the analysis script committed in the "
              r"pre-registration (\texttt{analyze\_cec2014.py}); for SEHHO-COBL at the "
              r"competition budget it is "
              r"the registered primary-endpoint interval. $p^{\text{SR}}_{\text{Holm}}$: "
              r"two-sided signed-rank test of the per-function deltas, Holm-corrected over the "
              r"six competitors of a block. "
              r"$p^{\text{F}}_{\text{Holm}}$: the Friedman post-hoc test against " + METHOD +
              r" implemented in \texttt{analyze\_cec2014.py}, which was written after the "
              r"pre-registration (Section~" + ref("sec:S2_prereg", home) + r"); the "
              r"pre-registration names no test for these secondary comparisons. It is "
              r"Holm-corrected in the same family; it depends on which other algorithms are in "
              r"the set and on CEC2014 absorbs HHO's zero-shift artefact (HHO reaches the error of "
              r"the origin on F23--F30, which the all-zero shift of the third component makes "
              r"reachable). "
              r"$^{*}$ marks $p<0.05$.",
              r"\end{flushleft}", r"\end{table}", ""]
    write(fname, "\n".join(lines), "tab:cec2014", home)


# ================================================================== S5.1: both configurations vs HHO
HHO_BLOCKS = [("CEC2017", "diagnostic", "cec2017_competition", 30, 300_000),
              ("CEC2017", "diagnostic", "cec2017_competition", 50, 500_000),
              ("CEC2017", "diagnostic", "cec2017_competition", 100, 1_000_000),
              ("CEC2017", "diagnostic", "cec2017_tight", 30, 15_000),
              ("CEC2017", "diagnostic", "cec2017_tight", 50, 15_000),
              ("CEC2022", "selection", "cec2022_matched", 10, 15_000),
              ("CEC2022", "selection", "cec2022_matched", 20, 15_000),
              ("CEC2022", "selection", "cec2022_competition", 10, 200_000),
              ("CEC2022", "selection", "cec2022_competition", 20, 1_000_000),
              ("CEC2014", "confirmatory", "cec2014", 30, 300_000),
              ("CEC2014", "confirmatory", "cec2014", 50, 500_000),
              ("CEC2014", "confirmatory", "cec2014", 30, 15_000),
              ("CEC2014", "confirmatory", "cec2014", 50, 15_000)]


def table_supp_hho(caption, fname, home="supp"):
    """S5.1 (REV-57): SEHHO-COBL and SEHHO-COBL-R against HHO in every block of the
    three suites, so that the main text's statement about both configurations rests
    on one generated table."""
    _LABELS[fname] = "tab:supp_hho"
    hho = _ana("revision_hho.csv")
    sp = _ana("revision_suite_pairwise.csv")
    cp = _ana("revision_cec2014_pairwise.csv")
    lines = [r"\begin{table}[htbp]", r"\centering", f"\\caption{{{caption}}}",
             r"\label{tab:supp_hho}", r"\adjustbox{max width=\textwidth}{%", r"\begin{tabular}{llcccc}",
             r"\toprule",
             r" & & \multicolumn{2}{c}{SEHHO-COBL vs HHO} & \multicolumn{2}{c}{" + METHOD +
             r" vs HHO} \\", r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}",
             r"Block & Budget & $\bar\delta$ [95\% CI] & W/L & $\bar\delta$ [95\% CI] & W/L \\",
             r"\midrule"]
    vals = {}
    prev = None
    for suite, role, tab, dim, fes in HHO_BLOCKS:
        if suite != prev:
            if prev is not None:
                lines.append(r"\midrule")
            lines.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{suite}, the {role} suite}}}} \\\\")
            prev = suite
        a = hho[(hho.table == tab) & (hho.dim == dim) & (hho.max_fes == fes)].iloc[0]
        if suite == "CEC2014":
            b = cp[(cp.dim == dim) & (cp.max_fes == fes) & (cp.opponent == "HHO")].iloc[0]
            src_b = "results/analysis/revision_cec2014_pairwise.csv (registered interval)"
        else:
            b = sp[(sp.table == tab) & (sp.dim == dim) & (sp.max_fes == fes)
                   & (sp.opponent == "HHO")].iloc[0]
            src_b = "results/analysis/revision_suite_pairwise.csv"
        blk = f"{suite} D={dim} {fes}"
        da = led(fname, "SEHHO-COBL vs HHO", blk, a["mean"], "SEHHO-COBL", "HHO",
                 "results/analysis/revision_hho.csv")
        db = led(fname, f"{METHOD} vs HHO", blk, b["mean"], METHOD, "HHO", src_b)
        key = "CEC2014" if suite == "CEC2014" else "other"
        vals.setdefault((key, "SEHHO-COBL"), []).append(a["mean"])
        vals.setdefault((key, METHOD), []).append(b["mean"])
        lines.append(f"\\quad $D={dim}$ & {fnum(fes)} & ${da}$ ${cfmt(a.pct_lo, a.pct_hi)}$ & "
                     f"{int(a.wins)}/{int(a.losses)} & ${db}$ ${cfmt(b.pct_lo, b.pct_hi)}$ & "
                     f"{int(b.wins)}/{int(b.losses)} \\\\")

    def rng_(key, who):
        v = vals[(key, who)]
        return f"${dfmt(max(v))}$ to ${dfmt(min(v))}$"
    tabs = [ref(t, home) for t in ("tab:cec2014", "tab:cec2017", "tab:cec2017_tight",
                                    "tab:cec2022_matched", "tab:cec2022_competition")]
    others = ", ".join(tabs[:-1]) + " and~" + tabs[-1]
    lines += [r"\bottomrule", r"\end{tabular}}", r"\begin{flushleft}\footnotesize",
              CONV_FN + r" Here A is the configuration named at the head of the column and B "
              r"is HHO. Intervals are 95\% percentile bootstrap intervals over the functions "
              r"(20{,}000 resamples; for " + METHOD + r" on CEC2014, the intervals of the "
              r"registered analysis); W/L counts the functions on which $\delta$ is negative / "
              r"positive. The " + METHOD + r" columns repeat the HHO rows of Tables~"
              + others + r"; the SEHHO-COBL columns are "
              r"computed in the same way. On CEC2017 and CEC2022 $\bar\delta$ runs from "
              + rng_("other", "SEHHO-COBL") + r" for SEHHO-COBL and from "
              + rng_("other", METHOD) + r" for " + METHOD + r"; on CEC2014, where HHO is "
              r"helped by the all-zero shift of the third component of every composition "
              r"function, from " + rng_("CEC2014", "SEHHO-COBL") + r" and from "
              + rng_("CEC2014", METHOD) + r". Both configurations differ from HHO in several "
              r"components, so these contrasts are consistent with, but do not isolate, the "
              r"value of replacing HHO's siege operators.",
              r"\end{flushleft}", r"\end{table}", ""]
    write(fname, "\n".join(lines), "tab:supp_hho", home)


# ================================================================== S2.6: precision
def table_supp_precision(caption, fname, home="supp"):
    _LABELS[fname] = "tab:supp_precision"
    p = _ana("revision_precision.csv")
    lines = [r"\begin{table}[htbp]", r"\centering", f"\\caption{{{caption}}}",
             r"\label{tab:supp_precision}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{lccccc}", r"\toprule",
             r"Block & per-function SE of $\delta$, median [IQR] & SD of $\delta$ over functions "
             r"& SE of $\bar\delta$: functions / runs only & one-stage 95\% CI & two-stage 95\% CI \\",
             r"\midrule"]
    for r in p[p.dim > 0].to_dict("records"):
        led(fname, "primary endpoint", f"D={int(r['dim'])}", r["mean"], METHOD, "SEHHO-COBL",
            "results/analysis/revision_precision.csv")
        lines.append(f"$D={int(r['dim'])}$, {fnum(r['max_fes'])} FEs & {r['se_run_median']:.3f} "
                     f"[{r['se_run_q1']:.3f}, {r['se_run_q3']:.3f}] & {r['sd_between_functions']:.3f} & "
                     f"{r['se_of_mean_functions']:.3f} / {r['se_of_mean_runs_only']:.3f} & "
                     f"${cfmt(r['one_stage_lo'], r['one_stage_hi'])}$ & "
                     f"${cfmt(r['two_stage_lo'], r['two_stage_hi'])}$ \\\\")
    allr = p[p.dim == 0].iloc[0]
    lines += [r"\bottomrule", r"\end{tabular}}", r"\begin{flushleft}\footnotesize",
              r"Precision of the registered primary endpoint (" + METHOD + r" against "
              r"SEHHO-COBL on CEC2014, 30 functions, 30 runs per arm). The per-function "
              r"standard error is the unbiased estimate of the sampling variance of $\delta$ "
              r"of \citet{cliff1993dominance}, at 30 against 30 runs. The standard error of the mean over functions "
              r"is computed twice: from the spread of $\delta$ between functions (the "
              r"interval the paper reports) and from the run-level errors alone. The "
              r"two-stage bootstrap resamples the functions and then, within each selected "
              r"function, the runs of each arm (20{,}000 resamples, seed 20261003). Over all "
              f"{int(allr.n_functions)} per-function contrasts of {METHOD} with the six "
              f"competitors in the four CEC2014 blocks the median per-function standard error "
              f"is {allr.se_run_median:.3f} (IQR {allr.se_run_q1:.3f}--{allr.se_run_q3:.3f}, "
              f"maximum {allr.se_run_max:.3f}). The unit of inference is the function, and the "
              r"run-level noise of 30 runs adds little to the interval.",
              r"\end{flushleft}", r"\end{table}", ""]
    write(fname, "\n".join(lines), "tab:supp_precision", home)


# ================================================================== S4.1: population rule
POP_LABELS = {"LPSR-6D": r"LPSR from $N_{\text{init}}=6D$", "LPSR-12D": r"LPSR from $N_{\text{init}}=12D$",
              "LPSR-18D": r"LPSR from $N_{\text{init}}=18D$ (L-SHADE's rule)",
              "N30": r"fixed $N=30$ (SEHHO-COBL)", "N4D": r"fixed $N=4D$",
              "N120": r"fixed $N=120$", "N120b": r"fixed $N=120$, budget-based schedule (control)"}


def table_poprule(caption, fname, home="supp"):
    """S4.1: how the population rule was chosen, on the selection suite."""
    _LABELS[fname] = "tab:poprule"
    ranks = _ana("revision_poprule_ranks.csv")
    blocks = sorted({(int(r.dim), int(r.max_fes)) for r in ranks.itertuples()})
    cands = sorted(ranks.candidate.unique())
    mean_rank = {c: float(ranks[ranks.candidate == c].avg_rank.mean()) for c in cands}
    best = min(cands, key=lambda c: mean_rank[c])
    lines = [r"\begin{table}[htbp]", r"\centering", f"\\caption{{{caption}}}",
             r"\label{tab:poprule}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{ll" + "c" * (len(blocks) + 1) + "}", r"\toprule",
             "Label & Population rule & " + " & ".join(
                 f"$D{{=}}{dm}$, {fnum(fes)} FEs" for dm, fes in blocks) + r" & mean \\",
             " & & " + " & ".join(f"$B/D={fnum(fes / dm)}$" for dm, fes in blocks) + r" & \\",
             r"\midrule"]
    for c in sorted(cands, key=lambda c: mean_rank[c]):
        cells = []
        for dm, fes in blocks:
            r = ranks[(ranks.dim == dm) & (ranks.max_fes == fes) & (ranks.candidate == c)].iloc[0]
            cells.append(f"{r.avg_rank:.2f} ({r.position})")
        cells.append(f"{mean_rank[c]:.2f}")
        name = POP_LABELS.get(c, esc(c))
        lab = c
        if c == best:
            lab, name = r"\textbf{" + c + "}", r"\textbf{" + name + "}"
            cells = [r"\textbf{" + x + "}" for x in cells]
        lines.append(f"{lab} & {name} & " + " & ".join(cells) + r" \\")
    # what the selected rule did in each block, read from the ranks
    tight = [(dm, fes) for dm, fes in blocks if fes == 15_000]
    large = [(dm, fes) for dm, fes in blocks if fes > 15_000]
    pos = lambda c, b: int(ranks[(ranks.dim == b[0]) & (ranks.max_fes == b[1]) &
                                 (ranks.candidate == c)].position.iloc[0])
    first_large = {b: ranks[(ranks.dim == b[0]) & (ranks.max_fes == b[1])].sort_values("avg_rank")
                   .candidate.iloc[0] for b in large}
    won_tight = all(pos(best, b) == 1 for b in tight)
    sel_txt = (f"{best} ranked first in {'both' if won_tight and len(tight) == 2 else 'some'} "
               f"tight blocks and " + " and ".join(f"{pos(best, b)}th" if pos(best, b) > 3 else
                                                   {1: '1st', 2: '2nd', 3: '3rd'}[pos(best, b)]
                                                   for b in large)
               + f" of {len(cands)} at the large budgets, where "
               + (f"{first_large[large[0]]} ranked first" if len(set(first_large.values())) == 1 else
                  ", ".join(f"{v} ranked first at $D={k[0]}$" for k, v in first_large.items())))
    control = ""
    if "N120" in mean_rank and "N120b" in mean_rank:
        import analyze_extras
        blk = analyze_extras.poprule_basis_control()
        out = [r for r in blk if not r["negligible"]]
        n_in = len(blk) - len(out)
        where = (f"in all {NUMBER_WORDS.get(len(blk), len(blk))} blocks" if not out else
                 f"in {NUMBER_WORDS.get(n_in, n_in)} of the {NUMBER_WORDS.get(len(blk), len(blk))} blocks")
        parts_ = []
        for r in out:
            dv = led(fname, "N120b vs N120", f"D={r['dim']} {r['max_fes']}", r["estimate"],
                     "N120b", "N120", "results/analysis/extras.csv (c)")
            parts_.append(f"$D{{=}}{r['dim']}$ with {fnum(r['max_fes'])} evaluations, "
                          f"${dv}$ ${cfmt(r['ci_lo'], r['ci_hi'])}$")
        exc = "; ".join(parts_)
        exc = ("" if not out else ("; the exception is " if len(out) == 1 else "; the exceptions are ") + exc)
        control = (r" N120b is a control: the N120 rule with the schedule basis changed to "
                   r"match what LPSR requires. The near-identical mean ranks ("
                   f"${mean_rank['N120']:.2f}$ and ${mean_rank['N120b']:.2f}$) and the per-block "
                   r"differences, which lie inside the negligible band $|\delta|<0.147$ " + where +
                   r" (Cliff's $\delta$ of N120b against N120, negative favouring N120b, 95\% "
                   r"bootstrap interval" + exc + r"), show that the basis change contributes "
                   r"little and the LPSR result is mainly about population size.")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\begin{flushleft}\footnotesize",
              r"Friedman average rank within each block (lower is better; position of 7 in "
              r"parentheses), over the 12 CEC2022 functions with 30 runs per cell; bold marks "
              r"the rule with the best mean rank, which is the one selected. All candidates "
              r"are gate-free and differ only in how the population is sized. This is the "
              r"only constant fitted in the study, and it is fitted on CEC2022, the selection "
              r"suite. " + sel_txt + r": averaging ranks over blocks does not prevent a rule "
              r"from winning through the tight budgets, and the selected $N_{\text{init}}=6D$ "
              r"is fixed at every budget, so it does not size the population to the budget."
              + control,
              r"\end{flushleft}", r"\end{table}", ""]
    write(fname, "\n".join(lines), "tab:poprule", home)


# ================================================================== S4.2: composition
COMP_ORDER = ["+Gate", "-COBL", "-Levy", "-Archive", "-SHADE", "-pAnneal", "Stripped"]
COMP_PRETTY = {"+Gate": r"re-introduce the escape-energy gate",
               "-COBL": r"remove chaotic opposition-based initialisation",
               "-Levy": r"remove the L\'evy perturbation",
               "-Archive": r"remove the external archive",
               "-SHADE": r"remove the success-history memory",
               "-pAnneal": r"remove the $p$ annealing",
               "Stripped": r"\emph{all three dropped parts removed at once}"}


def table_composition(caption, fname, home="supp"):
    _LABELS[fname] = "tab:composition"
    d = _ana("revision_composition.csv")
    blocks = sorted({(int(r.dim), int(r.max_fes)) for r in d.itertuples()})
    lines = [r"\begin{table}[htbp]", r"\centering", f"\\caption{{{caption}}}",
             r"\label{tab:composition}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{l" + "c" * len(blocks) + "}", r"\toprule",
             "Change & " + " & ".join(f"$D{{=}}{dim}$, {fnum(fes)} FEs" for dim, fes in blocks) + r" \\",
             r"\midrule"]
    marks = set()
    for c in COMP_ORDER:
        sub = d[d.change == c]
        if sub.empty:
            continue
        cells = []
        for dim, fes in blocks:
            r = sub[(sub.dim == dim) & (sub.max_fes == fes)].iloc[0]
            dv = led(fname, c, f"D={dim} {fes}", r["mean"], "changed (" + c + ")", "default",
                     "results/analysis/revision_composition.csv")
            mk = robust_mark(r)
            # the tie rule of main text Section 2.6: any unadjusted interval (percentile,
            # BCa or Student-t) that excludes zero with a bound within 0.02 of it
            tie = r"$^{\circ}$" if excludes_zero_tie(r) else ""
            marks |= {mk, tie} - {""}
            txt = f"${dv}$ ${cfmt(r.pct_lo, r.pct_hi)}${mk}{tie}"
            if r.pct_lo > 0 or r.pct_hi < 0:
                txt = r"\textbf{\boldmath " + txt + "}"
            cells.append(txt)
        if c == "Stripped":
            lines.append(r"\midrule")
        lines.append(f"{COMP_PRETTY.get(c, esc(c))} & " + " & ".join(cells) + r" \\")
    mark_txt = ""
    if r"$^{\S}$" in marks:
        mark_txt += r" $^{\S}$: the Student-$t$ or BCa interval includes zero (and so does its adjusted version)."
    if r"$^{\ddagger}$" in marks:
        mark_txt += (r" $^{\ddagger}$: the Bonferroni-adjusted interval over the four blocks "
                     r"of the row (98.75\%) includes zero.")
    if r"$^{\circ}$" in marks:
        mark_txt += (r" $^{\circ}$: the percentile, BCa or Student-$t$ interval excludes zero "
                     r"with its bound nearest zero within 0.02 of it (a tie).")
    cobl = d[(d.change == "-COBL") & (d.dim == 10) & (d.max_fes == 15000)].iloc[0]
    # REV-26: the two verdicts of negligibility the decisions rest on, under the
    # Bonferroni-adjusted Student-t interval, and what the rule would then decide
    dec = _ana("revision_composition_decisions.csv")
    t_adj = dec[(dec.interval == "t_adj") & (np.isclose(dec.margin, 0.147))]
    pann = d[(d.change == "-pAnneal")]
    pann_out = [r for r in pann.itertuples() if _adj_t_verdict(r._asdict(), 0.147) != "negligible"]
    cobl_out = _adj_t_verdict(cobl, 0.147) != "negligible"
    adj_txt = ""
    if cobl_out and pann_out:
        adj_txt = (r" Neither verdict of negligibility survives the Bonferroni-adjusted "
                   r"Student-$t$ interval: removing opposition-based initialisation gives "
                   f"${cfmt(cobl.t_adj_lo, cobl.t_adj_hi)}$ there, and removing the $p$ "
                   r"annealing " + " and ".join(
                       f"${cfmt(r.t_adj_lo, r.t_adj_hi)}$ at $D{{=}}{int(r.dim)}$ with "
                       f"{fnum(r.max_fes)} evaluations" for r in pann_out) + ".")
        if (t_adj.decision == "KEEP (by default)").all():
            adj_txt += (r" Under those intervals the rule would keep every part by default, so "
                        r"every decision here is directional; the frozen configuration does "
                        r"not change.")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\begin{flushleft}\footnotesize",
              r"Mean Cliff's $\delta$ over the 12 CEC2022 functions with a 95\% percentile "
              r"bootstrap interval (20{,}000 resamples). " + CONV + r" Here A is the changed "
              r"configuration and B the gate-free default, so a positive interval means the "
              r"part earns its place and a negative one that the method is better without it. "
              r"Bold marks an interval that excludes zero." + mark_txt + r" BCa, Student-$t$ "
              r"and adjusted intervals for every cell are in Table~"
              + ref("tab:supp_composition_intervals", home) + r". Twelve functions give too "
              r"little power for Holm-corrected significance to decide anything here, so the "
              r"decisions read the intervals. They follow the amended rule: a part is dropped "
              r"when no interval lies above zero and it is either negligible in every block "
              r"($|\delta|<0.147$) or has an interval below zero; otherwise it is kept. That "
              r"rule was adopted after the output of the originally coded rule on these data "
              r"had been seen. The original rule dropped a part only if it was negligible in "
              r"every block; it would have kept opposition-based initialisation and the L\'evy "
              r"perturbation, and agrees with the amended rule on the other four single "
              r"changes. Under the amended rule the archive and the success-history memory "
              r"earn their place; opposition-based initialisation, the L\'evy perturbation and "
              r"the $p$ annealing do not. Opposition-based initialisation never helps, and "
              r"removing it changes performance by a negligible amount (at $D{=}10$ with "
              f"15{{,}}000 evaluations ${dfmt(cobl['mean'])}$ ${cfmt(cobl.pct_lo, cobl.pct_hi)}$, "
              r"inside the band; the Student-$t$ bound reaches zero, "
              f"${cfmt(cobl.t_lo, cobl.t_hi)}$). The $p$ annealing is negligible in every block "
              r"only at the 0.147 margin." + adj_txt + r" The last row (stage E11, added after the original "
              r"rule's output had been seen) removes all three together. The first row "
              r"re-introduces the gate into the gate-free configuration; its harm at the "
              r"competition budgets is directionally consistent, within the CEC2017 function "
              r"family, with the large-budget result on CEC2017, for a greedier gate: here it is "
              r"scheduled on generations counted from the initial population, so under "
              r"population reduction it closes after about 40\% of the budget rather than half "
              r"and opens on about 12--13\% of updates rather than 15.3\%. At 15{,}000 "
              r"evaluations the two configurations are not separated.",
              r"\end{flushleft}", r"\end{table}", ""]
    write(fname, "\n".join(lines), "tab:composition", home)


def table_supp_composition_intervals(caption, fname, home="supp"):
    _LABELS[fname] = "tab:supp_composition_intervals"
    d = _ana("revision_composition.csv")
    dec = _ana("revision_composition_decisions.csv")
    lines = [r"\begin{table}[htbp]", r"\centering", f"\\caption{{{caption}}}",
             r"\label{tab:supp_composition_intervals}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{llrccccc}", r"\toprule",
             r"Change & Block & $\bar\delta$ & percentile & BCa & Student-$t$ & "
             r"percentile, 98.75\% & Student-$t$, 98.75\% \\", r"\midrule"]
    for c in COMP_ORDER:
        sub = d[d.change == c]
        dd = dec[(dec.change == c)]
        base = dd[(dd.interval == "pct") & (dd.margin == 0.147)].decision.iloc[0]
        alt = dd[dd.decision != base]
        alt_txt = ("same under every interval and margin" if alt.empty else
                   "differs under " + ", ".join(sorted({f"{r.interval.replace('_adj', ' adj.')}"
                                                        f" at {r.margin:g}" for r in alt.itertuples()})))
        lines.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{{COMP_PRETTY[c]}}} -- amended rule: "
                     f"{esc(base)}; {esc(alt_txt)}}} \\\\")
        for r in sub.to_dict("records"):
            dv = led(fname, c, f"D={r['dim']} {r['max_fes']}", r["mean"], "changed (" + c + ")",
                     "default", "results/analysis/revision_composition.csv")
            lines.append(f"\\quad & $D{{=}}{r['dim']}$, {fnum(r['max_fes'])} & ${dv}$ & "
                         f"${cfmt(r['pct_lo'], r['pct_hi'])}$ & ${cfmt(r['bca_lo'], r['bca_hi'])}$ & "
                         f"${cfmt(r['t_lo'], r['t_hi'])}$ & ${cfmt(r['pct_adj_lo'], r['pct_adj_hi'])}$ & "
                         f"${cfmt(r['t_adj_lo'], r['t_adj_hi'])}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\begin{flushleft}\footnotesize",
              r"Every cell of Table~" + ref("tab:composition", home) + r" with BCa, Student-$t$ "
              r"($t_{11}$, truncated to $[-1,1]$) and Bonferroni-adjusted intervals for the "
              r"family of the four blocks of one change. Percentile and BCa intervals are read "
              r"from the same resamples (seed 20260914, in the order of "
              r"\texttt{analyze\_composition.py}, so the percentile intervals are the ones it "
              r"reports). " + CONV + r" Here A is the changed configuration. The heading of "
              r"each change gives the decision of the amended rule with the 95\% percentile "
              r"interval at the 0.147 margin, and the intervals and margins (0.147, 0.10, "
              r"0.05) under which the same rule would decide differently; the frozen "
              r"configuration does not change. For the re-introduced gate, ``KEEP'' means "
              r"that the gate stays out.",
              r"\end{flushleft}", r"\end{table}", ""]
    write(fname, "\n".join(lines), "tab:supp_composition_intervals", home)


# ================================================================== S4.4: E15
def table_supp_e15(caption, fname, home="supp"):
    _LABELS[fname] = "tab:supp_e15"
    e = _ana("e15_original_rule.csv")
    pre = json.load(open(os.path.join(ANA, "PREREGISTRATION_E15_original_rule.json")))
    prim = _ana("revision_cec2014_primary.csv")
    reg = prim[prim.analysis == "registered primary endpoint"].set_index("dim")
    pw = _ana("revision_cec2014_pairwise.csv")
    blocks = [(30, 300_000), (50, 500_000), (30, 15_000), (50, 15_000)]
    rows = [("GF:OriginalRule", "SEHHO-COBL", "original-rule configuration vs SEHHO-COBL"),
            ("GF-Method", "SEHHO-COBL", METHOD + " vs SEHHO-COBL (E13)"),
            ("GF:OriginalRule", "GF-Method", "original-rule configuration vs " + METHOD)]
    lines = [r"\begin{table}[htbp]", r"\centering", f"\\caption{{{caption}}}",
             r"\label{tab:supp_e15}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{l" + "c" * len(blocks) + "}", r"\toprule",
             "Contrast & " + " & ".join(f"$D={dm}$, {fnum(fes)} FEs" for dm, fes in blocks) + r" \\",
             r"\midrule"]
    for a, b, name in rows:
        cells = []
        for dm, fes in blocks:
            r = e[(e.dim == dm) & (e.max_fes == fes) & (e.a == a) & (e.b == b)].iloc[0]
            lo, hi, src = r.ci_lo, r.ci_hi, "results/analysis/e15_original_rule.csv"
            if a == "GF-Method":
                # one interval per contrast: the registered E13 analysis
                if fes > 15_000:
                    q = reg.loc[dm]
                    lo, hi = q.pct_lo, q.pct_hi
                else:
                    q = pw[(pw.dim == dm) & (pw.max_fes == fes) & (pw.opponent == "SEHHO-COBL")].iloc[0]
                    lo, hi = q.pct_lo, q.pct_hi
                src = "registered E13 analysis (revision_cec2014_*.csv)"
            an = "original-rule configuration" if a == "GF:OriginalRule" else METHOD
            bn = METHOD if b == "GF-Method" else b
            dv = led(fname, name, f"D={dm} {fes}", r.mean_delta, an, bn, src)
            ph = r.p_holm if np.isfinite(r.p_holm) else np.nan
            txt = f"${dv}$ ${cfmt(lo, hi)}$" + (f", $p_{{\\text{{Holm}}}}$ {ptex(ph)}" if np.isfinite(ph) else "")
            if (lo > 0 or hi < 0) and a == "GF:OriginalRule" and b == "SEHHO-COBL" and fes > 15_000:
                txt = r"\textbf{\boldmath " + txt + "}"
            cells.append(txt)
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    # the last row's family-adjusted intervals (not part of the registration; Section 2.6)
    adj = _ana("revision_e15.csv")
    cells, flips = [], []
    for dm, fes in blocks:
        q = adj[(adj.dim == dm) & (adj.max_fes == fes)].iloc[0]
        cells.append(f"${cfmt(q.pct_adj_lo, q.pct_adj_hi)}$; ${cfmt(q.t_adj_lo, q.t_adj_hi)}$")
        if q["verdict_pct_0.147"] == "negligible" and (q["verdict_t_adj_0.147"] != "negligible"
                                                       or q["verdict_pct_adj_0.147"] != "negligible"):
            flips.append(f"$D={dm}$ with {fnum(fes)} evaluations")
    lines.append(r"\quad same, Bonferroni-adjusted (98.75\%): percentile; Student-$t$ & "
                 + " & ".join(cells) + r" \\")
    flip_txt = (r" The last row adjusts the third row's intervals for its family of four blocks "
                r"(Bonferroni, 98.75\%; not part of the registration, whose rule uses the "
                r"unadjusted 95\% interval)" +
                (r"; under the adjustment the interval at " + " and ".join(flips) +
                 r" reaches beyond the 0.147 margin, so that verdict of negligibility is "
                 r"directional (main text, Section~" + ref("subsec:stats", home) + r")."
                 if flips else r" and changes no verdict."))
    lines += [r"\bottomrule", r"\end{tabular}}", r"\begin{flushleft}\footnotesize",
              r"Stage E15 (exploratory): the configuration the original composition rule would "
              r"have produced (gate-free, LPSR from $N_{\text{init}}=6D$, only the $p$ annealing "
              r"removed, opposition-based initialisation and the L\'evy perturbation kept; "
              r"label \texttt{GF:OriginalRule} in the data), run in the cells of the "
              r"confirmatory evaluation. Registered internally at "
              + (datetime.fromisoformat(pre["written_at"]).astimezone(timezone.utc)
                 .strftime("%Y-%m-%d %H:%M UTC")) + r" before it ran, after the confirmatory "
              r"analysis and after the amendment it examines; it cannot change the frozen "
              r"configuration. " + CONV_FN + r" Here A is the first-named configuration. "
              r"Intervals are 95\% percentile bootstrap intervals over the 30 CEC2014 functions; "
              r"the middle row uses the registered E13 intervals. The registered sensitivity "
              r"criterion (bold) was that the first row exclude zero in both competition-budget "
              r"blocks; $p_{\text{Holm}}$ as registered (two-block family for the first row at "
              r"the competition budget, four-block family for the third row)." + flip_txt,
              r"\end{flushleft}", r"\end{table}", ""]
    write(fname, "\n".join(lines), "tab:supp_e15", home)


# ================================================================== Table 7
PROB_PRETTY = {"P1_pressure_vessel": "P1 Pressure vessel ($D{=}4$)",
               "P2_gear_train": "P2 Gear train ($D{=}4$)",
               "P3_heat_exchanger_RC01": "P3 Heat-exchanger network, RC01 ($D{=}9$)",
               "P4_blending_pooling_RC06": "P4 Blending--pooling--separation, RC06 ($D{=}38$)",
               "P5_batch_plant_RC14": "P5 Multi-product batch plant, RC14 ($D{=}10$)"}
REF_TEXT = {"P1_pressure_vessel": "thickness grid + SLSQP on $(x_3,x_4)$",
            "P2_gear_train": "enumeration of all $49^4$ tooth combinations",
            "P3_heat_exchanger_RC01": r"closed-form design, Eq.~(\ref{eq:rc01star_main})",
            "P4_blending_pooling_RC06": r"SciPy \texttt{SLSQP}, multi-start (Table~\ref{tab:engbudget})",
            "P5_batch_plant_RC14": "27 unit assignments $\\times$ multi-start SLSQP"}
REF_SHORT = {"P1_pressure_vessel": "grid + SLSQP",
             "P2_gear_train": "enumeration",
             "P3_heat_exchanger_RC01": r"Eq.~(\ref{eq:rc01star_main})",
             "P4_blending_pooling_RC06": "SLSQP",
             "P5_batch_plant_RC14": "enumeration + SLSQP"}
REF_METHOD = {"P1_pressure_vessel": "thickness grid + SLSQP on (x3, x4)",
              "P2_gear_train": "full enumeration of 49^4 tooth combinations",
              "P3_heat_exchanger_RC01": "equality elimination, closed form (rc01_analytic.py)",
              "P4_blending_pooling_RC06": "SciPy SLSQP, multi-start (E5c)",
              "P5_batch_plant_RC14": "27 integer unit assignments"}


def _refrow(refs, prob):
    s = refs[(refs.problem == prob) & refs.method.str.startswith(REF_METHOD[prob])]
    return s.iloc[0]


def _fullnum(v):
    if abs(v) < 1e-3:
        return f"{v:.6E}"
    if abs(v) >= 1e4:
        return f"{v:,.2f}".replace(",", "{,}")
    return f"{v:.4f}" if abs(v) >= 100 else f"{v:.6f}"


def table_engineering(df, caption, fname, home="main"):
    _LABELS[fname] = "tab:engineering"
    succ = _ana("revision_engineering_success.csv")
    refs = _ana("revision_classical_refs.csv")
    probs = sorted(df["problem"].unique())
    algos = [a for a in ORDER if a in set(df.algo)]
    lines = [r"\begin{table}[htbp]", r"\centering", f"\\caption{{{caption}}}",
             r"\label{tab:engineering}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{llrrrrrrr}", r"\toprule",
             r"Problem & Algorithm & Feasible & Success & Best feasible & Mean feasible & "
             r"Std & Mean violation & Evaluations \\", r"\midrule"]
    inv = {v: k for k, v in DISPLAY.items()}
    partial = {}        # problem -> [(algorithm, feasible runs, runs)] with 0 < feasible < runs
    for pr in probs:
        sub = df[df["problem"] == pr]
        # NEW-3: a mean over the feasible runs is conditional on feasibility, so means are
        # compared (bold) only among algorithms feasible in every run; a mean over fewer
        # feasible runs is marked and never bolded
        full = [a for a in algos if len(sub[sub.algo == a]) and sub[sub.algo == a].feasible.all()]
        best_mean = (min(sub[sub.algo == a].objective.mean() for a in full)
                     if len(full) >= 2 else None)
        lines.append(f"\\multicolumn{{9}}{{l}}{{\\textit{{{PROB_PRETTY.get(pr, esc(pr))}}}}} \\\\")
        for a in algos:
            s = sub[sub.algo == a]
            fs = s[s.feasible]
            rate = 100.0 * s.feasible.mean()
            sr = succ[(succ.problem == pr) & (succ.algo == inv.get(a, a))].iloc[0]
            scell = f"{100.0 * sr.success / sr.runs:.0f}\\%"
            if len(fs):
                mb = sci(fs.objective.mean())
                if len(fs) < len(s):
                    mb += r"$^{a}$"
                    partial.setdefault(pr, []).append((a, len(fs), len(s)))
                elif best_mean is not None and np.isclose(fs.objective.mean(), best_mean,
                                                          rtol=1e-6, atol=0.0):
                    mb = f"\\textbf{{{mb}}}"
                cells = [f"{rate:.0f}\\%", scell, sci(fs.objective.min()), mb,
                         sci(fs.objective.std(ddof=1)) if len(fs) > 1 else "--",
                         sci(s.total_violation.mean()), fnum(s.max_fes.iloc[0])]
            else:
                cells = [f"{rate:.0f}\\%", scell, "--", "--", "--", sci(s.total_violation.mean()),
                         fnum(s.max_fes.iloc[0])]
            lines.append(f" & {esc(a)} & " + " & ".join(cells) + r" \\")
        rr = _refrow(refs, pr)
        ev = "0" if int(rr.evaluations) == 0 else fnum(rr.evaluations)
        txt = REF_SHORT[pr].replace(r"\ref{eq:rc01star_main}", ref("eq:rc01star_main", home))
        lines.append(f" & \\textit{{reference: {txt}}} & \\textit{{--}} & \\textit{{--}} & "
                     f"\\textit{{{_fullnum(rr.objective)}}} & \\textit{{--}} & \\textit{{--}} & "
                     f"\\textit{{{sci(rr.total_violation) if rr.total_violation > 0 else '0'}}} & "
                     f"\\textit{{{ev}}} \\\\")
        if not sub.feasible.any():
            lines.append(r"\multicolumn{9}{l}{\footnotesize No metaheuristic reached the declared "
                         r"tolerance in any run.} \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    partial_txt = ""
    if partial:
        short = {"P1_pressure_vessel": "P1", "P2_gear_train": "P2",
                 "P3_heat_exchanger_RC01": "RC01", "P4_blending_pooling_RC06": "RC06",
                 "P5_batch_plant_RC14": "RC14"}
        parts = [short.get(pr, esc(pr)) + ": " + ", ".join(f"{esc(a)} {k} of {n}"
                                                          for a, k, n in lst)
                 for pr, lst in partial.items()]
        partial_txt = (r" $^{a}$: mean over the feasible runs only (" + "; ".join(parts)
                       + r"); a mean over a subset of runs is conditional on feasibility and is "
                       r"not comparable across algorithms with different numbers of feasible "
                       r"runs, so it carries no bold.")
    rc01 = df[(df["problem"] == "P3_heat_exchanger_RC01") & df.feasible]
    rc01_gap = (f" The best feasible RC01 design found by any metaheuristic is "
                f"${100 * (rc01.objective.min() - RC01_ANALYTIC) / RC01_ANALYTIC:.1f}\\%$ above the "
                f"closed-form design." if len(rc01) else "")
    p1 = succ[succ.problem == "P1_pressure_vessel"]
    at = p1[p1.runs_at_threshold == p1.runs]
    p1ref = _refrow(refs, "P1_pressure_vessel").objective
    p1best = df[(df.problem == "P1_pressure_vessel") & df.feasible].objective.min()
    rc14 = _refrow(refs, "P5_batch_plant_RC14")
    lines += [r"\end{tabular}}", r"\begin{flushleft}\footnotesize",
              r"Deb's feasibility rule, 30 runs, 15{,}000 evaluations per run, equality "
              r"tolerance $\varepsilon=10^{-4}$; a design is feasible when its total violation "
              r"$v$ (the unnormalised sum over constraints) is at most $10^{-8}$, a threshold "
              r"that is not scale-aware. ``Feasible'': percentage of runs returning a feasible "
              r"design; best/mean/std over feasible runs only; \textbf{bold}, the lowest mean "
              r"among the algorithms feasible in every run (within a relative $10^{-6}$)."
              + partial_txt + r" ``Success'': feasible and "
              r"within a relative $10^{-4}$ of the reference value $f^{*}$ of the italic row, "
              r"$(f-f^{*})/|f^{*}|\le10^{-4}$ (the suite's own success rate uses an absolute "
              r"$f-f^{*}\le10^{-8}$; \citealp{kumar2020rwco}). The italic row is a classical reference with the evaluations "
              r"it used, one per distinct decision vector (methods in Table~"
              + ref("tab:supp_classical_refs", home) + r"); for RC01, RC06 and RC14 it equals "
              r"the best-known value listed for the suite \citep{kumar2020rwco}. RC14's "
              f"{_fullnum(rc14.objective)} needs one unit per stage; every L-SHADE and jSO run "
              r"ends near 58{,}500, the cost of the configuration with two units in the first "
              r"stage (inferred from the objective; the runs do not store designs). Mean "
              r"violation: the "
              r"average $v$ of the returned design over all 30 runs; 1.00E-08 means that every "
              r"run sits on the threshold, here of P1's volume constraint (terms of order "
              r"$10^{6}$ in$^3$), which is worth "
              f"${texp(p1ref - p1best, 1)}$ in cost ({p1best:.6f} against {p1ref:.6f}). The RC01 "
              r"closed-form design is feasible under the declared rule but is not a certified "
              r"optimum of the tolerance-relaxed problem (Table~" + ref("tab:solverbaseline", home)
              + r")." + rc01_gap,
              r"\end{flushleft}", r"\end{table}", ""]
    write(fname, "\n".join(lines), "tab:engineering", home)


def table_supp_classical_refs(caption, fname, home="supp"):
    _LABELS[fname] = "tab:supp_classical_refs"
    refs = _ana("revision_classical_refs.csv")
    lines = [r"\begin{table}[htbp]", r"\centering", f"\\caption{{{caption}}}",
             r"\label{tab:supp_classical_refs}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{lp{5.2cm}rrrp{6.2cm}}", r"\toprule",
             r"Problem & Method & Objective & $v$ & Evaluations & Design vector / detail \\", r"\midrule"]
    def tex_text(t):
        return (esc(t).replace("49^4", "$49^4$").replace("(x3, x4)", "$(x_3, x_4)$")
                .replace("99x99", r"$99\times99$").replace(" x 3-start", r" $\times$ 3-start")
                .replace("r\\_min", r"$r_{\min}$"))
    for r in refs.to_dict("records"):
        x = r["x"] if isinstance(r["x"], str) and r["x"] else ""
        detail = r["detail"] if not str(r["detail"]).startswith("closed form") else "closed form"
        det = tex_text(detail) + (f"; $\\mathbf{{x}}$ = ({esc(', '.join(x.split()))})" if x else "")
        lines.append(f"{esc(r['problem'].split('_')[0])} {esc(r['problem'].split('_')[-1]) if 'RC' in r['problem'] else ''} & "
                     f"{tex_text(r['method'])} & {_fullnum(r['objective'])} & "
                     f"{sci(r['total_violation']) if r['total_violation'] > 0 else '0'} & "
                     f"{fnum(r['evaluations'])} & {det} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\begin{flushleft}\footnotesize",
              r"Classical references computed with the problem code of the experiments "
              r"(\texttt{scripts/classical\_references.py}; RC01 and RC06 rows from stage E5c). "
              r"Evaluations: one per distinct decision vector, so a finite-difference gradient "
              r"costs about $D+1$; for E5c rows, per run. $v$ is the total violation of the "
              r"design. P1's thicknesses are on the 0.0625-in grid and P2's tooth counts and "
              r"RC14's unit counts are integers, as in the experiments. The RC14 entry also "
              r"lists the best designs with two units in the first stage, whose costs match the "
              r"objectives at which the SHADE-family runs end (the runs do not store designs).",
              r"\end{flushleft}", r"\end{table}", ""]
    write(fname, "\n".join(lines), "tab:supp_classical_refs", home)


# ================================================================== Table 8 and S6.9: E16
E16_SHORT = {"P1_pressure_vessel": "P1 pressure vessel", "P2_gear_train": "P2 gear train",
             "P3_heat_exchanger_RC01": "RC01", "P4_blending_pooling_RC06": "RC06",
             "P5_batch_plant_RC14": "RC14"}


def _num(v, prob):
    if not np.isfinite(v):
        return "--"
    if prob == "P2_gear_train":
        return sci(v)
    return f"{v:,.1f}".replace(",", "{,}") if abs(v) >= 1000 else f"{v:.2f}"


def table_e16(caption, fname, home="main"):
    _LABELS[fname] = "tab:e16"
    e = _ana("revision_e16.csv")
    # the registration's file and time are reported in Supplementary Section S2.5 (REV-10)
    e = e.sort_values("evals_per_variable")
    lines = [r"\begin{table}[htbp]", r"\centering", f"\\caption{{{caption}}}",
             r"\label{tab:e16}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{lrrccccccl}", r"\toprule",
             r" & & & Feasible & Success & Mean feasible & Median feasible & & & \\",
             r"Problem & $D$ & $B/D$ & $6D$ / $18D$ & $6D$ / $18D$ & $6D$ / $18D$ & $6D$ / $18D$ & "
             r"$\delta$($6D$ vs $18D$) [95\% CI] & $p_{\text{Holm}}$ & Verdict \\", r"\midrule"]
    for r in e.to_dict("records"):
        dv = led(fname, E16_SHORT[r["problem"]], "delta", r["delta"], "L-SHADE at 6D",
                 "L-SHADE at 18D", "results/analysis/e16_engineering_population.csv (E16, registered)")
        lines.append(f"{E16_SHORT[r['problem']]} & {r['dim']} & {fnum(round(r['evals_per_variable']))} & "
                     f"{r['feasible_6D']} / {r['feasible_18D']} & {r['success_6D']} / {r['success_18D']} & "
                     f"{_num(r['mean_feasible_6D'], r['problem'])} / {_num(r['mean_feasible_18D'], r['problem'])} & "
                     f"{_num(r['median_feasible_6D'], r['problem'])} / {_num(r['median_feasible_18D'], r['problem'])} & "
                     f"${dv}$ ${cfmt(r['ci_lo'], r['ci_hi'])}$ & {ptex(r['mw_p_holm'])} & {esc(r['verdict'])} \\\\")
    rc01 = e[e.problem == "P3_heat_exchanger_RC01"].iloc[0]
    rc14 = e[e.problem == "P5_batch_plant_RC14"].iloc[0]
    lines += [r"\bottomrule", r"\end{tabular}}", r"\begin{flushleft}\footnotesize",
              r"Stock L-SHADE (published settings, $H=6$, archive $2.6N$, $p=0.11$, "
              r"LPSR to $N_{\min}=4$) with only $N_{\text{init}}$ changed, $\mathrm{round}(6D)$ "
              r"against $\mathrm{round}(18D)$; 15{,}000 evaluations, 30 runs per arm, Deb's rule "
              r"with $\varepsilon=10^{-4}$. Registered before any of its runs (Supplementary "
              r"Section~" + ref("sec:S2_prereg", home) + r") and exploratory by its own "
              r"registration. Counts are out "
              r"of 30 runs. Success: feasible and $(f-f^{*})/|f^{*}|\le10^{-4}$, with $f^{*}$ "
              r"the reference value of Table~" + ref("tab:engineering", home) + r" (counts under "
              r"the suite's absolute rule, $f-f^{*}\le10^{-8}$, are in Table~"
              + ref("tab:supp_e16", home) + r"); success rate and the mean among feasible runs are "
              r"not in the registration and are descriptive. " + CONV + r" Here A is $6D$; $\delta$ is computed over the runs on "
              r"the Deb-rule fitness the runs minimise (the objective for a feasible run, the "
              r"constant $F_{\text{ref}}$ plus the violation otherwise), with a 95\% percentile "
              r"bootstrap interval over runs (20{,}000 resamples); $p_{\text{Holm}}$ is a "
              r"two-sided Mann--Whitney test, Holm-corrected over the five problems. Verdicts "
              r"follow the registered rule. On RC06 no run of either arm is feasible, so "
              r"$\delta$ compares violations only. On RC14 both arms end at about "
              f"{rc14.median_feasible_6D:,.0f}".replace(",", "{,}") + r", about "
              f"{100 * rc14.excess_median_6D:.0f}\\% above the reference "
              f"{_fullnum(rc14.f_star)}. On RC01 the feasibility difference ("
              f"{rc01.feasible_6D}/30 against {rc01.feasible_18D}/30) is not significant "
              f"(Fisher's exact test, $p={rc01.fisher_p:.2f}$).",
              r"\end{flushleft}", r"\end{table}", ""]
    write(fname, "\n".join(lines), "tab:e16", home)


def table_supp_e16(caption, fname, home="supp"):
    _LABELS[fname] = "tab:supp_e16"
    e = _ana("revision_e16.csv").sort_values("evals_per_variable")
    lines = [r"\begin{table}[htbp]", r"\centering", f"\\caption{{{caption}}}",
             r"\label{tab:supp_e16}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{llrrrrrrrrr}", r"\toprule",
             r" & & & & & & \multicolumn{4}{c}{runs within a relative $\tau$ of $f^{*}$} & suite rule \\",
             r"\cmidrule(lr){7-10} \cmidrule(lr){11-11}",
             r"Problem & Arm & Feasible & Best feasible & Median feasible & Median $v$ & "
             r"$\tau=10^{-6}$ & $10^{-4}$ & $10^{-3}$ & $10^{-2}$ & $f-f^{*}\le10^{-8}$ \\", r"\midrule"]
    for r in e.to_dict("records"):
        for lab in ("6D", "18D"):
            lines.append(f"{E16_SHORT[r['problem']] if lab == '6D' else ''} & ${lab}$ & "
                         f"{r['feasible_' + lab]}/30 & {_num(r['best_feasible_' + lab], r['problem'])} & "
                         f"{_num(r['median_feasible_' + lab], r['problem'])} & "
                         f"{sci(r['median_violation_' + lab])} & {r['success_' + lab + '_tol1e-06']} & "
                         f"{r['success_' + lab]} & {r['success_' + lab + '_tol0.001']} & "
                         f"{r['success_' + lab + '_tol0.01']} & {r['success_' + lab + '_suite_abs1e-8']} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}}", r"\begin{flushleft}\footnotesize",
              r"Per-problem detail of stage E16 (Table~" + ref("tab:e16", home) + r"). $v$ is "
              r"the total constraint violation of the returned design (zero or at most "
              r"$10^{-8}$ for a feasible design). The success threshold of Table~"
              + ref("tab:e16", home) + r" is $\tau=10^{-4}$; the other columns show how the "
              r"counts depend on it. The last column applies the success rule of the CEC2020 suite "
              r"\citep{kumar2020rwco}, a feasible design with $f-f^{*}\le10^{-8}$ (absolute): on P2, "
              r"where $f^{*}=2.7\times10^{-12}$, it accepts every design with $f\le10^{-8}$; on "
              r"RC14, where $f^{*}\approx5.4\times10^{4}$, it requires agreement to "
              r"$2\times10^{-13}$ relative. $f^{*}$ as in Table~" + ref("tab:engineering", home) + r".",
              r"\end{flushleft}", r"\end{table}", ""]
    write(fname, "\n".join(lines), "tab:supp_e16", home)


# ================================================================== Table 9 (merged)
def table_rc01rc06(e5b, solver, eng, caption, fname, home="main"):
    """Table 9: every cell of the budget sweep and of the solver comparison."""
    _LABELS[fname] = "tab:engbudget"
    pretty = {"P3_heat_exchanger_RC01": "RC01 heat-exchanger network ($D{=}9$, 8 equalities)",
              "P4_blending_pooling_RC06": "RC06 blending--pooling--separation ($D{=}38$, 32 equalities)"}
    budgets = sorted(e5b.max_fes.unique())
    algos = [a for a in ORDER if a in set(e5b.algo)]
    lines = [r"\begin{table}[htbp]", r"\centering", f"\\caption{{{caption}}}",
             r"\label{tab:engbudget}", r"\label{tab:solverbaseline}", r"\small",
             r"\setlength{\tabcolsep}{4.5pt}",
             r"\begin{tabular}{l" + "c" * len(budgets) + "}", r"\toprule",
             r"\multicolumn{" + str(1 + len(budgets)) + r"}{l}{\emph{Panel A. Feasibility rate "
             r"(best feasible objective) against the evaluation budget}} \\", r"\midrule",
             "Algorithm & " + " & ".join(
                 [f"${b // 1000}$k" if b < 1_000_000 else "$10^{6}$" for b in budgets]) + r" \\",
             r"\midrule"]
    for pr in sorted(e5b["problem"].unique()):
        lines.append(f"\\multicolumn{{{1 + len(budgets)}}}{{l}}{{\\textit{{{pretty.get(pr, esc(pr))}}}}} \\\\")
        for a in algos:
            cells = []
            for b in budgets:
                s = e5b[(e5b["problem"] == pr) & (e5b.algo == a) & (e5b.max_fes == b)]
                if not len(s):
                    cells.append("--")
                    continue
                fs = s[s.feasible]
                cells.append(f"{100.0 * s.feasible.mean():.0f}\\% ({sci(fs.objective.min())})" if len(fs)
                             else "0\\%")
            lines.append(f"\\quad {esc(a)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", "", r"\vspace{0.8em}",
              r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lrrrr}", r"\toprule",
              r"\multicolumn{5}{l}{\emph{Panel B. Methods designed for explicit constraints, at "
              r"15{,}000 evaluations}} \\", r"\midrule",
              r"Method & Feasible & Best feasible & Excess over ref. & Evaluations \\",
              r"\midrule"]
    n_meta = eng.algo.nunique()
    meta = f"best of the {NUMBER_WORDS.get(n_meta, str(n_meta))} metaheuristics"
    for pr in sorted(solver["problem"].unique()):
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{pretty.get(pr, esc(pr))}}}}} \\\\")
        e = eng[eng["problem"] == pr]
        ef = e[e.feasible]
        if len(ef):
            best = ef.objective.min()
            who = ef.loc[ef.objective.idxmin(), "algo"]
            rate = 100.0 * e[e.algo == who].feasible.mean()
            gap = f"{100 * (best - RC01_ANALYTIC) / RC01_ANALYTIC:+.2f}\\%" if pr.startswith("P3") else "--"
            lines.append(f"\\quad {meta} ({esc(who)}) & {rate:.0f}\\% & {sci(best)} & {gap} & 15{{,}}000 \\\\")
        else:
            lines.append(f"\\quad {meta} & 0\\% & -- & -- & 15{{,}}000 \\\\")
        for a in ("SLSQP", "trust-constr", "Equality elimination (closed form)"):
            s2 = solver[(solver["problem"] == pr) & (solver.algo == a)]
            if not len(s2):
                continue
            sf = s2[s2.feasible]
            rate = 100.0 * s2.feasible.mean()
            if len(sf):
                best = sf.objective.min()
                gap = f"{100 * (best - RC01_ANALYTIC) / RC01_ANALYTIC:+.4f}\\%" if pr.startswith("P3") else "--"
                cell = sci(best)
            else:
                gap, cell = "--", "--"
            fes = "0" if "closed form" in a else fnum(s2.fes_used.mean())
            nm = (r"\textbf{equality elimination} (closed form)" if "closed" in a
                  else f"SciPy \\texttt{{{esc(a)}}}")
            lines.append(f"\\quad {nm} & {rate:.0f}\\% & {cell} & {gap} & {fes} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\begin{flushleft}\footnotesize",
              r"\textbf{Panel A}: feasibility rate over 30 runs at each budget, with the best "
              r"feasible objective in parentheses where any run succeeded; the 15k column is an "
              r"independent replication of Table~" + ref("tab:engineering", home) + r" with a "
              r"different seed stream (37\% against 40\% for SEHHO-COBL on RC01, 30 runs each). "
              r"\textbf{Panel B}: every row uses the same feasibility rule and equality tolerance "
              r"$\varepsilon=10^{-4}$; ``Feasible'' is the percentage of 30 runs whose returned "
              r"design satisfies every constraint. The two SciPy solvers are driven from random "
              r"multi-starts inside the same 15{,}000-evaluation budget, with one evaluation "
              r"charged per distinct decision vector (so a finite-difference gradient costs "
              r"about $D+1$) and the equalities passed to the solver explicitly. ``Excess over "
              r"ref.'' is measured against the closed-form feasible RC01 design of "
              r"Equation~(\ref{eq:rc01star_main}), $f(\mathbf{x}^{*})=189.3116$; that design is "
              r"feasible under the declared rule, so every entry in this column is a "
              r"\emph{lower bound} on the true excess, not a gap against a certified optimum. "
              r"Neither solver is uniformly better: SLSQP fails on RC01 and trust-constr on "
              r"RC06. RC06 has no known closed-form reference; its best SLSQP design equals the "
              r"best-known value the suite lists (1.8638). Equality tolerance and feasibility "
              r"threshold as in Table~" + ref("tab:engineering", home) + r"; the static "
              r"feasibility rule scopes every statement about the metaheuristics here.",
              r"\end{flushleft}", r"\end{table}", ""]
    write(fname, "\n".join(lines), "tab:engbudget", home)
    # The two former tables are one float now; leave a stub so that a stale \input
    # of the old file produces nothing (and no duplicate label) until it is removed.
    p = os.path.join(TEX, "tab_solverbaseline.tex")
    with open(p, "w") as fh:
        fh.write("% Merged into tab_engbudget.tex (Table 9, labels tab:engbudget and "
                 "tab:solverbaseline).\n% Remove the \\input of this file; it is "
                 "intentionally empty.\n")
    print("wrote", p, "(stub)")


# ================================================================== Table 10
LITERATURE = [
    # (B/D label, text, verified?).  Tuned initial population rate of restart
    # L-SHADE (RL-SHADE) in Tanabe & Fukunaga (2015), Table I, as checked against the
    # paper by the literature task (stage4/LITERATURE.md, section 1.1).  These are
    # literature values, not results of this study, and no CSV holds them.
    ("$10^2$", r"tuned $N_{\text{init}}=5.19D$, restart L-SHADE \citep{tanabe2015tuning}", True),
    ("$10^4$", r"tuned $N_{\text{init}}=13.63D$, no restarts, i.e.\ L-SHADE (same study)", True),
    ("$10^5$", r"tuned $N_{\text{init}}=16.39D$, restart L-SHADE (same study)", True),
]


def table_guide(caption, fname, home="main"):
    """Table 10: delta(6D vs 18D) indexed by evaluations per variable, row by row from
    revision_budget_guide.csv, with the bands the data do not cover."""
    _LABELS[fname] = "tab:guide"
    g = _ana("revision_budget_guide.csv")
    ps = _ana("revision_popsize.csv").set_index("N")
    pw = _ana("revision_suite_pairwise.csv")
    c14 = _ana("revision_cec2014_pairwise.csv")
    src_tab = {"measured": ref("tab:popisolation", home), "exploratory": ref("tab:popisolation", home),
               "in-sample": ref("tab:poprule", home), "E16": ref("tab:e16", home)}
    # printed status: the stage identifier E16 stays in the data, not in the main text (REV-10)
    status_txt = {"E16": "registered, exploratory"}
    lines = [r"\begin{table}[htbp]", r"\centering", f"\\caption{{{caption}}}",
             r"\label{tab:guide}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{rrllccll}", r"\toprule",
             r"$B/D$ & $D$ & Problems & Engine & $\delta$($6D$ vs $18D$) [95\% CI] & Favoured & "
             r"Status & Source \\", r"\midrule",
             r"$<300$ & -- & -- & -- & not measured & -- & untested & -- \\"]

    def row(r):
        dv = led(fname, f"{r['problems']} {r['engine']}", f"B/D={r['evals_per_variable']:.0f}",
                 r["delta"], "6D", "18D", r["source"], reversed_=bool(r["sign_reversed"]))
        eng = ("L-SHADE" if r["engine"] == "L-SHADE" else
               METHOD if "SEHHO-COBL-R" in r["engine"] else "gate-free engine")
        prob = r["problems"].replace("P1", "P1 pressure vessel").replace("P2", "P2 gear train")
        return (f"{fnum(round(r['evals_per_variable']))} & {r['dim']} & {esc(prob)} & {eng} & "
                f"${dv}$ ${cfmt(r['ci_lo'], r['ci_hi'])}$ & {r['favours']} & "
                f"{status_txt.get(r['status'], r['status'])} & "
                f"Table~{src_tab[r['status']]} \\\\")

    for r in g[g.evals_per_variable < 1700].to_dict("records"):
        lines.append(row(r))
    lines.append(r"1{,}700--3{,}750 & -- & -- & -- & not measured & -- & interpolated & -- \\")
    for r in g[(g.evals_per_variable >= 1700) & (g.evals_per_variable < 10_000)].to_dict("records"):
        lines.append(row(r))
    for r in g[g.evals_per_variable >= 10_000].to_dict("records"):
        lines.append(row(r))
        if r["evals_per_variable"] == 10_000 and r["problems"] == "CEC2017" and r["dim"] == 50 \
                and r["engine"] != "L-SHADE":
            q = ps.loc[120]
            dv = led(fname, "SEHHO-COBL fixed N", "B/D=10000", q["mean"], "N=120", "N=30",
                     "results/analysis/revision_popsize.csv (E6)")
            lines.append(f"10{{,}}000 & 30 & CEC2017 & SEHHO-COBL, $N{{=}}120$ vs $N{{=}}30$ & "
                         f"${dv}$ ${cfmt(q.pct_lo, q.pct_hi)}$ & $N{{=}}120$ & measured & "
                         f"Table~{ref('tab:popsize', home)} \\\\")
    for bd, txt, ok in LITERATURE:
        lines.append(f"{bd} & -- & CEC2014 & \\multicolumn{{3}}{{l}}{{{txt}}} & "
                     f"literature{'' if ok else ' [verify]'} & -- \\\\")
    # the caveat at 300 evaluations per variable: SEHHO-COBL's fixed N=30 against the
    # repaired configuration's 6D, both suites, D=50, 15,000 evaluations
    t17 = pw[(pw.table == "cec2017_tight") & (pw.dim == 50) & (pw.opponent == "SEHHO-COBL")].iloc[0]
    t14 = c14[(c14.dim == 50) & (c14.max_fes == 15_000) & (c14.opponent == "SEHHO-COBL")].iloc[0]
    led(fname, "note: SEHHO-COBL-R vs SEHHO-COBL", "CEC2017 D=50 15,000", t17["mean"], METHOD,
        "SEHHO-COBL", "results/analysis/revision_suite_pairwise.csv")
    led(fname, "note: SEHHO-COBL-R vs SEHHO-COBL", "CEC2014 D=50 15,000", t14["mean"], METHOD,
        "SEHHO-COBL", "results/analysis/revision_cec2014_pairwise.csv")
    low = g[(g.evals_per_variable >= 300) & (g.evals_per_variable <= 1700)]
    hi = g[(g.evals_per_variable == 10_000) & (g.engine == "L-SHADE")]
    lines += [r"\bottomrule", r"\end{tabular}}", r"\begin{flushleft}\footnotesize",
              CONV + r" Here A is $N_{\text{init}}=6D$ and B $18D$ (for the SEHHO-COBL row, "
              r"$N=120$ against $N=30$), so a negative value favours the smaller initial "
              r"population. $B/D$ is the evaluation budget per variable. Intervals are 95\% "
              r"percentile bootstrap intervals, over functions for the suites and over runs for "
              r"the design problems. Status: \emph{measured}, CEC2017, the diagnostic suite; "
              r"\emph{exploratory}, CEC2014 blocks run after the confirmatory analysis; "
              r"\emph{in-sample}, CEC2022, the suite the rule was selected on (the "
              r"gate-free engine before the composition step; LPSR-18D against LPSR-6D, sign "
              r"reversed from the analysis behind Table~" + ref("tab:poprule", home) + r"); "
              r"\emph{registered, exploratory}, the control on the five design problems "
              r"(Table~" + ref("tab:e16", home) + r"), registered before it ran and exploratory "
              r"by its registration; "
              r"\emph{interpolated}, a band between measurements, for which no recommendation "
              r"is made; \emph{literature}, the initial population rates that "
              r"\citet{tanabe2015tuning} obtained by tuning restart L-SHADE jointly with its other "
              r"parameters on CEC2014 F1--F16 at $D=2$, 10 and 20 (their Table~I), not results of "
              r"this study. The "
              r"rows come from different problems, dimensions and engines, and $B/D$ is "
              r"confounded with $D$ and with the problem class; this is not a controlled sweep. "
              f"From 300 to 1{{,}}700 evaluations per variable every measured row favours $6D$ "
              f"($\\delta$ from ${low.delta.min():+.2f}$ to ${low.delta.max():+.2f}$, $D$ from "
              f"{int(low.dim.min())} to {int(low.dim.max())}); at $10^4$ per variable L-SHADE's "
              f"$18D$ is better in {int((hi.favours == '18D').sum())} of {len(hi)} blocks "
              f"($\\delta$ from ${hi.delta.min():+.2f}$ to ${hi.delta.max():+.2f}$). Design-problem "
              r"qualifiers: RC06 compares violations only; on RC14 both arms end about 9\% "
              r"above the best-known value; on RC01 the feasibility difference is not "
              r"significant. At 300 per variable even $6D$ may be too large for the repaired "
              r"configuration: SEHHO-COBL with its fixed $N=30$ beats it at $D=50$ with "
              f"15{{,}}000 evaluations (${dfmt(t17['mean'], 2)}$ on CEC2017, ${dfmt(t14['mean'], 2)}$ "
              r"on CEC2014, " + METHOD + r" against SEHHO-COBL; the two differ in five respects). "
              + METHOD + r" fixes $N_{\text{init}}=6D$ at every budget and does not follow this "
              r"guide.",
              r"\end{flushleft}", r"\end{table}", ""]
    write(fname, "\n".join(lines), "tab:guide", home)


# ================================================================== S2.2, S2.4
def table_supp_singular(caption, fname, home="supp"):
    _LABELS[fname] = "tab:supp_singular"
    sv = _ana("revision_singular_values.csv")
    lines = [r"\begin{table}[htbp]", r"\centering", f"\\caption{{{caption}}}",
             r"\label{tab:supp_singular}", r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Suite & $D$ & Matrices & largest $=2$ & orthogonal & other & max over the rest \\",
             r"\midrule"]
    for (suite, dim), s in sv.groupby(["suite", "dim"]):
        k = s.kind.value_counts()
        rest = s[s.kind != "max 2"]
        lines.append(f"{suite} & {dim} & {len(s)} & {k.get('max 2', 0)} & {k.get('orthogonal', 0)} & "
                     f"{k.get('other', 0)} & {rest.sv_max.max():.4f} \\\\" if len(rest) else
                     f"{suite} & {dim} & {len(s)} & {k.get('max 2', 0)} & 0 & 0 & -- \\\\")
    tot = sv.groupby("suite").apply(lambda s: (int((s.kind == "max 2").sum()), len(s)), include_groups=False)
    lines += [r"\bottomrule", r"\end{tabular}", r"\begin{flushleft}\footnotesize",
              r"Every $D\times D$ block of the transformation-matrix files at the dimensions "
              r"used in the paper, read as \texttt{validate\_benchmarks.py} reads them "
              r"(\texttt{scripts/check\_cec\_data.py}); a composition function's file contributes "
              r"every block it contains. ``Orthogonal'': all singular values equal to 1. Over "
              r"each suite the largest singular value is exactly 2, and every entry is finite; "
              r"the matrices with largest singular value 2 number "
              + ", ".join(f"{a} of {b} ({s})" for s, (a, b) in tot.items()) + r".",
              r"\end{flushleft}", r"\end{table}", ""]
    write(fname, "\n".join(lines), "tab:supp_singular", home)


def table_supp_lineage(caption, fname, home="supp"):
    _LABELS[fname] = "tab:supp_lineage"
    lin = _ana("revision_cec2022_lineage.csv")
    lines = [r"\begin{table}[htbp]", r"\centering", f"\\caption{{{caption}}}",
             r"\label{tab:supp_lineage}", r"\begin{tabular}{llll}", r"\toprule",
             r"CEC2022 & shift vector & rotation matrices ($D=10,20$) & shuffle ($D=10,20$) \\",
             r"\midrule"]
    for fn in range(1, 13):
        s = lin[lin.cec2022_func == fn]

        def cell(kind):
            t = s[s.kind == kind]
            if t.empty:
                return "--"
            return "; ".join(f"{o} F{g}" for (o, g) in sorted({(o, int(g)) for o, g in zip(t.other_suite, t.other_func)}))
        lines.append(f"F{fn} & {cell('shift vector')} & {cell('rotation matrix')} & {cell('shuffle')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\begin{flushleft}\footnotesize",
              r"Input data of CEC2022 that are identical to CEC2017 or CEC2014 input data "
              r"(\texttt{scripts/check\_cec\_data.py}): a shift vector matches when all 100 "
              r"entries of its first row are equal; matrices and shuffles when every number in "
              r"the file is equal. All twelve CEC2022 shift vectors are CEC2017 shift vectors. "
              r"Whether two functions share a recipe (the same basic functions in the same "
              r"order and proportions) is a property of the C sources and is not tested here. "
              r"CEC2014 and CEC2017 share basic functions in their code but no shift vector or "
              r"rotation matrix at $D=30$ and 50.",
              r"\end{flushleft}", r"\end{table}", ""]
    write(fname, "\n".join(lines), "tab:supp_lineage", home)


# ================================================================== S2.3: port validation
def table_supp_validation(caption, fname, home="supp"):
    """S2.3: output-level validation of the jSO, L-SHADE and SHADE ports (REV-23).

    Reads the summary files that the validation scripts write with --summary-csv
    (they hold statistics only; the published reference results are not
    redistributed):
      results/analysis/validation_jso.csv           scripts/validate_jso.py
      results/analysis/validation_lshade_shade.csv  scripts/validate_lshade_shade.py
    Writes nothing if neither exists.
    """
    import importlib.util
    parts = []
    for f in ("validation_jso.csv", "validation_lshade_shade.csv"):
        p = os.path.join(ANA, f)
        if os.path.exists(p):
            parts.append(read_runs(p))
    if not parts:
        print("[S2.3] no validation summaries found: run validate_jso.py and "
              "validate_lshade_shade.py with --summary-csv (see the README)")
        return False
    _LABELS[fname] = "tab:supp_validation"
    v = pd.concat(parts, ignore_index=True)
    spec = importlib.util.spec_from_file_location(
        "vls", os.path.join(os.path.dirname(os.path.abspath(__file__)), "validate_lshade_shade.py"))
    vls = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vls)
    port = {"jSO": "jSO", "LSHADE": "L-SHADE", "SHADE": "SHADE"}
    refname = {"jSO": r"published jSO results (Brest et al.)",
               "LSHADE": r"CEC2014 entry, \texttt{lshade.cc} 1.0.0",
               "SHADE": r"SHADE 1.1 (other version)$^{a}$"}
    suite = {"jSO": "CEC2017", "LSHADE": "CEC2014", "SHADE": "CEC2014"}
    lines = [r"\begin{table}[htbp]", r"\centering", f"\\caption{{{caption}}}",
             r"\label{tab:supp_validation}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{lllcccccl}", r"\toprule",
             r"Port & Reference & Block & Runs & Comparable & Median & Geometric-mean & "
             r"Within & Rank-sum differences \\",
             r" & & & ours / ref. & functions & ratio & ratio & $2\times$ / $3\times$ & "
             r"(Holm), ours $\downarrow$ lower / $\uparrow$ higher \\", r"\midrule"]
    order = {"jSO": 0, "LSHADE": 1, "SHADE": 2}
    marks = []
    for r in sorted(v.to_dict("records"), key=lambda r: (order[r["algorithm"]], r["dim"])):
        gm = f"{r['gmean_ratio']:.3f}"
        w3 = r.get("gmean_ratio_within3", np.nan)
        if isinstance(w3, float) and np.isfinite(w3) and abs(w3 - r["gmean_ratio"]) > 5e-4:
            gm += f" ({w3:.3f})$^{{b}}$"
            marks.append(r)
        if r["algorithm"] == "jSO":
            rs = r"-- (summary statistics only)"
        else:
            items = str(r["rank_sum_different"]).split() if isinstance(r["rank_sum_different"], str) else []
            rs = (" ".join(f"{x[:-1]}$\\{'downarrow' if x.endswith('-') else 'uparrow'}$" for x in items)
                  if items else "none")
        lines.append(f"{port[r['algorithm']]} & {refname[r['algorithm']]} & "
                     f"{suite[r['algorithm']]}, $D={r['dim']}$ & {r['runs_ours']} / {r['runs_reference']} & "
                     f"{int(r['n_comparable'])} of {int(r['n_functions'])} & {r['median_ratio']:.3f} & {gm} & "
                     f"{100 * r['within2']:.0f}\\% / {100 * r['within3']:.0f}\\% & {rs} \\\\")
    dig = vls.REFERENCES
    bnote = ""
    if marks:
        bnote = (r" $^{b}$In parentheses, the geometric mean without the functions whose ratio "
                 r"lies outside $3\times$ (" + "; ".join(
                     f"{port[r['algorithm']]}, $D={r['dim']}$: "
                     + ", ".join(f for f in str(r['outside3']).split())
                     for r in marks) + r").")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\begin{flushleft}\footnotesize",
              r"Each port, run at the competition protocol ($10{,}000\times D$ evaluations), "
              r"against its authors' results. Ratios are our mean error divided by the "
              r"published mean error, over the functions on which both exceed $10^{-8}$ "
              r"(``comparable''); functions both sides solve are not counted. Rank-sum: "
              r"two-sided Mann--Whitney test per function on the runs, errors rounded to "
              r"$10^{-8}$, Holm-corrected over the functions of a block; it needs per-run "
              r"reference data, which exist for L-SHADE and SHADE but not for jSO. Sources, not "
              r"redistributed here (no licence is stated): jSO, \texttt{D30.rez} in Brest et "
              r"al.'s source archive (CEC2017 repository, sha256 \texttt{03e31fe6}\ldots); "
              r"L-SHADE, the per-run files of Tanabe and Fukunaga's CEC2014 competition entry "
              r"(CEC2014 repository, \texttt{CEC-2014-Results-PartA.zip}, folder "
              r"\texttt{L-SHADE}; sha256 of the 30 files of a dimension, concatenated, "
              f"\\texttt{{{dig['LSHADE']['digest'][30][:8]}}}\\ldots\\ and "
              f"\\texttt{{{dig['LSHADE']['digest'][50][:8]}}}\\ldots); "
              r"SHADE, the SHADE 1.1 runs of Tanabe's data release for the L-SHADE paper "
              r"(\texttt{Tanabe-CEC14-results.zip}, Internet Archive copy, folder "
              f"\\texttt{{SHADE11}}; \\texttt{{{dig['SHADE']['digest'][30][:8]}}}\\ldots\\ and "
              f"\\texttt{{{dig['SHADE']['digest'][50][:8]}}}\\ldots). "
              r"The published L-SHADE runs come from \texttt{lshade.cc} 1.0.0, which archives "
              r"the trial vector and re-accumulates a memory slot set to the terminal CR value; "
              r"our port archives the replaced parent and keeps the terminal value absorbing, "
              r"as the L-SHADE paper's Algorithm~1 specifies. "
              r"$^{a}$We found no published SHADE 1.0 results on CEC2014; SHADE 1.1 differs from "
              r"the SHADE 1.0 implemented here ($H=D$ against 100, $|\mathcal{A}|=2N$ against "
              r"$N$, $p=0.1$ against $p_i\sim U[2/N,0.2]$, a Lehmer $M_{CR}$ with a terminal "
              r"value against an arithmetic one), so its row is a plausibility check, not a "
              r"validation of the port." + bnote +
              r" Statistics from \texttt{scripts/validate\_jso.py} and "
              r"\texttt{scripts/validate\_lshade\_shade.py} (option \texttt{-{}-summary-csv}).",
              r"\end{flushleft}", r"\end{table}", ""]
    write(fname, "\n".join(lines), "tab:supp_validation", home)
    return True


# ================================================================== S5.6 per-function
def table_per_function(df, dim, caption, fname, label, fes_note="", home="supp"):
    _LABELS[fname] = f"tab:{label}"
    sub = df[df.dim == dim]
    algos = [a for a in ORDER if a in set(sub.algo)]
    funcs = sorted(sub.func.unique())
    lines = [r"\begin{table}[htbp]", r"\centering", f"\\caption{{{caption}}}",
             f"\\label{{tab:{label}}}",
             r"\adjustbox{max width=\textwidth,max totalheight=0.85\textheight}{%",
             r"\begin{tabular}{cl" + "c" * len(algos) + "}", r"\toprule",
             "Func. & Metric & " + " & ".join(esc(a) for a in algos) + r" \\", r"\midrule"]
    for f in funcs:
        means = {a: sub[(sub.func == f) & (sub.algo == a)]["error"].mean() for a in algos}
        stds = {a: sub[(sub.func == f) & (sub.algo == a)]["error"].std(ddof=1) for a in algos}
        best = {algos[i] for i in tied_best([means[a] for a in algos])}
        lines.append(f"F{f} & Mean & " + " & ".join(
            (f"\\textbf{{{sci(means[a])}}}" if a in best else sci(means[a])) for a in algos) + r" \\")
        lines.append(" & Std & " + " & ".join(
            (f"$\\pm$\\textbf{{{sci(stds[a])}}}" if a in best else f"$\\pm${sci(stds[a])}")
            for a in algos) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\begin{flushleft}\footnotesize",
              f"Error values $f(\\mathbf{{x}})-f^{{*}}$ over 30 independent runs; errors below "
              f"$10^{{-8}}$ are recorded as 0. {fes_note} Best mean per function in bold; "
              f"when several algorithms share the lowest mean exactly, every one of them is bold.",
              r"\end{flushleft}", r"\end{table}", ""]
    write(fname, "\n".join(lines), f"tab:{label}", home)


# ================================================================== main
def main():
    files = {k: os.path.join(RES, v) for k, v in (
        ("E1", "E1_cec2022_paper_budget.csv"), ("E2", "E2_cec2022_competition.csv"),
        ("E3", "E3_cec2017.csv"), ("E4", "E4_ablation_cec2022.csv"),
        ("E4b", "E4b_ablation_cec2017_30D.csv"), ("E5", "E5_engineering.csv"))}
    e9 = os.path.join(RES, "E9_newmethod_vs_baselines.csv")
    e12 = os.path.join(RES, "E12_cec2017_tight.csv")
    e10 = os.path.join(RES, "E10_newmethod_engineering.csv")

    # ---------------- main text
    table_gate_inference("The gate $\\times$ L\\'evy factorial on CEC2017 at $D=30$: interval "
                         "estimates, tests on both scales and equivalence verdicts at three "
                         "margins.", "tab_gateinference.tex")
    table_popsize(read_runs(os.path.join(RES, "E6_popsize_cec2017_30D.csv")),
                  "The population sweep in SEHHO-COBL on CEC2017 at $D=30$ with "
                  "$3\\times10^{5}$ evaluations.", "tab_popsize.tex")
    table_pop_isolation("The one-constant population control: rows 1 and 2 change only "
                        "$N_{\\text{init}}$ inside one algorithm.", "tab_popisolation.tex")
    table_cec2014_primary("Pre-registered evaluation on CEC2014.", "tab_cec2014primary.tex")
    eng = pd.concat([read_runs(files["E5"]), read_runs(e10)], ignore_index=True)
    eng["algo"] = eng.algo.replace(DISPLAY)
    table_engineering(eng, "Seven algorithms on five constrained design problems under Deb's "
                      "feasibility rule, with a classical reference for each problem.",
                      "tab_engineering.tex")
    table_e16("The one-constant population control on the five design problems: L-SHADE "
              "with $N_{\\text{init}}=6D$ against $18D$.", "tab_e16.tex")
    e5b = read_runs(os.path.join(RES, "E5b_engineering_budget.csv"))
    e5b["algo"] = e5b.algo.replace(DISPLAY)
    table_rc01rc06(e5b, read_runs(os.path.join(RES, "E5c_solver_baseline.csv")), eng,
                   "RC01 and RC06: more evaluations, and methods designed for explicit "
                   "constraints.", "tab_engbudget.tex")
    table_guide("A budget-indexed guide to the initial population: $\\delta$($6D$ vs $18D$) "
                "against evaluations per variable.", "tab_guide.tex")

    # ---------------- supplement
    table_supp_singular("Largest singular values of the CEC transformation matrices.",
                        "tab_supp_singular.tex")
    table_supp_validation("Output-level validation of the SHADE-family ports.",
                          "tab_supp_validation.tex")
    table_supp_lineage("Input data that CEC2022 shares with CEC2017 and CEC2014.",
                       "tab_supp_lineage.tex")
    table_supp_precision("Precision of the primary endpoint at 30 runs per cell.",
                         "tab_supp_precision.tex")
    p4c = os.path.join(RES, "E4c_ablation_cec2017_30D_15k.csv")
    blocks = [("CEC2017 $D{=}30$, 15{,}000 FEs", read_runs(p4c)),
              ("CEC2017 $D{=}30$, 300{,}000 FEs", read_runs(files["E4b"]))]
    table_ablation(pd.concat([b for _, b in blocks]), "SEHHO:Full",
                   "Ablation of SEHHO-COBL at two evaluation budgets on CEC2017 at $D=30$: "
                   "phase-gate replacements, the fixed-$p$ variant and single-module removals, "
                   "ordered by rank at 300{,}000 evaluations.", "tab_ablation.tex", "ablation", blocks)
    table_gate_factorial("Ranks of the gate $\\times$ L\\'evy factorial, and the tests on "
                         "per-function mean errors.", "tab_gatefactorial.tex")
    table_supp_gate_intervals("Every interval for the contrasts of the gate $\\times$ L\\'evy "
                              "factorial: percentile, BCa and Student-$t$, plain and "
                              "Bonferroni-adjusted, with the margin verdicts.",
                              "tab_supp_gate_intervals.tex")
    d4 = read_runs(files["E4"])
    # the note this table needs (REV-33): non-separation, and the sign of the gate
    # variants at D = 20, read from the analysis rather than asserted
    gate_vars = ["SEHHO:AlwaysPbest", "SEHHO:AlwaysPbest+NoLevy", "SEHHO:AlwaysPbest+FixedP",
                 "SEHHO:NoEscapeEnergy", "SEHHO:DeterministicPhase", "SEHHO:AlwaysBest"]
    outs = [paired_vs_full(d4[d4.dim == dm], "SEHHO:Full")[1] for dm in (10, 20)]
    o20 = outs[1]
    pos = [v.replace("SEHHO:", "") for v in gate_vars
           if -float(o20[o20.variant == v].cliffs_delta.iloc[0]) > 0]
    sig = sorted({v.replace("SEHHO:", "") for o in outs for v in o[o.significant_holm].variant})
    table_ablation(d4, "SEHHO:Full", "Ablation of SEHHO-COBL on CEC2022 at 15{,}000 evaluations.",
                   "tab_supp_ablation_cec2022.tex", "supp_ablation_cec2022",
                   [("CEC2022, $D=10$, 15{,}000 FEs", d4[d4.dim == 10]),
                    ("CEC2022, $D=20$, 15{,}000 FEs", d4[d4.dim == 20])],
                   note_extra=(r" On CEC2022 at 15{,}000 evaluations the variants are not "
                               r"separated from the full method: no Holm-corrected test is "
                               r"below 0.05" + ("" if not sig else " except for " + ", ".join(sig)) +
                               r". At $D=20$ the point estimates of the gate variants "
                               + ", ".join(pos) + r" are positive, i.e.\ favour keeping the "
                               r"gate. CEC2022 is built from CEC2017 data and CEC2014 and "
                               r"CEC2017 recipes (Table~" + ref("tab:supp_lineage", "supp") + r")."))
    table_poprule("Choosing the population rule on CEC2022, the selection suite.", "tab_poprule.tex")
    table_composition("Does each remaining part earn its place? Changes to the gate-free "
                      "configuration on CEC2022.", "tab_composition.tex")
    table_supp_composition_intervals("Every interval for the composition contrasts on CEC2022, "
                                     "and the decisions they would give.",
                                     "tab_supp_composition_intervals.tex")
    table_supp_e15("Sensitivity of the confirmatory result to the composition-rule amendment "
                   "(stage E15).", "tab_supp_e15.tex")
    table_cec2014("The full CEC2014 comparison: ranks, $\\delta$ and pairwise tests against " +
                  METHOD + ".", "tab_cec2014.tex")
    table_supp_hho("SEHHO-COBL and " + METHOD + " against HHO in every block of the three "
                   "suites.", "tab_supp_hho.tex")
    d17 = load_comparison(files["E3"], e9, "E3_cec2017")
    table_ranks(d17, "cec2017", "CEC2017 at $D=30$, 50 and 100 under the competition budget "
                "($10{,}000\\times D$ evaluations, 30 runs). Diagnostic suite: the decisions to "
                "change the gate and the population rule were taken on it, so this comparison "
                "is an optimistic estimate on the suite that motivated the changes.",
                "tab_cec2017.tex", "cec2017_competition")
    d12 = read_runs(e12)
    d12["algo"] = d12.algo.replace(DISPLAY)
    table_ranks(d12, "cec2017_tight", "CEC2017 at $D=30$ and 50 under the tight budget of "
                "15{,}000 evaluations (30 runs). Diagnostic suite.", "tab_cec2017_tight.tex",
                "cec2017_tight")
    d1 = load_comparison(files["E1"], e9, "E1_cec2022_paper_budget")
    table_ranks(d1, "cec2022_matched", "CEC2022 at $D=10$ and 20 under the matched budget of "
                "15{,}000 evaluations (30 runs). Selection suite: the population rule and the "
                "composition of " + METHOD + " were chosen on it, so these results are "
                "in-sample.", "tab_cec2022_matched.tex", "cec2022_matched")
    d2 = load_comparison(files["E2"], e9, "E2_cec2022_competition")
    table_ranks(d2, "cec2022_competition", "CEC2022 at $D=10$ and 20 under the competition "
                "budget ($2\\times10^{5}$ and $10^{6}$ evaluations, 30 runs). Selection suite, "
                "in-sample.", "tab_cec2022_competition.tex", "cec2022_competition")
    table_supp_popisolation("The one-constant population control, every row with every "
                            "interval.", "tab_supp_popisolation.tex")
    table_supp_classical_refs("Classical reference designs for the five design problems.",
                              "tab_supp_classical_refs.tex")
    table_supp_e16("Stage E16 per problem: feasibility, best and median designs, and success "
                   "counts at four thresholds.", "tab_supp_e16.tex")
    # per-function tables; every title states suite, dimension and budget
    for key, stem, note, tag, suite, word in (
            ("E1", "supp_cec2022_matched", "Selection suite, matched budget.",
             "E1_cec2022_paper_budget", "CEC2022", "matched budget"),
            ("E2", "supp_cec2022_competition", "Selection suite, competition budget.",
             "E2_cec2022_competition", "CEC2022", "competition budget"),
            ("E3", "supp_cec2017", "Diagnostic suite, competition budget of $10{,}000\\times D$.",
             "E3_cec2017", "CEC2017", "competition budget")):
        d = load_comparison(files[key], e9, tag)
        for dim in sorted(d.dim.unique()):
            fes = int(d[d.dim == dim].max_fes.iloc[0])
            table_per_function(d, dim, f"Per-function results, {suite} at $D={dim}$, "
                               f"{fnum(fes)} evaluations ({word}).",
                               f"tab_{stem}_D{dim}.tex", f"{stem}_D{dim}", note)
    for dim in sorted(d12.dim.unique()):
        table_per_function(d12, dim, f"Per-function results, CEC2017 at $D={dim}$, 15{{,}}000 "
                           f"evaluations (tight budget).", f"tab_supp_cec2017_tight_D{dim}.tex",
                           f"supp_cec2017_tight_D{dim}", "Diagnostic suite, tight budget.")
    d13 = read_runs(os.path.join(RES, "E13_cec2014.csv"))
    d13["algo"] = d13.algo.replace(DISPLAY)
    for dim in sorted(d13.dim.unique()):
        for fes, tag, word in ((10_000 * dim, "competition", "competition budget"),
                               (15_000, "tight", "tight budget")):
            sub = d13[(d13.dim == dim) & (d13.max_fes == fes)]
            table_per_function(sub, dim, f"Per-function results, CEC2014 at $D={dim}$, "
                               f"{fnum(fes)} evaluations ({word}).",
                               f"tab_supp_cec2014_{tag}_D{dim}.tex", f"supp_cec2014_{tag}_D{dim}",
                               "Confirmatory suite: no design decision was taken on its results.")

    # ---------------- ledger and index
    led_df = pd.DataFrame(LEDGER)
    led_df.to_csv(os.path.join(ANA, "revision_delta_ledger.csv"), index=False)
    print(f"wrote {os.path.join(ANA, 'revision_delta_ledger.csv')} ({len(led_df)} deltas)")
    seen, idx = set(), []
    for f, lab, home in WRITTEN:
        if f in seen:
            continue
        seen.add(f)
        idx.append(dict(file=f, label=lab, document=home))
    pd.DataFrame(idx).to_csv(os.path.join(ANA, "revision_table_index.csv"), index=False)
    print(f"wrote {os.path.join(ANA, 'revision_table_index.csv')} ({len(idx)} tables)")


if __name__ == "__main__":
    main()
