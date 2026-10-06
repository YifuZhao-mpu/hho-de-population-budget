"""Turn raw per-run results into the paper's statistical tables.

Every comparison family is Holm-corrected and every pairwise test is reported
with an effect size (Cliff's delta, with the Vargha-Delaney A12 alongside).
Outputs land in results/analysis/ as CSV plus ready-to-include LaTeX.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sehho.runfiles import read_runs  # noqa: E402  (pandas.read_csv; SEHHO_ERROR_DECIMALS, Section 2.6)
from sehho.stats import (friedman, friedman_holm_vs_control, ranksum_family,
                         summarise_outcomes, cliffs_delta, cliffs_magnitude,
                         vargha_delaney_a12, holm)
from scipy import stats as sps

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "analysis")
os.makedirs(OUT, exist_ok=True)
CONTROL = "SEHHO-COBL"


def _fmt(v, prec=2):
    if not np.isfinite(v):
        return "--"
    if v == 0:
        return "0.00E+00"
    return f"{v:.{prec}E}".replace("E-0", "E-").replace("E+0", "E+")


def compare_block(df, control=CONTROL, value="error", label=""):
    """Full statistical battery for one (suite, dim) block."""
    algos = sorted(df["algo"].unique(), key=lambda a: (a != control, a))
    funcs = sorted(df["func"].unique())
    # mean matrix for Friedman (problems x algorithms)
    M = np.array([[df[(df.func == f) & (df.algo == a)][value].mean() for a in algos]
                  for f in funcs])
    fr, post = friedman_holm_vs_control(M, algos, control)

    per_run = {a: {f: df[(df.func == f) & (df.algo == a)][value].to_numpy()
                   for f in funcs} for a in algos}
    rows = ranksum_family(per_run, control)
    outcomes = summarise_outcomes(rows)

    # per-function summary table
    summ = []
    for f in funcs:
        rec = {"func": f}
        for a in algos:
            s = df[(df.func == f) & (df.algo == a)][value]
            rec[f"{a}_mean"] = s.mean()
            rec[f"{a}_std"] = s.std(ddof=1)
        best = min(algos, key=lambda a: rec[f"{a}_mean"])
        rec["best"] = best
        summ.append(rec)
    summ = pd.DataFrame(summ)

    ranks = pd.DataFrame({"algorithm": algos, "avg_rank": fr["avg_ranks"]})
    ranks["rank1_count"] = [int((summ["best"] == a).sum()) for a in algos]
    ranks = ranks.sort_values("avg_rank").reset_index(drop=True)
    ranks["position"] = np.arange(1, len(ranks) + 1)

    post_df = pd.DataFrame(post)
    if len(post_df):
        post_df["win_tie_loss"] = post_df["algorithm"].map(
            lambda a: f"{outcomes[a]['win']}/{outcomes[a]['tie']}/{outcomes[a]['loss']}")
        # aggregate effect size: mean of the per-function Cliff's delta values,
        # which are already computed one function at a time in ranksum_family
        eff = {}
        for a in outcomes:
            ds = [r["cliffs_delta"] for r in rows if r["algorithm"] == a]
            eff[a] = float(np.mean(ds))
        post_df["mean_cliffs_delta"] = post_df["algorithm"].map(eff)
        post_df["magnitude"] = post_df["mean_cliffs_delta"].map(cliffs_magnitude)

    meta = dict(label=label, n_problems=fr["n"], n_algorithms=fr["k"],
                friedman_chi2=fr["chi2"], iman_davenport_F=fr["iman_davenport_F"],
                friedman_p=fr["p"])
    return dict(meta=meta, summary=summ, ranks=ranks, posthoc=post_df,
                pairwise=pd.DataFrame(rows))


def paired_vs_full(df, control, value="error", label=""):
    """Ablation analysis: paired signed-rank of `control` against each variant.

    Pairs are the per-function mean errors, matching the ablation protocol used
    in the manuscript; the family of variant comparisons is Holm-corrected and
    reported with an effect size.
    """
    algos = sorted(df["algo"].unique())
    funcs = sorted(df["func"].unique())
    M = np.array([[df[(df.func == f) & (df.algo == a)][value].mean() for a in algos]
                  for f in funcs])
    fr = friedman(M)
    ci = algos.index(control)
    rows, praw = [], []
    for j, a in enumerate(algos):
        if a == control:
            continue
        x, y = M[:, ci], M[:, j]
        diff = y - x
        nz = np.abs(diff) > 0
        if nz.sum() == 0:
            p, stat = 1.0, np.nan
        else:
            stat, p = sps.wilcoxon(x[nz], y[nz], alternative="two-sided")
        # Effect size must be computed per function and then averaged: error
        # scales differ by many orders of magnitude across a suite, so pooling
        # all runs together would be dominated by whichever functions happen to
        # have the largest values and would understate real per-function effects.
        ds, a12s = [], []
        for f in funcs:
            xf = df[(df.func == f) & (df.algo == control)][value].to_numpy()
            yf = df[(df.func == f) & (df.algo == a)][value].to_numpy()
            ds.append(cliffs_delta(xf, yf))
            a12s.append(vargha_delaney_a12(xf, yf))
        d = float(np.mean(ds))
        rows.append(dict(variant=a, avg_rank=fr["avg_ranks"][j],
                         full_rank=fr["avg_ranks"][ci],
                         wins=int((x < y).sum()), losses=int((x > y).sum()),
                         ties=int((x == y).sum()), p=float(p),
                         cliffs_delta=d, magnitude=cliffs_magnitude(d),
                         a12=float(np.mean(a12s))))
        praw.append(p)
    for r, pa in zip(rows, holm(praw)):
        r["p_holm"] = float(pa)
        r["significant_holm"] = bool(pa < 0.05)
    out = pd.DataFrame(rows).sort_values("avg_rank").reset_index(drop=True)
    out.insert(0, "block", label)
    return fr, out


# ---------------------------------------------------------------- reports
def write_block(res, stem):
    res["summary"].to_csv(os.path.join(OUT, f"{stem}_per_function.csv"), index=False)
    res["ranks"].to_csv(os.path.join(OUT, f"{stem}_ranks.csv"), index=False)
    res["posthoc"].to_csv(os.path.join(OUT, f"{stem}_posthoc_holm.csv"), index=False)
    res["pairwise"].to_csv(os.path.join(OUT, f"{stem}_pairwise_tests.csv"), index=False)
    m = res["meta"]
    lines = [f"### {m['label']}",
             f"Friedman over {m['n_problems']} functions x {m['n_algorithms']} algorithms: "
             f"chi2={m['friedman_chi2']:.2f}, Iman-Davenport F={m['iman_davenport_F']:.2f}, "
             f"p={m['friedman_p']:.3e}", "",
             res["ranks"].to_string(index=False), ""]
    if len(res["posthoc"]):
        p = res["posthoc"][["algorithm", "avg_rank", "z", "p", "p_holm",
                            "significant_holm", "win_tie_loss",
                            "mean_cliffs_delta", "magnitude"]]
        lines += ["Post-hoc vs " + CONTROL + " (Holm-corrected) and per-function "
                  "rank-sum win/tie/loss (Holm-corrected across the whole family):",
                  p.to_string(index=False), ""]
    return "\n".join(lines)


def main():
    report = ["# SEHHO-COBL revision: statistical analysis",
              "All p-values are Holm step-down corrected within their family.",
              "Effect size is Cliff's delta of the control against the competitor "
              "(negative favours " + CONTROL + "); A12 is the probability the "
              "control produces the smaller error.", ""]

    # -------- E1 / E2 / E3 : main comparisons --------
    for fname, title in [("E1_cec2022_paper_budget.csv",
                          "CEC2022, paper budget (15,000 FEs), strictly FE-matched"),
                         ("E2_cec2022_competition.csv",
                          "CEC2022, competition budget"),
                         ("E3_cec2017.csv", "CEC2017, competition budget (10,000 x D)")]:
        path = os.path.join(RES, fname)
        if not os.path.exists(path):
            continue
        df = read_runs(path)
        report.append(f"\n## {title}\n")
        for d in sorted(df["dim"].unique()):
            sub = df[df.dim == d]
            res = compare_block(sub, label=f"{title} - D={d}")
            stem = fname.replace(".csv", "") + f"_D{d}"
            report.append(write_block(res, stem))

    # -------- E4 : ablations --------
    for fname, title in [("E4_ablation_cec2022.csv", "Ablations on CEC2022 (15,000 FEs)"),
                         ("E4b_ablation_cec2017_30D.csv",
                          "Ablations on CEC2017 D=30 (300,000 FEs)")]:
        path = os.path.join(RES, fname)
        if not os.path.exists(path):
            continue
        df = read_runs(path)
        report.append(f"\n## {title}\n")
        allout = []
        for d in sorted(df["dim"].unique()):
            sub = df[df.dim == d]
            fr, out = paired_vs_full(sub, "SEHHO:Full", label=f"D={d}")
            allout.append(out)
            report.append(f"### D={d}  (Friedman chi2={fr['chi2']:.2f}, "
                          f"Iman-Davenport F={fr['iman_davenport_F']:.2f}, p={fr['p']:.3e})")
            cols = ["variant", "avg_rank", "wins", "losses", "p", "p_holm",
                    "significant_holm", "cliffs_delta", "magnitude", "a12"]
            report.append(out[cols].to_string(index=False))
            report.append(f"(full-method average rank = {out['full_rank'].iloc[0]:.2f})\n")
        pd.concat(allout).to_csv(
            os.path.join(OUT, fname.replace(".csv", "_paired_holm.csv")), index=False)

    # -------- E5 : engineering --------
    path = os.path.join(RES, "E5_engineering.csv")
    if os.path.exists(path):
        df = read_runs(path)
        report.append("\n## Constrained engineering problems (feasibility rule)\n")
        recs = []
        for prob in sorted(df["problem"].unique()):
            for a in sorted(df["algo"].unique()):
                s = df[(df.problem == prob) & (df.algo == a)]
                feas = s[s.feasible]
                recs.append(dict(
                    problem=prob, algorithm=a, runs=len(s),
                    feasible_runs=int(s.feasible.sum()),
                    feasibility_rate=100.0 * s.feasible.mean(),
                    best_feasible=feas.objective.min() if len(feas) else np.nan,
                    mean_feasible=feas.objective.mean() if len(feas) else np.nan,
                    std_feasible=feas.objective.std(ddof=1) if len(feas) > 1 else np.nan,
                    mean_total_violation=s.total_violation.mean(),
                    max_total_violation=s.total_violation.max(),
                    mean_max_violation=s.max_violation.mean()))
        eng = pd.DataFrame(recs)
        eng.to_csv(os.path.join(OUT, "E5_engineering_feasibility.csv"), index=False)
        report.append(eng.to_string(index=False))
        report.append("")
        for prob in sorted(df["problem"].unique()):
            sub = eng[eng.problem == prob]
            if sub.feasibility_rate.max() == 0:
                report.append(f"**{prob}: no algorithm produced a single feasible design "
                              f"in any run at this budget.**")

    text = "\n".join(report)
    with open(os.path.join(OUT, "ANALYSIS.md"), "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
