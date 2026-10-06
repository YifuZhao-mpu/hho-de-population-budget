"""Publication figures for the revised manuscript and its supplement.

Writes vector PDFs into code_release/results/figures/; copying them into the
manuscript's figure directory is a separate step.

Revision conventions (Stage 4 plan, REV-19, REV-42):

* Greyscale-safe by construction: identity is carried by line style, marker
  shape, fill and hatching, never by colour alone; series are labelled
  directly where there are few of them.
* Every figure is drawn at the width at which it is printed, with no text
  smaller than 8 pt at that width.  The widths are those of the staged
  manuscript's \\includegraphics (main text and supplement share the 16 cm =
  6.3 in text width); PRINT_WIDTH below holds them.  Including a figure
  narrower than its PRINT_WIDTH shrinks its text below 8 pt: if an include
  width changes, change PRINT_WIDTH and re-run this script.
* The configuration obtained by dismantling SEHHO-COBL is "GF-Method" in the
  raw CSVs (its seeds hash that string) and SEHHO-COBL-R in every figure;
  L-SHADE is "LSHADE" in the data for the same reason.

Run:  python3 scripts/make_figures.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sehho.runfiles import read_runs  # noqa: E402  (pandas.read_csv; SEHHO_ERROR_DECIMALS, Section 2.6)
from sehho.stats import friedman  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results")
ANA = os.path.join(RES, "analysis")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

FULL_W = 6.3          # inches: the 16 cm text width of the manuscript
# print width of each figure = its \\includegraphics width in the staged manuscript
PRINT_WIDTH = {"fig_phase_gate.pdf": 0.72 * FULL_W,      # main.tex: 0.72\\textwidth
               "fig_ablation_ranks.pdf": FULL_W,          # main.tex: \\textwidth
               "fig_budget_guide.pdf": 0.9 * FULL_W,      # main.tex: 0.9\\textwidth
               "fig_scalability.pdf": FULL_W, "fig_feasibility.pdf": FULL_W}
INK, MID, LIGHT, FAINT = "#111111", "#555555", "#9a9a9a", "#d9d9d9"
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.5,
    "axes.edgecolor": "#444444", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": "#333333", "ytick.color": "#333333",
    "legend.frameon": False, "savefig.bbox": None,
    "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 150,
})

DISPLAY = {"GF-Method": "SEHHO-COBL-R", "LSHADE": "L-SHADE"}
ALGO_ORDER = ["SEHHO-COBL-R", "SEHHO-COBL", "jSO", "L-SHADE", "SHADE", "DE", "HHO"]
# line style + marker per algorithm: distinguishable in greyscale
ALGO_STYLE = {
    "SEHHO-COBL-R": dict(color=INK, ls="-", lw=1.6, marker="s", mfc=INK),
    "SEHHO-COBL": dict(color=INK, ls="-", lw=1.0, marker="o", mfc="white"),
    "jSO": dict(color=MID, ls="--", lw=1.1, marker="^", mfc=MID),
    "L-SHADE": dict(color=MID, ls="-.", lw=1.1, marker="v", mfc="white"),
    "SHADE": dict(color=LIGHT, ls=":", lw=1.4, marker="D", mfc=LIGHT),
    "DE": dict(color=LIGHT, ls="--", lw=1.0, marker="x", mfc=LIGHT),
    "HHO": dict(color=LIGHT, ls="-", lw=1.0, marker="+", mfc=LIGHT),
}
HATCH = {"SEHHO-COBL-R": "", "SEHHO-COBL": "////", "jSO": "", "L-SHADE": "\\\\\\\\",
         "SHADE": "....", "DE": "xxxx", "HHO": ""}
BAR_FACE = {"SEHHO-COBL-R": INK, "SEHHO-COBL": "white", "jSO": MID, "L-SHADE": "white",
            "SHADE": "white", "DE": "white", "HHO": FAINT}


def _save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, metadata={"CreationDate": None})   # no timestamp: byte-reproducible PDFs
    plt.close(fig)
    print("wrote", p)


# ---------------------------------------------------------------- Figure 1(b)
def fig_phase_gate():
    """q(t) of the escape-energy gate and the four substitute gates of the ablation."""
    T = 500
    t = np.arange(1, T + 1) / T
    E1 = 2 * (1 - t)
    q = np.where(E1 > 1, 1 - 1 / np.maximum(E1, 1e-12), 0.0)
    fig, ax = plt.subplots(figsize=(PRINT_WIDTH["fig_phase_gate.pdf"], 2.7), layout="constrained")
    ax.axvspan(0.5, 1.0, color="#f2f2f2", lw=0, zorder=0)
    ax.plot(t, np.ones_like(t), color=MID, ls=(0, (1, 1.5)), lw=1.3)
    ax.plot(t, (t < 0.5).astype(float), color=MID, ls="--", lw=1.1, drawstyle="steps-post")
    ax.plot(t, np.full_like(t, q.mean()), color=LIGHT, ls="-.", lw=1.2)
    ax.plot(t, np.zeros_like(t), color=MID, ls=(0, (4, 1.5, 1, 1.5, 1, 1.5)), lw=1.1)
    ax.plot(t, q, color=INK, lw=2.0)
    # direct labels instead of a legend
    ax.text(0.98, 1.03, r"always pbest, $q\equiv1$", ha="right", va="bottom", fontsize=8)
    ax.text(0.02, 0.86, r"split $\mathbb{1}[t<T/2]$", ha="left", va="top", fontsize=8, color=MID)
    ax.text(0.98, q.mean() + 0.03, fr"constant, $\bar q={q.mean():.3f}$", ha="right",
            va="bottom", fontsize=8, color="#555555")
    ax.text(0.98, -0.035, r"always best, $q\equiv0$", ha="right", va="top", fontsize=8,
            color=MID)
    ax.annotate("escape energy\n(SEHHO-COBL)", xy=(0.30, q[149]), xytext=(0.22, 0.60),
                fontsize=8, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.6))
    ax.text(0.75, 0.62, "$q(t)=0$ for\n$t\\geq T/2$", ha="center", va="center", fontsize=8,
            color=MID)
    ax.set_xlabel("normalised iteration $t/T$")
    ax.set_ylabel("P(exploration phase)")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.14, 1.16)
    ax.set_yticks([0, 0.5, 1])
    _save(fig, "fig_phase_gate.pdf")


# ---------------------------------------------------------------- Figure 2
GATE_VARIANTS = ("NoEscapeEnergy", "AlwaysPbest", "AlwaysBest", "DeterministicPhase",
                 "AlwaysPbest+FixedP", "AlwaysPbest+NoLevy")


def _variant_kind(n):
    if n == "Full":
        return "full"
    if n == "FixedP":
        return "fixedp"
    return "gate" if n in GATE_VARIANTS else "module"


KIND_STYLE = {"full": dict(facecolor=INK, edgecolor=INK, hatch=""),
              "gate": dict(facecolor=MID, edgecolor=MID, hatch=""),
              "fixedp": dict(facecolor="white", edgecolor=INK, hatch="////"),
              "module": dict(facecolor="white", edgecolor=INK, hatch="")}
KIND_LABEL = {"full": "full (SEHHO-COBL)", "gate": "phase-gate variant",
              "fixedp": "$p$ annealing removed", "module": "module removed"}


def fig_ablation_ranks():
    specs = [("E4c_ablation_cec2017_30D_15k.csv", "15,000 evaluations"),
             ("E4b_ablation_cec2017_30D.csv", "300,000 evaluations")]
    panels = []
    for fname, title in specs:
        df = read_runs(os.path.join(RES, fname))
        df = df[df.dim == 30]
        algos = sorted(df.algo.unique())
        funcs = sorted(df.func.unique())
        M = np.array([[df[(df.func == f) & (df.algo == a)]["error"].mean() for a in algos]
                      for f in funcs])
        panels.append((title, [a.replace("SEHHO:", "") for a in algos], friedman(M)["avg_ranks"]))
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 3.5), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.04, wspace=0.06)
    for ax, (title, names, ranks) in zip(axes, panels):
        order = np.argsort(ranks)
        names = [names[i] for i in order]
        ranks = ranks[order]
        y = np.arange(len(names))
        for yi, n, r in zip(y, names, ranks):
            st = KIND_STYLE[_variant_kind(n)]
            ax.barh(yi, r, height=0.68, linewidth=0.7, **st)
            ax.text(r + 0.12, yi, f"{r:.2f}", va="center", fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlim(0, 12.8)
        ax.set_xticks([0, 4, 8, 12])
        ax.grid(axis="y", visible=False)
        ax.set_title(title)
    fig.supxlabel("Friedman average rank over the twelve variants (lower is better)", fontsize=8)
    handles = [Patch(linewidth=0.7, label=KIND_LABEL[k], **KIND_STYLE[k])
               for k in ("full", "gate", "fixedp", "module")]
    fig.legend(handles=handles, loc="outside upper center", ncol=4, handlelength=1.4,
               columnspacing=1.0, handletextpad=0.4)
    _save(fig, "fig_ablation_ranks.pdf")


# ---------------------------------------------------------------- Figure 3
SOURCE_STYLE = {
    "CEC2017 L-SHADE": dict(marker="o", mfc=INK, mec=INK, ms=5.5,
                            label="L-SHADE, CEC2017 (diagnostic)"),
    "CEC2014 L-SHADE (exploratory)": dict(marker="o", mfc="white", mec=INK, ms=5.5,
                                          label="L-SHADE, CEC2014 (exploratory)"),
    "repaired engine CEC2017": dict(marker="s", mfc=MID, mec=MID, ms=5,
                                    label="SEHHO-COBL-R engine, CEC2017"),
    "repaired engine CEC2014": dict(marker="s", mfc="white", mec=MID, ms=5,
                                    label="SEHHO-COBL-R engine, CEC2014 (exploratory)"),
    "CEC2022 selection data (in-sample)": dict(marker="D", mfc="white", mec=LIGHT, ms=4.8,
                                               label="gate-free engine, CEC2022 (selection)"),
    "E16 design problems": dict(marker="^", mfc=INK, mec=INK, ms=6,
                                label="L-SHADE, design problems (exploratory)"),
}


def _source_key(r):
    if r["source_marker"] == "repaired engine":
        return "repaired engine " + ("CEC2017" if r["problems"] == "CEC2017" else "CEC2014")
    return r["source_marker"]


def fig_budget_guide():
    """delta(6D vs 18D) against evaluations per variable, from revision_budget_guide.csv."""
    g = read_runs(os.path.join(ANA, "revision_budget_guide.csv"))
    g["key"] = [_source_key(r) for r in g.to_dict("records")]
    # spread points that share an abscissa, on the log scale
    order = list(SOURCE_STYLE)
    g = g.sort_values(["evals_per_variable", "key", "dim"]).reset_index(drop=True)
    g["x"] = g.evals_per_variable.astype(float)
    for bd, idx in g.groupby("evals_per_variable").groups.items():
        k = len(idx)
        offs = (np.arange(k) - (k - 1) / 2) * 0.028
        if bd <= 300:            # never draw a measured point inside the untested band
            offs = offs - offs.min() + 0.006
        g.loc[idx, "x"] = bd * 10 ** offs
    fig, ax = plt.subplots(figsize=(PRINT_WIDTH["fig_budget_guide.pdf"], 4.1), layout="constrained")
    ax.set_xscale("log")
    ax.set_xlim(180, 75_000)
    ax.set_ylim(-1.08, 1.08)
    ax.axvspan(180, 300, facecolor="#efefef", edgecolor="#bdbdbd", hatch="////", lw=0, zorder=0)
    ax.axhspan(-0.147, 0.147, color="#e9e9e9", lw=0, zorder=0)
    ax.axhline(0, color="#777777", lw=0.7, zorder=1)
    ax.text(186, 0.0, "untested\n($B/D<300$)", fontsize=8, ha="left", va="center", color=MID,
            rotation=90)
    ax.text(620, 0.16, r"$|\delta|<0.147$", fontsize=8, ha="center", va="bottom", color=MID)
    for key in order:
        s = g[g.key == key]
        if s.empty:
            continue
        st = dict(SOURCE_STYLE[key])
        lab = st.pop("label")
        ax.errorbar(s.x, s.delta, yerr=[s.delta - s.ci_lo, s.ci_hi - s.delta], fmt="none",
                    ecolor=st["mec"], elinewidth=0.8, capsize=1.8, zorder=2)
        ax.plot(s.x, s.delta, ls="none", label=lab, zorder=3, mew=0.9, **st)
    names = {"P1": "P1", "P2": "P2", "P3 RC01": "RC01", "P4 RC06": "RC06", "P5 RC14": "RC14"}
    e16 = g[g.key == "E16 design problems"]
    nudge = {"P1": (-7, 4), "P2": (-7, -2), "P3 RC01": (7, 0), "P4 RC06": (12, 26),
             "P5 RC14": (-8, 2)}
    for r in e16.to_dict("records"):
        dx, dy = nudge.get(r["problems"], (5, 3))
        ax.annotate(f"{names.get(r['problems'], r['problems'])} ($D={r['dim']}$)",
                    (r["x"], r["delta"]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8, ha="left" if dx > 0 else ("right" if dx < 0 else "center"),
                    va="center",
                    arrowprops=(dict(arrowstyle="-", color=MID, lw=0.6, shrinkA=0, shrinkB=3)
                                if dy > 10 else None))
    ax.set_xticks([300, 500, 750, 1500, 3750, 10_000, 20_000, 50_000])
    ax.set_xticklabels(["300", "500", "750", "1,500", "3,750", "10,000", "20,000", "50,000"])
    ax.minorticks_off()
    ax.set_xlabel("evaluation budget per variable, $B/D$ (log scale)")
    ax.set_ylabel(r"$\delta$($6D$ vs $18D$), 95% CI")
    ax.text(70_000, 0.97, "18D better", fontsize=8, ha="right", va="top", color=MID)
    ax.text(70_000, -0.97, "6D better", fontsize=8, ha="right", va="bottom", color=MID)
    fig.legend(loc="outside lower center", ncol=2, handletextpad=0.3, columnspacing=1.6,
               fontsize=8)
    ax.grid(axis="x", visible=False)
    _save(fig, "fig_budget_guide.pdf")


# ---------------------------------------------------------------- convergence (withdrawn)
# The convergence figures of the version that went through the internal pre-submission review
# were withdrawn in the revision (its item REV-18):
# results/traces.csv holds only the per-checkpoint mean and median of 30 runs, so no
# dispersion band can be drawn from it, and the traces were recorded before the repaired
# configuration existed.  The per-function tables carry the final-error distributions.


# ---------------------------------------------------------------- S5.4
def fig_scalability():
    e9 = read_runs(os.path.join(RES, "E9_newmethod_vs_baselines.csv"))
    frames = []
    for fname in ("E1_cec2022_paper_budget.csv", "E3_cec2017.csv"):
        d = read_runs(os.path.join(RES, fname))
        d = pd.concat([d, e9[e9.tag == fname[:-4]]], ignore_index=True)
        d["suite_label"] = "CEC2022, 15,000 evaluations" if "2022" in fname else \
            "CEC2017, $10{,}000\\times D$ evaluations"
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["algo"] = df.algo.replace(DISPLAY)
    rows = []
    for (suite, dim), sub in df.groupby(["suite_label", "dim"]):
        algos = sorted(sub.algo.unique())
        funcs = sorted(sub.func.unique())
        M = np.array([[sub[(sub.func == f) & (sub.algo == a)]["error"].mean() for a in algos]
                      for f in funcs])
        for a, r in zip(algos, friedman(M)["avg_ranks"]):
            rows.append(dict(suite=suite, dim=dim, algo=a, rank=r))
    R = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 3.0), sharey=True, layout="constrained")
    for ax, suite in zip(axes, ["CEC2022, 15,000 evaluations", "CEC2017, $10{,}000\\times D$ evaluations"]):
        s = R[R.suite == suite]
        ends = []
        for a in ALGO_ORDER:
            t = s[s.algo == a].sort_values("dim")
            if not len(t):
                continue
            st = dict(ALGO_STYLE[a])
            ax.plot(t.dim, t["rank"], ms=4, mew=0.8, **st)
            ends.append((t["rank"].iloc[-1], a, t.dim.iloc[-1]))
        # direct labels at the right end; lines that end at (almost) the same rank
        # share one label, and labels are nudged apart where ranks are close
        ends.sort()
        merged = []
        for rk, a, x in ends:
            if merged and abs(rk - merged[-1][0]) < 0.1:
                merged[-1] = (merged[-1][0], merged[-1][1] + " / " + a, x)
            else:
                merged.append((rk, a, x))
        last = -9
        dx = 0.04 * (s.dim.max() - s.dim.min())
        for rk, a, x in merged:
            yy = max(rk, last + 0.42)
            ax.annotate(a, (x, rk), xytext=(x + dx, yy), textcoords="data", fontsize=8,
                        va="center")
            last = yy
        ax.set_xticks(sorted(s.dim.unique()))
        ax.set_xlim(s.dim.min() - 0.05 * (s.dim.max() - s.dim.min()),
                    s.dim.max() + 0.42 * (s.dim.max() - s.dim.min()))
        ax.set_xlabel("dimension $D$")
        ax.set_title(suite, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Friedman average rank")
    axes[0].invert_yaxis()
    _save(fig, "fig_scalability.pdf")


# ---------------------------------------------------------------- S6.10
def fig_feasibility():
    frames = [read_runs(os.path.join(RES, "E5_engineering.csv")),
              read_runs(os.path.join(RES, "E10_newmethod_engineering.csv"))]
    df = pd.concat(frames, ignore_index=True)
    df["algo"] = df.algo.replace(DISPLAY)
    probs = sorted(df["problem"].unique())
    algos = [a for a in ALGO_ORDER if a in set(df.algo)]
    M = np.array([[100 * df[(df["problem"] == pr) & (df.algo == a)].feasible.mean()
                   for pr in probs] for a in algos])
    fig, ax = plt.subplots(figsize=(FULL_W, 3.1), layout="constrained")
    w = 0.8 / len(algos)
    x = np.arange(len(probs))
    for i, a in enumerate(algos):
        xs = x + i * w - 0.4 + w / 2
        ax.bar(xs, M[i], width=w * 0.92, label=a, facecolor=BAR_FACE[a], edgecolor=INK,
               hatch=HATCH[a], linewidth=0.6)
        for xi, v in zip(xs, M[i]):
            if 0 < v < 100:
                ax.text(xi, v + 2, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["P1\npressure vessel", "P2\ngear train", "RC01\nheat exchanger",
                        "RC06\nblending\u2013pooling", "RC14\nbatch plant"])
    ax.set_ylabel("feasible runs (%)")
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="x", visible=False)
    ax.text(x[3], 4, "no feasible run", ha="center", va="bottom", fontsize=8, color=MID)
    fig.legend(ncol=len(algos), loc="outside lower center", handlelength=1.6,
               columnspacing=0.9)
    _save(fig, "fig_feasibility.pdf")


if __name__ == "__main__":
    fig_phase_gate()
    fig_ablation_ranks()
    fig_budget_guide()
    fig_scalability()
    fig_feasibility()
