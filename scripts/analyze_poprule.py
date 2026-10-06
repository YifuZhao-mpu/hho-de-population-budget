"""Choose the population rule for the gate-free method, on the design suite only.

The ablation shows the escape-energy gate is harmful, but it also shows
something larger: on CEC2017 at D=30 and 300,000 evaluations, raising the
population from the published N=30 to N=120 is worth Cliff's delta +0.60,
against +0.17 for deleting the gate.  Population sizing, not the phase policy,
is the dominant design error in the published method.

This script picks the replacement rule.  Two properties of the protocol matter
more than the result:

1.  **The choice is made on CEC2022 only.**  CEC2017 is never consulted while
    choosing.  It is still not a held-out suite: the decision to repair the
    population rule was taken on it (E6), so it is the diagnostic suite, and
    the confirmatory suite is CEC2014 (E13).  A rule chosen on the suite it is
    then evaluated on would be worth nothing.
2.  **Both budgets are weighted equally.**  The published method is competitive
    at 15,000 evaluations and collapses at competition budgets; a rule that wins
    on average by trading the first for the second would not be a fix.  Each
    candidate is therefore ranked within each (dimension, budget) block
    separately, and the blocks are combined only at the end.

Run:  python3 scripts/analyze_poprule.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sehho.runfiles import read_runs  # noqa: E402  (pandas.read_csv; SEHHO_ERROR_DECIMALS, Section 2.6)
from sehho.stats import friedman, cliffs_delta, holm
from scipy import stats as sps

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
SRC = os.path.join(RES, "E7_poprule_cec2022.csv")
RNG = np.random.default_rng(20260914)
BOOT = 20000


def boot_ci(v, level=0.95):
    m = v[RNG.integers(0, len(v), (BOOT, len(v)))].mean(axis=1)
    return float(np.quantile(m, (1 - level) / 2)), float(np.quantile(m, 1 - (1 - level) / 2))


def per_fn_delta(df, a, b, funcs):
    return np.array([cliffs_delta(df[(df.func == f) & (df.cand == a)]["error"].to_numpy(),
                                  df[(df.func == f) & (df.cand == b)]["error"].to_numpy())
                     for f in funcs])


def main():
    if not os.path.exists(SRC):
        print(f"{SRC} not found — run: python3 scripts/run_experiments.py E7")
        return 1
    d = read_runs(SRC)
    d["cand"] = d.algo.str.slice(4).str.split("@").str[0]
    d["block"] = d.algo.str.split("@").str[1]
    cands = sorted(d.cand.unique())
    funcs = sorted(d.func.unique())

    blocks, rank_rows = [], {c: [] for c in cands}
    for dim in sorted(d.dim.unique()):
        for blk in ("matched", "competition"):
            sub = d[(d.dim == dim) & (d.block == blk)]
            if sub.empty:
                continue
            fes = int(sub.max_fes.iloc[0])
            M = np.array([[sub[(sub.func == f) & (sub.cand == c)]["error"].mean()
                           for c in cands] for f in funcs])
            fr = friedman(M)
            label = f"D={dim}, {fes:,} FEs"
            blocks.append((label, dict(zip(cands, fr["avg_ranks"]))))
            for c, r in zip(cands, fr["avg_ranks"]):
                rank_rows[c].append(r)

    print("Friedman average rank within each block (lower is better)\n")
    hdr = f"{'candidate':<12s}" + "".join(f"{lab:>22s}" for lab, _ in blocks) + f"{'mean':>9s}"
    print(hdr)
    print("-" * len(hdr))
    mean_rank = {c: float(np.mean(rank_rows[c])) for c in cands}
    for c in sorted(cands, key=lambda c: mean_rank[c]):
        print(f"{c:<12s}" + "".join(f"{r[c]:>22.2f}" for _, r in blocks)
              + f"{mean_rank[c]:>9.2f}")

    winner = min(cands, key=lambda c: mean_rank[c])
    worst_block = max(range(len(blocks)), key=lambda i: blocks[i][1][winner])
    print(f"\nwinner by mean rank: {winner}")
    print(f"  its weakest block: {blocks[worst_block][0]} "
          f"(rank {blocks[worst_block][1][winner]:.2f})")

    print(f"\nEffect size of each candidate against {winner}, per block")
    print("(positive delta favours the winner; 95% bootstrap CI over functions)\n")
    rows = []
    for dim in sorted(d.dim.unique()):
        for blk in ("matched", "competition"):
            sub = d[(d.dim == dim) & (d.block == blk)]
            if sub.empty:
                continue
            fes = int(sub.max_fes.iloc[0])
            print(f"  -- D={dim}, {fes:,} FEs --")
            praw, tmp = [], []
            for c in cands:
                if c == winner:
                    continue
                dl = per_fn_delta(sub, c, winner, funcs)
                lo, hi = boot_ci(dl)
                nz = np.abs(dl) > 0
                p = float(sps.wilcoxon(dl[nz])[1]) if nz.sum() else 1.0
                praw.append(p)
                tmp.append(dict(dim=dim, block=blk, max_fes=fes, candidate=c,
                                mean_delta=float(dl.mean()), ci_lo=lo, ci_hi=hi, p=p))
            for r, ph in zip(tmp, holm(praw)):
                r["p_holm"] = float(ph)
                mark = "*" if ph < 0.05 else " "
                print(f"     {r['candidate']:<12s} {r['mean_delta']:+.4f} "
                      f"[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]  p_Holm={ph:.4f}{mark}")
            rows += tmp

    out = pd.DataFrame(rows)
    os.makedirs(os.path.join(RES, "analysis"), exist_ok=True)
    out.to_csv(os.path.join(RES, "analysis", "poprule.csv"), index=False)

    sel = os.path.join(RES, "analysis", "selected_config.json")
    # The stored selected_config.json (2026-09-14) predates the relabelling of
    # CEC2017 and still says "held_out"; no code reads that field (stages read
    # only "winner"), and the timestamped file is deliberately left as written.
    json.dump({"winner": winner,
               "mean_rank": mean_rank,
               "blocks": [{"label": lab, "ranks": r} for lab, r in blocks],
               "design_suite": "cec2022 (D=10,20; matched and competition budgets)",
               "diagnostic_suite": "cec2017 (D=30,50,100)"},
              open(sel, "w"), indent=2)
    print(f"\nwrote {sel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
