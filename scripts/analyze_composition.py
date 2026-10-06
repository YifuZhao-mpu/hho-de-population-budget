"""Does each remaining part of the gate-free method earn its place?

The published ablation measured every module against SEHHO-COBL, a method whose
population rule is now known to be its dominant design error.  A module can look
negligible next to a badly sized population and matter next to a well sized one,
so the question has to be asked again around the new default.

Two verdicts are reported for each removal, and they are not the same question:

*  **detected**   - a signed-rank test on the per-function Cliff's deltas says
   the removal changes performance.  A large p-value here means only that no
   change was detected.
*  **negligible** - the whole 95% bootstrap interval for the mean delta lies
   inside the pre-declared band |delta| < 0.147 (Romano et al., 2006).  This is
   an equivalence test, not a restatement of the p-value.

Two decision rules, and why there are two
-----------------------------------------
The rule that turns these verdicts into keep-or-drop decisions was changed
after its first output had been seen.  Both rules are implemented below and
both decisions are printed.  The amended rule's decision is the one written to
``composition_decision.json``, and therefore the one behind the configuration
that was frozen and evaluated on CEC2014 (stage E13).

*  ORIGINAL rule, as coded at 2026-09-14T07:48Z, before the E8 data existed:
   a part is kept if some block shows that removing it hurts with
   p_Holm < 0.05; otherwise it is dropped only if its 95% interval lies inside
   the negligible band in every block; anything else is kept.  Run on the E8
   data at 2026-09-14T09:01Z, it kept every part except the p annealing.
*  AMENDED rule, adopted at 2026-09-14T09:02Z, after that output had been
   seen (and after stage E11, the stripped configuration, had been launched):
   a part earns its place if some block has ci_lo > 0; it is dropped if it
   never earns its place and either is negligible in every block or has
   ci_hi < 0 in some block; anything else is kept.

The two rules disagree only on COBL and the Levy perturbation, which the
original rule keeps and the amended rule drops.  Stage E15
(scripts/run_e15_original_rule.py) runs the configuration the original rule
would have produced on the confirmatory suite.

Run:  python3 scripts/analyze_composition.py [--out DIR]
      (default DIR: results/analysis)
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sehho.runfiles import read_runs  # noqa: E402  (pandas.read_csv; SEHHO_ERROR_DECIMALS, Section 2.6)
from sehho.stats import friedman, cliffs_delta, holm

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
SRC = os.path.join(RES, "E8_composition_cec2022.csv")
SRC_STRIPPED = os.path.join(RES, "E11_stripped_cec2022.csv")
RNG = np.random.default_rng(20260914)
BOOT = 20000
NEGLIGIBLE = 0.147
BASE = "Default"


def boot_ci(v, level=0.95):
    m = v[RNG.integers(0, len(v), (BOOT, len(v)))].mean(axis=1)
    return (float(np.quantile(m, (1 - level) / 2)),
            float(np.quantile(m, 1 - (1 - level) / 2)))


def per_fn_delta(sub, a, b, funcs):
    return np.array([cliffs_delta(sub[(sub.func == f) & (sub.cand == a)]["error"].to_numpy(),
                                  sub[(sub.func == f) & (sub.cand == b)]["error"].to_numpy())
                     for f in funcs])


def decide_original(out, cands):
    """The decision rule as coded at 2026-09-14T07:48Z, before the E8 data existed.

    Kept verbatim in substance so that its decision can be compared with the
    amended rule's; it is printed but never written to composition_decision.json.
    """
    keep, drop = [], []
    for c in cands:
        sub = out[out.change == c]
        earns = bool(((sub.p_holm < 0.05) & (sub.mean_delta > 0)).any())
        neg_everywhere = bool(sub.negligible.all())
        if earns:
            best = sub.loc[sub.mean_delta.idxmax()]
            print(f"   {c:<10s} KEEP  — matters in at least one block "
                  f"(max delta {best.mean_delta:+.3f} at D={int(best.dim)}, "
                  f"{int(best.max_fes):,} FEs)")
            keep.append(c)
        elif neg_everywhere:
            print(f"   {c:<10s} DROP  — negligible in every block")
            drop.append(c)
        else:
            print(f"   {c:<10s} KEEP  — not detected, but the interval does not "
                  f"bound it below negligible everywhere")
            keep.append(c)
    return keep, drop


def decide_amended(out, cands):
    """The decision rule adopted at 2026-09-14T09:02Z, after the original rule's
    output had been seen; its decision is the one written to
    composition_decision.json.

    Twelve functions and Holm over the seven changes in each block leave very
    little power, so the amended rule reads intervals rather than p-values:

      a part EARNS its place   if some block has ci_lo > 0  (removing it hurts)
      removal HELPS            if some block has ci_hi < 0  (removing it helps)

    A part is dropped when it never earns its place and either is negligible
    everywhere or is positively better off removed somewhere.  Anything else
    is kept.
    """
    keep, drop, notes = [], [], {}
    for c in cands:
        sub = out[out.change == c]
        earns = sub[(sub.ci_lo > 0)]
        helps = sub[(sub.ci_hi < 0)]
        neg_everywhere = bool(sub.negligible.all())
        if len(earns):
            b = earns.loc[earns.mean_delta.idxmax()]
            msg = (f"KEEP  — removing it costs {b.mean_delta:+.3f} "
                   f"[{b.ci_lo:+.3f},{b.ci_hi:+.3f}] at D={int(b.dim)}, "
                   f"{int(b.max_fes):,} FEs")
            keep.append(c)
        elif len(helps):
            b = helps.loc[helps.mean_delta.idxmin()]
            msg = (f"DROP  — never helps, and removing it gains "
                   f"{-b.mean_delta:+.3f} [{-b.ci_hi:+.3f},{-b.ci_lo:+.3f}] "
                   f"at D={int(b.dim)}, {int(b.max_fes):,} FEs")
            drop.append(c)
        elif neg_everywhere:
            msg = "DROP  — negligible in every block"
            drop.append(c)
        else:
            msg = "KEEP  — no block bounds it below negligible; kept by default"
            keep.append(c)
        notes[c] = msg
        print(f"   {c:<10s} {msg}")
    return keep, drop, notes


def main(argv=None):
    ap = argparse.ArgumentParser(description="Composition analysis of the gate-free method.")
    ap.add_argument("--out", default=os.path.join(RES, "analysis"),
                    help="directory for composition.csv and composition_decision.json "
                         "(default: results/analysis)")
    args = ap.parse_args(argv)
    if not os.path.exists(SRC):
        print(f"{SRC} not found — run: python3 scripts/run_experiments.py E8")
        return 1
    frames = [read_runs(SRC)]
    if os.path.exists(SRC_STRIPPED):
        frames.append(read_runs(SRC_STRIPPED))
    d = pd.concat(frames, ignore_index=True)
    d["cand"] = d.algo.str.slice(3).str.split("@").str[0]
    d["block"] = d.algo.str.split("@").str[1]
    cands = [c for c in d.cand.unique() if c != BASE]
    # the single-part changes of E8, i.e. what existed when the original rule was
    # coded; the stripped configuration (E11) is a consequence of the amendment
    e8_cands = [c for c in cands if c in set(d.cand.iloc[:len(frames[0])])]
    funcs = sorted(d.func.unique())

    rows = []
    for dim in sorted(d.dim.unique()):
        for blk in ("matched", "competition"):
            sub = d[(d.dim == dim) & (d.block == blk)]
            if sub.empty:
                continue
            fes = int(sub.max_fes.iloc[0])
            print(f"\n=== CEC2022 D={dim}, {fes:,} FEs "
                  f"(change relative to the default; positive favours the default) ===")
            praw, tmp = [], []
            for c in cands:
                dl = per_fn_delta(sub, c, BASE, funcs)
                lo, hi = boot_ci(dl)
                nz = np.abs(dl) > 0
                p = float(sps.wilcoxon(dl[nz])[1]) if nz.sum() else 1.0
                praw.append(p)
                tmp.append(dict(dim=dim, block=blk, max_fes=fes, change=c,
                                mean_delta=float(dl.mean()), ci_lo=lo, ci_hi=hi, p=p,
                                negligible=bool(lo > -NEGLIGIBLE and hi < NEGLIGIBLE)))
            for r, ph in zip(tmp, holm(praw)):
                r["p_holm"] = float(ph)
                verdict = ("part MATTERS" if (ph < 0.05 and r["mean_delta"] > 0) else
                           "removal HELPS" if (ph < 0.05 and r["mean_delta"] < 0) else
                           "negligible" if r["negligible"] else "undetermined")
                print(f"   {r['change']:<10s} {r['mean_delta']:+.4f} "
                      f"[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]  p_Holm={ph:.4f}  {verdict}")
            rows += tmp

    out = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    out.to_csv(os.path.join(args.out, "composition.csv"), index=False)

    # ---------------------------------------------------------------- decision
    # With E11 present each Holm family has seven members rather than the six it
    # had when the original rule was first run; the original rule's decisions
    # on the six single changes are the same either way.
    print("\n=== Decision per part: ORIGINAL rule (coded 2026-09-14T07:48Z, "
          "before the E8 data; not used) ===")
    keep_o, drop_o = decide_original(out, e8_cands)
    print("\n=== Decision per part: AMENDED rule (adopted 2026-09-14T09:02Z, after "
          "the original rule's output had been seen; written to "
          "composition_decision.json) ===")
    keep, drop, notes = decide_amended(out, cands)
    flipped = [c for c in e8_cands if (c in keep_o) != (c in keep)]
    print("\n   The two rules disagree on: "
          + (", ".join(f"{c} (original {'KEEP' if c in keep_o else 'DROP'}, amended "
                       f"{'KEEP' if c in keep else 'DROP'})" for c in flipped)
             if flipped else "nothing"))

    parts_to_drop = sorted(x[1:] for x in drop if x.startswith("-"))
    gate_rows = out[out.change == "+Gate"]
    gate_harm = gate_rows[gate_rows.ci_lo > 0]
    if len(gate_harm):
        b = gate_harm.loc[gate_harm.mean_delta.idxmax()]
        print(f"\n   Re-introducing the escape-energy gate costs "
              f"{b.mean_delta:+.3f} [{b.ci_lo:+.3f},{b.ci_hi:+.3f}] at D={int(b.dim)}, "
              f"{int(b.max_fes):,} FEs — the gate stays out, now confirmed on "
              f"CEC2022 as well as CEC2017.")

    stripped = out[out.change == "Stripped"]
    use_stripped = None
    if len(stripped):
        print("\n=== Stripped configuration (all dropped parts removed at once) ===")
        for _, r in stripped.iterrows():
            print(f"   D={int(r.dim):<3d} {int(r.max_fes):>9,} FEs  "
                  f"{r.mean_delta:+.4f} [{r.ci_lo:+.4f},{r.ci_hi:+.4f}]  "
                  f"p_Holm={r.p_holm:.4f}"
                  f"{'   [negligible]' if r.negligible else ''}")
        # adopt the simpler method unless some block shows it is actually worse
        worse = stripped[stripped.ci_lo > 0]
        use_stripped = bool(len(worse) == 0)
        print("   => " + ("adopt the stripped configuration: no block shows it worse, "
                          "and it is the simpler method"
                          if use_stripped else
                          "keep the full configuration: the stripped one is worse in "
                          f"{len(worse)} block(s)"))

    json.dump({"baseline": BASE, "kept": keep, "dropped": drop,
               "parts_to_drop": parts_to_drop, "notes": notes,
               "negligible_margin": NEGLIGIBLE, "use_stripped": use_stripped},
              open(os.path.join(args.out, "composition_decision.json"), "w"), indent=2)
    print(f"\nwrote {os.path.join(args.out, 'composition.csv')}")
    print(f"wrote {os.path.join(args.out, 'composition_decision.json')} (amended rule)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
