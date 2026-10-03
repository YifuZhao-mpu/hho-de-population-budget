"""The confirmatory CEC2014 analysis, exactly as pre-registered.

Everything this script computes was fixed in
`results/analysis/PREREGISTRATION_cec2014.json`, an internal pre-registration
written (and timestamped inside the file, 2026-09-15T08:30Z) before the suite
was run.  The hash that file records, `frozen_configuration_sha256`, is the
sha256 of `json.dumps(frozen_configuration, sort_keys=True)`: it covers the
frozen configuration, not the rest of the file.

Before computing anything the script refuses to run unless
  1. that recorded hash matches the frozen configuration stored in the file,
     and
  2. the frozen configuration equals the configuration stage E13 runs with,
     `run_experiments._final_cfg()` (read back from selected_config.json and
     composition_decision.json),
so neither the registered configuration nor the code path that produced the
data can drift from the pre-registration unnoticed.

Why a third suite exists at all: CEC2017 was used to decide *which* components
to change -- the gate ablation and the population sweep were both run on it --
even though the final constants were chosen on CEC2022.  Choosing what to
repair is a design decision, so CEC2017 is not a holdout for the repaired
method.  CEC2014 was consulted for no design decision.

The verdict file keeps the field name `preregistration_sha256` for the
configuration hash, so that the stored cec2014_verdict.json is what this script
reproduces; the name predates the clarification above.

Run:  python3 scripts/analyze_cec2014.py [--out DIR]
      (default DIR: results/analysis)
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sehho.stats import friedman, friedman_holm_vs_control, cliffs_delta, holm

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
SRC = os.path.join(RES, "E13_cec2014.csv")
PRE = os.path.join(RES, "analysis", "PREREGISTRATION_cec2014.json")
METHOD = "GF-Method"          # R-SHADE in the paper
PUBLISHED = "SEHHO-COBL"
BOOT = 20000
SEED = 20260915


def boot_ci(v, rng, level=0.95):
    m = v[rng.integers(0, len(v), (BOOT, len(v)))].mean(axis=1)
    return (float(np.quantile(m, (1 - level) / 2)),
            float(np.quantile(m, 1 - (1 - level) / 2)))


def per_fn_delta(sub, a, b, funcs):
    """delta(a, b) per function; negative means `a` produces smaller errors."""
    return np.array([cliffs_delta(sub[(sub.func == f) & (sub.algo == a)]["error"].to_numpy(),
                                  sub[(sub.func == f) & (sub.algo == b)]["error"].to_numpy())
                     for f in funcs])


def signed_rank(v):
    nz = np.abs(v) > 0
    return float(sps.wilcoxon(v[nz])[1]) if nz.sum() else 1.0


def check_preregistration(pre):
    """Refuse to analyse unless the registered configuration is intact and is
    the configuration stage E13 runs with."""
    frozen = pre["frozen_configuration"]
    digest = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    if digest != pre["frozen_configuration_sha256"]:
        raise SystemExit("REFUSING TO RUN: the frozen configuration in the "
                         "pre-registration does not match the hash recorded with it "
                         f"(recorded {pre['frozen_configuration_sha256'][:16]}..., "
                         f"computed {digest[:16]}...)")
    spec = importlib.util.spec_from_file_location(
        "run_experiments", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "run_experiments.py"))
    rx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rx)
    used = rx._final_cfg()
    if used != frozen:
        raise SystemExit("REFUSING TO RUN: stage E13 runs with a configuration that is "
                         f"not the pre-registered one:\n  registered {frozen}\n"
                         f"  E13 uses    {used}")
    return digest


def main(argv=None):
    ap = argparse.ArgumentParser(description="The pre-registered CEC2014 analysis.")
    ap.add_argument("--out", default=os.path.join(RES, "analysis"),
                    help="directory for cec2014_confirmatory.csv and "
                         "cec2014_verdict.json (default: results/analysis)")
    args = ap.parse_args(argv)
    if not os.path.exists(SRC):
        print(f"{SRC} not found — run: python3 scripts/run_experiments.py E13")
        return 1
    pre = json.load(open(PRE))
    check_preregistration(pre)
    d = pd.read_csv(SRC)
    rng = np.random.default_rng(SEED)
    funcs = sorted(d.func.unique())

    print("=" * 74)
    print("CONFIRMATORY EVALUATION ON CEC2014")
    print("=" * 74)
    print(f"pre-registered {pre['written_at']}")
    print(f"frozen configuration sha256 {pre['frozen_configuration_sha256'][:16]}... "
          f"(matches the stored configuration and the one E13 runs with)")
    print(f"{len(funcs)} functions, {int(d.run.max()) + 1} runs per cell, "
          f"{len(d):,} runs total\n")

    comp = [(dim, 10_000 * dim) for dim in (30, 50)]
    tight = [(dim, 15_000) for dim in (30, 50)]

    # ---------------------------------------------------------- primary
    print("-" * 74)
    print("PRIMARY ENDPOINT — R-SHADE vs the published method, competition budget")
    print("declared success criterion: the 95% interval excludes zero in")
    print("R-SHADE's favour in BOTH blocks")
    print("-" * 74)
    prim, praw = [], []
    for dim, fes in comp:
        sub = d[(d.dim == dim) & (d.max_fes == fes)]
        dl = per_fn_delta(sub, METHOD, PUBLISHED, funcs)
        lo, hi = boot_ci(dl, rng)
        p = signed_rank(dl)
        praw.append(p)
        prim.append(dict(block=f"D={dim}, {fes:,} FEs", dim=dim, max_fes=fes,
                         mean_delta=float(dl.mean()), ci_lo=lo, ci_hi=hi, p=p,
                         favours_method=bool(hi < 0),
                         wins=int((dl < 0).sum()), losses=int((dl > 0).sum())))
    for r, ph in zip(prim, holm(praw)):
        r["p_holm"] = float(ph)
        print(f"  {r['block']:<22s} mean delta {r['mean_delta']:+.4f} "
              f"[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]  p_Holm={ph:.4f}  "
              f"functions {r['wins']}/{r['losses']}  "
              f"{'REPLICATES' if r['favours_method'] else 'does not replicate'}")
    replicated = all(r["favours_method"] for r in prim)
    partial = any(r["favours_method"] for r in prim)
    verdict = ("REPLICATED" if replicated else
               "PARTIAL REPLICATION" if partial else "FAILED TO REPLICATE")
    print(f"\n  => {verdict}")

    # ---------------------------------------------------------- secondary
    out_rows = list(prim)
    for label, blocks in (("competition", comp), ("tight", tight)):
        print("\n" + "-" * 74)
        print(f"SECONDARY — full ranking, {label} budget")
        print("-" * 74)
        for dim, fes in blocks:
            sub = d[(d.dim == dim) & (d.max_fes == fes)]
            algos = sorted(sub.algo.unique(), key=lambda a: (a != METHOD, a))
            M = np.array([[sub[(sub.func == f) & (sub.algo == a)]["error"].mean()
                           for a in algos] for f in funcs])
            fr, post = friedman_holm_vs_control(M, algos, METHOD)
            order = sorted(zip(algos, fr["avg_ranks"]), key=lambda t: t[1])
            print(f"\n  D={dim}, {fes:,} FEs")
            for a, r in order:
                tag = "  <<< R-SHADE" if a == METHOD else ""
                extra = ""
                if a != METHOD:
                    row = next((r for r in post if r["algorithm"] == a), None)
                    if row is not None:
                        row = pd.Series(row)
                        dl = per_fn_delta(sub, METHOD, a, funcs)
                        lo, hi = boot_ci(dl, rng)
                        extra = (f"   p_Holm={row.p_holm:.4f}  delta={dl.mean():+.3f} "
                                 f"[{lo:+.3f},{hi:+.3f}]")
                        out_rows.append(dict(block=f"D={dim}, {fes:,} FEs", dim=dim,
                                             max_fes=fes, opponent=a,
                                             mean_delta=float(dl.mean()), ci_lo=lo, ci_hi=hi,
                                             p_holm=float(row.p_holm), family=label))
                print(f"     {a:<12s} {r:.2f}{tag}{extra}")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "cec2014_confirmatory.csv")
    pd.DataFrame(out_rows).to_csv(path, index=False)
    # "preregistration_sha256" holds the hash of the frozen configuration (see
    # the module docstring); the field name is kept so the stored file reproduces
    json.dump({"primary": prim, "verdict": verdict,
               "preregistration_sha256": pre["frozen_configuration_sha256"]},
              open(os.path.join(args.out, "cec2014_verdict.json"), "w"), indent=2)
    print(f"\nwrote {path}")
    print("\nNOTE: the configuration was frozen before this stage ran and is not "
          "altered by anything above, whichever way the verdict went.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
