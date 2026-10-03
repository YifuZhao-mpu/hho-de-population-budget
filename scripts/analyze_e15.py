"""Analysis of stage E15, exactly as fixed in its pre-registration.

E15 asks whether the CEC2014 replication of the repair depends on the
composition-rule amendment (see `run_e15_original_rule.py`).  Everything
computed here was fixed in
`results/analysis/PREREGISTRATION_E15_original_rule.json` before E15 ran; the
script refuses to run if the configuration in the data is not the registered
one.  The result is exploratory (post hoc relative to the confirmatory
analysis) and is reported whatever it shows.

Run:  python3 scripts/analyze_e15.py
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sehho.stats import cliffs_delta, holm

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
SRC = os.path.join(RES, "E15_original_rule_cec2014.csv")
E13 = os.path.join(RES, "E13_cec2014.csv")
PRE = os.path.join(RES, "analysis", "PREREGISTRATION_E15_original_rule.json")
ORIG = "GF:OriginalRule"
METHOD = "GF-Method"          # R-SHADE in the paper
PUBLISHED = "SEHHO-COBL"
BOOT = 20000
SEED = 20260915
NEGLIGIBLE = 0.147


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


def classify(lo, hi):
    if -NEGLIGIBLE < lo and hi < NEGLIGIBLE:
        return "negligible"
    if lo > 0:
        return "R-SHADE better"
    if hi < 0:
        return "original-rule configuration better"
    return "inconclusive"


def main():
    pre = json.load(open(PRE))
    cfg_hash = hashlib.sha256(json.dumps(pre["configuration"], sort_keys=True).encode()).hexdigest()
    if cfg_hash != pre["configuration_sha256"]:
        raise SystemExit("pre-registration configuration hash mismatch")
    df = pd.concat([pd.read_csv(SRC), pd.read_csv(E13)], ignore_index=True)
    funcs = sorted(df[df.algo == ORIG].func.unique())
    if len(funcs) != 30 or (df[df.algo == ORIG].groupby(["func", "dim", "max_fes"]).size() != 30).any():
        raise SystemExit("E15 is incomplete: expected 30 functions x 30 runs per cell")
    rng = np.random.default_rng(SEED)
    rows = []
    for d in (30, 50):
        for fes in (10_000 * d, 15_000):
            sub = df[(df.dim == d) & (df.max_fes == fes)]
            blk = "competition" if fes > 15_000 else "tight"
            for a, b, family in ((ORIG, PUBLISHED, "sensitivity_primary" if blk == "competition" else "descriptive"),
                                 (METHOD, PUBLISHED, "reference_E13"),
                                 (ORIG, METHOD, "secondary")):
                dl = per_fn_delta(sub, a, b, funcs)
                lo, hi = boot_ci(dl, rng)
                rows.append(dict(dim=d, max_fes=fes, budget=blk, a=a, b=b, family=family,
                                 mean_delta=float(dl.mean()), ci_lo=lo, ci_hi=hi,
                                 p=signed_rank(dl),
                                 a_lower_mean=int(sum(sub[(sub.algo == a) & (sub.func == f)]["error"].mean()
                                                      < sub[(sub.algo == b) & (sub.func == f)]["error"].mean()
                                                      for f in funcs))))
    out = pd.DataFrame(rows)
    for fam in ("sensitivity_primary", "secondary"):
        m = out.family == fam
        out.loc[m, "p_holm"] = holm(out.loc[m, "p"].to_numpy())
    out["class_vs_rshade"] = [classify(r.ci_lo, r.ci_hi) if r.family == "secondary" else ""
                              for r in out.itertuples()]
    prim = out[out.family == "sensitivity_primary"]
    both = bool((prim.ci_hi < 0).all())
    one = bool((prim.ci_hi < 0).any())
    verdict = ("replicates under the original rule" if both else
               "partial: one block only" if one else
               "does not replicate under the original rule")
    pd.set_option("display.width", 200)
    print(out[["dim", "budget", "a", "b", "family", "mean_delta", "ci_lo", "ci_hi", "p", "p_holm",
               "a_lower_mean", "class_vs_rshade"]].to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print(f"\nE15 sensitivity verdict (exploratory): {verdict}")
    os.makedirs(os.path.join(RES, "analysis"), exist_ok=True)
    out.to_csv(os.path.join(RES, "analysis", "e15_original_rule.csv"), index=False)
    json.dump({"verdict": verdict, "exploratory": True,
               "preregistration_configuration_sha256": cfg_hash,
               "primary_blocks": prim[["dim", "mean_delta", "ci_lo", "ci_hi", "p_holm"]].to_dict("records")},
              open(os.path.join(RES, "analysis", "e15_verdict.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
