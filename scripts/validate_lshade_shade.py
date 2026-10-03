"""Validation: compare the stored L-SHADE and SHADE runs with their authors' published runs.

The paper's L-SHADE and SHADE numbers on CEC2014 (stage E13, 10000*D
evaluations, 30 runs) are compared, function by function, with the per-run
result files that Tanabe & Fukunaga published for the same suite at the same
budget (51 runs).  Nothing is optimised: the script only reads
``results/E13_cec2014.csv`` (regenerate it with ``run_experiments.py E13``) and
the reference files, and prints.

Reference data
--------------
The reference files are not redistributed in this package (no licence is
stated).  Each file holds one function at one dimension: 14 rows x 51 columns,
column = run, row k = error after (0.01, 0.02, 0.03, 0.05, 0.1, ..., 0.9, 1.0) *
MaxFES (CEC2014 technical report, section 2.1).  Only row 14 (MaxFES) is used.

L-SHADE -- the CEC2014 competition entry (Tanabe's ``lshade.cc`` version 1.0.0),
in the official CEC2014 repository:

    https://github.com/P-N-Suganthan/CEC2014  (branch master)
    CEC-2014-Results-PartA.zip     6,002,877 bytes
        sha256 355b427798ba7fa7826fb65a1dae7b030ee8634b6790b1d55c4dd02d5cb1f143
      -> L-SHADE/L-SHADE_<f>_<D>.txt          (pass the L-SHADE directory)

SHADE -- SHADE 1.1, from Tanabe's data release for the L-SHADE paper.  The
original address (https://sites.google.com/site/tanaberyoji/data/Tanabe-CEC14-results.zip)
no longer resolves; the Internet Archive holds the file:

    http://web.archive.org/web/20201016162111id_/https://sites.google.com/site/tanaberyoji/data/Tanabe-CEC14-results.zip?attredirects=0
    Tanabe-CEC14-results.zip       6,577,575 bytes
        sha256 7dc2be55d6489288b7614b4cc14256b882db6e58ba729c566603877ea06a410c
      -> Tanabe-CEC14-results/SHADE11/SHADE11_<f>_<D>.txt   (pass the SHADE11 directory)

Its ``L-SHADE`` folder is byte-identical to the competition files above, so
either copy can serve as the L-SHADE reference.  Because individual files are
small and numerous, the script checks one digest per dimension: the sha256 of
the 30 files of that dimension concatenated in function order F1..F30.

What the comparison can and cannot show
---------------------------------------
* L-SHADE: same algorithm, settings, suite source, input data and budget.  The
  published runs come from lshade.cc 1.0.0, whose archive stores the successful
  trial vector instead of the defeated parent (the bug its author fixed in
  1.0.1); this package archives the parent.  This package's terminal CR value
  is absorbing, as in the paper's Algorithm 1; the released code re-accumulates
  the memory slot.  Runs: 30 here, 51 published.
* SHADE: same protocol, *different algorithm version*.  This package's SHADE is
  SHADE 1.0 (Tanabe & Fukunaga, 2013: H=100, |A|=N, p_i ~ U[2/N, 0.2], weighted
  arithmetic mean for M_CR), for which we found no published CEC2014 results.  The
  only per-function SHADE results on CEC2014 at 10000*D that we could find are
  SHADE 1.1's (H=D, |A|=2N, p=0.1, weighted Lehmer mean for M_CR with a terminal
  value, plus an archive-update bug fixed in 1.1.1).  Agreement there is a
  plausibility check, not a validation of the port.

Statistics follow validate_jso.py: ratio = our mean / published mean, computed
where both exceed 1e-8; functions where both are ~0 are counted separately and
functions where exactly one side is ~0 are listed.  The same is done for
medians.  Because both sides are per-run data, a two-sided rank-sum test per
function (Holm-corrected over the 30 functions of each block, errors rounded to
the CEC resolution of 1e-8) and Cliff's delta are reported as well.

Usage
-----
    python3 scripts/validate_lshade_shade.py --lshade-dir /path/to/L-SHADE \\
                                             --shade-dir  /path/to/SHADE11 \\
                                             [--summary-csv results/analysis/validation_lshade_shade.csv]
    # either directory may be omitted (that block is skipped); they can also be
    # given as the environment variables LSHADE_REF_DIR and SHADE11_REF_DIR

With --summary-csv the per-block statistics printed below are also written, one
row per (algorithm, dimension), for the supplementary validation table
(make_tables.py).  The file holds only statistics computed here, not the
reference data.
"""
import argparse
import functools
import hashlib
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sehho.bench import CEC2014_FUNCS
from sehho.stats import holm, cliffs_delta

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
E13_CSV = os.path.join(RES, "E13_cec2014.csv")
E13_TAG = "E13_cec2014_competition"
ZERO = 1e-8          # CEC convention: errors below 1e-8 count as 0

REFERENCES = {
    # algorithm name in our CSV -> reference description
    "LSHADE": dict(
        prefix="L-SHADE", env="LSHADE_REF_DIR", flag="--lshade-dir",
        label="L-SHADE 1.0.0, CEC2014 competition entry (Tanabe & Fukunaga)",
        # sha256 of the 30 files of one dimension concatenated in order F1..F30
        digest={30: "8d4e25ca07dfda00cc5b1d14fa50d8295353ab127fd0a55be24cf784830b1dd6",
                50: "4281c427c760e3e3fd4ce0230d60296309502fb6d1b59e9c9f1539a567e33dc5"},
        note=("same algorithm and settings; published runs from lshade.cc 1.0.0, which "
              "archives the trial vector (fixed in 1.0.1) and re-accumulates the terminal "
              "CR slot -- this package archives the parent and keeps the slot terminal")),
    "SHADE": dict(
        prefix="SHADE11", env="SHADE11_REF_DIR", flag="--shade-dir",
        label="SHADE 1.1 (Tanabe & Fukunaga, data release of the L-SHADE paper)",
        digest={30: "9de73ba84f24894daacc9d1b55b1ab450491b47c750606a8c3d9356f211f6ddf",
                50: "ab071caa8407e92b899c8ed812f7eded42274ba03f4f09e23cb83a3026891e0a"},
        note=("DIFFERENT VERSION: ours is SHADE 1.0 (H=100, |A|=N, p_i~U[2/N,0.2], "
              "arithmetic M_CR); the reference is SHADE 1.1 (H=D, |A|=2N, p=0.1, Lehmer "
              "M_CR with terminal value). Plausibility check only, not a port validation")),
}


def load_reference(ref_dir, prefix, dim, expected_digest=None):
    """Final-error vectors {func: array(51)} of one published algorithm at one dimension."""
    h = hashlib.sha256()
    ref = {}
    for f in CEC2014_FUNCS:
        path = os.path.join(ref_dir, f"{prefix}_{f}_{dim}.txt")
        if not os.path.exists(path):
            raise SystemExit(f"{path} not found: extract the reference archive (see this "
                             f"script's docstring) and pass the directory holding the "
                             f"{prefix}_<f>_<D>.txt files")
        with open(path, "rb") as fh:
            raw = fh.read()
        h.update(raw)
        M = np.loadtxt(path, ndmin=2)
        if M.shape[0] != 14:
            raise SystemExit(f"{path}: expected 14 checkpoint rows, found {M.shape[0]}")
        last = M[-1].astype(float)
        ref[f] = np.where(last < ZERO, 0.0, last)
    digest = h.hexdigest()
    ok = expected_digest is None or digest == expected_digest
    print(f"reference: {os.path.normpath(ref_dir)}/{prefix}_<f>_{dim}.txt "
          f"({len(next(iter(ref.values())))} runs per function)\n"
          f"           sha256(F1..F30 concatenated) {digest}"
          + ("" if ok else f"\nWARNING: not the published files this script was checked "
                           f"against (expected {expected_digest})"))
    return ref, ok


def _ratio_summary(ours, pub, label):
    """validate_jso.py statistics for one location statistic (mean or median)."""
    both0, one0, ratios = [], [], {}
    for f in CEC2014_FUNCS:
        o, p = ours[f], pub[f]
        if o <= ZERO and p <= ZERO:
            both0.append(f)
        elif o <= ZERO or p <= ZERO:
            one0.append(f)
        else:
            ratios[f] = o / p
    r = np.array(list(ratios.values()))
    out = dict(both0=both0, one0=one0, ratios=ratios, n=len(r),
               median=float(np.median(r)) if len(r) else np.nan,
               gmean=float(np.exp(np.mean(np.log(r)))) if len(r) else np.nan,
               within2=float(np.mean((r > 0.5) & (r < 2))) if len(r) else np.nan,
               within3=float(np.mean((r > 1 / 3) & (r < 3))) if len(r) else np.nan,
               outside2=[f for f, q in ratios.items() if not 0.5 < q < 2])
    print(f"{label}: both ~0 (solved) on {len(both0)} functions; comparable n={out['n']}  "
          f"median ratio={out['median']:.3f}  geometric-mean ratio={out['gmean']:.3f}")
    print(f"{' ' * len(label)}  within 2x of published: {out['within2'] * 100:.0f}%   "
          f"within 3x: {out['within3'] * 100:.0f}%")
    if out["outside2"]:
        print(f"{' ' * len(label)}  outside 2x: " + ", ".join(
            f"F{f} ({ratios[f]:.3g}, ours {'lower' if ratios[f] < 1 else 'higher'})"
            for f in out["outside2"]))
    if one0:
        print(f"{' ' * len(label)}  one side ~0 (no ratio): " + ", ".join(
            f"F{f} (ours {ours[f]:.3g}, published {pub[f]:.3g})" for f in one0))
    return out


def compare(ours_runs, ref_runs, title=""):
    """Print the per-function table and the summaries; return them as a dict.

    ``ours_runs`` and ``ref_runs`` map function -> 1-D array of final errors.
    """
    om = {f: float(np.mean(ours_runs[f])) for f in CEC2014_FUNCS}
    pm = {f: float(np.mean(ref_runs[f])) for f in CEC2014_FUNCS}
    omed = {f: float(np.median(ours_runs[f])) for f in CEC2014_FUNCS}
    pmed = {f: float(np.median(ref_runs[f])) for f in CEC2014_FUNCS}

    praw, delta = [], {}
    for f in CEC2014_FUNCS:
        # compared at the CEC resolution (1e-8): on plateau functions (e.g. F23)
        # both sides take values one or two ulps apart, which is not a difference
        x = np.round(np.asarray(ours_runs[f], float), 8)
        y = np.round(np.asarray(ref_runs[f], float), 8)
        if np.ptp(np.concatenate([x, y])) == 0:      # e.g. every run of both is 0
            praw.append(1.0)
        else:
            praw.append(float(stats.mannwhitneyu(x, y, alternative="two-sided").pvalue))
        delta[f] = cliffs_delta(x, y)
    padj = dict(zip(CEC2014_FUNCS, holm(praw)))

    if title:
        print(f"\n{title}")
    print(f"{'func':>5} {'ours mean':>11} {'publ mean':>11} {'ratio':>9}  {'ours med':>11} "
          f"{'publ med':>11} {'ratio':>9}  {'p(Holm)':>8} {'delta':>6}")
    for f in CEC2014_FUNCS:
        def rat(a, b):
            if a > ZERO and b > ZERO:
                return f"{a / b:9.3f}"
            return "   both~0" if (a <= ZERO and b <= ZERO) else "  one~0 "
        flag = " *" if padj[f] < 0.05 else ""
        print(f"F{f:<4d} {om[f]:11.4e} {pm[f]:11.4e} {rat(om[f], pm[f])}  {omed[f]:11.4e} "
              f"{pmed[f]:11.4e} {rat(omed[f], pmed[f])}  {padj[f]:8.3g} {delta[f]:+6.2f}{flag}")
    print()
    means = _ratio_summary(om, pm, "Means  ")
    medians = _ratio_summary(omed, pmed, "Medians")
    sig = [f for f in CEC2014_FUNCS if padj[f] < 0.05]
    print(f"Rank-sum, two-sided, Holm over {len(CEC2014_FUNCS)} functions: "
          f"{len(sig)} differ at alpha=0.05"
          + (": " + ", ".join(f"F{f} (ours {'lower' if delta[f] < 0 else 'higher'}, "
                              f"delta {delta[f]:+.2f})" for f in sig) if sig else ""))
    print("(delta = Cliff's delta of ours vs published, errors rounded to 1e-8; negative = "
          "ours lower; * = significant after Holm)")
    return dict(means=means, medians=medians, p_holm=padj, delta=delta, significant=sig,
                ours_mean=om, pub_mean=pm, ours_median=omed, pub_median=pmed)


@functools.lru_cache(maxsize=2)
def _read_runs(csv):
    # round_trip: pandas' default parser can be one ulp off the stored value
    return pd.read_csv(csv, float_precision="round_trip")


def load_ours(csv, algo, dim):
    df = _read_runs(csv)
    d = df[(df.tag == E13_TAG) & (df.algo == algo) & (df.dim == dim)]
    if d.empty:
        raise SystemExit(f"no {algo} runs at D={dim} with tag {E13_TAG} in {csv}")
    runs = d.groupby("func").size()
    print(f"ours:      {os.path.normpath(csv)} [{E13_TAG}], {algo}, D={dim}: "
          f"{runs.min()}-{runs.max()} runs per function, MaxFE="
          f"{', '.join(str(int(b)) for b in sorted(d.max_fes.unique()))}")
    if set(d.max_fes) != {10000 * dim} or set(d.fes_used) != {10000 * dim}:
        print(f"WARNING: the stored runs are not all at MaxFE={10000 * dim}")
    return {f: d.loc[d.func == f, "error"].to_numpy(float) for f in CEC2014_FUNCS}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lshade-dir", default=os.environ.get("LSHADE_REF_DIR"),
                    help="directory with L-SHADE_<f>_<D>.txt (default: $LSHADE_REF_DIR)")
    ap.add_argument("--shade-dir", default=os.environ.get("SHADE11_REF_DIR"),
                    help="directory with SHADE11_<f>_<D>.txt (default: $SHADE11_REF_DIR)")
    ap.add_argument("--dims", type=int, nargs="+", default=[30, 50])
    ap.add_argument("--csv", default=E13_CSV, help="stored runs (default: results/E13_cec2014.csv)")
    ap.add_argument("--summary-csv", default=None,
                    help="also write the per-block summary statistics to this CSV")
    args = ap.parse_args(argv)
    dirs = {"LSHADE": args.lshade_dir, "SHADE": args.shade_dir}
    if not any(dirs.values()):
        raise SystemExit("no reference directory: pass --lshade-dir and/or --shade-dir "
                         "(see this script's docstring for the official sources)")
    print("Comparing stored runs with published runs; no optimisation is run.")
    all_ok = True
    summary = []
    for algo, ref_dir in dirs.items():
        spec = REFERENCES[algo]
        if not ref_dir:
            print(f"\n[{algo}] skipped: no {spec['flag']} / ${spec['env']}")
            continue
        for dim in args.dims:
            print("\n" + "=" * 100)
            print(f"[{algo}] D={dim}, MaxFE={10000 * dim}: ours vs {spec['label']}")
            print(f"note: {spec['note']}")
            ref, ok = load_reference(ref_dir, spec["prefix"], dim, spec["digest"].get(dim))
            all_ok &= ok
            ours = load_ours(args.csv, algo, dim)
            res = compare(ours, ref)
            summary.append(_summary_row(algo, spec, dim, ours, ref, ok, res))
    if not all_ok:
        print("\nWARNING: at least one reference set did not match its expected digest")
    if args.summary_csv and summary:
        os.makedirs(os.path.dirname(os.path.abspath(args.summary_csv)), exist_ok=True)
        pd.DataFrame(summary).to_csv(args.summary_csv, index=False)
        print(f"\nwrote {args.summary_csv}")
    return 0


def _summary_row(algo, spec, dim, ours, ref, digest_ok, res):
    """One row of the supplementary validation table: the statistics printed above."""
    m, md = res["means"], res["medians"]
    r = np.array(list(m["ratios"].values()))
    in3 = r[(r > 1 / 3) & (r < 3)]
    sig = res["significant"]
    return dict(
        algorithm=algo, reference=spec["label"], dim=dim, budget=10000 * dim,
        runs_ours=int(min(len(v) for v in ours.values())),
        runs_reference=int(min(len(v) for v in ref.values())),
        reference_digest_ok=bool(digest_ok), n_functions=len(CEC2014_FUNCS),
        both_zero=len(m["both0"]), one_zero=" ".join(f"F{f}" for f in m["one0"]),
        n_comparable=m["n"], median_ratio=m["median"], gmean_ratio=m["gmean"],
        gmean_ratio_within3=float(np.exp(np.mean(np.log(in3)))) if len(in3) else np.nan,
        within2=m["within2"], within3=m["within3"],
        outside2=" ".join(f"F{f}" for f in m["outside2"]),
        outside3=" ".join(f"F{f}" for f, q in m["ratios"].items() if not 1 / 3 < q < 3),
        n_comparable_medians=md["n"], median_ratio_medians=md["median"],
        gmean_ratio_medians=md["gmean"], within2_medians=md["within2"],
        n_rank_sum_different=len(sig),
        rank_sum_different=" ".join(f"F{f}{'-' if res['delta'][f] < 0 else '+'}" for f in sig),
        note=spec["note"])


if __name__ == "__main__":
    raise SystemExit(main())
