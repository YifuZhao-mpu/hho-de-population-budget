"""End-to-end validation: reproduce Brest's own published jSO CEC2017 results.

If both the compiled benchmark and the jSO port are faithful, running our jSO at
the competition protocol (MaxFE = 10000*D) must land on the same mean error
values that the jSO authors published with their source archive.

Reference data
--------------
The published results are the ``D<dim>.rez`` files of Brest, Sepesy Maucec and
Boskovic's jSO source archive, distributed in the official CEC2017 repository.
They are not redistributed in this package (their licence is not stated):

    https://github.com/P-N-Suganthan/CEC2017-BoundContrained  (branch master)
    Codes-of-Top-Methods-and-results.zip           22,821,612 bytes
        sha256 0249e3cc39070cc9d16578fa64ddc88aa2abe1ee72c5147b595707ac6502e361
      -> jSO-SOURCE-RESULTS-29May2017.tar.gz
        sha256 720f958ebb1ab2c02aaea4be408314325773c72ba5912fb4853a94fdf11b251f
        -> jSO-SOURCE-RESULTS/D30.rez
           sha256 03e31fe64ff1ada98afab3688ca70ea4232248b78aa690cf8d94d6d4114df2f4

Each line of a .rez file is ``fNN best worst median mean std``.  Point this
script at the extracted ``jSO-SOURCE-RESULTS`` directory with ``--rez-dir`` or
with the environment variable ``JSO_REZ_DIR``.

Usage
-----
    # run our jSO (RUNS runs per function, 10000*D evaluations), write
    # results/validate_jso_D<dim>.csv, and compare it with the published means
    python3 scripts/validate_jso.py 30 30 --rez-dir /path/to/jSO-SOURCE-RESULTS

    # compare the stored runs only: nothing is optimised and nothing is written
    python3 scripts/validate_jso.py 30 --compare-only --rez-dir /path/to/jSO-SOURCE-RESULTS

    # ... and write the summary statistics for the supplementary validation table
    python3 scripts/validate_jso.py 30 --compare-only --rez-dir /path/to/jSO-SOURCE-RESULTS \\
        --summary-csv results/analysis/validation_jso.csv

The summary file holds only the statistics computed here, not the reference data.
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sehho.runner import Cell, run_all
from sehho.bench import CEC2017_FUNCS

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
# sha256 of the published reference file the paper's numbers were computed against
REZ_SHA256 = {30: "03e31fe64ff1ada98afab3688ca70ea4232248b78aa690cf8d94d6d4114df2f4"}


def load_reference(rez_dir, dim):
    path = os.path.join(rez_dir, f"D{dim}.rez")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found: extract jSO-SOURCE-RESULTS from the official "
                         f"archive (see this script's docstring) and pass --rez-dir or "
                         f"set JSO_REZ_DIR")
    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    print(f"reference: {path}\n           sha256 {digest}")
    if dim in REZ_SHA256 and digest != REZ_SHA256[dim]:
        print(f"WARNING: this is not the published D{dim}.rez the paper used "
              f"(expected sha256 {REZ_SHA256[dim]})")
    ref = {}
    for line in open(path):
        p = line.split()
        if len(p) == 6 and p[0].startswith("f"):
            ref[int(p[0][1:])] = dict(best=float(p[1]), worst=float(p[2]),
                                      median=float(p[3]), mean=float(p[4]), std=float(p[5]))
    return ref


def compare(df, ref):
    g = df.groupby("func")["error"].agg(["mean", "median", "std"])
    print(f"\n{'func':>5} {'ours mean':>12} {'Brest mean':>12} {'ratio':>9}   {'ours med':>11} {'Brest med':>11}")
    ratios = []
    for f in CEC2017_FUNCS:
        om, rm = g.loc[f, "mean"], ref[f]["mean"]
        if rm > 1e-8 and om > 1e-8:
            ratios.append(om / rm)
            rat = f"{om/rm:9.3f}"
        else:
            rat = "     both~0" if (om <= 1e-8 and rm <= 1e-8) else "        --"
        print(f"F{f:<4d} {om:12.4e} {rm:12.4e} {rat}   {g.loc[f,'median']:11.4e} {ref[f]['median']:11.4e}")
    ratios = np.array(ratios)
    print(f"\nFunctions where both are ~0 (solved): "
          f"{sum(1 for f in CEC2017_FUNCS if g.loc[f,'mean']<=1e-8 and ref[f]['mean']<=1e-8)}")
    print(f"Comparable functions: n={len(ratios)}  median ratio={np.median(ratios):.3f}  "
          f"geometric mean ratio={np.exp(np.mean(np.log(ratios))):.3f}")
    print(f"Within 2x of published: {np.mean((ratios>0.5)&(ratios<2))*100:.0f}%   "
          f"within 3x: {np.mean((ratios>1/3)&(ratios<3))*100:.0f}%")
    return dict(n=len(ratios), median=float(np.median(ratios)),
                gmean=float(np.exp(np.mean(np.log(ratios)))),
                within2=float(np.mean((ratios > 0.5) & (ratios < 2))),
                within3=float(np.mean((ratios > 1 / 3) & (ratios < 3))),
                both0=int(sum(1 for f in CEC2017_FUNCS
                              if g.loc[f, "mean"] <= 1e-8 and ref[f]["mean"] <= 1e-8)),
                runs=int(df.groupby("func").size().min()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dim", nargs="?", type=int, default=30)
    ap.add_argument("runs", nargs="?", type=int, default=30,
                    help="runs per function (ignored with --compare-only)")
    ap.add_argument("--rez-dir", default=os.environ.get("JSO_REZ_DIR"),
                    help="the extracted jSO-SOURCE-RESULTS directory "
                         "(default: $JSO_REZ_DIR)")
    ap.add_argument("--compare-only", action="store_true",
                    help="compare results/validate_jso_D<dim>.csv without re-running")
    ap.add_argument("--summary-csv", default=None,
                    help="also write the summary statistics to this CSV")
    args = ap.parse_args(argv)
    if not args.rez_dir:
        raise SystemExit("no reference directory: pass --rez-dir or set JSO_REZ_DIR "
                         "(see this script's docstring for the official source)")
    ref = load_reference(args.rez_dir, args.dim)
    out_csv = os.path.join(RES, f"validate_jso_D{args.dim}.csv")

    if args.compare_only:
        if not os.path.exists(out_csv):
            raise SystemExit(f"{out_csv} not found: run without --compare-only first")
        df = pd.read_csv(out_csv)
        print(f"Comparing stored runs {os.path.normpath(out_csv)}: "
              f"{df.groupby('func').size().min()} runs per function, MaxFE="
              f"{', '.join(str(int(b)) for b in sorted(df.max_fes.unique()))}; "
              f"no optimisation is run")
        if set(df.dim) != {args.dim} or set(df.max_fes) != {10000 * args.dim}:
            print(f"WARNING: the stored runs are not jSO at D={args.dim} with "
                  f"MaxFE={10000 * args.dim}")
    else:
        cells = [Cell("cec2017", f, args.dim, "jSO", 10000 * args.dim, args.runs,
                      tag="validate_jso") for f in CEC2017_FUNCS]
        print(f"Running jSO on CEC2017 D={args.dim}, {args.runs} runs, "
              f"MaxFE={10000*args.dim} ...")
        df = run_all(cells, out_csv=out_csv)
    st = compare(df, ref)
    if args.summary_csv:
        with open(os.path.join(args.rez_dir, f"D{args.dim}.rez"), "rb") as fh:
            ok = hashlib.sha256(fh.read()).hexdigest() == REZ_SHA256.get(args.dim)
        os.makedirs(os.path.dirname(os.path.abspath(args.summary_csv)), exist_ok=True)
        pd.DataFrame([dict(
            algorithm="jSO", reference="jSO, Brest et al.'s published CEC2017 results (D30.rez)",
            dim=args.dim, budget=10000 * args.dim, runs_ours=st["runs"], runs_reference=51,
            reference_digest_ok=bool(ok), n_functions=len(CEC2017_FUNCS), both_zero=st["both0"],
            n_comparable=st["n"], median_ratio=st["median"], gmean_ratio=st["gmean"],
            within2=st["within2"], within3=st["within3"], n_rank_sum_different=np.nan,
            rank_sum_different="", note="reference file holds summary statistics only "
            "(best, worst, median, mean, std), so no per-run test is possible")]).to_csv(
            args.summary_csv, index=False)
        print(f"\nwrote {args.summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
