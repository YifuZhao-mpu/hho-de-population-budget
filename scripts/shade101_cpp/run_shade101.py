"""Build and run Tanabe's SHADE 1.0.1 C++ code on this package's CEC2014 data.

This is the code-to-code check of Supplementary Section S2.3: the SHADE port
of this package is SHADE 1.0, for which no published CEC2014 results exist, so
its authors' own C++ release was run on the same suite, data and budget.  The
51 runs per function made this way are released as
``results/validation_shade101_cpp_cec2014.csv`` and analysed by
``scripts/validate_engine_variants.py analyze``.  They are our runs of
third-party code; the code itself is not redistributed.

Third-party inputs (not in this package; checked by sha256 before use)
--------------------------------------------------------------------------
SHADE 1.0.1 for CEC2013, Tanabe's release.  The original address,
https://sites.google.com/site/tanaberyoji/software/SHADE1.0.1_CEC2013.zip,
no longer resolves; the Internet Archive holds the file:

    http://web.archive.org/web/20201016162057id_/https://sites.google.com/site/tanaberyoji/software/SHADE1.0.1_CEC2013.zip?attredirects=0
    SHADE1.0.1_CEC2013.zip   1,774,242 bytes
        sha256 92fafa12f90149f5df00402770b9ddb3961a922bfcef3fd7579e2d7440737d86

The CEC2014 test function in Tanabe's own build, cec14_test_func.cc
(sha256 eb7da784d9e027c02c333fb0ca4000be829287a45e781714f71fd567c5584be6),
which is the same file in three places; pass any of them:

    * folder LSHADE_CEC14/ of Tanabe's L-SHADE release, Internet Archive
      http://web.archive.org/web/20201016162105id_/https://sites.google.com/site/tanaberyoji/software/LSHADE1.0.0_CEC2014.zip?attredirects=0
      (LSHADE1.0.0_CEC2014.zip, sha256 233bb102ca165329099a2fa0721ceae82ae1ebfb4755e24848e54df1868739d9);
    * LSHADE_CEC2014.zip inside Top-Methods-Part-A.rar of the official CEC2014
      repository, https://github.com/P-N-Suganthan/CEC2014 (branch master);
    * the extracted file itself (--cec14-src).

It reads the CEC2014 input data from cec_native/input_data_2014 of this package,
which is identical to the competition data at D=30 and 50.

The two edits (nothing else in the release is changed)
------------------------------------------------------
1. The evaluation call: ``test_func`` (CEC2013) becomes ``cec14_test_func`` in
   de.h and search_algorithm.cc.
2. The optimum: the CEC2013 table of optima in search_algorithm.cc is replaced
   by the CEC2014 rule F_i* = 100 i.

The driver is this package's harness_main.cc (N = 100, H = 100, |A| = N, as in
the release's de_test.cc; 10,000*D evaluations).  Run r of function f at
dimension D is seeded with srand(100000*D + 1000*f + r + 1).  The C++ code sets
errors below 1e-8 to 0 itself; ``error`` in the CSV applies the same rule to
the printed value, and ``raw_error`` is the printed value.  The code counts its
own evaluations and returns the best of the first 10,000*D; the CSV therefore
has no ``fes_used`` column.

Usage
-----
    python3 scripts/shade101_cpp/run_shade101.py --shade-zip SHADE1.0.1_CEC2013.zip \\
        (--lshade-zip LSHADE1.0.0_CEC2014.zip | --cec14-src cec14_test_func.cc) \\
        [--dims 30 50] [--runs 51] [--workers 40] [--out results/validation_shade101_cpp_cec2014.csv]
    # --check compares the new runs with the released CSV instead of overwriting it
Requires g++ (the released runs: g++ 11.4.0, -O3, Ubuntu 22.04, x86-64).
"""
import argparse
import hashlib
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.normpath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(PKG, "cec_native", "input_data_2014")
OUT = os.path.join(PKG, "results", "validation_shade101_cpp_cec2014.csv")
SHA_SHADE_ZIP = "92fafa12f90149f5df00402770b9ddb3961a922bfcef3fd7579e2d7440737d86"
SHA_LSHADE_ZIP = "233bb102ca165329099a2fa0721ceae82ae1ebfb4755e24848e54df1868739d9"
SHA_CEC14 = "eb7da784d9e027c02c333fb0ca4000be829287a45e781714f71fd567c5584be6"
ALGO = "SHADE-1.0.1-C++"
TAG = "validation_shade101_cpp"
SEED_RULE = "100000*dim+1000*func+run+1"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _replace_once(text, old, new, what):
    if text.count(old) != 1:
        raise SystemExit(f"edit '{what}': expected exactly one occurrence, found {text.count(old)}")
    return text.replace(old, new)


def prepare(args, build):
    with open(args.shade_zip, "rb") as fh:
        raw = fh.read()
    if _sha(raw) != SHA_SHADE_ZIP:
        raise SystemExit(f"{args.shade_zip}: sha256 {_sha(raw)} is not the release this "
                         f"check used ({SHA_SHADE_ZIP})")
    z = zipfile.ZipFile(io.BytesIO(raw))
    src = {n: z.read(f"SHADE1.0.1_CEC2013/{n}").decode("utf-8")
           for n in ("de.h", "search_algorithm.cc", "shade.cc")}
    if args.cec14_src:
        cec14 = open(args.cec14_src, "rb").read()
    else:
        with open(args.lshade_zip, "rb") as fh:
            lz = fh.read()
        if _sha(lz) != SHA_LSHADE_ZIP:
            raise SystemExit(f"{args.lshade_zip}: sha256 {_sha(lz)} is not {SHA_LSHADE_ZIP}")
        cec14 = zipfile.ZipFile(io.BytesIO(lz)).read("LSHADE_CEC14/cec14_test_func.cc")
    if _sha(cec14) != SHA_CEC14:
        raise SystemExit(f"cec14_test_func.cc: sha256 {_sha(cec14)} is not {SHA_CEC14}")
    # edit 1: the evaluation call
    src["de.h"] = _replace_once(src["de.h"], "void test_func(double *, double *,int,int,int);",
                                "void cec14_test_func(double *, double *,int,int,int);",
                                "declaration")
    src["search_algorithm.cc"] = _replace_once(
        src["search_algorithm.cc"], "    test_func(pop[i],", "    cec14_test_func(pop[i],", "call")
    # edit 2: the optimum, F_i* = 100 i on CEC2014
    pat = re.compile(r"  //set optimal value\n  switch\(function_number\) \{\n.*?\n  \}\n", re.S)
    if len(pat.findall(src["search_algorithm.cc"])) != 1:
        raise SystemExit("edit 'optimum': the CEC2013 table of optima was not found exactly once")
    src["search_algorithm.cc"] = pat.sub(
        "  //CEC2014: F_i* = 100 i  (edit for this check)\n  optimum = function_number * 100;\n\n",
        src["search_algorithm.cc"])
    os.makedirs(build, exist_ok=True)
    for n, t in src.items():
        with open(os.path.join(build, n), "w", encoding="utf-8") as fh:
            fh.write(t)
    with open(os.path.join(build, "cec14_test_func.cc"), "wb") as fh:
        fh.write(cec14)
    shutil.copy(os.path.join(HERE, "harness_main.cc"), os.path.join(build, "main.cc"))
    link = os.path.join(build, "input_data")
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(DATA, link)
    cmd = ["g++", "-O3", "-o", "shade101", "main.cc", "search_algorithm.cc", "shade.cc",
           "cec14_test_func.cc", "-lm"]
    subprocess.run(cmd, cwd=build, check=True)
    print(f"built {os.path.join(build, 'shade101')} ({' '.join(cmd)})")


def run_all(args, build):
    exe = os.path.join(build, "shade101")
    jobs = [(f, d, r) for d in args.dims for f in range(30, 0, -1) for r in range(args.runs)]

    def one(job):
        f, d, r = job
        seed = 100000 * d + 1000 * f + r + 1
        out = subprocess.run([exe, str(f), str(d), str(r), str(seed)], cwd=build, check=True,
                             capture_output=True, text=True).stdout.split()
        assert [int(out[0]), int(out[1]), int(out[2])] == [f, d, r], out
        raw = float(out[3])
        return dict(tag=TAG, suite="cec2014", func=f, dim=d, algo=ALGO, run=r, seed=seed,
                    seed_rule=SEED_RULE, max_fes=10000 * d,
                    error=0.0 if raw < 1e-8 else raw, raw_error=raw)

    with ThreadPoolExecutor(args.workers) as ex:
        rows = list(ex.map(one, jobs))
    return pd.DataFrame(rows).sort_values(["dim", "func", "run"]).reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shade-zip", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--lshade-zip")
    g.add_argument("--cec14-src")
    ap.add_argument("--dims", type=int, nargs="+", default=[30, 50])
    ap.add_argument("--runs", type=int, default=51)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    ap.add_argument("--build-dir", default=os.path.join(PKG, "build", "shade101"))
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--check", action="store_true",
                    help="compare with the released CSV instead of writing it")
    args = ap.parse_args(argv)
    prepare(args, args.build_dir)
    new = run_all(args, args.build_dir)
    if args.check:
        old = pd.read_csv(OUT, float_precision="round_trip")
        old = old[old.dim.isin(args.dims) & (old.run < args.runs)]
        old = old.sort_values(["dim", "func", "run"]).reset_index(drop=True)
        same = (len(old) == len(new) and np.array_equal(old.seed, new.seed)
                and np.array_equal(old.raw_error.to_numpy(), new.raw_error.to_numpy()))
        print(f"{len(new)} runs re-made; identical to the released CSV: {same}")
        return 0 if same else 1
    new.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(new)} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
