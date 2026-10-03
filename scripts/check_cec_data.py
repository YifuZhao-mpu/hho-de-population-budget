"""Two facts about the released CEC input data, made auditable (W6, W7).

1. Singular values of the transformation matrices (REV-44).  For each suite
   and each dimension the paper uses (CEC2014: 30, 50; CEC2017: 30, 50, 100;
   CEC2022: 10, 20), every d x d block of every M_<f>_D<d>.txt file is
   decomposed, exactly as validate_benchmarks.py (section 3b) reads them.
   Counted: blocks whose largest singular value is 2 (to 1e-9), blocks that
   are orthogonal (all singular values 1 to 1e-9), and any other kind.

2. Lineage of CEC2022 (REV-37).  Every CEC2022 shift vector, rotation matrix
   and shuffle (permutation) file is compared with every file of the same kind
   in the CEC2017 and CEC2014 input data.  A shift file matches when its first
   row equals the other file's first row in all 100 entries; the number of
   further rows that also match is reported.  A matrix or shuffle file matches
   when every number in it is equal.  Only data are compared: whether two
   functions share a *recipe* (the same basic functions in the same order and
   proportions) is a property of the C sources and is not tested here.

3. CEC2014 against CEC2017: the same comparison of shift vectors and of the
   rotation matrices at D = 30 and 50 (the two suites share basic functions
   in their C code; this checks whether they also share data).

Matrices are split into d x d blocks exactly as validate_benchmarks.py does,
so a composition function's file contributes every block it contains,
including blocks beyond the components the function uses.

Writes results/analysis/revision_singular_values.csv and
results/analysis/revision_cec2022_lineage.csv.

Run:  python3 scripts/check_cec_data.py
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.join(HERE, "..", "cec_native")
OUT = os.path.join(HERE, "..", "results", "analysis")
SUITES = {"2014": ((30, 50), range(1, 31)), "2017": ((30, 50, 100), range(1, 31)),
          "2022": ((10, 20), range(1, 13))}
TOL = 1e-9


def _numbers(path):
    with open(path) as fh:
        return np.array([float(t) for t in fh.read().split()])


def _rows(path):
    with open(path) as fh:
        return [np.array([float(t) for t in line.split()]) for line in fh if line.strip()]


def singular_values():
    rows = []
    for suite, (dims, funcs) in SUITES.items():
        for d in dims:
            for fn in funcs:
                p = os.path.join(NATIVE, f"input_data_{suite}", f"M_{fn}_D{d}.txt")
                if not os.path.exists(p):
                    continue
                raw = _numbers(p)
                for b in range(raw.size // (d * d)):
                    M = raw[b * d * d:(b + 1) * d * d].reshape(d, d)
                    sv = np.linalg.svd(M, compute_uv=False)
                    kind = ("max 2" if abs(sv.max() - 2.0) < TOL else
                            "orthogonal" if np.all(np.abs(sv - 1.0) < TOL) else "other")
                    rows.append(dict(suite=f"CEC{suite}", dim=d, func=fn, block=b,
                                     sv_max=float(sv.max()), sv_min=float(sv.min()), kind=kind,
                                     finite=bool(np.all(np.isfinite(M)))))
    return pd.DataFrame(rows)


def lineage():
    rows = []
    d22 = os.path.join(NATIVE, "input_data_2022")
    others = {s: os.path.join(NATIVE, f"input_data_{s}") for s in ("2017", "2014")}
    for fn in range(1, 13):
        # shift vectors
        s22 = _rows(os.path.join(d22, f"shift_data_{fn}.txt"))
        for s, dd in others.items():
            for g in range(1, 31):
                p = os.path.join(dd, f"shift_data_{g}.txt")
                if not os.path.exists(p):
                    continue
                so = _rows(p)
                if len(so[0]) == len(s22[0]) and np.array_equal(so[0], s22[0]):
                    n = sum(1 for a, b in zip(s22, so) if np.array_equal(a, b))
                    rows.append(dict(cec2022_func=fn, kind="shift vector", dim="all",
                                     other_suite=f"CEC{s}", other_func=g,
                                     detail=f"first row identical; {n} of {len(s22)} rows identical"))
        # rotation matrices and shuffles at the CEC2022 dimensions
        for d in (10, 20):
            for kind, stem in (("rotation matrix", "M"), ("shuffle", "shuffle_data")):
                p22 = os.path.join(d22, f"{stem}_{fn}_D{d}.txt")
                if not os.path.exists(p22):
                    continue
                m22 = _numbers(p22)
                for s, dd in others.items():
                    for g in range(1, 31):
                        p = os.path.join(dd, f"{stem}_{g}_D{d}.txt")
                        if not os.path.exists(p):
                            continue
                        mo = _numbers(p)
                        if mo.size == m22.size and np.array_equal(mo, m22):
                            rows.append(dict(cec2022_func=fn, kind=kind, dim=d,
                                             other_suite=f"CEC{s}", other_func=g,
                                             detail="identical file contents"))
                        elif mo.size > m22.size and np.array_equal(mo[:m22.size], m22):
                            rows.append(dict(cec2022_func=fn, kind=kind, dim=d,
                                             other_suite=f"CEC{s}", other_func=g,
                                             detail="CEC2022 file is the leading part of the other"))
    return pd.DataFrame(rows)


def cec2014_vs_cec2017():
    d14 = os.path.join(NATIVE, "input_data_2014")
    d17 = os.path.join(NATIVE, "input_data_2017")
    found = []
    for f in range(1, 31):
        a = _rows(os.path.join(d14, f"shift_data_{f}.txt"))[0]
        for g in range(1, 31):
            b = _rows(os.path.join(d17, f"shift_data_{g}.txt"))[0]
            if a.size == b.size and np.array_equal(a, b):
                found.append(("shift vector", f, g, "all"))
            for d in (30, 50):
                pa = os.path.join(d14, f"M_{f}_D{d}.txt")
                pb = os.path.join(d17, f"M_{g}_D{d}.txt")
                if os.path.exists(pa) and os.path.exists(pb):
                    A, B = _numbers(pa), _numbers(pb)
                    if A.size == B.size and np.array_equal(A, B):
                        found.append(("rotation matrix", f, g, d))
    return found


def main():
    os.makedirs(OUT, exist_ok=True)
    sv = singular_values()
    sv.to_csv(os.path.join(OUT, "revision_singular_values.csv"), index=False)
    print("Transformation matrices: largest singular value (dimensions used in the paper)")
    for suite, s in sv.groupby("suite"):
        k = s.kind.value_counts()
        others = s[s.kind != "max 2"]
        print(f"  {suite}: {len(s)} matrices; largest singular value 2 in {k.get('max 2', 0)}, "
              f"orthogonal (all singular values 1) {k.get('orthogonal', 0)}, other "
              f"{k.get('other', 0)}; max over the suite {s.sv_max.max():.12g}; "
              f"max among the rest {others.sv_max.max() if len(others) else float('nan'):.12g}; "
              f"all finite {bool(s.finite.all())}")
        for d, t in s.groupby("dim"):
            kk = t.kind.value_counts()
            print(f"      D={d}: {len(t)} matrices, max 2 in {kk.get('max 2', 0)}, orthogonal "
                  f"{kk.get('orthogonal', 0)}")
    lin = lineage()
    lin.to_csv(os.path.join(OUT, "revision_cec2022_lineage.csv"), index=False)
    print("\nCEC2022 data found in CEC2017 / CEC2014 input files")
    for fn in range(1, 13):
        s = lin[lin.cec2022_func == fn]
        parts = []
        for (kind, other), t in s.groupby(["kind", "other_suite"], sort=False):
            parts.append(f"{kind} = {other} " + ", ".join(
                sorted({f"F{g}" + (f" (D={d})" if d != 'all' else "") for g, d in
                        zip(t.other_func, t.dim)})))
        print(f"  F{fn:<2d}: " + ("; ".join(parts) if parts else "no identical data found"))
    shifts17 = lin[(lin.kind == "shift vector") & (lin.other_suite == "CEC2017")]
    print(f"\n  shift vectors identical to a CEC2017 function's: "
          f"{shifts17.cec2022_func.nunique()} of 12 CEC2022 functions")
    shared = cec2014_vs_cec2017()
    print(f"\nCEC2014 against CEC2017: {len(shared)} identical shift vectors or rotation "
          f"matrices (D = 30, 50)" + (": " + ", ".join(f"{k} F{f}/F{g} D={d}" for k, f, g, d in shared)
                                       if shared else ""))
    print(f"wrote {os.path.join(OUT, 'revision_singular_values.csv')}")
    print(f"wrote {os.path.join(OUT, 'revision_cec2022_lineage.csv')}")


if __name__ == "__main__":
    main()
