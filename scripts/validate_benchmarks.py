"""Validate the compiled competition suites against independent references.

1. CEC2022 native  vs  the official Python transliteration shipped by the
   competition organisers (exact agreement expected).
2. CEC2017 native  vs  cec2017-py (Tilley), an independent MIT-licensed port of
   the same C code; disagreements are reported per function.
3. Optimum recovery: f(o) - f* should be ~0 for every function and dimension.
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sehho.bench import CECProblem, CEC2014_FUNCS, CEC2017_FUNCS, CEC2022_FUNCS

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.join(os.path.dirname(HERE), "cec_native")

def rel(a, b):
    return np.max(np.abs(a - b) / np.maximum(1.0, np.abs(b)))

# ---------------- 1. CEC2022 vs official Python reference ----------------
print("=" * 70)
print("1. CEC2022 native  vs  official Python reference implementation")
sys.path.insert(0, os.path.join(NATIVE, "reference"))
cwd = os.getcwd(); os.chdir(NATIVE)
os.makedirs("input_data", exist_ok=True)
# the reference reads 'input_data/...' relative to cwd
import shutil
if not os.path.exists("input_data/M_1_D10.txt"):
    for f in os.listdir("input_data_2022"):
        shutil.copy(os.path.join("input_data_2022", f), os.path.join("input_data", f))
import official_cec2022_reference as ref
rng = np.random.default_rng(12345)
worst = 0.0
for d in (10, 20):
    for fn in CEC2022_FUNCS:
        X = rng.uniform(-100, 100, (8, d))
        mine = CECProblem("cec2022", fn, d)(X)
        theirs = np.array([ref.cec22_test_func(X[i].copy(), d, 1, fn)[0] for i in range(X.shape[0])])
        r = rel(mine, theirs)
        worst = max(worst, r)
        flag = "ok " if r < 1e-12 else "DIFF"
        print(f"   [{flag}] D={d:3d} F{fn:<2d}  max rel diff = {r:.3e}")
os.chdir(cwd)
print(f"   -> worst relative difference over all CEC2022 checks: {worst:.3e}")

# ---------------- 2. CEC2017 vs cec2017-py ----------------
print("=" * 70)
print("2. CEC2017 native  vs  cec2017-py (independent MIT port)")
try:
    ref = os.environ.get("CEC2017_PY_DIR")  # folder holding the cec2017-py package, if it is not installed
    if ref:
        sys.path.insert(0, ref)
    import warnings; warnings.filterwarnings("ignore")
    from cec2017 import functions as TF
    allf = TF.all_functions
    for d in (10, 30, 50, 100):
        diffs = []
        for fn in CEC2017_FUNCS:
            X = rng.uniform(-100, 100, (6, d))
            mine = CECProblem("cec2017", fn, d)(X)
            theirs = allf[fn - 1](X)
            diffs.append((fn, rel(mine, theirs)))
        bad = [(f, r) for f, r in diffs if r > 1e-10]
        print(f"   D={d:3d}: {len(diffs)-len(bad):2d}/{len(diffs)} agree to 1e-10; "
              f"disagreeing: {[f'F{f}({r:.1e})' for f, r in bad]}")
except Exception as e:
    print("   (skipped:", e, ")")

# ---------------- 3. optimum recovery ----------------
print("=" * 70)
print("3. Optimum recovery  f(o) - f*   (official shift vectors)")
def load_shift(suite, fn, d, comp):
    p = os.path.join(NATIVE, f"input_data_{suite}", f"shift_data_{fn}.txt")
    raw = open(p).read().split()
    v = np.array([float(t) for t in raw])
    return v[:d] if not comp else v[:d]
for suite, funcs, dims, comp_start in (("2014", CEC2014_FUNCS, (30, 50), 23),
                                       ("2017", CEC2017_FUNCS, (10, 30, 50, 100), 21),
                                       ("2022", CEC2022_FUNCS, (10, 20), 9)):
    worst_f, worst_v = None, 0.0
    for d in dims:
        for fn in funcs:
            p = CECProblem(f"cec{suite}", fn, d)
            o = load_shift(suite, fn, d, fn >= comp_start)[None, :]
            err = float(p(o)[0] - p.f_star)
            if abs(err) > abs(worst_v):
                worst_v, worst_f = err, f"F{fn} D={d}"
    print(f"   CEC{suite}: largest |f(o) - f*| = {abs(worst_v):.3e}  at {worst_f}")

# ---------------- 3b. transformation matrices ----------------
# The optimum-recovery check above passes even if the transformation matrices
# are garbage, because z = M(x-o) vanishes at x=o whatever M is.  Parsing of M
# is therefore tested separately.
#
# Most of these are not pure rotations: in most matrices a rotation is composed
# with a diagonal scaling and the largest singular value is exactly 2; some are
# orthogonal (all singular values 1); and 40 blocks of the CEC2022 composition
# functions F9-F12 at D=10 lie in between (largest singular value 1.28-1.98;
# per-suite counts in scripts/check_cec_data.py).  The useful invariant is that
# the maximum over each suite is exactly 2: it is identical across all three
# suites, and a matrix read with the wrong format specifier does not reproduce it.
print("=" * 70)
print("3b. Transformation matrices: singular values (max must be exactly 2)")
for suite, dims in (("2014", (30, 50)), ("2017", (30, 50, 100)), ("2022", (10, 20))):
    lo, hi, n, finite = 9e9, -9e9, 0, True
    for d in dims:
        for fn in range(1, 31):
            fp = os.path.join(NATIVE, f"input_data_{suite}", f"M_{fn}_D{d}.txt")
            if not os.path.exists(fp):
                continue
            raw = np.array([float(t) for t in open(fp).read().split()])
            finite = finite and bool(np.all(np.isfinite(raw)))
            for b0 in range(raw.size // (d * d)):
                M = raw[b0 * d * d:(b0 + 1) * d * d].reshape(d, d)
                sv = np.linalg.svd(M, compute_uv=False)
                lo, hi, n = min(lo, sv.min()), max(hi, sv.max()), n + 1
    ok = "OK" if (abs(hi - 2.0) < 1e-9 and finite) else "FAIL"
    print(f"   [{ok}] CEC{suite}: {n} matrices, singular values in "
          f"[{lo:.6f}, {hi:.6f}], all finite = {finite}")

# ---------------- 4. speed ----------------
print("=" * 70)
print("4. Throughput (batch of 30 candidates)")
for suite, fns, dims in (("cec2022", [1, 6, 12], [10, 20]), ("cec2017", [1, 15, 30], [30, 100]),
                         ("cec2014", [1, 15, 30], [30, 50])):
    for d in dims:
        tot = 0.0
        for fn in fns:
            p = CECProblem(suite, fn, d)
            X = rng.uniform(-100, 100, (30, d))
            p(X)
            t0 = time.perf_counter()
            for _ in range(200):
                p(X)
            tot += (time.perf_counter() - t0) / 200 / 30
        print(f"   {suite} D={d:3d}: {tot/len(fns)*1e6:6.2f} us per evaluation")
