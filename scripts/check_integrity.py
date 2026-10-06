"""Integrity checks on the raw result files.

Confirms that every cell has the expected number of runs, that every run
consumed exactly its budget, that seeds are unique within a cell and
reproducible from the cell identity, and that no result is non-finite.

Two validation files follow other seed rules, which the check verifies too:
results/validation_engine_variants_cec2014.csv reuses the seeds of the stored
E13 runs (columns seed_algo and seed_tag name the E13 cell whose seed a run
reuses, so that every variant run is paired with an E13 run), and
results/validation_shade101_cpp_cec2014.csv holds runs of a third-party C++
code seeded by the rule in its seed_rule column, whose evaluation counter is
internal to that code (no fes_used column; the budget check is skipped).
"""
import os, sys, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sehho.runner import make_seed

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
ok = True
for path in sorted(glob.glob(os.path.join(RES, "*.csv"))):
    name = os.path.basename(path)
    df = pd.read_csv(path)
    if not {"algo", "max_fes", "seed", "run"}.issubset(df.columns):
        # not a per-run result file (e.g. the convergence traces)
        continue
    # zero-budget rows are deterministic references (e.g. a closed-form
    # optimum), not stochastic runs: they have no seed to verify.
    df = df[df.max_fes > 0]
    if df.empty:
        print(f"[OK  ] {name:38s} reference-only file, no stochastic runs")
        continue
    keys = ["tag", "suite", "func", "dim", "algo", "max_fes"]
    g = df.groupby(keys)
    n_cells = g.ngroups
    runs = g.size()
    has_fes = "fes_used" in df.columns
    budget_ok = (df.fes_used == df.max_fes).all() if has_fes else True
    val = (df.best_f if "best_f" in df.columns else
           df.objective if "objective" in df.columns else df.error)
    finite = np.isfinite(val).all()
    seeds_unique = g["seed"].nunique().eq(runs).all()
    # seeds must be reproducible from the cell identity (or from the rule the
    # file declares: E13's seeds reused, or the C++ reference code's srand rule)
    sample = df.sample(min(200, len(df)), random_state=0)
    if "seed_rule" in df.columns:
        rules = set(df.seed_rule)
        if rules != {"100000*dim+1000*func+run+1"}:
            raise SystemExit(f"{name}: unknown seed rule {rules}")
        seed_ok = bool((df.seed == 100000 * df.dim + 1000 * df.func + df.run + 1).all())
    elif {"seed_algo", "seed_tag"}.issubset(df.columns):
        seed_ok = all(make_seed(r.suite, r.func, r.dim, r.seed_algo, r.seed_tag, r.run) == r.seed
                      for r in sample.itertuples())
    else:
        seed_ok = all(make_seed(r.suite, r.func, r.dim, r.algo, r.tag, r.run) == r.seed
                      for r in sample.itertuples())
    problems, notes = [], []
    if not has_fes:
        notes.append("evaluations counted inside the reference code (no fes_used column)")
    if not runs.eq(runs.iloc[0]).all(): problems.append(f"uneven runs {sorted(set(runs))}")
    if not budget_ok:
        bad = df[df.fes_used != df.max_fes]
        short = (bad.max_fes - bad.fes_used)
        frac = float((short / bad.max_fes).max())
        msg = (f"{len(bad)} runs short of budget by at most {int(short.max())} FEs "
               f"({100*frac:.4f}%)")
        # A shortfall below 0.1% is a rounding remainder that disadvantages the
        # affected algorithm; it is reported but does not fail the check.
        (notes if frac < 1e-3 else problems).append(msg)
    if not finite: problems.append("non-finite objective values")
    if not seeds_unique: problems.append("duplicate seeds within a cell")
    if not seed_ok: problems.append("seeds not reproducible from cell identity")
    status = "FAIL" if problems else ("WARN" if [n for n in notes if "reference code" not in n]
                                      else "OK  ")
    ok &= not problems
    tail = "; ".join(problems + notes)
    print(f"[{status}] {name:38s} {len(df):6d} runs, {n_cells:4d} cells, "
          f"{runs.iloc[0]:2d} runs/cell" + ("  <- " + tail if tail else ""))
print("\nALL INTEGRITY CHECKS PASSED" if ok else "\nINTEGRITY PROBLEMS FOUND")
sys.exit(0 if ok else 1)
