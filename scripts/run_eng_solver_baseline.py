"""Dedicated constrained-solver baselines for the two process problems.

A comparison among general-purpose metaheuristics cannot establish that a
problem is hard, only that those particular heuristics find it hard. This
script adds the two references such a claim actually needs:

1. **Equality elimination** (RC01 only): the closed-form reduction of
   `scripts/rc01_analytic.py`, which costs zero function evaluations.
2. **A dedicated constrained solver**: SciPy's SLSQP and trust-constr driven
   from random multi-starts, given the same 15,000-evaluation budget the
   metaheuristics receive, with the equality constraints handed to the solver
   explicitly instead of being folded into a feasibility rule.

Evaluation accounting is deliberately conservative towards the solver: one
evaluation is charged per *distinct* decision vector, so the objective and the
32 constraint functions queried at the same point cost 1, not 33. Finite-
difference gradients are therefore charged at roughly (n+1) evaluations per
gradient step, as they should be.

Run:  python3 scripts/run_eng_solver_baseline.py
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize, NonlinearConstraint

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sehho.engineering import ENGINEERING_PROBLEMS
from sehho.runner import make_seed

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
BUDGET = 15_000
RUNS = 30
PROBLEMS = ["P3_heat_exchanger_RC01", "P4_blending_pooling_RC06"]
RC01_ANALYTIC = 35.0 * (50.0 / 3.0) ** 0.6


class Counter:
    """Charges one evaluation per distinct decision vector.

    Decisions are mapped to the unit box before being handed to the solver --
    standard practice, and applied in the solver's favour. Equality *rows* are
    deliberately NOT rescaled: dividing a row by its typical magnitude also
    divides its tolerance, which drives the scaled target below the solver's
    own convergence threshold and makes it stop pushing on exactly the rows
    that matter. Unit-box scaling alone was verified to be the stronger
    configuration for both problems.
    """

    def __init__(self, problem, budget):
        self.p = problem
        self.budget = budget
        self.used = 0
        self._key = None
        self._val = None

    def _eval(self, x):
        k = np.asarray(x, dtype=float).tobytes()
        if k != self._key:
            if self.used >= self.budget:
                raise StopIteration
            self._key = k
            self._val = self.p.parts(np.asarray(x, dtype=float)[None, :])
            self.used += 1
        return self._val

    def f(self, x):
        try:
            return float(self._eval(x)[0][0])
        except StopIteration:
            return 1e30

    def eq(self, x):
        """Equality residual vector (target 0)."""
        try:
            self._eval(x)
        except StopIteration:
            return np.zeros(self.p.n_eq)
        _, H = self.p.constraints(self.p.repair(np.asarray(x, float)[None, :]))
        return np.zeros(self.p.n_eq) if H is None else H[0]

    def ineq(self, x):
        try:
            self._eval(x)
        except StopIteration:
            return np.zeros(max(self.p.n_ineq, 1))
        G, _ = self.p.constraints(self.p.repair(np.asarray(x, float)[None, :]))
        return np.zeros(max(self.p.n_ineq, 1)) if G is None else G[0]


def one_run(problem, method, seed, budget=BUDGET):
    """Multi-start `method` in a scaled space until the budget is exhausted."""
    rng = np.random.default_rng(seed)
    c = Counter(problem, budget)
    lb, ub = problem.lb, problem.ub
    span = ub - lb

    def to_x(u):
        return lb + np.clip(np.asarray(u, float), 0.0, 1.0) * span

    fu = lambda u: c.f(to_x(u))
    equ = (lambda u: c.eq(to_x(u))) if problem.n_eq else None
    inqu = (lambda u: c.ineq(to_x(u))) if problem.n_ineq else None

    bounds = [(0.0, 1.0)] * problem.dim
    cons = []
    if problem.n_eq:
        cons.append(NonlinearConstraint(equ, -problem.eps_eq, problem.eps_eq))
    if problem.n_ineq:
        cons.append(NonlinearConstraint(inqu, -np.inf, 0.0))

    best_x, best_key, starts = None, None, 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        while c.used < budget:
            u0 = rng.uniform(0.0, 1.0, problem.dim)
            starts += 1
            try:
                r = minimize(fu, u0, method=method, bounds=bounds,
                             constraints=cons,
                             options={"maxiter": 200} if method == "SLSQP"
                             else {"maxiter": 200, "verbose": 0})
                cand = to_x(r.x)
            except (StopIteration, ValueError, np.linalg.LinAlgError):
                cand = None
            for y in ([cand] if cand is not None else []) + [to_x(u0)]:
                rep = problem.report(y)
                # feasibility rule: feasible beats infeasible, then objective
                key = (0 if rep["feasible"] else 1,
                       rep["objective"] if rep["feasible"] else rep["total_violation"])
                if best_key is None or key < best_key:
                    best_key, best_x = key, np.asarray(y, float).copy()
    rep = problem.report(best_x)
    rep.update(fes_used=c.used, starts=starts)
    return rep


def main():
    rows = []
    for key in PROBLEMS:
        p = ENGINEERING_PROBLEMS[key]()
        for method in ("SLSQP", "trust-constr"):
            t0 = time.perf_counter()
            for r in range(RUNS):
                seed = make_seed("engineering", 0, 0, method, key, r)
                rec = one_run(p, method, seed)
                rec.update(tag=key, suite="engineering", func=0, dim=0,
                           problem=key, algo=method, run=r, seed=seed,
                           max_fes=BUDGET)
                rows.append(rec)
            print(f"  {key:26s} {method:13s} {RUNS} runs in "
                  f"{(time.perf_counter()-t0)/60:5.1f} min", flush=True)
        # equality-elimination reference (RC01 only): exact, zero evaluations
        if key == "P3_heat_exchanger_RC01":
            x = np.array([0.0, 50/3, 0.0, 0.0, 2e6, 600.0, 100.0, 600.0, 700.0])
            rec = p.report(x)
            rec.update(tag=key, suite="engineering", func=0, dim=0, problem=key,
                       algo="Equality elimination (closed form)",
                       run=0, seed=0, max_fes=0, fes_used=0, starts=0)
            rows.append(rec)

    df = pd.DataFrame(rows)
    out = os.path.join(RES, "E5c_solver_baseline.csv")
    df.to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(df)} runs)\n")

    for key in PROBLEMS:
        print(f"### {key}")
        sub = df[df["problem"] == key]
        for a in sub.algo.unique():
            s = sub[sub.algo == a]
            fe = s[s.feasible]
            gap = (f"{100*(fe.objective.min()-RC01_ANALYTIC)/RC01_ANALYTIC:+.2f}%"
                   if len(fe) and key == "P3_heat_exchanger_RC01" else "--")
            print(f"  {a:36s} feasible {100*s.feasible.mean():5.1f}%  "
                  f"best {(f'{fe.objective.min():.4f}' if len(fe) else '--'):>12s}  "
                  f"gap-vs-analytic {gap:>8s}  mean FEs {s.fes_used.mean():.0f}")
        print()


if __name__ == "__main__":
    main()
