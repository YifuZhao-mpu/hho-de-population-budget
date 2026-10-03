"""Classical reference designs for the five constrained design problems (W5, REV-53).

A metaheuristic's result on a design problem means little without the value a
classical method reaches, and at what cost.  This script computes one
classical reference per problem with the problem code the experiments use
(``sehho/engineering.py``), so every reference is evaluated under the same
repair, the same feasibility rule and the same tolerances as Table 7:

* P1 pressure vessel: the shell and head thicknesses take values on the
  0.0625-in grid.  Every grid pair (x1, x2) is enumerated; a pair is skipped
  only when an analytic lower bound on the cost (below) already exceeds the
  incumbent; for every other pair the two continuous variables (x3, x4) are
  solved by SLSQP from a start on the active volume constraint.
* P2 gear train: full enumeration of the 49^4 integer tooth combinations.
* P3 RC01: the closed-form equality elimination of ``rc01_analytic.py`` (0
  evaluations) and SciPy trust-constr (stage E5c); read from
  ``results/E5c_solver_baseline.csv``, nothing is recomputed.
* P4 RC06: SciPy SLSQP (stage E5c), read from the same file.
* P5 RC14: the 27 integer assignments of parallel units (N1, N2, N3 in
  {1, 2, 3}) are enumerated; for each, a multi-start SLSQP over the seven
  continuous variables.

Evaluation accounting follows ``run_eng_solver_baseline.py``: one evaluation
per distinct decision vector (the objective and all constraints at one point
cost 1; a finite-difference gradient costs about D + 1).

The P1 lower bound: every term of the cost is increasing in x3 and x4, and
any feasible design has x4 <= 200 (bound) and therefore, from the volume
constraint, x3 >= r_min, where pi r^2 200 + 4/3 pi r^3 = 1,296,000; with
x4 >= 10 this gives
    f >= 0.6224 x1 r_min 10 + 1.7781 x2 r_min^2 + 3.1661 x1^2 10 + 19.84 x1^2 r_min,
which is valid for every design with thicknesses (x1, x2).

Writes results/analysis/revision_classical_refs.csv.

Run:  python3 scripts/classical_references.py
"""

import itertools
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sehho.engineering import ENGINEERING_PROBLEMS

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
OUT = os.path.join(RES, "analysis", "revision_classical_refs.csv")
SEED = 20261003


class Counter:
    """One evaluation per distinct decision vector (consecutive repeats are free)."""

    def __init__(self, problem):
        self.p = problem
        self.used = 0
        self._key = None
        self._val = None

    def parts(self, x):
        x = np.asarray(x, dtype=float)
        k = x.tobytes()
        if k != self._key:
            self._key = k
            f, v, w = self.p.parts(x[None, :])
            G, H = self.p.constraints(self.p.repair(x[None, :]))
            self._val = (float(f[0]), float(v[0]), G, H)
            self.used += 1
        return self._val


# ------------------------------------------------------------------ P1
def pressure_vessel():
    p = ENGINEERING_PROBLEMS["P1_pressure_vessel"]()
    c = Counter(p)
    grid = np.arange(1, 100) * 0.0625                 # 0.0625 .. 6.1875
    r_min = brentq(lambda r: np.pi * r * r * 200 + 4 / 3 * np.pi * r ** 3 - 1_296_000, 1, 200)
    best = (np.inf, None)
    solved, pruned = 0, 0

    def nlp(x1, x2):
        # x3 <= min(x1/0.0193, x2/0.00954, 200); the volume constraint binds at
        # the optimum, so start on it
        x3_hi = min(x1 / 0.0193, x2 / 0.00954, 200.0)
        if x3_hi < r_min:
            return None
        x30 = max(r_min, min(x3_hi, 0.98 * x3_hi))
        x40 = (1_296_000 - 4 / 3 * np.pi * x30 ** 3) / (np.pi * x30 ** 2)
        x40 = float(np.clip(x40 * 1.0001, 10, 200))
        f = lambda z: c.parts([x1, x2, z[0], z[1]])[0]
        g = lambda z: -c.parts([x1, x2, z[0], z[1]])[2][0]     # SLSQP wants g >= 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = minimize(f, [x30, x40], method="SLSQP", bounds=[(10, 200), (10, 200)],
                         constraints=[{"type": "ineq", "fun": g}],
                         options={"maxiter": 200, "ftol": 1e-12})
        # SLSQP stops within its own tolerance of the volume constraint, which
        # is of order 1e6, so the 1e-8 absolute feasibility threshold is not
        # met.  Polish: keep x3 inside its thickness limits and raise x4 to the
        # value on which the volume constraint holds (one extra evaluation).
        x3 = float(min(max(r.x[0], 10.0), x3_hi))
        x4 = (1_296_000 - 4 / 3 * np.pi * x3 ** 3) / (np.pi * x3 ** 2)
        x4 = float(np.clip(np.nextafter(x4, np.inf), 10, 200))
        x = np.array([x1, x2, x3, x4])
        c.parts(x)
        rep = p.report(x)
        return x, rep

    for x1, x2 in itertools.product(grid, grid):
        lb = (0.6224 * x1 * r_min * 10 + 1.7781 * x2 * r_min ** 2
              + 3.1661 * x1 ** 2 * 10 + 19.84 * x1 ** 2 * r_min)
        if lb >= best[0]:
            pruned += 1
            continue
        out = nlp(x1, x2)
        if out is None:
            pruned += 1
            continue
        solved += 1
        x, rep = out
        if rep["feasible"] and rep["objective"] < best[0]:
            best = (rep["objective"], x)
    x = best[1]
    rep = p.report(x)
    return dict(problem="P1_pressure_vessel", method="thickness grid + SLSQP on (x3, x4)",
                objective=rep["objective"], feasible=rep["feasible"],
                total_violation=rep["total_violation"], evaluations=c.used,
                x=" ".join(f"{v:.10g}" for v in x),
                detail=(f"{len(grid)}x{len(grid)} thickness grid; {solved} pairs solved by "
                        f"SLSQP, {pruned} skipped by the analytic lower bound or because no "
                        f"feasible (x3, x4) exists (r_min = {r_min:.4f})"))


# ------------------------------------------------------------------ P2
def gear_train():
    p = ENGINEERING_PROBLEMS["P2_gear_train"]()
    v = np.arange(12, 61, dtype=float)
    X = np.array(np.meshgrid(v, v, v, v, indexing="ij")).reshape(4, -1).T
    f = np.asarray(p.objective(X))
    fmin = float(f.min())
    n_opt = int(np.sum(f == fmin))
    x = X[int(np.argmin(f))]
    rep = p.report(x)
    return dict(problem="P2_gear_train", method="full enumeration of 49^4 tooth combinations",
                objective=rep["objective"], feasible=rep["feasible"],
                total_violation=rep["total_violation"], evaluations=int(len(X)),
                x=" ".join(f"{v:.10g}" for v in x),
                detail=f"{n_opt} of {len(X):,} combinations attain the minimum")


# ------------------------------------------------------------------ RC14
def batch_plant(starts=3):
    p = ENGINEERING_PROBLEMS["P5_batch_plant_RC14"]()
    c = Counter(p)
    rng = np.random.default_rng(SEED)
    lb, ub = p.lb[3:], p.ub[3:]
    rows = []
    for N in itertools.product((1.0, 2.0, 3.0), repeat=3):
        best = (np.inf, None)
        for _ in range(starts):
            z0 = lb + rng.uniform(0, 1, 7) * (ub - lb)
            f = lambda z: c.parts(np.r_[N, z])[0]
            g = lambda z: -np.asarray(c.parts(np.r_[N, z])[2][0])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    r = minimize(f, z0, method="SLSQP", bounds=list(zip(lb, ub)),
                                 constraints=[{"type": "ineq", "fun": g}],
                                 options={"maxiter": 300, "ftol": 1e-12})
                    z = r.x
                except (ValueError, np.linalg.LinAlgError):
                    continue
            rep = p.report(np.r_[N, z])
            if rep["feasible"] and rep["objective"] < best[0]:
                best = (rep["objective"], np.r_[N, z])
        rows.append((N, best))
    feas = [(N, b) for N, b in rows if b[1] is not None]
    N_best, (fbest, xbest) = min(feas, key=lambda t: t[1][0])
    rep = p.report(xbest)
    second = sorted(b[0] for N, b in feas if N[0] == 2.0)
    per_n = "; ".join(f"N={tuple(int(v) for v in N)}: {b[0]:.2f}" for N, b in
                      sorted(feas, key=lambda t: t[1][0])[:4])
    return dict(problem="P5_batch_plant_RC14",
                method=f"27 integer unit assignments x {starts}-start SLSQP",
                objective=rep["objective"], feasible=rep["feasible"],
                total_violation=rep["total_violation"], evaluations=c.used,
                x=" ".join(f"{v:.10g}" for v in xbest),
                detail=(f"best assignments: {per_n}; best with two units in stage 1: "
                        f"{second[0]:.2f}" if second else per_n))


# ------------------------------------------------------------------ RC01, RC06
def from_e5c():
    d = pd.read_csv(os.path.join(RES, "E5c_solver_baseline.csv"))
    out = []
    for prob, algo, method in (
            ("P3_heat_exchanger_RC01", "Equality elimination (closed form)",
             "equality elimination, closed form (rc01_analytic.py)"),
            ("P3_heat_exchanger_RC01", "trust-constr", "SciPy trust-constr, multi-start (E5c)"),
            ("P4_blending_pooling_RC06", "SLSQP", "SciPy SLSQP, multi-start (E5c)")):
        s = d[(d.problem == prob) & (d.algo == algo)]
        f = s[s.feasible]
        best = f.loc[f.objective.idxmin()]
        x = ("0 16.66666667 0 0 2000000 600 100 600 700"
             if "closed" in algo else "")
        out.append(dict(problem=prob, method=method, objective=float(best.objective),
                        feasible=True, total_violation=float(best.total_violation),
                        evaluations=int(best.fes_used), x=x,
                        detail=(f"{int(s.feasible.sum())}/{len(s)} runs feasible; evaluations "
                                f"per run" if len(s) > 1 else "closed form, x* = (0, 50/3, 0, "
                                "0, 2e6, 600, 100, 600, 700)")))
    return out


def main():
    rows = from_e5c()
    for fn in (pressure_vessel, gear_train, batch_plant):
        r = fn()
        rows.append(r)
    df = pd.DataFrame(rows).sort_values(["problem", "evaluations"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 120)
    for _, r in df.iterrows():
        print(f"{r.problem:<26s} {r.method:<52s} f = {r.objective:.10g}  feasible={r.feasible}  "
              f"v={r.total_violation:.2e}  evaluations={r.evaluations:,}")
        print(f"{'':<26s} x = {r.x}")
        print(f"{'':<26s} {r.detail}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
