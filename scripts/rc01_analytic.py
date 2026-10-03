"""Closed-form reduction of CEC2020 RC01 (heat-exchanger network design).

RC01 is presented in the metaheuristics literature as a hard equality-constrained
process-design problem. It is not. Its feasible set collapses to a two-parameter
family under three lines of algebra, and a closed-form feasible reference design
x* is available. This script derives it and verifies every step numerically
against the same implementation that produced the paper's experimental numbers.

What x* is, and is not: it minimises the objective over the exact-equality
feasible family derived below.  Under the declared equality tolerance
(eps = 1e-4) it is feasible, but it is not a certified optimum of the
tolerance-relaxed problem the experiments actually solve, whose infimum lies
below f(x*) by at least a relative 3e-9 % along the one direction examined
(Supplementary Material); its exact value is not determined.  It is therefore
called the closed-form feasible reference design, and every gap quoted against
it is a lower bound on the true excess.

Run:  python3 scripts/rc01_analytic.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sehho.engineering import HeatExchangerNetwork


def derive():
    """The reduction, step by step.

    Constraints (official CEC2020 `cec20_func.m`, prob_k == 1):

        h1: 200 x1 x4 - x3                                    = 0
        h2: 200 x2 x6 - x5                                    = 0
        h3: x3 - 1e4 (x7 - 100)                               = 0
        h4: x5 - 1e4 (300 - x7)                               = 0
        h5: x3 - 1e4 (600 - x8)                               = 0
        h6: x5 - 1e4 (900 - x9)                               = 0
        h7: x4 ln|x8-100| - x4 ln(600-x7) - x8 + x7 + 500     = 0
        h8: x6 ln|x9-x7| - x6 ln 600 - x9 + x7 + 600          = 0

    Step 1.  h3 gives x7 = 100 + x3/1e4;  h5 gives x8 = 600 - x3/1e4.

    Step 2.  Substituting into h7, both logarithm arguments become the same
             number, 500 - x3/1e4:

                 |x8 - 100| = |500 - x3/1e4| = 500 - x3/1e4   (x3 <= 100)
                 600 - x7   =  500 - x3/1e4

             so x4 (ln A - ln A) = 0 for *every* x4 -- the logarithmic term,
             the one the literature calls the hard part, vanishes identically.
             What remains is

                 -x8 + x7 + 500 = 2 x3 / 1e4,

             so h7 = 0 forces  x3 = 0,  hence  x7 = 100,  x8 = 600.

    Step 3.  With x3 = 0:
                 h4  ->  x5 = 1e4 (300 - 100) = 2e6
                 h6  ->  x9 = 900 - x5/1e4 = 700
                 h1  ->  x1 x4 = 0
                 h2  ->  x2 x6 = 1e4
                 h8  ->  x6 ln 600 - x6 ln 600 - 700 + 100 + 600 = 0
                         identically, for every x6.

    Step 4.  Minimise f = 35 x1^0.6 + 35 x2^0.6 over this family; f is
             increasing in both arguments.  x1 x4 = 0 is satisfied by x1 = 0
             (any x4), and f is increasing in x1, so x1 = 0 is best. x2 = 1e4/x6
             is decreasing in x6, and x6 <= 600, so x2 >= 1e4/600 = 50/3,
             attained at x6 = 600.

                 f(x*) = 35 (50/3)^0.6 = 189.3116296866...

             This is the minimum over the exact-equality family, not a
             certified optimum of the tolerance-relaxed problem (module
             docstring).
    """
    x = np.zeros(9)
    x[0] = 0.0          # x1
    x[1] = 50.0 / 3.0   # x2 = 1e4 / x6
    x[2] = 0.0          # x3  (forced by h7)
    x[3] = 0.0          # x4  (free; x1 x4 = 0 already satisfied by x1 = 0)
    x[4] = 2.0e6        # x5  (h4)
    x[5] = 600.0        # x6  (upper bound -> minimises x2)
    x[6] = 100.0        # x7  (h3)
    x[7] = 600.0        # x8  (h5)
    x[8] = 700.0        # x9  (h6)
    return x


def main():
    p = HeatExchangerNetwork()
    x = derive()
    f_ref = 35.0 * (50.0 / 3.0) ** 0.6

    print(__doc__.strip())
    print("\n" + "=" * 68)
    print("Closed-form candidate")
    print("=" * 68)
    names = ["x1 (area 1)", "x2 (area 2)", "x3 (duty 1)", "x4 (approach 1)",
             "x5 (duty 2)", "x6 (approach 2)", "x7", "x8", "x9"]
    for n, v, lo, hi in zip(names, x, p.lb, p.ub):
        ok = "ok" if lo - 1e-12 <= v <= hi + 1e-12 else "OUT OF BOUNDS"
        print(f"  {n:<18s} = {v:>12.6g}   bounds [{lo:g}, {hi:g}]  {ok}")

    _, H = p.constraints(x[None, :])
    print("\nEquality residuals (tolerance eps = %g):" % p.eps_eq)
    for k, hv in enumerate(H[0], 1):
        slack = max(0.0, abs(hv) - p.eps_eq)
        print(f"  h{k}: {hv:+.6e}   excess over eps = {slack:.3e}")

    rep = p.report(x)
    print(f"\n  total violation : {rep['total_violation']:.6e}"
          f"   (feasible iff <= {p.feas_tol:g})")
    print(f"  FEASIBLE        : {rep['feasible']}")
    print(f"  objective       : {rep['objective']:.10f}")
    print(f"  closed form     : 35*(50/3)^0.6 = {f_ref:.10f}")
    assert rep["feasible"], "closed-form point is not feasible"
    assert abs(rep["objective"] - f_ref) < 1e-9

    # The h8 identity: it holds for every x6 once x3 = 0.
    print("\n  check: h8 is satisfied for every x6 once x3 = 0 ->", end=" ")
    worst = 0.0
    for x6 in np.linspace(50.0, 600.0, 23):
        y = x.copy()
        y[5], y[1] = x6, 1.0e4 / x6
        _, Hy = p.constraints(y[None, :])
        worst = max(worst, abs(Hy[0, 7]))
    print(f"max |h8| over x6 in [50, 600] = {worst:.3e}")

    # Within the exact-equality family: f is increasing in x1 and x2, and
    # x2 >= 1e4/600.
    print("  check: no point of the exact-equality family beats it ->", end=" ")
    best = np.inf
    for x6 in np.linspace(50.0, 600.0, 2001):
        y = x.copy()
        y[5], y[1] = x6, 1.0e4 / x6
        r = p.report(y)
        if r["feasible"]:
            best = min(best, r["objective"])
    print(f"min over the family = {best:.10f}")
    assert best >= f_ref - 1e-9

    print("\n" + "=" * 68)
    print(f"RC01 closed-form feasible reference design: f(x*) = {f_ref:.6f}")
    print("(feasible under eps = 1e-4; not a certified optimum of the relaxed problem)")
    print("=" * 68)
    return f_ref


if __name__ == "__main__":
    main()
