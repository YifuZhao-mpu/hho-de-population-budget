"""Compare this re-implementation of SEHHO-COBL with the original's published means.

The algorithm was originally implemented in MATLAB by its authors; that code is
not available, and this package re-implements it from the published
specification.  The per-function means reported in the earlier version of the
manuscript (CEC2022, D=10, N=30, T=500, 30 runs) are compared against this
implementation under the same 15,000-evaluation budget.

What the comparison shows: the mean errors agree to within a factor of two on
most functions (8 of the 10 comparable ones; F1 and F3 are not compared because
their published mean error is zero).  F11 differs most (ratio 0.15) and F5 is
next (0.33), both with the lower error here.  This is evidence that the
re-implementation is close to the original on this one block, not proof that
the two are equivalent, so it does not by itself carry conclusions drawn here
over to the original code.  The paper's analytical result (the escape energy
enters only through |E| >= 1, so the inherited framework is a Bernoulli gate)
follows from the published specification and does not depend on either
implementation.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

FSTAR = {1: 300, 2: 400, 3: 600, 4: 800, 5: 900, 6: 1800,
         7: 2000, 8: 2200, 9: 2300, 10: 2400, 11: 2600, 12: 2700}
# Means reported in the earlier manuscript version (Table "10-dimensional results").
PUBLISHED = {1: 3.000e2, 2: 4.066e2, 3: 6.000e2, 4: 8.071e2, 5: 9.001e2, 6: 1.815e3,
             7: 2.009e3, 8: 2.216e3, 9: 2.529e3, 10: 2.504e3, 11: 2.880e3, 12: 2.863e3}

df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                              "results", "E1_cec2022_paper_budget.csv"))
s = df[(df.dim == 10) & (df.algo == "SEHHO-COBL")]
print(f"{'Func':>5} {'published mean':>15} {'this package':>14} "
      f"{'published err':>14} {'our err':>12} {'ratio':>8}")
ratios = []
for f in range(1, 13):
    ours = s[s.func == f].best_f.mean()
    pe, oe = PUBLISHED[f] - FSTAR[f], ours - FSTAR[f]
    if pe > 1e-6:
        r = oe / pe
        ratios.append(r)
        rs = f"{r:8.3f}"
    else:
        rs = "  solved"
    print(f"F{f:<4d} {PUBLISHED[f]:15.4e} {ours:14.4e} {pe:14.4e} {oe:12.4e} {rs}")
ratios = np.array(ratios)
print(f"\nComparable functions: {len(ratios)}; median error ratio "
      f"{np.median(ratios):.3f}, geometric mean {np.exp(np.mean(np.log(ratios))):.3f}")
print(f"Within a factor of two of the published mean error: "
      f"{100*np.mean((ratios > 0.5) & (ratios < 2)):.0f}%")
