"""Budget study for the two equality-constrained process problems.

The manuscript reported HEN and BPS as penalised-objective stress tests because
no run was ever feasible.  This asks the sharper question: is the declared
feasibility tolerance reachable at all, and if so at what evaluation budget?
"""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sehho.runner import Cell, run_all

BUDGETS = [15_000, 50_000, 200_000, 1_000_000]
ALGOS = ["SEHHO-COBL", "LSHADE", "jSO"]
cells = []
for prob in ("P3_heat_exchanger_RC01", "P4_blending_pooling_RC06"):
    for fes in BUDGETS:
        for a in ALGOS:
            cells.append(Cell("engineering", 0, 0, a, fes, 30,
                              cfg=dict(problem=prob), tag=prob))
df = run_all(cells, workers=int(os.environ.get("SEHHO_WORKERS", 6)),
             out_csv="results/E5b_engineering_budget.csv")
g = (df.groupby(["problem", "max_fes", "algo"])
       .agg(feas_rate=("feasible", lambda s: 100 * s.mean()),
            best=("objective", lambda s: s.min()),
            mean_viol=("total_violation", "mean")).reset_index())
print(g.to_string(index=False))
