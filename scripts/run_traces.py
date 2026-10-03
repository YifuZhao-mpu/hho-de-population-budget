"""Convergence traces for the paper's figures.

Traces record the incumbent best value at a fixed grid of evaluation counts, so
curves from algorithms with different population sizes remain comparable: the
x axis is function evaluations, never generations.
"""

import os
import sys
import numpy as np
import pandas as pd
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sehho.bench import CECProblem, Budget
from sehho.runner import make_seed, _dispatch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts_common import ABLATION_CFG

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
RUNS = 30
NPTS = 120


def trace_cell(job):
    suite, func, dim, algo, cfg, max_fes, tag = job
    p = CECProblem(suite, func, dim)
    grid = np.unique(np.linspace(max_fes / NPTS, max_fes, NPTS).astype(np.int64))
    acc = []
    for r in range(RUNS):
        rng = np.random.default_rng(make_seed(suite, func, dim, algo, tag, r))
        b = Budget(p, max_fes, trace_points=grid)
        _dispatch(algo, b, rng, cfg)
        acc.append(np.asarray(b.trace, dtype=float) - p.f_star)
    A = np.vstack(acc)
    return pd.DataFrame(dict(tag=tag, suite=suite, func=func, dim=dim, algo=algo,
                             fes=grid, mean_error=A.mean(axis=0),
                             median_error=np.median(A, axis=0)))


def main():
    jobs = []
    MAIN = ["SEHHO-COBL", "HHO", "DE", "SHADE", "LSHADE", "jSO"]
    for f in (1, 6, 9, 12):
        for a in MAIN:
            jobs.append(("cec2022", f, 20, a, {}, 15000, "trace_main_cec2022_D20"))
    for f in (4, 10, 21, 30):
        for a in MAIN:
            jobs.append(("cec2017", f, 30, a, {}, 300000, "trace_main_cec2017_D30"))
    for f in (1, 6, 9, 12):
        for name, cfg in ABLATION_CFG.items():
            jobs.append(("cec2022", f, 20, f"SEHHO:{name}", dict(cfg), 15000,
                         "trace_ablation_cec2022_D20"))
    print(f"{len(jobs)} trace cells x {RUNS} runs", flush=True)
    with Pool(int(os.environ.get("SEHHO_WORKERS", 40))) as pool:
        parts = pool.map(trace_cell, jobs)
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(os.path.join(RES, "traces.csv"), index=False)
    print("wrote results/traces.csv", len(df), "rows")


if __name__ == "__main__":
    main()
