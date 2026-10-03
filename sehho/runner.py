"""Parallel experiment driver.

A *unit of work* is one (suite, function, dimension, algorithm/variant) cell and
contains all independent runs for that cell.  Grouping runs this way means the
native suite loads its shift/rotation data once per cell instead of once per
run.

Seeding is deterministic: the seed of a run is a hash of the cell identity and
the run index, so every number in the paper can be regenerated from this file
alone.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field, asdict
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .bench import CECProblem, Budget
from .algorithms import (SEHHOConfig, sehhocobl, hho, de_rand1bin, shade,
                         lshade, lshade_n, jso)


def make_seed(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "little") % (2 ** 63 - 1)


@dataclass
class Cell:
    suite: str
    func: int
    dim: int
    algo: str                     # display name
    max_fes: int
    runs: int = 30
    cfg: dict = field(default_factory=dict)   # SEHHOConfig overrides
    tag: str = ""                 # experiment tag, e.g. "main" / "ablation"


# Display-name prefixes that are configurations of the same optimiser rather
# than different algorithms: "SEHHO:" for the published method's ablations,
# "POP:" for the population-rule search, "GF:" for the gate-free method and its
# own ablations.  Everything after the prefix is a label only; the behaviour
# comes entirely from the cell's cfg, so a run is fully described by its seed
# and that dict.
_SEHHO_PREFIXES = ("SEHHO", "POP:", "GF")


def _dispatch(algo: str, budget, rng, cfg: dict):
    if algo.startswith(_SEHHO_PREFIXES):
        return sehhocobl(budget, rng, SEHHOConfig(**cfg))
    if algo == "HHO":
        return hho(budget, rng, N=cfg.get("N", 30))
    if algo == "DE":
        return de_rand1bin(budget, rng, N=cfg.get("N", 100))
    if algo == "SHADE":
        return shade(budget, rng, N=cfg.get("N", 100))
    if algo == "LSHADE":
        return lshade(budget, rng)
    if algo.startswith("LSHADE-N"):
        # "LSHADE-N6" etc: stock L-SHADE with only N_init changed to that
        # multiple of D.  Isolates the population rule from everything else.
        return lshade_n(budget, rng, float(algo.split("LSHADE-N")[1]))
    if algo == "jSO":
        return jso(budget, rng)
    raise ValueError(f"unknown algorithm {algo!r}")


def _make_problem(cell: Cell):
    if cell.suite == "engineering":
        from .engineering import ENGINEERING_PROBLEMS
        p = ENGINEERING_PROBLEMS[cell.cfg.get("problem", cell.tag)]()
        p.name = p.name
        return p
    return CECProblem(cell.suite, cell.func, cell.dim)


def run_cell(cell: Cell):
    problem = _make_problem(cell)
    is_eng = cell.suite == "engineering"
    algo_cfg = {k: v for k, v in cell.cfg.items() if k != "problem"}
    rows = []
    for r in range(cell.runs):
        seed = make_seed(cell.suite, cell.func, cell.dim, cell.algo, cell.tag, r)
        rng = np.random.default_rng(seed)
        budget = Budget(problem, cell.max_fes)
        t0 = time.perf_counter()
        best = _dispatch(cell.algo, budget, rng, algo_cfg)
        dt = time.perf_counter() - t0
        err = best - problem.f_star
        row = dict(
            tag=cell.tag, suite=cell.suite, func=cell.func, dim=cell.dim,
            algo=cell.algo, run=r, seed=seed, max_fes=cell.max_fes,
            fes_used=budget.used, best_f=best,
            # CEC convention: errors below 1e-8 are reported as 0
            error=0.0 if err < 1e-8 else err,
            raw_error=err, seconds=dt,
        )
        if is_eng:
            # the optimiser only ever sees the feasibility-rule scalar, so the
            # engineering quantities are recomputed from the returned design
            row["problem"] = cell.cfg.get("problem", "")
            row.update(problem.report(budget.best_x))
        rows.append(row)
    return rows


def run_all(cells, workers: int = None, out_csv: str = None, quiet: bool = False):
    workers = workers or int(os.environ.get("SEHHO_WORKERS", 0)) or max(1, (os.cpu_count() or 4) - 8)
    t0 = time.perf_counter()
    rows = []
    done = 0
    with Pool(workers) as pool:
        for res in pool.imap_unordered(run_cell, cells, chunksize=1):
            rows.extend(res)
            done += 1
            if not quiet and (done % 25 == 0 or done == len(cells)):
                el = time.perf_counter() - t0
                rate = done / max(el, 1e-9)
                print(f"   {done}/{len(cells)} cells  {el/60:6.1f} min elapsed  "
                      f"ETA {(len(cells)-done)/max(rate,1e-9)/60:6.1f} min", flush=True)
    df = pd.DataFrame(rows)
    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df.to_csv(out_csv, index=False)
        if not quiet:
            print(f"   wrote {out_csv}  ({len(df)} runs)")
    return df
