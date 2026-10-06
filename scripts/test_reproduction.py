"""Regression test: the published configurations must still reproduce bit for bit.

Adding the population-control options to ``SEHHOConfig`` touched the main loop
of ``sehhocobl``.  Any such edit risks silently changing results that are
already in the manuscript, and floating-point arithmetic makes that easy to do
by accident -- ``a * t / T`` and ``a * (t/T)`` differ in the last bit, and
``pN = ceil(p*N)`` can round either side of an integer, which changes which
individual is selected and therefore the whole run.

This test re-runs a sample of cells and compares against the stored CSVs.  The
tolerance is 1e-12 relative, which admits the ~1e-16 noise of writing a float
to text and reading it back, and nothing else.  Besides the SEHHO-COBL cells it
re-executes SHADE, L-SHADE and jSO cells, which run the shared engine
``algorithms._shade_family`` with its validation switches at their defaults.

Run:  python3 scripts/test_reproduction.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sehho.runner import Cell, run_cell

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
TOL = 1e-12
RUNS = 3

CASES = [
    # (stored csv, tag, suite, dim, max_fes, {variant: cfg}, functions[, algorithm prefix])
    ("E4b_ablation_cec2017_30D.csv", "E4b_ablation_cec2017_30D", "cec2017", 30, 300_000,
     {"Full": dict(),
      "AlwaysPbest": dict(phase_mode="always_pbest"),
      "AlwaysPbest+NoLevy": dict(phase_mode="always_pbest", levy_mode="off"),
      "NoSHADE": dict(use_shade=False),
      "NoCOBL": dict(use_cobl=False),
      "NoArchive": dict(use_archive=False),
      "FixedP": dict(p_schedule="fixed")},
     (1, 14, 21)),
    ("E4c_ablation_cec2017_30D_15k.csv", "E4c_ablation_cec2017_30D_15k", "cec2017", 30, 15_000,
     {"Full": dict(), "AlwaysPbest": dict(phase_mode="always_pbest")},
     (3, 17)),
    # the SHADE-family baselines (shared engine, default switches)
    ("E12_cec2017_tight.csv", "E12_cec2017_tight", "cec2017", 30, 15_000,
     {"jSO": dict(), "LSHADE": dict(), "SHADE": dict()}, (1, 10), ""),
    ("E13_cec2014.csv", "E13_cec2014_competition", "cec2014", 30, 300_000,
     {"LSHADE": dict()}, (10,), ""),
]


def main():
    worst, checked, failures = 0.0, 0, []
    for case in CASES:
        fname, tag, suite, dim, fes, variants, funcs = case[:7]
        prefix = case[7] if len(case) > 7 else "SEHHO:"
        path = os.path.join(RES, fname)
        if not os.path.exists(path):
            print(f"  [skip] {fname} not present")
            continue
        stored = pd.read_csv(path)
        for name, cfg in variants.items():
            for fn in funcs:
                cell = Cell(suite, fn, dim, f"{prefix}{name}", fes, RUNS,
                            cfg=dict(cfg), tag=tag)
                got = pd.DataFrame(run_cell(cell)).set_index("run").sort_index()
                ref = (stored[(stored.algo == f"{prefix}{name}") & (stored.func == fn)
                              & (stored.dim == dim) & (stored.tag == tag)]
                       .set_index("run").sort_index().loc[list(range(RUNS))])
                if not np.array_equal(got["seed"].to_numpy(), ref["seed"].to_numpy()):
                    failures.append(f"{name} F{fn}: seeds differ")
                    continue
                a, b = got["raw_error"].to_numpy(), ref["raw_error"].to_numpy()
                rel = float((np.abs(a - b) / np.maximum(np.abs(b), 1e-300)).max())
                worst = max(worst, rel)
                checked += 1
                if rel > TOL:
                    failures.append(f"{name} F{fn}: max relative difference {rel:.2e}\n"
                                    f"      now    {a}\n      stored {b}")
    print(f"\nchecked {checked} cells x {RUNS} runs; "
          f"worst relative difference {worst:.2e} (tolerance {TOL:.0e})")
    if failures:
        print("\nFAILED:")
        for f in failures:
            print("  " + f)
        return 1
    print("PASS: every published configuration still reproduces.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
