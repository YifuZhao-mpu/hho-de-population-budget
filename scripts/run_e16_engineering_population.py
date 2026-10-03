"""Stage E16: does the population result transfer to the five design problems?

Stage E14 changed only L-SHADE's initial population (6D against the published
18D) on CEC2017 and CEC2014.  This stage runs the same one-constant control on
the five constrained engineering problems of Section 7 at the budget used there
(15,000 evaluations), so that the paper's population recommendation is tested
where a designer would apply it.  Everything except N_init is stock L-SHADE, as
in E14; the 18D arm is run explicitly so both arms share code path and seeding
protocol.

The configuration, comparisons, statistics and the directional expectation are
fixed in `results/analysis/PREREGISTRATION_E16_engineering_population.json`,
which this script checks before running anything.  The result is reported
whatever it shows.

Run:  python3 scripts/run_e16_engineering_population.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sehho.engineering import ENGINEERING_PROBLEMS
from sehho.runner import Cell, run_all
import run_experiments as rx

ARMS = ("LSHADE-N6", "LSHADE-N18")
BUDGET = 15_000
PRE = os.path.join(rx.RESULTS, "analysis", "PREREGISTRATION_E16_engineering_population.json")
OUT = os.path.join(rx.RESULTS, "E16_engineering_population.csv")


def cells():
    out = []
    for key in ENGINEERING_PROBLEMS:
        for arm in ARMS:
            out.append(Cell("engineering", 0, ENGINEERING_PROBLEMS[key].dim, arm, BUDGET,
                            rx.RUNS, cfg=dict(problem=key), tag=f"E16_engpop_{key}"))
    return out


def main():
    if not os.path.exists(PRE):
        raise SystemExit(f"{PRE} missing: the pre-registration must exist before E16 runs")
    pre = json.load(open(PRE))
    if sorted(pre["design"]["problems"]) != sorted(ENGINEERING_PROBLEMS) or \
            list(pre["design"]["arms"]) != list(ARMS) or pre["design"]["budget"] != BUDGET:
        raise SystemExit("design does not match the pre-registration; refusing to run")
    if os.path.exists(OUT):
        raise SystemExit(f"{OUT} already exists; refusing to overwrite")
    cs = cells()
    print(f"E16: {len(cs)} cells x {rx.RUNS} runs = {len(cs) * rx.RUNS} runs")
    run_all(cs, workers=int(os.environ.get("SEHHO_WORKERS", 0)) or None, out_csv=OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
