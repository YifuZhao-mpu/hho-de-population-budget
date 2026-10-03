"""Stage E15: does the CEC2014 replication depend on the composition-rule amendment?

The composition of the repaired method (stages E8/E11) was decided by a rule
that was amended after its first output had been seen.  The rule as coded
before E8 ran kept COBL and the Levy perturbation and dropped only the p
annealing; the amended rule also dropped COBL and Levy, and the stripped
configuration is the one that was frozen and evaluated on CEC2014 (E13).

This stage runs the configuration the original rule would have produced --
the gate-free method with LPSR from 6D and only the p annealing removed -- in
exactly the E13 cells, so the pre-registered primary contrast can be computed
for it as well.  It is a post-hoc sensitivity check and is reported as
exploratory whatever it shows.  Its configuration, comparisons and analysis
script are fixed in `results/analysis/PREREGISTRATION_E15_original_rule.json`,
which this script checks before running anything.

Run:  python3 scripts/run_e15_original_rule.py
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sehho.bench import CEC2014_FUNCS
from sehho.runner import Cell, run_all
import run_experiments as rx

ALGO = "GF:OriginalRule"
PRE = os.path.join(rx.RESULTS, "analysis", "PREREGISTRATION_E15_original_rule.json")
OUT = os.path.join(rx.RESULTS, "E15_original_rule_cec2014.csv")


def original_rule_cfg():
    """The gate-free LPSR-6D default with only the p annealing removed."""
    _, base = rx._selected_pop()
    return dict(base, p_schedule="fixed")


def cfg_sha256(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def cells():
    cfg = original_rule_cfg()
    out = []
    for d in (30, 50):
        for fes in (10_000 * d, 15_000):
            blk = "competition" if fes > 15_000 else "tight"
            out += [Cell("cec2014", f, d, ALGO, fes, rx.RUNS, cfg=dict(cfg),
                         tag=f"E15_original_rule_cec2014_{blk}")
                    for f in CEC2014_FUNCS]
    return out


def main():
    if not os.path.exists(PRE):
        raise SystemExit(f"{PRE} missing: the pre-registration must exist before E15 runs")
    pre = json.load(open(PRE))
    cfg = original_rule_cfg()
    if cfg != pre["configuration"] or cfg_sha256(cfg) != pre["configuration_sha256"]:
        raise SystemExit("configuration does not match the pre-registration; refusing to run")
    if os.path.exists(OUT):
        raise SystemExit(f"{OUT} already exists; refusing to overwrite")
    cs = cells()
    print(f"E15: {len(cs)} cells x {rx.RUNS} runs = {len(cs) * rx.RUNS} runs, cfg={cfg}")
    run_all(cs, workers=int(os.environ.get("SEHHO_WORKERS", 0)) or None, out_csv=OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
