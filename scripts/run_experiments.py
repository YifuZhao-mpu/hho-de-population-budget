"""Run the full experimental programme for the SEHHO-COBL revision.

Usage:  python3 scripts/run_experiments.py E1 [E2 ...]      (default: all)

E1  CEC2022, paper budget (15,000 FEs), strictly FE-matched, 6 algorithms.
E2  CEC2022, competition budget (2e5 at D=10, 1e6 at D=20), 6 algorithms.
E3  CEC2017 at D = 30, 50, 100, competition budget 10,000*D, 6 algorithms.
E4  Core ablations of SEHHO-COBL (phase gate, p schedule, and each module).
E5  Five constrained engineering problems under the feasibility rule.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sehho.bench import CEC2014_FUNCS, CEC2017_FUNCS, CEC2022_FUNCS
from sehho.runner import Cell, run_all
from sehho.engineering import ENGINEERING_PROBLEMS

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
RUNS = 30
MAIN_ALGOS = ["SEHHO-COBL", "HHO", "DE", "SHADE", "LSHADE", "jSO"]

# --------------------------------------------------------------------------
# Ablation design.
#
# The first block isolates the contribution that the manuscript's central claim
# ("framework-engine decoupling") rests on: the HHO escape energy used as a
# phase indicator that switches pbest breadth.  Because E enters the algorithm
# only through the test |E| >= 1, that mechanism is exactly a Bernoulli gate
# with exploration probability q(t) = max(0, 1 - 1/(2(1-t/T))); the variants
# below replace it with alternatives that keep the interface but remove one
# property each.  The second block is the conventional module ablation.
# --------------------------------------------------------------------------
ABLATIONS = {
    "Full":                dict(),
    # --- phase-gate ablations (the decisive set) ---
    "NoEscapeEnergy":      dict(phase_mode="constant_gate"),   # same mix, no schedule
    "AlwaysPbest":         dict(phase_mode="always_pbest"),    # SHADE-style, no gate
    "AlwaysBest":          dict(phase_mode="always_best"),     # greedy, no gate
    "DeterministicPhase":  dict(phase_mode="deterministic"),   # t<T/2, no mixing
    "FixedP":              dict(p_schedule="fixed"),           # gate kept, no p anneal
    "AlwaysPbest+FixedP":  dict(phase_mode="always_pbest", p_schedule="fixed"),
    # Deleting the gate also makes the Levy perturbation eligible on every
    # update instead of only in the exploration phase, raising its activation
    # from qbar*0.3 = 4.6% to 30% and extending it past t = T/2, where the
    # proposed gate is identically zero.  Without this cell the AlwaysPbest
    # result cannot separate elite breadth from perturbation frequency.
    "AlwaysPbest+NoLevy":  dict(phase_mode="always_pbest", levy_mode="off"),
    # --- module ablations ---
    "NoCOBL":              dict(use_cobl=False),
    "NoSHADE":             dict(use_shade=False),
    "NoLevy":              dict(levy_mode="off"),
    "NoArchive":           dict(use_archive=False),
}


# --------------------------------------------------------------------------
# The gate-free method and its population rule.
#
# The ablation establishes two things about SEHHO-COBL: the inherited
# escape-energy gate is harmful at realistic budgets, and the fixed population
# of N=30 is the single most costly setting in the method (Cliff's delta of
# +0.60 against N=120 at 300,000 evaluations, against +0.17 for the gate).
# Deleting the gate is therefore only half the fix.  This stage searches the
# population rule.
#
# PROTOCOL.  Every constant below is chosen on CEC2022 alone, at D=10 and D=20
# and at both budgets; CEC2017 is never consulted while choosing.  CEC2017 is
# nevertheless the *diagnostic* suite, not a held-out one: the decision to
# replace the gate and the population rule was taken on it (E4b, E6).  The
# confirmatory suite is CEC2014 (E13).  The search is run on the smaller suite,
# even though the larger one has more functions, so that no constant is fitted
# on the suite the repair was diagnosed on.
# --------------------------------------------------------------------------
GATE_FREE = dict(phase_mode="always_pbest")

POP_CANDIDATES = {
    # fixed population, the manuscript's scheme
    "N30":      dict(N=30),                                  # as published
    "N120":     dict(N=120),                                 # best fixed N in E6
    "N120b":    dict(N=120, schedule_basis="budget"),         # control, see below
    "N4D":      dict(N_init_factor=4.0),                      # fixed but D-scaled
    # linear population size reduction, as in L-SHADE; the schedule basis must
    # change with it because T is no longer known in advance
    "LPSR-6D":  dict(pop_mode="lpsr", N_init_factor=6.0,  schedule_basis="budget"),
    "LPSR-12D": dict(pop_mode="lpsr", N_init_factor=12.0, schedule_basis="budget"),
    "LPSR-18D": dict(pop_mode="lpsr", N_init_factor=18.0, schedule_basis="budget"),
}
# "N120b" exists so that the LPSR candidates are not confounded with the change
# of schedule basis they force: it is N120 with the basis changed and nothing
# else, so any difference between N120 and N120b is the basis alone.

CEC2022_COMPETITION_FES = {10: 200_000, 20: 1_000_000}


def e7():
    """Population rule for the gate-free method, searched on CEC2022 only."""
    cells = []
    for name, pop in POP_CANDIDATES.items():
        cfg = dict(GATE_FREE, **pop)
        for d in (10, 20):
            for fes, block in ((15_000, "matched"),
                               (CEC2022_COMPETITION_FES[d], "competition")):
                cells += [Cell("cec2022", f, d, f"POP:{name}@{block}", fes, RUNS,
                               cfg=dict(cfg), tag="E7_poprule_cec2022")
                          for f in CEC2022_FUNCS]
    return cells, "E7_poprule_cec2022.csv"


def e1():
    cells = [Cell("cec2022", f, d, a, 15000, RUNS, tag="E1_cec2022_paper_budget")
             for d in (10, 20) for f in CEC2022_FUNCS for a in MAIN_ALGOS]
    return cells, "E1_cec2022_paper_budget.csv"


def e2():
    cells = []
    for d, fes in ((10, 200_000), (20, 1_000_000)):
        cells += [Cell("cec2022", f, d, a, fes, RUNS, tag="E2_cec2022_competition")
                  for f in CEC2022_FUNCS for a in MAIN_ALGOS]
    return cells, "E2_cec2022_competition.csv"


def e3():
    cells = []
    for d in (30, 50, 100):
        cells += [Cell("cec2017", f, d, a, 10_000 * d, RUNS, tag="E3_cec2017")
                  for f in CEC2017_FUNCS for a in MAIN_ALGOS]
    return cells, "E3_cec2017.csv"


def e4():
    cells = []
    for name, cfg in ABLATIONS.items():
        for d in (10, 20):
            cells += [Cell("cec2022", f, d, f"SEHHO:{name}", 15000, RUNS,
                           cfg=dict(cfg), tag="E4_ablation_cec2022")
                      for f in CEC2022_FUNCS]
    return cells, "E4_ablation_cec2022.csv"


def e4b():
    cells = []
    for name, cfg in ABLATIONS.items():
        cells += [Cell("cec2017", f, 30, f"SEHHO:{name}", 300_000, RUNS,
                       cfg=dict(cfg), tag="E4b_ablation_cec2017_30D")
                  for f in CEC2017_FUNCS]
    return cells, "E4b_ablation_cec2017_30D.csv"


def e4c():
    """Budget isolation: the same ablation at 15,000 FEs on CEC2017 D=30.

    The contrast between "inert at 15,000 FEs" and "harmful at 300,000" was
    drawn across CEC2022 D=10/20 and CEC2017 D=30, so suite, dimension and
    budget all moved together.  This repeats the whole ablation at the small
    budget on the *same* suite and dimension as E4b, leaving budget as the only
    difference.
    """
    cells = []
    for name, cfg in ABLATIONS.items():
        cells += [Cell("cec2017", f, 30, f"SEHHO:{name}", 15_000, RUNS,
                       cfg=dict(cfg), tag="E4c_ablation_cec2017_30D_15k")
                  for f in CEC2017_FUNCS]
    return cells, "E4c_ablation_cec2017_30D_15k.csv"


def e5():
    cells = []
    for key in ENGINEERING_PROBLEMS:
        for a in MAIN_ALGOS:
            cells.append(Cell("engineering", 0, ENGINEERING_PROBLEMS[key].dim, a,
                              15000, RUNS, cfg=dict(problem=key), tag=key))
    return cells, "E5_engineering.csv"



def e6():
    """Does SEHHO-COBL's fixed N=30 limit it at large budgets?

    The manuscript fixes the population at 30 for every experiment.  At the
    competition budget that is far smaller than the population L-SHADE and jSO
    give themselves, so this sweep separates "the design is weaker" from "the
    population was too small for this budget".
    """
    from sehho.bench import CEC2017_FUNCS
    cells = []
    for N in (30, 60, 120, 270, 540):
        cells += [Cell("cec2017", f, 30, "SEHHO-COBL", 300_000, RUNS,
                       cfg=dict(N=N), tag=f"E6_popsize_N{N}")
                  for f in CEC2017_FUNCS]
    return cells, "E6_popsize_cec2017_30D.csv"


def _selected_pop():
    """The population rule chosen by E7, read back from its analysis output.

    Kept in a file rather than hard-coded so that the choice is traceable to the
    experiment that made it: the rule used by every later stage is literally the
    one E7 selected, and changing E7 changes them all.
    """
    import json
    path = os.path.join(RESULTS, "analysis", "selected_config.json")
    if not os.path.exists(path):
        raise SystemExit("run E7 and scripts/analyze_poprule.py first — "
                         f"{path} does not exist")
    name = json.load(open(path))["winner"]
    return name, dict(GATE_FREE, **POP_CANDIDATES[name])


# Composition of the gate-free method: does each remaining part earn its place?
# Measured around the *new* default, not around the published one -- a module
# that is negligible next to SEHHO-COBL may not be negligible next to a method
# with a different population rule, and vice versa.  Run on the design suite,
# CEC2022, so that dropping a part is decided off both the diagnostic suite
# (CEC2017) and the confirmatory one (CEC2014).
COMPOSITION = {
    "Default":   dict(),
    "+Gate":     dict(phase_mode="escape_energy"),   # put the gate back
    "-COBL":     dict(use_cobl=False),
    "-Levy":     dict(levy_mode="off"),
    "-Archive":  dict(use_archive=False),
    "-SHADE":    dict(use_shade=False),
    "-pAnneal":  dict(p_schedule="fixed"),
}


def e8():
    """Composition ablation of the gate-free method, on the design suite."""
    _, base = _selected_pop()
    cells = []
    for name, over in COMPOSITION.items():
        cfg = dict(base, **over)
        for d in (10, 20):
            for fes, blk in ((15_000, "matched"),
                             (CEC2022_COMPETITION_FES[d], "competition")):
                cells += [Cell("cec2022", f, d, f"GF:{name}@{blk}", fes, RUNS,
                               cfg=dict(cfg), tag="E8_composition_cec2022")
                          for f in CEC2022_FUNCS]
    return cells, "E8_composition_cec2022.csv"


# Parts the composition ablation finds do not earn their place.  Removing three
# things at once is not the sum of removing each, so the combination is tested
# rather than assumed.
STRIPPED = dict(use_cobl=False, levy_mode="off", p_schedule="fixed")


def e14():
    """Isolate the initial population size, inside each algorithm.

    The claim that L-SHADE's N_init = 18D is mis-sized at tight budgets cannot
    be read off a comparison between L-SHADE and our own configuration: those
    two differ in the memory size, the archive capacity and the elite fraction
    as well, and in implementation details (bound handling, steady-state versus
    generational replacement, the CR memory mean, the terminal CR value, F
    sampling, rounding of the pbest pool).  Each arm below differs from its
    own reference (stock L-SHADE, or R-SHADE) in **one** constant and nothing
    else:

      * `LSHADE-N6`  -- stock L-SHADE with N_init = 6D instead of 18D;
      * `LSHADE-N18` -- stock L-SHADE, included explicitly so the two arms of
        the contrast come from identical code paths and seeding;
      * `GF@18D`     -- our own configuration moved to N_init = 18D, the same
        contrast in the other direction.

    Both budgets and both suites, so the comparison is available wherever the
    claim is made.  Nothing here may alter R-SHADE's configuration; on CEC2014
    this is an unregistered comparison and is reported as exploratory, per the
    pre-registration's own clause.
    """
    base = _final_cfg()
    arms = [("LSHADE-N6", {}), ("LSHADE-N18", {}),
            ("GF@18D", dict(base, N_init_factor=18.0))]
    cells = []
    for suite, funcs, dims in (("cec2017", CEC2017_FUNCS, (30, 50)),
                               ("cec2014", CEC2014_FUNCS, (30, 50))):
        for d in dims:
            for fes in (10_000 * d, 15_000):
                for name, cfg in arms:
                    cells += [Cell(suite, f, d, name, fes, RUNS, cfg=dict(cfg),
                                   tag=f"E14_popisolation_{suite}")
                              for f in funcs]
    return cells, "E14_pop_isolation.csv"


def e13():
    """Confirmatory evaluation on CEC2014, a suite consulted for nothing before
    this stage ran.

    CEC2017 cannot serve as a held-out suite: it was used to diagnose *which*
    components to change (the gate ablation of Section 6.2 and the population
    sweep), even though the final constants were chosen on CEC2022.  Choosing
    what to repair is a design decision, so an evaluation reserved from the
    whole construction is needed.  CEC2014 is that evaluation.

    The configuration is frozen before this stage runs and may not be altered
    on the basis of anything it returns; the analysis, the primary endpoint and
    the success criterion are fixed in
    `results/analysis/PREREGISTRATION_cec2014.json`, an internal pre-registration
    written (and timestamped inside the file) first.  The hash it records
    covers the frozen configuration only, not the whole file.
    """
    base = _final_cfg()
    algos = MAIN_ALGOS + ["GF-Method"]
    cells = []
    for d in (30, 50):
        for fes in (10_000 * d, 15_000):
            for a in algos:
                cfg = dict(base) if a == "GF-Method" else {}
                cells += [Cell("cec2014", f, d, a, fes, RUNS, cfg=dict(cfg),
                               tag=f"E13_cec2014_{'competition' if fes > 15_000 else 'tight'}")
                          for f in CEC2014_FUNCS]
    return cells, "E13_cec2014.csv"


def e12():
    """Tight budget on the diagnostic suite.

    E9 shows the repaired method first on CEC2022 at 15,000 evaluations, but
    CEC2022 is the suite its population rule was chosen on, so that result is
    in-sample.  This repeats the tight-budget comparison on CEC2017, against
    every baseline.  No constant was chosen on CEC2017, but the repair was
    diagnosed on it, so this is a diagnostic-suite result rather than a
    held-out one; the confirmatory evaluation is E13 (CEC2014).
    """
    base = _final_cfg()
    algos = MAIN_ALGOS + ["GF-Method"]
    cells = []
    for d in (30, 50):
        for a in algos:
            cfg = dict(base) if a == "GF-Method" else {}
            cells += [Cell("cec2017", f, d, a, 15_000, RUNS, cfg=dict(cfg),
                           tag="E12_cec2017_tight")
                      for f in CEC2017_FUNCS]
    return cells, "E12_cec2017_tight.csv"


def e11():
    """The stripped configuration against the default, on the design suite."""
    _, base = _selected_pop()
    cfg = dict(base, **STRIPPED)
    cells = []
    for d in (10, 20):
        for fes, blk in ((15_000, "matched"), (CEC2022_COMPETITION_FES[d], "competition")):
            cells += [Cell("cec2022", f, d, f"GF:Stripped@{blk}", fes, RUNS,
                           cfg=dict(cfg), tag="E8_composition_cec2022")
                      for f in CEC2022_FUNCS]
    return cells, "E11_stripped_cec2022.csv"


def _final_cfg():
    """The configuration every headline result uses.

    Composed from the design-suite experiments only: E7 chose the population
    rule, E8/E11 chose which parts to keep.  Stored alongside the results so
    that the configuration behind a number is always recoverable.
    """
    import json
    _, base = _selected_pop()
    path = os.path.join(RESULTS, "analysis", "composition_decision.json")
    strip = STRIPPED if (os.path.exists(path)
                         and json.load(open(path)).get("use_stripped")) else {}
    return dict(base, **strip)


def e9():
    """The new method against the baselines. Only the new method is run here;
    the five baselines already exist in E1, E2 and E3 under identical cells."""
    base = _final_cfg()
    cells = []
    for d in (10, 20):
        cells += [Cell("cec2022", f, d, "GF-Method", 15_000, RUNS, cfg=dict(base),
                       tag="E1_cec2022_paper_budget") for f in CEC2022_FUNCS]
        cells += [Cell("cec2022", f, d, "GF-Method", CEC2022_COMPETITION_FES[d],
                       RUNS, cfg=dict(base), tag="E2_cec2022_competition")
                  for f in CEC2022_FUNCS]
    for d in (30, 50, 100):
        cells += [Cell("cec2017", f, d, "GF-Method", 10_000 * d, RUNS,
                       cfg=dict(base), tag="E3_cec2017") for f in CEC2017_FUNCS]
    return cells, "E9_newmethod_vs_baselines.csv"


def e10():
    """The new method on the five constrained engineering problems."""
    base = _final_cfg()
    cells = []
    for key in ENGINEERING_PROBLEMS:
        cells.append(Cell("engineering", 0, ENGINEERING_PROBLEMS[key].dim,
                          "GF-Method", 15_000, RUNS,
                          cfg=dict(base, problem=key), tag=key))
    return cells, "E10_newmethod_engineering.csv"


STAGES = {"E1": e1, "E2": e2, "E3": e3, "E4": e4, "E4b": e4b, "E4c": e4c,
          "E5": e5, "E6": e6, "E7": e7, "E8": e8, "E9": e9, "E10": e10, "E11": e11, "E12": e12, "E13": e13, "E14": e14}

if __name__ == "__main__":
    wanted = sys.argv[1:] or ["E1", "E5", "E4", "E2", "E4b", "E3", "E6"]
    for key in wanted:
        cells, name = STAGES[key]()
        print(f"\n=== {key}: {len(cells)} cells x {RUNS} runs "
              f"= {len(cells)*RUNS} runs ===", flush=True)
        t0 = time.perf_counter()
        run_all(cells, out_csv=os.path.join(RESULTS, name))
        print(f"=== {key} done in {(time.perf_counter()-t0)/60:.1f} min ===", flush=True)
