"""Validation: the shared SHADE-family engine against its authors' released code.

The SHADE, L-SHADE and jSO ports share one engine, ``algorithms._shade_family``,
which differs from Tanabe's released C++ code in two declared rules
(Supplementary Section S2.3; README, "Known deviations and disclosures"):

* the terminal value of the CR memory is *absorbing* (the L-SHADE paper's
  Algorithm 1), whereas ``lshade.cc`` 1.0.0/1.0.1 and SHADE 1.1 re-accumulate
  the slot, and SHADE 1.0's own code has no terminal value at all;
* the archive stores the *replaced parent* (the L-SHADE paper and
  ``lshade.cc`` 1.0.1), whereas ``lshade.cc`` 1.0.0, the CEC2014 competition
  code that produced the published L-SHADE runs, stores the trial vector.

The engine's switches ``terminal`` and ``archive_trial`` reproduce the C++
behaviour; their defaults are the engine used for every experiment of the
paper, and they change no random draw.  This script produces and analyses the
validation runs behind every number that main text Section 2.4 and
Supplementary Section S2.3 quote about them.

Sub-commands
------------
run             Re-runs stock L-SHADE and SHADE on CEC2014 (D=30 and 50,
                10,000*D evaluations, all 30 functions, 30 runs) with the
                switched engine.  Each variant run reuses the seed of the stored
                E13 run of the same algorithm, function, dimension and index
                (E13 tag ``E13_cec2014_competition``), so every variant run is
                paired with an E13 run.  Output (appended per dimension):
                    results/validation_engine_variants_cec2014.csv
                The variants:
                    LSHADE[reaccumulate,parent]  terminal="reaccumulate"
                    LSHADE[absorbing,trial]      archive_trial=True
                    LSHADE[reaccumulate,trial]   both switched: the behaviour of
                                                 lshade.cc 1.0.0
                    SHADE[none]                  terminal="none": SHADE 1.0's own
                                                 memory update
check-defaults  Replays a sample of stored E13 L-SHADE, SHADE and jSO runs through
                this script's variant entry points with the switches at their
                defaults, and fails unless every one is reproduced bit for bit.
analyze         Prints, and writes to results/analysis/validation_engine_variants.csv,
                (a) every engine against the published L-SHADE runs (needs
                    --lshade-dir; see validate_lshade_shade.py for the source and
                    its checksums),
                (b) the paired effects of each switch (E13 seeds),
                (c) the released SHADE and SHADE[none] against our runs of
                    Tanabe's SHADE 1.0.1 C++ code
                    (results/validation_shade101_cpp_cec2014.csv; see
                    scripts/shade101_cpp/), and
                (d) the sensitivity of the paper's CEC2014 contrast of the repaired
                    configuration (GF-Method) with L-SHADE to the L-SHADE runs used.
                Nothing is optimised.

Statistics follow validate_lshade_shade.py: ratios of mean errors where both
exceed 1e-8, two-sided rank-sum tests per function (errors at the CEC
resolution of 1e-8) Holm-corrected over the 30 functions of a block, and, for
paired runs, two-sided Wilcoxon signed-rank tests, Holm-corrected over the 30
functions.  Cliff's delta is reported for arm A against arm B (negative favours
A), as everywhere in the paper.

Usage
-----
    python3 scripts/validate_engine_variants.py run --dims 30 50 [--workers 40]
    python3 scripts/validate_engine_variants.py check-defaults
    python3 scripts/validate_engine_variants.py analyze --lshade-dir /path/to/L-SHADE \\
            [--summary-csv results/analysis/validation_engine_variants.csv]
"""
import argparse
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import sehho.algorithms as A                                  # noqa: E402
from sehho.bench import CECProblem, Budget, CEC2014_FUNCS     # noqa: E402
from sehho.runner import make_seed                            # noqa: E402
from sehho.stats import holm, cliffs_delta                    # noqa: E402

RES = os.path.join(HERE, "..", "results")
VARIANT_CSV = os.path.join(RES, "validation_engine_variants_cec2014.csv")
CPP_CSV = os.path.join(RES, "validation_shade101_cpp_cec2014.csv")
SUMMARY_CSV = os.path.join(RES, "analysis", "validation_engine_variants.csv")
E13_CSV = os.path.join(RES, "E13_cec2014.csv")
E13_TAG = "E13_cec2014_competition"
TAG = "validation_engine_variants"
ZERO = 1e-8
FUNCS = list(CEC2014_FUNCS)


# ------------------------------------------------------------------ the variants
def lshade_switched(budget, rng, terminal="absorbing", archive_trial=False):
    """Stock L-SHADE (the arguments of algorithms.lshade) with the two switches."""
    N = int(round(18 * budget.dim))
    return A._shade_family(budget, rng, N_init=N, H=6, arc_rate=2.6, p_rate=0.11,
                           memory_f0=0.5, memory_cr0=0.5, lpsr=True, jso_mode=False,
                           terminal=terminal, archive_trial=archive_trial)


def shade_switched(budget, rng, terminal="absorbing", N=100):
    """Stock SHADE 1.0 (the arguments of algorithms.shade) with the terminal switch."""
    return A._shade_family(budget, rng, N_init=N, H=N, arc_rate=1.0, p_rate=0.1,
                           memory_f0=0.5, memory_cr0=0.5, lpsr=False, jso_mode=False,
                           cr_lehmer=False, p_random=True, terminal=terminal)


def jso_switched(budget, rng, terminal="absorbing"):
    """Stock jSO (the arguments of algorithms.jso); used only by check-defaults."""
    d = budget.dim
    N = int(round(math.sqrt(d) * math.log(d) * 25))
    return A._shade_family(budget, rng, N_init=N, H=5, arc_rate=1.0, p_rate=0.25,
                           memory_f0=0.3, memory_cr0=0.8, lpsr=True, jso_mode=True,
                           terminal=terminal)


# name: (E13 algorithm whose seeds are reused, terminal, archive_trial)
VARIANTS = {
    "LSHADE[reaccumulate,parent]": ("LSHADE", "reaccumulate", False),
    "LSHADE[absorbing,trial]": ("LSHADE", "absorbing", True),
    "LSHADE[reaccumulate,trial]": ("LSHADE", "reaccumulate", True),
    "SHADE[none]": ("SHADE", "none", False),
}


def _call(base, terminal, archive_trial, budget, rng):
    if base == "LSHADE":
        return lshade_switched(budget, rng, terminal, archive_trial)
    if base == "SHADE":
        assert not archive_trial
        return shade_switched(budget, rng, terminal)
    if base == "jSO":
        return jso_switched(budget, rng, terminal)
    raise ValueError(base)


def _unit(args):
    name, base, terminal, archive_trial, func, dim, run = args
    prob = CECProblem("cec2014", func, dim)
    seed = make_seed("cec2014", func, dim, base, E13_TAG, run)
    budget = Budget(prob, 10_000 * dim)
    t0 = time.perf_counter()
    best = _call(base, terminal, archive_trial, budget, np.random.default_rng(seed))
    err = best - prob.f_star
    return dict(tag=TAG, suite="cec2014", func=func, dim=dim, algo=name, run=run, seed=seed,
                seed_algo=base, seed_tag=E13_TAG, terminal=terminal,
                archive_trial=bool(archive_trial), max_fes=budget.max_fes,
                fes_used=budget.used, best_f=best, error=0.0 if err < ZERO else err,
                raw_error=err, seconds=time.perf_counter() - t0)


def cmd_run(args):
    names = args.variants or list(VARIANTS)
    for dim in args.dims:
        units = [(n, *VARIANTS[n], f, dim, r) for n in names for f in FUNCS
                 for r in range(args.runs)]
        units.sort(key=lambda u: -u[4])          # hybrid and composition functions first
        t0, rows = time.perf_counter(), []
        with Pool(args.workers) as pool:
            for i, row in enumerate(pool.imap_unordered(_unit, units, chunksize=1), 1):
                rows.append(row)
                if i % 300 == 0 or i == len(units):
                    print(f"  D={dim}: {i}/{len(units)} runs, "
                          f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)
        new = pd.DataFrame(rows).sort_values(["dim", "algo", "func", "run"])
        if os.path.exists(args.out):
            old = pd.read_csv(args.out, float_precision="round_trip")
            old = old[~((old.dim == dim) & old.algo.isin(names))]
            new = pd.concat([old, new], ignore_index=True).sort_values(
                ["dim", "algo", "func", "run"])
        new.to_csv(args.out, index=False)
        print(f"  wrote {args.out} ({len(new)} runs)")
    return 0


def cmd_check_defaults(args):
    """Stored E13 runs replayed through the switch entry points at their defaults."""
    e13 = pd.read_csv(E13_CSV, float_precision="round_trip")
    e13 = e13[e13.tag == E13_TAG].set_index(["algo", "dim", "func", "run"]).sort_index()
    units = [(base, base, "absorbing", False, func, dim, run) for base in ("LSHADE", "SHADE", "jSO")
             for dim in (30, 50) for func in args.funcs for run in range(args.runs)]
    with Pool(min(args.workers, len(units))) as pool:
        got = pool.map(_unit, units, chunksize=1)
    worst, bad, n = 0.0, [], 0
    for row in got:
        key = (row["seed_algo"], row["dim"], row["func"], row["run"])
        r = e13.loc[key]
        n += 1
        if row["seed"] != r.seed:
            bad.append(f"{key}: seed differs")
            continue
        rel = abs(row["raw_error"] - r.raw_error) / max(abs(r.raw_error), 1e-300)
        worst = max(worst, rel)
        if rel > 1e-12:
            bad.append(f"{key}: {row['raw_error']!r} vs stored {r.raw_error!r}")
    print(f"replayed {n} stored E13 runs (L-SHADE, SHADE, jSO; D=30 and 50; functions "
          f"{', '.join(f'F{f}' for f in args.funcs)}) with the switches at their defaults; "
          f"worst relative difference {worst:.2e}")
    if bad:
        print("FAILED:\n  " + "\n  ".join(bad))
        return 1
    print("PASS: the defaults are the released engine.")
    return 0


# ------------------------------------------------------------------ analysis helpers
def _runs(df, algo, dim, col="algo"):
    d = df[(df[col] == algo) & (df.dim == dim)]
    out = {f: d[d.func == f].sort_values("run").error.to_numpy(float) for f in FUNCS}
    if any(len(v) == 0 for v in out.values()):
        raise SystemExit(f"missing runs for {algo} at D={dim}")
    return out


def _vs_reference(ours, ref, label, dim):
    """validate_lshade_shade.compare, condensed to one summary row."""
    import contextlib
    import io
    from validate_lshade_shade import compare
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = compare(ours, ref)
    m = res["means"]
    sig = res["significant"]
    row = dict(part="vs_reference", label=label, dim=dim, n_comparable=m["n"],
               median_ratio=m["median"], gmean_ratio=m["gmean"], within2=m["within2"],
               within3=m["within3"], outside2=" ".join(f"F{f}" for f in m["outside2"]),
               n_different=len(sig),
               different=" ".join(f"F{f}{'-' if res['delta'][f] < 0 else '+'}" for f in sig),
               min_p_holm=float(min(res["p_holm"].values())))
    print(f"  {label:<46s} D={dim}: comparable {m['n']:2d}, median ratio {m['median']:.3f}, "
          f"geometric mean {m['gmean']:.3f}, within 2x {100 * m['within2']:.0f}%, "
          f"rank-sum differences (Holm) {row['different'] or 'none'}; smallest p_Holm "
          f"{row['min_p_holm']:.2f}")
    return row


def _paired(a, b, label, dim):
    """Two engines on the same seeds: identical runs, mean changes, signed-rank (Holm).

    Percentages are the change of a's mean error relative to b's."""
    praw, ratios, means, ident, total = [], {}, {}, 0, 0
    for f in FUNCS:
        x, y = np.round(a[f], 8), np.round(b[f], 8)
        ident += int(np.sum(x == y))
        total += len(x)
        praw.append(1.0 if np.all(x == y) else
                    float(stats.wilcoxon(x, y, zero_method="zsplit").pvalue))
        mx, my = float(x.mean()), float(y.mean())
        means[f] = (mx, my)
        if mx > ZERO and my > ZERO:
            ratios[f] = mx / my
    padj = dict(zip(FUNCS, holm(praw)))
    moved = {f: q for f, q in ratios.items() if q > 1.1 or q < 1 / 1.1}
    sig = [f for f in FUNCS if padj[f] < 0.05]
    pct = lambda f: (f"{100 * (ratios[f] - 1):+.1f}%" if f in ratios else
                     f"{means[f][0]:.3g} vs {means[f][1]:.3g}")
    row = dict(part="paired", label=label, dim=dim, identical_runs=ident, paired_runs=total,
               n_comparable=len(ratios),
               moved_10pct=" ".join(f"F{f}:{100 * (q - 1):+.0f}%" for f, q in moved.items()),
               n_moved_10pct=len(moved), n_different=len(sig),
               different=" ".join(f"F{f}({pct(f)})" for f in sig),
               min_p_holm=float(min(padj.values())))
    print(f"  {label:<46s} D={dim}: identical final errors in {ident} of {total} paired runs; "
          f"mean error moves by more than 10% on {len(moved)} of {len(ratios)} comparable "
          f"functions ({row['moved_10pct'] or 'none'}); Holm-significant: "
          f"{row['different'] or 'none'}")
    return row


def _sensitivity(method, runs, label, dim):
    dl = np.array([cliffs_delta(method[f], runs[f]) for f in FUNCS])
    row = dict(part="sensitivity", label=label, dim=dim, mean_delta=float(dl.mean()),
               wins=int((dl < 0).sum()), losses=int((dl > 0).sum()))
    print(f"  {label:<46s} D={dim}: mean delta(SEHHO-COBL-R vs these L-SHADE runs) "
          f"{row['mean_delta']:+.4f}, functions won/lost {row['wins']}/{row['losses']}")
    return row


def cmd_analyze(args):
    e13 = pd.read_csv(E13_CSV, float_precision="round_trip")
    e13 = e13[e13.tag == E13_TAG]
    var = pd.read_csv(args.variants_csv, float_precision="round_trip")
    cpp = pd.read_csv(args.cpp_csv, float_precision="round_trip") \
        if os.path.exists(args.cpp_csv) else None
    rows = []
    ref = {}
    if args.lshade_dir:
        import contextlib
        import io
        from validate_lshade_shade import load_reference, REFERENCES
        for dim in args.dims:
            with contextlib.redirect_stdout(io.StringIO()):
                ref[dim], ok = load_reference(args.lshade_dir, "L-SHADE", dim,
                                              REFERENCES["LSHADE"]["digest"][dim])
            if not ok:
                print(f"WARNING: the L-SHADE reference files at D={dim} are not the published "
                      f"ones this script was checked against")
    for dim in args.dims:
        print("\n" + "=" * 100)
        print(f"CEC2014, D={dim}, 10,000*D evaluations")
        released = {"LSHADE": _runs(e13, "LSHADE", dim), "SHADE": _runs(e13, "SHADE", dim)}
        v = {n: _runs(var, n, dim) for n in VARIANTS}
        if ref:
            print(" (a) each L-SHADE engine against the 51 published runs of lshade.cc 1.0.0")
            for lab, runs in (("released [absorbing, parent]", released["LSHADE"]),
                              ("variant [reaccumulate, parent]", v["LSHADE[reaccumulate,parent]"]),
                              ("variant [absorbing, trial]", v["LSHADE[absorbing,trial]"]),
                              ("variant [reaccumulate, trial] = lshade.cc 1.0.0",
                               v["LSHADE[reaccumulate,trial]"])):
                rows.append(_vs_reference(runs, ref[dim], "L-SHADE " + lab, dim))
        print(" (b) paired effects of the switches (E13 seeds; a vs b)")
        rows.append(_paired(released["LSHADE"], v["LSHADE[reaccumulate,parent]"],
                            "L-SHADE terminal: absorbing vs reaccumulate", dim))
        rows.append(_paired(v["LSHADE[reaccumulate,parent]"], v["LSHADE[reaccumulate,trial]"],
                            "L-SHADE archive: parent vs trial (reaccumulate)", dim))
        rows.append(_paired(released["LSHADE"], v["LSHADE[absorbing,trial]"],
                            "L-SHADE archive: parent vs trial (absorbing)", dim))
        rows.append(_paired(released["SHADE"], v["SHADE[none]"],
                            "SHADE terminal: absorbing vs none", dim))
        if cpp is not None:
            print(" (c) SHADE against our runs of Tanabe's SHADE 1.0.1 C++ code (51 runs)")
            c = _runs(cpp, "SHADE-1.0.1-C++", dim)
            rows.append(_vs_reference(released["SHADE"], c, "SHADE released [absorbing]", dim))
            rows.append(_vs_reference(v["SHADE[none]"], c, "SHADE [none] (SHADE 1.0 update)", dim))
        print(" (d) the paper's contrast SEHHO-COBL-R vs L-SHADE (E13), by L-SHADE runs used")
        method = _runs(e13, "GF-Method", dim)
        rows.append(_sensitivity(method, released["LSHADE"], "released engine (E13)", dim))
        for n in ("LSHADE[reaccumulate,parent]", "LSHADE[absorbing,trial]",
                  "LSHADE[reaccumulate,trial]"):
            rows.append(_sensitivity(method, v[n], "variant " + n, dim))
        if ref:
            rows.append(_sensitivity(method, ref[dim], "published runs of lshade.cc 1.0.0 (51)",
                                     dim))
    if args.summary_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.summary_csv)), exist_ok=True)
        pd.DataFrame(rows).to_csv(args.summary_csv, index=False)
        print(f"\nwrote {args.summary_csv}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the engine variants with E13's seeds")
    r.add_argument("--dims", type=int, nargs="+", default=[30, 50])
    r.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=None)
    r.add_argument("--runs", type=int, default=30)
    r.add_argument("--workers", type=int,
                   default=int(os.environ.get("SEHHO_WORKERS", 0)) or max(1, (os.cpu_count() or 4) - 8))
    r.add_argument("--out", default=VARIANT_CSV)
    c = sub.add_parser("check-defaults", help="replay stored E13 runs with default switches")
    c.add_argument("--funcs", type=int, nargs="+", default=[1, 10, 30])
    c.add_argument("--runs", type=int, default=2)
    c.add_argument("--workers", type=int, default=36)
    a = sub.add_parser("analyze", help="print and write the validation statistics")
    a.add_argument("--lshade-dir", default=os.environ.get("LSHADE_REF_DIR"),
                   help="directory with L-SHADE_<f>_<D>.txt (see validate_lshade_shade.py)")
    a.add_argument("--dims", type=int, nargs="+", default=[30, 50])
    a.add_argument("--variants-csv", default=VARIANT_CSV)
    a.add_argument("--cpp-csv", default=CPP_CSV)
    a.add_argument("--summary-csv", default=None)
    args = ap.parse_args(argv)
    return {"run": cmd_run, "check-defaults": cmd_check_defaults, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
