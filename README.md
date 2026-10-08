# SEHHO-COBL — reproducibility package

Everything needed to regenerate every number, table and figure in the
manuscript: the optimisers, the benchmark suites, the per-run raw results, the
seeds that produced them, and the analysis scripts. Read
[Known deviations and disclosures](#known-deviations-and-disclosures) before
relying on any single file.

Public repository: https://github.com/YifuZhao-mpu/hho-de-population-budget
(the release cited in the manuscript is tagged `v1.1` and archived on Zenodo,
https://doi.org/10.5281/zenodo.23227108). Code is MIT-licensed
and data CC BY 4.0; third-party benchmark sources keep their authors' terms
(see [Licences and access](#licences-and-access)).

## Quick start

```bash
cd cec_native && ./build.sh && cd ..        # compile the official CEC suites
python3 scripts/validate_benchmarks.py      # benchmark correctness checks
python3 scripts/validate_jso.py 30 --compare-only --rez-dir <jSO-SOURCE-RESULTS> \
        --summary-csv results/analysis/validation_jso.csv
                                            # stored jSO runs vs the jSO authors' published results
python3 scripts/validate_jso.py 30 30 --rez-dir <jSO-SOURCE-RESULTS>
                                            # ... or re-run them first (rewrites results/validate_jso_D30.csv)
python3 scripts/validate_lshade_shade.py --lshade-dir <L-SHADE> --shade-dir <SHADE11> \
        --summary-csv results/analysis/validation_lshade_shade.csv
                                            # stored L-SHADE/SHADE runs (E13) vs Tanabe & Fukunaga's published runs
                                            # (--summary-csv: statistics for the supplementary validation table)
python3 scripts/validate_engine_variants.py run          # the engine with its two switches set to the
                                            # C++ codes' behaviour, E13's seeds -> results/validation_engine_variants_cec2014.csv
python3 scripts/validate_engine_variants.py check-defaults
                                            # the switches at their defaults replay stored E13 runs bit for bit
python3 scripts/shade101_cpp/run_shade101.py --shade-zip <SHADE1.0.1_CEC2013.zip> --lshade-zip <LSHADE1.0.0_CEC2014.zip>
                                            # Tanabe's SHADE 1.0.1 C++ code on our CEC2014 data (code not redistributed)
                                            # -> results/validation_shade101_cpp_cec2014.csv (add --check to compare)
python3 scripts/validate_engine_variants.py analyze --lshade-dir <L-SHADE> \
        --summary-csv results/analysis/validation_engine_variants.csv
                                            # every validation statistic of Supplementary Section S2.3
python3 scripts/cec2020_entries.py <2020-RW-Constrained-Optimisation folder>
                                            # checkpoint figures of the CEC2020 competition entries quoted in main text
                                            # Section 4.3 and Supplementary S6.1/S6.8, from the organisers' per-run files
                                            # (sha256 checked; not redistributed; needs openpyxl, see requirements.txt)
                                            # -> results/analysis/cec2020_entries.csv
python3 scripts/resolution_check.py          # the precision check of main text Section 2.6: every analysis
                                            # that reads per-run errors, and every table, re-run on errors
                                            # rounded to 1e-8 (post hoc; a few minutes; the figure and
                                            # validation scripts are not re-run)
                                            # -> results/analysis/resolution_changes.csv,
                                            #    results/analysis/resolution_table_cells.csv
python3 scripts/run_experiments.py          # E1-E6: the re-evaluation of SEHHO-COBL
python3 scripts/run_experiments.py E4c      # the same ablation at 15,000 evaluations
python3 scripts/run_eng_budget.py           # E5b: feasibility vs budget on RC01/RC06
python3 scripts/run_experiments.py E7       # population rule search (design suite)
python3 scripts/analyze_poprule.py          # ... and the choice it makes
python3 scripts/run_experiments.py E8 E11   # composition of the repaired method
python3 scripts/analyze_composition.py      # ... the parts it drops (prints the original and the amended rule)
python3 scripts/run_experiments.py E9 E10 E12   # the repaired method vs. baselines
python3 scripts/run_experiments.py E13      # confirmatory suite (CEC2014)
python3 scripts/analyze_cec2014.py          # ... the pre-registered analysis
python3 scripts/run_experiments.py E14      # population rule isolated inside L-SHADE and SEHHO-COBL-R
python3 scripts/analyze_pop_isolation.py    # ... its contrasts (CEC2014 part exploratory)
python3 scripts/run_e15_original_rule.py    # E15: the configuration the original composition rule gives
python3 scripts/analyze_e15.py              # ... its pre-registered (exploratory) analysis
python3 scripts/run_e16_engineering_population.py
                                            # E16: L-SHADE at N_init = 6D vs 18D on the five design problems
python3 scripts/analyze_e16.py              # ... its pre-registered (exploratory) analysis
python3 scripts/test_reproduction.py        # published configurations still reproduce
python3 scripts/rc01_analytic.py            # closed-form feasible reference design of CEC2020 RC01
python3 scripts/run_eng_solver_baseline.py  # equality elimination + SciPy solvers
python3 scripts/run_traces.py               # convergence traces (per-checkpoint mean and median only; not plotted)
python3 scripts/analyze.py                  # legacy six-algorithm statistics -> results/analysis/ (see below)
python3 scripts/analyze_gate_factorial.py   # 2x2 gate x Levy factorial, both budgets
python3 scripts/analyze_gate_inference.py   # intervals + equivalence + interaction tests
python3 scripts/analyze_extras.py           # numbers quoted in the text but held by no table
                                            # (suite gap, RC01 Fisher test, N120b control, E6 effect,
                                            # E15 configuration vs SEHHO-COBL-R on CEC2022, HHO at the
                                            # CEC2014 zero shift) -> results/analysis/extras.csv
python3 scripts/classical_references.py     # classical reference designs of the five design problems,
                                            # with evaluation counts -> results/analysis/revision_classical_refs.csv
python3 scripts/check_cec_data.py           # singular values of the CEC matrices; CEC2022 input data
                                            # shared with CEC2017/CEC2014 -> results/analysis/revision_*.csv
python3 scripts/analyze_revision.py         # re-analyses requested in an internal pre-submission review (see "Revision analyses")
                                            # -> results/analysis/revision_*.csv
python3 scripts/make_tables.py              # LaTeX tables -> results/tex/
python3 scripts/make_figures.py             # figures -> results/figures/
python3 scripts/check_integrity.py          # budget / run-count audit of every CSV
```

`<jSO-SOURCE-RESULTS>` is the directory extracted from the jSO authors' archive
(see [Validation](#validation)); it can also be given as the environment
variable `JSO_REZ_DIR`. Likewise `<L-SHADE>` and `<SHADE11>` are folders of
Tanabe & Fukunaga's published CEC2014 result archives (environment variables
`LSHADE_REF_DIR` and `SHADE11_REF_DIR`).

### Using an optimiser on your own problem

Every optimiser takes a `Budget` (which counts every evaluation and refuses to
exceed the limit) and a NumPy generator. A constrained problem subclasses
`EngProblem`, which applies Deb's feasibility rule exactly as in the paper
(equality tolerance 1e-4, feasible when the total violation is at most 1e-8).
Run from the package root, or put it on `PYTHONPATH`:

```python
import numpy as np
from sehho.algorithms import lshade_n
from sehho.bench import Budget
from sehho.engineering import EngProblem


class Bracket(EngProblem):
    """Minimise f(x) subject to g(x) <= 0 (one inequality here)."""
    name = "bracket"
    dim = 3
    n_ineq = 1
    LB = [0.1, 0.1, 0.1]
    UB = [5.0, 5.0, 5.0]
    f_ref = 1e3                      # must exceed f everywhere in the box (checked at run time)

    def objective(self, X):          # X holds one candidate per row
        return X[:, 0] * X[:, 1] + 2.0 * X[:, 2]

    def constraints(self, X):        # (G, H): G <= 0 and H = 0, one column per constraint
        G = (4.0 - X[:, 0] * X[:, 1] * X[:, 2])[:, None]     # x1 x2 x3 >= 4
        return G, None


problem = Bracket()
budget = Budget(problem, max_fes=15_000)
best = lshade_n(budget, np.random.default_rng(1), n_factor=6)   # L-SHADE with N_init = 6D
print(best, problem.report(budget.best_x))   # 5.6582..., feasible (optimum 4*sqrt(2) = 5.6569)
```

`lshade_n` is stock L-SHADE with only the initial population changed
(`n_factor` times the number of variables); choose `n_factor` for the budget,
not from the default (the paper's Table 10 gives the evidence by evaluations per
variable). Integer or discrete variables are handled by overriding
`EngProblem.repair`, as `PressureVessel` and `BatchPlant` in
`sehho/engineering.py` do; equality constraints are returned as the second
element `H` of `constraints`.

### Three suites, three roles

* **CEC2017 — diagnostic.** Stages E4b, E4c and E6 use it to find out *what* is
  wrong with SEHHO-COBL (the gate ablation and the population sweep). Choosing
  which component to repair is a design decision, so **CEC2017 is not a holdout
  for the repaired method** and the paper does not treat it as one.
* **CEC2022 — selection.** Stages E7, E8 and E11 choose every constant and every
  keep-or-drop decision, writing `results/analysis/selected_config.json` and
  `results/analysis/composition_decision.json`. Stages E9, E10 and E12–E14 read
  those through `_final_cfg()`.
* **CEC2014 — confirmatory.** Stage E13. No design decision was taken on its
  results. Its data are not wholly separate from the selection suite: CEC2022
  F6, F8, F9 and F10 reuse the D=10 and D=20 rotation matrices of CEC2014 F18,
  F22, F23 and F24 (`scripts/check_cec_data.py`), which was not considered when
  the roles were assigned and was identified in pre-submission checking. The
  configuration, the primary endpoint and its test, three secondary predictions
  stated in words and the success criterion were fixed in
  `results/analysis/PREREGISTRATION_cec2014.json`, an internal pre-registration
  timestamped inside the file (2026-09-15T08:30:15Z), **before** the stage was
  run; the hash it records covers the frozen configuration only (see
  [What the pre-registration hash covers](#what-the-pre-registration-hash-covers)).
  Nothing in its output was permitted to change the configuration.
  `analyze_cec2014.py` implements that pre-registration and refuses to run if
  the registered configuration is not the one E13 runs with. The later stages
  E14 and E15 also use CEC2014; their comparisons are not in that
  pre-registration and are reported as exploratory.

The same contrast on CEC2017, the suite on which the repair was diagnosed, is
0.10–0.15 larger in magnitude in Cliff's delta than on CEC2014 (0.146 at D=30,
0.097 at D=50), the sign expected if diagnosing on it inflated the estimate,
but the bootstrap interval of the difference includes zero
(`scripts/analyze_extras.py`) and the gap cannot be separated from differences
between the suites.

The configuration obtained by dismantling SEHHO-COBL is called *the repaired
configuration* in the paper's text and **SEHHO-COBL-R** in its tables and
figures; it is `GF-Method` in this code and in every raw CSV. It is not renamed
in the data because `make_seed()` hashes the algorithm name, so renaming would
make the released seeds irreproducible; `make_tables.py` and `make_figures.py`
map the name at display time. L-SHADE is `LSHADE` in the data for the same
reason and is mapped the same way. Comments, docstrings and printed labels of
analysis scripts written before the renaming call the repaired configuration
"R-SHADE", the name used in an earlier draft; it is not Tanabe & Fukunaga's
restart SHADE. Scripts whose
sha256 is recorded in a pre-registration are left exactly as registered.

### The two gate-ablation analyses, and why there are two

`analyze_gate_factorial.py` reports the 2x2 the way an ablation table usually
is reported: Friedman ranks within the four cells, and a signed-rank test on
per-function **mean errors**. That answers "which configuration is better".

`analyze_gate_inference.py` answers the two questions a p-value cannot. Saying
the Levy perturbation *has no effect* is a claim of absence, and a large
p-value is equally consistent with absence and with low power; saying the gate
matters *more at a larger budget* is a claim that two effects differ, which
does not follow from one being significant and the other not (Gelman & Stern,
2006). Both are therefore answered with intervals: each contrast is summarised
per function by Cliff's delta over the 30 runs, aggregated over the 29
functions with a 95% percentile bootstrap interval, and compared against the
negligible margin |delta| < 0.147 of Romano et al. (2006), which is declared in
that script; the script's file was last modified 32 minutes after the
mean-error factorial output was written (file times record the last edit, not
creation). Holm is applied within each of three families.
`analyze_revision.py` adds BCa, Student-t and family-adjusted intervals for the
same contrasts and the verdicts at the margins 0.05 and 0.10 as well as 0.147.

The two scripts summarise the same runs differently and disagree on one
contrast (the gate effect with Levy off at 300,000 evaluations: p_Holm = 0.169
on mean errors, 0.0016 on Cliff's delta). Both numbers are in the manuscript,
with the discrepancy explained rather than resolved in the paper's favour.

Requirements: Python with `numpy`, `scipy`, `pandas`, `matplotlib`, and a C++
compiler; no MATLAB is required. `requirements.txt` pins the exact environment in
which every result was verified: Python 3.13.12, NumPy 2.4.4, SciPy 1.17.1,
pandas 3.0.2, Matplotlib 3.10.8 (and their runtime dependencies); the CEC
libraries were built by `cec_native/build.sh` with g++ 11.4.0 (Ubuntu
11.4.0-1ubuntu1~22.04.3, `-O3 -fPIC -shared`) on Ubuntu 22.04.5 LTS (x86-64).
Install with `python3 -m pip install -r requirements.txt`.

## What is in here

```
cec_native/     official CEC2014, CEC2017 and CEC2022 C/C++ sources + their
                input data, compiled to libcec14/17/22.so.  The CEC2014 port is
                produced by patch_cec14.py rather than hand-edited, so the diff
                against the competition source is auditable.
sehho/
  bench.py      ctypes bindings and the hard function-evaluation budget
  algorithms.py SEHHO-COBL (with every ablation switch) and the baselines
  engineering.py five constrained design problems, Deb feasibility rule
  stats.py      Friedman, rank-sum, Holm correction, Cliff's delta, A12
  intervals.py  percentile / BCa / Student-t / Bonferroni-adjusted intervals for a mean
                of per-function deltas, margin verdicts, two-stage bootstrap
  runner.py     deterministic seeding and parallel execution
scripts/        experiment drivers, validation, analysis, tables, figures
results/        per-run raw CSVs, analysis tables, LaTeX tables, figures; the
                validation re-runs of Supplementary Section S2.3 are
                results/validation_engine_variants_cec2014.csv (our engine with its
                two switches) and results/validation_shade101_cpp_cec2014.csv (our
                runs of Tanabe's SHADE 1.0.1 C++ code, which is not redistributed)
results/legacy/ superseded raw files kept for transparency; no script reads them
scripts/shade101_cpp/  harness and build script for the SHADE 1.0.1 code-to-code check
requirements.txt exact package versions of the verified environment (and the compiler)
```

## Benchmark provenance

The landscapes are **not re-implemented**. `cec_native/` contains the
competition C/C++ sources distributed by P. N. Suganthan, together with their
official `input_data` shift/rotation/shuffle files:

* CEC2014 — `cec14_test_func.cpp`, turned into `patched_cec14_test_func.cpp` by
  `patch_cec14.py`
* CEC2017 — `cec17_test_func.cpp` from *CEC2017-BoundContrained*
* CEC2022 — `cec22_test_func.cpp` from *2022-SO-BO*

Five mechanical edits were applied. For CEC2014 they are made by
`cec_native/patch_cec14.py`, which writes `patched_cec14_test_func.cpp` from the
competition source, so the difference is auditable; for every suite they are
visible by diffing `patched_*.cpp` against the originals:

1. **Headers.** `#include <WINDOWS.H>` and `<malloc.h>` removed; `<stdlib.h>`
   and `<string.h>` included instead.
2. **Globals.** The global variables that the official `main.cpp` defines are
   defined in the library, so that it links on its own.
3. **Data directory.** The hard-coded `input_data/` prefix becomes a directory
   that can be set at run time.
4. **`%Lf` -> `%lf`, a genuine portability bug in the CEC2014 and CEC2017
   sources.** They read their rotation matrices and shift vectors with
   `fscanf(fpt, "%Lf", ...)` into `double` arrays. `%Lf` is the `long double`
   conversion; MSVC historically accepted it as `double`, but under glibc it
   writes an 80/128-bit value into a 64-bit slot, which silently corrupts every
   entry. All five occurrences in each source (two in the matrix loader, three
   in the shift loader) were changed to `%lf`. Without this fix the suites
   return `inf`/`nan`.
5. **C linkage** for the test function, so that `ctypes` can find it.

The CEC2022 source carries edits 1-3 and 5; it already reads its data with
`%lf`. The `patched_*.cpp` build copies also normalise CRLF line endings to LF,
which changes no code. `cec_native/build.sh` only compiles the three patched sources.

No objective or constraint expression was modified.

### Validation

`scripts/validate_benchmarks.py` performs three checks:

* **CEC2022 against the organisers' own Python reference** (shipped in the same
  archive): agreement is exact — maximum relative difference `0.000e+00` over
  all 12 functions × {10, 20} dimensions.
* **CEC2017 against `cec2017-py`**, an independent MIT-licensed port. Where the
  two disagree the discrepancy is traceable to the *port*, not to the
  competition source: `cec2017-py` multiplies the Bent Cigar tail by `10e6`
  (= 1e7) instead of the `pow(10.0, 6.0)` of the C code, which accounts for the
  large disagreements on F1 and every hybrid/composition that uses Bent Cigar.
  The check needs `cec2017-py` installed, or its folder named in the environment
  variable `CEC2017_PY_DIR`; without it the script skips this check.
* **Optimum recovery**: `f(o) − f*` is 0 for all CEC2022 functions. On CEC2017
  the residuals are the documented artefacts of the original C code (the
  truncated Schwefel constant, ~1e-4, and the Levy offset, ~0.096·D).

`scripts/validate_jso.py` is the end-to-end check. It compares this package's
jSO port, run on CEC2017 at the competition protocol (D=30, 300,000
evaluations, 30 runs per function, stored in `results/validate_jso_D30.csv`),
against the result files Brest et al. shipped with their own jSO source
archive. Those files are not redistributed here (their licence is not stated);
they are in the official CEC2017 repository:

| item | value |
|---|---|
| repository | https://github.com/P-N-Suganthan/CEC2017-BoundContrained (branch `master`) |
| archive | `Codes-of-Top-Methods-and-results.zip`, 22,821,612 bytes, sha256 `0249e3cc39070cc9d16578fa64ddc88aa2abe1ee72c5147b595707ac6502e361` |
| inside it | `jSO-SOURCE-RESULTS-29May2017.tar.gz`, sha256 `720f958ebb1ab2c02aaea4be408314325773c72ba5912fb4853a94fdf11b251f` |
| reference file | `jSO-SOURCE-RESULTS/D30.rez` (`fNN best worst median mean std`), sha256 `03e31fe64ff1ada98afab3688ca70ea4232248b78aa690cf8d94d6d4114df2f4` |

Extract the tarball and pass the `jSO-SOURCE-RESULTS` directory with
`--rez-dir` (or `JSO_REZ_DIR`). The script prints the reference file's sha256
and warns if it is not the file above. `--compare-only` recomputes the
statistics from the stored runs without optimising anything:

| statistic | value |
|---|---|
| comparable functions (both means > 1e-8), D=30 | 25 of 29 |
| median ratio (ours / published) | 1.000 |
| geometric-mean ratio | 1.010 |
| functions within 2× of published | 96 % |
| functions within 3× of published | 100 % |

This single test exercises the compiled benchmark, the jSO port and the budget
accounting together.

`scripts/validate_lshade_shade.py` does the same for L-SHADE and SHADE on
CEC2014. It compares the stored E13 runs (D=30 and 50, 10,000·D evaluations,
30 runs per function) with per-run result files that Tanabe & Fukunaga
published for the same suite source, input data and budget (51 runs). Nothing
is optimised. The files are not redistributed here (no licence is stated):

| item | value |
|---|---|
| L-SHADE reference | the CEC2014 competition entry (`lshade.cc` 1.0.0): https://github.com/P-N-Suganthan/CEC2014 (branch `master`), `CEC-2014-Results-PartA.zip`, 6,002,877 bytes, sha256 `355b427798ba7fa7826fb65a1dae7b030ee8634b6790b1d55c4dd02d5cb1f143` → folder `L-SHADE/` |
| SHADE reference | SHADE 1.1 runs from Tanabe's data release `Tanabe-CEC14-results.zip`, 6,577,575 bytes, sha256 `7dc2be55d6489288b7614b4cc14256b882db6e58ba729c566603877ea06a410c` → folder `Tanabe-CEC14-results/SHADE11/`. The original address (`sites.google.com/site/tanaberyoji/data/Tanabe-CEC14-results.zip`) no longer resolves; the Internet Archive copy is http://web.archive.org/web/20201016162111id_/https://sites.google.com/site/tanaberyoji/data/Tanabe-CEC14-results.zip?attredirects=0 . Its `L-SHADE/` folder is byte-identical to the competition files. |
| file format | `<name>_<f>_<D>.txt`: 14 rows (error after 0.01 … 1.0 × MaxFES) × 51 runs; the script uses row 14 |
| digests checked | sha256 of the 30 files of one dimension concatenated F1…F30: L-SHADE D=30 `8d4e25ca…`, D=50 `4281c427…`; SHADE 1.1 D=30 `9de73ba8…`, D=50 `ab071caa…` (full values in the script) |

Pass the folders with `--lshade-dir` and `--shade-dir`. Besides the jSO
statistics, for means and for medians, the script lists every function outside
2× or with only one side ~0 and, because both sides are per-run data, runs a
two-sided rank-sum test per function (Holm over the 30 functions; errors
compared at the CEC resolution of 1e-8):

| statistic (means) | L-SHADE D=30 | L-SHADE D=50 | SHADE 1.1† D=30 | SHADE 1.1† D=50 |
|---|---|---|---|---|
| comparable functions (both > 1e-8) | 24 of 30 | 26 of 30 | 25 of 30 | 27 of 30 |
| median ratio (ours / published) | 1.000 | 1.001 | 1.003 | 0.995 |
| geometric-mean ratio | 1.696 (0.997 without F6) | 0.995 | 0.991 | 0.901 |
| within 2× / 3× | 92 % / 96 % | 96 % / 100 % | 92 % / 100 % | 89 % / 96 % |
| rank-sum differences (Holm) | F10↓ F18↑ F30↑ | F6↓ F10↓ F18↑ | F6↓ F10↓ F25↑ F28↓ | F6↓ F9↑ F19↑ F24↓ F25↑ |

↓ ours lower, ↑ ours higher. † SHADE 1.1 is not the SHADE implemented here
(that is SHADE 1.0: H=100 not D, |A|=N not 2N, p_i ~ U[2/N, 0.2] not 0.1,
arithmetic not Lehmer M_CR), so those two columns are a plausibility check, not
a validation.

The L-SHADE outliers are F6 at D=30 (ratio 3.5·10^5: 2 of our 30 runs end at
0.46 and 0.98, the largest of the 51 published errors is 7·10^-6; both medians
are 0) and F10 at both dimensions (0.34 and 0.38, ours lower). The rank-sum
differences come from the two code-level differences listed under [Known
deviations and disclosures](#known-deviations-and-disclosures): the absorbing
terminal CR value (F10) and the parent archive (F18 and F30 at D=30, F6 and F18
at D=50). The engine was re-run, with E13's seeds, with both set to the
competition code's behaviour: no function then differs from the published runs,
at D=30 or at D=50. The two rules are keyword switches of `_shade_family`
(`terminal`, `archive_trial`) whose defaults are the behaviour of every
experiment, and no random draw depends on them, so a variant run is paired with
the E13 run of the same seed. `scripts/validate_engine_variants.py run` makes
the runs (L-SHADE with each switch and both, SHADE with SHADE 1.0's own memory
update; D=30 and 50, all 30 functions, 30 runs: 7,200 runs, in
`results/validation_engine_variants_cec2014.csv`), `check-defaults` replays
stored E13 L-SHADE, SHADE and jSO runs through the same entry points with the
switches at their defaults (bit for bit), and `analyze` prints every statistic
quoted in Supplementary Section S2.3 and main text Section 2.4
(`results/analysis/validation_engine_variants.csv`; the comparisons with the
published runs need `--lshade-dir`, the folder described above).

We found no published SHADE 1.0 results on CEC2014 (sources checked in October
2026: the CEC2014 competition results, Tanabe's archived software and data
pages, and the SHADE and L-SHADE papers). In their place Tanabe and Fukunaga's
own SHADE 1.0.1 C++ code was run on this package's CEC2014 data, 51 runs per
function at D=30 and 50. The code is not redistributed; our runs are
(`results/validation_shade101_cpp_cec2014.csv`), and
`scripts/shade101_cpp/run_shade101.py` rebuilds and re-runs them from the
official files, which it checks by sha256 before use:

| item | value |
|---|---|
| SHADE 1.0.1 for CEC2013 (Tanabe) | original address `https://sites.google.com/site/tanaberyoji/software/SHADE1.0.1_CEC2013.zip` (no longer resolves); Internet Archive copy http://web.archive.org/web/20201016162057id_/https://sites.google.com/site/tanaberyoji/software/SHADE1.0.1_CEC2013.zip?attredirects=0 ; `SHADE1.0.1_CEC2013.zip`, 1,774,242 bytes, sha256 `92fafa12f90149f5df00402770b9ddb3961a922bfcef3fd7579e2d7440737d86` |
| CEC2014 test function used with it | Tanabe's `cec14_test_func.cc`, sha256 `eb7da784d9e027c02c333fb0ca4000be829287a45e781714f71fd567c5584be6`, folder `LSHADE_CEC14/` of `LSHADE1.0.0_CEC2014.zip` (Internet Archive http://web.archive.org/web/20201016162105id_/https://sites.google.com/site/tanaberyoji/software/LSHADE1.0.0_CEC2014.zip?attredirects=0 , sha256 `233bb102ca165329099a2fa0721ceae82ae1ebfb4755e24848e54df1868739d9`); the same file is in `Top-Methods-Part-A.rar` of the official CEC2014 repository |
| edits | the evaluation call (`test_func` becomes `cec14_test_func`) and the optimum (F_i* = 100 i); nothing else |
| driver | `scripts/shade101_cpp/harness_main.cc` (this package): N = 100, H = 100, \|A\| = N, 10,000·D evaluations; run r of function f at dimension D is seeded with `srand(100000*D + 1000*f + r + 1)` |

Our SHADE differs from it on F5 and F12 at D=30 and on F5, F11, F12 and F16 at
D=50, always lower; without the terminal value, which SHADE 1.0 does not have,
no function differs (`validate_engine_variants.py analyze`).

## Function-evaluation accounting

Every optimiser reaches the objective only through `sehho.bench.Budget`, which
counts each candidate and raises `BudgetExceeded` rather than allowing an
overrun. Consequences worth stating explicitly:

* COBL's `4N` initial candidates are paid for **out of the same budget** every
  competitor receives, so the comparison is FE-matched rather than
  generation-matched.
* HHO's two rapid-dive branches consume extra evaluations; it therefore
  completes fewer iterations under an equal budget, which is the intended
  behaviour of an FE-matched protocol.
* The reported value is always the best objective ever *evaluated* within the
  budget, identical in definition across algorithms.

## Baseline provenance

| Algorithm | Source followed |
|---|---|
| SHADE | Tanabe & Fukunaga (2013), i.e. SHADE 1.0, implemented from the paper (not ported from author code): N=100, H=N, \|A\|=N, p_i ~ U[2/N, 0.2], weighted arithmetic mean for M_CR |
| L-SHADE | Tanabe & Fukunaga's own `lshade.cc` (version 1.0.0, the CEC2014 competition code): N=18D, H=6, \|A\|=2.6N, p=0.11, LPSR, parent-midpoint bound handling; the archive stores the replaced parent, as in their corrected 1.0.1 |
| jSO | Brest et al.'s own `lshade.cc` (jSO CEC2017 archive): N=25·√D·ln D, H=5, terminal memory slot fixed at 0.9, staged F/CR clamps, weighted F, p annealing |
| HHO | Heidari et al. (2019), all four siege branches |
| DE | DE/rand/1/bin, F=0.5 and CR=0.9 (first choices Storn & Price suggest), N fixed at 100 rather than their rule of thumb of 5D to 10D |

The three SHADE-family baselines share one engine (`algorithms._shade_family`);
its terminal-CR rule differs from the reference codes, and L-SHADE's archive
from the competition code, see
[Known deviations and disclosures](#known-deviations-and-disclosures).

## Reproducing an individual number

Seeds are deterministic and stored with every run:

```python
from sehho.runner import make_seed
make_seed("cec2017", 4, 30, "jSO", "E3_cec2017", 0)
```

The `seed` column of any results CSV is exactly this value, so a single run can
be replayed in isolation.

## Statistics

`sehho/stats.py` implements the comparison protocol of Derrac et al. (2011)
with two additions:

* **Holm step-down correction** applied across each whole family of tests — all
  functions × all competitors together, which is the level at which the
  aggregate win/tie/loss counts are formed;
* **effect sizes**: Cliff's delta with the Romano et al. magnitude thresholds,
  and the Vargha–Delaney A12 statistic. A12 is computed and tested but is not
  reported in any of the paper's tables, which use Cliff's delta throughout.

## Revision analyses

Everything an internal pre-submission review of the manuscript asked to be
recomputed is recomputed from the stored per-run CSVs; no optimiser is run (see
[The word "review" in file comments](#the-word-review-in-file-comments)).

**Sign convention.** Every table reports Cliff's delta for arm A against arm B,
A being the changed, new or first-named arm; negative values favour A
(`cliffs_delta(a, b) < 0` means `a` has the smaller errors). The ablation
tables, which earlier reported the full method against the variant, now report
the variant against the full method. Every delta a table prints is listed, with
its arm order, its source file and whether its sign was reversed relative to
that file, in `results/analysis/revision_delta_ledger.csv`;
`results/analysis/revision_table_index.csv` lists every table file with its
label and whether it belongs to the main text or the supplement.

**Scripts and outputs** (all under `results/analysis/`):

| Script | Output | Content |
|---|---|---|
| `analyze_revision.py gate` | `revision_gate_inference.csv` | gate x Levy factorial (E4b, E4c): percentile, BCa and Student-t intervals, Bonferroni (98.75%) versions within each Holm family of four, p_Holm on the delta and the mean-error scales, verdicts at \|delta\| < 0.147 / 0.10 / 0.05 |
| `analyze_revision.py pop` | `revision_pop_isolation.csv` | the one-constant population control (E14), every row with every interval |
| `analyze_revision.py cec2014` | `revision_cec2014_primary.csv`, `revision_cec2014_suitegap.csv`, `revision_cec2014_excluded_functions.csv`, `revision_cec2014_pairwise.csv`, `revision_precision.csv` | the registered endpoint with BCa / Student-t / two-block Bonferroni intervals; the same endpoint without F18, F22, F23, F24 (post hoc); the CEC2017 counterpart and the suite gap; signed-rank tests on per-function delta, Holm over the six competitors of a block, beside the Friedman post-hoc p-values of the committed analysis script `analyze_cec2014.py` (the pre-registration names no test for these secondary comparisons); per-function standard errors of delta at 30 runs and the two-stage bootstrap |
| `analyze_revision.py suites` | `revision_suite_pairwise.csv` | the same pairwise statistics for CEC2017 (E3+E9, E12) and CEC2022 (E1+E9, E2+E9) |
| `analyze_revision.py comp` | `revision_composition.csv`, `revision_composition_decisions.csv` | composition of the repaired configuration (E8, E11): every interval, and the decision the amended rule gives under each interval type and margin |
| `analyze_revision.py poprule` | `revision_poprule.csv`, `revision_poprule_ranks.csv` | the population-rule search (E7): intervals against the selected rule; ranks and positions per block |
| `analyze_revision.py popsize` | `revision_popsize.csv` | delta(N vs N=30) for N = 60, 120, 270, 540 (E6) |
| `analyze_revision.py e16` | `revision_e16.csv` | E16 with success rates, mean and median among feasible runs |
| `analyze_revision.py eng` | `revision_engineering_success.csv` | success rates of the seven algorithms (E5, E10); runs on the feasibility threshold |
| `analyze_revision.py guide` | `revision_budget_guide.csv` | delta(6D vs 18D) against evaluations per variable, from E14, E7 and E16 (Table 10, Figure 3) |
| `classical_references.py` | `revision_classical_refs.csv` | a classical reference design per problem with its evaluation count |
| `check_cec_data.py` | `revision_singular_values.csv`, `revision_cec2022_lineage.csv` | singular values of every transformation matrix; CEC2022 data identical to CEC2017 or CEC2014 data |
| `analyze_revision.py e15` | `revision_e15.csv` | E15's registered secondary comparison (original-rule configuration vs SEHHO-COBL-R) with BCa, Student-t and Bonferroni-adjusted (98.75%) intervals over its four blocks (Table S-e15; the adjustment is not part of the registration) |
| `analyze_revision.py contrast` | `revision_effect_contrast.csv` | the direct per-function contrast of the population effect (N=120 vs N=30, E6) with the gate effect (E4b), both at 300,000 evaluations on CEC2017 D=30, as main text Section 2.6 requires for a difference of effects |
| `validate_jso.py`, `validate_lshade_shade.py` with `--summary-csv` | `validation_jso.csv`, `validation_lshade_shade.csv` | the port-validation statistics behind the supplementary validation table (statistics only; the reference files are not redistributed) |
| `validate_engine_variants.py analyze` with `--summary-csv` | `validation_engine_variants.csv` | the engine-variant and SHADE 1.0.1 code-to-code statistics of Supplementary Section S2.3 (statistics only) |

`analyze_revision.py` draws, for every contrast an earlier script already
analysed, from a generator seeded and consumed exactly as in that script
(20260913 `analyze_gate_inference.py`; 20260914 `analyze_poprule.py`,
`analyze_composition.py`; 20260915 `analyze_cec2014.py`,
`analyze_pop_isolation.py`, `analyze_extras.py`), so its percentile intervals
are the stored ones; it checks this and prints any mismatch. BCa intervals are
read from the same 20,000 resamples; Student-t intervals are truncated to
[-1, 1], the range of Cliff's delta. Contrasts no earlier script computed use
seed 20261003. Inference is conditional on a suite's functions treated as
exchangeable instances of the problem class the suite represents.

Equivalence verdicts at a margin m: *negligible* when the whole interval lies
inside (-m, m); *nonzero* when it excludes zero; *inconclusive* otherwise. An
unadjusted interval (percentile, BCa or Student-t) that excludes zero with its
bound nearest zero within 0.02 of it is reported as a *tie* (main text, Section
2.6); family-adjusted intervals decide only whether a verdict is directional.
The column `tie_002` of the analysis files is broader: it also flags an
interval that spans zero with a bound within 0.02 of zero, and the tables use
it only where the interval excludes zero (`excludes_zero_tie` in
`make_tables.py` and `analyze_revision.py`). Ties are reported for the gate x
budget interaction with the Levy perturbation off (Table 3), for the CEC2014
D=30 competition block of the population control inside SEHHO-COBL-R (Table 5,
`°`, and Table 10, "tie"), and for four composition decisions (Table S-composition).

Where a table counts or bolds the lowest mean error ("#1" columns, the bold
marks of the per-function tables), exact ties for the lowest mean are credited
to every tied algorithm. In the W/T/L columns, a function whose two samples
agree run by run to within a relative 1e-5 (`numpy.allclose` in
`stats.ranksum_family`) is counted as not different without a rank-sum test.

In the engineering table (Table 7), bold marks the lowest mean among the
algorithms that are feasible in every run; a mean over only some of the runs is
conditional on feasibility, is marked `a` and is not compared.

Success on a design problem: feasible and (f - f*)/|f*| <= 1e-4, where f* is
the problem's reference value (`classical_references.py`; for RC01, RC06 and
RC14 it equals the best-known value listed by Kumar et al. 2020). The suite's
own success rate (Kumar et al. 2020) uses an absolute f - f* <= 1e-8; it is
reported beside ours (`success_suite_abs1e_8` in `revision_engineering_success.csv`,
`success_*_suite_abs1e-8` in `revision_e16.csv`, last column of the
supplementary E16 table).

Both configurations against HHO (`revision_hho.csv`, with the HHO rows of the
pairwise files) are tabulated in the supplement (`tab_supp_hho.tex`). For the
gate factorial (Table 3) and the composition step, `make_tables.py` also reads
the verdict of the Bonferroni-adjusted Student-t interval and marks, or lists
in the table note, every verdict it overturns.

Code changes in this revision: `sehho/intervals.py` is new (analysis only), and
the `jso()` docstring in `sehho/algorithms.py`, which called jSO the CEC2017
winner, now says that it ranked second, behind EBOwithCMAR. In release v1.1
`algorithms._shade_family` gained the two validation switches `terminal` and
`archive_trial`; their defaults are the behaviour of every experiment, and
`scripts/test_reproduction.py` (which now also re-executes SHADE, L-SHADE and
jSO cells) and `validate_engine_variants.py check-defaults` confirm that the
stored runs still reproduce. No other executable line of the optimisers
changed.

Not done, and why: the convergence traces were not re-run with per-run
logging. `traces.csv` holds only the per-checkpoint mean and median of 30 runs
and predates SEHHO-COBL-R, so it can show neither run-to-run dispersion nor the
repaired configuration; the convergence figures were therefore withdrawn from
the manuscript, and `make_figures.py` no longer draws them (the file is kept as
data). A normalised or maximum relative constraint violation cannot be computed
from the stored runs, which record only the total and the largest violation,
not the design vector.

The revision analyses were run in the environment pinned in `requirements.txt`.
Figures are drawn at the width at which the manuscript includes them (the
`PRINT_WIDTH` table in `scripts/make_figures.py`: 6.3 in = `\textwidth`, except
the phase-gate panel at 0.72 and the budget guide at 0.9 of it), so no text
prints below 8 pt; if an include width changes, change `PRINT_WIDTH` and re-run
the script.

## Licences and access

**Access.** The package is openly available, without registration or request,
from the public repository
https://github.com/YifuZhao-mpu/hho-de-population-budget; the release cited in
the manuscript is tagged `v1.1`. It contains the code, the per-run results of
every experiment (with seeds), including the validation re-runs of
Supplementary Section S2.3, the pre-registration files and the analysis,
table and figure scripts. Release v1.1 adds those validation re-runs and their
scripts to v1.0 and corrects tables and text: no per-run result or statistic of v1.0
changed, while one classification of the budget guide (`revision_budget_guide.csv`,
one row now a tie) and the '#1' counts of the tables (exact ties now credited to every
tied algorithm) were corrected.

**The authors' code** — everything under `sehho/` and `scripts/`, and
`cec_native/build.sh` and `cec_native/patch_cec14.py` — is original to this work
and is released under the MIT licence.

**The authors' data** — the per-run results and analysis outputs under
`results/`, including the pre-registration files in `results/analysis/` — are
released under the Creative Commons Attribution 4.0 International licence
(CC BY 4.0).

**Third-party material keeps its authors' terms.** The official benchmark
sources and their input data in `cec_native/` are redistributed from the
competition packages and remain the property of their authors (Liang, Qu,
Suganthan for CEC2014; Awad, Ali, Suganthan, Liang, Qu for CEC2017; Kumar,
Price, Mohamed, Hadi, Suganthan for CEC2022 and for the CEC2020 real-world
constrained problems RC01/RC06/RC14). The original sources are included
unmodified; the `patched_*.cpp` build copies differ from them only by the
documented mechanical edits of [Benchmark provenance](#benchmark-provenance)
(five for CEC2014 and CEC2017, four for CEC2022), of which the substantive one
changes `%Lf` to `%lf` in the CEC2014 and CEC2017 data readers.
`cec2017-py`, used only as an independent cross-check during validation, is
MIT-licensed (© 2022 Duncan Tilley). The published reference results used to
validate the jSO, L-SHADE and SHADE ports, and Tanabe's SHADE 1.0.1 C++ code run
for the SHADE check, are not redistributed (their sources and checksums are
listed above); `results/validation_shade101_cpp_cec2014.csv` holds only our runs
of that code.

## Known deviations and disclosures

**SEHHO-COBL is a re-implementation.** SEHHO-COBL, the authors' earlier
design, described in an unrefereed preprint, was originally implemented in MATLAB; that code is
not available, and `sehho/algorithms.py` re-implements it in Python from the
preprint's specification. Every result in this package labelled SEHHO-COBL concerns this
re-implementation. `scripts/validate_sehhocobl.py` (it only reads
`results/E1_cec2022_paper_budget.csv` and prints) compares it with the
per-function means of the earlier, MATLAB-based version of the manuscript on
CEC2022 at D=10 with 15,000 evaluations. F1 and F3 are not compared, because
the published mean error is zero (ours is 0 and 1.1e-5). On the other 10
functions the median ratio of our mean error to the published one is
0.986 and the geometric mean 0.704, and 8 of the 10 are within a factor of two.
The two outside are F11 (0.149: 41.7 against 280) and F5 (0.333); the next
largest discrepancy is F6 (0.574). In all three the re-implementation has the
*lower* error.

**E5b was re-run.** `results/E5b_engineering_budget.csv` (Table 19 of the paper)
was re-run on 2026-10-02 with the released code, after the floor→ceil fix to the
generation count in `algorithms.sehhocobl`. The earlier file, in which 180
SEHHO-COBL runs at 50,000, 200,000 and 10^6 evaluations stopped up to 20
evaluations short of their budget, is kept as
`results/legacy/E5b_engineering_budget_pre_ceil_fix.csv` for transparency. No
script reads `results/legacy/`.

**Terminal-CR semantics of the SHADE-family baselines.** In
`algorithms._shade_family` a memory slot whose successful CR values are all 0
becomes the terminal value ⊥ (−1, which makes every CR drawn from it 0), and the
rule is absorbing: once a slot is ⊥ it stays ⊥, and jSO's averaging keeps it at
⊥. This is the rule the L-SHADE paper specifies (its Algorithm 1; M_CR "will
remain fixed at ⊥ until the end of the search"), and it is the reference this
package follows. The released codes do something else. Tanabe's `lshade.cc`
(1.0.0, lines 199–221, unchanged in 1.0.1), SHADE 1.1's `shade.cc` and Brest et
al.'s jSO `lshade.cc` (lines 255–289 of that file in the archive above) all
zero the slot before re-accumulating it, so their `== -1` test never fires: ⊥
is set only by a generation in which every successful CR was 0, and jSO then
averages it with the slot's previous value instead of keeping it. SHADE 1.0
(2013) has no ⊥ at all, yet the shared engine applies it to SHADE too. SHADE is
implemented from the 2013 paper rather than ported from its authors' code.

The two rules agree more often than the description suggests, because once
every slot is ⊥ every successful CR is 0 and the re-accumulating rule sets ⊥
again; they diverge only while the memory is mixed. (A probe of how often ⊥ is
set, made with an instrumented copy of the engine during the revision, is not
part of the package and no number from it is used.) The effect on results was
measured with the engine's `terminal` switch (`scripts/validate_engine_variants.py`;
runs in `results/validation_engine_variants_cec2014.csv`): all 30 functions
re-run with E13's seeds, 30 runs.
For L-SHADE with a re-accumulating slot, final errors are identical in 562 of
900 paired runs at D=30 and 534 of 900 at D=50; mean errors move by more than
10 % on 3 of 24 comparable functions at D=30 (F10 −70 %, F12 +10 %, F22 +15 %
for the absorbing rule) and on 1 of 26 at D=50 (F10 −60 %); paired signed-rank
tests (Holm) find F22 at D=30 and F10 at D=50. Against SHADE without ⊥ (SHADE
1.0's own update), 522 and 521 of 900 runs are identical; the released rule's
mean errors differ by more than 10 % on F10 (+60 %) and F12 (−19 %) at D=30 and
on F10 (−50 %), F12 (−25 %) and F19 (−10 %) at D=50, with F5 (−0.3 % / −0.5 %),
F12, and at D=50 also F16 and F19 significant. The jSO port reproduces the
published jSO means at the competition budget (see [Validation](#validation)).
The paper's CEC2014 contrast of SEHHO-COBL-R with L-SHADE does not depend on the
choice: the mean Cliff's delta is +0.518 with the released L-SHADE and +0.527
with the competition code's behaviour at D=30 (+0.419 and +0.424 at D=50).

**L-SHADE's archive.** The engine archives the parent that a trial vector
replaced, as the L-SHADE paper describes and as Tanabe's corrected release
1.0.1 (June 2014) does. The competition code 1.0.0, which produced the
published runs used in [Validation](#validation), copies the trial vector
instead: it overwrites `pop[i]` with the child before archiving `pop[i]`, which
its author lists as a bug fixed in 1.0.1. In the same re-runs this difference,
not the terminal-CR rule, accounts for the published runs' lower errors on F18
and F30 at D=30 (our means are 48 % and 84 % higher with the parent archive than
with the trial-vector one; on F22 the parent archive is 27 % lower) and for the
F6 and F18 differences at D=50. Release 1.0.1 also changed the default memory
size and archive rate to H=5 and 1.4N; the engine keeps H=6 and 2.6N, the values
in the paper and in the code that produced the published runs.

**Seeds are per variant, not shared.** `make_seed()` hashes the suite,
function, dimension, algorithm or variant name, stage tag and run index, so two
variants compared in the same cell (for example the ablation variants of E4b,
or the same variant in E4b and E4c) never share seeds. What is identical across
variants is the seed *protocol*, not the seeds: comparisons are between
independent samples, not common random numbers. In E14, whose tag does not
encode the budget, the 15,000-evaluation and competition-budget runs of the
same arm do share seeds; every E14 contrast is within one budget.

**The composition rule was amended after its first output was seen.**
`scripts/analyze_composition.py` contains both rules and prints both
decisions. The rule as coded at 2026-09-14T07:48Z, before the E8 data existed,
kept a part unless it was negligible in every block; run on the E8 data at
09:01Z it kept everything except the p annealing. At 09:02Z, after that output
had been seen (and after stage E11, the stripped configuration, had been
launched), the rule was rewritten to also drop a part that never helps and has
an interval below zero somewhere. The two rules differ only on COBL and the
Levy perturbation (original: keep; amended: drop). The amended rule's decision
is the one in `results/analysis/composition_decision.json` and behind the
configuration frozen for CEC2014; running the current script reproduces that
file byte for byte. An earlier comment in the script, which described the rule as
fixed before the data existed, was wrong and has been removed.

**E15: a sensitivity check of that amendment.** Stage E15
(`scripts/run_e15_original_rule.py`, analysed by `scripts/analyze_e15.py`)
runs the configuration the original rule would have produced — the gate-free
method with LPSR from N_init = 6D and only the p annealing removed, COBL and
the Levy perturbation kept — in exactly the E13 cells (CEC2014, 30 functions,
D ∈ {30, 50}, competition and 15,000-evaluation budgets, 30 runs: 3,600 runs,
algorithm label `GF:OriginalRule`), so that the pre-registered primary contrast
can be computed for it too. It was registered before it was run, in
`results/analysis/PREREGISTRATION_E15_original_rule.json` (written
2026-10-02T11:05:38Z; file sha256 `a52a8345fa252331a4c5fc1bacdce9c4f954547c6f8ce52c0079fc04eb2c77c4`),
and the runner refuses to start if its configuration does not match that file.
That runner check is what ties the data to the registered configuration:
`analyze_e15.py` checks only that the configuration written in the
pre-registration matches the hash recorded beside it (and that E15 is
complete), although its docstring says it checks the configuration in the data;
the script is left as registered, because its sha256 is part of the
pre-registration.
Because it was designed after the confirmatory analysis and after the
amendment it examines, it is exploratory and is reported whatever it shows; it
cannot change the frozen configuration or the E13 result. On the design suite
the same configuration is the `-pAnneal` arm of stage E8; its contrast with
SEHHO-COBL-R there is computed by `scripts/analyze_extras.py`.

<a id="what-the-pre-registration-hash-covers"></a>
**What the pre-registration hash covers.** `PREREGISTRATION_cec2014.json`
records `frozen_configuration_sha256` = `55df5437…`, which is
`sha256(json.dumps(frozen_configuration, sort_keys=True))`: a hash of the
frozen configuration, **not** of the comparisons, statistic and success
criterion written in the same file. The sha256 of the whole file as shipped is
`f57c3e7d30c77bde8a3da8ae70b767671568e4f63ef9f9e3d469ebe4187f3eef`; it was
computed and recorded only afterwards and so does not itself prove when the
file was written. The file is an internal pre-registration with an
in-file timestamp (2026-09-15T08:30:15Z), not an external registry entry, and
no third-party timestamp exists. In `results/analysis/cec2014_verdict.json`
the field `preregistration_sha256` holds that same configuration hash; the file
is a timestamped artefact and is not regenerated, so the field keeps its name.
According to the authors' working log, `analyze_cec2014.py` was written at
08:32Z while E13 was running and was edited after the data existed (10:44Z),
to fix a crash in its secondary section; the primary output was the same
before and after. The pre-registration names no test for the secondary
comparisons; the Friedman post-hoc tests reported for them are this script's.
In the revision (2026-10-02) the script was edited again: it now checks, before
computing anything, that
the recorded hash matches the stored configuration and that this configuration
equals the one E13 runs with (`run_experiments._final_cfg()`), and refuses to
run otherwise; with those checks in place it reproduces the stored
`cec2014_verdict.json` and `cec2014_confirmatory.csv` byte for byte.

**`"held_out"` in `selected_config.json` is historical.** The file was written
by `analyze_poprule.py` on 2026-09-14, when CEC2017 was still described as
held out. It is a timestamped design-stage artefact and is deliberately not
edited; no code reads that field (stages read only `"winner"`). CEC2017 is the
diagnostic suite, and the current `analyze_poprule.py` writes the same
information under `"diagnostic_suite"`.

**The "+Gate" composition variant uses a greedier gate.** In E8, "+Gate" puts
the escape-energy gate back into the gate-free method, which uses population
reduction. `sehhocobl` evaluates the gate on t/T with T computed from the
*initial* population, while under LPSR the run lasts about three to four times
T generations, so the gate closes after about 40% of the budget instead of 50% and explores
on about 12–13% of updates instead of 15.3% (approximate values that follow
from the LPSR schedule; the package holds no instrumented runs of this). The "+Gate" row of the composition table therefore
measures a somewhat greedier gate than the published one; the main gate
ablation (E4/E4b/E4c, fixed population, exact T) is unaffected.

**E14's L-SHADE at 18D is a separate run.** The `LSHADE-N18` arm of E14 is stock
L-SHADE re-run with E14's own seed stream (tag `E14_popisolation_*`), not the
L-SHADE runs of E3/E12/E13, so the "SEHHO-COBL-R vs L-SHADE at 18D" contrast in the
population-isolation table differs from the same contrast in the main tables
by sampling error (at most 0.024 in Cliff's delta). The CEC2014 half of E14 is
not in the CEC2014 pre-registration and is exploratory.

**Legacy six-algorithm analysis outputs.** `scripts/analyze.py` writes
`results/analysis/E1_*`, `E2_*`, `E3_*` (`*_ranks`, `*_posthoc_holm`,
`*_pairwise_tests`, `*_per_function`), `E4_*_paired_holm`, `E4b_*_paired_holm`,
`E5_engineering_feasibility.csv` and `ANALYSIS.md`. They predate SEHHO-COBL-R: the
comparisons use the six original algorithms with SEHHO-COBL as the control,
and they are **not** the paper's tables, which `make_tables.py` computes
directly from the raw runs with SEHHO-COBL-R as the control. Twelve stray copies of
the `E3_cec2017_*` files (`tmp3_*`) have been removed.

<a id="the-word-review-in-file-comments"></a>
## The word "review" in file comments

Before submission the manuscript went through an internal pre-submission
review. Where code comments, docstrings, printed labels or the registration
files mention "review", "peer review", "adversarial review", "Stage 3",
roadmap items `REV-xx` or work orders `W1`–`W11`, they refer to that internal
pre-submission review of the manuscript, not to a journal's review. In
particular, the field `"status"` of
`results/analysis/PREREGISTRATION_E16_engineering_population.json` ("requested
by peer review (Stage 3, roadmap item REV-09)") and the "adversarial review"
in the rationale of `results/analysis/PREREGISTRATION_cec2014.json` refer to
that internal review. The registration files are hash-registered provenance and
are left exactly as written.

## Verification scripts

| Script | What it checks |
|---|---|
| `scripts/validate_benchmarks.py` | compiled suites against independent references; optimum recovery; throughput |
| `scripts/validate_jso.py` | our jSO against the jSO authors' published result files (`--compare-only` uses the stored runs) |
| `scripts/validate_lshade_shade.py` | our L-SHADE and SHADE (E13, CEC2014) against Tanabe & Fukunaga's published per-run results (reference folders passed in; nothing is optimised) |
| `scripts/validate_engine_variants.py` | the engine with its two switches set to the C++ codes' behaviour (`run`), the switches' defaults against stored E13 runs (`check-defaults`), and every statistic of the engine-variant and SHADE 1.0.1 checks (`analyze`) |
| `scripts/shade101_cpp/run_shade101.py` | builds Tanabe's SHADE 1.0.1 code from the official archive (sha256 checked) with the two documented edits, runs it on our CEC2014 data, and `--check` compares with the released runs |
| `scripts/validate_sehhocobl.py` | our SEHHO-COBL re-implementation against the means published in the earlier, MATLAB-based version of the manuscript |
| `scripts/test_stats.py` | Holm, Friedman, Cliff's delta, A12 against hand-computed values and scipy; the revision's BCa and Student-t intervals against scipy, margin verdicts, run-level variance of delta |
| `scripts/analyze_revision.py` | before computing anything new, recomputes every stored percentile interval it extends (gate inference, population control, CEC2014 endpoint and secondary comparisons, composition, population rule, suite gap, E6 contrast) and reports any mismatch |
| `scripts/check_cec_data.py` | largest singular values of every CEC transformation matrix; CEC2022 shift vectors, matrices and shuffles that are identical to CEC2017 or CEC2014 data |
| `scripts/check_integrity.py` | run counts, budget consumption, seed uniqueness and reproducibility (for the validation files, E13's reused seeds and the C++ code's srand rule) |
| `scripts/rc01_analytic.py` | derives and numerically verifies RC01's closed-form feasible reference design |
| `scripts/resolution_check.py` | re-runs every analysis script that reads per-run errors (the registered analyses, `analyze.py`, the gate analyses, `analyze_extras.py`, `analyze_revision.py`) and `make_tables.py` on two temporary copies, one with the recorded errors and one with every per-run error rounded to 1e-8, and compares them (post hoc; main text Section 2.6, Supplementary S2.6 and Table S-resolution). `make_figures.py`, the `validate_*.py` scripts, `check_integrity.py` and `test_reproduction.py` also read per-run errors and are not re-run. The full-precision run reproduces every released file except `selected_config.json`, whose released copy keeps the historical key `held_out` (see the note on `held_out` above). `results/analysis/resolution_changes.csv` lists every changed interval side, margin verdict (0.147, 0.10, 0.05; unadjusted and adjusted), tie flag, Holm or unadjusted test decision, label, decision, count and rank (two decimals) of the 64 analysis files, and every estimate that moves by more than 0.02, with both values; `results/analysis/resolution_table_cells.csv` lists every printed table cell or note that changes, with its table and document. The rounding reaches the readers through `sehho/runfiles.py` (`SEHHO_ERROR_DECIMALS`, unset by default, when nothing changes) and, for the pre-registered analysis scripts, which are left as registered, through rounded copies of the per-run files. `analyze_revision.py resolution` gives the earlier check of the 37 headline contrasts with all their intervals |
| `scripts/cec2020_entries.py` | recomputes the CEC2020 competition entries' feasibility counts and costs on RC01, RC06 and RC14 that the paper quotes, from the organisers' per-run files (https://github.com/P-N-Suganthan/2020-RW-Constrained-Optimisation; each archive checked by sha256) |

## A note on CEC2020 RC01

RC01 (heat-exchanger network design) is ranked by the suite's authors among
the hardest problems of its group, and its closed-form value below equals the
best-known objective they list for it. It is not hard. `scripts/rc01_analytic.py`
shows that constraints `h3` and `h5` make the logarithmic term of `h7` cancel
identically, forcing `x3 = 0`; the feasible set then collapses to the two-parameter family
`x2*x6 = 1e4` (with `x1*x4 = 0`) with every other coordinate fixed, and the
objective is minimised over that family in closed form (not uniquely: `x4` is
free once `x1 = 0`) at

    x* = (0, 50/3, 0, 0, 2e6, 600, 100, 600, 700),   f(x*) = 35*(50/3)^0.6 = 189.3116

verified feasible (total violation 0) by the same code that produced the
paper's numbers. This is a closed-form feasible reference design, not a
certified optimum: under the declared equality tolerance the infimum of the
problem the experiments solve lies below f(x*) by at least a relative 3e-9 %
along the one direction examined; its exact value is not determined. Every gap
quoted against x* is therefore a lower bound on the true excess. None of the
seven metaheuristics came within 11.5 % of it at 15,000 evaluations (for the
three re-run at budgets up to 1e6, see `E5b_engineering_budget.csv`), while
SciPy `trust-constr` reaches 189.3204 at 15,000. Among the CEC2020 competition
entries, EnMODE (Deb's rule with the same tolerance, constraints activated in
stages), SASS and sCMAgES (repair with analytic constraint Jacobians) return
189.3116 in all 25 of their released runs at 1e5 evaluations; EnMODE has no
feasible run before 8e4 (`scripts/cec2020_entries.py`).
Anyone reporting RC01 results should report the gap against this reference design.

## Note on this public copy

In `logs/`, absolute local paths were replaced by `<repo>`, `<project>`, `<python-lib>` and `<scratch>`; the logs are otherwise verbatim. Compiled CEC libraries (`*.so`) are not included: build them with `cec_native/build.sh`. Code: MIT (`LICENSE`); data: CC BY 4.0 (`LICENSE-DATA.md`); `cec_native/` is third-party material under its authors' terms.
