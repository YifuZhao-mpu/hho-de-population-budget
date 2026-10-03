"""Unit tests for the statistics used in the paper."""
import os, sys
import numpy as np
from scipy import stats as sps
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sehho.stats import (holm, cliffs_delta, vargha_delaney_a12, friedman,
                         friedman_holm_vs_control, cliffs_magnitude)

ok = True
def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")

# --- Holm -----------------------------------------------------------------
a = holm([0.01, 0.04, 0.03])
check("Holm hand-computed", np.allclose(a, [0.03, 0.06, 0.06]), f"-> {np.round(a,4)}")
check("Holm monotone in sorted order",
      np.all(np.diff(np.sort(holm(np.random.default_rng(0).random(50)))) >= -1e-12))
check("Holm never below raw p", np.all(holm([0.2, 0.5]) >= np.array([0.2, 0.5])))
check("Holm capped at 1", holm([0.9, 0.95])[1] <= 1.0)
check("Holm n=1 identity", np.allclose(holm([0.03]), [0.03]))

# --- effect sizes ---------------------------------------------------------
check("Cliff delta fully separated (a<b)", cliffs_delta([1, 2, 3], [4, 5, 6]) == -1.0)
check("Cliff delta fully separated (a>b)", cliffs_delta([4, 5, 6], [1, 2, 3]) == 1.0)
check("Cliff delta identical samples", cliffs_delta([1, 2, 3], [1, 2, 3]) == 0.0)
check("A12 fully separated", vargha_delaney_a12([1, 2, 3], [4, 5, 6]) == 1.0)
check("A12 identical samples", vargha_delaney_a12([1, 2], [1, 2]) == 0.5)
check("Cliff magnitudes", [cliffs_magnitude(d) for d in (0.1, 0.2, 0.4, 0.8)]
      == ["negligible", "small", "medium", "large"])
# delta and A12 must agree on direction
rng = np.random.default_rng(3)
x, y = rng.normal(0, 1, 40), rng.normal(0.6, 1, 40)
check("delta and A12 agree on direction",
      (cliffs_delta(x, y) < 0) == (vargha_delaney_a12(x, y) > 0.5))

# --- Friedman -------------------------------------------------------------
M = rng.random((14, 5))
fr = friedman(M)
chi2_scipy, p_scipy = sps.friedmanchisquare(*[M[:, j] for j in range(M.shape[1])])
check("Friedman chi2 matches scipy", np.isclose(fr["chi2"], chi2_scipy),
      f"ours={fr['chi2']:.6f} scipy={chi2_scipy:.6f}")
check("average ranks sum to k(k+1)/2",
      np.isclose(fr["avg_ranks"].sum(), 5 * 6 / 2))
# a deliberately dominant column must earn rank 1
M2 = rng.random((14, 4)); M2[:, 2] -= 5
check("dominant column ranks first", np.argmin(friedman(M2)["avg_ranks"]) == 2)
_, post = friedman_holm_vs_control(M2, ["a", "b", "ctrl", "d"], "ctrl")
check("post-hoc detects dominance", all(r["significant_holm"] for r in post))

# --- revision intervals (sehho/intervals.py) ---------------------------------
from sehho.intervals import (boot_indices, boot_means, percentile_ci, bca_ci, t_ci,
                             margin_verdict, near_zero_bound, cliff_var_unbiased,
                             run_boot_deltas, sign_matrix, adjusted_level)
v = np.clip(np.random.default_rng(11).normal(-0.3, 0.35, 29), -1, 1)
idx = boot_indices(np.random.default_rng(5), len(v))
m = boot_means(v, idx)
ref = sps.bootstrap((v,), np.mean, n_resamples=20000, method="BCa",
                    random_state=np.random.default_rng(5)).confidence_interval
check("BCa matches scipy on the same resamples",
      np.allclose(bca_ci(v, m), (ref.low, ref.high), atol=1e-12),
      f"ours={np.round(bca_ci(v, m), 4)} scipy={np.round((ref.low, ref.high), 4)}")
check("percentile interval = numpy quantiles of the bootstrap means",
      np.allclose(percentile_ci(m), np.quantile(m, [0.025, 0.975])))
tt = sps.t.interval(0.95, len(v) - 1, loc=v.mean(), scale=sps.sem(v))
check("Student-t matches scipy", np.allclose(t_ci(v, bounds=None), tt))
check("Student-t truncated to [-1, 1]", t_ci(np.r_[-1.0, -1.0, -0.9])[0] >= -1.0)
check("Bonferroni level for four contrasts", np.isclose(adjusted_level(4), 0.9875))
check("negated data give the mirrored percentile interval",
      np.allclose(percentile_ci(boot_means(-v, idx)), [-percentile_ci(m)[1], -percentile_ci(m)[0]]))
check("margin verdicts", [margin_verdict(-0.11, 0.045, 0.147), margin_verdict(-0.11, 0.045, 0.10),
                          margin_verdict(-0.24, -0.09, 0.147), margin_verdict(-0.12, -0.01, 0.147)]
      == ["negligible", "inconclusive", "nonzero", "negligible"])
check("tie rule (bound within 0.02 of zero)", near_zero_bound(-0.131, -0.0002) and
      not near_zero_bound(-0.2, -0.09))
a, b = rng.normal(0, 1, 30), rng.normal(0.5, 1, 30)
d = run_boot_deltas(sign_matrix(a, b), np.random.default_rng(3), 20000)
check("run-level bootstrap is centred on delta", abs(d.mean() - cliffs_delta(a, b)) < 0.01)
check("Cliff's unbiased variance agrees with the run bootstrap",
      abs(np.sqrt(cliff_var_unbiased(a, b)) - d.std()) < 0.01,
      f"analytic {np.sqrt(cliff_var_unbiased(a, b)):.4f} bootstrap {d.std():.4f}")

print("\nALL TESTS PASSED" if ok else "\nSOME TESTS FAILED")
sys.exit(0 if ok else 1)
