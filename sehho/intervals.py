"""Interval estimates for a mean of per-function effect sizes (revision).

Every contrast in the paper is summarised per function by Cliff's delta over
the runs of the two arms, and the reported effect is the mean of those deltas
over the functions of a suite.  The unit of inference is therefore the
function, and inference is conditional on the suite's functions treated as
exchangeable instances of the problem class the suite stands for.

This module adds to the percentile bootstrap used so far (reviewer item
REV-26, work order W1, and REV-27, work order W3):

* the BCa (bias-corrected and accelerated) bootstrap interval (Efron, 1987),
  computed from the *same* resamples as the percentile interval;
* the Student-t interval, mean +/- t_{n-1} s / sqrt(n), truncated to [-1, 1],
  the range of Cliff's delta;
* family-adjusted (Bonferroni) versions of all three at level 1 - alpha/k for a
  family of k contrasts within which a claim is decided;
* equivalence verdicts at any margin (REV-24);
* the two-stage bootstrap (functions, then runs within each function) and the
  per-function standard error of Cliff's delta at the run level (REV-27).

Conventions are those of sehho/stats.py and of the analysis scripts:
cliffs_delta(a, b) < 0 means `a` has the smaller errors; bootstrap indices are
drawn as rng.integers(0, n, (B, n)) with B = 20,000; quantiles use numpy's
default (linear) method.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps

BOOT = 20000
LEVEL = 0.95


# ------------------------------------------------------------------ resampling
def boot_indices(rng, n, boot=BOOT):
    """The resampling matrix every analysis script in the package draws."""
    return rng.integers(0, n, (boot, n))


def boot_means(v, idx):
    return np.asarray(v, dtype=float)[idx].mean(axis=1)


# ------------------------------------------------------------------ intervals
def percentile_ci(means, level=LEVEL):
    """Percentile interval, computed exactly as the existing scripts compute it."""
    return (float(np.quantile(means, (1 - level) / 2)),
            float(np.quantile(means, 1 - (1 - level) / 2)))


def bca_ci(v, means, level=LEVEL):
    """BCa interval for the mean of `v` from the bootstrap means `means`.

    z0 uses the mid-rank proportion of bootstrap means below the estimate, so
    that ties with the estimate (common: per-function deltas are multiples of
    1/900) do not bias it.  The acceleration is the jackknife value, which for
    a mean reduces to the sample skewness term sum(d^3) / (6 (sum d^2)^1.5).
    """
    v = np.asarray(v, dtype=float)
    theta = float(v.mean())
    if np.ptp(v) == 0:
        return theta, theta
    b = len(means)
    prop = (np.sum(means < theta) + 0.5 * np.sum(means == theta)) / b
    prop = min(max(prop, 1.0 / (2 * b)), 1 - 1.0 / (2 * b))
    z0 = sps.norm.ppf(prop)
    d = v - theta
    a = float(np.sum(d ** 3) / (6.0 * np.sum(d ** 2) ** 1.5))
    alpha = 1 - level
    out = []
    for q in (alpha / 2, 1 - alpha / 2):
        zq = sps.norm.ppf(q)
        adj = sps.norm.cdf(z0 + (z0 + zq) / (1 - a * (z0 + zq)))
        out.append(float(np.quantile(means, min(max(adj, 0.0), 1.0))))
    return out[0], out[1]


def t_ci(v, level=LEVEL, bounds=(-1.0, 1.0)):
    """Student-t interval for the mean of `v`, truncated to `bounds`.

    Cliff's delta lies in [-1, 1], so an interval reaching beyond that range is
    truncated to it (default); pass bounds=None for the untruncated interval.
    """
    v = np.asarray(v, dtype=float)
    n = len(v)
    m = float(v.mean())
    se = float(v.std(ddof=1) / np.sqrt(n))
    h = float(sps.t.ppf(1 - (1 - level) / 2, n - 1)) * se
    lo, hi = m - h, m + h
    if bounds is not None:
        lo, hi = max(lo, bounds[0]), min(hi, bounds[1])
    return lo, hi


def adjusted_level(k, level=LEVEL):
    """Bonferroni level for a family of k contrasts (0.9875 for k = 4)."""
    return 1 - (1 - level) / k


def all_intervals(v, idx, k=1, level=LEVEL):
    """Percentile, BCa and Student-t intervals, plain and family-adjusted.

    `idx` is the bootstrap index matrix of this contrast; all bootstrap
    intervals of the contrast are read from the same resamples.
    """
    v = np.asarray(v, dtype=float)
    m = boot_means(v, idx)
    adj = adjusted_level(k, level)
    out = dict(mean=float(v.mean()), n=len(v), sd=float(v.std(ddof=1)),
               se=float(v.std(ddof=1) / np.sqrt(len(v))), k_family=int(k),
               adj_level=adj)
    for name, (lo, hi) in (("pct", percentile_ci(m, level)), ("bca", bca_ci(v, m, level)),
                           ("t", t_ci(v, level)), ("pct_adj", percentile_ci(m, adj)),
                           ("bca_adj", bca_ci(v, m, adj)), ("t_adj", t_ci(v, adj))):
        out[f"{name}_lo"], out[f"{name}_hi"] = lo, hi
    return out


# ------------------------------------------------------------------ verdicts
def side(lo, hi):
    """'below' / 'above' zero, or 'spans' it."""
    return "below" if hi < 0 else "above" if lo > 0 else "spans"


def margin_verdict(lo, hi, margin):
    """Three-valued equivalence verdict at |delta| < margin.

    negligible   the whole interval lies inside (-margin, +margin);
    nonzero      the interval excludes zero and is not inside the band;
    inconclusive the interval includes zero and reaches outside the band.
    """
    if lo > -margin and hi < margin:
        return "negligible"
    if lo > 0 or hi < 0:
        return "nonzero"
    return "inconclusive"


def near_zero_bound(lo, hi, tol=0.02):
    """True when the bound nearest zero lies within `tol` of it (REV-26: tie)."""
    if lo > 0:
        return lo < tol
    if hi < 0:
        return -hi < tol
    return min(abs(lo), abs(hi)) < tol


# ------------------------------------------------------------------ signed rank
def signed_rank(v):
    """Two-sided signed-rank test of per-function deltas against zero.

    Zeros are dropped first, exactly as in the existing analysis scripts.
    """
    v = np.asarray(v, dtype=float)
    nz = np.abs(v) > 0
    return float(sps.wilcoxon(v[nz])[1]) if nz.sum() else 1.0


# ------------------------------------------------------------------ run level
def sign_matrix(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.sign(a[:, None] - b[None, :])


def cliff_var_unbiased(a, b):
    """Cliff's (1993) unbiased estimate of the sampling variance of delta."""
    S = sign_matrix(a, b)
    m, n = S.shape
    d = S.mean()
    di = S.mean(axis=1)
    dj = S.mean(axis=0)
    num = (m * m * np.sum((di - d) ** 2) + n * n * np.sum((dj - d) ** 2)
           - np.sum((S - d) ** 2))
    return float(max(num / (m * n * (m - 1) * (n - 1)), 0.0))


def run_boot_deltas(S, rng, reps):
    """`reps` run-level bootstrap replicates of delta for one function.

    Resampling the runs of arm A with multinomial counts cA and those of arm B
    with counts cB gives delta* = cA' S cB / (m n), so no comparison has to be
    repeated.
    """
    m, n = S.shape
    out = np.empty(reps)
    chunk = 2048
    ar_m, ar_n = np.arange(m), np.arange(n)
    for s in range(0, reps, chunk):
        r = min(chunk, reps - s)
        ia = rng.integers(0, m, (r, m))
        ib = rng.integers(0, n, (r, n))
        ca = (ia[:, :, None] == ar_m[None, None, :]).sum(axis=1)
        cb = (ib[:, :, None] == ar_n[None, None, :]).sum(axis=1)
        out[s:s + r] = np.einsum("ri,ij,rj->r", ca, S, cb) / (m * n)
    return out


def two_stage_bootstrap(runs_a, runs_b, rng, boot=BOOT, level=LEVEL):
    """Bootstrap of the mean per-function delta that resamples functions and then
    the runs of each arm within every selected function.

    `runs_a`, `runs_b`: lists (one entry per function) of per-run errors.
    Each selected function receives an independent run-level replicate.
    Returns (mean of the observed deltas, percentile interval, bootstrap means).
    """
    nf = len(runs_a)
    S = [sign_matrix(a, b) for a, b in zip(runs_a, runs_b)]
    obs = np.array([s.mean() for s in S])
    idx = rng.integers(0, nf, (boot, nf))
    counts = np.bincount(idx.ravel(), minlength=nf)
    pools = [run_boot_deltas(S[f], rng, int(counts[f])) for f in range(nf)]
    vals = np.empty(idx.shape)
    pos = np.zeros(nf, dtype=int)
    flat = idx.ravel()
    out = vals.ravel()
    for t, f in enumerate(flat):
        out[t] = pools[f][pos[f]]
        pos[f] += 1
    means = vals.mean(axis=1)
    return float(obs.mean()), percentile_ci(means, level), means
