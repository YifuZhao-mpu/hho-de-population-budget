"""Non-parametric statistics with multiplicity control and effect sizes.

Implements the protocol recommended by Derrac et al. (2011) for comparing
stochastic optimisers, plus the two additions the manuscript previously lacked:

* **Holm step-down correction** over every family of pairwise tests, so the
  reported significance counts are not inflated by testing one method against
  many competitors on many functions;
* **effect sizes** (Cliff's delta and the Vargha-Delaney A12 statistic), so a
  significant p-value is accompanied by the magnitude of the difference.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


# ---------------------------------------------------------------- multiplicity
def holm(pvals):
    """Holm step-down adjusted p-values (monotone, family-wise error control)."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj


# ---------------------------------------------------------------- effect sizes
def cliffs_delta(a, b):
    """Cliff's delta in [-1, 1]; negative means `a` tends to be smaller."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = np.sign(a[:, None] - b[None, :])
    return float(diff.mean())


def cliffs_magnitude(d):
    """Romano et al. (2006) thresholds."""
    d = abs(d)
    if d < 0.147:
        return "negligible"
    if d < 0.33:
        return "small"
    if d < 0.474:
        return "medium"
    return "large"


def vargha_delaney_a12(a, b):
    """P(a < b) + 0.5 P(a == b): probability that `a` wins on a minimisation task."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    gt = (a[:, None] < b[None, :]).sum()
    eq = (a[:, None] == b[None, :]).sum()
    return float((gt + 0.5 * eq) / (a.size * b.size))


# ---------------------------------------------------------------- Friedman
def friedman(M):
    """Friedman test on an (n_problems, n_algorithms) matrix of scores.

    Returns average ranks (lower is better), the chi-square statistic, the
    Iman-Davenport F statistic and its p-value.
    """
    M = np.asarray(M, dtype=float)
    n, k = M.shape
    ranks = np.apply_along_axis(stats.rankdata, 1, M)
    avg = ranks.mean(axis=0)
    chi2 = (12.0 * n / (k * (k + 1))) * (np.sum(avg ** 2) - k * (k + 1) ** 2 / 4.0)
    denom = n * (k - 1) - chi2
    if denom <= 0:
        F, p = np.inf, 0.0
    else:
        F = (n - 1) * chi2 / denom
        p = float(stats.f.sf(F, k - 1, (k - 1) * (n - 1)))
    return dict(ranks=ranks, avg_ranks=avg, chi2=float(chi2),
                iman_davenport_F=float(F), p=p, n=n, k=k)


def friedman_holm_vs_control(M, names, control):
    """Friedman post-hoc: each algorithm against `control`, Holm-corrected."""
    fr = friedman(M)
    avg = fr["avg_ranks"]
    n, k = fr["n"], fr["k"]
    ci = names.index(control)
    se = np.sqrt(k * (k + 1) / (6.0 * n))
    rows, praw = [], []
    for j, nm in enumerate(names):
        if j == ci:
            continue
        z = (avg[ci] - avg[j]) / se
        p = 2.0 * stats.norm.sf(abs(z))
        praw.append(p)
        rows.append(dict(algorithm=nm, avg_rank=avg[j], control_rank=avg[ci], z=z, p=p))
    for r, pa in zip(rows, holm(praw)):
        r["p_holm"] = pa
        r["significant_holm"] = bool(pa < 0.05)
    return fr, rows


# ---------------------------------------------------------------- pairwise
def ranksum_family(per_run, control, alpha=0.05):
    """Per-function two-sided rank-sum tests of `control` against each competitor.

    `per_run` maps algorithm -> {function -> array of run values}.  Every test in
    the returned family (all functions x all competitors) is Holm-corrected
    together, which is the multiplicity level that matters for the aggregate
    win/tie/loss counts reported in the paper.
    """
    algos = [a for a in per_run if a != control]
    funcs = sorted(per_run[control].keys())
    rows, praw = [], []
    for a in algos:
        for f in funcs:
            x = np.asarray(per_run[control][f], dtype=float)
            y = np.asarray(per_run[a][f], dtype=float)
            if np.allclose(x, y):
                p, U = 1.0, 0.5 * x.size * y.size
            else:
                U, p = stats.mannwhitneyu(x, y, alternative="two-sided")
            d = cliffs_delta(x, y)
            rows.append(dict(algorithm=a, func=f, U=float(U), p=float(p),
                             cliffs_delta=d, magnitude=cliffs_magnitude(d),
                             a12=vargha_delaney_a12(x, y),
                             median_control=float(np.median(x)),
                             median_other=float(np.median(y))))
            praw.append(p)
    adj = holm(praw)
    for r, pa in zip(rows, adj):
        r["p_holm"] = float(pa)
        if pa >= alpha:
            r["outcome"] = "tie"
        else:
            r["outcome"] = "win" if r["cliffs_delta"] < 0 else "loss"
    return rows


def summarise_outcomes(rows):
    """Aggregate win/tie/loss per competitor from `ranksum_family` output."""
    out = {}
    for r in rows:
        d = out.setdefault(r["algorithm"], dict(win=0, tie=0, loss=0))
        d[r["outcome"]] += 1
    return out
