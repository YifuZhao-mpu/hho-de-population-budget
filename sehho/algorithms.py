"""Optimisers used in the SEHHO-COBL study.

Every algorithm has the same contract::

    best_f = algorithm(budget, rng, **params)

``budget`` is a :class:`sehho.bench.Budget` that hard-caps the number of
objective evaluations, so all algorithms compared in a table have consumed
*exactly* the same evaluation budget.  ``rng`` is a ``numpy.random.Generator``
seeded per (algorithm, problem, run), which makes every reported number
reproducible from the seed alone.

Baseline provenance
-------------------
``lshade``  ports Tanabe & Fukunaga's own ``lshade.cpp`` (the implementation
            distributed with the CEC competition packages).
``jso``     ports Brest, Sepesy Maucec & Boskovic's own ``lshade.cc`` from the
            jSO CEC2017 submission archive.
``shade``   follows Tanabe & Fukunaga (2013) with the parameters of that paper.
``hho``     follows Heidari et al. (2019), including the extra evaluations that
            the two rapid-dive branches consume.
``de_rand1bin`` is textbook DE/rand/1/bin.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import functools
import math
import numpy as np

from .bench import Budget

_LEVY_BETA = 1.5


@functools.lru_cache(maxsize=8)
def _levy_sigma(beta: float) -> float:
    return (math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
            / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)


def levy(rng, size, beta: float = _LEVY_BETA) -> np.ndarray:
    """Mantegna's algorithm for symmetric Levy-stable steps."""
    sigma = _levy_sigma(beta)
    u = rng.standard_normal(size) * sigma
    v = rng.standard_normal(size)
    return u / np.abs(v) ** (1 / beta)


def _eval_capped(budget: Budget, X: np.ndarray):
    """Evaluate at most ``budget.remaining`` rows; returns (values, n_used)."""
    n = min(X.shape[0], budget.remaining)
    if n <= 0:
        return np.empty(0), 0
    return budget.evaluate(X[:n]), n


# --------------------------------------------------------------------------
# Proposed algorithm
# --------------------------------------------------------------------------

@dataclass
class SEHHOConfig:
    """SEHHO-COBL configuration.

    The defaults are the manuscript's fixed settings.  The remaining fields are
    the ablation switches; each isolates exactly one design decision.

    ``phase_mode`` controls how the exploration/exploitation phase indicator is
    produced.  Note the structural fact this exposes: in SEHHO-COBL the escape
    energy :math:`E = E_1 (2r-1)` enters the search **only** through the test
    :math:`|E| \\ge 1`, so the HHO "framework" is exactly a Bernoulli gate with
    per-iteration exploration probability
    :math:`q(t) = \\max(0, 1 - 1/(2(1-t/T)))`.  ``phase_mode`` replaces that gate
    by alternatives with the same interface.

    - ``escape_energy``  : the proposed gate, q(t) above.
    - ``constant_gate``  : Bernoulli gate with the *run-average* of q(t), which
      preserves the overall exploration share but destroys its schedule.
    - ``always_pbest``   : always sample pbest from the top p_N (SHADE-style).
    - ``always_best``    : always use the incumbent best.
    - ``deterministic``  : explore iff t < T/2 (same support as q(t), no mixing).
    """

    N: int = 30
    H: int = 10
    p_max: float = 0.25
    p_min: float = 0.05
    cobl_multiplier: int = 2
    levy_prob: float = 0.30
    levy_beta: float = _LEVY_BETA
    archive_factor: float = 1.0
    use_cobl: bool = True
    use_shade: bool = True
    use_archive: bool = True
    phase_mode: str = "escape_energy"
    levy_mode: str = "phase"          # phase | always | off
    p_schedule: str = "linear"        # linear | fixed
    fixed_F: float = 0.5
    fixed_CR: float = 0.9

    # --- population control -------------------------------------------------
    # The manuscript's method holds N fixed for the whole run.  On the
    # evidence of Section 5.4 that is its single most costly design decision,
    # so the sizing rule is exposed as configuration rather than hard-coded.
    #
    # ``pop_mode``        : "fixed" keeps N constant; "lpsr" applies the linear
    #                       population size reduction of L-SHADE, shrinking N
    #                       from its initial value to ``N_min`` in proportion to
    #                       the budget consumed.
    # ``N_init_factor``   : when set, the initial population is
    #                       round(N_init_factor * D) instead of ``N``, so the
    #                       sizing scales with the problem rather than being a
    #                       constant that happens to suit one dimensionality.
    # ``schedule_basis``  : what the time-varying schedules are a function of.
    #                       "generation" reproduces the manuscript (t/T, where T
    #                       is fixed in advance from the budget); "budget" uses
    #                       the fraction of evaluations actually spent, which is
    #                       the only well-defined choice once N varies.
    pop_mode: str = "fixed"           # fixed | lpsr
    N_init_factor: float | None = None
    N_min: int = 4
    schedule_basis: str = "generation"   # generation | budget


def _tent_chaos(n: int, rng) -> np.ndarray:
    """Tent map sequence with the manuscript's degeneracy safeguards."""
    z = np.empty(n)
    zk = 0.7
    for k in range(n):
        if abs(zk) < 1e-10 or abs(zk - 0.5) < 1e-10:
            zk += 0.01 * rng.random()
        zk = 2 * zk if zk < 0.5 else 2 * (1 - zk)
        z[k] = zk
    return z


def cobl_init(budget: Budget, rng, N: int, multiplier: int = 2):
    """Chaotic opposition-based initialisation; returns (X, f) of size N.

    Generates ``multiplier*N`` tent-map candidates plus their opposites
    (``2*multiplier*N`` total), evaluates all of them **against the budget**, and
    keeps the best N.  Under the FE-matched protocol these extra evaluations are
    paid for out of the same budget every competitor receives.
    """
    lb, ub, d = budget.lb, budget.ub, budget.dim
    m = multiplier * N
    z = _tent_chaos(m * d, rng).reshape(m, d)
    X = lb + z * (ub - lb)
    # break the cross-dimensional correlation induced by the shared chaotic seed
    for j in range(d):
        X[:, j] = X[rng.permutation(m), j]
    Xo = lb + ub - X
    pool = np.clip(np.vstack([X, Xo]), lb, ub)
    f, n = _eval_capped(budget, pool)
    pool = pool[:n]
    order = np.argsort(f)[:N]
    return pool[order].copy(), f[order].copy()


def uniform_init(budget: Budget, rng, N: int):
    lb, ub, d = budget.lb, budget.ub, budget.dim
    X = rng.uniform(lb, ub, (N, d))
    f, n = _eval_capped(budget, X)
    return X[:n].copy(), f.copy()


def _phase_explore(mode: str, t: int, T: int, N: int, rng, mean_q: float) -> np.ndarray:
    """Boolean array (N,): True where individual i is in the exploration phase."""
    if mode == "escape_energy":
        E1 = 2.0 * (1.0 - t / T)
        # |E| = E1 * |2u-1|, and |2u-1| ~ U(0,1)
        return E1 * np.abs(2.0 * rng.random(N) - 1.0) >= 1.0
    if mode == "constant_gate":
        return rng.random(N) < mean_q
    if mode == "always_pbest":
        return np.ones(N, dtype=bool)
    if mode == "always_best":
        return np.zeros(N, dtype=bool)
    if mode == "deterministic":
        return np.full(N, t < T / 2.0, dtype=bool)
    raise ValueError(f"unknown phase_mode {mode!r}")


def _mean_exploration_share(T: int) -> float:
    """Run-average of q(t) = max(0, 1 - 1/(2(1-t/T))) over t = 1..T."""
    t = np.arange(1, T + 1)
    E1 = 2.0 * (1.0 - t / T)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(E1 > 1.0, 1.0 - 1.0 / np.maximum(E1, 1e-300), 0.0)
    return float(np.mean(q))


def sehhocobl(budget: Budget, rng, cfg: SEHHOConfig | None = None) -> float:
    cfg = cfg or SEHHOConfig()
    lb, ub, d = budget.lb, budget.ub, budget.dim
    N = cfg.N if cfg.N_init_factor is None else max(4, int(round(cfg.N_init_factor * d)))

    if cfg.use_cobl:
        X, f = cobl_init(budget, rng, N, cfg.cobl_multiplier)
    else:
        X, f = uniform_init(budget, rng, N)
    if X.shape[0] < N:                       # budget too small to even initialise
        budget.finish_trace()
        return budget.best_f
    N = X.shape[0]

    # Number of generations the remaining budget affords; T is needed up front
    # because the phase gate and the p schedule are functions of t/T.  Ceiling
    # division guarantees T*N >= remaining, so the budget is spent exactly: the
    # final generation is truncated by the budget guard in the inner loop.
    # Floor division would leave the remainder unspent and quietly give this
    # algorithm a slightly smaller budget than its competitors.
    T = max(1, -(-budget.remaining // N))
    mean_q = _mean_exploration_share(T)

    M_F = np.full(cfg.H, 0.5)
    M_CR = np.full(cfg.H, 0.5)
    midx = 0
    arc_cap = int(round(cfg.archive_factor * N)) if cfg.use_archive else 0
    archive = np.empty((0, d))

    best_i = int(np.argmin(f))
    xstar = X[best_i].copy()
    fstar = float(f[best_i])
    span = ub - lb
    N_max = N

    # With LPSR the number of generations is not known in advance, so the loop
    # is driven by the budget.  With a fixed population the two forms are
    # identical: t runs 1..T and the budget guard is unchanged.
    t = 0
    while True:
        t += 1
        if cfg.pop_mode == "fixed" and t > T:
            break
        if budget.exhausted:
            break
        # Fraction of the run elapsed, used by every time-varying schedule.
        if cfg.schedule_basis == "budget":
            frac = min(1.0, budget.used / budget.max_fes)
        else:
            frac = t / T
        if cfg.p_schedule != "linear":
            p = cfg.p_max
        elif cfg.schedule_basis == "budget":
            p = cfg.p_max - (cfg.p_max - cfg.p_min) * frac
        else:
            # Deliberately written as `* t / T`, not `* frac`, even though frac
            # is t/T: floating-point multiplication and division do not
            # associate, the two forms differ in the last bit, and pN =
            # ceil(p*N) can round either side of an integer.  Keeping the
            # original expression keeps every previously published run
            # reproducible bit for bit.
            p = cfg.p_max - (cfg.p_max - cfg.p_min) * t / T
        pN = max(2, int(math.ceil(p * N)))
        order = np.argsort(f)

        # per-generation randomness, drawn in bulk for speed
        if cfg.use_shade:
            ridx = rng.integers(0, cfg.H, N)
            F = np.clip(M_F[ridx] + 0.1 * np.tan(np.pi * (rng.random(N) - 0.5)), 0.1, 1.0)
            CR = np.clip(M_CR[ridx] + 0.1 * rng.standard_normal(N), 0.0, 1.0)
        else:
            F = np.full(N, cfg.fixed_F)
            CR = np.full(N, cfg.fixed_CR)
        explore = _phase_explore(cfg.phase_mode, t, T, N, rng, mean_q)
        pbest_pick = order[rng.integers(0, pN, N)]
        r1_all = rng.integers(0, N, N)
        # r2 ranges over population + archive.  The draw uses the archive size at
        # the start of the generation; the archive can only grow while it is
        # still filling (the first two or three generations), after which it
        # stays at capacity and the range is exact.
        r2_all = rng.integers(0, N + archive.shape[0], N)
        jrand = rng.integers(0, d, N)
        cross = rng.random((N, d))
        levy_draw = rng.random(N) < cfg.levy_prob
        amp = 0.01 * (1.0 - frac) ** 2

        S_F, S_CR, S_df = [], [], []

        for i in range(N):
            if budget.exhausted:
                break
            xi = X[i]
            x_pb = X[pbest_pick[i]] if explore[i] else xstar

            r1 = r1_all[i]
            if r1 == i:
                r1 = (r1 + 1) % N
            npool = N + archive.shape[0]
            r2 = r2_all[i]
            guard = 0
            while (r2 == i or r2 == r1) and guard < 8:
                r2 = int(rng.integers(0, npool))
                guard += 1
            x_r2 = X[r2] if r2 < N else archive[r2 - N]

            Fi = F[i]
            v = xi + Fi * (x_pb - xi) + Fi * (X[r1] - x_r2)

            if cfg.levy_mode == "always":
                apply_levy = levy_draw[i]
            elif cfg.levy_mode == "phase":
                apply_levy = explore[i] and levy_draw[i]
            else:
                apply_levy = False
            if apply_levy:
                v = v + amp * levy(rng, d, cfg.levy_beta) * span

            mask = cross[i] < CR[i]
            mask[jrand[i]] = True
            u = np.where(mask, v, xi)
            np.clip(u, lb, ub, out=u)

            fu = float(budget.evaluate(u[None, :])[0])
            if fu < f[i]:
                S_F.append(Fi)
                S_CR.append(CR[i])
                S_df.append(f[i] - fu)
                if arc_cap > 0:
                    if archive.shape[0] < arc_cap:
                        archive = np.vstack([archive, xi[None, :]])
                    else:
                        archive[rng.integers(0, arc_cap)] = xi
                X[i] = u
                f[i] = fu
                if fu < fstar:
                    fstar = fu
                    xstar = u.copy()

        if cfg.use_shade and S_F:
            w = np.asarray(S_df, dtype=float)
            tot = w.sum()
            w = w / tot if tot > 0 else np.full(len(w), 1.0 / len(w))
            sf = np.asarray(S_F)
            scr = np.asarray(S_CR)
            M_F[midx] = np.sum(w * sf * sf) / max(np.sum(w * sf), 1e-300)
            M_CR[midx] = np.sum(w * scr)
            midx = (midx + 1) % cfg.H

        if cfg.pop_mode == "lpsr":
            # Same rule as L-SHADE: the planned size falls linearly with the
            # evaluations consumed, and the worst individuals are dropped.
            plan = int(round(((cfg.N_min - N_max) / budget.max_fes) * budget.used
                             + N_max))
            if N > plan:
                cut = min(N - plan, N - cfg.N_min)
                keep = np.argsort(f)[: N - cut]
                X, f = X[keep].copy(), f[keep].copy()
                N = X.shape[0]
                arc_cap = int(round(cfg.archive_factor * N)) if cfg.use_archive else 0
                if archive.shape[0] > arc_cap:
                    archive = archive[:arc_cap].copy()

    budget.finish_trace()
    return budget.best_f


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------

def hho(budget: Budget, rng, N: int = 30) -> float:
    """Harris Hawks Optimization (Heidari et al., 2019)."""
    lb, ub, d = budget.lb, budget.ub, budget.dim
    X = rng.uniform(lb, ub, (N, d))
    f, n = _eval_capped(budget, X)
    if n < N:
        budget.finish_trace()
        return budget.best_f
    k = int(np.argmin(f))
    rabbit, rabbit_f = X[k].copy(), float(f[k])

    # ceiling division so the budget is exhausted within T generations
    T = max(1, -(-budget.remaining // N))
    for t in range(1, T + 1):
        if budget.exhausted:
            break
        Xm = X.mean(axis=0)
        E1 = 2.0 * (1.0 - t / T)
        for i in range(N):
            if budget.exhausted:
                break
            E = E1 * (2.0 * rng.random() - 1.0)
            aE = abs(E)
            if aE >= 1.0:                                  # exploration
                q = rng.random()
                if q >= 0.5:
                    xr = X[rng.integers(0, N)]
                    new = xr - rng.random() * np.abs(xr - 2.0 * rng.random() * X[i])
                else:
                    new = (rabbit - Xm) - rng.random() * (lb + rng.random() * (ub - lb))
                new = np.clip(new, lb, ub)
                fn = float(budget.evaluate(new[None, :])[0])
                X[i], f[i] = new, fn
            else:                                          # exploitation
                r = rng.random()
                if r >= 0.5 and aE < 0.5:                  # hard besiege
                    new = rabbit - E * np.abs(rabbit - X[i])
                    new = np.clip(new, lb, ub)
                    fn = float(budget.evaluate(new[None, :])[0])
                    X[i], f[i] = new, fn
                elif r >= 0.5:                             # soft besiege
                    J = 2.0 * (1.0 - rng.random())
                    new = (rabbit - X[i]) - E * np.abs(J * rabbit - X[i])
                    new = np.clip(new, lb, ub)
                    fn = float(budget.evaluate(new[None, :])[0])
                    X[i], f[i] = new, fn
                else:                                      # rapid dives
                    # soft dive uses the individual as reference, hard dive the
                    # population mean (Heidari et al., 2019, Eqs. 10 and 12)
                    J = 2.0 * (1.0 - rng.random())
                    ref = X[i] if aE >= 0.5 else Xm
                    Y = np.clip(rabbit - E * np.abs(J * rabbit - ref), lb, ub)
                    fY = float(budget.evaluate(Y[None, :])[0])
                    if fY < f[i]:
                        X[i], f[i] = Y, fY
                    elif not budget.exhausted:
                        Z = np.clip(Y + rng.random(d) * levy(rng, d), lb, ub)
                        fZ = float(budget.evaluate(Z[None, :])[0])
                        if fZ < f[i]:
                            X[i], f[i] = Z, fZ
            if f[i] < rabbit_f:
                rabbit_f = float(f[i])
                rabbit = X[i].copy()
    budget.finish_trace()
    return budget.best_f


def de_rand1bin(budget: Budget, rng, N: int = 100, F: float = 0.5, CR: float = 0.9) -> float:
    """Textbook DE/rand/1/bin."""
    lb, ub, d = budget.lb, budget.ub, budget.dim
    X = rng.uniform(lb, ub, (N, d))
    f, n = _eval_capped(budget, X)
    if n < N:
        budget.finish_trace()
        return budget.best_f
    ar = np.arange(N)
    while not budget.exhausted:
        # three mutually distinct donors per target, all different from i
        idx = np.empty((N, 3), dtype=np.int64)
        for c in range(3):
            col = rng.integers(0, N, N)
            clash = (col == ar) | np.any(col[:, None] == idx[:, :c], axis=1)
            while clash.any():
                col[clash] = rng.integers(0, N, int(clash.sum()))
                clash = (col == ar) | np.any(col[:, None] == idx[:, :c], axis=1)
            idx[:, c] = col
        V = X[idx[:, 0]] + F * (X[idx[:, 1]] - X[idx[:, 2]])
        mask = rng.random((N, d)) < CR
        mask[np.arange(N), rng.integers(0, d, N)] = True
        U = np.clip(np.where(mask, V, X), lb, ub)
        m = min(N, budget.remaining)
        fu = budget.evaluate(U[:m])
        better = fu < f[:m]
        X[:m][better] = U[:m][better]
        f[:m][better] = fu[better]
    budget.finish_trace()
    return budget.best_f


def _shade_family(budget, rng, *, N_init, H, arc_rate, p_rate, memory_f0, memory_cr0,
                  lpsr, jso_mode, cr_lehmer=True, p_random=False) -> float:
    """Shared engine for SHADE / L-SHADE / jSO.

    Structure, bound handling ("medium with parent"), archive policy, memory
    update and LPSR follow the reference C++ implementations released by the
    algorithms' own authors.
    """
    lb, ub, d = budget.lb, budget.ub, budget.dim
    max_fes = budget.max_fes
    N = N_init
    X = rng.uniform(lb, ub, (N, d))
    f, n = _eval_capped(budget, X)
    if n < N:
        budget.finish_trace()
        return budget.best_f

    M_F = np.full(H, memory_f0)
    M_CR = np.full(H, memory_cr0)
    midx = 0
    arc_cap = int(round(N * arc_rate))
    archive = np.empty((0, d))
    p_best_rate = p_rate
    p_num = max(2, int(round(N * p_best_rate)))
    N_max, N_min = N, 4

    while not budget.exhausted:
        nfes = budget.used
        order = np.argsort(f)
        ridx = rng.integers(0, H, N)

        if jso_mode:
            mu_f = np.where(ridx == H - 1, 0.9, M_F[ridx])
            mu_cr = np.where(ridx == H - 1, 0.9, M_CR[ridx])
        else:
            mu_f, mu_cr = M_F[ridx], M_CR[ridx]

        CR = np.where(mu_cr < 0, 0.0,
                      np.clip(mu_cr + 0.1 * rng.standard_normal(N), 0.0, 1.0))
        # Cauchy resampling until F > 0, then truncate at 1
        F = mu_f + 0.1 * np.tan(np.pi * (rng.random(N) - 0.5))
        for _ in range(64):
            bad = F <= 0
            if not bad.any():
                break
            F[bad] = mu_f[bad] + 0.1 * np.tan(np.pi * (rng.random(bad.sum()) - 0.5))
        F = np.where(F <= 0, 0.001, np.minimum(F, 1.0))

        if jso_mode:
            if nfes < 0.25 * max_fes:
                CR = np.maximum(CR, 0.7)
            if nfes < 0.50 * max_fes:
                CR = np.maximum(CR, 0.6)
            if nfes < 0.60 * max_fes:
                F = np.minimum(F, 0.7)

        if p_random:
            # SHADE (2013) draws p_i ~ U[2/N, 0.2] independently per individual
            p_i = rng.uniform(2.0 / N, 0.2, N)
            pn_i = np.maximum(2, np.round(p_i * N).astype(int))
            pbest = order[(rng.random(N) * pn_i).astype(int)]
        else:
            pbest = order[rng.integers(0, p_num, N)]
        if jso_mode and nfes < 0.5 * max_fes:
            clash = pbest == np.arange(N)
            for _ in range(16):
                if not clash.any():
                    break
                pbest[clash] = order[rng.integers(0, p_num, int(clash.sum()))]
                clash = pbest == np.arange(N)

        r1 = rng.integers(0, N, N)
        clash = r1 == np.arange(N)
        while clash.any():
            r1[clash] = rng.integers(0, N, int(clash.sum()))
            clash = r1 == np.arange(N)
        na = archive.shape[0]
        r2 = rng.integers(0, N + na, N)
        clash = (r2 == np.arange(N)) | (r2 == r1)
        for _ in range(32):
            if not clash.any():
                break
            r2[clash] = rng.integers(0, N + na, int(clash.sum()))
            clash = (r2 == np.arange(N)) | (r2 == r1)

        pool = np.vstack([X, archive]) if na else X
        Fc = F[:, None]
        if jso_mode:
            if nfes < 0.2 * max_fes:
                jF = 0.7 * Fc
            elif nfes < 0.4 * max_fes:
                jF = 0.8 * Fc
            else:
                jF = 1.2 * Fc
        else:
            jF = Fc
        V = X + jF * (X[pbest] - X) + Fc * (X[r1] - pool[r2])

        mask = rng.random((N, d)) < CR[:, None]
        mask[np.arange(N), rng.integers(0, d, N)] = True
        U = np.where(mask, V, X)
        # bound handling: midpoint between the violated bound and the parent
        low, high = U < lb, U > ub
        U = np.where(low, (lb + X) / 2.0, U)
        U = np.where(high, (ub + X) / 2.0, U)

        m = min(N, budget.remaining)
        fu = budget.evaluate(U[:m])

        better = fu < f[:m]
        equal = fu == f[:m]
        if better.any():
            parents = X[:m][better]
            if arc_cap > 1:
                room = arc_cap - archive.shape[0]
                if room > 0:
                    archive = np.vstack([archive, parents[:room]])
                    parents = parents[room:]
                if parents.shape[0] and archive.shape[0]:
                    slots = rng.integers(0, archive.shape[0], parents.shape[0])
                    archive[slots] = parents
            S_F = F[:m][better]
            S_CR = CR[:m][better]
            S_df = f[:m][better] - fu[better]
            w = S_df / max(S_df.sum(), 1e-300)
            old_f, old_cr = M_F[midx], M_CR[midx]
            new_f = np.sum(w * S_F * S_F) / max(np.sum(w * S_F), 1e-300)
            denom_cr = np.sum(w * S_CR)
            if denom_cr == 0 or old_cr == -1:
                # every successful CR was 0: the slot becomes terminal
                new_cr = -1.0
            elif cr_lehmer:                      # L-SHADE / jSO: weighted Lehmer
                new_cr = np.sum(w * S_CR * S_CR) / denom_cr
            else:                                # SHADE 2013: weighted arithmetic
                new_cr = denom_cr
            if jso_mode:
                new_f = (new_f + old_f) / 2.0
                new_cr = -1.0 if new_cr == -1.0 else (new_cr + old_cr) / 2.0
            M_F[midx], M_CR[midx] = new_f, new_cr
            midx = (midx + 1) % H

        upd = better | equal
        X[:m][upd] = U[:m][upd]
        f[:m][upd] = fu[upd]

        if lpsr:
            plan = int(round(((N_min - N_max) / max_fes) * budget.used + N_max))
            if N > plan:
                cut = min(N - plan, N - N_min)
                keep = np.argsort(f)[: N - cut]
                X, f = X[keep].copy(), f[keep].copy()
                N = X.shape[0]
                arc_cap = int(N * arc_rate)
                if archive.shape[0] > arc_cap:
                    archive = archive[:arc_cap]
                if jso_mode:
                    p_best_rate = p_rate * (1.0 - 0.5 * budget.used / max_fes)
                p_num = max(2, int(round(N * p_best_rate)))
    budget.finish_trace()
    return budget.best_f


def shade(budget: Budget, rng, N: int = 100) -> float:
    """SHADE (Tanabe & Fukunaga, 2013): fixed N=100, H=N, |A|=N, p_i~U[2/N,0.2]."""
    return _shade_family(budget, rng, N_init=N, H=N, arc_rate=1.0, p_rate=0.1,
                         memory_f0=0.5, memory_cr0=0.5, lpsr=False, jso_mode=False,
                         cr_lehmer=False, p_random=True)


def lshade(budget: Budget, rng) -> float:
    """L-SHADE (Tanabe & Fukunaga, 2014): N=18D, H=6, |A|=2.6N, p=0.11, LPSR."""
    N = int(round(18 * budget.dim))
    return _shade_family(budget, rng, N_init=N, H=6, arc_rate=2.6, p_rate=0.11,
                         memory_f0=0.5, memory_cr0=0.5, lpsr=True, jso_mode=False)


def lshade_n(budget: Budget, rng, n_factor: float) -> float:
    """L-SHADE with **only** its initial population size changed.

    Every other published setting is untouched: H=6, |A|=2.6N, p=0.11, LPSR,
    the same bound handling and the same memory update.  This exists so that
    the effect of the initial population can be estimated without changing the
    algorithm around it -- comparing our own configuration against stock
    L-SHADE confounds population size with three other constants.
    """
    N = max(4, int(round(n_factor * budget.dim)))
    return _shade_family(budget, rng, N_init=N, H=6, arc_rate=2.6, p_rate=0.11,
                         memory_f0=0.5, memory_cr0=0.5, lpsr=True, jso_mode=False)


def jso(budget: Budget, rng) -> float:
    """jSO (Brest et al., 2017): ranked second at CEC2017, behind EBOwithCMAR;
    N=25 sqrt(D) ln(D), H=5."""
    d = budget.dim
    N = int(round(math.sqrt(d) * math.log(d) * 25))
    return _shade_family(budget, rng, N_init=N, H=5, arc_rate=1.0, p_rate=0.25,
                         memory_f0=0.3, memory_cr0=0.8, lpsr=True, jso_mode=True)


ALGORITHMS = {
    "SEHHO-COBL": lambda b, r: sehhocobl(b, r),
    "HHO": lambda b, r: hho(b, r, N=30),
    "DE": lambda b, r: de_rand1bin(b, r, N=100),
    "SHADE": lambda b, r: shade(b, r, N=100),
    "LSHADE": lambda b, r: lshade(b, r),
    "jSO": lambda b, r: jso(b, r),
}
