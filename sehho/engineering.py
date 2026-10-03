"""Constrained engineering design problems with feasibility-first handling.

Three of the five problems (heat-exchanger network, blending-pooling-separation,
multi-product batch plant) are RC01, RC06 and RC14 of the CEC2020 real-world
constrained suite; their objective and constraint expressions here are
transcribed from Kumar et al.'s official ``cec20_func.m`` and their bounds from
``Cal_par.m``.  The pressure-vessel and gear-train problems are the classical
formulations.

Constraint handling
-------------------
All problems use Deb's feasibility rule rather than a penalised objective:

* a feasible point always beats an infeasible one;
* two feasible points are compared by objective value;
* two infeasible points are compared by total violation.

This is expressed as a single scalar so that unmodified optimisers can be used::

    score(x) = f(x)                      if v(x) == 0
             = F_REF_p + v(x)            otherwise

where ``F_REF_p`` is a per-problem constant that provably exceeds the objective
anywhere in the box, so the scalar ordering is identical to the rule above.
``_check_ref`` asserts this at run time.

Total violation uses the competition tolerance for equalities::

    v(x) = sum_i max(0, g_i(x)) + sum_j max(0, |h_j(x)| - eps)

and a point counts as feasible when v(x) <= 1e-8.
"""

from __future__ import annotations

import numpy as np

EPS_EQ = 1e-4        # CEC2020 real-world protocol equality tolerance
FEAS_TOL = 1e-8      # total violation at or below this counts as feasible;
                     # without it, designs sitting exactly on an active
                     # inequality (e.g. the classical pressure-vessel optimum,
                     # where g3 evaluates to 8e-11) are misreported.


class EngProblem:
    """Constrained problem exposing the feasibility-rule scalar to optimisers."""

    name = "engineering problem"
    dim = 0
    f_ref = 1e12          # must exceed sup f over the box
    n_ineq = 0
    n_eq = 0

    def __init__(self, eps_eq: float = EPS_EQ, feas_tol: float = FEAS_TOL):
        self.eps_eq = eps_eq
        self.feas_tol = feas_tol
        self.lb = np.asarray(self.LB, dtype=float)
        self.ub = np.asarray(self.UB, dtype=float)
        self.f_star = 0.0          # reported objectives are absolute, not errors

    # --- to be provided by subclasses -------------------------------------
    def repair(self, X):
        """Map raw decision vectors to the physically admissible encoding."""
        return X

    def objective(self, X):
        raise NotImplementedError

    def constraints(self, X):
        """Return (G, H) with shapes (n, n_ineq) and (n, n_eq)."""
        raise NotImplementedError

    # --- shared machinery --------------------------------------------------
    def parts(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Xr = self.repair(X)
        f = np.asarray(self.objective(Xr), dtype=float).ravel()
        G, H = self.constraints(Xr)
        G = np.zeros((X.shape[0], 0)) if G is None else np.atleast_2d(G)
        H = np.zeros((X.shape[0], 0)) if H is None else np.atleast_2d(H)
        gv = np.maximum(0.0, G).sum(axis=1) if G.size else np.zeros(X.shape[0])
        hv = (np.maximum(0.0, np.abs(H) - self.eps_eq).sum(axis=1)
              if H.size else np.zeros(X.shape[0]))
        v = gv + hv
        worst = np.zeros(X.shape[0])
        if G.size:
            worst = np.maximum(worst, np.maximum(0.0, G).max(axis=1))
        if H.size:
            worst = np.maximum(worst, np.maximum(0.0, np.abs(H) - self.eps_eq).max(axis=1))
        return f, v, worst

    def __call__(self, X):
        f, v, _ = self.parts(X)
        feas = v <= self.feas_tol
        if np.any(feas) and np.any(f[feas] >= self.f_ref):
            raise AssertionError(
                f"{self.name}: feasible objective {f[feas].max():.3e} exceeds "
                f"F_REF {self.f_ref:.3e}; the feasibility-rule scalar would misorder")
        out = np.where(feas, f, self.f_ref + v)
        return np.nan_to_num(out, nan=self.f_ref * 10, posinf=self.f_ref * 10)

    def report(self, x):
        """Full diagnostic record for one solution."""
        f, v, worst = self.parts(np.asarray(x)[None, :])
        return dict(objective=float(f[0]), total_violation=float(v[0]),
                    max_violation=float(worst[0]),
                    feasible=bool(v[0] <= self.feas_tol))


# --------------------------------------------------------------------------
# P1 - pressure vessel (classical)
# --------------------------------------------------------------------------
class PressureVessel(EngProblem):
    name = "Pressure vessel"
    dim = 4
    n_ineq = 4
    LB = [0.0625, 0.0625, 10.0, 10.0]
    UB = [6.1875, 6.1875, 200.0, 200.0]
    f_ref = 1.0e7          # sup f over the box is ~7.8e5

    def repair(self, X):
        X = X.copy()
        # thicknesses are available only in multiples of 0.0625 in
        X[:, :2] = np.ceil(X[:, :2] / 0.0625) * 0.0625
        X[:, :2] = np.clip(X[:, :2], self.lb[:2], self.ub[:2])
        return X

    def objective(self, X):
        x1, x2, x3, x4 = X.T
        return (0.6224 * x1 * x3 * x4 + 1.7781 * x2 * x3 ** 2
                + 3.1661 * x1 ** 2 * x4 + 19.84 * x1 ** 2 * x3)

    def constraints(self, X):
        x1, x2, x3, x4 = X.T
        G = np.column_stack([
            -x1 + 0.0193 * x3,
            -x2 + 0.00954 * x3,
            -np.pi * x3 ** 2 * x4 - (4.0 / 3.0) * np.pi * x3 ** 3 + 1_296_000.0,
            x4 - 240.0,
        ])
        return G, None


# --------------------------------------------------------------------------
# P2 - gear train (classical, unconstrained integer)
# --------------------------------------------------------------------------
class GearTrain(EngProblem):
    name = "Gear train"
    dim = 4
    LB = [12.0] * 4
    UB = [60.0] * 4
    f_ref = 1.0e3

    def repair(self, X):
        return np.clip(np.round(X), self.lb, self.ub)

    def objective(self, X):
        x1, x2, x3, x4 = X.T
        return (1.0 / 6.931 - (x2 * x3) / (x1 * x4)) ** 2

    def constraints(self, X):
        return None, None


# --------------------------------------------------------------------------
# P3 - RC01 heat exchanger network (CEC2020 real-world suite)
# --------------------------------------------------------------------------
class HeatExchangerNetwork(EngProblem):
    name = "Heat exchanger network (RC01)"
    dim = 9
    n_eq = 8
    LB = [0, 0, 0, 0, 1000, 0, 100, 100, 100]
    UB = [10, 200, 100, 200, 2_000_000, 600, 600, 600, 900]
    f_ref = 1.0e5          # sup f over the box is ~1.0e3

    def objective(self, X):
        return 35.0 * X[:, 0] ** 0.6 + 35.0 * X[:, 1] ** 0.6

    def constraints(self, X):
        x = X
        H = np.column_stack([
            200.0 * x[:, 0] * x[:, 3] - x[:, 2],
            200.0 * x[:, 1] * x[:, 5] - x[:, 4],
            x[:, 2] - 10000.0 * (x[:, 6] - 100.0),
            x[:, 4] - 10000.0 * (300.0 - x[:, 6]),
            x[:, 2] - 10000.0 * (600.0 - x[:, 7]),
            x[:, 4] - 10000.0 * (900.0 - x[:, 8]),
            # the official code regularises the logarithms with +1e-8
            x[:, 3] * np.log(np.abs(x[:, 7] - 100.0) + 1e-8)
            - x[:, 3] * np.log((600.0 - x[:, 6]) + 1e-8) - x[:, 7] + x[:, 6] + 500.0,
            x[:, 5] * np.log(np.abs(x[:, 8] - x[:, 6]) + 1e-8)
            - x[:, 5] * np.log(600.0) - x[:, 8] + x[:, 6] + 600.0,
        ])
        return None, H


# --------------------------------------------------------------------------
# P4 - RC06 blending-pooling-separation (CEC2020 real-world suite)
# --------------------------------------------------------------------------
class BlendingPoolingSeparation(EngProblem):
    name = "Blending-pooling-separation (RC06)"
    dim = 38
    n_eq = 32
    LB = [0.0] * 38
    UB = [90, 150, 90, 150, 90, 90, 150, 90, 90, 90, 150, 150, 90, 90, 150, 90,
          150, 90, 150, 90, 1, 1.2, 1, 1, 1, 0.5, 1, 1, 0.5, 0.5, 0.5, 1.2, 0.5,
          1.2, 1.2, 0.5, 1.2, 1.2]
    f_ref = 1.0e3          # sup f over the box is ~2.8

    def objective(self, X):
        return 0.9979 + 0.00432 * X[:, 4] + 0.01517 * X[:, 12]

    def constraints(self, X):
        x = X.T            # 1-based indices below match the official source
        def v(i):
            return x[i - 1]
        H = np.column_stack([
            v(1) + v(2) + v(3) + v(4) - 300.0,
            v(6) - v(7) - v(8),
            v(9) - v(10) - v(11) - v(12),
            v(14) - v(15) - v(16) - v(17),
            v(18) - v(19) - v(20),
            v(5) * v(21) - v(6) * v(22) - v(9) * v(23),
            v(5) * v(24) - v(6) * v(25) - v(9) * v(26),
            v(5) * v(27) - v(6) * v(28) - v(9) * v(29),
            v(13) * v(30) - v(14) * v(31) - v(18) * v(32),
            v(13) * v(33) - v(14) * v(34) - v(18) * v(35),
            v(13) * v(36) - v(14) * v(37) - v(18) * v(38),
            v(1) / 3.0 + v(15) * v(31) - v(5) * v(21),
            v(1) / 3.0 + v(15) * v(34) - v(5) * v(24),
            v(1) / 3.0 + v(15) * v(37) - v(5) * v(27),
            v(2) / 3.0 + v(10) * v(23) - v(13) * v(30),
            v(2) / 3.0 + v(10) * v(26) - v(13) * v(33),
            v(2) / 3.0 + v(10) * v(29) - v(13) * v(36),
            v(3) / 3.0 + v(7) * v(22) + v(11) * v(23) + v(16) * v(31) + v(19) * v(32) - 30.0,
            v(3) / 3.0 + v(7) * v(25) + v(11) * v(26) + v(16) * v(34) + v(19) * v(35) - 50.0,
            v(3) / 3.0 + v(7) * v(28) + v(11) * v(29) + v(16) * v(37) + v(19) * v(38) - 30.0,
            v(21) + v(24) + v(27) - 1.0,
            v(22) + v(25) + v(28) - 1.0,
            v(23) + v(26) + v(29) - 1.0,
            v(30) + v(33) + v(36) - 1.0,
            v(31) + v(34) + v(37) - 1.0,
            v(32) + v(35) + v(38) - 1.0,
            v(25), v(28), v(23), v(37), v(32), v(35),
        ])
        return None, H


# --------------------------------------------------------------------------
# P5 - RC14 multi-product batch plant (CEC2020 real-world suite)
# --------------------------------------------------------------------------
class BatchPlant(EngProblem):
    name = "Multi-product batch plant (RC14)"
    dim = 10
    n_ineq = 10
    LB = [0.51, 0.51, 0.51, 250, 250, 250, 6, 4, 40, 10]
    UB = [3.49, 3.49, 3.49, 2500, 2500, 2500, 20, 16, 700, 450]
    f_ref = 1.0e7          # sup f over the box is ~2.5e5

    _S = np.array([[2, 3, 4], [4, 6, 3]], dtype=float)
    _t = np.array([[8, 20, 8], [16, 4, 4]], dtype=float)

    def repair(self, X):
        X = X.copy()
        X[:, :3] = np.round(X[:, :3])      # integer number of parallel units
        return X

    def objective(self, X):
        N1, N2, N3, V1, V2, V3 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5]
        alp, beta = 250.0, 0.6
        return alp * (N1 * V1 ** beta + N2 * V2 ** beta + N3 * V3 ** beta)

    def constraints(self, X):
        S, t = self._S, self._t
        N1, N2, N3 = X[:, 0], X[:, 1], X[:, 2]
        V1, V2, V3 = X[:, 3], X[:, 4], X[:, 5]
        TL1, TL2 = X[:, 6], X[:, 7]
        B1, B2 = X[:, 8], X[:, 9]
        H, Q1, Q2 = 6000.0, 40000.0, 20000.0
        G = np.column_stack([
            Q1 * TL1 / B1 + Q2 * TL2 / B2 - H,
            S[0, 0] * B1 + S[1, 0] * B2 - V1,
            S[0, 1] * B1 + S[1, 1] * B2 - V2,
            S[0, 2] * B1 + S[1, 2] * B2 - V3,
            t[0, 0] - N1 * TL1,
            t[0, 1] - N2 * TL1,
            t[0, 2] - N3 * TL1,
            t[1, 0] - N1 * TL2,
            t[1, 1] - N2 * TL2,
            t[1, 2] - N3 * TL2,
        ])
        return G, None


ENGINEERING_PROBLEMS = {
    "P1_pressure_vessel": PressureVessel,
    "P2_gear_train": GearTrain,
    "P3_heat_exchanger_RC01": HeatExchangerNetwork,
    "P4_blending_pooling_RC06": BlendingPoolingSeparation,
    "P5_batch_plant_RC14": BatchPlant,
}
