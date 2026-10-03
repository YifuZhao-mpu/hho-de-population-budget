"""Benchmark suites and the function-evaluation budget harness.

The CEC2017 and CEC2022 suites are evaluated by the *official* competition C/C++
code (P. N. Suganthan's distributions), compiled to shared libraries by
``cec_native/build.sh``.  Nothing about the landscape definitions is
re-implemented here, so the numbers are directly comparable with the
competition literature.

Every algorithm in this package obtains objective values exclusively through
:class:`Budget`, which counts each evaluated candidate and refuses to evaluate
past ``max_fes``.  This makes the "strictly FE-matched" claim in the manuscript
mechanically enforced rather than a convention that each algorithm is trusted to
respect.
"""

from __future__ import annotations

import ctypes
import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_NATIVE = os.path.join(os.path.dirname(_HERE), "cec_native")

# CEC2017: official suite is f1..f30 with f2 withdrawn -> 29 usable functions.
CEC2017_FUNCS = [i for i in range(1, 31) if i != 2]
CEC2017_DIMS = (10, 30, 50, 100)

CEC2022_FUNCS = list(range(1, 13))

# CEC2014 Part A: 30 functions, all of them (unlike CEC2017, F2 is not withdrawn).
CEC2014_FUNCS = list(range(1, 31))
CEC2022_DIMS = (10, 20)


class BudgetExceeded(RuntimeError):
    """Raised when an algorithm asks for more evaluations than it is allowed."""


class _NativeSuite:
    """ctypes binding to one compiled competition suite."""

    def __init__(self, sofile: str, entry: str, data_dir: str):
        self._lib = ctypes.CDLL(os.path.join(_NATIVE, sofile))
        self._fn = getattr(self._lib, entry)
        self._fn.restype = None
        self._fn.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._lib.cec_set_data_dir.argtypes = [ctypes.c_char_p]
        self._lib.cec_set_data_dir.restype = None
        self._lib.cec_set_data_dir(os.path.join(_NATIVE, data_dir).encode())
        # Reusable input/output buffers keyed by (rows, dim).  Re-creating the
        # ctypes pointers on every call dominates the cost of single-candidate
        # evaluation, which is the common case for the per-individual loops.
        self._buf: dict[tuple[int, int], tuple] = {}

    def _buffers(self, m: int, d: int):
        key = (m, d)
        b = self._buf.get(key)
        if b is None:
            xin = np.empty((m, d), dtype=np.float64)
            out = np.empty(m, dtype=np.float64)
            b = (xin, out,
                 xin.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                 out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
            self._buf[key] = b
        return b

    def __call__(self, X: np.ndarray, func_num: int) -> np.ndarray:
        m, d = X.shape
        xin, out, xptr, optr = self._buffers(m, d)
        xin[...] = X
        self._fn(xptr, optr, d, m, func_num)
        return out.copy()


_SUITES: dict[str, _NativeSuite] = {}


def _suite(name: str) -> _NativeSuite:
    """Load a suite lazily; one shared instance per process."""
    if name not in _SUITES:
        if name == "cec2014":
            _SUITES[name] = _NativeSuite("libcec14.so", "cec14_test_func", "input_data_2014")
        elif name == "cec2017":
            _SUITES[name] = _NativeSuite("libcec17.so", "cec17_test_func", "input_data_2017")
        elif name == "cec2022":
            _SUITES[name] = _NativeSuite("libcec22.so", "cec22_test_func", "input_data_2022")
        else:
            raise ValueError(f"unknown suite {name!r}")
    return _SUITES[name]


class CECProblem:
    """A single (suite, function, dimension) landscape on [-100, 100]^D."""

    def __init__(self, suite: str, func_num: int, dim: int):
        self.suite = suite
        self.func_num = func_num
        self.dim = dim
        self.lb = np.full(dim, -100.0)
        self.ub = np.full(dim, 100.0)
        # CEC2014 and CEC2017 both bias F_i by 100*i; CEC2022 uses its own table.
        self.f_star = (_CEC2022_FSTAR[func_num] if suite == "cec2022"
                       else 100.0 * func_num)
        self.name = f"{suite.upper()}-F{func_num}"
        self._eval = _suite(suite)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self._eval(X, self.func_num)


# CEC2022 biases from the competition definition (f1..f12).
_CEC2022_FSTAR = {
    1: 300.0, 2: 400.0, 3: 600.0, 4: 800.0, 5: 900.0, 6: 1800.0,
    7: 2000.0, 8: 2200.0, 9: 2300.0, 10: 2400.0, 11: 2600.0, 12: 2700.0,
}


class Budget:
    """Hard function-evaluation budget shared by every algorithm.

    ``evaluate`` raises :class:`BudgetExceeded` rather than silently truncating,
    so an algorithm that miscounts fails loudly instead of quietly consuming a
    different budget from its competitors.  ``remaining`` lets a well-behaved
    algorithm shrink its final batch to land exactly on ``max_fes``.
    """

    __slots__ = ("problem", "max_fes", "used", "best_f", "best_x",
                 "_trace_at", "trace", "_trace_i", "lb", "ub", "dim")

    def __init__(self, problem, max_fes: int, trace_points: np.ndarray | None = None):
        self.problem = problem
        self.max_fes = int(max_fes)
        self.used = 0
        self.best_f = np.inf
        self.best_x = None
        self.lb = problem.lb
        self.ub = problem.ub
        self.dim = problem.dim
        self._trace_at = None if trace_points is None else np.asarray(trace_points, dtype=np.int64)
        self.trace = [] if trace_points is not None else None
        self._trace_i = 0

    @property
    def remaining(self) -> int:
        return self.max_fes - self.used

    @property
    def exhausted(self) -> bool:
        return self.used >= self.max_fes

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X[None, :]
        n = X.shape[0]
        if n > self.remaining:
            raise BudgetExceeded(
                f"{self.problem.name}: requested {n} evaluations, {self.remaining} left "
                f"of {self.max_fes}"
            )
        f = self.problem(X)
        self.used += n
        # Track the incumbent so every algorithm reports the same quantity:
        # the best objective value ever evaluated within the budget.
        k = 0 if n == 1 else int(np.argmin(f))
        if f[k] < self.best_f:
            self.best_f = float(f[k])
            self.best_x = X[k].copy()
        if self._trace_at is not None:
            while (self._trace_i < len(self._trace_at)
                   and self.used >= self._trace_at[self._trace_i]):
                self.trace.append(self.best_f)
                self._trace_i += 1
        return f

    def finish_trace(self):
        if self._trace_at is not None:
            while self._trace_i < len(self._trace_at):
                self.trace.append(self.best_f)
                self._trace_i += 1
