"""Reading the per-run result files, with the optional precision switch.

``read_runs(path)`` is ``pandas.read_csv(path)``.  Only when the environment
variable SEHHO_ERROR_DECIMALS is set (scripts/resolution_check.py sets it; see
main text Section 2.6 and Supplementary Section S2.6) is the ``error`` column of
a per-run file rounded to that many decimals after reading.  Files under
results/analysis/ (stored analyses, not runs) are never rounded.  With the
variable unset the function returns exactly what ``pandas.read_csv`` returns.
"""
import os

import numpy as np
import pandas as pd


def error_decimals():
    """The number of decimals from SEHHO_ERROR_DECIMALS, or None when it is unset."""
    v = os.environ.get("SEHHO_ERROR_DECIMALS")
    return int(v) if v else None


def read_runs(path, **kwargs):
    df = pd.read_csv(path, **kwargs)
    dec = error_decimals()
    if (dec is not None and "error" in df.columns
            and os.path.basename(os.path.dirname(os.path.abspath(path))) != "analysis"):
        df["error"] = np.round(df["error"].to_numpy(dtype=float), dec)
    return df
