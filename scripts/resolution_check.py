"""Precision check over every released analysis and every generated table (main text
Section 2.6, Supplementary Section S2.6; post hoc, added after the main analyses).

Cliff's delta compares the recorded errors at full double precision (errors below
1e-8 are recorded as 0), so runs that end on a plateau and agree to about twelve
significant digits are still ordered.  This script measures what that convention
changes.  It builds two temporary copies of the package and re-runs, in both, every
analysis that reads per-run errors and the table generator:

    analyze_poprule, analyze_composition, analyze_cec2014, analyze_pop_isolation,
    analyze_e15, analyze_e16, analyze, analyze_gate_factorial,
    analyze_gate_inference, analyze_extras, analyze_revision, make_tables

  * "full":    the released per-run files, unchanged;
  * "rounded": the same, with the `error` column of every per-run file in results/
               rounded to 1e-8 (written into the copy, so that the pre-registered
               analysis scripts, which are left exactly as registered, read rounded
               errors too) and SEHHO_ERROR_DECIMALS=8 set for every other reader.

It then compares the two runs:

  1. every CSV and JSON file the analyses write to results/analysis/, row by row
     (rows matched on their identifying columns):
       - the side of every interval (unadjusted and Bonferroni-adjusted;
         percentile, BCa, Student-t, and the single intervals of the registered
         analyses): below zero, above zero, or spanning it;
       - the equivalence verdict of every delta-scale interval at the margins
         0.147, 0.10 and 0.05 (adjusted and unadjusted);
       - the tie rule of Section 2.6 for every unadjusted delta-scale interval
         (it excludes zero with its bound nearest zero within 0.02 of it);
       - every Holm-corrected p-value at 0.05 and every unadjusted p-value at 0.05;
       - every categorical column (magnitude labels, rank-sum outcomes, W/T/L
         strings, verdicts, decisions, significance flags);
       - every count (wins, losses, ties, rank positions, #1 counts, feasible and
         success counts);
       - every rank (printed to two decimals: listed when the printed value changes);
       - every delta-scale estimate whose value moves by more than 0.02;
       - every leaf of the JSON verdict and decision files;
  2. every cell of every generated LaTeX table (results/tex/), and every table
     note: each printed cell whose text changes is listed with both texts.

Outputs (both rewritten on every run):
  results/analysis/resolution_changes.csv      one row per individual change in an
                                               analysis file (part 1)
  results/analysis/resolution_table_cells.csv  one row per changed printed table
                                               cell or note (part 2)

The full-precision run must reproduce the released results; any file in which it
does not is reported (it is compared with the rounded run, not with the released
file).  One is expected: selected_config.json, whose released copy keeps the
historical key "held_out" (README, "held_out in selected_config.json is historical").  The rounded run fails the reproduction checks of analyze_revision.py by
design; every compared file must have been rewritten by both runs, otherwise the
script stops.

Run:  python3 scripts/resolution_check.py        (a few minutes; the two runs are parallel)
"""
import csv
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "results", "analysis", "resolution_changes.csv")
OUT_CELLS = os.path.join(ROOT, "results", "analysis", "resolution_table_cells.csv")
DECIMALS = 8
SCRIPTS = ["analyze_poprule", "analyze_composition", "analyze_cec2014", "analyze_pop_isolation",
           "analyze_e15", "analyze_e16", "analyze", "analyze_gate_factorial",
           "analyze_gate_inference", "analyze_extras", "analyze_revision", "make_tables"]
# analysis outputs that are not compared: this check's own outputs, the precision
# contrast of analyze_revision.py (itself a full-vs-rounded comparison), and indices
SKIP = {"resolution_changes.csv", "resolution_table_cells.csv", "revision_resolution.csv",
        "revision_delta_ledger.csv", "revision_table_index.csv", "ANALYSIS.md"}
KEYS = ["table", "suite", "dim", "max_fes", "budget", "block", "quantity", "family", "contrast",
        "change", "candidate", "versus", "opponent", "a", "b", "N", "arm_a", "arm_b", "analysis",
        "engine", "problems", "item", "evals_per_variable", "algorithm", "func", "variant",
        "effect", "problem", "algo", "interval", "margin", "source_marker"]
TOL, SHIFT = 0.02, 0.02
FREE_TEXT = {"detail", "note", "x", "seed"}      # descriptive text, compared through the tables
MARGINS = (0.147, 0.10, 0.05)
P_COLS = re.compile(r"^(p|p_.*|.*_p|.*_p_holm|min_p_holm)$")
COUNT_COLS = re.compile(r"^(wins|losses|ties|wins_for_gate_off|first_better|second_better|"
                        r"rank1_count|position|n_better|feasible.*|success.*|runs_at_threshold|"
                        r"runs_violation_exactly_1e_8|feas_6D|feas_18D|functions_changed)$")
RANK_COLS = re.compile(r"^(avg_rank|control_rank|full_rank|rank_gate_on|rank_gate_off|rank_change)$")
EST_COLS = re.compile(r"^(mean|mean_delta|delta|estimate|cliffs_delta|mean_cliffs_delta|mean_four|"
                      r"F\d+|gap_cec2017_minus_cec2014|cec2017_mean|cec2014_mean|population_effect|"
                      r"gate_effect)$")

sys.path.insert(0, ROOT)
from sehho.intervals import margin_verdict, side  # noqa: E402  (the package's own definitions)


# ------------------------------------------------------------------ the two runs
def _round_error_column(path):
    """Round the `error` column of one per-run CSV in place, leaving every other byte."""
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows or "error" not in rows[0]:
        return False
    j = rows[0].index("error")
    for r in rows[1:]:
        if j < len(r) and r[j] not in ("", "nan", "NaN"):
            r[j] = repr(float(np.round(float(r[j]), DECIMALS)))
    with open(path, "w", newline="") as fh:
        csv.writer(fh, lineterminator="\n").writerows(rows)
    return True


def _prepare(tmp, rounded):
    for d in ("sehho", "scripts"):
        shutil.copytree(os.path.join(ROOT, d), os.path.join(tmp, d),
                        ignore=shutil.ignore_patterns("__pycache__", "*.so", "*.o"))
    shutil.copytree(os.path.join(ROOT, "results"), os.path.join(tmp, "results"),
                    ignore=shutil.ignore_patterns("figures", "legacy"))
    if rounded:
        n = sum(_round_error_column(p) for p in glob.glob(os.path.join(tmp, "results", "*.csv")))
        print(f"  rounded the error column of {n} per-run files to 1e-{DECIMALS}")
    # every compared output must be rewritten: age the copies so that a rewrite is visible
    old = time.time() - 86400
    for p in glob.glob(os.path.join(tmp, "results", "analysis", "*")) + \
            glob.glob(os.path.join(tmp, "results", "tex", "*")):
        os.utime(p, (old, old))
    return old


def _run_chain(tmp, rounded):
    env = dict(os.environ, MPLBACKEND="Agg")
    env.pop("SEHHO_ERROR_DECIMALS", None)
    if rounded:
        env["SEHHO_ERROR_DECIMALS"] = str(DECIMALS)
    script = " && ".join(f"{{ {sys.executable} scripts/{s}.py > log_{s}.txt 2>&1; true; }}"
                         for s in SCRIPTS)
    return subprocess.Popen(["bash", "-c", script], cwd=tmp, env=env)


# ------------------------------------------------------------------ helpers
def _num(v):
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _fmt(v):
    f = _num(v)
    if f is None or isinstance(v, (bool, np.bool_)):
        return str(v)
    return f"{f:.6g}"


def _keys(a, b):
    ks = [c for c in KEYS if c in a.columns and c in b.columns]
    if ks and not a.duplicated(ks).any() and not b.duplicated(ks).any():
        ka = a[ks].apply(lambda r: "|".join(map(str, r)), axis=1)
        kb = b[ks].apply(lambda r: "|".join(map(str, r)), axis=1)
        if set(ka) == set(kb):
            return ks, a.set_index(ka), b.set_index(kb).loc[ka.values]
    return None, a.reset_index(drop=True), b.reset_index(drop=True)


def _row_label(r, ks):
    cols = ks or [c for c in KEYS if c in r.index]
    return "; ".join(f"{c}={r[c]}" for c in cols if c in r.index and pd.notna(r[c]))


def _interval_pairs(cols):
    pairs = []
    for c in cols:
        if c.endswith("_lo") and c[:-3] + "_hi" in cols:
            pairs.append((c[:-3], c, c[:-3] + "_hi"))
        elif c == "ci_lo" and "ci_hi" in cols:
            pairs.append(("ci", c, "ci_hi"))
    return pairs


def _is_adjusted(prefix):
    return "adj" in prefix


# ------------------------------------------------------------------ part 1: analysis files
def compare_frame(name, a, b, out):
    ks, a, b = _keys(a, b)
    if len(a) != len(b):
        raise SystemExit(f"{name}: {len(a)} rows at full precision, {len(b)} rounded")
    cols = [c for c in a.columns if c in b.columns]
    pairs = _interval_pairs(cols)
    paired_cols = {c for _, lo, hi in pairs for c in (lo, hi)}
    # stored sides and margin verdicts are recomputed from the intervals below
    derived = {c for c in cols if re.match(r"^side_", c) or re.match(r"^verdict_(pct|bca|t)_0\.", c)}
    delta_scale = name != "extras.csv"
    for (ia, ra), (_, rb) in zip(a.iterrows(), b.iterrows()):
        label = _row_label(ra, ks)
        cid = f"{name} | {label}"

        def add(kind, quantity, va, vb):
            out.append(dict(kind=kind, file=name, row=label, quantity=quantity,
                            full=_fmt(va), rounded=_fmt(vb), comparison=cid))

        for prefix, lo, hi in pairs:
            la, ha, lb, hb = (_num(ra[lo]), _num(ra[hi]), _num(rb[lo]), _num(rb[hi]))
            if None in (la, ha, lb, hb):
                continue
            adj = _is_adjusted(prefix)
            iv_a, iv_b = f"[{la:+.4f}, {ha:+.4f}]", f"[{lb:+.4f}, {hb:+.4f}]"
            if side(la, ha) != side(lb, hb):
                add("adjusted interval side" if adj else "interval side", f"{prefix} interval",
                    f"{side(la, ha)} {iv_a}", f"{side(lb, hb)} {iv_b}")
            if delta_scale and max(abs(la), abs(ha), abs(lb), abs(hb)) <= 1.0:
                for m in MARGINS:
                    va, vb = margin_verdict(la, ha, m), margin_verdict(lb, hb, m)
                    if va != vb:
                        add("adjusted margin verdict" if adj else "margin verdict",
                            f"{prefix} verdict at {m:g}", f"{va} {iv_a}", f"{vb} {iv_b}")
                if not adj:
                    ta = bool((0 < la < TOL) or (-TOL < ha < 0))
                    tb = bool((0 < lb < TOL) or (-TOL < hb < 0))
                    if ta != tb:
                        add("tie flag (Section 2.6)", f"{prefix} tie", f"{ta} {iv_a}", f"{tb} {iv_b}")
        for c in cols:
            if c in paired_cols or c in derived or c in KEYS or c in FREE_TEXT:
                continue
            va, vb = ra[c], rb[c]
            if (pd.isna(va) and pd.isna(vb)) or (isinstance(va, str) and va == vb):
                continue
            fa, fb = _num(va), _num(vb)
            if P_COLS.match(c) and fa is not None and fb is not None:
                if (fa < 0.05) != (fb < 0.05):
                    add("Holm decision" if "holm" in c else "unadjusted test decision",
                        f"{c} < 0.05", va, vb)
            elif COUNT_COLS.match(c) and fa is not None and fb is not None:
                if fa != fb:
                    add("#1 count" if c == "rank1_count" else "rank position" if c == "position" else
                        "win/tie/loss count" if c in ("wins", "losses", "ties", "first_better",
                                                      "second_better", "wins_for_gate_off")
                        else "count", c, va, vb)
            elif RANK_COLS.match(c) and fa is not None and fb is not None:
                if f"{fa:.2f}" != f"{fb:.2f}":
                    add("rank", c, va, vb)
            elif EST_COLS.match(c) and fa is not None and fb is not None:
                if abs(fb - fa) > SHIFT:
                    add("estimate shift > 0.02", c, va, vb)
            elif fa is None or fb is None or isinstance(va, (bool, np.bool_)):
                if str(va) != str(vb):
                    kind = ("near-zero bound flag (tie_002)" if c.endswith("tie_002") else
                            "magnitude label" if c == "magnitude" else
                            "per-function rank-sum outcome" if c == "outcome" else
                            "W/T/L" if c == "win_tie_loss" else
                            "Holm decision" if c.startswith("significant") else
                            "decision or label")
                    add(kind, c, va, vb)


def _flatten(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix, obj


def compare_json(name, a, b, out):
    fa, fb = dict(_flatten(a)), dict(_flatten(b))
    for k in sorted(set(fa) | set(fb)):
        va, vb = fa.get(k), fb.get(k)
        if va == vb or (isinstance(va, float) and isinstance(vb, float) and np.isnan(va) and np.isnan(vb)):
            continue
        na, nb = _num(va), _num(vb)
        if na is not None and nb is not None and not isinstance(va, bool):
            if P_COLS.match(k.split(".")[-1]) and (na < 0.05) != (nb < 0.05):
                out.append(dict(kind="Holm decision" if "holm" in k else "unadjusted test decision",
                                file=name, row=k, quantity=f"{k} < 0.05", full=_fmt(va),
                                rounded=_fmt(vb), comparison=f"{name} | {k}"))
            elif abs(nb - na) > SHIFT and abs(na) <= 1 and abs(nb) <= 1:
                out.append(dict(kind="estimate shift > 0.02", file=name, row=k, quantity=k,
                                full=_fmt(va), rounded=_fmt(vb), comparison=f"{name} | {k}"))
            continue
        same_words = isinstance(va, str) and isinstance(vb, str) and _skeleton(va) == _skeleton(vb)
        out.append(dict(kind="numbers inside a stored note" if same_words else "decision or label",
                        file=name, row=k, quantity=k, full=str(va), rounded=str(vb),
                        comparison=f"{name} | {k}"))


# ------------------------------------------------------------------ part 2: tables
_AMP = re.compile(r"(?<!\\)&")
_NUM = re.compile(r"[-+]?\d[\d,{}.]*(?:\\times10\^\{?-?\d+\}?)?")


def _cells(line):
    s = line.strip()
    s = re.sub(r"\\\\\s*(\[[^\]]*\])?\s*$", "", s)
    return [c.strip() for c in _AMP.split(s)]


def _skeleton(s):
    return _NUM.sub("#", s)


def _cell_kind(header, ca, cb):
    plain_a = re.sub(r"\\textbf\{|\\emph\{|\}|\$", "", ca).strip()
    plain_b = re.sub(r"\\textbf\{|\\emph\{|\}|\$", "", cb).strip()
    if re.fullmatch(r"\d+/\d+/\d+", plain_a) and re.fullmatch(r"\d+/\d+/\d+", plain_b):
        return "W/T/L cell"
    if re.fullmatch(r"\d+/\d+", plain_a) and re.fullmatch(r"\d+/\d+", plain_b):
        return "W/L cell"
    words = re.compile(r"negligible|small|medium|large")
    if words.search(ca) or words.search(cb):
        if words.findall(ca) != words.findall(cb):
            return "magnitude label"
    if _skeleton(ca) != _skeleton(cb):
        return "printed mark or label"
    h = header.lower()
    if "rank" in h:
        return "rank"
    if "#1" in h or "\\#1" in header:
        return "#1 count"
    if "holm" in h or re.search(r"(^|[^a-z])p($|[^a-z])", h):
        return "p-value"
    if "delta" in h or "[" in ca:
        return "delta or interval value"
    return "other printed number"


def _row_keys(lines):
    """A key for every tabular row: block number (rules and group lines start a block) and
    the row's first cell (a row with an empty first cell is keyed by the row above), so that
    rows a table orders by a statistic are matched by name, not by position."""
    keys, block, last = [], 0, ""
    for x in lines:
        if "\\midrule" in x or (x.startswith("\\multicolumn") and "&" not in x):
            block += 1
        if "&" in x:
            first = _cells(x)[0]
            last = first if first not in ("", "\\quad") else f"{last} +"
            if first == "\\quad":
                last = f"{block}:{_cells(x)[1]}"
            keys.append((block, last))
        else:
            keys.append(None)
    seen = {}
    for i, k in enumerate(keys):        # a label that repeats is matched by its occurrence
        if k is not None:
            seen[k] = seen.get(k, 0) + 1
            keys[i] = (k[0], k[1], seen[k])
    return keys


def _column_heads(rows):
    """Column labels from the heading rows (multicolumn groups expanded)."""
    expanded = []
    for r in rows:
        cols = []
        for c in _cells(r):
            m = re.match(r"\\multicolumn\{(\d+)\}\{[^}]*\}\{(.*)\}\s*$", c)
            cols += [m.group(2)] * int(m.group(1)) if m else [c]
        expanded.append(cols)
    if not expanded:
        return []
    n = len(expanded[-1])
    return [" / ".join(e[j] for e in expanded if len(e) == n and e[j]) for j in range(n)]


def compare_tables(full_tex, rnd_tex, index, out):
    import difflib
    for f in sorted(glob.glob(os.path.join(full_tex, "*.tex"))):
        name = os.path.basename(f)
        la = open(f).read().split("\n")
        lb = open(os.path.join(rnd_tex, name)).read().split("\n")
        label, doc = index.get(name, ("", ""))

        def add(kind, row, column, full, rounded):
            out.append(dict(kind=kind, table=name, label=label, document=doc, row=row,
                            column=column, full=full, rounded=rounded))

        if len(la) != len(lb):
            add("table structure", "", "", f"{len(la)} lines", f"{len(lb)} lines")
            continue
        ka, kb = _row_keys(la), _row_keys(lb)
        pos_b = {k: i for i, k in enumerate(kb) if k is not None}
        if [k for k in ka if k] != [k for k in kb if k] and set(pos_b) == {k for k in ka if k}:
            moved = [k[1] for k, q in zip(ka, kb) if k and k != q]
            add("row order (rows sorted by a statistic)", "; ".join(dict.fromkeys(moved))[:200], "",
                "", "")
        header, group = [], ""
        for i, x in enumerate(la):
            if "\\midrule" in x:
                if not header:      # the column heads: every row above the first rule
                    header = _column_heads([l for l in la[:i] if "&" in l])
                continue
            if x.startswith("\\multicolumn") and "&" not in x:
                group = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}\{", "", x)[:80]
            y = lb[pos_b[ka[i]]] if ka[i] is not None and ka[i] in pos_b else lb[i]
            if x == y:
                continue
            if "&" in x and "&" in y:
                ca, cb = _cells(x), _cells(y)
                if len(ca) != len(cb):
                    add("table structure", ca[0][:60], "", x.strip(), y.strip())
                    continue
                rowlab = ca[0]
                if ca[0] == "\\quad" and group:
                    rowlab = f"{group} / {ca[1]}"
                elif ca[0].startswith("\\quad") and group:
                    rowlab = f"{group} / {ca[0]}"
                elif ca[0] == "" and ka[i]:
                    rowlab = ka[i][1]
                for j, (u, v) in enumerate(zip(ca, cb)):
                    if u != v:
                        hdr = header[j] if len(header) == len(ca) else f"column {j + 1}"
                        add(_cell_kind(hdr, u, v), rowlab[:120], hdr[:90], u, v)
            else:
                # a note, caption or group line: list the changed tokens with some context
                ta, tb = re.split(r"(\s+)", x), re.split(r"(\s+)", y)
                sm = difflib.SequenceMatcher(a=ta, b=tb, autojunk=False)
                for op, i1, i2, j1, j2 in sm.get_opcodes():
                    if op != "equal":
                        add("table note text", "".join(ta[max(0, i1 - 16):i1])[-90:], "",
                            "".join(ta[i1:i2]), "".join(tb[j1:j2]))


# ------------------------------------------------------------------ main
def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    full = tempfile.mkdtemp(prefix="resolution_full_")
    rnd = tempfile.mkdtemp(prefix="resolution_rounded_")
    try:
        print("preparing the full-precision copy and the rounded copy")
        aged = _prepare(full, False)
        _prepare(rnd, True)
        procs = [_run_chain(full, False), _run_chain(rnd, True)]
        for p in procs:
            p.wait()
        # every compared output must have been rewritten by both runs
        compared, stale = [], []
        for p in sorted(glob.glob(os.path.join(full, "results", "analysis", "*"))):
            name = os.path.basename(p)
            if name in SKIP or not name.endswith((".csv", ".json")):
                continue
            q = os.path.join(rnd, "results", "analysis", name)
            fresh = [os.path.getmtime(x) > aged + 3600 for x in (p, q)]
            if all(fresh):
                compared.append(name)
            elif any(fresh):
                raise SystemExit(f"{name} was rewritten by one run only: see the logs in {full} / {rnd}")
            else:
                stale.append(name)
        for p in glob.glob(os.path.join(rnd, "results", "tex", "*.tex")) + \
                glob.glob(os.path.join(full, "results", "tex", "*.tex")):
            if os.path.getmtime(p) <= aged + 3600:
                raise SystemExit(f"{p} was not rewritten: see log_make_tables.txt")
        # does the full-precision re-run reproduce the released files?
        not_repro = [n for n in compared
                     if _sha(os.path.join(full, "results", "analysis", n)) !=
                     _sha(os.path.join(ROOT, "results", "analysis", n))]
        not_repro += ["tex/" + os.path.basename(p) for p in
                      sorted(glob.glob(os.path.join(full, "results", "tex", "*.tex")))
                      if _sha(p) != _sha(os.path.join(ROOT, "results", "tex", os.path.basename(p)))]

        rows = []
        for name in compared:
            pa = os.path.join(full, "results", "analysis", name)
            pb = os.path.join(rnd, "results", "analysis", name)
            if name.endswith(".json"):
                compare_json(name, json.load(open(pa)), json.load(open(pb)), rows)
            else:
                compare_frame(name, pd.read_csv(pa), pd.read_csv(pb), rows)
        idx = pd.read_csv(os.path.join(full, "results", "analysis", "revision_table_index.csv"))
        index = {r.file: (r.label, r.document) for r in idx.itertuples()}
        cells = []
        compare_tables(os.path.join(full, "results", "tex"), os.path.join(rnd, "results", "tex"),
                       index, cells)
    finally:
        shutil.rmtree(full, ignore_errors=True)
        shutil.rmtree(rnd, ignore_errors=True)

    out = pd.DataFrame(rows, columns=["kind", "file", "row", "quantity", "full", "rounded", "comparison"])
    out.to_csv(OUT, index=False)
    cel = pd.DataFrame(cells, columns=["kind", "table", "label", "document", "row", "column",
                                       "full", "rounded"])
    cel.to_csv(OUT_CELLS, index=False)

    print(f"\ncompared {len(compared)} analysis files and every table; not error-dependent or "
          f"not re-run here (unchanged): {', '.join(stale) or 'none'}")
    print("full-precision re-run differs from the released file: " + (", ".join(not_repro) or "none"))
    print(f"\nanalysis files: {len(out)} individual changes in {out.comparison.nunique()} rows")
    for k, g in out.groupby("kind"):
        print(f"  {k:<32s} {len(g):4d} changes in {g.comparison.nunique():3d} rows")
    print(f"\nprinted table cells and notes: {len(cel)} changes in {cel.table.nunique()} tables")
    for (d, k), g in cel.groupby(["document", "kind"]):
        print(f"  {d:<5s} {k:<26s} {len(g):4d}")
    print(f"wrote {os.path.relpath(OUT, ROOT)} and {os.path.relpath(OUT_CELLS, ROOT)}")


if __name__ == "__main__":
    main()
