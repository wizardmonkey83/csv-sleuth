from __future__ import annotations
 
import numpy as np
import pandas as pd
 
from .base_tables import TableSpec
from .findings import Finding
 
 
def _rows(rng: np.random.Generator, n: int, low: float, high: float) -> np.ndarray:
    """Pick a random share of row positions, between `low` and `high` of the table."""
    k = max(1, int(round(n * rng.uniform(low, high))))
    return np.sort(rng.choice(n, size=k, replace=False))
 
 
def _reserved_columns(spec: TableSpec, plants: list[str]) -> set[str]:
    """Columns other injectors will use, so two problems never land on the same column."""
    reserved = set()
    if "outlier" in plants:
        reserved.add(spec.outlier_column)
    if "unit_mismatch" in plants and spec.unit:
        reserved.add(spec.unit["column"])
    return reserved
 
 
def inject_outlier(df, rng, spec, plants):
    col = spec.outlier_column
    row = int(rng.integers(len(df)))
    value = round(float(df[col].median() * rng.uniform(40, 80)), 2)
    df = df.copy()
    df[col] = df[col].astype(float)
    df.loc[row, col] = value
    return df, Finding("outlier", [col], {"value": value})
 
 
def inject_sentinel(df, rng, spec, plants):
    reserved = _reserved_columns(spec, plants)
    options = [c for c in spec.sentinel_columns if c not in reserved]
    col = options[int(rng.integers(len(options)))]
    values = spec.sentinel_columns[col]
    value = values[int(rng.integers(len(values)))]
    rows = _rows(rng, len(df), 0.01, 0.03)
    df = df.copy()
    df.loc[rows, col] = value
    return df, Finding("sentinel_value", [col], {"value": value})
 
 
def inject_unit_mismatch(df, rng, spec, plants):
    """Record a contiguous stretch of time in the wrong unit, as if one source was misconfigured."""
    unit = spec.unit
    col = unit["column"]
    n = len(df)
    length = max(5, int(round(n * rng.uniform(0.06, 0.12))))
    start = int(rng.integers(0, n - length))
    rows = np.arange(start, start + length)  # the table is still sorted by date here
 
    df = df.copy()
    normal_max = float(df.drop(index=rows)[col].max())
    if unit["kind"] == "scale":
        df.loc[rows, col] = np.round(df.loc[rows, col] * unit["factor"], 2)
    else:  # Celsius recorded as Fahrenheit
        df.loc[rows, col] = np.round(df.loc[rows, col] * 9 / 5 + 32, 2)
    wrong_min = float(df.loc[rows, col].min())
    dates = df.loc[rows, spec.date_column]
    return df, Finding("unit_mismatch", [col], {
        "correct_unit": unit["correct_unit"],
        "wrong_unit": unit["wrong_unit"],
        "window_start": dates.min().strftime(spec.date_format),
        "window_end": dates.max().strftime(spec.date_format),
        "threshold": round((normal_max + wrong_min) / 2, 2),  # used to recount affected rows
    })
 
 
def inject_inconsistent_category(df, rng, spec, plants):
    col = spec.category_columns[int(rng.integers(len(spec.category_columns)))]
    values = sorted(df[col].astype(str).unique())
    canonical = values[int(rng.integers(len(values)))]
 
    candidates = [canonical.lower(), canonical.upper(), canonical + " ", " " + canonical]
    candidates = list(dict.fromkeys(v for v in candidates if v != canonical))  # dedupe, keep order
    picks = rng.choice(len(candidates), size=min(2, len(candidates)), replace=False)
    variants = [candidates[i] for i in sorted(picks)]
 
    positions = np.flatnonzero(df[col].astype(str).to_numpy() == canonical)
    k = max(1, int(round(len(positions) * rng.uniform(0.25, 0.40))))
    chosen = np.sort(rng.choice(positions, size=k, replace=False))
    df = df.copy()
    df[col] = df[col].astype(str)
    df.loc[chosen, col] = [variants[int(i)] for i in rng.integers(len(variants), size=k)]
    return df, Finding("inconsistent_category", [col], {"canonical": canonical, "variants": variants})
 
 
def inject_mixed_dates(df, rng, spec, plants):
    col = spec.date_column
    rows = _rows(rng, len(df), 0.15, 0.35)
    formats = [spec.alt_date_formats[int(i)] for i in rng.integers(len(spec.alt_date_formats), size=len(rows))]
 
    original = df[col]
    text = original.dt.strftime(spec.date_format)
    for fmt in sorted(set(formats)):
        rows_for_fmt = rows[[f == fmt for f in formats]]
        text.iloc[rows_for_fmt] = original.iloc[rows_for_fmt].dt.strftime(fmt).to_numpy()
    df = df.copy()
    df[col] = text
    return df, Finding("mixed_date_format", [col], {
        "standard_format": spec.date_format,
        "standard_pattern": spec.date_pattern,
        "other_formats": spec.alt_date_formats,
    })
 
 
def inject_duplicates(df, rng, spec, plants):
    """Copy a few rows and slot each copy in near its original, so they aren't all adjacent."""
    n = len(df)
    rows = _rows(rng, n, 0.02, 0.04)
    order_key = np.concatenate([np.arange(n, dtype=float), rows + rng.uniform(0.5, 30, size=len(rows))])
    combined = pd.concat([df, df.iloc[rows]], ignore_index=True)
    combined = combined.iloc[np.argsort(order_key, kind="stable")].reset_index(drop=True)
    return combined, Finding("duplicate_rows", [], {})
 
 
INJECTORS = {
    "outlier": inject_outlier,
    "sentinel_value": inject_sentinel,
    "unit_mismatch": inject_unit_mismatch,
    "inconsistent_category": inject_inconsistent_category,
    "mixed_date_format": inject_mixed_dates,
    "duplicate_rows": inject_duplicates,
}