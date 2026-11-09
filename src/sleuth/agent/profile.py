from __future__ import annotations

import pandas as pd
import os 


 
import sys
from pathlib import Path
from typing import TypedDict
 
import numpy as np
import pandas as pd
 
TOP_VALUES = 10
MAX_CATEGORIES = 50
EXAMPLES = 8
SAMPLE_ROWS = 5
MAX_VALUE_CHARS = 40
MISSING = "<missing>"
 
 
class ColumnProfile(TypedDict):
    name: str
    dtype: str
    missing: int
    unique: int
    stats: dict[str, float] | None
    top_values: list[tuple[str, int]] | None
    examples: list[str] | None
 
 
class DataProfile(TypedDict):
    rows: int
    duplicate_rows: int
    columns: list[ColumnProfile]
    sample: str
 
 
def build_profile(csv_path: str | Path) -> DataProfile:
    """Summarize the CSV exactly as it is on disk."""
    df = pd.read_csv(csv_path) 
    return {
        "rows": len(df),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": [_profile_column(df[name]) for name in df.columns],
        "sample": df.head(SAMPLE_ROWS).to_string(index=False),
    }
 
 
def _profile_column(s: pd.Series) -> ColumnProfile:
    column: ColumnProfile = {
        "name": str(s.name),
        "dtype": str(s.dtype),
        "missing": int(s.isna().sum()),
        "unique": int(s.nunique(dropna=True)),
        "stats": None,
        "top_values": None,
        "examples": None,
    }
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        d = s.describe()
        column["stats"] = {key: round(float(d[key]), 4) for key in ("min", "25%", "50%", "75%", "max", "mean")}
    elif column["unique"] <= MAX_CATEGORIES:
        counts = s.value_counts(dropna=False).head(TOP_VALUES)
        column["top_values"] = [(_short(value), int(count)) for value, count in counts.items()]
    else:
        values = s.dropna().astype(str)
        positions = sorted(set(np.linspace(0, len(values) - 1, EXAMPLES).astype(int)))
        column["examples"] = [_short(values.iloc[i]) for i in positions]
    return column
 
 
def _short(value) -> str:
    if pd.isna(value):
        return MISSING
    text = str(value)
    return text if len(text) <= MAX_VALUE_CHARS else text[:MAX_VALUE_CHARS] + "..."
 
 
def _show(value: str) -> str:
    # repr() adds quotes, which makes leading or trailing spaces visible: 'East '
    return value if value == MISSING else repr(value)
 
 
def render_profile(profile: DataProfile) -> str:
    """Compact text version of the profile, for the model's prompt."""
    lines = [
        f"Rows: {profile['rows']} | Columns: {len(profile['columns'])} | "
        f"Duplicate rows: {profile['duplicate_rows']}",
        "",
        "Columns:",
    ]
    for c in profile["columns"]:
        lines.append(f"- {c['name']} ({c['dtype']}): {c['missing']} missing, {c['unique']} unique")
        if c["stats"]:
            st = c["stats"]
            lines.append(
                f"    min {st['min']:g} | p25 {st['25%']:g} | median {st['50%']:g} | "
                f"p75 {st['75%']:g} | max {st['max']:g} | mean {st['mean']:g}"
            )
        if c["top_values"]:
            shown = ", ".join(f"{_show(v)} {n}" for v, n in c["top_values"])
            more = " (more not shown)" if c["unique"] > TOP_VALUES else ""
            lines.append(f"    top values: {shown}{more}")
        if c["examples"]:
            lines.append("    examples: " + ", ".join(_show(v) for v in c["examples"]))
    lines += ["", f"First {SAMPLE_ROWS} rows:", profile["sample"]]
    return "\n".join(lines)
 
 
def profile_node(state: dict) -> dict:
    """LangGraph node: reads the CSV path from state and adds the profile."""
    return {"profile": build_profile(state["csv_path"])}
 
 
if __name__ == "__main__":
    print(render_profile(build_profile(sys.argv[1])))