import sys
from pathlib import Path
from typing import TypedDict
 
import numpy as np
import pandas as pd
 
# How much detail the profile includes. Smaller numbers mean a shorter prompt.
TOP_VALUES = 10  # most common values to show for a text column
MAX_CATEGORIES = 50  # above this many unique values, show examples instead of counts
EXAMPLES = 8  # how many example values to show for columns with many unique values
SAMPLE_ROWS = 5  # how many rows to show at the end
MAX_VALUE_CHARS = 40  # longer values are cut off with "..."
MISSING = "<missing>"  # how empty cells are shown
 
 
class ColumnProfile(TypedDict):
    """Summary of one column. Only one of stats, top_values, or examples is filled in."""
 
    name: str
    dtype: str
    missing: int
    unique: int
    stats: dict[str, float] | None
    top_values: list[tuple[str, int]] | None
    examples: list[str] | None
 
 
class DataProfile(TypedDict):
    """Summary of the whole table."""
 
    rows: int
    duplicate_rows: int
    columns: list[ColumnProfile]
    sample: str
 
 
 
def build_profile(csv_path: str | Path) -> DataProfile:
    """Read the CSV and summarize it."""
    df = pd.read_csv(csv_path)
 
    columns = []
    for column_name in df.columns:
        columns.append(profile_column(df[column_name]))
 
    return {
        "rows": len(df),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": columns,
        "sample": df.head(SAMPLE_ROWS).to_string(index=False),
    }
 
 
def profile_column(column: pd.Series) -> ColumnProfile:
    """Summarize one column. What we include depends on the kind of column."""
    profile: ColumnProfile = {
        "name": str(column.name),
        "dtype": str(column.dtype),
        "missing": int(column.isna().sum()),
        "unique": int(column.nunique()),
        "stats": None,
        "top_values": None,
        "examples": None,
    }
 
    is_number = pd.api.types.is_numeric_dtype(column) and not pd.api.types.is_bool_dtype(column)
 
    if is_number:
        profile["stats"] = numeric_stats(column)
    elif profile["unique"] <= MAX_CATEGORIES:
        profile["top_values"] = most_common_values(column)
    else:
        profile["examples"] = example_values(column)
 
    return profile
 
 
def numeric_stats(column: pd.Series) -> dict[str, float]:
    """Min, quartiles, max, and mean of a numeric column."""
    summary = column.describe()
    return {
        "min": round(float(summary["min"]), 4),
        "p25": round(float(summary["25%"]), 4),
        "median": round(float(summary["50%"]), 4),
        "p75": round(float(summary["75%"]), 4),
        "max": round(float(summary["max"]), 4),
        "mean": round(float(summary["mean"]), 4),
    }
 
 
def most_common_values(column: pd.Series) -> list[tuple[str, int]]:
    """The most frequent values and how often each appears, e.g. [("East", 404), ...]."""
    counts = column.value_counts(dropna=False).head(TOP_VALUES)
 
    result = []
    for value, count in counts.items():
        result.append((shorten(value), int(count)))
    return result
 
 
def example_values(column: pd.Series) -> list[str]:
    """A few values spread evenly through the column, from first to last.
 
    Spreading them out (instead of taking the first few) means the examples
    come from the whole file, so an odd format partway through can show up.
    """
    values = column.dropna().astype(str)
 
    positions = np.linspace(0, len(values) - 1, EXAMPLES).astype(int)
    positions = sorted(set(positions))
 
    examples = []
    for position in positions:
        examples.append(shorten(values.iloc[position]))
    return examples
 
 
def shorten(value) -> str:
    """Turn a cell into text, cutting off long values and labeling empty ones."""
    if pd.isna(value):
        return MISSING
 
    text = str(value)
    if len(text) > MAX_VALUE_CHARS:
        return text[:MAX_VALUE_CHARS] + "..."
    return text
 
 
def render_profile(profile: DataProfile) -> str:
    """Write the profile as compact text, to put in the model's prompt."""
    lines = [
        f"Rows: {profile['rows']} | Columns: {len(profile['columns'])} | "
        f"Duplicate rows: {profile['duplicate_rows']}",
        "",
        "Columns:",
    ]
 
    for column in profile["columns"]:
        lines.extend(render_column(column))
 
    lines.append("")
    lines.append(f"First {SAMPLE_ROWS} rows:")
    lines.append(profile["sample"])
    return "\n".join(lines)
 
 
def render_column(column: ColumnProfile) -> list[str]:
    """The text lines for one column."""
    lines = [
        f"- {column['name']} ({column['dtype']}): "
        f"{column['missing']} missing, {column['unique']} unique"
    ]
 
    stats = column["stats"]
    if stats:
        lines.append(
            f"    min {stats['min']:g} | p25 {stats['p25']:g} | median {stats['median']:g} | "
            f"p75 {stats['p75']:g} | max {stats['max']:g} | mean {stats['mean']:g}"
        )
 
    if column["top_values"]:
        parts = []
        for value, count in column["top_values"]:
            parts.append(f"{quoted(value)} {count}")
        line = "    top values: " + ", ".join(parts)
        if column["unique"] > TOP_VALUES:
            line += " (more not shown)"
        lines.append(line)
 
    if column["examples"]:
        parts = []
        for value in column["examples"]:
            parts.append(quoted(value))
        lines.append("    examples: " + ", ".join(parts))
 
    return lines
 
 
def quoted(value: str) -> str:
    """Put quotes around a value so spaces are visible: 'East ' vs 'East'."""
    if value == MISSING:
        return value
    return repr(value)
 
 
def profile_node(state: dict) -> dict:
    """Read the CSV path from the state and add the profile to it."""
    profile = build_profile(state["csv_path"])
    return {"profile": profile}