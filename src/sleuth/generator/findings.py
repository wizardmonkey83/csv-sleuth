from __future__ import annotations
 
from dataclasses import dataclass, field
 
import numpy as np
import pandas as pd
 
# Patterns are built into the data when a base table is generated.
PATTERN_TYPES = ("trend", "seasonality", "segment_gap", "simpsons_paradox")
# Corruptions are added afterward by injectors, in exactly this order.
CORRUPTION_TYPES = (
    "outlier",
    "sentinel_value",
    "unit_mismatch",
    "inconsistent_category",
    "mixed_date_format",
    "duplicate_rows",
)
FINDING_TYPES = PATTERN_TYPES + CORRUPTION_TYPES
 
 
class FindingCheckError(Exception):
    """A planted finding is missing or not detectable; the build must stop."""
 
 
@dataclass
class Finding:
    type: str
    columns: list[str]
    details: dict = field(default_factory=dict)
    description: str = ""
 
    def to_dict(self, finding_id: str) -> dict:
        return {
            "id": finding_id,
            "type": self.type,
            "columns": list(self.columns),
            "details": to_python(self.details),
            "description": self.description,
        }
 
 
def to_python(value):
    """Convert numpy values to plain Python so they can be written as JSON."""
    if isinstance(value, dict):
        return {str(k): to_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_python(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (np.str_, np.bool_)):
        return value.item()
    return value
 
 
def remeasure(finding: Finding, df: pd.DataFrame) -> None:
    """Recount a corruption on the final table, confirm it's detectable, and describe it.
 
    Later injectors can change earlier counts (duplicating rows can duplicate
    sentinel rows too), so the answer key always uses counts from the final table.
    Patterns were already verified when the base table was built.
    """
    t, d = finding.type, finding.details
    col = finding.columns[0] if finding.columns else None
 
    if t == "duplicate_rows":
        count = int(df.duplicated().sum())
    elif t == "sentinel_value":
        count = int((df[col] == d["value"]).sum())
    elif t == "outlier":
        count = int((df[col] == d["value"]).sum())
        if count and d["value"] < 10 * df[col].median():
            raise FindingCheckError(f"outlier in {col} is not extreme enough")
    elif t == "unit_mismatch":
        count = int((df[col] > d["threshold"]).sum())
    elif t == "inconsistent_category":
        s = df[col].astype(str)
        canon = d["canonical"]
        count = int(((s != canon) & (s.str.strip().str.lower() == canon.lower())).sum())
    elif t == "mixed_date_format":
        count = int((~df[col].astype(str).str.fullmatch(d["standard_pattern"])).sum())
    else:
        return
 
    if count == 0:
        raise FindingCheckError(f"{t} in {col or 'table'} is not present in the final data")
    d["count"] = count
    finding.description = describe(finding)
 
 
def describe(f: Finding) -> str:
    """A plain-English statement of the finding, for reports and the LLM judge."""
    d, col = f.details, (f.columns[0] if f.columns else "")
    if f.type == "duplicate_rows":
        return f"{d['count']} rows are exact duplicates of other rows."
    if f.type == "sentinel_value":
        return f"{d['count']} rows in '{col}' hold the placeholder value {d['value']} instead of real data."
    if f.type == "outlier":
        return f"An extreme value ({d['value']}) in '{col}' appears in {d['count']} row(s) and distorts the average."
    if f.type == "unit_mismatch":
        return (
            f"{d['count']} rows in '{col}' between {d['window_start']} and {d['window_end']} "
            f"are recorded in {d['wrong_unit']} instead of {d['correct_unit']}."
        )
    if f.type == "inconsistent_category":
        variants = ", ".join(repr(v) for v in d["variants"])
        return f"'{d['canonical']}' in '{col}' is also written as {variants} ({d['count']} rows)."
    if f.type == "mixed_date_format":
        return f"'{col}' mixes date formats; {d['count']} rows are not in the standard {d['standard_format']} format."
    return f.description