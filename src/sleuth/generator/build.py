from __future__ import annotations
 
import argparse
import json
from pathlib import Path
 
import numpy as np
import pandas as pd
 
from .base_tables import BUILDERS, TABLE_PATTERNS
from .findings import CORRUPTION_TYPES, FINDING_TYPES, PATTERN_TYPES, remeasure
from .injectors import INJECTORS
 
DEV_SEED, TEST_SEED = 1000, 9000  # separate ranges, so dev and test can never overlap
TABLES = list(BUILDERS)  # fixed order: sales, signups, sensor
 
 
def allowed_types(table: str) -> list[str]:
    corruptions = [t for t in CORRUPTION_TYPES if not (t == "unit_mismatch" and table == "signups")]
    return list(TABLE_PATTERNS[table]) + corruptions
 
 
def plan_split(prefix: str, n: int, base_seed: int) -> list[dict]:
    """Decide each dataset's table and plants, spreading finding types evenly.
 
    Rather than picking plants purely at random, each dataset takes the allowed
    types used least so far (random tie-breaks), so every type appears often
    enough to measure per-type recall. A couple of datasets are nearly clean,
    to measure whether the agent invents problems.
    """
    rng = np.random.default_rng(base_seed)
    clean_slots = {n // 4, (3 * n) // 4} if n >= 8 else {n // 2}
    counts = {t: 0 for t in FINDING_TYPES}
    plans = []
    for i in range(n):
        table = TABLES[i % len(TABLES)]
        allowed = allowed_types(table)
        k = int(rng.integers(0, 2)) if i in clean_slots else int(rng.integers(3, 6))
        tiebreak = {t: float(rng.random()) for t in allowed}
        ranked = sorted(allowed, key=lambda t: (counts[t], tiebreak[t]))
        plants = sorted(ranked[:k], key=FINDING_TYPES.index)
        for t in plants:
            counts[t] += 1
        plans.append({"dataset_id": f"{prefix}_{i:03d}", "seed": base_seed + i, "table": table, "plants": plants})
    return plans
 
 
def generate(plan: dict) -> tuple[pd.DataFrame, dict]:
    """Build one dataset from its plan. The same plan always gives the same output."""
    rng = np.random.default_rng(plan["seed"])
    plants = plan["plants"]
    patterns = tuple(t for t in plants if t in PATTERN_TYPES)
 
    df, spec, findings = BUILDERS[plan["table"]](rng, patterns)
    for t in CORRUPTION_TYPES:  # fixed order, duplicates last
        if t in plants:
            df, finding = INJECTORS[t](df, rng, spec, plants)
            findings.append(finding)
 
    if pd.api.types.is_datetime64_any_dtype(df[spec.date_column]):
        df[spec.date_column] = df[spec.date_column].dt.strftime(spec.date_format)
 
    for f in findings:
        remeasure(f, df)  # recounts on the final table; raises if a finding isn't detectable
 
    key = {
        "dataset_id": plan["dataset_id"],
        "seed": plan["seed"],
        "base_table": plan["table"],
        "rows": len(df),
        "columns": list(df.columns),
        "findings": [f.to_dict(f"f{i}") for i, f in enumerate(findings, start=1)],
    }
    return df, key
 
 
def build_split(out_dir: Path, prefix: str, n: int, base_seed: int) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for plan in plan_split(prefix, n, base_seed):
        df, key = generate(plan)
        df.to_csv(out_dir / f"{plan['dataset_id']}.csv", index=False)
        (out_dir / f"{plan['dataset_id']}.json").write_text(json.dumps(key, indent=2) + "\n")
        manifest.append({
            "dataset_id": key["dataset_id"],
            "seed": key["seed"],
            "base_table": key["base_table"],
            "rows": key["rows"],
            "finding_types": [f["type"] for f in key["findings"]],
        })
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
 
 
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Sleuth's dev and test datasets.")
    parser.add_argument("--out", default="data", help="output folder (default: data)")
    parser.add_argument("--dev", type=int, default=20, help="number of dev datasets")
    parser.add_argument("--test", type=int, default=10, help="number of held-out test datasets")
    args = parser.parse_args()
 
    out = Path(args.out)
    for name, n, seed in [("dev", args.dev, DEV_SEED), ("test", args.test, TEST_SEED)]:
        manifest = build_split(out / name, name, n, seed)
        per_type = {t: sum(t in m["finding_types"] for m in manifest) for t in FINDING_TYPES}
        print(f"{name}: {len(manifest)} datasets -> {out / name}")
        print("  findings per type:", {t: c for t, c in per_type.items() if c})
 
 
if __name__ == "__main__":
    main()