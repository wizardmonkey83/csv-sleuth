from __future__ import annotations
 
from dataclasses import dataclass
 
import numpy as np
import pandas as pd
 
from .findings import Finding, FindingCheckError
 
START = pd.Timestamp("2025-01-01")  # fixed, never "today"
 
 
@dataclass
class TableSpec:
    name: str
    date_column: str
    date_format: str  # strftime format for the standard date column
    date_pattern: str  # regex that matches the standard format
    alt_date_formats: list[str]
    category_columns: list[str]
    sentinel_columns: dict[str, list[int]]  # column -> placeholder values it may get
    outlier_column: str
    unit: dict | None  # how to plant a unit mismatch, or None if this table can't have one
 
 
def _pick(rng: np.random.Generator, options: list) -> str:
    return str(options[int(rng.integers(len(options)))])
 
 
# ---------------------------------------------------------------- sales
 
 
def build_sales(rng: np.random.Generator, patterns: tuple[str, ...]):
    n = int(rng.integers(800, 2500))
    regions = ["East", "West", "North", "South"]
    prices = {"Widget": 20.0, "Gadget": 35.0, "Gizmo": 12.5, "Doohickey": 48.0}
 
    region = rng.choice(regions, size=n, p=[0.35, 0.30, 0.20, 0.15])
    product = rng.choice(list(prices), size=n)
    channel = rng.choice(["online", "store"], size=n, p=[0.6, 0.4])
    date = START + pd.to_timedelta(rng.integers(0, 365, size=n), unit="D")
    month = date.month.to_numpy()
 
    rate = np.full(n, 4.0)  # expected units per order before patterns
    findings = []
    if "trend" in patterns:
        change = round(float(rng.uniform(0.04, 0.08)), 3)
        direction = _pick(rng, ["up", "down"])
        rate *= (1 + change if direction == "up" else 1 - change) ** (month - 1)
        findings.append(Finding(
            "trend", ["date", "units"], {"direction": direction, "monthly_change": change},
            f"Units per order trend {direction} over the year (about {change:.0%} per month).",
        ))
    if "seasonality" in patterns:
        rate *= np.where(month == 12, 2.5, 1.0)
        findings.append(Finding(
            "seasonality", ["date", "units"], {"peak_month": 12, "multiplier": 2.5},
            "Units per order spike in December (about 2.5x the expected level).",
        ))
    if "segment_gap" in patterns:
        weak = _pick(rng, regions)
        rate *= np.where(region == weak, 0.4, 1.0)
        findings.append(Finding(
            "segment_gap", ["region", "units"], {"segment": weak, "metric": "units", "ratio": 0.4},
            f"The {weak} region sells far fewer units per order than the other regions.",
        ))
 
    units = rng.poisson(rate) + 1
    unit_price = np.round(np.array([prices[p] for p in product]) * rng.normal(1, 0.05, n), 2)
    df = pd.DataFrame({
        "date": date,
        "region": region,
        "product": product,
        "channel": channel,
        "units": units,
        "unit_price": unit_price,
        "revenue": np.round(units * unit_price, 2),
    }).sort_values("date", kind="stable").reset_index(drop=True)
 
    _verify_sales(df, findings)
    spec = TableSpec(
        name="sales",
        date_column="date",
        date_format="%Y-%m-%d",
        date_pattern=r"\d{4}-\d{2}-\d{2}",
        alt_date_formats=["%m/%d/%Y", "%d %b %Y"],
        category_columns=["region", "product", "channel"],
        sentinel_columns={"units": [-1, 9999], "unit_price": [-1, 9999]},
        outlier_column="revenue",
        unit={"column": "unit_price", "kind": "scale", "factor": 100,
              "correct_unit": "dollars", "wrong_unit": "cents"},
    )
    return df, spec, findings
 
 
def _verify_sales(df: pd.DataFrame, findings: list[Finding]) -> None:
    month = df["date"].dt.month
    for f in findings:
        if f.type == "trend":
            q1 = df.loc[month <= 3, "units"].mean()
            q3 = df.loc[month.between(7, 9), "units"].mean()
            ok = q3 > 1.15 * q1 if f.details["direction"] == "up" else q3 < q1 / 1.15
        elif f.type == "seasonality":
            ok = df.loc[month == 12, "units"].mean() > 1.5 * df.loc[month == 11, "units"].mean()
        elif f.type == "segment_gap":
            weak = df["region"] == f.details["segment"]
            ok = df.loc[weak, "units"].mean() < 0.75 * df.loc[~weak, "units"].mean()
        else:
            continue
        if not ok:
            raise FindingCheckError(f"sales {f.type} is not detectable")
 
 
# ---------------------------------------------------------------- signups
 
 
def build_signups(rng: np.random.Generator, patterns: tuple[str, ...]):
    n = int(rng.integers(1500, 4000))
    countries = ["US", "UK", "DE", "IN", "BR"]
 
    segment = rng.choice(["enterprise", "consumer"], size=n, p=[0.4, 0.6])
    country = rng.choice(countries, size=n, p=[0.35, 0.20, 0.15, 0.18, 0.12])
    plan = rng.choice(["free", "pro", "team"], size=n, p=[0.6, 0.3, 0.1])
    age = rng.integers(18, 70, size=n)
    session_minutes = np.round(rng.gamma(2.0, 6.0, size=n), 1)
    signup_date = START + pd.to_timedelta(rng.integers(0, 365, size=n), unit="D")
 
    base = np.where(segment == "enterprise", 0.55, 0.20)
    findings = []
    if "simpsons_paradox" in patterns:
        # Treatment goes mostly to consumers, who convert less, so it looks worse
        # overall even though it helps within each segment.
        p_treat = np.where(segment == "enterprise", 0.2, 0.75)
        treated = rng.random(n) < p_treat
        base = base + np.where(treated, 0.10, 0.0)
        findings.append(Finding(
            "simpsons_paradox", ["treatment", "converted", "segment"],
            {"overall": "treatment converts worse", "within_segments": "treatment converts better"},
            "The treatment group converts worse overall but better within each segment (Simpson's paradox).",
        ))
    else:
        treated = rng.random(n) < 0.5  # treatment has no effect
 
    rate = base
    if "segment_gap" in patterns:
        weak = _pick(rng, countries)
        rate = rate * np.where(country == weak, 0.4, 1.0)
        findings.append(Finding(
            "segment_gap", ["country", "converted"], {"segment": weak, "metric": "conversion rate", "ratio": 0.4},
            f"Signups from {weak} convert at a much lower rate than other countries.",
        ))
 
    converted = _exact_conversions(rng, rate)
    df = pd.DataFrame({
        "signup_date": signup_date,
        "plan": plan,
        "country": country,
        "segment": segment,
        "age": age,
        "session_minutes": session_minutes,
        "treatment": np.where(treated, "new_onboarding", "control"),
        "converted": converted,
    }).sort_values("signup_date", kind="stable").reset_index(drop=True)
 
    _verify_signups(df, findings)
    spec = TableSpec(
        name="signups",
        date_column="signup_date",
        date_format="%Y-%m-%d",
        date_pattern=r"\d{4}-\d{2}-\d{2}",
        alt_date_formats=["%m/%d/%Y", "%d %b %Y"],
        category_columns=["plan", "country", "segment"],
        sentinel_columns={"age": [-1, 0, 999], "session_minutes": [-1, 9999]},
        outlier_column="session_minutes",
        unit=None,
    )
    return df, spec, findings
 
 
def _exact_conversions(rng: np.random.Generator, rate: np.ndarray) -> np.ndarray:
    """Within each group of rows sharing a rate, convert exactly round(rate * size) of them.
 
    Exact counts make patterns like Simpson's paradox hold by construction
    instead of depending on luck.
    """
    converted = np.zeros(len(rate), dtype=int)
    for r in np.unique(rate):  # np.unique returns sorted values, so the order is fixed
        idx = np.flatnonzero(rate == r)
        k = int(round(r * len(idx)))
        converted[rng.choice(idx, size=k, replace=False)] = 1
    return converted
 
 
def _verify_signups(df: pd.DataFrame, findings: list[Finding]) -> None:
    for f in findings:
        if f.type == "simpsons_paradox":
            overall = df.groupby("treatment")["converted"].mean()
            by_seg = df.groupby(["segment", "treatment"])["converted"].mean().unstack()
            ok = overall["new_onboarding"] < overall["control"] and bool(
                (by_seg["new_onboarding"] > by_seg["control"]).all()
            )
        elif f.type == "segment_gap":
            weak = df["country"] == f.details["segment"]
            ok = df.loc[weak, "converted"].mean() < 0.7 * df.loc[~weak, "converted"].mean()
        else:
            continue
        if not ok:
            raise FindingCheckError(f"signups {f.type} is not detectable")
 
 
# ---------------------------------------------------------------- sensor
 
 
def build_sensor(rng: np.random.Generator, patterns: tuple[str, ...]):
    sensors = [f"S{i:02d}" for i in range(1, int(rng.integers(4, 8)))]  # 3 to 6 sensors
    hours = int(rng.integers(24 * 20, 24 * 45))  # 20 to 45 days of hourly readings
    stamps = START + pd.to_timedelta(np.arange(hours), unit="h")
 
    timestamp = np.tile(stamps.to_numpy(), len(sensors))
    sensor_id = np.repeat(sensors, hours)
    hour_of_day = np.tile(stamps.hour.to_numpy(), len(sensors))
    day = np.tile(np.arange(hours) / 24, len(sensors))
    n = len(timestamp)
 
    temp = 21 + 4 * np.sin(2 * np.pi * (hour_of_day - 9) / 24) + rng.normal(0, 0.8, n)
    findings = []
    if "trend" in patterns:
        per_day = round(float(rng.uniform(0.08, 0.15)), 3)
        temp = temp + per_day * day
        findings.append(Finding(
            "trend", ["timestamp", "temperature_c"], {"direction": "up", "per_day": per_day},
            f"Temperature rises steadily over time (about {per_day} °C per day).",
        ))
    if "segment_gap" in patterns:
        hot = _pick(rng, sensors)
        temp = temp + np.where(sensor_id == hot, 3.0, 0.0)
        findings.append(Finding(
            "segment_gap", ["sensor_id", "temperature_c"], {"segment": hot, "metric": "temperature_c", "offset": 3.0},
            f"Sensor {hot} reads about 3 °C warmer than the other sensors.",
        ))
 
    df = pd.DataFrame({
        "timestamp": timestamp,
        "sensor_id": sensor_id,
        "temperature_c": np.round(temp, 2),
        "humidity": np.round(np.clip(rng.normal(45, 5, n), 10, 95), 1),
        "battery_v": np.round(rng.uniform(3.0, 3.7, n), 2),
    }).sort_values(["timestamp", "sensor_id"], kind="stable").reset_index(drop=True)
 
    _verify_sensor(df, findings)
    spec = TableSpec(
        name="sensor",
        date_column="timestamp",
        date_format="%Y-%m-%d %H:%M:%S",
        date_pattern=r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}",
        alt_date_formats=["%m/%d/%Y %H:%M", "%d %b %Y %H:%M"],
        category_columns=["sensor_id"],
        sentinel_columns={"humidity": [-1, 9999], "temperature_c": [-999, 9999], "battery_v": [-1, 0]},
        outlier_column="humidity",
        unit={"column": "temperature_c", "kind": "c_to_f",
              "correct_unit": "Celsius", "wrong_unit": "Fahrenheit"},
    )
    return df, spec, findings
 
 
def _verify_sensor(df: pd.DataFrame, findings: list[Finding]) -> None:
    for f in findings:
        if f.type == "trend":
            days = (df["timestamp"] - df["timestamp"].min()).dt.days
            ok = df.loc[days >= days.max() - 2, "temperature_c"].mean() > df.loc[days <= 2, "temperature_c"].mean() + 1
        elif f.type == "segment_gap":
            hot = df["sensor_id"] == f.details["segment"]
            ok = df.loc[hot, "temperature_c"].mean() > df.loc[~hot, "temperature_c"].mean() + 2
        else:
            continue
        if not ok:
            raise FindingCheckError(f"sensor {f.type} is not detectable")
 
 
BUILDERS = {"sales": build_sales, "signups": build_signups, "sensor": build_sensor}
TABLE_PATTERNS = {
    "sales": ("trend", "seasonality", "segment_gap"),
    "signups": ("segment_gap", "simpsons_paradox"),
    "sensor": ("trend", "segment_gap"),
}
