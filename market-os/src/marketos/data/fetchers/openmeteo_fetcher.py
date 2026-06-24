"""Open-Meteo weather fetcher — keyless, no rate limit beyond fair use.

Sources ERA5 reanalysis (1940-present) + forecast for economically-weighted locations.
Weather features proxy:
  - Energy demand (heating/cooling degree days → EIA inventory surprises)
  - Agricultural commodity supply shocks
  - Retail foot traffic / consumer spending

Locations chosen for financial signal, not coverage:
  - US Gulf Coast (energy hub) — Houston TX
  - US Northeast (heating demand bell-weather) — New York NY
  - India (monsoon → power demand) — Mumbai
  - Europe energy hub — Frankfurt DE
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_BASE = "https://api.open-meteo.com/v1"
_ARCHIVE = "https://archive-api.open-meteo.com/v1"

# (name, lat, lon, tz)
LOCATIONS = {
    "us_gulf":     (29.76, -95.37, "America/Chicago"),    # Houston
    "us_northeast":(40.71, -74.01, "America/New_York"),   # New York
    "india_west":  (19.08,  72.88, "Asia/Kolkata"),       # Mumbai
    "europe_hub":  (50.11,   8.68, "Europe/Berlin"),      # Frankfurt
}

DAILY_VARS = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "precipitation_sum", "windspeed_10m_max",
    "shortwave_radiation_sum",  # solar → solar power generation proxy
]


def fetch_weather_history(location_key: str, days: int = 365) -> pd.DataFrame:
    """Fetch historical daily weather for one location."""
    if location_key not in LOCATIONS:
        return pd.DataFrame()
    lat, lon, tz = LOCATIONS[location_key]
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days)
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "daily": ",".join(DAILY_VARS),
        "timezone": tz,
    }
    try:
        r = requests.get(f"{_ARCHIVE}/archive", params=params, timeout=30)
        r.raise_for_status()
        data = r.json().get("daily", {})
        df = pd.DataFrame(data)
        df["time"] = pd.to_datetime(df["time"]).dt.tz_localize("UTC")
        df.set_index("time", inplace=True)
        df.columns = [f"{location_key}_{c}" for c in df.columns]
        return df
    except Exception as e:
        logger.warning(f"open-meteo {location_key}: {e}")
        return pd.DataFrame()


def compute_weather_features(days: int = 90) -> pd.DataFrame:
    """Compute cross-location weather features for the feature store.

    Returns one row (latest date) with features for all locations:
    - Heating/cooling degree days (HDD/CDD) vs seasonal norms
    - Precipitation anomaly (rolling z-score)
    - Wind/solar generation proxy for energy stocks
    """
    frames = []
    for loc in LOCATIONS:
        df = fetch_weather_history(loc, days=days + 30)
        if df.empty:
            continue
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, axis=1).sort_index().dropna(how="all")
    if combined.empty:
        return pd.DataFrame()

    row: dict = {}
    now = datetime.now(timezone.utc)

    for loc in LOCATIONS:
        temp_col = f"{loc}_temperature_2m_mean"
        precip_col = f"{loc}_precipitation_sum"
        wind_col = f"{loc}_windspeed_10m_max"
        solar_col = f"{loc}_shortwave_radiation_sum"

        if temp_col in combined.columns and len(combined) >= 30:
            s = combined[temp_col].dropna()
            if len(s) >= 30:
                latest = float(s.iloc[-1])
                mean_30 = float(s.iloc[-30:].mean())
                std_30 = float(s.iloc[-30:].std())
                row[f"wx_{loc}_temp_latest"] = latest
                row[f"wx_{loc}_temp_anom"] = (latest - mean_30) / max(std_30, 0.1)
                # Heating degree days (base 65°F ≈ 18.3°C)
                row[f"wx_{loc}_hdd"] = max(0.0, 18.3 - latest)
                row[f"wx_{loc}_cdd"] = max(0.0, latest - 18.3)

        if precip_col in combined.columns and len(combined) >= 30:
            s = combined[precip_col].dropna()
            if len(s) >= 30:
                latest_p = float(s.iloc[-1])
                mean_p = float(s.iloc[-30:].mean())
                std_p = float(s.iloc[-30:].std())
                row[f"wx_{loc}_precip_anom"] = (latest_p - mean_p) / max(std_p, 0.1)
                row[f"wx_{loc}_precip_7d"] = float(s.iloc[-7:].sum())

        if wind_col in combined.columns:
            s = combined[wind_col].dropna()
            if len(s) >= 7:
                row[f"wx_{loc}_wind_7d_mean"] = float(s.iloc[-7:].mean())

        if solar_col in combined.columns:
            s = combined[solar_col].dropna()
            if len(s) >= 7:
                row[f"wx_{loc}_solar_7d_mean"] = float(s.iloc[-7:].mean())

    # Cross-location aggregates
    hdd_cols = [k for k in row if k.endswith("_hdd")]
    cdd_cols = [k for k in row if k.endswith("_cdd")]
    if hdd_cols:
        row["wx_global_hdd_sum"] = sum(row[c] for c in hdd_cols)
    if cdd_cols:
        row["wx_global_cdd_sum"] = sum(row[c] for c in cdd_cols)

    row["asof_ts"] = now
    row["knowledge_ts"] = now
    return pd.DataFrame([row])
