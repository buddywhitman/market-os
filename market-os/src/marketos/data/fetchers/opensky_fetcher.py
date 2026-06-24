"""OpenSky Network fetcher — keyless for aggregated stats, 100 req/day anonymous.

API docs: https://openskynetwork.github.io/opensky-api/rest.html

Signal theory: Flight counts are a high-frequency proxy for:
  - Business activity (corporate jet / commercial aviation)
  - Travel demand (airline stocks: not in our universe directly but correlated)
  - Freight/logistics (supply chain health)
  - Energy demand (jet fuel consumption → crude demand)

We don't track individual flights (privacy) — only country-level and route-level aggregates.
Anonymous API: 100 req/day, limited to last 2hr data.
Registered API: 4000 req/day (free with email registration).

Key metrics:
  - US total flights (last hour) vs seasonal baseline
  - US → China / US → India routes (trade/diplomatic signal)
  - Global flight count breadth (economic expansion indicator)
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_BASE = "https://opensky-network.org/api"
_HEADERS = {"User-Agent": "MarketOS/0.1 research@guaqai.me"}

# ICAO country codes for key economies
COUNTRIES = {
    "us": "US", "cn": "CN", "eu_de": "DE", "in": "IN",
    "jp": "JP", "gb": "GB",
}

# Bounding boxes (min_lon, min_lat, max_lon, max_lat) for dense regions
REGIONS = {
    "us_east": (-80, 35, -70, 45),       # US East Coast (NYC + DC + BOS)
    "us_west": (-125, 32, -115, 48),      # US West Coast (LAX + SFO + SEA)
    "europe":  (0, 46, 20, 55),           # Europe core
    "china":   (100, 20, 125, 42),        # China major corridors
}


def fetch_state_count(region_key: str) -> int:
    """Count aircraft in a bounding box right now."""
    if region_key not in REGIONS:
        return 0
    lon_min, lat_min, lon_max, lat_max = REGIONS[region_key]
    try:
        r = requests.get(
            f"{_BASE}/states/all",
            params={"lamin": lat_min, "lomin": lon_min, "lamax": lat_max, "lomax": lon_max},
            headers=_HEADERS, timeout=20
        )
        r.raise_for_status()
        data = r.json()
        states = data.get("states") or []
        return len(states)
    except Exception as e:
        logger.warning(f"OpenSky {region_key}: {e}")
        return 0


def fetch_departures_count(airport: str, hours_back: int = 1) -> int:
    """Count departures from an airport in the last N hours."""
    end = int(datetime.now(timezone.utc).timestamp())
    begin = end - hours_back * 3600
    try:
        r = requests.get(
            f"{_BASE}/flights/departure",
            params={"airport": airport, "begin": begin, "end": end},
            headers=_HEADERS, timeout=20
        )
        if r.status_code == 404:
            return 0
        r.raise_for_status()
        data = r.json()
        return len(data) if isinstance(data, list) else 0
    except Exception as e:
        logger.warning(f"OpenSky departures {airport}: {e}")
        return 0


def compute_aviation_features() -> pd.DataFrame:
    """Compute aviation activity features as economic proxies.

    Uses bounding box counts (not per-flight data) to respect privacy
    and stay within anonymous rate limits.
    """
    now = datetime.now(timezone.utc)
    row: dict = {"asof_ts": now, "knowledge_ts": now}

    # Bounding box counts — 4 API calls
    total_flights = 0
    for region, bbox in REGIONS.items():
        count = fetch_state_count(region)
        row[f"aviation_{region}_count"] = count
        total_flights += count

    row["aviation_total_tracked"] = total_flights

    # Regional ratios
    us_total = row.get("aviation_us_east_count", 0) + row.get("aviation_us_west_count", 0)
    eu_total = row.get("aviation_europe_count", 0)
    cn_total = row.get("aviation_china_count", 0)

    if total_flights > 0:
        row["aviation_us_share"] = us_total / max(total_flights, 1)
        row["aviation_eu_share"] = eu_total / max(total_flights, 1)
        row["aviation_cn_share"] = cn_total / max(total_flights, 1)
        row["aviation_us_cn_ratio"] = us_total / max(cn_total, 1)

    # Hour of day (UTC) — aviation is cyclical, control for time-of-day
    row["aviation_hour_utc"] = now.hour
    row["aviation_is_peak"] = int(9 <= now.hour <= 20)

    return pd.DataFrame([row])
