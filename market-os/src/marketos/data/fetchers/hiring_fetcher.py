"""Hiring-trends fetcher — open requisitions from public ATS APIs (no auth required).

Signal theory: a company's open job requisitions are a high-frequency, forward-looking
read on management's *expectation* of demand — published weeks before it shows in revenue.
The literature (LinkUp / Revelio job-postings alpha) finds:
  - Sales & marketing req growth   → leads REVENUE  by ~1-2 quarters (demand pull)
  - R&D / engineering req growth    → leads PRODUCT cycles & capex (longer lag)
  - Sharp req *declines* / freezes   → leading recession / guidance-cut signal
So the tradeable signal is the *change* and the *function mix*, not the raw headcount.

Data sources — all free, no key, all expose per-posting timestamps or recency buckets:
  Greenhouse  GET  boards-api.greenhouse.io/v1/boards/{token}/jobs   (updated_at, first_published ISO)
  Lever       GET  api.lever.co/v0/postings/{token}?mode=json        (createdAt epoch-ms, categories.team)
  Workday     POST {tenant}.{dc}.myworkdayjobs.com/wday/cxs/{tenant}/{site}/jobs  (postedOn relative)
  Ashby       GET  api.ashbyhq.com/posting-api/job-board/{token}     (publishedAt, departmentName)

Registry below is seeded with LIVE-VERIFIED tokens (probed 2026-06-22). Adding a company
is one row. Workday site-slugs are embedded in each careers page's HTML config and can't be
guessed blindly — TODO: a slug-discovery scraper to unlock AMD/RTX/LMT/GEV/VST/CEG/ETN/MSTR.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone, timedelta

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_H = {"User-Agent": "MarketOS/0.1", "Accept": "application/json"}
_TIMEOUT = 15

# ── Function classification (team/title → coarse function) ───────────────────
# Order matters: first match wins. Sales/eng are the highest-signal buckets.
_FUNCTION_PATTERNS = [
    ("sales",    re.compile(r"sales|account exec|business develop|revenue|go.to.market|gtm|partnerships?", re.I)),
    ("eng",      re.compile(r"engineer|software|developer|sre|devops|infrastructure|platform|backend|frontend|ml|machine learning", re.I)),
    ("research", re.compile(r"research|scientist|r&d|applied science", re.I)),
    ("ops",      re.compile(r"operations|supply chain|manufactur|logistics|production|facilit", re.I)),
    ("gtm_mktg", re.compile(r"marketing|growth|demand gen|brand|content", re.I)),
    ("ga",       re.compile(r"finance|legal|hr|people|recruit|administr|accounting", re.I)),
]


def _classify_function(text: str) -> str:
    if not text:
        return "other"
    for name, pat in _FUNCTION_PATTERNS:
        if pat.search(text):
            return name
    return "other"


# ── ATS registry: ticker → adapter config (LIVE-VERIFIED 2026-06-22) ─────────
# provider ∈ {greenhouse, lever, workday, ashby}
HIRING_REGISTRY: dict[str, dict] = {
    "PLTR": {"provider": "lever",      "token": "palantir"},                                   # 238 reqs, precise dates + teams
    "COIN": {"provider": "greenhouse", "token": "coinbase"},                                   # 102 reqs, ISO timestamps
    "RKLB": {"provider": "greenhouse", "token": "rocketlab"},                                  # 332 reqs
    "NVDA": {"provider": "workday",    "tenant": "nvidia",   "dc": "wd5", "site": "NVIDIAExternalCareerSite"},
    "AVGO": {"provider": "workday",    "tenant": "broadcom", "dc": "wd1", "site": "External_Career"},
    "NOC":  {"provider": "workday",    "tenant": "ngc",      "dc": "wd1", "site": "Northrop_Grumman_External_Site"},
}

# Theme grouping for sector-level hiring velocity (broadcast features)
_THEME_TICKERS = {
    "ai":      ["NVDA", "PLTR", "AVGO"],
    "space":   ["RKLB"],
    "crypto":  ["COIN"],
    "defense": ["NOC"],
}


# ── Normalized posting record ─────────────────────────────────────────────────
def _posting(title: str, team: str | None, posted_dt: datetime | None,
             rel_days: float | None) -> dict:
    """One normalized job posting. posted_dt = precise UTC datetime if known;
    rel_days = approximate age in days when only a relative string is available."""
    fn = _classify_function(f"{team or ''} {title or ''}")
    return {"title": title or "", "team": team, "function": fn,
            "posted_dt": posted_dt, "rel_days": rel_days}


# ── Adapters ──────────────────────────────────────────────────────────────────
# Each adapter returns {"postings": [...normalized...], "total": int|None}.
# "total" = the TRUE open-req count (may exceed len(postings) when the source
# paginates, e.g. Workday). len(postings) is the *sample* used for recency/function
# distributions. For Greenhouse/Lever/Ashby the API returns everything → total == len.

def _fetch_greenhouse(token: str) -> dict:
    try:
        r = requests.get(f"https://boards-api.greenhouse.io/v1/boards/{token}/jobs",
                         headers=_H, params={"content": "false"}, timeout=_TIMEOUT)
        r.raise_for_status()
        jobs = r.json().get("jobs", [])
    except Exception as e:
        logger.warning(f"Greenhouse {token}: {e}")
        return {"postings": [], "total": None}
    out = []
    for j in jobs:
        dt = _parse_iso(j.get("first_published") or j.get("updated_at"))
        depts = j.get("departments") or []
        team = depts[0].get("name") if depts and isinstance(depts[0], dict) else None
        out.append(_posting(j.get("title"), team, dt, None))
    return {"postings": out, "total": len(out)}


def _fetch_lever(token: str) -> dict:
    try:
        r = requests.get(f"https://api.lever.co/v0/postings/{token}",
                         headers=_H, params={"mode": "json"}, timeout=_TIMEOUT)
        r.raise_for_status()
        jobs = r.json()
    except Exception as e:
        logger.warning(f"Lever {token}: {e}")
        return {"postings": [], "total": None}
    out = []
    for j in jobs:
        ms = j.get("createdAt")
        dt = (datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
              if isinstance(ms, (int, float)) else None)
        team = (j.get("categories") or {}).get("team")
        out.append(_posting(j.get("text"), team, dt, None))
    return {"postings": out, "total": len(out)}


def _fetch_ashby(token: str) -> dict:
    try:
        r = requests.get(f"https://api.ashbyhq.com/posting-api/job-board/{token}",
                         headers=_H, timeout=_TIMEOUT)
        r.raise_for_status()
        jobs = r.json().get("jobs", [])
    except Exception as e:
        logger.warning(f"Ashby {token}: {e}")
        return {"postings": [], "total": None}
    out = []
    for j in jobs:
        dt = _parse_iso(j.get("publishedAt"))
        out.append(_posting(j.get("title"), j.get("departmentName"), dt, None))
    return {"postings": out, "total": len(out)}


_WD_REL = re.compile(r"(\d+)\+?\s*day", re.I)


def _fetch_workday(tenant: str, dc: str, site: str, max_pages: int = 10) -> dict:
    """Workday CXS: POST paginated (20/page). `total` field gives the TRUE req count;
    postedOn is a relative string ('Posted Today' / 'Posted 30+ Days Ago') → approx age.
    We sample up to max_pages*20 of the freshest postings for recency/function stats."""
    url = f"https://{tenant}.{dc}.myworkdayjobs.com/wday/cxs/{tenant}/{site}/jobs"
    out, true_total = [], None
    for page in range(max_pages):
        try:
            r = requests.post(url, headers=_H,
                              json={"limit": 20, "offset": page * 20, "searchText": ""},
                              timeout=_TIMEOUT)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            if page == 0:
                logger.warning(f"Workday {tenant}/{site}: {e}")
            break
        if true_total is None:
            true_total = data.get("total")
        postings = data.get("jobPostings") or []
        if not postings:
            break
        for j in postings:
            rel = (j.get("postedOn") or "").lower()
            if "today" in rel:
                rel_days = 0.0
            elif "yesterday" in rel:
                rel_days = 1.0
            else:
                m = _WD_REL.search(rel)
                rel_days = float(m.group(1)) if m else None
            out.append(_posting(j.get("title"), None, None, rel_days))
        if len(postings) < 20:
            break
    return {"postings": out, "total": true_total if true_total is not None else len(out)}


_ADAPTERS = {
    "greenhouse": lambda c: _fetch_greenhouse(c["token"]),
    "lever":      lambda c: _fetch_lever(c["token"]),
    "ashby":      lambda c: _fetch_ashby(c["token"]),
    "workday":    lambda c: _fetch_workday(c["tenant"], c["dc"], c["site"]),
}


def _parse_iso(s) -> datetime | None:
    if not s or not isinstance(s, str):
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def fetch_company_jobs(provider: str, **config) -> dict:
    """On-demand lookup for ANY company (mode-b): fetch_company_jobs('greenhouse', token='stripe').
    Returns {"postings": [...normalized...], "total": int} — same shape the pipeline consumes.
    Use this to ad-hoc inspect a careers site for a ticker not yet in HIRING_REGISTRY."""
    fn = _ADAPTERS.get(provider)
    if not fn:
        raise ValueError(f"unknown ATS provider {provider!r}; use one of {list(_ADAPTERS)}")
    return fn(config)


# ── Feature extraction ────────────────────────────────────────────────────────
def _ticker_features(postings: list[dict], total: int | None, now: datetime) -> dict:
    """Per-company hiring features. `total` = true open-req count (headline level);
    `postings` = the (possibly sampled) freshest records used for recency/function stats."""
    n = len(postings)
    if n == 0 and not total:
        return {}
    f: dict = {"req_count": int(total) if total else n,
               "req_sampled": n}  # sample size behind the distribution stats below

    # Recency — precise where dates exist, else relative-day buckets (Workday)
    ages = []
    for p in postings:
        if p["posted_dt"] is not None:
            ages.append((now - p["posted_dt"]).total_seconds() / 86400.0)
        elif p["rel_days"] is not None:
            ages.append(p["rel_days"])
    if ages:
        ages_s = pd.Series(ages, dtype=float)
        f["median_age_days"] = float(ages_s.median())
        f["recent_30d_frac"] = float((ages_s <= 30).mean())   # hiring intensity
        f["recent_7d_frac"] = float((ages_s <= 7).mean())     # acceleration proxy
        f["stale_90d_frac"] = float((ages_s >= 90).mean())    # backfill / freeze proxy

    # Function mix — sales = demand-side lead, eng/research = supply-side lead
    funcs = pd.Series([p["function"] for p in postings])
    vc = funcs.value_counts(normalize=True)
    for fn_name in ["sales", "eng", "research", "ops", "gtm_mktg"]:
        if fn_name in vc.index:
            f[f"{fn_name}_frac"] = float(vc[fn_name])
    # Demand/supply hiring tilt: (sales+mktg) − (eng+research)
    demand = float(vc.get("sales", 0) + vc.get("gtm_mktg", 0))
    supply = float(vc.get("eng", 0) + vc.get("research", 0))
    f["demand_supply_tilt"] = demand - supply

    return f


def compute_hiring_features(universe: list[str], prior: dict | None = None) -> pd.DataFrame:
    """One broadcast row of hiring features for every registry ticker in `universe`.

    Per ticker → `hiring_{tick}_*`: req_count, recency, function mix.
    Momentum (vs `prior` snapshot) → `hiring_{tick}_req_mom`, `_req_mom_pct`.
    Theme aggregates → `hiring_{theme}_req_total`, `_req_mom_pct`.

    `prior` = the last stored `_hiring`/`hiring` features dict (pass store.get_latest_family).
    """
    now = datetime.now(timezone.utc)
    row: dict = {"asof_ts": now, "knowledge_ts": now}
    prior = prior or {}

    per_ticker_count: dict[str, int] = {}
    for ticker, cfg in HIRING_REGISTRY.items():
        if ticker not in universe:
            continue
        try:
            res = _ADAPTERS[cfg["provider"]](cfg)
        except Exception as e:
            logger.warning(f"hiring {ticker}: {e}")
            continue
        feats = _ticker_features(res["postings"], res.get("total"), now)
        if not feats:
            continue
        prefix = f"hiring_{ticker.lower()}"
        for k, v in feats.items():
            row[f"{prefix}_{k}"] = v
        per_ticker_count[ticker] = feats["req_count"]

        # Momentum vs prior snapshot — the actual tradeable signal
        prev = prior.get(f"{prefix}_req_count")
        if isinstance(prev, (int, float)) and prev > 0:
            row[f"{prefix}_req_mom"] = feats["req_count"] - prev
            row[f"{prefix}_req_mom_pct"] = (feats["req_count"] - prev) / prev

    # ── Theme-level hiring velocity (sector demand proxy) ────────────────────
    for theme, tickers in _THEME_TICKERS.items():
        total = sum(per_ticker_count.get(t, 0) for t in tickers)
        if total == 0:
            continue
        row[f"hiring_{theme}_req_total"] = total
        prev_total = prior.get(f"hiring_{theme}_req_total")
        if isinstance(prev_total, (int, float)) and prev_total > 0:
            row[f"hiring_{theme}_req_mom_pct"] = (total - prev_total) / prev_total

    return pd.DataFrame([row])
