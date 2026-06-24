"""GitHub API fetcher — free tier: 60 req/hr unauth, 5000/hr with token.

Set env var: GITHUB_TOKEN (optional — Personal Access Token, no scope needed)

Signal theory: GitHub activity is a leading indicator for AI/software companies:
  - Stars momentum → developer mindshare → enterprise adoption funnel
  - Commit frequency → development pace / R&D velocity
  - Issue open/close ratio → product stability / technical debt
  - Fork growth → ecosystem health

Universe map: tickers → primary GitHub orgs
  NVDA  → NVIDIA CUDA/cuDNN repos
  AMD   → ROCm / HIP ecosystem
  MSFT  → microsoft (Windows, Azure, GitHub itself, VS Code)
  PLTR  → palantir
  COIN  → coinbase
  PATH  → UiPath
  RKLB  → rocketlab  (hardware — lower signal)
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone, timedelta

import pandas as pd
import requests

logger = logging.getLogger(__name__)

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
_BASE = "https://api.github.com"

# ticker → (org, [key_repos])
TICKER_REPOS = {
    "NVDA": ("NVIDIA", ["cuda-samples", "apex", "NeMo", "TensorRT-LLM"]),
    "AMD":  ("ROCm", ["ROCm", "HIP", "rocBLAS", "MIOpen"]),
    "MSFT": ("microsoft", ["vscode", "TypeScript", "semantic-kernel", "phi"]),
    "PLTR": ("palantir", ["foundry-platform", "javapoet"]),
    "COIN": ("coinbase", ["coinbase-sdk-python", "coinbase-advanced-py"]),
    "PATH": ("UiPath", ["uipath-automation-suite", "studio"]),
}


def _headers() -> dict:
    h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
         "User-Agent": "MarketOS/0.1"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h


def _get(path: str, params: dict | None = None) -> dict | list | None:
    try:
        r = requests.get(f"{_BASE}{path}", headers=_headers(),
                         params=params, timeout=15)
        if r.status_code == 403:
            logger.warning(f"GitHub rate limited: {path}")
            return None
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"GitHub {path}: {e}")
        return None


def fetch_repo_stats(org: str, repo: str) -> dict:
    """Fetch star count, fork count, open issues, recent commit activity."""
    data = _get(f"/repos/{org}/{repo}")
    if not data or "id" not in data:
        return {}
    result = {
        "stars": int(data.get("stargazers_count", 0)),
        "forks": int(data.get("forks_count", 0)),
        "open_issues": int(data.get("open_issues_count", 0)),
        "watchers": int(data.get("watchers_count", 0)),
        "size_kb": int(data.get("size", 0)),
    }
    # Weekly commit activity (last 52 weeks)
    activity = _get(f"/repos/{org}/{repo}/stats/commit_activity")
    if activity and isinstance(activity, list) and len(activity) >= 4:
        last4 = activity[-4:]
        result["commits_4w"] = sum(w.get("total", 0) for w in last4)
        result["commits_52w"] = sum(w.get("total", 0) for w in activity)
        # Momentum: last 4w vs prior 4w
        prior4 = activity[-8:-4] if len(activity) >= 8 else []
        if prior4:
            prior_count = sum(w.get("total", 0) for w in prior4)
            result["commit_momentum"] = result["commits_4w"] - prior_count
    return result


def compute_github_features(universe: list[str]) -> pd.DataFrame:
    """Aggregate GitHub metrics per ticker.

    Features per ticker:
    - gh_{tick}_stars_total: sum across key repos
    - gh_{tick}_forks_total: sum across key repos
    - gh_{tick}_commits_4w: recent development pace
    - gh_{tick}_commits_52w: annual volume
    - gh_{tick}_commit_mom: 4w vs prior 4w change
    - gh_{tick}_issues_ratio: open_issues / (open_issues + implied_closed)
    - gh_{tick}_ecosystem_size: total stars (mindshare)
    """
    now = datetime.now(timezone.utc)
    row: dict = {"asof_ts": now, "knowledge_ts": now}

    for ticker, (org, repos) in TICKER_REPOS.items():
        if ticker not in universe:
            continue
        agg = {"stars": 0, "forks": 0, "open_issues": 0, "commits_4w": 0,
               "commits_52w": 0, "commit_momentum": 0}
        repo_count = 0
        for repo in repos[:2]:  # cap at 2 repos per ticker to preserve rate limit
            stats = fetch_repo_stats(org, repo)
            if not stats:
                continue
            for k in agg:
                agg[k] += stats.get(k, 0)
            repo_count += 1

        if repo_count == 0:
            continue

        prefix = f"gh_{ticker.lower()}"
        row[f"{prefix}_stars_total"] = agg["stars"]
        row[f"{prefix}_forks_total"] = agg["forks"]
        row[f"{prefix}_open_issues"] = agg["open_issues"]
        row[f"{prefix}_commits_4w"] = agg["commits_4w"]
        row[f"{prefix}_commits_52w"] = agg["commits_52w"]
        row[f"{prefix}_commit_mom"] = agg["commit_momentum"]
        if agg["commits_52w"] > 0:
            row[f"{prefix}_commit_recency"] = agg["commits_4w"] / (agg["commits_52w"] / 13)

    # Cross-ticker AI ecosystem stars momentum (universe-level signal)
    ai_stars = sum(row.get(f"gh_{t.lower()}_stars_total", 0)
                   for t in ["NVDA", "AMD", "MSFT", "PLTR"] if f"gh_{t.lower()}_stars_total" in row)
    if ai_stars > 0:
        row["gh_ai_ecosystem_stars"] = ai_stars

    return pd.DataFrame([row])
