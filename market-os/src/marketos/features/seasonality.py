"""Seasonality & calendar feature family.

Calendar effects are among the most robust anomalies in the literature: turn-of-month
strength (Ariel 1987), the January effect, day-of-week patterns (French 1980), the
options-expiration week, and "sell in May" (Bouman-Jacobsen 2002). These are deterministic
functions of the timestamp — trivially causal (the date of a bar is known at the bar).

We encode each cyclical variable with sin/cos pairs so a model sees the cycle's geometry
rather than an arbitrary integer ordering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_seasonality_features(index: pd.DatetimeIndex, *, knowledge_lag: str = "0min") -> pd.DataFrame:
    """Compute calendar/seasonality features for a DatetimeIndex."""
    idx = pd.DatetimeIndex(index)
    out = pd.DataFrame(index=idx)

    dow = idx.dayofweek          # 0=Mon
    dom = idx.day
    month = idx.month
    doy = idx.dayofyear
    week = idx.isocalendar().week.astype(int).values
    quarter = ((month - 1) // 3) + 1           # 1-4
    moq = ((month - 1) % 3) + 1               # month within quarter: 1, 2, or 3

    # ── Cyclical encodings (geometry, not ordinal) ────────────────────────────
    out["dow_sin"]   = np.sin(2 * np.pi * dow   / 5)
    out["dow_cos"]   = np.cos(2 * np.pi * dow   / 5)
    out["dom_sin"]   = np.sin(2 * np.pi * dom   / 31)
    out["dom_cos"]   = np.cos(2 * np.pi * dom   / 31)
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    out["doy_sin"]   = np.sin(2 * np.pi * doy   / 365)
    out["doy_cos"]   = np.cos(2 * np.pi * doy   / 365)
    out["woy_sin"]   = np.sin(2 * np.pi * week  / 52)
    out["woy_cos"]   = np.cos(2 * np.pi * week  / 52)
    out["moq_sin"]   = np.sin(2 * np.pi * moq   / 3)
    out["moq_cos"]   = np.cos(2 * np.pi * moq   / 3)

    # Quarter-position cyclicals (day within ~91-day quarter window)
    quarter_day = np.where(quarter == 1, doy,
                  np.where(quarter == 2, doy - 90,
                  np.where(quarter == 3, doy - 181,
                                         doy - 273)))
    out["quarter_pos_sin"] = np.sin(2 * np.pi * quarter_day / 91.0)
    out["quarter_pos_cos"] = np.cos(2 * np.pi * quarter_day / 91.0)

    # ── Named calendar anomalies (binary flags) ───────────────────────────────
    out["is_monday"]    = (dow == 0).astype(float)
    out["is_tuesday"]   = (dow == 1).astype(float)
    out["is_wednesday"] = (dow == 2).astype(float)
    out["is_thursday"]  = (dow == 3).astype(float)
    out["is_friday"]    = (dow == 4).astype(float)

    out["is_january"]  = (month == 1).astype(float)
    out["is_december"] = (month == 12).astype(float)

    out["is_q1"] = (quarter == 1).astype(float)
    out["is_q2"] = (quarter == 2).astype(float)
    out["is_q3"] = (quarter == 3).astype(float)
    out["is_q4"] = (quarter == 4).astype(float)

    # Turn of month: last 1 + first 3 trading days approximated by calendar day
    days_in_month = idx.to_series().dt.daysinmonth.values
    out["turn_of_month"]     = ((dom <= 3) | (dom >= days_in_month - 1)).astype(float)
    out["turn_month_start"]  = (dom <= 3).astype(float)
    out["turn_month_end"]    = (dom >= days_in_month - 1).astype(float)

    # Week-of-month bins (approximate)
    out["is_week1_of_month"] = (dom <= 7).astype(float)
    out["is_week2_of_month"] = ((dom > 7)  & (dom <= 14)).astype(float)
    out["is_week3_of_month"] = ((dom > 14) & (dom <= 21)).astype(float)
    out["is_week4_of_month"] = (dom > 21).astype(float)

    # More DOM ranges
    out["is_mid_month"]   = ((dom >= 10) & (dom <= 20)).astype(float)
    out["is_start_month"] = (dom <= 5).astype(float)
    out["is_end_month"]   = (dom >= 25).astype(float)

    # "Sell in May" — weak May–Oct window
    out["sell_in_may"] = month.isin([5, 6, 7, 8, 9, 10]).astype(float)

    # Halloween effect (Nov–Apr = historically bullish half-year)
    out["halloween_effect"] = month.isin([11, 12, 1, 2, 3, 4]).astype(float)

    # Quarter-end (window dressing)
    out["is_quarter_end_month"] = month.isin([3, 6, 9, 12]).astype(float)
    out["is_quarter_start_month"] = month.isin([1, 4, 7, 10]).astype(float)

    # Options expiration week (3rd Friday of month → days 15–21)
    out["opex_week"] = ((dom >= 15) & (dom <= 21)).astype(float)
    # Expiry day: 3rd Friday
    out["opex_day"]  = ((dom >= 15) & (dom <= 21) & (dow == 4)).astype(float)
    # Week before OPEX (days 8-14)
    out["pre_opex_week"] = ((dom >= 8) & (dom <= 14)).astype(float)

    # NFP (Non-Farm Payrolls) — first Friday of month
    out["nfp_day"]  = ((dom <= 7) & (dow == 4)).astype(float)
    out["nfp_week"] = (dom <= 7).astype(float)

    # FOMC meeting week proxy: ~3rd week of every other month (Jan/Mar/May/Jul/Sep/Nov)
    fomc_months    = month.isin([1, 3, 5, 7, 9, 11])
    fomc_week_dom  = (dom >= 15) & (dom <= 21)
    out["fomc_week_proxy"] = (fomc_months & fomc_week_dom).astype(float)

    # Santa Claus / year-end rally
    out["santa_rally"] = ((month == 12) & (dom >= 24)).astype(float)
    out["year_end_rally"] = ((month == 12) & (dom >= 20)).astype(float)

    # January effects
    out["jan_effect_window"] = ((month == 1) & (dom <= 15)).astype(float)

    # Quarter-end windows (last 10 trading days of each quarter, approximated)
    out["q1_end"] = ((month == 3)  & (dom >= 20)).astype(float)
    out["q2_end"] = ((month == 6)  & (dom >= 20)).astype(float)
    out["q3_end"] = ((month == 9)  & (dom >= 20)).astype(float)
    out["q4_end"] = ((month == 12) & (dom >= 20)).astype(float)

    # Seasonal regimes
    out["is_summer"] = month.isin([6, 7, 8]).astype(float)
    out["is_winter"] = month.isin([12, 1, 2]).astype(float)
    out["is_spring"] = month.isin([3, 4, 5]).astype(float)
    out["is_fall"]   = month.isin([9, 10, 11]).astype(float)
    out["spring_rally"] = month.isin([3, 4]).astype(float)

    # ── Position metrics ───────────────────────────────────────────────────────
    out["month_position"]  = dom / days_in_month          # 0-1, where in month
    out["week_of_year"]    = week / 52.0                  # 0-1 normalized
    out["year_position"]   = doy / 365.0                  # 0-1, where in year
    out["quarter_position"] = np.clip(quarter_day / 91.0, 0, 1)  # 0-1, where in quarter
    out["month_of_quarter"] = moq / 3.0                   # 0.33 / 0.67 / 1.0

    # Half-year
    out["is_first_half"]  = (month <= 6).astype(float)
    out["is_second_half"] = (month > 6).astype(float)
    out["quarter_code"]   = quarter.astype(float) / 4.0   # 0.25, 0.5, 0.75, 1.0

    out["asof_ts"]     = idx
    out["knowledge_ts"] = idx + pd.Timedelta(knowledge_lag)
    return out
