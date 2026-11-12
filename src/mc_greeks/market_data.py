"""Phase 8C: canonical point-in-time schema for index-option quotes, CSV ingestion, quote validation.

The repository ships **no** market data. Licensed data (e.g. OptionMetrics, Cboe DataShop) are
read from local CSV files the user is entitled to; nothing licensed is committed. For tests and
demonstrations, ``synthetic_chain`` builds clearly labelled synthetic quotes (source = "SYNTHETIC");
synthetic data must never be reported as historical results.

One row = one option quote snapshot. Required columns:

    quote_time        timestamp of the snapshot (timezone-aware, or naive = US/Eastern by convention)
    underlying        "SPX" or "XSP"
    expiration        last trading date of the contract (date)
    settlement        "AM" (SPX standard, settled on the opening print) or "PM" (weeklies, XSP)
    option_type       "C" or "P"
    strike            > 0
    bid, ask          >= 0, quoted per index point
    bid_size, ask_size  contracts (may be 0 or missing)
    quote_updated     time of the last quote change (freshness)
    underlying_price  index level at quote_time
    multiplier        index points -> dollars (100 for SPX and XSP)
    exercise_style    "European"
    rate              continuously compounded risk-free rate for the expiry (decimal)

Optional: forward, provided_iv, provided_delta, provided_gamma, provided_vega, source.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .implied_vol import black_price

REQUIRED = ["quote_time", "underlying", "expiration", "settlement", "option_type", "strike", "bid", "ask",
            "bid_size", "ask_size", "quote_updated", "underlying_price", "multiplier", "exercise_style", "rate"]
OPTIONAL = ["forward", "provided_iv", "provided_delta", "provided_gamma", "provided_vega", "source"]
FLAGS = ("missing", "crossed", "zero_ask", "negative_bid", "bad_strike", "expired", "stale", "wide", "below_intrinsic")


def load_quotes_csv(path, tz: str = "America/New_York") -> pd.DataFrame:
    """Read a quote file into the canonical schema (types coerced; missing required columns are an error)."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    for c in ("quote_time", "quote_updated"):
        df[c] = pd.to_datetime(df[c])
        if df[c].dt.tz is None:
            df[c] = df[c].dt.tz_localize(tz)
    df["expiration"] = pd.to_datetime(df["expiration"]).dt.date
    for c in ("strike", "bid", "ask", "bid_size", "ask_size", "underlying_price", "multiplier", "rate"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["option_type"] = df["option_type"].str.upper().str[0]
    if "source" not in df.columns:
        df["source"] = "unspecified"
    return df


def time_to_expiry(quote_time: pd.Series, expiration: pd.Series, settlement: pd.Series) -> np.ndarray:
    """Year fraction (calendar days / 365) from the quote to settlement: 09:30 ET on the expiry date for
    AM-settled contracts, 16:00 ET for PM-settled ones."""
    exp_date = pd.to_datetime(expiration).dt.tz_localize(quote_time.dt.tz)
    hours = np.where(np.asarray(settlement) == "AM", 9.5, 16.0)
    settle = exp_date + pd.to_timedelta(hours, unit="h")
    return ((settle - quote_time).dt.total_seconds() / (365.0 * 86400.0)).to_numpy()


def validate_quotes(df: pd.DataFrame, max_staleness_s: float = 60.0, max_rel_spread: float = 0.5,
                    min_abs_spread_allowed: float = 0.10) -> pd.DataFrame:
    """Add one boolean column per flag and 'valid' (no flag raised). Never drops rows.

    wide: ask - bid > max(min_abs_spread_allowed, max_rel_spread x mid).
    below_intrinsic: ask < discounted intrinsic on the spot (r only; a coarse sanity check, since
    dividends lower the forward; use the inferred forward for exact bounds).
    """
    out = df.copy()
    T = time_to_expiry(out.quote_time, out.expiration, out.settlement)
    mid = 0.5 * (out.bid + out.ask)
    out["flag_missing"] = out[["bid", "ask", "strike", "underlying_price"]].isna().any(axis=1)
    out["flag_crossed"] = out.bid > out.ask
    out["flag_zero_ask"] = out.ask <= 0
    out["flag_negative_bid"] = out.bid < 0
    out["flag_bad_strike"] = ~(out.strike > 0)
    out["flag_expired"] = T <= 0
    out["flag_stale"] = (out.quote_time - out.quote_updated).dt.total_seconds() > max_staleness_s
    out["flag_wide"] = (out.ask - out.bid) > np.maximum(min_abs_spread_allowed, max_rel_spread * mid)
    D = np.exp(-out.rate.to_numpy() * np.maximum(T, 0.0))
    cp = np.where(out.option_type == "C", 1.0, -1.0)
    intrinsic = D * np.maximum(cp * (out.underlying_price.to_numpy() - out.strike.to_numpy()), 0.0)
    out["flag_below_intrinsic"] = out.ask.to_numpy() < intrinsic - 1e-9
    out["valid"] = ~out[[f"flag_{f}" for f in FLAGS]].any(axis=1)
    out["T"] = T
    return out


def select_contract(chain: pd.DataFrame, as_of, target_days: float, option_type: str, moneyness: float = 1.0) -> pd.Series:
    """Point-in-time selection: among valid quotes with quote_time <= as_of (the latest snapshot per
    contract), the expiry closest to target_days, then the strike closest to moneyness x underlying.
    Uses nothing observed after as_of."""
    c = chain[(chain.quote_time <= as_of) & chain.get("valid", True) & (chain.option_type == option_type)]
    if c.empty:
        raise LookupError("no valid quote at or before as_of")
    c = c.sort_values("quote_time").groupby(["expiration", "strike", "option_type"]).tail(1)
    days = c["T"] * 365.0
    c = c[np.abs(days - target_days) == np.abs(days - target_days).min()]
    return c.loc[(c.strike - moneyness * c.underlying_price).abs().idxmin()]


def synthetic_chain(quote_time, underlying_price: float, expirations, strikes, sigma: float = 0.18, rate: float = 0.04,
                    div_yield: float = 0.013, half_spread: float = 0.05, rel_half_spread: float = 0.01,
                    underlying: str = "SPX", settlement: str = "PM") -> pd.DataFrame:
    """SYNTHETIC quotes priced with Black-Scholes-Merton at a flat volatility plus a bid/ask spread.
    For tests and demonstrations only (source = 'SYNTHETIC'); not market data."""
    qt = pd.Timestamp(quote_time)
    if qt.tz is None:
        qt = qt.tz_localize("America/New_York")
    rows = []
    for e in expirations:
        exp = pd.Timestamp(e).date()
        T = time_to_expiry(pd.Series([qt]), pd.Series([exp]), pd.Series([settlement]))[0]
        F = underlying_price * math.exp((rate - div_yield) * T)
        D = math.exp(-rate * T)
        for K in strikes:
            for cp, t in ((1.0, "C"), (-1.0, "P")):
                # an expired contract is worth its intrinsic value (and is flagged 'expired' by validate_quotes)
                mid = float(black_price(F, K, T, D, sigma, cp)) if T > 0 else max(cp * (underlying_price - K), 0.0)
                hs = max(half_spread, rel_half_spread * mid)
                rows.append(dict(quote_time=qt, underlying=underlying, expiration=exp, settlement=settlement, option_type=t,
                                 strike=float(K), bid=max(round(mid - hs, 2), 0.0), ask=round(mid + hs, 2), bid_size=10, ask_size=10,
                                 quote_updated=qt, underlying_price=underlying_price, multiplier=100.0, exercise_style="European",
                                 rate=rate, forward=F, provided_iv=sigma, source="SYNTHETIC"))
    return pd.DataFrame(rows)
