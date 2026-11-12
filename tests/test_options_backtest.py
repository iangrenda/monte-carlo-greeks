"""Phase 8C/8D: Black-76 pricing and implied vol, quote schema and validation, and hand-checked trade ledgers.
All market data here is SYNTHETIC."""
import math

import numpy as np
import pandas as pd
import pytest

from mc_greeks import analytical
from mc_greeks import implied_vol as iv
from mc_greeks import market_data as md
from mc_greeks import options_backtest as ob
from mc_greeks.models import BlackScholesParams


# --- pricing on the forward -------------------------------------------------------------------------

def test_black_price_matches_black_scholes_and_parity():
    S, r, sig, T, K = 100.0, 0.04, 0.22, 0.75, 104.0
    F, D = S * math.exp(r * T), math.exp(-r * T)
    p = BlackScholesParams(S, r, sig, T)
    c = float(iv.black_price(F, K, T, D, sig, 1.0))
    assert c == pytest.approx(analytical.call_price(p, K), rel=1e-12)
    put = float(iv.black_price(F, K, T, D, sig, -1.0))
    assert c - put == pytest.approx(D * (F - K), rel=1e-12)                       # put-call parity


def test_greeks_match_closed_forms_and_finite_differences():
    S, r, q, sig, T, K = 100.0, 0.04, 0.015, 0.25, 0.5, 97.0
    F, D = S * math.exp((r - q) * T), math.exp(-r * T)
    g = iv.black_greeks(S, F, K, T, D, sig, 1.0)
    p = BlackScholesParams(S, r, sig, T)
    g0 = iv.black_greeks(S, S * math.exp(r * T), K, T, D, sig, 1.0)               # q = 0 vs the analytical module
    assert float(g0["delta"]) == pytest.approx(analytical.call_delta(p, K), rel=1e-12)
    assert float(g0["gamma"]) == pytest.approx(analytical.call_gamma(p, K), rel=1e-12)
    assert float(g0["vega"]) == pytest.approx(analytical.call_vega(p, K), rel=1e-12)
    price = lambda s, v: float(iv.black_price(s * math.exp((r - q) * T), K, T, D, v, 1.0))
    h = 1e-3
    assert float(g["delta"]) == pytest.approx((price(S + h, sig) - price(S - h, sig)) / (2 * h), rel=1e-7)
    assert float(g["gamma"]) == pytest.approx((price(S + h, sig) - 2 * price(S, sig) + price(S - h, sig)) / h**2, rel=1e-4)
    assert float(g["vega"]) == pytest.approx((price(S, sig + 1e-6) - price(S, sig - 1e-6)) / 2e-6, rel=1e-6)
    gp = iv.black_greeks(S, F, K, T, D, sig, -1.0)
    assert float(g["delta"] - gp["delta"]) == pytest.approx(D * F / S, rel=1e-12)  # call delta - put delta = d(D F)/dS


@pytest.mark.parametrize("cp", [1.0, -1.0])
@pytest.mark.parametrize("K", [80.0, 100.0, 125.0])
def test_implied_vol_round_trip_and_bounds(cp, K):
    F, T, D, sig = 101.0, 0.3, 0.99, 0.31
    price = float(iv.black_price(F, K, T, D, sig, cp))
    assert iv.implied_vol(price, F, K, T, D, cp) == pytest.approx(sig, abs=1e-9)
    lo, hi = iv.no_arbitrage_bounds(F, K, D, cp)
    assert math.isnan(iv.implied_vol(lo - 0.01, F, K, T, D, cp)) and math.isnan(iv.implied_vol(hi + 0.01, F, K, T, D, cp))


def test_implied_forward_from_put_call_parity():
    ch = md.synthetic_chain("2025-03-03 15:45", 5800.0, ["2025-04-17"], np.arange(5500, 6101, 25))
    c = ch[ch.option_type == "C"].sort_values("strike")
    p = ch[ch.option_type == "P"].sort_values("strike")
    T = md.time_to_expiry(c.quote_time, c.expiration, c.settlement)[0]
    D = math.exp(-0.04 * T)
    F = iv.implied_forward(c.strike, 0.5 * (c.bid + c.ask).to_numpy(), 0.5 * (p.bid + p.ask).to_numpy(), D)
    assert F == pytest.approx(c.forward.iloc[0], abs=0.02)                        # mids carry quote rounding to 0.01


# --- schema and validation ---------------------------------------------------------------------------

def test_time_to_expiry_am_vs_pm():
    qt = pd.Series([pd.Timestamp("2025-01-10 15:00", tz="America/New_York")] * 2)
    exp = pd.Series([pd.Timestamp("2025-01-17").date()] * 2)
    T = md.time_to_expiry(qt, exp, pd.Series(["AM", "PM"]))
    assert (T[1] - T[0]) * 365 * 24 == pytest.approx(6.5)
    assert T[1] * 365 * 24 == pytest.approx(7 * 24 + 1)


def test_validation_flags_each_defect(tmp_path):
    ch = md.synthetic_chain("2025-03-03 15:45", 5800.0, ["2025-03-21", "2025-02-28"], [5700.0, 5800.0, 5900.0])
    v = md.validate_quotes(ch)
    live = v.expiration == pd.Timestamp("2025-03-21").date()
    assert v[live].valid.all() and v[~live].flag_expired.all()
    bad = ch[ch.expiration == pd.Timestamp("2025-03-21").date()].copy().reset_index(drop=True)
    bad.loc[0, ["bid", "ask"]] = [10.0, 9.0]                                       # crossed
    bad.loc[1, "ask"] = 0.0                                                         # zero ask
    bad.loc[2, "quote_updated"] = bad.loc[2, "quote_time"] - pd.Timedelta("10min") # stale
    bad.loc[3, ["bid", "ask"]] = [1.0, 60.0]                                       # wide
    bad.loc[4, "ask"] = 1.0                                                         # deep ITM call below intrinsic
    bad.loc[4, "bid"] = 0.5
    bad.loc[4, "strike"] = 5000.0
    v = md.validate_quotes(bad)
    assert v.loc[0, "flag_crossed"] and v.loc[1, "flag_zero_ask"] and v.loc[2, "flag_stale"] and v.loc[3, "flag_wide"]
    assert v.loc[4, "flag_below_intrinsic"]
    assert v.loc[5:, "valid"].all() and (~v.valid).sum() == 5                      # untouched rows stay valid
    f = tmp_path / "q.csv"
    ch.to_csv(f, index=False)
    back = md.load_quotes_csv(f)
    assert len(back) == len(ch) and set(md.REQUIRED) <= set(back.columns) and (back.source == "SYNTHETIC").all()
    pd.DataFrame({"strike": [1]}).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError):
        md.load_quotes_csv(tmp_path / "bad.csv")


def test_contract_selection_is_point_in_time():
    early = md.synthetic_chain("2025-03-03 10:00", 5800.0, ["2025-04-02", "2025-04-17"], [5750.0, 5800.0, 5850.0])
    late = md.synthetic_chain("2025-03-03 15:00", 6100.0, ["2025-04-02", "2025-04-17"], [5800.0, 6100.0])   # later: big move
    chain = md.validate_quotes(pd.concat([early, late], ignore_index=True))
    pick = md.select_contract(chain, pd.Timestamp("2025-03-03 12:00", tz="America/New_York"), 30, "C")
    assert pick.quote_time == early.quote_time.iloc[0] and pick.strike == 5800.0 and str(pick.expiration) == "2025-04-02"
    with pytest.raises(LookupError):
        md.select_contract(chain, pd.Timestamp("2025-03-03 09:00", tz="America/New_York"), 30, "C")


# --- ledger: hand-checked trade histories ----------------------------------------------------------------

def test_long_call_held_to_cash_settlement():
    led = ob.Ledger()
    c = ob.Costs(option_commission=0.65)
    f = ob.trade_option(led, 0, "C5800", 2, bid=5.00, ask=5.10, multiplier=100, costs=c)
    assert f.price == 5.10 and f.cash == pytest.approx(-2 * 510 - 1.30)              # -1021.30
    cash = ob.settle_option(led, 1, "C5800", "C", 5800.0, 5807.0, 100)
    assert cash == pytest.approx(2 * 700.0) and led.position("C5800") == 0
    assert led.reconcile()["pnl"] == pytest.approx(1400 - 1021.30)                   # 378.70


def test_short_put_expires_worthless_and_round_trip_costs_the_spread():
    led = ob.Ledger()
    c = ob.Costs(option_commission=0.65, option_slippage=0.05)
    ob.trade_option(led, 0, "P5600", -1, bid=3.00, ask=3.20, multiplier=100, costs=c)   # sells at 2.95
    ob.settle_option(led, 1, "P5600", "P", 5600.0, 5650.0, 100)
    assert led.reconcile()["pnl"] == pytest.approx(295.0 - 0.65)
    led2 = ob.Ledger()
    ob.trade_option(led2, 0, "X", 3, bid=10.0, ask=10.4, multiplier=100, costs=c)
    ob.trade_option(led2, 1, "X", -3, bid=10.0, ask=10.4, multiplier=100, costs=c)
    # paid 10.45, received 9.95: spread 0.40 + 2 x slippage 0.05 = 0.50 per contract, plus 6 commissions
    assert led2.reconcile()["pnl"] == pytest.approx(-3 * 0.50 * 100 - 6 * 0.65)
    with pytest.raises(ValueError):
        ob.option_fill_price(5.0, 4.0, 1, c)
    with pytest.raises(ValueError):
        ob.Ledger(fills=[ob.Fill(0, "Y", 1, 1.0, 100, 0.0)]).reconcile()            # open position without a mark


def test_hedge_units_sign_and_size():
    # long 1 SPX call with delta 0.5: +50 dollars per index point; an E-mini future is 50 per point -> sell 1
    assert ob.hedge_units(0.5, 1, 100, 50.0) == pytest.approx(-1.0)
    assert ob.hedge_units(-0.25, -2, 100, 50.0, hedge_ratio=0.9) == pytest.approx(-0.9)


def test_delta_hedged_history_matches_a_hand_calculation():
    """Long 1 call (mult 100), hedged with a future worth 50 per point, three days, then settlement.

    day 0: buy call at ask 20.0 (bid 19.6); delta 0.50 -> target -1 future at 5800
    day 1: delta 0.70 -> target -1.4 -> rounds to -1: no trade (unchanged); future 5810
    day 2: delta 0.90 -> target -2 -> sell 1 more at 5830
    day 3 (expiry): settlement 5850 -> call intrinsic (5850 - 5800) = 50; buy back 2 futures at 5850
    Option: -2000 - 0.65 + 5000 = 2999.35
    Futures: short 1 from 5800 to 5850 = -2500; short 1 from 5830 to 5850 = -1000; costs 4 units x 1.25 = 5
    Total = 2999.35 - 3500 - 5 = -505.65
    """
    days = [dict(time=0, bid=19.6, ask=20.0, delta=0.50, hedge_price=5800.0),
            dict(time=1, bid=24.0, ask=24.4, delta=0.70, hedge_price=5810.0),
            dict(time=2, bid=36.0, ask=36.4, delta=0.90, hedge_price=5830.0),
            dict(time=3, bid=49.8, ask=50.2, delta=1.00, hedge_price=5850.0, settlement_level=5850.0)]
    costs = ob.Costs(option_commission=0.65, hedge_cost_per_unit=1.25)
    led, log = ob.run_delta_hedged(days, dict(instrument="C5800", option_type="C", strike=5800.0, multiplier=100),
                                   1, costs, hedge_unit_multiplier=50.0, hedge_unit_delta=50.0)
    assert [r["hedge_pos"] for r in log] == [-1.0, -1.0, -2.0, 0.0]
    assert log[-1]["option_pos"] == 0 and log[-1]["open_value"] == 0
    assert log[-1]["pnl"] == pytest.approx(-505.65)
    assert led.fees() == pytest.approx(0.65 + 4 * 1.25)
    # mark-to-market on day 1 (mid 24.2): option 2420 - 2000.65 = 419.35; future short 1: -(5810-5800)*50 = -500; fee 1.25
    assert log[1]["pnl"] == pytest.approx(419.35 - 500 - 1.25)
