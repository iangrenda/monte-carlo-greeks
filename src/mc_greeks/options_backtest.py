"""Phase 8D: cash-flow ledger for option positions and hedges, with execution costs and settlement.

Conventions:
* Option prices are in index points; dollars = points x multiplier (100 for SPX and XSP).
* Buys fill at the ask plus slippage, sells at the bid minus slippage (never at the mid).
* Every fill pays a per-contract commission (options) or a per-unit cost (hedge instrument).
* SPX/XSP options are European and cash-settled: at expiry a long position receives
  multiplier x intrinsic value at the settlement level. AM-settled SPX uses the opening
  settlement value (SOQ), PM-settled contracts the closing level; the caller supplies the
  right level.
* The index itself is not tradable. Delta hedges use a stated proxy (e.g. index futures or an
  index ETF) with its own price, unit size, cost and a hedge ratio, so tracking error appears
  in the P&L.
* Reconciliation: the sum of all cash flows plus the mark-to-market of open positions equals
  the total P&L; ``Ledger.reconcile`` checks it.

Research only: nothing here connects to a broker.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Costs:
    option_commission: float = 0.65      # dollars per contract
    option_slippage: float = 0.0         # index points added to buys / subtracted from sells
    hedge_cost_per_unit: float = 0.0     # dollars per hedge unit traded
    hedge_slippage: float = 0.0          # price units of the hedge instrument


@dataclass
class Fill:
    time: object
    instrument: str
    qty: float                           # + buy, - sell (contracts or hedge units)
    price: float                         # executed price (points for options, hedge price for hedges)
    multiplier: float
    fee: float

    @property
    def cash(self) -> float:
        """Cash flow of the fill (negative when buying)."""
        return -self.qty * self.price * self.multiplier - self.fee


@dataclass
class Ledger:
    fills: list = field(default_factory=list)
    settlements: list = field(default_factory=list)      # (time, instrument, qty, value per unit, cash)

    def position(self, instrument: str) -> float:
        q = sum(f.qty for f in self.fills if f.instrument == instrument)
        q -= sum(s[2] for s in self.settlements if s[1] == instrument)
        return q

    def cash(self) -> float:
        return sum(f.cash for f in self.fills) + sum(s[4] for s in self.settlements)

    def fees(self) -> float:
        return sum(f.fee for f in self.fills)

    def reconcile(self, marks: dict | None = None) -> dict:
        """P&L = cash + value of open positions at `marks` ({instrument: (price, multiplier)})."""
        marks = marks or {}
        open_value = 0.0
        for inst in {f.instrument for f in self.fills}:
            q = self.position(inst)
            if abs(q) > 1e-12:
                if inst not in marks:
                    raise ValueError(f"open position in {inst} has no mark")
                p, mult = marks[inst]
                open_value += q * p * mult
        return {"cash": self.cash(), "open_value": open_value, "pnl": self.cash() + open_value, "fees": self.fees()}


def option_fill_price(bid: float, ask: float, side: int, costs: Costs) -> float:
    """Executable price: buy (side +1) at ask + slippage; sell (side -1) at bid - slippage (floored at 0)."""
    if not (np.isfinite(bid) and np.isfinite(ask)) or ask <= 0 or bid > ask:
        raise ValueError(f"not an executable quote: bid={bid}, ask={ask}")
    return ask + costs.option_slippage if side > 0 else max(bid - costs.option_slippage, 0.0)


def trade_option(ledger: Ledger, time, instrument: str, qty: float, bid: float, ask: float, multiplier: float, costs: Costs) -> Fill:
    side = 1 if qty > 0 else -1
    f = Fill(time, instrument, qty, option_fill_price(bid, ask, side, costs), multiplier, abs(qty) * costs.option_commission)
    ledger.fills.append(f)
    return f


def trade_hedge(ledger: Ledger, time, instrument: str, qty: float, price: float, unit_multiplier: float, costs: Costs) -> Fill:
    px = price + (costs.hedge_slippage if qty > 0 else -costs.hedge_slippage)
    f = Fill(time, instrument, qty, px, unit_multiplier, abs(qty) * costs.hedge_cost_per_unit)
    ledger.fills.append(f)
    return f


def settle_option(ledger: Ledger, time, instrument: str, option_type: str, strike: float, settlement_level: float, multiplier: float) -> float:
    """Cash-settle the whole open quantity at intrinsic value; returns the cash received (negative if short)."""
    q = ledger.position(instrument)
    value = max(settlement_level - strike, 0.0) if option_type == "C" else max(strike - settlement_level, 0.0)
    cash = q * value * multiplier
    ledger.settlements.append((time, instrument, q, value, cash))
    return cash


def hedge_units(option_delta_per_contract: float, option_qty: float, option_multiplier: float,
                hedge_unit_delta: float, hedge_ratio: float = 1.0) -> float:
    """Hedge quantity that offsets the option position's index delta.

    option dollar-delta per index point = qty x delta x multiplier; one hedge unit has
    `hedge_unit_delta` dollars per index point (e.g. 50 for an E-mini S&P future, about 0.1 x 100
    shares for SPY). hedge_ratio < 1 or > 1 encodes an imperfect proxy."""
    return -hedge_ratio * option_qty * option_delta_per_contract * option_multiplier / hedge_unit_delta


def run_delta_hedged(days, option, qty: float, costs: Costs, hedge_unit_multiplier: float, hedge_unit_delta: float,
                     hedge_ratio: float = 1.0, rebalance_threshold: float = 0.0, round_lots: bool = True):
    """Open an option position on day 0, delta-hedge at each day's close, settle at expiry.

    days: list of dicts, one per trading day up to and including expiry, each with keys
      time, bid, ask, delta (model delta per unit, w.r.t. the index), hedge_price,
      and on the last day settlement_level.
    option: dict with instrument, option_type, strike, multiplier.
    Hedge trades happen when the target differs from the current hedge by more than
    rebalance_threshold units; quantities are rounded to whole units if round_lots.
    Returns (ledger, log): the log has one row per day with positions and marks."""
    led = Ledger()
    inst, mult = option["instrument"], option["multiplier"]
    trade_option(led, days[0]["time"], inst, qty, days[0]["bid"], days[0]["ask"], mult, costs)
    log = []
    for i, d in enumerate(days):
        last = i == len(days) - 1
        h = led.position("HEDGE")
        if last:
            settle_option(led, d["time"], inst, option["option_type"], option["strike"], d["settlement_level"], mult)
            if abs(h) > 0:
                trade_hedge(led, d["time"], "HEDGE", -h, d["hedge_price"], hedge_unit_multiplier, costs)
        else:
            target = hedge_units(d["delta"], led.position(inst), mult, hedge_unit_delta, hedge_ratio)
            if round_lots:
                target = float(np.round(target))
            if abs(target - h) > rebalance_threshold:
                trade_hedge(led, d["time"], "HEDGE", target - h, d["hedge_price"], hedge_unit_multiplier, costs)
        mid = 0.5 * (d["bid"] + d["ask"])
        marks = {"HEDGE": (d["hedge_price"], hedge_unit_multiplier)}
        if abs(led.position(inst)) > 1e-12:
            marks[inst] = (mid, mult)
        rec = led.reconcile(marks)
        log.append({"time": d["time"], "option_pos": led.position(inst), "hedge_pos": led.position("HEDGE"), **rec})
    return led, log
