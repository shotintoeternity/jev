"""Synthetic daily-P&L scenarios: one true cause and two planted traps each, with an answer key.

Tables (SQLite):
  positions(date, desk, instrument_id, currency, quantity, trade_id)
  prices(date, instrument_id, price_local, last_update)     -- last_update < date means the price was carried over
  fx_rates(date, currency, rate_to_usd, as_of)               -- as_of < date means the rate was carried over
  pnl_daily(date, desk, pnl_local, pnl_usd)                  -- change in market value; pnl_local ignores FX entirely

The question is always: why was total P&L on 2026-09-15 sharply negative?
"""

import random
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

DATES = ["2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11", "2026-09-14", "2026-09-15", "2026-09-16"]
DAY = "2026-09-15"
INSTRUMENTS = [  # id, desk, currency, quantity, start price
    ("DE10Y", "rates", "EUR", 150_000, 98.0),
    ("FR10Y", "rates", "EUR", 100_000, 97.0),
    ("UST10Y", "rates", "USD", 80_000, 95.0),
    ("CORP_A", "credit", "USD", -20_000, 101.0),
    ("CORP_B", "credit", "EUR", 60_000, 99.0),
    ("CORP_C", "credit", "USD", 50_000, 102.0),
]
QUESTION = (
    "The fund's total daily P&L on 2026-09-15 was sharply negative. Find out why. "
    "The database has tables positions, prices, fx_rates and pnl_daily."
)


@dataclass
class Scenario:
    name: str
    cause: str
    traps: list[str]
    tweak: callable = field(repr=False)


def _base(seed: int):
    rng = random.Random(seed)
    prices = {}
    for iid, *_, p0 in INSTRUMENTS:
        p = p0
        for d in DATES:
            prices[(d, iid)] = [round(p, 3), d]
            p *= 1 + rng.uniform(-0.002, 0.002)
    fx = {}
    r = 1.10
    for d in DATES:
        fx[(d, "EUR")] = [round(r, 4), d]
        fx[(d, "USD")] = [1.0, d]
        r *= 1 + rng.uniform(-0.001, 0.001)
    positions = [[d, desk, iid, ccy, q, f"T-{i + 100}"] for d in DATES for i, (iid, desk, ccy, q, _) in enumerate(INSTRUMENTS)]
    return prices, fx, positions


def _eur_fx_drop(prices, fx, positions):
    for d in ("2026-09-15", "2026-09-16"):
        fx[(d, "EUR")][0] = round(fx[(d, "EUR")][0] * 0.973, 4)  # about 1.10 -> 1.07


def _dup_row_every_day(prices, fx, positions):
    for d in DATES:
        positions.append([d, "credit", "CORP_C", "USD", 50_000, "T-105"])  # same trade twice, every day: no effect on daily P&L


def _stale_de10y(prices, fx, positions):
    # The 09-14 price was carried over from 09-11; the real fall shows up all at once on 09-15.
    prices[("2026-09-14", "DE10Y")] = [prices[("2026-09-11", "DE10Y")][0], "2026-09-11"]
    for d in ("2026-09-15", "2026-09-16"):
        prices[(d, "DE10Y")][0] = round(prices[(d, "DE10Y")][0] * 0.972, 3)


def _small_eur_fx(prices, fx, positions):
    for d in ("2026-09-15", "2026-09-16"):
        fx[(d, "EUR")][0] = round(fx[(d, "EUR")][0] * 0.997, 4)


def _dup_short_on_day(prices, fx, positions):
    positions.append(["2026-09-15", "credit", "CORP_A", "USD", -20_000, "T-103"])  # short booked twice on 09-15 only


def _stale_fx_row(prices, fx, positions):
    fx[("2026-09-15", "EUR")] = [fx[("2026-09-14", "EUR")][0], "2026-09-14"]


def _corp_b_dip(prices, fx, positions):
    for d in ("2026-09-15", "2026-09-16"):
        prices[(d, "CORP_B")][0] = round(prices[(d, "CORP_B")][0] * 0.994, 3)


SCENARIOS = [
    Scenario(
        "fx",
        cause="EUR/USD fell about 2.7% (from roughly 1.10 to 1.07) on 2026-09-15, cutting the USD value of the EUR-denominated bonds, mostly the rates desk's.",
        traps=[
            "a duplicated CORP_C position row on the credit desk",
            "pnl_local showing only a small change, as if prices barely moved",
        ],
        tweak=lambda *a: (_eur_fx_drop(*a), _dup_row_every_day(*a)),
    ),
    Scenario(
        "stale_price",
        cause="The DE10Y price for 2026-09-14 was stale (carried over from 2026-09-11), so two days of price declines were booked at once on 2026-09-15.",
        traps=[
            "the EUR/USD exchange rate falling on 2026-09-15",
            "pnl_local showing only a small change, as if prices barely moved",
        ],
        tweak=lambda *a: (_stale_de10y(*a), _small_eur_fx(*a)),
    ),
    Scenario(
        "dup_trade",
        cause="The CORP_A short position (trade T-103) was booked twice on 2026-09-15, so the credit desk's market value dropped by about 2 million USD.",
        traps=[
            "a stale EUR/USD rate carried over from 2026-09-14",
            "a fall in the CORP_B bond price",
        ],
        tweak=lambda *a: (_dup_short_on_day(*a), _stale_fx_row(*a), _corp_b_dip(*a)),
    ),
]


def build(s: Scenario, path: Path, seed: int = 5) -> Path:
    prices, fx, positions = _base(seed)
    s.tweak(prices, fx, positions)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    db = sqlite3.connect(path)
    db.executescript(
        "create table positions(date text, desk text, instrument_id text, currency text, quantity real, trade_id text);"
        "create table prices(date text, instrument_id text, price_local real, last_update text);"
        "create table fx_rates(date text, currency text, rate_to_usd real, as_of text);"
        "create table pnl_daily(date text, desk text, pnl_local real, pnl_usd real);"
    )
    db.executemany("insert into positions values (?,?,?,?,?,?)", positions)
    db.executemany("insert into prices values (?,?,?,?)", [(d, i, p, lu) for (d, i), (p, lu) in prices.items()])
    db.executemany("insert into fx_rates values (?,?,?,?)", [(d, c, r, a) for (d, c), (r, a) in fx.items()])
    # P&L = change in market value per desk. pnl_local ignores FX; pnl_usd converts.
    mv = {}
    for d, desk, iid, ccy, q, _ in positions:
        p = prices[(d, iid)][0]
        loc, usd = mv.get((d, desk), (0.0, 0.0))
        mv[(d, desk)] = (loc + q * p, usd + q * p * fx[(d, ccy)][0])
    rows = []
    for a, b in zip(DATES, DATES[1:]):
        for desk in ("rates", "credit"):
            (l0, u0), (l1, u1) = mv[(a, desk)], mv[(b, desk)]
            rows.append((b, desk, round(l1 - l0, 2), round(u1 - u0, 2)))
    db.executemany("insert into pnl_daily values (?,?,?,?)", rows)
    db.commit()
    db.close()
    return path


if __name__ == "__main__":
    for s in SCENARIOS:
        p = build(s, Path(__file__).parent / "dbs" / f"{s.name}.sqlite")
        db = sqlite3.connect(p)
        tot = db.execute("select date, round(sum(pnl_usd)), round(sum(pnl_local)) from pnl_daily group by date order by date").fetchall()
        print(s.name, tot)
