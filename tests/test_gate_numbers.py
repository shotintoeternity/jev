from evals.gate.agent import _numbers


def test_dates_and_identifiers_are_not_quantities():
    assert _numbers("Credit pnl on 09-15 and 9/16 for trade T-103 in DE10Y and CORP_A (Q9, P2)") == []


def test_real_quantities_are_kept():
    got = _numbers("fell by -2,042,760 USD, about 2.05M, or 904k, a 2.7% drop from 1.0972")
    assert [round(x, 4) for x in got] == [-2042760.0, 2050000.0, 904000.0, 2.7, 1.0972]


def test_years_are_ignored():
    assert _numbers("in 2026 the rate was 1.07") == [1.07]
