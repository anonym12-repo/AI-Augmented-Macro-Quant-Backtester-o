"""Unit tests for execution timing and portfolio accounting."""

import pandas as pd
import pytest

from engine.backtest import BacktestEngine


def test_signal_executes_on_next_bar_without_lookahead() -> None:
    """A close signal cannot participate in the same day's price move."""
    data = pd.DataFrame(
        {
            "Close": [100.0, 100.0, 200.0, 200.0],
            "signal": [0, 0, 1, 1],
        }
    )

    result = BacktestEngine(
        initial_capital=100_000.0, commission_bps=0.0, slippage_bps=0.0
    ).run(data)
    curve = result.equity_curve

    assert curve["position"].tolist() == [0, 0, 0, 1]
    assert curve.loc[2, "strategy_return"] == 0.0
    assert curve.loc[2, "equity"] == curve.loc[1, "equity"]


def test_transaction_cost_matches_turnover_exactly() -> None:
    """Turnover is charged at commission plus slippage in basis points."""
    data = pd.DataFrame(
        {
            "Close": [100.0, 100.0, 110.0, 110.0, 110.0],
            "signal": [0, 1, 1, 0, 0],
        }
    )
    commission_bps = 3.0
    slippage_bps = 1.0
    result = BacktestEngine(
        initial_capital=100_000.0,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
    ).run(data)
    curve = result.equity_curve
    total_cost = (commission_bps + slippage_bps) / 10_000

    assert curve.loc[2, "turnover"] == 1.0
    assert curve.loc[2, "net_strategy_return"] == pytest.approx(0.1 - total_cost)
    assert curve.loc[3, "turnover"] == 0.0
    assert curve.loc[3, "net_strategy_return"] == pytest.approx(0.0)
    assert curve.loc[4, "turnover"] == 1.0
    assert curve.loc[4, "net_strategy_return"] == pytest.approx(-total_cost)


def test_flat_position_preserves_capital() -> None:
    """A flat strategy neither gains nor loses from asset price changes."""
    data = pd.DataFrame(
        {
            "Close": [100.0, 95.0, 110.0, 80.0],
            "signal": [0, 0, 0, 0],
        }
    )

    result = BacktestEngine(initial_capital=100_000.0).run(data)

    assert (result.equity_curve["position"] == 0).all()
    assert (result.equity_curve["equity"] == 100_000.0).all()


def test_position_reversal_charges_double_turnover() -> None:
    """A direct long-to-short reversal incurs two units of turnover."""
    data = pd.DataFrame(
        {
            "Close": [100.0, 100.0, 100.0, 100.0, 100.0],
            "signal": [0, 1, -1, -1, -1],
        }
    )
    commission_bps = 3.0
    slippage_bps = 1.0
    result = BacktestEngine(
        initial_capital=100_000.0,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
    ).run(data)
    curve = result.equity_curve
    cost_per_unit = (commission_bps + slippage_bps) / 10_000

    assert curve.loc[3, "position"] == -1
    assert curve.loc[3, "turnover"] == 2.0
    assert curve.loc[3, "net_strategy_return"] == pytest.approx(-2 * cost_per_unit)
