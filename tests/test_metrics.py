"""Unit tests for performance and trade-level analytics."""

import numpy as np
import pandas as pd
import pytest

from engine.backtest import BacktestResult
from engine.metrics import PerformanceEvaluator


def test_deterministic_equity_metrics() -> None:
    """CAGR, Sharpe, and drawdown match manual calculations."""
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    result = BacktestResult(
        equity_curve=pd.DataFrame(
            {"Close": [100.0, 110.0, 99.0], "equity": [100_000.0, 110_000.0, 99_000.0]},
            index=dates,
        ),
        trade_log=pd.DataFrame(
            {
                "pnl_percent": [10.0, -5.0, 0.0],
                "direction": ["LONG", "SHORT", "LONG"],
            }
        ),
        initial_capital=100_000.0,
        total_trades=3,
    )

    sheet = PerformanceEvaluator(result).generate_tear_sheet()
    returns = pd.Series([0.0, 0.1, -0.1])
    expected_sharpe = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    expected_cagr = (99_000.0 / 100_000.0) ** (365.25 / 2) - 1.0

    assert sheet.loc["Sharpe Ratio", "Strategy"] == pytest.approx(expected_sharpe)
    assert sheet.loc["Maximum Drawdown", "Strategy"] == pytest.approx(-0.10)
    assert sheet.loc["CAGR", "Strategy"] == pytest.approx(expected_cagr)


def test_trade_level_metrics() -> None:
    """Win rate, count, and profit factor use the trade P&L series."""
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    result = BacktestResult(
        equity_curve=pd.DataFrame(
            {"Close": [100.0, 100.0, 100.0], "equity": [100_000.0] * 3},
            index=dates,
        ),
        trade_log=pd.DataFrame({"pnl_percent": [10.0, -5.0, 0.0]}),
        initial_capital=100_000.0,
        total_trades=3,
    )

    sheet = PerformanceEvaluator(result).generate_tear_sheet()

    assert sheet.loc["Win Rate (%)", "Strategy"] == pytest.approx(100 / 3)
    assert sheet.loc["Total Trades", "Strategy"] == 3
    assert sheet.loc["Profit Factor", "Strategy"] == pytest.approx(2.0)
