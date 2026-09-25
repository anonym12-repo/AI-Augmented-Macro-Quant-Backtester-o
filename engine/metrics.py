"""Performance analytics for backtest results."""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from .backtest import BacktestResult


class PerformanceEvaluator:
    """Calculate risk and return statistics for a strategy and benchmark.

    The benchmark is a buy-and-hold investment in the asset represented by the
    ``Close`` column.  Daily statistics assume 252 trading sessions per year;
    CAGR uses the elapsed calendar time between the first and last observation.

    Args:
        result: Output produced by :class:`~engine.backtest.BacktestEngine`.
        trading_days: Number of trading days used for annualization.
    """

    _METRIC_NAMES: Final[tuple[str, ...]] = (
        "CAGR",
        "Annualized Volatility",
        "Sharpe Ratio",
        "Maximum Drawdown",
        "Max Drawdown Duration (days)",
    )

    def __init__(self, result: BacktestResult, trading_days: int = 252) -> None:
        """Initialize an evaluator from a completed backtest."""
        if not isinstance(result, BacktestResult):
            raise TypeError("result must be a BacktestResult")
        if trading_days <= 0:
            raise ValueError("trading_days must be positive")
        self.result = result
        self.trading_days = trading_days
        self._curve = self._prepare_curve(result)
        self._benchmark_equity = (
            self._curve["Close"] / self._curve["Close"].iloc[0] * result.initial_capital
        )

    def generate_tear_sheet(self) -> pd.DataFrame:
        """Return strategy and buy-and-hold metrics in side-by-side columns.

        Returns:
            DataFrame indexed by metric name with ``Strategy`` and
            ``Benchmark`` columns. Trade-level metrics are strategy-only and
            have ``NaN`` in the benchmark column.
        """
        strategy_equity = self._curve["equity"]
        benchmark_equity = self._benchmark_equity
        sheet = pd.DataFrame(
            {
                "Strategy": self._equity_metrics(strategy_equity),
                "Benchmark": self._equity_metrics(benchmark_equity),
            }
        )
        trade_metrics = self._trade_metrics()
        for name, value in trade_metrics.items():
            sheet.loc[name, "Strategy"] = value
            sheet.loc[name, "Benchmark"] = np.nan
        return sheet

    def _equity_metrics(self, equity: pd.Series) -> dict[str, float]:
        """Calculate annualized and drawdown metrics for an equity series."""
        returns = equity.pct_change().fillna(0.0)
        return {
            "CAGR": self._cagr(equity),
            "Annualized Volatility": float(returns.std(ddof=1) * np.sqrt(self.trading_days)),
            "Sharpe Ratio": self._sharpe(returns),
            "Maximum Drawdown": float(self._drawdown(equity).min()),
            "Max Drawdown Duration (days)": float(self._drawdown_duration(equity)),
        }

    def _trade_metrics(self) -> dict[str, float]:
        """Calculate win rate, count, and profit factor from the trade ledger."""
        trades = self.result.trade_log
        if "pnl_percent" not in trades.columns:
            raise ValueError("trade_log must contain a 'pnl_percent' column")
        pnl = pd.to_numeric(trades["pnl_percent"], errors="coerce").dropna()
        gross_profit = float(pnl[pnl > 0].sum())
        gross_loss = float(-pnl[pnl < 0].sum())
        profit_factor = np.inf if gross_loss == 0 and gross_profit > 0 else (
            gross_profit / gross_loss if gross_loss else 0.0
        )
        return {
            "Win Rate (%)": float((pnl > 0).mean() * 100.0) if len(pnl) else 0.0,
            "Total Trades": float(len(pnl)),
            "Profit Factor": float(profit_factor),
        }

    @staticmethod
    def _cagr(equity: pd.Series) -> float:
        """Return CAGR using elapsed calendar years."""
        if len(equity) < 2 or equity.iloc[0] <= 0 or equity.iloc[-1] <= 0:
            return np.nan
        elapsed = equity.index[-1] - equity.index[0]
        years = elapsed.total_seconds() / (365.25 * 24 * 60 * 60)
        return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) if years > 0 else np.nan

    def _sharpe(self, returns: pd.Series) -> float:
        """Return the annualized zero-risk-free-rate Sharpe ratio."""
        volatility = returns.std(ddof=1)
        return (
            float(returns.mean() / volatility * np.sqrt(self.trading_days))
            if volatility > 0
            else 0.0
        )

    @staticmethod
    def _drawdown(equity: pd.Series) -> pd.Series:
        """Return drawdown from each observation's running equity high."""
        return equity.div(equity.cummax()).sub(1.0)

    @staticmethod
    def _drawdown_duration(equity: pd.Series) -> int:
        """Return the longest calendar-day interval spent below a prior high."""
        drawdown = PerformanceEvaluator._drawdown(equity)
        highs = drawdown.eq(0)
        high_dates = equity.index[highs]
        if len(high_dates) < 2:
            return int((equity.index[-1] - equity.index[0]).days) if len(equity) else 0
        intervals = high_dates.to_series().diff().dt.days.dropna()
        if not intervals.empty and drawdown.iloc[-1] == 0:
            return int(intervals.max())
        trailing = int((equity.index[-1] - high_dates[-1]).days)
        return max(int(intervals.max()), trailing)

    @staticmethod
    def _prepare_curve(result: BacktestResult) -> pd.DataFrame:
        """Validate and normalize the result's equity curve."""
        required = {"Close", "equity"}
        missing = required.difference(result.equity_curve.columns)
        if missing:
            raise ValueError(f"equity_curve is missing required columns: {sorted(missing)}")
        curve = result.equity_curve[["Close", "equity"]].copy().sort_index()
        curve = curve.apply(pd.to_numeric, errors="coerce").dropna()
        if curve.empty or (curve <= 0).any().any():
            raise ValueError("equity_curve must contain positive numeric Close and equity values")
        if not isinstance(curve.index, pd.DatetimeIndex):
            raise TypeError("equity_curve must use a DatetimeIndex")
        return curve


__all__: Final[tuple[str, ...]] = ("PerformanceEvaluator",)
