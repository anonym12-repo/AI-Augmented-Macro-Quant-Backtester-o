"""Vectorized backtesting engine with explicit next-bar execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Container for the outputs of a backtest run.

    Attributes:
        equity_curve: Time-indexed prices, positions, returns, and equity.
        trade_log: Discrete completed and end-of-test marked trades.
        initial_capital: Starting portfolio value.
        total_trades: Number of rows in ``trade_log``.
    """

    equity_curve: pd.DataFrame
    trade_log: pd.DataFrame
    initial_capital: float
    total_trades: int


class BacktestEngine:
    """Simulate a single-asset long/short strategy after signal generation.

    Signals are interpreted as desired market positions in ``{-1, 0, 1}``.
    Returns are calculated from close-to-close prices, and transaction costs
    are charged whenever the held position changes.

    Attributes:
        initial_capital: Starting portfolio value in dollars.
        commission_bps: Commission charged per unit of turnover.
        slippage_bps: Slippage charged per unit of turnover.
    """

    _TRADE_COLUMNS: Final[tuple[str, ...]] = (
        "entry_date",
        "exit_date",
        "direction",
        "entry_price",
        "exit_price",
        "pnl_percent",
        "net_pnl_dollars",
    )

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission_bps: float = 5.0,
        slippage_bps: float = 1.0,
    ) -> None:
        """Initialize the engine.

        Args:
            initial_capital: Positive starting equity.
            commission_bps: Commission in basis points per unit turnover.
            slippage_bps: Slippage in basis points per unit turnover.

        Raises:
            ValueError: If a monetary amount or cost rate is negative/invalid.
        """
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if commission_bps < 0 or slippage_bps < 0:
            raise ValueError("commission_bps and slippage_bps cannot be negative")
        self.initial_capital = float(initial_capital)
        self.commission_bps = float(commission_bps)
        self.slippage_bps = float(slippage_bps)

    def run(self, data: pd.DataFrame) -> BacktestResult:
        """Run the backtest on a DataFrame containing ``Close`` and ``signal``.

        Args:
            data: Chronologically indexed market data with raw desired signals.

        Returns:
            A ``BacktestResult`` containing the equity curve and trade ledger.

        Raises:
            ValueError: If required columns are absent, prices are invalid, or
                signals contain values outside ``{-1, 0, 1}``.
        """
        self._validate_input(data)
        frame = data.copy().sort_index()
        frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
        frame["signal"] = pd.to_numeric(frame["signal"], errors="coerce")
        frame = frame.dropna(subset=["Close", "signal"])
        if frame.empty:
            raise ValueError("Input DataFrame has no usable rows after cleaning")
        frame["signal"] = frame["signal"].astype("int8")

        # A close-based signal from day T is deliberately shifted here: it can
        # only be acted on during day T+1. This prevents using information from
        # the closing price to claim execution at that same closing price.
        frame["position"] = frame["signal"].shift(1).fillna(0).astype("int8")
        frame["asset_return"] = frame["Close"].pct_change().fillna(0.0)
        frame["strategy_return"] = frame["position"] * frame["asset_return"]
        frame["turnover"] = frame["position"].diff().abs().fillna(frame["position"].abs())
        frame["trade_occurred"] = frame["turnover"].gt(0)
        total_cost_bps = (self.commission_bps + self.slippage_bps) / 10_000
        frame["net_strategy_return"] = (
            frame["strategy_return"] - frame["turnover"] * total_cost_bps
        )
        frame["equity"] = self.initial_capital * (
            1.0 + frame["net_strategy_return"]
        ).cumprod()

        equity_columns = [
            "Close",
            "signal",
            "position",
            "asset_return",
            "strategy_return",
            "turnover",
            "trade_occurred",
            "net_strategy_return",
            "equity",
        ]
        equity_curve = frame[equity_columns].copy()
        trade_log = self._build_trade_log(frame)
        return BacktestResult(
            equity_curve=equity_curve,
            trade_log=trade_log,
            initial_capital=self.initial_capital,
            total_trades=len(trade_log),
        )

    def _build_trade_log(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Build discrete trades from contiguous non-zero positions."""
        records: list[dict[str, object]] = []
        active_direction = 0
        entry_date: object = None
        entry_price = 0.0
        entry_turnover = 0.0
        total_cost_bps = (self.commission_bps + self.slippage_bps) / 10_000

        for timestamp, row in frame.iterrows():
            position = int(row["position"])
            price = float(row["Close"])
            if position == active_direction:
                continue

            if active_direction != 0:
                records.append(
                    self._trade_record(
                        entry_date,
                        timestamp,
                        active_direction,
                        entry_price,
                        price,
                        entry_turnover + abs(active_direction),
                        total_cost_bps,
                        self.initial_capital,
                    )
                )
                entry_date = None

            if position != 0:
                active_direction = position
                entry_date = timestamp
                entry_price = price
                entry_turnover = abs(position - 0)
            else:
                active_direction = 0

        if active_direction != 0 and entry_date is not None:
            last_timestamp = frame.index[-1]
            last_price = float(frame.iloc[-1]["Close"])
            records.append(
                self._trade_record(
                    entry_date,
                    last_timestamp,
                    active_direction,
                    entry_price,
                    last_price,
                    entry_turnover,
                    total_cost_bps,
                    self.initial_capital,
                )
            )

        return pd.DataFrame.from_records(records, columns=self._TRADE_COLUMNS)

    @staticmethod
    def _trade_record(
        entry_date: object,
        exit_date: object,
        direction: int,
        entry_price: float,
        exit_price: float,
        turnover: float,
        total_cost_bps: float,
        initial_capital: float,
    ) -> dict[str, object]:
        """Return one trade record, with P&L measured in price terms."""
        signed_return = direction * (exit_price / entry_price - 1.0)
        net_return = signed_return - turnover * total_cost_bps
        return {
            "entry_date": entry_date,
            "exit_date": exit_date,
            "direction": "LONG" if direction == 1 else "SHORT",
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_percent": signed_return * 100.0,
            "net_pnl_dollars": net_return * initial_capital,
        }

    @staticmethod
    def _validate_input(data: pd.DataFrame) -> None:
        """Validate required columns and raw signal domain."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        required = {"Close", "signal"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"Input DataFrame is missing required columns: {sorted(missing)}")
        close = pd.to_numeric(data["Close"], errors="coerce")
        if close.empty or close.isna().all() or close.le(0).any():
            raise ValueError("'Close' must contain only positive numeric prices")
        signals = pd.to_numeric(data["signal"], errors="coerce")
        if signals.isna().any() or not signals.isin((-1, 0, 1)).all():
            raise ValueError("'signal' must contain only -1, 0, or 1")


__all__: Final[tuple[str, ...]] = ("BacktestEngine", "BacktestResult")
