"""Vectorized technical signal strategies for OHLCV data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Final

import pandas as pd


class SignalStrategy(ABC):
    """Interface implemented by strategies that produce raw market positions."""

    @abstractmethod
    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return ``data`` plus strategy indicators and an integer signal."""

    @staticmethod
    def _validate_close(data: pd.DataFrame) -> pd.Series:
        """Return a numeric close series or raise a descriptive error."""
        if "Close" not in data.columns:
            raise ValueError("Input DataFrame must contain a 'Close' column")
        close = pd.to_numeric(data["Close"], errors="coerce")
        if close.isna().all():
            raise ValueError("'Close' must contain at least one numeric value")
        return close


@dataclass(frozen=True, slots=True)
class MovingAverageCrossover(SignalStrategy):
    """Generate positions from a fast/slow simple moving-average crossover.

    For each observation ``t``, the simple moving average is the arithmetic
    mean of the preceding ``window`` closing prices, including the close at
    ``t``.  Thus, the signal at ``t`` uses no information after that close.

    Attributes:
        fast_window: Lookback period for the fast simple moving average.
        slow_window: Lookback period for the slow simple moving average.
        allow_short: If false, bearish crossovers produce a flat position.
    """

    fast_window: int = 20
    slow_window: int = 50
    allow_short: bool = True

    def __post_init__(self) -> None:
        if self.fast_window <= 0 or self.slow_window <= 0:
            raise ValueError("Moving-average windows must be positive")
        if self.fast_window >= self.slow_window:
            raise ValueError("fast_window must be smaller than slow_window")

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return SMA columns and a vectorized long/short/flat signal.

        Args:
            data: DataFrame containing a ``Close`` column.

        Returns:
            A copy of ``data`` with ``SMA_fast``, ``SMA_slow``, and integer
            ``signal`` columns.  Warm-up rows remain flat until both averages
            are available.
        """
        close = self._validate_close(data)
        result = data.copy()
        result["SMA_fast"] = close.rolling(self.fast_window, min_periods=self.fast_window).mean()
        result["SMA_slow"] = close.rolling(self.slow_window, min_periods=self.slow_window).mean()

        bullish = result["SMA_fast"].gt(result["SMA_slow"])
        bearish = result["SMA_fast"].lt(result["SMA_slow"])
        result["signal"] = bullish.astype("int8")
        if self.allow_short:
            result.loc[bearish, "signal"] = -1
        result.loc[result[["SMA_fast", "SMA_slow"]].isna().any(axis=1), "signal"] = 0
        result["signal"] = result["signal"].astype("int8")
        return result


@dataclass(frozen=True, slots=True)
class RSIStrategy(SignalStrategy):
    """Generate positions from Wilder's exponentially smoothed RSI.

    Wilder's relative strength index is ``100 - 100 / (1 + RS)``, where
    ``RS = average_gain / average_loss``.  Both averages use an exponential
    smoothing factor of ``1 / window`` (equivalent to Wilder's recursive
    moving average).  An oversold reading enters long, and an overbought
    reading enters short or exits to flat.  Between thresholds, the previous
    position is maintained using a vectorized forward-fill.

    Attributes:
        window: RSI lookback period.
        oversold: RSI level below which a long position is requested.
        overbought: RSI level above which a short/flat position is requested.
        allow_short: If false, overbought readings exit to a flat position.
    """

    window: int = 14
    oversold: float = 30.0
    overbought: float = 70.0
    allow_short: bool = True

    def __post_init__(self) -> None:
        if self.window <= 0:
            raise ValueError("RSI window must be positive")
        if not 0 <= self.oversold < self.overbought <= 100:
            raise ValueError("RSI thresholds must satisfy 0 <= oversold < overbought <= 100")

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return an RSI column and the corresponding vectorized position.

        Args:
            data: DataFrame containing a ``Close`` column.

        Returns:
            A copy of ``data`` with ``RSI`` and integer ``signal`` columns.
            The initial RSI warm-up period is flat, and no execution shift is
            applied.
        """
        close = self._validate_close(data)
        result = data.copy()
        delta = close.diff()
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)
        average_gain = gains.ewm(
            alpha=1 / self.window, adjust=False, min_periods=self.window
        ).mean()
        average_loss = losses.ewm(
            alpha=1 / self.window, adjust=False, min_periods=self.window
        ).mean()
        relative_strength = average_gain.div(average_loss)
        rsi = 100 - (100 / (1 + relative_strength))
        rsi = rsi.mask(average_loss.eq(0) & average_gain.gt(0), 100)
        rsi = rsi.mask(average_loss.eq(0) & average_gain.eq(0), 50)
        result["RSI"] = rsi

        entries = pd.Series(pd.NA, index=result.index, dtype="Int8")
        entries.loc[rsi.lt(self.oversold)] = 1
        entries.loc[rsi.gt(self.overbought)] = -1 if self.allow_short else 0
        result["signal"] = entries.ffill().fillna(0).astype("int8")
        return result


__all__: Final[tuple[str, ...]] = (
    "MovingAverageCrossover",
    "RSIStrategy",
    "SignalStrategy",
)
