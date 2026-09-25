"""Historical FX market-data loading and caching."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Final

import pandas as pd
import yfinance as yf


class FXDataLoader:
    """Download, clean, and cache daily foreign-exchange market data.

    The loader stores one CSV per ticker under ``data/cache``.  A cache written
    on the current trading day (or on the most recent Friday when called over
    a weekend) is reused; otherwise the full requested history is downloaded
    again so that the backtest has the latest available observations.

    Attributes:
        start_date: Inclusive first date requested from Yahoo Finance.
        end_date: Inclusive logical end date for the returned data.
        cache_dir: Directory in which ticker CSV files are stored.
    """

    DEFAULT_TICKERS: Final[tuple[str, ...]] = (
        "EURUSD=X",
        "GBPUSD=X",
        "USDJPY=X",
    )
    _PRICE_COLUMNS: Final[tuple[str, ...]] = (
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
    )
    _OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    )

    def __init__(
        self,
        start_date: date = date(2014, 1, 1),
        end_date: date | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        """Initialize the loader.

        Args:
            start_date: Inclusive beginning of the historical data window.
            end_date: Inclusive end of the data window. Defaults to today.
            cache_dir: Optional override for the CSV cache directory.
        """
        self.start_date = start_date
        self.end_date = end_date or date.today()
        self.cache_dir = Path(cache_dir) if cache_dir else Path(__file__).parent / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_data(self, ticker: str) -> pd.DataFrame:
        """Return cleaned daily OHLCV data for ``ticker``.

        Args:
            ticker: Yahoo Finance symbol, such as ``"EURUSD=X"``.

        Returns:
            A date-indexed DataFrame with cleaned OHLCV columns.

        Raises:
            ValueError: If ``ticker`` is empty or Yahoo Finance returns no data.
            RuntimeError: If the downloaded data cannot be cleaned.
        """
        if not ticker or not ticker.strip():
            raise ValueError("ticker must be a non-empty Yahoo Finance symbol")

        ticker = ticker.strip().upper()
        cache_path = self.cache_dir / f"{self._cache_name(ticker)}.csv"
        if cache_path.exists() and self._cache_is_current(cache_path):
            return self._read_cache(cache_path)

        # yfinance treats `end` as exclusive, so request the day after the
        # logical end date to ensure today's bar is included when available.
        download_end = self.end_date + timedelta(days=1)
        downloaded = yf.download(
            ticker,
            start=self.start_date.isoformat(),
            end=download_end.isoformat(),
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        cleaned = self._clean(downloaded)
        if cleaned.empty:
            raise ValueError(f"No market data returned for ticker {ticker!r}")

        cleaned.to_csv(cache_path, index_label="Date")
        return cleaned

    def _clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize yfinance output and apply the loader's cleaning rules."""
        if data.empty:
            return pd.DataFrame(columns=self._OUTPUT_COLUMNS)

        frame = data.copy()
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = [
                next((str(part) for part in column if str(part) in self._OUTPUT_COLUMNS), str(column[0]))
                for column in frame.columns
            ]
        frame = frame.loc[:, ~frame.columns.duplicated()]

        missing_columns = [column for column in self._PRICE_COLUMNS if column not in frame]
        if missing_columns:
            raise RuntimeError(f"Downloaded data is missing price columns: {missing_columns}")
        if "Volume" not in frame:
            frame["Volume"] = pd.NA

        frame.index = pd.to_datetime(frame.index, errors="coerce").tz_localize(None)
        frame = frame[~frame.index.isna()].sort_index()
        frame = frame.loc[
            (frame.index.date >= self.start_date)
            & (frame.index.date <= self.end_date)
        ]
        frame = frame[list(self._OUTPUT_COLUMNS)].apply(pd.to_numeric, errors="coerce")

        # Forward-fill gaps created by holidays or incomplete provider rows.
        frame = frame.ffill()
        frame = frame.dropna(subset=list(self._PRICE_COLUMNS))

        # FX feeds commonly publish no meaningful volume and report zero.
        # When a feed does provide positive volume, remove missing/zero rows.
        volume = frame["Volume"]
        if volume.notna().any() and (volume > 0).any():
            frame = frame.loc[volume.gt(0)]
        return frame

    def _read_cache(self, cache_path: Path) -> pd.DataFrame:
        """Read and validate a cached CSV using the same cleaning rules."""
        cached = pd.read_csv(cache_path, index_col="Date", parse_dates=["Date"])
        return self._clean(cached)

    def _cache_is_current(self, cache_path: Path) -> bool:
        """Return whether the cache was written on the latest possible session."""
        modified = datetime.fromtimestamp(cache_path.stat().st_mtime).date()
        latest_session = self.end_date
        while latest_session.weekday() >= 5:
            latest_session -= timedelta(days=1)
        return modified >= latest_session

    @staticmethod
    def _cache_name(ticker: str) -> str:
        """Return a filesystem-safe cache filename stem."""
        return ticker.replace("=", "_").replace("/", "_")

