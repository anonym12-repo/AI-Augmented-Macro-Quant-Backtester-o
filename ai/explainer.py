"""Cohere-assisted explanations of mechanical trade entries."""

from __future__ import annotations

import os
from typing import Final

import pandas as pd
import cohere
from dotenv import load_dotenv


class TradeExplainer:
    """Translate indicator conditions into factual, non-predictive explanations.

    The explainer sends only the supplied trade record and recent indicator
    observations to Cohere. It is intentionally limited to explaining why the
    already-triggered mechanical signal occurred; it does not make forecasts
    or provide investment advice.

    Args:
        model: Cohere model used for explanations. Defaults to
            ``command-r-08-2024``.
        api_key: Optional API key override; otherwise ``COHERE_API_KEY`` is
            loaded from the environment or a local ``.env`` file.
        timeout: Request timeout in seconds.
    """

    DEFAULT_MODEL: Final[str] = "command-r-08-2024"
    _SYSTEM_PROMPT: Final[str] = (
        "You are a quantitative research assistant explaining a mechanical "
        "backtest trade. Use only the trade details and tabular observations "
        "provided by the user. Do not predict market direction, fabricate "
        "macroeconomic news or other facts, and do not provide financial advice. "
        "Reference ONLY the exact parameter values provided in the strategy "
        "parameters (for example, if fast_window is 20, refer to it as the "
        "20-period SMA). Do not invent other periods. "
        "Your sole task is to translate the technical indicator conditions "
        "visible on the specific entry date into a clear, factual, plain-English "
        "explanation of why the mechanical system triggered the trade. Mention "
        "only observable values or relationships in the supplied data. Respond "
        "in 2-3 sentences."
    )

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the explainer without requiring credentials at import time."""
        if not model.strip():
            raise ValueError("model must be a non-empty string")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        load_dotenv()
        self.model = model
        self._api_key = api_key or os.getenv("COHERE_API_KEY")
        self._timeout = timeout
        self._client: cohere.ClientV2 | None = None

    def generate_rationale(
        self,
        trade_record: dict[str, object],
        recent_data: pd.DataFrame,
        strategy_name: str,
        strategy_params: dict[str, object] | None = None,
    ) -> str:
        """Generate a concise rationale for one mechanically triggered trade.

        Args:
            trade_record: Mapping containing entry date, direction, and entry
                price, plus any other available trade fields.
            recent_data: Recent OHLCV and indicator rows leading up to entry.
            strategy_name: Name of the strategy that generated the signal.
            strategy_params: Exact parameters used by the strategy.

        Returns:
            A 2-3 sentence explanation, or a factual fallback string when
            credentials are unavailable or the API request fails.
        """
        self._validate_inputs(trade_record, recent_data, strategy_name)
        entry_date = trade_record.get("entry_date")
        direction = str(trade_record.get("direction", "UNKNOWN")).upper()
        entry_price = trade_record.get("entry_price")
        table = self._format_data(recent_data)
        formatted_params = ", ".join(
            f"{name}={value}" for name, value in (strategy_params or {}).items()
        ) or "None provided"
        user_prompt = (
            f"Strategy: {strategy_name.strip()}\n"
            f"Strategy Parameters: {formatted_params}\n"
            f"Trade entry date: {entry_date}\n"
            f"Direction: {direction}\n"
            f"Entry price: {entry_price}\n\n"
            "Timing: The signal was calculated using the close of the session "
            "preceding the entry date (T-1), and that signal triggered execution "
            "at the open/close of the entry date (T).\n\n"
            "Recent observations (the final row should correspond to, or be "
            "the latest observation before, the entry date):\n"
            f"{table}"
        )

        if not self._api_key:
            return self._fallback(direction, strategy_name)

        try:
            if self._client is None:
                self._client = cohere.ClientV2(api_key=self._api_key)
            messages = [
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            response = self._client.chat(
                model=self.model,
                messages=messages,
                temperature=0,
                max_tokens=150,
            )
            content = response.message.content[0].text
            if not content or not content.strip():
                return self._fallback(direction, strategy_name)
            return content.strip()
        except Exception as e:
            print(f"\n--- COHERE API ERROR: {e} ---\n")
            return self._fallback(direction, strategy_name)

    @staticmethod
    def _format_data(data: pd.DataFrame) -> str:
        """Serialize recent observations as a bounded Markdown table."""
        table = data.copy()
        table.index.name = table.index.name or "Date"
        table = table.reset_index()
        table = table.tail(20)
        table = table.map(
            lambda value: f"{value:.6g}" if isinstance(value, float) else str(value)
        )
        headers = [str(column) for column in table.columns]
        separator = ["---"] * len(headers)
        rows = [headers, separator, *table.astype(str).values.tolist()]
        return "\n".join("| " + " | ".join(row) + " |" for row in rows)

    @staticmethod
    def _validate_inputs(
        trade_record: dict[str, object],
        recent_data: pd.DataFrame,
        strategy_name: str,
    ) -> None:
        """Validate the minimum factual context required by the prompt."""
        required = {"entry_date", "direction", "entry_price"}
        missing = required.difference(trade_record)
        if missing:
            raise ValueError(f"trade_record is missing required fields: {sorted(missing)}")
        if not isinstance(recent_data, pd.DataFrame) or recent_data.empty:
            raise ValueError("recent_data must be a non-empty pandas DataFrame")
        if not isinstance(strategy_name, str) or not strategy_name.strip():
            raise ValueError("strategy_name must be a non-empty string")

    @staticmethod
    def _fallback(direction: str, strategy_name: str) -> str:
        """Return a transparent non-predictive explanation when unavailable."""
        return (
            f"The {strategy_name.strip()} generated a {direction} signal from "
            "the indicator conditions in the supplied entry data. "
            "No additional market prediction or fundamental explanation was generated."
        )


__all__: Final[tuple[str, ...]] = ("TradeExplainer",)
