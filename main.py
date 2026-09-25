"""Run the end-to-end EUR/USD backtesting demonstration."""

from __future__ import annotations

from datetime import date

import pandas as pd

from ai.explainer import TradeExplainer
from data.data_loader import FXDataLoader
from engine.backtest import BacktestEngine, BacktestResult
from engine.metrics import PerformanceEvaluator
from engine.signals import MovingAverageCrossover, RSIStrategy, SignalStrategy


TICKER = "EURUSD=X"


def run_strategy(
    data: pd.DataFrame,
    strategy: SignalStrategy,
    engine: BacktestEngine,
) -> tuple[pd.DataFrame, BacktestResult]:
    """Generate signals and execute one strategy.

    Args:
        data: Cleaned OHLCV market data.
        strategy: Technical strategy to apply.
        engine: Configured execution engine.

    Returns:
        The indicator-enriched strategy data and its backtest result.
    """
    signal_data = strategy.generate(data)
    return signal_data, engine.run(signal_data)


def print_tear_sheets(
    ma_result: BacktestResult,
    rsi_result: BacktestResult,
) -> None:
    """Print strategy and buy-and-hold metrics in a combined table."""
    ma_sheet = PerformanceEvaluator(ma_result).generate_tear_sheet()
    rsi_sheet = PerformanceEvaluator(rsi_result).generate_tear_sheet()
    combined = pd.DataFrame(
        {
            "MA Crossover": ma_sheet["Strategy"],
            "RSI Strategy": rsi_sheet["Strategy"],
            "Buy & Hold": ma_sheet["Benchmark"],
        }
    )
    print("\nPerformance tear sheets")
    print(combined.to_string(float_format=lambda value: f"{value:,.4f}"))


def print_trade_rationales(
    strategy_data: pd.DataFrame,
    result: BacktestResult,
    strategy_name: str,
) -> None:
    """Print AI explanations for the first three mechanical strategy trades."""
    print(f"\nTrade rationales: {strategy_name}")
    if result.trade_log.empty:
        print("No trades were generated.")
        return

    explainer = TradeExplainer()
    for trade_number, (_, trade) in enumerate(result.trade_log.head(3).iterrows(), 1):
        entry_date = pd.Timestamp(trade["entry_date"])
        recent_data = strategy_data.loc[strategy_data.index <= entry_date].tail(10)
        rationale = explainer.generate_rationale(
            trade_record=trade.to_dict(),
            recent_data=recent_data,
            strategy_name=strategy_name,
            strategy_params={"fast_window": 20, "slow_window": 50},
        )
        print(
            f"\nTrade {trade_number}: {trade['direction']} | "
            f"Entry {entry_date.date()} @ {float(trade['entry_price']):.6f}"
        )
        print(f"  {rationale}")


def main() -> None:
    """Load EUR/USD data, run both strategies, and print research outputs."""
    print(f"Loading {TICKER} daily data from 2015-01-01...")
    loader = FXDataLoader(start_date=date(2015, 1, 1))
    market_data = loader.fetch_data(TICKER)
    print(f"Loaded {len(market_data):,} observations through {market_data.index[-1].date()}.")

    engine = BacktestEngine(
        initial_capital=100_000.0,
        commission_bps=3.0,
        slippage_bps=1.0,
    )
    ma_data, ma_result = run_strategy(
        market_data,
        MovingAverageCrossover(fast_window=20, slow_window=50, allow_short=True),
        engine,
    )
    rsi_data, rsi_result = run_strategy(
        market_data,
        RSIStrategy(window=14, oversold=30, overbought=70, allow_short=True),
        engine,
    )

    print_tear_sheets(ma_result, rsi_result)
    print_trade_rationales(ma_data, ma_result, "MovingAverageCrossover")


if __name__ == "__main__":
    main()
