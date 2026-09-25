"""Unit tests for vectorized technical signal strategies."""

import pandas as pd

from engine.signals import MovingAverageCrossover, RSIStrategy


def test_moving_average_crossover_fires_on_expected_indices() -> None:
    """Bullish and bearish crossovers occur on the bar where they form."""
    data = pd.DataFrame({"Close": [1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0]})

    result = MovingAverageCrossover(
        fast_window=2, slow_window=3, allow_short=True
    ).generate(data)

    assert result.loc[3, "signal"] == 1
    assert result.loc[5, "signal"] == -1


def test_rsi_is_bounded_and_thresholds_generate_positions() -> None:
    """Sustained losses and gains produce valid oversold and overbought signals."""
    prices = [100.0 - index for index in range(20)] + [
        81.0 + index for index in range(20)
    ]
    result = RSIStrategy(window=5, oversold=30, overbought=70).generate(
        pd.DataFrame({"Close": prices})
    )

    valid_rsi = result["RSI"].dropna()
    assert valid_rsi.between(0, 100).all()
    assert (result.loc[result["RSI"] < 30, "signal"] == 1).all()
    assert (result.loc[result["RSI"] > 70, "signal"] == -1).all()
    assert (result["RSI"] < 30).any()
    assert (result["RSI"] > 70).any()


def test_moving_average_warmup_rows_remain_flat() -> None:
    """Rows before both moving averages exist cannot generate positions."""
    data = pd.DataFrame({"Close": range(1, 10)})
    result = MovingAverageCrossover(fast_window=3, slow_window=5).generate(data)

    assert (result.iloc[:4]["signal"] == 0).all()
    assert result.iloc[4]["SMA_fast"] == 4.0
    assert result.iloc[4]["SMA_slow"] == 3.0


def test_rsi_maintains_position_in_neutral_zone() -> None:
    """An oversold long remains active while RSI moves back into neutral."""
    prices = [100.0] + [99.0 - index for index in range(10)] + [
        90.0,
        91.0,
        92.0,
        93.0,
        94.0,
    ]
    result = RSIStrategy(window=5).generate(pd.DataFrame({"Close": prices}))
    oversold_index = result.index[result["RSI"] < 30][0]
    neutral_indices = result.index[
        result["RSI"].between(30, 70, inclusive="neither")
    ]

    assert len(neutral_indices) > 0
    assert result.loc[oversold_index, "signal"] == 1
    assert (result.loc[neutral_indices, "signal"] == 1).all()


def test_strategies_respect_allow_short_false() -> None:
    """Bearish and overbought conditions become flat when shorts are disabled."""
    ma_data = pd.DataFrame({"Close": [1.0, 2.0, 3.0, 2.0, 1.0]})
    ma_result = MovingAverageCrossover(
        fast_window=2, slow_window=3, allow_short=False
    ).generate(ma_data)

    rsi_prices = [100.0 - index for index in range(10)] + [
        91.0 + index for index in range(10)
    ]
    rsi_result = RSIStrategy(
        window=3, oversold=30, overbought=70, allow_short=False
    ).generate(pd.DataFrame({"Close": rsi_prices}))

    assert (ma_result.loc[ma_result["SMA_fast"] < ma_result["SMA_slow"], "signal"] == 0).all()
    assert (rsi_result.loc[rsi_result["RSI"] > 70, "signal"] == 0).all()
