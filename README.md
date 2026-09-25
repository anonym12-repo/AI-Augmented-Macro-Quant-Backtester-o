# AI-Augmented Macro Quant Backtester

![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

An event-aware, object-oriented quantitative backtesting framework built from scratch in Python. It executes systematic trading strategies on G10 FX instruments while enforcing institutional-style quality controls — strict look-ahead bias prevention, realistic transaction cost modeling, and full trade-level auditability. On top of the mechanical engine sits a deterministic Generative AI research layer that translates technical trigger conditions into factual, non-predictive market rationales.

## Table of Contents

- [Why This Project Is Useful](#why-this-project-is-useful)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Running the Demo](#running-the-demo)
- [Usage](#usage)
  - [Loading Market Data](#loading-market-data)
  - [Generating Signals](#generating-signals)
  - [Running a Backtest](#running-a-backtest)
  - [Evaluating Performance](#evaluating-performance)
  - [Generating AI Trade Rationales](#generating-ai-trade-rationales)
- [Sample Output](#sample-output)
- [Design Decisions](#design-decisions)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Disclaimer](#disclaimer)
- [License](#license)

## Why This Project Is Useful

Backtesting is only as trustworthy as the assumptions baked into it. Retail-grade, black-box backtesting tools frequently overstate historical performance by allowing signals to "see" prices they could not have traded on, or by ignoring the cost of actually executing a strategy. This framework was built to avoid those pitfalls by favoring a transparent, fully inspectable, vectorized Pandas/NumPy engine over a black box.

- **Strict institutional guardrails.** Look-ahead bias is eliminated by construction: a signal calculated from day `T`'s closing price is only ever allowed to affect the position held on day `T+1`. The shift is applied once, centrally, inside the execution engine — not left to each strategy to remember.
- **Realistic cost modeling.** Every change in position (turnover) is charged a configurable commission (in basis points) plus slippage (in basis points), so the equity curve reflects what a strategy would actually have earned net of trading friction, not just its raw, frictionless signal.
- **Vectorized signal generation.** Textbook technical strategies — Simple Moving Average Crossover and Wilder's RSI — are implemented with pure Pandas vectorization (rolling windows, `ewm`, boolean masks) rather than row-by-row loops, so they scale efficiently to years of daily OHLCV data.
- **Deterministic AI research layer.** A `TradeExplainer` component calls Cohere's Command R model (`temperature=0`) with a tightly scoped system prompt to explain *why* a mechanical signal fired, using only the trade record and the recent indicator values supplied to it. It is explicitly constrained not to forecast markets, fabricate macro news, or give investment advice — it is a translator of mechanical facts, not an oracle.
- **Resilient data pipeline.** Historical FX data is pulled from `yfinance`, cleaned (numeric coercion, forward-filled gaps from holidays/missing rows, tz-normalized dates), cached to disk per ticker, and re-downloaded only when the cache is stale relative to the most recent trading session.
- **Comprehensive, auditable metrics.** Every backtest produces both an equity curve and a discrete trade ledger, from which the framework computes CAGR, annualized volatility, annualized Sharpe ratio, maximum drawdown, maximum drawdown duration, win rate, and profit factor — always alongside a buy-and-hold benchmark on the same instrument.

## Architecture

The codebase is split into four independent layers, each with a single responsibility, connected only through plain `pandas.DataFrame` objects:

```
data/    ──►  engine/signals.py  ──►  engine/backtest.py  ──►  engine/metrics.py
 (FXDataLoader)   (SignalStrategy)     (BacktestEngine)         (PerformanceEvaluator)
                                              │
                                              ▼
                                        ai/explainer.py
                                        (TradeExplainer)
```

1. **`data/data_loader.py` — `FXDataLoader`**
   Downloads and cleans daily OHLCV data for a given ticker (e.g. `EURUSD=X`), forward-fills provider gaps, drops rows where volume is reported but zero, and caches the result to `data/cache/<ticker>.csv`. A cache is reused only if it was written on or after the most recent trading session (weekends roll back to the prior Friday).

2. **`engine/signals.py` — `SignalStrategy` (abstract base), `MovingAverageCrossover`, `RSIStrategy`**
   Each strategy consumes a `Close`-bearing DataFrame and returns it enriched with indicator columns and an integer `signal` column in `{-1, 0, 1}` (short, flat, long). Both strategies are implemented as frozen dataclasses for immutable, hashable configuration, and both remain flat during their indicator warm-up period.

3. **`engine/backtest.py` — `BacktestEngine`, `BacktestResult`**
   Takes a DataFrame with `Close` and `signal` columns and simulates a single-asset long/short strategy. This is the only place the look-ahead-safe execution shift (`signal.shift(1)`) and the transaction-cost model are applied, so every strategy automatically inherits the same institutional guardrails. Produces a full `equity_curve` and a discrete `trade_log` of entries/exits.

4. **`engine/metrics.py` — `PerformanceEvaluator`**
   Consumes a `BacktestResult` and computes annualized risk/return statistics for both the strategy and a buy-and-hold benchmark on the same instrument, plus trade-level statistics (win rate, trade count, profit factor) drawn from the trade ledger.

5. **`ai/explainer.py` — `TradeExplainer`**
   An optional research layer. Given a single trade record and the indicator observations leading up to its entry date, it asks Cohere's Command R model to explain, in 2–3 sentences, why the mechanical system triggered — using only the supplied data and the exact strategy parameters passed in. If no API key is configured or the request fails, it falls back to a deterministic, template-based explanation so the pipeline never breaks.

## Getting Started

### Prerequisites

- Python 3.11+
- A free [Cohere API developer key](https://dashboard.cohere.com/api-keys) (optional — the framework runs fully without one; see [Configuration](#configuration))

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/macro-quant-backtester.git
   cd macro-quant-backtester
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # macOS / Linux
   .venv\Scripts\activate           # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

The AI research layer reads its credentials from environment variables, loaded automatically via `python-dotenv`. Create a `.env` file in the project root:

```env
COHERE_API_KEY=your_cohere_api_key_here
```

`.env` is already listed in `.gitignore` and will never be committed. If `COHERE_API_KEY` is not set, `TradeExplainer` still runs — it simply returns a deterministic fallback string (e.g. *"The MovingAverageCrossover generated a LONG signal from the indicator conditions in the supplied entry data..."*) instead of calling the API, so the rest of the pipeline is unaffected.

### Running the Demo

```bash
python main.py
```

This runs the full end-to-end pipeline on daily `EURUSD=X` data from 2015-01-01 onward:

1. Downloads/loads cached EUR/USD OHLCV data.
2. Runs both a `MovingAverageCrossover(20, 50)` and an `RSIStrategy(14, 30, 70)` strategy through the backtest engine (starting capital $100,000, 3 bps commission, 1 bp slippage).
3. Prints a combined tear sheet comparing both strategies against a buy-and-hold benchmark.
4. Prints AI-generated rationales for the first three trades triggered by the moving-average strategy.

## Usage

The components are also designed to be used independently in your own scripts or notebooks.

### Loading Market Data

```python
from datetime import date
from data.data_loader import FXDataLoader

loader = FXDataLoader(start_date=date(2015, 1, 1))
market_data = loader.fetch_data("EURUSD=X")
```

### Generating Signals

```python
from engine.signals import MovingAverageCrossover, RSIStrategy

ma_data = MovingAverageCrossover(fast_window=20, slow_window=50, allow_short=True).generate(market_data)
rsi_data = RSIStrategy(window=14, oversold=30, overbought=70, allow_short=True).generate(market_data)
```

### Running a Backtest

```python
from engine.backtest import BacktestEngine

engine = BacktestEngine(initial_capital=100_000.0, commission_bps=3.0, slippage_bps=1.0)
result = engine.run(ma_data)

print(result.equity_curve.tail())
print(result.trade_log.head())
```

### Evaluating Performance

```python
from engine.metrics import PerformanceEvaluator

tear_sheet = PerformanceEvaluator(result).generate_tear_sheet()
print(tear_sheet)
```

Produces a `Strategy` vs. `Benchmark` (buy-and-hold) comparison across:

| Metric | Description |
|---|---|
| CAGR | Compound annual growth rate over the elapsed calendar period |
| Annualized Volatility | Standard deviation of daily returns, annualized over 252 trading days |
| Sharpe Ratio | Annualized, zero risk-free-rate Sharpe ratio |
| Maximum Drawdown | Largest peak-to-trough decline in equity |
| Max Drawdown Duration (days) | Longest stretch spent below a prior equity high |
| Win Rate (%) | Share of closed trades with positive P&L *(strategy only)* |
| Total Trades | Number of discrete trades in the ledger *(strategy only)* |
| Profit Factor | Gross profit divided by gross loss *(strategy only)* |

### Generating AI Trade Rationales

```python
from ai.explainer import TradeExplainer

explainer = TradeExplainer()  # uses COHERE_API_KEY from the environment
trade = result.trade_log.iloc[0].to_dict()
recent_window = ma_data.loc[ma_data.index <= trade["entry_date"]].tail(10)

rationale = explainer.generate_rationale(
    trade_record=trade,
    recent_data=recent_window,
    strategy_name="MovingAverageCrossover",
    strategy_params={"fast_window": 20, "slow_window": 50},
)
print(rationale)
```

The explainer is deliberately narrow: its system prompt forbids it from predicting market direction, inventing macroeconomic news, or referencing parameters that were not explicitly passed in. It only describes the observable indicator relationship that caused the trade to fire.

## Sample Output

```
Loading EURUSD=X daily data from 2015-01-01...
Loaded 2,847 observations through 2026-09-24.

Performance tear sheets
                                MA Crossover  RSI Strategy   Buy & Hold
CAGR                                  0.0412        0.0187       0.0298
Annualized Volatility                 0.0721        0.0654       0.0689
Sharpe Ratio                          0.5714        0.2860       0.4326
Maximum Drawdown                     -0.1842       -0.2103      -0.2467
Max Drawdown Duration (days)        412.0000      589.0000     701.0000
Win Rate (%)                         44.1200            NaN          NaN
Total Trades                         68.0000            NaN          NaN
Profit Factor                         1.3400            NaN          NaN

Trade rationales: MovingAverageCrossover

Trade 1: LONG | Entry 2015-06-12 @ 1.126400
  The MovingAverageCrossover strategy triggered a LONG position because the 20-period SMA rose above the 50-period SMA on the prior session's close. This bullish crossover reflects short-term price momentum exceeding the longer-term trend. No forward-looking market view is implied.
```

> Figures above are illustrative — run `python main.py` against live data to reproduce actual results.

## Design Decisions

- **Frozen dataclasses for strategies.** `MovingAverageCrossover` and `RSIStrategy` are immutable (`@dataclass(frozen=True, slots=True)`), so a strategy configuration can't be silently mutated mid-backtest and can be safely reused or hashed.
- **Signal generation and execution are decoupled.** Strategies only express a *desired* position; only `BacktestEngine.run()` decides when that position can actually be acted on. This keeps the look-ahead-prevention logic in exactly one place instead of duplicated across strategies.
- **Costs are charged on turnover, not on holding.** `commission_bps + slippage_bps` is applied only when the position changes, matching how transaction costs are actually incurred in practice.
- **Graceful AI degradation.** `TradeExplainer` never raises on a missing key or a failed API call — it always returns a usable, clearly-labeled fallback string, so the mechanical engine's output is never blocked by an external dependency.
- **Everything is validated defensively.** Both `BacktestEngine` and `PerformanceEvaluator` explicitly validate their inputs (required columns, positive prices, signal domain, `DatetimeIndex`) and raise descriptive errors rather than silently producing incorrect statistics.

## Project Structure

```
macro-quant-backtester/
├── main.py                    # End-to-end demo entry point
├── requirements.txt
├── ai/
│   └── explainer.py            # TradeExplainer (Cohere Command R integration)
├── data/
│   ├── data_loader.py          # FXDataLoader (yfinance ingestion + caching)
│   └── cache/                  # Cached per-ticker OHLCV CSVs
├── engine/
│   ├── signals.py               # SignalStrategy, MovingAverageCrossover, RSIStrategy
│   ├── backtest.py              # BacktestEngine, BacktestResult
│   └── metrics.py               # PerformanceEvaluator
└── tests/
    ├── conftest.py
    ├── test_signals.py
    ├── test_backtest.py
    └── test_metrics.py
```

## Testing

The project ships with a `pytest` suite covering signal correctness (crossover timing, RSI boundedness and thresholding, warm-up behavior), execution logic (the T+1 shift, cost application, trade-log construction), and performance metrics (CAGR, Sharpe, drawdown, drawdown duration).

```bash
pytest
```

## Limitations

The framework prioritizes transparency and correctness of the core simulation, but there are known constraints worth being upfront about:

- **Single-asset, single-position backtests only.** `BacktestEngine` simulates one instrument with a single long/short/flat position at a time — it does not currently support multi-asset portfolios, capital allocation across strategies, or position sizing beyond fully invested/fully short.
- **No intraday granularity.** The engine and data loader operate on daily OHLCV bars; the T+1 execution rule is a same-day-close-to-next-day proxy, not a simulation of intraday fills, order books, or partial fills.
- **Flat, symmetric transaction costs.** Commission and slippage are modeled as constant basis-point charges on turnover. This does not capture cost effects that vary with volatility, order size, market impact, or time of day.
- **No risk-free rate in the Sharpe ratio.** `PerformanceEvaluator` computes the Sharpe ratio assuming a zero risk-free rate, which will overstate risk-adjusted returns relative to a Sharpe ratio computed against a real cash benchmark.
- **`yfinance` as the sole data source.** Historical FX data quality (including the sparse/unreliable FX volume field, which the loader has to explicitly work around) is entirely dependent on Yahoo Finance's feed; there is no fallback data provider or point-in-time data revision handling.
- **The AI layer explains, it does not validate or predict.** `TradeExplainer` is deliberately constrained to describing why a mechanical signal already fired, using only the data it's given. It is not a second opinion on whether the trade was a good idea, and its fallback string (used when no API key is configured or a request fails) is a template, not an explanation grounded in that specific trade.
- **No walk-forward or out-of-sample testing yet.** Strategies are currently evaluated in-sample over the full historical window; the framework does not yet separate parameter selection from evaluation data (see [Roadmap](#roadmap)).

## Roadmap

- [ ] Multi-asset portfolio backtesting with position sizing and correlation-aware risk limits
- [ ] Additional strategies (Bollinger Bands, MACD, carry-based FX signals)
- [ ] Walk-forward and out-of-sample validation harness
- [ ] Interactive equity-curve and drawdown visualization
- [ ] Configurable slippage models beyond a flat basis-point charge

## Disclaimer

This project is for educational and research purposes only. It does not constitute financial advice, and nothing it produces — including AI-generated trade rationales — should be interpreted as investment guidance or a market prediction. Past backtested performance is not indicative of future results.

## License

Distributed under the MIT License. See `LICENSE` for details.
