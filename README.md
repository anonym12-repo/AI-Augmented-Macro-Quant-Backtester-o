# AI-Augmented Macro Quant Backtester

An event-driven, object-oriented quantitative backtesting framework built entirely from scratch in Python. Designed to execute systematic trading strategies on global macro instruments (G10 FX) while adhering strictly to institutional quality controls, such as look-ahead bias prevention and realistic transaction cost modeling. This project bridges rigorous data engineering with a deterministic Generative AI research layer, parsing technical conditions into factual market rationales.

## Core Architectural Design
Engineered with the architectural discipline of systems-level programming, this framework bypasses retail black-box tools in favor of a transparent, vectorized Pandas and NumPy engine.
* **Vectorized Signal Generation:** Implements textbook strategies (Moving Average Crossover, Wilder’s RSI) using pure vectorization to process large OHLCV datasets efficiently without row-wise iteration.
* **Intelligent Data Pipeline:** Ingests live market data via `yfinance`, automatically handling missing FX volume data, forward-filling holiday gaps, and applying timezone-aware normalizations.
* **Modular Extensibility:** Built with strict Python 3.11+ type hinting and frozen `dataclasses`, ensuring clear separation between signal generation, execution, and performance analytics.

## Institutional Quantitative Guardrails
Hedge fund backtesting requires conservative assumptions to prevent overstating historical performance. This engine mathematically enforces real-world trading frictions:
* **Strict T+1 Execution:** Eliminates look-ahead bias by mandating that a signal calculated on day T's closing price can only be executed on day T+1.
* **Realistic Friction Modeling:** Penalizes the strategy equity curve using a configurable basis-point model for broker commissions and bid-ask slippage whenever position turnover occurs.
* **Risk-Adjusted Analytics:** Calculates annualized Sharpe ratios, CAGR, and peak-to-trough Maximum Drawdown durations to evaluate strategies against a standard buy-and-hold benchmark.

## Deterministic AI Research Layer
Applying advanced Generative AI engineering principles, the engine features an integrated LLM explainer that translates mechanical quantitative triggers into human-readable text.
* **Anti-Hallucination Controls:** Utilizes the Cohere Command R model via the V2 Python SDK, configured with strict system constraints and temperature set to 0 to ensure the AI acts as a mechanical explainer rather than a market predictor.
* **Contextual Data Grounding:** Dynamically injects strategy parameters (e.g., specific moving average windows) and a Markdown-formatted table of the trailing 10 days of indicator data directly into the prompt context.
* **Graceful Degradation:** Features robust error handling that defaults to a transparent fallback string if API authentication or rate limits fail, ensuring the backtest never crashes.

## Validation & Testing Suite
To prove the framework's mathematical correctness, a comprehensive `pytest` suite validates all critical engine functions.
* **Execution Timing:** Asserts that synthetic price surges on day T do not leak into the strategy's returns when a signal fires on the exact same close.
* **Cost Accounting:** Verifies that double turnover during long-to-short position reversals correctly charges twice the transaction friction.
* **Indicator Bounds:** Confirms that warm-up periods remain flat and momentum oscillators respect strict mathematical limits.
