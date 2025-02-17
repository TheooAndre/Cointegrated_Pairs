# Cointegrated Crypto Pairs Trading Analyzer

This repository contains a Python-based tool for analyzing cointegration among cryptocurrency pairs on the Binance Futures market. The script is designed for quantitative trading applications and is especially useful for developing pairs trading strategies in high-frequency and systematic trading environments.

## Key Features

- **Asynchronous Data Fetching:**  
  Efficiently collects OHLCV data using [ccxt.async_support](https://github.com/ccxt/ccxt) from Binance Futures, ensuring timely and reliable market data retrieval.

- **Liquidity Filtering Options:**  
  Supports dynamic filtering based on:
  - **Volume:** Filters markets using minimum volume thresholds.
  - **Open Interest:** Filters markets using minimum open interest requirements.
  - **Combined Filters:** Allows the user to select one, the other, or both filters interactively via terminal prompts.

- **Robust Cointegration Analysis:**  
  Utilizes the Engle-Granger two-step method (via `statsmodels.tsa.stattools.coint`) to test for cointegration among pairs, ensuring the identification of statistically robust long-run relationships.

- **Parallel Processing:**  
  Leverages Python’s `ThreadPoolExecutor` to efficiently compute cointegration tests for numerous pairs simultaneously, significantly reducing overall processing time.

- **Interactive Pair Lookup:**  
  After analysis, the script:
  - Saves the top cointegrated pairs to a CSV file.
  - Provides an interactive terminal interface to search for cointegrated pairs by asset symbol.
  - Offers an option to list all unique cointegrated pairs from the full analysis.

- **Modular and Configurable:**  
  The tool is built using a clean, modular structure with a configuration dataclass, making it easy to adjust parameters (e.g., timeframe, analysis period, liquidity thresholds, top-N pairs) without changing the core code.

## Use Cases

- **Quantitative Trading:**  
  Ideal for developing pairs trading strategies, allowing traders to identify and exploit long-run equilibrium relationships between crypto assets.
  
- **Research & Analysis:**  
  Useful for systematic trading research in banks or investment funds looking to integrate advanced time series analysis into their trading models.

## Technologies

- **Python 3**
- **ccxt (async support)**
- **pandas & numpy**
- **statsmodels**
- **asyncio & concurrent.futures**
- **logging**

---

This tool demonstrates advanced programming practices, including asynchronous programming and parallel processing, and showcases expertise in quantitative finance and statistical analysis. It is a powerful addition to any quantitative research or trading team seeking to develop robust pairs trading strategies in cryptocurrency markets.
