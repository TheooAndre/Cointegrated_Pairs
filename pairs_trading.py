#!/usr/bin/env python3
import ccxt.async_support as ccxt_async
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
from itertools import combinations
import time
import warnings
from statsmodels.tools.sm_exceptions import CollinearityWarning
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
import asyncio
import os

# -----------------------------
# Configuration
# -----------------------------
@dataclass
class Config:
    timeframe: str = '4h'
    days: int = 90
    coint_threshold: float = 0.05
    market_type: str = 'future'
    output_file: str = "pairs_to_trade.csv"
    min_open_interest: float = 50_000_000
    min_volume: float = 500_000_000
    # Liquidity filter flags (default to False)
    enable_volume_filter: bool = False
    enable_open_interest_filter: bool = False
    top_n_pairs: int = 10
    max_workers: int = 8
    min_data_points: int = 50
    rate_limit_sleep: float = 0.1

# -----------------------------
# Logging Configuration
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pairs_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -----------------------------
# Market Data Handling
# -----------------------------
class MarketDataFetcher:
    def __init__(self, config: Config):
        self.config = config
        self.exchange = ccxt_async.binance({
            'enableRateLimit': True,
            'options': {'defaultType': config.market_type}
        })
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.exchange.close()
    
    async def fetch_ohlcv(self, symbol: str, since: int) -> Optional[pd.DataFrame]:
        try:
            data = await self.exchange.fetch_ohlcv(
                symbol,
                timeframe=self.config.timeframe,
                since=since,
                limit=self.config.days * 24
            )
            if not data:
                logger.warning(f"No OHLCV data returned for {symbol}")
                return None
            df = pd.DataFrame(
                data,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            return None

    async def get_futures_volume(self, symbol: str) -> float:
        try:
            ticker = await self.exchange.fapiPublic_getTickerDaily({'symbol': symbol})
            await asyncio.sleep(self.config.rate_limit_sleep)
            return float(ticker['quoteVolume'])
        except Exception as e:
            try:
                ticker = await self.exchange.fetch_ticker(f"{symbol}")
                await asyncio.sleep(self.config.rate_limit_sleep)
                return float(ticker['quoteVolume'])
            except Exception as e2:
                logger.error(f"Error fetching volume for {symbol}: {e2}")
                return 0

    async def get_open_interest(self, symbol: str) -> float:
        try:
            response = await self.exchange.fapiPublicGetOpenInterest({'symbol': symbol})
            await asyncio.sleep(self.config.rate_limit_sleep)
            if isinstance(response, dict) and 'openInterest' in response:
                return float(response['openInterest'])
            else:
                logger.warning(f"Unexpected open interest response for {symbol}: {response}")
                return 0
        except Exception as e:
            logger.error(f"Error fetching open interest for {symbol}: {e}")
            try:
                ticker = await self.exchange.fetch_ticker(f"{symbol}")
                if 'info' in ticker and 'openInterest' in ticker['info']:
                    return float(ticker['info']['openInterest'])
                return 0
            except Exception as e2:
                logger.error(f"Backup method for open interest failed for {symbol}: {e2}")
                return 0

    async def load_markets(self) -> List[Dict]:
        try:
            markets = await self.exchange.load_markets()
            valid_markets = []
            for market in markets.values():
                if market['quote'] == 'USDT' and market['linear'] and market['active']:
                    try:
                        oi = await self.get_open_interest(market['id'])
                        if oi > 0:
                            market['openInterest'] = oi
                            valid_markets.append(market)
                    except Exception as e:
                        logger.error(f"Error checking market {market['id']}: {e}")
                    await asyncio.sleep(self.config.rate_limit_sleep)
            return valid_markets
        except Exception as e:
            logger.error(f"Error loading markets: {e}")
            return []

# -----------------------------
# Statistical Analysis
# -----------------------------
class PairsAnalyzer:
    def __init__(self, config: Config):
        self.config = config
    
    def calculate_cointegration(self, series1: pd.Series, series2: pd.Series) -> Optional[Dict[str, float]]:
        if len(series1) < self.config.min_data_points or len(series2) < self.config.min_data_points:
            return None
        if np.isclose(series1.var(), 0) or np.isclose(series2.var(), 0):
            return None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", CollinearityWarning)
                score, pvalue, _ = coint(series1, series2, autolag='BIC')
            return {'score': score, 'pvalue': pvalue}
        except Exception as e:
            logger.error(f"Cointegration test failed: {e}")
            return None

    async def analyze_pairs(self, closes: pd.DataFrame) -> Tuple[List[Tuple], List[Tuple]]:
        pairs = list(combinations(closes.columns, 2))
        results = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_pair = {
                executor.submit(self.calculate_cointegration, closes[pair[0]], closes[pair[1]]): pair
                for pair in pairs
            }
            for future in as_completed(future_to_pair):
                pair = future_to_pair[future]
                result = future.result()
                if result and result['pvalue'] < self.config.coint_threshold:
                    results.append((*pair, result['pvalue'], result['score']))
        results.sort(key=lambda x: x[2])
        top_pairs = results[:self.config.top_n_pairs]
        return results, top_pairs

# -----------------------------
# Main Application
# -----------------------------
async def main():
    config = Config()
    
    # Prompt user for liquidity filter type
    filter_choice = input("Select liquidity filter type:\n"
                          "Enter 'v' for volume filter only,\n"
                          "Enter 'oi' for open interest filter only,\n"
                          "Enter 'b' or 'y' for both filters,\n"
                          "or press Enter for no filter: ").strip().lower()
    if filter_choice == 'v':
        config.enable_volume_filter = True
        config.enable_open_interest_filter = False
    elif filter_choice == 'oi':
        config.enable_volume_filter = False
        config.enable_open_interest_filter = True
    elif filter_choice in ('b', 'y'):
        config.enable_volume_filter = True
        config.enable_open_interest_filter = True
    else:
        config.enable_volume_filter = False
        config.enable_open_interest_filter = False

    logger.info("Starting pairs trading analysis")
    
    async with MarketDataFetcher(config) as market_fetcher:
        # Load and filter markets
        markets = await market_fetcher.load_markets()
        logger.info(f"Initial markets: {len(markets)}")
        if not markets:
            logger.error("No markets found")
            return
        
        # Apply liquidity filters based on user selection
        if config.enable_volume_filter or config.enable_open_interest_filter:
            filtered = []
            for market in markets:
                passes_volume = True
                passes_oi = True
                if config.enable_volume_filter:
                    vol = await market_fetcher.get_futures_volume(market['id'])
                    if vol <= config.min_volume / 14:
                        passes_volume = False
                if config.enable_open_interest_filter:
                    oi = await market_fetcher.get_open_interest(market['id'])
                    if oi > config.min_open_interest:
                        market['openInterest'] = oi
                    else:
                        passes_oi = False
                if passes_volume and passes_oi:
                    filtered.append(market)
                await asyncio.sleep(config.rate_limit_sleep)
        else:
            filtered = markets
        
        logger.info(f"After liquidity filter: {len(filtered)}")
        if not filtered:
            logger.error("No markets passed liquidity filter")
            return
        
        # Fetch price data
        since = int(time.time() * 1000) - config.days * 86400 * 1000
        ohlcv_data = {}
        for market in filtered:
            df = await market_fetcher.fetch_ohlcv(market['symbol'], since)
            if df is not None:
                ohlcv_data[market['symbol']] = df
            await asyncio.sleep(config.rate_limit_sleep)
        if not ohlcv_data:
            logger.error("No OHLCV data fetched")
            return
        
        # Create aligned price matrix
        closes = pd.DataFrame({k: v['close'] for k, v in ohlcv_data.items()})
        closes = closes.dropna(axis=1, how='any')
        logger.info(f"Aligned data shape: {closes.shape}")
        if len(closes.columns) < 2:
            logger.error("Insufficient data for pair analysis")
            return
        
        # Analyze pairs: get both the full list and top pairs
        pairs_analyzer = PairsAnalyzer(config)
        all_pairs, top_pairs = await pairs_analyzer.analyze_pairs(closes)
        if not top_pairs:
            logger.error("No cointegrated pairs found")
            return
        
        # Generate output for top pairs and save to CSV
        output = []
        for pair in top_pairs:
            output.append({
                'Pair': f"{pair[0]}-{pair[1]}",
                'P-Value': f"{pair[2]:.4f}",
                'Score': f"{pair[3]:.2f}"
            })
        pd.DataFrame(output).to_csv(config.output_file, index=False)
        logger.info(f"Saved {len(top_pairs)} pairs to {config.output_file}")
        
        # New Feature: Continuously lookup cointegrated pairs by asset or list all pairs
        while True:
            selected_input = input("\nEnter asset symbol (without USDT, e.g., BTC), 'list' to show all unique pairs, or 'x' to exit: ").strip().upper()
            if selected_input == 'X':
                os.system('clear')
                print("Exiting lookup.")
                break
            elif selected_input == 'LIST':
                os.system('clear')
                print("\nAll unique cointegrated pairs:")
                for pair in all_pairs:
                    print(f"{pair[0]} - {pair[1]}: p-value={pair[2]:.4f}, score={pair[3]:.2f}")
            else:
                matching_pairs = [
                    pair for pair in all_pairs 
                    if pair[0].split('/')[0] == selected_input or pair[1].split('/')[0] == selected_input
                ]
                if matching_pairs:
                    os.system('clear')
                    print(f"\nCointegrated pairs containing {selected_input}:")
                    for pair in matching_pairs:
                        print(f"{pair[0]} - {pair[1]}: p-value={pair[2]:.4f}, score={pair[3]:.2f}")
                else:
                    os.system('clear')
                    print(f"\nNo cointegrated pairs found for asset {selected_input}. Please try another asset or press 'x' to exit.")

if __name__ == '__main__':
    asyncio.run(main())
