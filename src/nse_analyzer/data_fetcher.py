"""
data_fetcher.py
Fetch OHLCV data for NSE stocks using yfinance.
Stores CSV files in data/ folder, overwrites on each run.
Located at: src/nse_analyzer/data_fetcher.py
"""

import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime
import warnings

from .utils.config import ConfigManager

warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")


class NSEDataFetcher:
    """Fetch and store OHLCV data for NSE stocks."""

    def __init__(self, stock_group: str = "stocks", data_dir: Path | None = None):
        """
        Initialize fetcher with stock list from config.

        Args:
            stock_group: Key in stocks.yaml to load stocks from
            data_dir: Optional custom data directory (for testing)
        """
        self.stocks = ConfigManager.load_stocks(stock_group)
        self.stock_group = stock_group
        self.data_dir = data_dir or ConfigManager.get_data_dir()

    def fetch_ohlcv(
        self, ticker: str, period: str = "2y", interval: str = "1d"
    ) -> pd.DataFrame | None:
        """
        Fetch OHLCV data for a stock.

        Args:
            ticker: Stock symbol (e.g., 'RELIANCE.NS')
            period: Data period ('1y', '2y', '5y', 'max')
            interval: Candle interval ('1d', '1h', '15m', etc.)

        Returns:
            pd.DataFrame with OHLCV data or None if fetch fails
        """
        try:
            print(f"Fetching {ticker}...")
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True,
            )

            if df.empty:
                print(f"⚠️  No data for {ticker}")
                return None

            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Keep only OHLCV columns
            df = df[["Open", "High", "Low", "Close", "Volume"]]

            # Reset index to ensure Date column is clean
            df = df.reset_index()

            return df

        except Exception as e:
            print(f"❌ Error fetching {ticker}: {e}")
            return None

    def save_to_csv(self, ticker: str, df: pd.DataFrame, folder: str = "ohlcv") -> bool:
        """
        Save OHLCV data to CSV, overwriting previous file.

        Args:
            ticker: Stock symbol
            df: DataFrame with OHLCV data
            folder: Subfolder in data/ to save to (default: 'ohlcv')

        Returns:
            True if successful, False otherwise
        """
        try:
            # Use self.data_dir which was set in __init__
            output_dir = self.data_dir / folder
            output_dir.mkdir(parents=True, exist_ok=True)

            stock_name = ticker.replace(".NS", "")
            csv_path = output_dir / f"{stock_name}.csv"

            df.to_csv(csv_path, index=False)
            print(f"✅ Saved {csv_path} ({len(df)} rows)")
            return True
        except Exception as e:
            print(f"❌ Error saving {ticker}: {e}")
            return False

    def fetch_all(self, period: str = "2y", interval: str = "1d") -> dict:
        """
        Fetch and save all stocks.

        Args:
            period: Data period
            interval: Candle interval

        Returns:
            dict with results for each stock
        """
        results = {}
        print(f"\n{'='*60}")
        print(
            f"NSE DATA FETCHER - {self.stock_group.upper()} - "
            f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print(f"{'='*60}\n")

        for ticker in self.stocks:
            df = self.fetch_ohlcv(ticker, period=period, interval=interval)
            if df is not None:
                success = self.save_to_csv(ticker, df)
                status = "success" if success else "save_failed"
                results[ticker] = {"status": status, "rows": len(df)}
            else:
                results[ticker] = {"status": "fetch_failed", "rows": 0}

        print(f"\n{'='*60}")
        print("Summary:")
        for ticker, result in results.items():
            status = "✅" if result["status"] == "success" else "❌"
            print(f"  {status} {ticker}: {result['status']} ({result['rows']} rows)")
        print(f"{'='*60}\n")

        return results


if __name__ == "__main__":
    # Example: Fetch for default stocks
    fetcher = NSEDataFetcher()
    fetcher.fetch_all(period="2y", interval="1d")

    # Example: Fetch for nifty50
    # fetcher = NSEDataFetcher("nifty50")
    # fetcher.fetch_all()
