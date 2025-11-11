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

# Suppress FutureWarning from yfinance
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

# Get project root (src/nse_analyzer/../..)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_FOLDER = PROJECT_ROOT / "data"

# Ensure data folder exists
DATA_FOLDER.mkdir(parents=True, exist_ok=True)


class NSEDataFetcher:
    """Fetch and store OHLCV data for NSE stocks."""

    def __init__(self, data_dir: Path = DATA_FOLDER):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.stocks = [
            "RELIANCE.NS",
            "TCS.NS",
            "HDFCBANK.NS",
            "INFY.NS",
            "KOTAKBANK.NS",
        ]

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

    def save_to_csv(self, ticker: str, df: pd.DataFrame) -> bool:
        """
        Save OHLCV data to CSV, overwriting previous file.

        Args:
            ticker: Stock symbol
            df: DataFrame with OHLCV data

        Returns:
            True if successful, False otherwise
        """
        try:
            stock_name = ticker.replace(".NS", "")
            csv_path = self.data_dir / f"{stock_name}.csv"

            # Save without index to avoid MultiIndex issues
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
            f"NSE DATA FETCHER - Started at "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
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
    # Run when executed directly
    fetcher = NSEDataFetcher()
    fetcher.fetch_all(period="2y", interval="1d")
