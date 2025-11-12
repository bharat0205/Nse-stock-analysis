"""
calculator.py
Combined technical indicators calculator for all stocks using pandas-ta.
Fetches OHLCV data, calculates all indicators, and saves to CSV.
Located at: src/nse_analyzer/indicators/technical/calculator.py
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings

from nse_analyzer.indicators.technical.rsi import calculate_rsi
from nse_analyzer.indicators.technical.macd import calculate_macd
from nse_analyzer.indicators.technical.supertrend import calculate_supertrend
from nse_analyzer.indicators.technical.bollinger_bands import (
    calculate_bollinger_bands,
)
from nse_analyzer.utils.config import ConfigManager

warnings.filterwarnings("ignore")


class TechnicalIndicatorCalculator:
    """Calculate all technical indicators for NSE stocks using pandas-ta."""

    def __init__(self, stock_group: str = "stocks", data_dir: Path | None = None):
        """
        Initialize technical indicator calculator.

        Args:
            stock_group: Stock group from config (default: 'stocks')
            data_dir: Optional custom data directory for testing
        """
        self.stock_group = stock_group
        self.data_dir = data_dir or ConfigManager.get_data_dir()
        self.ohlcv_dir = self.data_dir / "ohlcv"
        self.indicators_dir = self.data_dir / "indicators"
        self.indicators_dir.mkdir(parents=True, exist_ok=True)

    def load_ohlcv_data(self, ticker: str) -> pd.DataFrame | None:
        """
        Load OHLCV data from CSV file.

        Args:
            ticker: Stock symbol (e.g., 'RELIANCE.NS')

        Returns:
            DataFrame with OHLCV data or None if file doesn't exist
        """
        stock_name = ticker.replace(".NS", "")
        csv_path = self.ohlcv_dir / f"{stock_name}.csv"

        if not csv_path.exists():
            print(f"⚠️  No OHLCV data found for {ticker} at {csv_path}")
            return None

        try:
            df = pd.read_csv(csv_path)
            df["Date"] = pd.to_datetime(df["Date"])
            print(f"✅ Loaded {len(df)} rows for {ticker}")
            return df
        except Exception as e:
            print(f"❌ Error loading {ticker}: {e}")
            return None

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all 4 technical indicators and merge with OHLCV data.

        Indicators calculated:
        - RSI (14 period)
        - MACD (12, 26, 9)
        - Supertrend (10 period, 3.0 multiplier)
        - Bollinger Bands (20 period, 2.0 std dev)

        Args:
            df: DataFrame with OHLCV data (Date, Open, High, Low, Close, Volume)

        Returns:
            DataFrame with original OHLCV + all indicator columns
        """
        try:
            # Ensure we have a copy to avoid modifying original
            df = df.copy()

            print("  Calculating RSI (period=14)...")
            df = calculate_rsi(df, period=14)

            print("  Calculating MACD (12, 26, 9)...")
            df = calculate_macd(df, fast=12, slow=26, signal=9)

            print("  Calculating Supertrend (period=10, multiplier=3.0)...")
            df = calculate_supertrend(df, period=10, multiplier=3.0)

            print("  Calculating Bollinger Bands (period=20, std=2.0)...")
            df = calculate_bollinger_bands(df, period=20, std_dev=2.0)

            return df

        except Exception as e:
            print(f"❌ Error calculating indicators: {e}")
            raise

    def process_stock(self, ticker: str) -> bool:
        """
        Load OHLCV data for a stock, calculate indicators, and save to CSV.

        Args:
            ticker: Stock symbol (e.g., 'RELIANCE.NS')

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load OHLCV data
            df = self.load_ohlcv_data(ticker)
            if df is None:
                return False

            # Calculate indicators
            df = self.calculate_all_indicators(df)

            # Save to CSV
            stock_name = ticker.replace(".NS", "")
            output_path = self.indicators_dir / f"{stock_name}_indicators.csv"
            df.to_csv(output_path, index=False)

            print(f"✅ Saved {len(df)} rows with indicators to:")
            print(f"   {output_path}\n")
            return True

        except Exception as e:
            print(f"❌ Error processing {ticker}: {e}\n")
            return False

    def process_all_stocks(self) -> dict:
        """
        Process all stocks from config file.

        Loads OHLCV data, calculates all indicators, and saves results.

        Returns:
            Dictionary with processing results for each stock
            {ticker: "success" or "failed"}
        """
        try:
            stocks = ConfigManager.load_stocks(self.stock_group)
        except Exception as e:
            print(f"❌ Error loading stock list: {e}")
            return {}

        results = {}

        print(f"\n{'='*70}")
        print("TECHNICAL INDICATORS CALCULATOR")
        print(f"{'='*70}")
        print(f"Stock Group: {self.stock_group.upper()}")
        print(f"Number of Stocks: {len(stocks)}")
        print(f"OHLCV Data Dir: {self.ohlcv_dir}")
        print(f"Output Dir: {self.indicators_dir}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")

        # Process each stock
        success_count = 0
        failed_count = 0

        for i, ticker in enumerate(stocks, 1):
            print(f"[{i}/{len(stocks)}] Processing {ticker}...")
            success = self.process_stock(ticker)
            if success:
                results[ticker] = "success"
                success_count += 1
            else:
                results[ticker] = "failed"
                failed_count += 1

        # Print summary
        print(f"\n{'='*70}")
        print("PROCESSING SUMMARY")
        print(f"{'='*70}")
        print(f"Total Stocks: {len(stocks)}")
        print(f"✅ Successful: {success_count}")
        print(f"❌ Failed: {failed_count}")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if failed_count > 0:
            print("\nFailed Stocks:")
            for ticker, status in results.items():
                if status == "failed":
                    print(f"  ❌ {ticker}")

        print(f"{'='*70}\n")

        return results

    def get_indicator_summary(self, ticker: str) -> dict | None:
        """
        Get summary statistics of indicators for a stock.

        Args:
            ticker: Stock symbol

        Returns:
            Dictionary with indicator statistics or None if file not found
        """
        stock_name = ticker.replace(".NS", "")
        csv_path = self.indicators_dir / f"{stock_name}_indicators.csv"

        if not csv_path.exists():
            return None

        try:
            df = pd.read_csv(csv_path)
            summary = {
                "ticker": ticker,
                "rows": len(df),
                "date_range": f"{df['Date'].min()} to {df['Date'].max()}",
                "indicators": {
                    "RSI_14": {
                        "min": df["RSI_14"].min(),
                        "max": df["RSI_14"].max(),
                        "mean": df["RSI_14"].mean(),
                        "latest": df["RSI_14"].iloc[-1],
                    },
                    "MACD": {
                        "latest": df["MACD"].iloc[-1],
                        "signal": df["MACD_Signal"].iloc[-1],
                        "histogram": df["MACD_Histogram"].iloc[-1],
                    },
                    "Supertrend": {
                        "latest": df["Supertrend"].iloc[-1],
                        "direction": df["Supertrend_Direction"].iloc[-1],
                    },
                    "BB_20": {
                        "upper": df["BB_Upper_20"].iloc[-1],
                        "middle": df["BB_Middle_20"].iloc[-1],
                        "lower": df["BB_Lower_20"].iloc[-1],
                    },
                },
            }
            return summary
        except Exception as e:
            print(f"❌ Error reading {csv_path}: {e}")
            return None

    def display_summary(self, ticker: str) -> None:
        """
        Display formatted summary of indicators for a stock.

        Args:
            ticker: Stock symbol
        """
        summary = self.get_indicator_summary(ticker)
        if summary is None:
            print(f"❌ No indicators found for {ticker}")
            return

        print(f"\n{'='*60}")
        print(f"INDICATORS SUMMARY - {summary['ticker']}")
        print(f"{'='*60}")
        print(f"Date Range: {summary['date_range']}")
        print(f"Rows: {summary['rows']}\n")

        ind = summary["indicators"]

        print("RSI (14):")
        print(f"  Latest: {ind['RSI_14']['latest']:.2f}")
        print(f"  Min: {ind['RSI_14']['min']:.2f}, Max: {ind['RSI_14']['max']:.2f}")
        print(f"  Mean: {ind['RSI_14']['mean']:.2f}\n")

        print("MACD (12, 26, 9):")
        print(f"  MACD Line: {ind['MACD']['latest']:.6f}")
        print(f"  Signal Line: {ind['MACD']['signal']:.6f}")
        print(f"  Histogram: {ind['MACD']['histogram']:.6f}\n")

        print("Supertrend (10, 3.0):")
        print(f"  Supertrend: {ind['Supertrend']['latest']:.2f}")
        print(f"Direction:{'🟢UP' if ind['Supertrend']['direction']>0 else'🔴DOWN'}")
        print()

        print("Bollinger Bands (20, 2.0):")
        print(f"  Upper: {ind['BB_20']['upper']:.2f}")
        print(f"  Middle: {ind['BB_20']['middle']:.2f}")
        print(f"  Lower: {ind['BB_20']['lower']:.2f}\n")

        print(f"{'='*60}\n")


# ==================== Main Execution ====================


if __name__ == "__main__":
    # Example 1: Process all stocks in default group
    print("\n>>> EXAMPLE 1: Process all stocks")
    calculator = TechnicalIndicatorCalculator()
    results = calculator.process_all_stocks()

    # Example 2: Display summary for specific stocks
    print("\n>>> EXAMPLE 2: Display summaries")
    for ticker in ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]:
        calculator.display_summary(ticker)

    # Example 3: Process specific stock group (if available)
    # print("\n>>> EXAMPLE 3: Process nifty50 group")
    # nifty_calc = TechnicalIndicatorCalculator("nifty50")
    # nifty_results = nifty_calc.process_all_stocks()
