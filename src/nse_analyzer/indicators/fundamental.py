"""
fundamental.py
Fetch fundamental metrics from yfinance for NSE stocks.
Located at: src/nse_analyzer/indicators/fundamental.py
"""

import sys
import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import Dict
import warnings

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "nse_analyzer"))
from utils.config import ConfigManager

warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

DATA_FOLDER = ConfigManager.get_data_dir()


class FundamentalIndicators:
    """Fetch and calculate fundamental indicators for NSE stocks."""

    def __init__(self, stock_group: str = "stocks", data_dir: Path | None = None):
        """
        Initialize with stock list from config.

        Args:
            stock_group: Key in stocks.yaml to load stocks from
            data_dir: Optional custom data directory (for testing)
        """
        self.stocks = ConfigManager.load_stocks(stock_group)
        self.stock_group = stock_group
        self.data_dir = data_dir or ConfigManager.get_data_dir()

    def fetch_fundamental_data(self, ticker: str) -> Dict | None:
        """
        Fetch fundamental data for a stock from yfinance.

        Args:
            ticker: Stock symbol (e.g., 'RELIANCE.NS')

        Returns:
            Dict with fundamental metrics or None if fetch fails
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Extract key fundamental metrics
            fundamentals = {
                "ticker": ticker,
                # Growth Metrics
                "eps_growth_yoy": self._get_value(info, "epsTrailingTwelveMonths"),
                "revenue_growth_yoy": self._get_value(info, "revenueGrowth"),
                # Profitability
                "roe": self._get_value(info, "returnOnEquity"),
                "roce": self._get_value(info, "returnOnCapital"),
                "net_profit_margin": self._get_value(info, "profitMargins"),
                "operating_margin": self._get_value(info, "operatingMargins"),
                # Valuation
                "pe_ratio": self._get_value(info, "trailingPE"),
                "pb_ratio": self._get_value(info, "priceToBook"),
                "peg_ratio": self._get_value(info, "pegRatio"),
                # Debt & Liquidity
                "debt_to_equity": self._get_value(info, "debtToEquity"),
                "interest_coverage": self._get_value(info, "interestCoverage"),
                "current_ratio": self._get_value(info, "currentRatio"),
                "quick_ratio": self._get_value(info, "quickRatio"),
                # Dividend
                "dividend_yield": self._get_value(info, "dividendYield"),
                "dividend_payout_ratio": self._get_value(info, "payoutRatio"),
                # Cash Flow
                "operating_cash_flow": self._get_value(info, "operatingCashflow"),
                "free_cash_flow": self._get_value(info, "freeCashflow"),
                # Financial Health
                "earnings_per_share": self._get_value(info, "epsTrailingTwelveMonths"),
                "book_value_per_share": self._get_value(info, "bookValue"),
            }

            return fundamentals

        except Exception as e:
            print(f"❌ Error fetching fundamentals for {ticker}: {e}")
            return None

    @staticmethod
    def _get_value(data: Dict, key: str, default=None):
        """Safely get value from dictionary."""
        return data.get(key, default)

    def fetch_all_fundamentals(self) -> pd.DataFrame:
        """
        Fetch fundamentals for all stocks and return as DataFrame.

        Returns:
            pd.DataFrame with fundamentals for all stocks
        """
        results = []

        print(f"\n{'='*60}")
        print(f"FUNDAMENTAL INDICATORS FETCHER - {self.stock_group.upper()}")
        print(f"{'='*60}\n")

        for ticker in self.stocks:
            print(f"Fetching fundamentals for {ticker}...")
            fundamentals = self.fetch_fundamental_data(ticker)
            if fundamentals:
                results.append(fundamentals)
                print(f"✅ {ticker}: Success")
            else:
                print(f"❌ {ticker}: Failed")

        if results:
            df = pd.DataFrame(results)
            return df
        return pd.DataFrame()

    def save_fundamentals_csv(
        self,
        df: pd.DataFrame,
        filename: str = "fundamentals.csv",
        subfolder: str = "fundamentals",
    ) -> bool:
        """
        Save fundamentals DataFrame to CSV.

        Args:
            df: DataFrame with fundamentals
            filename: Output filename
            subfolder: Subfolder in data/ to save to

        Returns:
            True if successful, False otherwise
        """
        try:
            output_dir = self.data_dir / subfolder
            output_dir.mkdir(parents=True, exist_ok=True)

            csv_path = output_dir / filename
            df.to_csv(csv_path, index=False)
            print(f"\n✅ Saved fundamentals to {csv_path}")
            return True
        except Exception as e:
            print(f"❌ Error saving fundamentals: {e}")
            return False

    def get_fundamental_score(
        self, ticker: str, fundamentals: Dict = None
    ) -> float | None:
        """
        Calculate a composite fundamental score (0-100).

        Args:
            ticker: Stock symbol
            fundamentals: Dict of fundamental metrics

        Returns:
            Score between 0-100 or None if calculation fails
        """
        if fundamentals is None:
            fundamentals = self.fetch_fundamental_data(ticker)

        if not fundamentals:
            return None

        score = 0
        weights = {
            "pe_ratio": 15,
            "pb_ratio": 10,
            "roe": 15,
            "roce": 15,
            "debt_to_equity": 10,
            "dividend_yield": 10,
            "current_ratio": 10,
        }

        # PE Ratio (inverse scoring)
        pe = fundamentals.get("pe_ratio")
        if pe and pe > 0:
            pe_score = min(100, (30 / pe) * 100)
            score += (pe_score / 100) * weights["pe_ratio"]

        # ROE (direct scoring)
        roe = fundamentals.get("roe")
        if roe and roe > 0:
            roe_score = min(100, roe * 100)
            score += (roe_score / 100) * weights["roe"]

        # ROCE (direct scoring)
        roce = fundamentals.get("roce")
        if roce and roce > 0:
            roce_score = min(100, roce * 100)
            score += (roce_score / 100) * weights["roce"]

        # Debt to Equity (inverse scoring)
        de = fundamentals.get("debt_to_equity")
        if de and de > 0:
            de_score = max(0, 100 - (de * 50))
            score += (de_score / 100) * weights["debt_to_equity"]

        # Dividend Yield (direct scoring)
        dy = fundamentals.get("dividend_yield")
        if dy and dy > 0:
            dy_score = min(100, dy * 100)
            score += (dy_score / 100) * weights["dividend_yield"]

        # Current Ratio (direct scoring)
        cr = fundamentals.get("current_ratio")
        if cr and cr > 0:
            cr_score = min(100, cr * 50)
            score += (cr_score / 100) * weights["current_ratio"]

        return min(100, score)


if __name__ == "__main__":
    # Example: Fetch for default stocks
    fetcher = FundamentalIndicators()
    df = fetcher.fetch_all_fundamentals()
    print("\nFundamental Metrics:")
    print(df.to_string())
    fetcher.save_fundamentals_csv(df)

    # Example: Fetch for nifty50
    # fetcher = FundamentalIndicators("nifty50")
    # df = fetcher.fetch_all_fundamentals()
    # fetcher.save_fundamentals_csv(df, "fundamentals_nifty50.csv")
