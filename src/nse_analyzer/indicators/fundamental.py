"""
fundamental.py
Fetch fundamental metrics from yfinance for NSE stocks.
Located at: src/nse_analyzer/indicators/fundamental.py
"""

import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import Optional, Dict
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_FOLDER = PROJECT_ROOT / "data"


class FundamentalIndicators:
    """Fetch and calculate fundamental indicators for NSE stocks."""

    def __init__(self):
        self.stocks = [
            "RELIANCE.NS",
            "TCS.NS",
            "HDFCBANK.NS",
            "INFY.NS",
            "KOTAKBANK.NS",
        ]

    def fetch_fundamental_data(self, ticker: str) -> Optional[Dict]:
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
                "debt_to_assets": (
                    self._get_value(info, "totalDebt")
                    / self._get_value(info, "totalAssets")
                    if self._get_value(info, "totalAssets")
                    else None
                ),
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

        print("\n" + "=" * 60)
        print("FUNDAMENTAL INDICATORS FETCHER")
        print("=" * 60 + "\n")

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
        else:
            return pd.DataFrame()

    def save_fundamentals_csv(
        self, df: pd.DataFrame, filename: str = "fundamentals.csv"
    ) -> bool:
        """
        Save fundamentals DataFrame to CSV.

        Args:
            df: DataFrame with fundamentals
            filename: Output filename

        Returns:
            True if successful, False otherwise
        """
        try:
            csv_path = DATA_FOLDER / filename
            DATA_FOLDER.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False)
            print(f"\n✅ Saved fundamentals to {csv_path}")
            return True
        except Exception as e:
            print(f"❌ Error saving fundamentals: {e}")
            return False

    def get_fundamental_score(
        self, ticker: str, fundamentals: Dict = None
    ) -> Optional[float]:
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
            "pe_ratio": 15,  # Lower PE is better
            "pb_ratio": 10,  # Lower PB is better
            "roe": 15,  # Higher ROE is better
            "roce": 15,  # Higher ROCE is better
            "debt_to_equity": 10,  # Lower D/E is better
            "dividend_yield": 10,  # Higher yield is better
            "current_ratio": 10,  # Higher liquidity is better
        }

        # PE Ratio (inverse scoring)
        pe = fundamentals.get("pe_ratio")
        if pe and pe > 0:
            pe_score = min(100, (30 / pe) * 100)  # 30 is benchmark
            score += (pe_score / 100) * weights["pe_ratio"]

        # ROE (direct scoring)
        roe = fundamentals.get("roe")
        if roe and roe > 0:
            roe_score = min(100, roe * 100)  # 1.0 = 100%
            score += (roe_score / 100) * weights["roe"]

        # ROCE (direct scoring)
        roce = fundamentals.get("roce")
        if roce and roce > 0:
            roce_score = min(100, roce * 100)
            score += (roce_score / 100) * weights["roce"]

        # Debt to Equity (inverse scoring)
        de = fundamentals.get("debt_to_equity")
        if de and de > 0:
            de_score = max(0, 100 - (de * 50))  # Lower is better
            score += (de_score / 100) * weights["debt_to_equity"]

        # Dividend Yield (direct scoring)
        dy = fundamentals.get("dividend_yield")
        if dy and dy > 0:
            dy_score = min(100, dy * 100)
            score += (dy_score / 100) * weights["dividend_yield"]

        # Current Ratio (direct scoring)
        cr = fundamentals.get("current_ratio")
        if cr and cr > 0:
            cr_score = min(100, cr * 50)  # 2.0 is target
            score += (cr_score / 100) * weights["current_ratio"]

        return min(100, score)


if __name__ == "__main__":
    fetcher = FundamentalIndicators()
    df = fetcher.fetch_all_fundamentals()
    print("\nFundamental Metrics:")
    print(df.to_string())
    fetcher.save_fundamentals_csv(df)
