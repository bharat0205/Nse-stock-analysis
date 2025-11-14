"""
fundamental.py
Fetch comprehensive fundamental metrics from yfinance for NSE stocks.
Located at: src/nse_analyzer/indicators/fundamental.py
"""

import sys
import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import Dict, List
import warnings

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from nse_analyzer.utils.config import ConfigManager

warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")


class FundamentalIndicators:
    """Fetch and calculate comprehensive fundamental indicators for NSE stocks."""

    def __init__(self, stock_group: str = "stocks", data_dir: Path = None):
        """
        Initialize with stock list from config OR auto-detect from ohlcv folder.

        Args:
            stock_group: Key in stocks.yaml to load stocks from
            data_dir: Optional custom data directory (for testing)
        """
        self.config_manager = ConfigManager()
        self.data_dir = data_dir or self.config_manager.get_data_dir()
        self.stock_group = stock_group

        # Try to load from config, fallback to auto-detect from ohlcv folder
        try:
            self.stocks = self.config_manager.load_stocks(stock_group)
        except Exception:
            print("⚠️  Could not load from config, auto-detecting from ohlcv folder...")
            self.stocks = self._detect_stocks_from_ohlcv()

        print(f"✅ Loaded {len(self.stocks)} stocks: {self.stocks}")

    def _detect_stocks_from_ohlcv(self) -> List[str]:
        """
        Auto-detect stocks from ../data/ohlcv/ folder.

        Returns:
            List of stock tickers (with .NS suffix added if missing)
        """
        ohlcv_dir = self.data_dir / "ohlcv"
        if not ohlcv_dir.exists():
            print(f"❌ OHLCV directory not found: {ohlcv_dir}")
            return []

        stocks = []
        for csv_file in ohlcv_dir.glob("*.csv"):
            # Extract stock name from filename (e.g., RELIANCE_ohlcv.csv -> RELIANCE.NS)
            stock_name = csv_file.stem.replace("_ohlcv", "")
            if not stock_name.endswith(".NS"):
                stock_name += ".NS"
            stocks.append(stock_name)

        return sorted(stocks)

    def fetch_fundamental_data(self, ticker: str) -> Dict | None:
        """
        Fetch comprehensive fundamental data for a stock from yfinance.

        Args:
            ticker: Stock symbol (e.g., 'RELIANCE.NS')

        Returns:
            Dict with fundamental metrics or None if fetch fails
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Get historical financials for growth calculations
            try:
                financials = stock.financials
                cashflow = stock.cashflow
            except Exception:
                financials = None
                cashflow = None

            # Calculate growth metrics
            eps_growth = self._calculate_eps_growth(stock)
            revenue_growth = self._get_value(info, "revenueGrowth")
            ocf_growth = self._calculate_ocf_growth(cashflow)

            # Extract comprehensive fundamental metrics
            fundamentals = {
                "ticker": ticker,
                "company_name": self._get_value(info, "longName", ticker),
                # ===== GROWTH METRICS =====
                "eps_growth_yoy": eps_growth,
                "revenue_growth_yoy": revenue_growth,
                "ocf_growth_yoy": ocf_growth,
                # ===== PROFITABILITY =====
                "roe": self._get_value(info, "returnOnEquity"),
                "roce": self._get_value(info, "returnOnAssets"),  # Proxy for ROCE
                "net_profit_margin": self._get_value(info, "profitMargins"),
                "operating_margin": self._get_value(info, "operatingMargins"),
                "gross_margin": self._get_value(info, "grossMargins"),
                # ===== VALUATION =====
                "pe_ratio": self._get_value(info, "trailingPE"),
                "forward_pe": self._get_value(info, "forwardPE"),
                "pb_ratio": self._get_value(info, "priceToBook"),
                "peg_ratio": self._get_value(info, "pegRatio"),
                "price_to_sales": self._get_value(info, "priceToSalesTrailing12Months"),
                # ===== DEBT & LIQUIDITY =====
                "debt_to_equity": self._get_value(info, "debtToEquity"),
                "interest_coverage": self._calculate_interest_coverage(financials),
                "current_ratio": self._get_value(info, "currentRatio"),
                "quick_ratio": self._get_value(info, "quickRatio"),
                # ===== DIVIDEND METRICS =====
                "dividend_yield": self._get_value(info, "dividendYield"),
                "dividend_payout_ratio": self._get_value(info, "payoutRatio"),
                "dividend_per_share": self._get_value(info, "dividendRate"),
                "ex_dividend_date": self._get_value(info, "exDividendDate"),
                # ===== CASH FLOW =====
                "operating_cash_flow": self._get_value(info, "operatingCashflow"),
                "free_cash_flow": self._get_value(info, "freeCashflow"),
                "fcf_per_share": self._calculate_fcf_per_share(info),
                # ===== FINANCIAL HEALTH =====
                "earnings_per_share": self._get_value(info, "trailingEps"),
                "forward_eps": self._get_value(info, "forwardEps"),
                "book_value_per_share": self._get_value(info, "bookValue"),
                "market_cap": self._get_value(info, "marketCap"),
                "enterprise_value": self._get_value(info, "enterpriseValue"),
                # ===== ADDITIONAL METRICS =====
                "beta": self._get_value(info, "beta"),
                "52_week_high": self._get_value(info, "fiftyTwoWeekHigh"),
                "52_week_low": self._get_value(info, "fiftyTwoWeekLow"),
            }

            return fundamentals

        except Exception as e:
            print(f"❌ Error fetching fundamentals for {ticker}: {e}")
            return None

    def _calculate_eps_growth(self, stock) -> float | None:
        """Calculate YoY EPS growth from historical data."""
        try:
            financials = stock.financials
            if financials is None or financials.empty:
                return None

            # Get Net Income for last 2 years
            if "Net Income" in financials.index:
                net_income = financials.loc["Net Income"]
                if len(net_income) >= 2:
                    latest = net_income.iloc[0]
                    previous = net_income.iloc[1]
                    if previous != 0 and pd.notna(latest) and pd.notna(previous):
                        growth = ((latest - previous) / abs(previous)) * 100
                        return round(growth, 2)
            return None
        except Exception:
            return None

    def _calculate_ocf_growth(self, cashflow) -> float | None:
        """Calculate YoY Operating Cash Flow growth."""
        try:
            if cashflow is None or cashflow.empty:
                return None

            if "Operating Cash Flow" in cashflow.index:
                ocf = cashflow.loc["Operating Cash Flow"]
                if len(ocf) >= 2:
                    latest = ocf.iloc[0]
                    previous = ocf.iloc[1]
                    if previous != 0 and pd.notna(latest) and pd.notna(previous):
                        growth = ((latest - previous) / abs(previous)) * 100
                        return round(growth, 2)
            return None
        except Exception:
            return None

    def _calculate_interest_coverage(self, financials) -> float | None:
        """Calculate Interest Coverage Ratio (EBIT / Interest Expense)."""
        try:
            if financials is None or financials.empty:
                return None

            ebit = None
            interest = None

            if "EBIT" in financials.index:
                ebit = financials.loc["EBIT"].iloc[0]
            if "Interest Expense" in financials.index:
                interest = abs(financials.loc["Interest Expense"].iloc[0])

            if ebit and interest and interest != 0:
                return round(ebit / interest, 2)
            return None
        except Exception:
            return None

    def _calculate_fcf_per_share(self, info: Dict) -> float | None:
        """Calculate Free Cash Flow per Share."""
        try:
            fcf = info.get("freeCashflow")
            shares = info.get("sharesOutstanding")

            if fcf and shares and shares != 0:
                return round(fcf / shares, 2)
            return None
        except Exception:
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

        print(f"\n{'='*70}")
        print(f"FUNDAMENTAL INDICATORS FETCHER - {self.stock_group.upper()}")
        print(f"{'='*70}\n")

        for idx, ticker in enumerate(self.stocks, 1):
            print(f"[{idx}/{len(self.stocks)}] Fetching fundamentals for {ticker}...")

            fundamentals = self.fetch_fundamental_data(ticker)

            if fundamentals:
                results.append(fundamentals)
                print(f"  ✅ Success")
            else:
                print(f"  ❌ Failed")

        if results:
            df = pd.DataFrame(results)
            print(f"\n✅ Successfully fetched fundamentals for {len(results)} stocks")
            return df

        print("\n❌ No fundamental data fetched")
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
            print(f"   Columns: {list(df.columns)}")
            print(f"   Rows: {len(df)}")

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
            "pe_ratio": 12,
            "pb_ratio": 8,
            "roe": 15,
            "roce": 12,
            "debt_to_equity": 10,
            "dividend_yield": 8,
            "current_ratio": 8,
            "revenue_growth_yoy": 12,
            "net_profit_margin": 10,
            "ocf_growth_yoy": 5,
        }

        # PE Ratio (inverse scoring: lower is better, up to threshold)
        pe = fundamentals.get("pe_ratio")
        if pe and pe > 0:
            pe_score = min(100, (25 / pe) * 100) if pe > 0 else 0
            score += (pe_score / 100) * weights["pe_ratio"]

        # PB Ratio (inverse scoring)
        pb = fundamentals.get("pb_ratio")
        if pb and pb > 0:
            pb_score = min(100, (3 / pb) * 100)
            score += (pb_score / 100) * weights["pb_ratio"]

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
        if de is not None and de >= 0:
            de_score = max(0, 100 - (de * 20))
            score += (de_score / 100) * weights["debt_to_equity"]

        # Dividend Yield (direct scoring)
        dy = fundamentals.get("dividend_yield")
        if dy and dy > 0:
            dy_score = min(100, dy * 2000)  # Scale up small percentages
            score += (dy_score / 100) * weights["dividend_yield"]

        # Current Ratio (optimal around 1.5-2)
        cr = fundamentals.get("current_ratio")
        if cr and cr > 0:
            cr_score = min(100, cr * 50) if cr < 3 else max(0, 100 - ((cr - 3) * 20))
            score += (cr_score / 100) * weights["current_ratio"]

        # Revenue Growth (direct scoring)
        rg = fundamentals.get("revenue_growth_yoy")
        if rg:
            rg_score = min(100, max(0, rg * 5))  # Scale percentage
            score += (rg_score / 100) * weights["revenue_growth_yoy"]

        # Net Profit Margin (direct scoring)
        npm = fundamentals.get("net_profit_margin")
        if npm and npm > 0:
            npm_score = min(100, npm * 100)
            score += (npm_score / 100) * weights["net_profit_margin"]

        # OCF Growth (direct scoring)
        ocf_g = fundamentals.get("ocf_growth_yoy")
        if ocf_g:
            ocf_score = min(100, max(0, ocf_g * 2))
            score += (ocf_score / 100) * weights["ocf_growth_yoy"]

        return round(min(100, score), 2)


if __name__ == "__main__":
    # Example: Fetch for default stocks (auto-detect from ohlcv or config)
    fetcher = FundamentalIndicators()
    df = fetcher.fetch_all_fundamentals()

    if not df.empty:
        print("\n" + "=" * 70)
        print("FUNDAMENTAL METRICS SUMMARY")
        print("=" * 70)
        print(df[["ticker", "pe_ratio", "pb_ratio", "roe", "debt_to_equity"]].to_string())

        fetcher.save_fundamentals_csv(df)

        # Calculate scores
        print("\n" + "=" * 70)
        print("FUNDAMENTAL SCORES (0-100)")
        print("=" * 70)
        for _, row in df.iterrows():
            score = fetcher.get_fundamental_score(row["ticker"], row.to_dict())
            print(f"{row['ticker']}: {score}")
