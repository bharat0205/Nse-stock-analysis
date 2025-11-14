"""
test_fundamental_indicators.py
Test fundamental indicators fetcher

Run with: pytest tests/test_fundamental_indicators.py -v
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nse_analyzer.indicators.fundamental import FundamentalIndicators


class TestFundamentalIndicators:
    """Test suite for Fundamental Indicators"""

    @pytest.fixture
    def fetcher(self):
        """Create a FundamentalIndicators instance"""
        return FundamentalIndicators()

    def test_initialization(self, fetcher):
        """Test that fetcher initializes correctly"""
        assert fetcher is not None
        assert len(fetcher.stocks) > 0
        print(f"\n✅ Initialized with {len(fetcher.stocks)} stocks")

    def test_fetch_single_stock(self, fetcher):
        """Test fetching fundamentals for a single stock"""
        ticker = "RELIANCE.NS"
        fundamentals = fetcher.fetch_fundamental_data(ticker)

        assert fundamentals is not None
        assert fundamentals["ticker"] == ticker

        # Check key metrics exist
        required_fields = [
            "eps_growth_yoy",
            "revenue_growth_yoy",
            "roe",
            "roce",
            "net_profit_margin",
            "pe_ratio",
            "pb_ratio",
            "debt_to_equity",
            "ocf_growth_yoy",
            "interest_coverage",
            "dividend_yield",
        ]

        for field in required_fields:
            assert field in fundamentals, f"Missing field: {field}"

        print(f"\n✅ {ticker} fundamentals:")
        for field in required_fields:
            print(f"   {field}: {fundamentals[field]}")

    def test_fetch_all_stocks(self, fetcher):
        """Test fetching fundamentals for all stocks"""
        df = fetcher.fetch_all_fundamentals()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "ticker" in df.columns

        print(f"\n✅ Fetched fundamentals for {len(df)} stocks")
        print(f"   Columns: {list(df.columns)[:10]}...")

    def test_fundamental_score(self, fetcher):
        """Test fundamental score calculation"""
        ticker = "TCS.NS"
        score = fetcher.get_fundamental_score(ticker)

        assert score is not None
        assert 0 <= score <= 100

        print(f"\n✅ Fundamental score for {ticker}: {score}/100")

    def test_save_to_csv(self, fetcher, tmp_path):
        """Test saving fundamentals to CSV"""
        df = fetcher.fetch_all_fundamentals()

        # Use temporary directory for testing
        fetcher.data_dir = tmp_path
        success = fetcher.save_fundamentals_csv(df, filename="test_fundamentals.csv")

        assert success is True

        # Verify file exists
        csv_path = tmp_path / "fundamentals" / "test_fundamentals.csv"
        assert csv_path.exists()

        # Verify file content
        loaded_df = pd.read_csv(csv_path)
        assert len(loaded_df) == len(df)

        print(f"\n✅ Saved and verified CSV at {csv_path}")

    def test_auto_detect_from_ohlcv(self, tmp_path):
        """Test auto-detection of stocks from ohlcv folder"""
        # Create fake ohlcv directory
        ohlcv_dir = tmp_path / "ohlcv"
        ohlcv_dir.mkdir()

        # Create dummy CSV files
        (ohlcv_dir / "RELIANCE_ohlcv.csv").touch()
        (ohlcv_dir / "TCS_ohlcv.csv").touch()

        # Initialize fetcher with custom data dir and invalid stock_group
        # This forces fallback to auto-detect
        fetcher = FundamentalIndicators(stock_group="nonexistent", data_dir=tmp_path)

        assert len(fetcher.stocks) == 2
        assert "RELIANCE.NS" in fetcher.stocks
        assert "TCS.NS" in fetcher.stocks

        print(f"\n✅ Auto-detected stocks: {fetcher.stocks}")



if __name__ == "__main__":
    # Run tests manually
    print("=" * 70)
    print("FUNDAMENTAL INDICATORS TEST SUITE")
    print("=" * 70)

    fetcher = FundamentalIndicators()

    print("\n1️⃣  Testing single stock fetch...")
    fundamentals = fetcher.fetch_fundamental_data("RELIANCE.NS")
    print(f"   Result: {fundamentals is not None}")

    print("\n2️⃣  Testing all stocks fetch...")
    df = fetcher.fetch_all_fundamentals()
    print(f"   Fetched: {len(df)} stocks")

    print("\n3️⃣  Testing fundamental score...")
    score = fetcher.get_fundamental_score("TCS.NS")
    print(f"   Score: {score}/100")

    print("\n4️⃣  Testing CSV save...")
    success = fetcher.save_fundamentals_csv(df)
    print(f"   Success: {success}")

    print("\n✅ All manual tests complete!")
