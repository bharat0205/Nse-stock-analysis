"""
test_fundamental_indicators.py
Unit tests for FundamentalIndicators
Run with: pytest tests/test_fundamental_indicators.py -v
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nse_analyzer.indicators.fundamental import FundamentalIndicators
from nse_analyzer.utils.config import ConfigManager


class TestFundamentalIndicators:
    """Test suite for FundamentalIndicators."""

    @pytest.fixture
    def fetcher(self):
        """Create a FundamentalIndicators instance."""
        fetcher_instance = FundamentalIndicators(stock_group="stocks")
        # Use only 2 stocks for testing
        fetcher_instance.stocks = ["RELIANCE.NS", "TCS.NS"]
        return fetcher_instance

    def test_fetcher_initialization(self, fetcher):
        """Test that fetcher initializes with stocks from config."""
        # Check that stocks list is not empty (scalable)
        assert len(fetcher.stocks) > 0, "Should have at least 1 stock"

        # Check that stock symbols are valid format
        for stock in fetcher.stocks:
            assert ".NS" in stock, f"Stock {stock} should have .NS suffix"
            assert isinstance(stock, str), f"Stock should be string, got {type(stock)}"

    def test_config_stocks_loaded(self):
        """Test that stocks are loaded from config."""
        stocks = ConfigManager.load_stocks("stocks")
        assert len(stocks) > 0, "Config should have stocks defined"
        assert all(".NS" in s for s in stocks), "All stocks should have .NS suffix"

    def test_fetch_fundamental_data_valid_stock(self, fetcher):
        """Test fetching fundamentals for a valid stock."""
        fundamentals = fetcher.fetch_fundamental_data("RELIANCE.NS")
        assert fundamentals is not None, "Should return dict"
        assert isinstance(fundamentals, dict), "Should be dictionary"
        assert "ticker" in fundamentals, "Should have ticker"
        assert "pe_ratio" in fundamentals, "Should have PE ratio"
        assert "roe" in fundamentals, "Should have ROE"

    def test_fetch_fundamental_data_invalid_stock(self, fetcher):
        """Test fetching for invalid stock."""
        fundamentals = fetcher.fetch_fundamental_data("INVALID.NS")
        # Should handle gracefully
        assert fundamentals is None or isinstance(
            fundamentals, dict
        ), "Should return None or dict"

    def test_fetch_all_fundamentals(self, fetcher):
        """Test fetching fundamentals for all stocks."""
        df = fetcher.fetch_all_fundamentals()
        assert isinstance(df, pd.DataFrame), "Should return DataFrame"
        if len(df) > 0:
            assert "ticker" in df.columns, "Should have ticker column"
            assert "pe_ratio" in df.columns, "Should have PE ratio column"
            # Check number of rows matches fetcher.stocks
            assert len(df) <= len(
                fetcher.stocks
            ), "Should have at most as many rows as stocks"

    def test_get_value_method(self, fetcher):
        """Test the _get_value helper method."""
        test_dict = {"key1": 100, "key2": None}
        assert fetcher._get_value(test_dict, "key1") == 100
        assert fetcher._get_value(test_dict, "key2") is None
        assert fetcher._get_value(test_dict, "key3", "default") == "default"

    def test_fundamental_score_calculation(self, fetcher):
        """Test fundamental score calculation."""
        test_fundamentals = {
            "pe_ratio": 20,
            "pb_ratio": 2.5,
            "roe": 0.15,
            "roce": 0.14,
            "debt_to_equity": 0.5,
            "dividend_yield": 0.02,
            "current_ratio": 1.5,
        }
        score = fetcher.get_fundamental_score("TEST.NS", test_fundamentals)
        assert score is not None, "Should return score"
        assert 0 <= score <= 100, "Score should be between 0-100"

    def test_save_fundamentals_csv(self, fetcher):
        """Test saving fundamentals to CSV."""
        df = pd.DataFrame(
            {
                "ticker": ["RELIANCE.NS", "TCS.NS"],
                "pe_ratio": [25.5, 30.2],
                "roe": [0.15, 0.18],
                "dividend_yield": [0.025, 0.015],
            }
        )
        success = fetcher.save_fundamentals_csv(df, "test_fundamentals.csv")
        assert success, "Save should succeed"

    def test_scalability_dynamic_stock_list(self):
        """Test that system scales with different stock group sizes."""
        # Test with different stock groups (when available)
        stock_groups = ConfigManager.list_stock_groups()

        for group in stock_groups:
            fetcher = FundamentalIndicators(stock_group=group)
            assert len(fetcher.stocks) > 0, f"Stock group '{group}' should have stocks"
            # Verify all are strings
            assert all(
                isinstance(s, str) for s in fetcher.stocks
            ), f"All stocks in '{group}' should be strings"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
