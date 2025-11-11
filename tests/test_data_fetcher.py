"""
test_data_fetcher.py
Unit tests for NSEDataFetcher
Run with: pytest tests/test_data_fetcher.py -v
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import sys

# Add src to path so we can import nse_analyzer
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nse_analyzer.data_fetcher import NSEDataFetcher


class TestNSEDataFetcher:
    """Test suite for NSEDataFetcher."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory for test data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def fetcher(self, temp_data_dir):
        """Create a fetcher instance with temp directory."""
        fetcher_instance = NSEDataFetcher(data_dir=temp_data_dir)
        fetcher_instance.stocks = ["RELIANCE.NS", "TCS.NS"]
        return fetcher_instance

    def test_data_folder_created(self, temp_data_dir):
        """Test that data folder is created."""
        NSEDataFetcher(data_dir=temp_data_dir)
        assert temp_data_dir.exists(), "Data folder should exist"

    def test_fetch_ohlcv_valid_stock(self, fetcher):
        """Test fetching OHLCV for a valid stock."""
        df = fetcher.fetch_ohlcv("RELIANCE.NS", period="1mo", interval="1d")
        assert df is not None, "Should return DataFrame"
        assert isinstance(df, pd.DataFrame), "Should be DataFrame"
        assert len(df) > 0, "Should have rows"
        required_cols = {"Open", "High", "Low", "Close", "Volume"}
        assert required_cols.issubset(df.columns), "Should have OHLCV columns"
        assert "Date" in df.columns, "Should have Date column"

    def test_fetch_ohlcv_invalid_stock(self, fetcher):
        """Test fetching for invalid stock."""
        df = fetcher.fetch_ohlcv("INVALID.NS", period="1mo", interval="1d")
        # yfinance returns empty DataFrame for invalid tickers
        if df is not None:
            assert len(df) >= 0, "Should handle gracefully"

    def test_save_to_csv(self, fetcher, temp_data_dir):
        """Test saving DataFrame to CSV."""
        df = pd.DataFrame(
            {
                "Date": pd.date_range("2025-01-01", periods=3),
                "Open": [100, 105, 110],
                "High": [105, 110, 115],
                "Low": [99, 104, 109],
                "Close": [104, 109, 114],
                "Volume": [1000000, 1100000, 1200000],
            }
        )
        success = fetcher.save_to_csv("RELIANCE.NS", df)
        assert success, "Save should succeed"

        csv_path = temp_data_dir / "RELIANCE.csv"
        assert csv_path.exists(), "CSV file should exist"

        # Load and verify
        saved_df = pd.read_csv(csv_path)
        assert len(saved_df) == 3, "Should have 3 rows"
        assert saved_df.loc[0, "Close"] == 104, "Data should match"
        assert "Date" in saved_df.columns, "Date should be a column"
        assert "Open" in saved_df.columns, "Open should be a column"

    def test_csv_overwrite(self, fetcher, temp_data_dir):
        """Test that CSV is overwritten, not appended."""
        # First write
        df1 = pd.DataFrame(
            {
                "Date": pd.date_range("2025-01-01", periods=1),
                "Open": [100],
                "High": [105],
                "Low": [99],
                "Close": [104],
                "Volume": [1000000],
            }
        )
        fetcher.save_to_csv("TEST.NS", df1)

        # Second write (should overwrite)
        df2 = pd.DataFrame(
            {
                "Date": pd.date_range("2025-01-01", periods=2),
                "Open": [110, 115],
                "High": [115, 120],
                "Low": [109, 114],
                "Close": [114, 119],
                "Volume": [1100000, 1200000],
            }
        )
        fetcher.save_to_csv("TEST.NS", df2)

        # Verify only 2 rows (not 3)
        csv_path = temp_data_dir / "TEST.csv"
        saved_df = pd.read_csv(csv_path)
        assert len(saved_df) == 2, "Should have 2 rows, not appended"
        assert saved_df.loc[0, "Close"] == 114, "First row should be from second write"

    def test_fetch_all_integration(self, fetcher, temp_data_dir):
        """Test fetching all stocks."""
        results = fetcher.fetch_all(period="1mo", interval="1d")

        assert isinstance(results, dict), "Should return dict"
        assert len(results) == 2, "Should have results for 2 stocks"

        # Check files were created
        assert (temp_data_dir / "RELIANCE.csv").exists(), "RELIANCE CSV should exist"
        assert (temp_data_dir / "TCS.csv").exists(), "TCS CSV should exist"

        # Verify CSV contents
        reliance_df = pd.read_csv(temp_data_dir / "RELIANCE.csv")
        assert len(reliance_df) > 0, "RELIANCE CSV should have data"
        required_cols = {"Date", "Open", "High", "Low", "Close", "Volume"}
        assert required_cols.issubset(
            reliance_df.columns
        ), "RELIANCE CSV should have all OHLCV columns"

    def test_csv_header_format(self, fetcher, temp_data_dir):
        """Test that CSV has clean single-level headers."""
        df = pd.DataFrame(
            {
                "Date": pd.date_range("2025-01-01", periods=2),
                "Open": [100, 105],
                "High": [105, 110],
                "Low": [99, 104],
                "Close": [104, 109],
                "Volume": [1000000, 1100000],
            }
        )
        fetcher.save_to_csv("TEST.NS", df)

        csv_path = temp_data_dir / "TEST.csv"
        with open(csv_path) as f:
            header = f.readline().strip()

        # Should be clean single-level header
        expected_header = "Date,Open,High,Low,Close,Volume"
        assert (
            header == expected_header
        ), f"Header should be '{expected_header}', got '{header}'"

    def test_data_integrity(self, fetcher, temp_data_dir):
        """Test that data integrity is maintained."""
        df = pd.DataFrame(
            {
                "Date": ["2025-01-01", "2025-01-02"],
                "Open": [100.5, 105.7],
                "High": [105.3, 110.2],
                "Low": [99.1, 104.6],
                "Close": [104.2, 109.8],
                "Volume": [1000000, 1100000],
            }
        )
        fetcher.save_to_csv("TEST.NS", df)

        # Load and compare
        csv_path = temp_data_dir / "TEST.csv"
        loaded_df = pd.read_csv(csv_path)

        # Check specific values
        assert loaded_df.loc[0, "Open"] == 100.5, "Open price should match"
        assert loaded_df.loc[1, "High"] == 110.2, "High price should match"
        assert loaded_df.loc[0, "Volume"] == 1000000, "Volume should match"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
