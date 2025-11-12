"""
test_technical_indicators.py
Unit tests for all technical indicators using pandas-ta.
Run with: pytest tests/test_technical_indicators.py -v
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nse_analyzer.indicators.technical.rsi import RSI, calculate_rsi
from nse_analyzer.indicators.technical.macd import MACD, calculate_macd
from nse_analyzer.indicators.technical.supertrend import (
    Supertrend,
    calculate_supertrend,
)
from nse_analyzer.indicators.technical.bollinger_bands import (
    BollingerBands,
    calculate_bollinger_bands,
)
from nse_analyzer.indicators.technical.calculator import (
    TechnicalIndicatorCalculator,
)


class TestTechnicalIndicators:
    """Test suite for technical indicators."""

    @pytest.fixture
    def sample_ohlcv_df(self):
        """Create sample OHLCV DataFrame with realistic data."""
        dates = pd.date_range("2025-01-01", periods=100)
        import numpy as np

        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)
        high = close + np.random.rand(100) * 2
        low = close - np.random.rand(100) * 2

        df = pd.DataFrame(
            {
                "Date": dates,
                "Open": close + np.random.randn(100) * 0.3,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": np.random.randint(1000000, 5000000, 100),
            }
        )
        return df

    # ==================== RSI Tests ====================

    def test_rsi_calculation(self, sample_ohlcv_df):
        """Test RSI calculation with pandas-ta."""
        df = calculate_rsi(sample_ohlcv_df, period=14)
        assert "RSI_14" in df.columns, "RSI_14 column should exist"
        assert len(df) == len(sample_ohlcv_df), "Should maintain row count"
        assert df["RSI_14"].notna().sum() > 0, "Should have RSI values"
        # RSI should be between 0-100
        rsi_values = df["RSI_14"].dropna()
        assert (rsi_values >= 0).all() and (
            rsi_values <= 100
        ).all(), "RSI should be between 0-100"

    def test_rsi_class(self, sample_ohlcv_df):
        """Test RSI class initialization and calculation."""
        rsi = RSI(period=14)
        df = rsi.calculate(sample_ohlcv_df)
        assert "RSI_14" in df.columns
        assert len(df) == len(sample_ohlcv_df)

    def test_rsi_custom_period(self, sample_ohlcv_df):
        """Test RSI with custom period."""
        df = calculate_rsi(sample_ohlcv_df, period=21)
        assert "RSI_21" in df.columns, "Should use custom period in column name"

    # ==================== MACD Tests ====================

    def test_macd_calculation(self, sample_ohlcv_df):
        """Test MACD calculation with pandas-ta."""
        df = calculate_macd(sample_ohlcv_df, fast=12, slow=26, signal=9)
        assert "MACD" in df.columns, "MACD column should exist"
        assert "MACD_Signal" in df.columns, "Signal line should exist"
        assert "MACD_Histogram" in df.columns, "Histogram should exist"
        assert len(df) == len(sample_ohlcv_df), "Should maintain row count"

    def test_macd_class(self, sample_ohlcv_df):
        """Test MACD class."""
        macd = MACD(fast=12, slow=26, signal=9)
        df = macd.calculate(sample_ohlcv_df)
        assert "MACD" in df.columns
        assert len(df) == len(sample_ohlcv_df)

    def test_macd_histogram_sign(self, sample_ohlcv_df):
        """Test that MACD histogram is difference of MACD and Signal."""
        df = calculate_macd(sample_ohlcv_df, fast=12, slow=26, signal=9)
        # Histogram should be approximately MACD - Signal
        diff = abs((df["MACD"] - df["MACD_Signal"]) - df["MACD_Histogram"])
        assert diff.max() < 1e-6, "Histogram should equal MACD - Signal"

    # ==================== Supertrend Tests ====================

    def test_supertrend_calculation(self, sample_ohlcv_df):
        """Test Supertrend calculation with pandas-ta."""
        df = calculate_supertrend(sample_ohlcv_df, period=10, multiplier=3.0)
        assert "Supertrend" in df.columns, "Supertrend column should exist"
        assert "Supertrend_Direction" in df.columns, "Direction column should exist"
        assert len(df) == len(sample_ohlcv_df), "Should maintain row count"

    def test_supertrend_class(self, sample_ohlcv_df):
        """Test Supertrend class."""
        st = Supertrend(period=10, multiplier=3.0)
        df = st.calculate(sample_ohlcv_df)
        assert "Supertrend" in df.columns
        assert len(df) == len(sample_ohlcv_df)

    def test_supertrend_custom_params(self, sample_ohlcv_df):
        """Test Supertrend with custom parameters."""
        df = calculate_supertrend(sample_ohlcv_df, period=7, multiplier=2.0)
        assert "Supertrend" in df.columns

    # ==================== Bollinger Bands Tests ====================

    def test_bollinger_bands_calculation(self, sample_ohlcv_df):
        """Test Bollinger Bands calculation with pandas-ta."""
        df = calculate_bollinger_bands(sample_ohlcv_df, period=20, std_dev=2.0)
        assert "BB_Upper_20" in df.columns, "Upper band should exist"
        assert "BB_Middle_20" in df.columns, "Middle band should exist"
        assert "BB_Lower_20" in df.columns, "Lower band should exist"
        assert "BB_Bandwidth_20" in df.columns, "Bandwidth should exist"
        assert "BB_Percent_B_20" in df.columns, "Percent B should exist"
        assert len(df) == len(sample_ohlcv_df), "Should maintain row count"

    def test_bollinger_bands_class(self, sample_ohlcv_df):
        """Test Bollinger Bands class."""
        bb = BollingerBands(period=20, std_dev=2.0)
        df = bb.calculate(sample_ohlcv_df)
        assert "BB_Upper_20" in df.columns
        assert len(df) == len(sample_ohlcv_df)

    def test_bollinger_bands_relationship(self, sample_ohlcv_df):
        """Test relationship between BB bands (Upper > Middle > Lower)."""
        df = calculate_bollinger_bands(sample_ohlcv_df, period=20, std_dev=2.0)
        # Upper band should be > Middle > Lower (in most cases)
        valid_rows = df[
            df[["BB_Upper_20", "BB_Middle_20", "BB_Lower_20"]].notna().all(axis=1)
        ]
        assert (
            valid_rows["BB_Upper_20"] > valid_rows["BB_Middle_20"]
        ).all(), "Upper band should > Middle"
        assert (
            valid_rows["BB_Middle_20"] > valid_rows["BB_Lower_20"]
        ).all(), "Middle should > Lower"

    # ==================== Integration Tests ====================

    def test_all_indicators_together(self, sample_ohlcv_df):
        """Test calculating all indicators on same DataFrame."""
        df = sample_ohlcv_df.copy()
        df = calculate_rsi(df)
        df = calculate_macd(df)
        df = calculate_supertrend(df)
        df = calculate_bollinger_bands(df)

        # Verify all columns exist
        assert "RSI_14" in df.columns
        assert "MACD" in df.columns
        assert "Supertrend" in df.columns
        assert "BB_Upper_20" in df.columns

        # Verify data integrity
        assert len(df) == len(sample_ohlcv_df)
        assert df["Date"].is_monotonic_increasing
        # Original OHLCV should be unchanged
        assert (df["Close"] == sample_ohlcv_df["Close"]).all()

    def test_date_alignment(self, sample_ohlcv_df):
        """Test that dates align correctly with indicators."""
        df = sample_ohlcv_df.copy()
        df = calculate_rsi(df)
        df = calculate_macd(df)

        # Date column should remain unchanged
        assert (df["Date"] == sample_ohlcv_df["Date"]).all()

    def test_no_data_loss(self, sample_ohlcv_df):
        """Test that OHLCV data is not lost when adding indicators."""
        df = sample_ohlcv_df.copy()
        original_ohlcv = df[["Open", "High", "Low", "Close", "Volume"]]

        df = calculate_rsi(df)
        df = calculate_macd(df)
        df = calculate_supertrend(df)
        df = calculate_bollinger_bands(df)

        new_ohlcv = df[["Open", "High", "Low", "Close", "Volume"]]
        pd.testing.assert_frame_equal(original_ohlcv, new_ohlcv)

    # ==================== Validation Tests ====================

    def test_invalid_dataframe_missing_ohlcv(self):
        """Test with invalid DataFrame missing OHLCV columns."""
        invalid_df = pd.DataFrame({"Col1": [1, 2, 3], "Col2": [4, 5, 6]})

        with pytest.raises(ValueError):
            calculate_rsi(invalid_df)

        with pytest.raises(ValueError):
            calculate_macd(invalid_df)

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        empty_df = pd.DataFrame(
            {"Date": [], "Open": [], "High": [], "Low": [], "Close": [], "Volume": []}
        )

        result_rsi = calculate_rsi(empty_df)
        assert len(result_rsi) == 0

    # ==================== Calculator Tests ====================

    def test_calculator_initialization(self):
        """Test TechnicalIndicatorCalculator initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            calc = TechnicalIndicatorCalculator(
                stock_group="stocks", data_dir=Path(tmpdir)
            )
            assert calc.indicators_dir.exists()
            assert calc.stock_group == "stocks"

    def test_calculator_ohlcv_loading(self, sample_ohlcv_df):
        """Test loading OHLCV data from CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            ohlcv_dir = tmpdir_path / "ohlcv"
            ohlcv_dir.mkdir()

            # Save sample data
            csv_path = ohlcv_dir / "RELIANCE.csv"
            sample_ohlcv_df.to_csv(csv_path, index=False)

            # Create calculator and load
            calc = TechnicalIndicatorCalculator(data_dir=tmpdir_path)
            loaded_df = calc.load_ohlcv_data("RELIANCE.NS")

            assert loaded_df is not None
            assert len(loaded_df) == len(sample_ohlcv_df)

    def test_calculator_missing_file(self):
        """Test loading non-existent OHLCV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            calc = TechnicalIndicatorCalculator(data_dir=Path(tmpdir))
            result = calc.load_ohlcv_data("NONEXISTENT.NS")
            assert result is None

    def test_calculator_all_indicators(self, sample_ohlcv_df):
        """Test calculator calculate_all_indicators method."""
        calc = TechnicalIndicatorCalculator()
        result_df = calc.calculate_all_indicators(sample_ohlcv_df)

        assert "RSI_14" in result_df.columns
        assert "MACD" in result_df.columns
        assert "Supertrend" in result_df.columns
        assert "BB_Upper_20" in result_df.columns

    # ==================== Scalability Tests ====================

    def test_scalability_different_sizes(self, sample_ohlcv_df):
        """Test that indicators work with different data sizes."""
        for size in [30, 50, 100, 200]:
            df = sample_ohlcv_df.iloc[:size].reset_index(drop=True)
            if len(df) >= 14:
                result = calculate_rsi(df)
                assert len(result) == len(df)

            # MACD slow period + signal period = 26 + 9 = 35 minimum rows needed
            if len(df) >= 35:
                result = calculate_macd(df)
                assert len(result) == len(df)

    def test_scalability_multiple_stocks(self):
        """Test calculator can handle multiple stocks in config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            ohlcv_dir = tmpdir_path / "ohlcv"
            ohlcv_dir.mkdir()

            # Create sample data for 3 stocks
            dates = pd.date_range("2025-01-01", periods=50)
            for stock in ["RELIANCE", "TCS", "INFY"]:
                df = pd.DataFrame(
                    {
                        "Date": dates,
                        "Open": [100 + i * 0.5 for i in range(50)],
                        "High": [105 + i * 0.5 for i in range(50)],
                        "Low": [95 + i * 0.5 for i in range(50)],
                        "Close": [102 + i * 0.5 for i in range(50)],
                        "Volume": [1000000 + i * 10000 for i in range(50)],
                    }
                )
                df.to_csv(ohlcv_dir / f"{stock}.csv", index=False)

            calc = TechnicalIndicatorCalculator(data_dir=tmpdir_path)
            for stock in ["RELIANCE.NS", "TCS.NS", "INFY.NS"]:
                result = calc.load_ohlcv_data(stock)
                assert result is not None
                result_with_ind = calc.calculate_all_indicators(result)
                assert "RSI_14" in result_with_ind.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
