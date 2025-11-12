"""
bollinger_bands.py
Bollinger Bands indicator using pandas-ta.
Located at: src/nse_analyzer/indicators/technical/bollinger_bands.py
"""

import pandas as pd
import pandas_ta as ta
from pathlib import Path
from nse_analyzer.indicators.technical.base import TechnicalIndicator


class BollingerBands(TechnicalIndicator):
    """Calculate Bollinger Bands indicator."""

    def __init__(
        self, period: int = 20, std_dev: float = 2.0, data_dir: Path | None = None
    ):
        """
        Initialize Bollinger Bands calculator.

        Args:
            period: MA period (default: 20)
            std_dev: Standard deviation multiplier (default: 2.0)
            data_dir: Optional custom data directory
        """
        super().__init__(data_dir)
        self.period = period
        self.std_dev = std_dev

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Bollinger Bands using pandas-ta.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with BB columns added
        """
        if not self.validate_ohlcv(df):
            raise ValueError("DataFrame must contain OHLCV columns")

        df = df.copy()

        # Use pandas-ta to calculate Bollinger Bands
        bb_result = ta.bbands(df["Close"], length=self.period, std=self.std_dev)

        # Get the actual column names from pandas-ta result
        # pandas-ta returns: BBL_20_2.0_2.0, BBM_20_2.0_2.0,
        #  BBU_20_2.0_2.0, BBB_20_2.0_2.0, BBP_20_2.0_2.0
        # Find the correct column names dynamically
        bb_columns = bb_result.columns.tolist()

        # Extract columns by their suffixes
        lower_col = [col for col in bb_columns if col.startswith("BBL_")][0]
        middle_col = [col for col in bb_columns if col.startswith("BBM_")][0]
        upper_col = [col for col in bb_columns if col.startswith("BBU_")][0]
        bandwidth_col = [col for col in bb_columns if col.startswith("BBB_")][0]
        percent_b_col = [col for col in bb_columns if col.startswith("BBP_")][0]

        # Assign to cleaner column names
        df[f"BB_Lower_{self.period}"] = bb_result[lower_col]
        df[f"BB_Middle_{self.period}"] = bb_result[middle_col]
        df[f"BB_Upper_{self.period}"] = bb_result[upper_col]
        df[f"BB_Bandwidth_{self.period}"] = bb_result[bandwidth_col]
        df[f"BB_Percent_B_{self.period}"] = bb_result[percent_b_col]

        return df


def calculate_bollinger_bands(
    df: pd.DataFrame, period: int = 20, std_dev: float = 2.0
) -> pd.DataFrame:
    """
    Standalone function to calculate Bollinger Bands.

    Args:
        df: DataFrame with OHLCV data
        period: MA period
        std_dev: Standard deviation multiplier

    Returns:
        DataFrame with Bollinger Bands columns added
    """
    bb_calculator = BollingerBands(period=period, std_dev=std_dev)
    return bb_calculator.calculate(df)
