"""
rsi.py
Relative Strength Index (RSI) indicator using pandas-ta.
Located at: src/nse_analyzer/indicators/technical/rsi.py
"""

import pandas as pd
import pandas_ta as ta
from pathlib import Path
from nse_analyzer.indicators.technical.base import TechnicalIndicator


class RSI(TechnicalIndicator):
    """Calculate Relative Strength Index (RSI) indicator."""

    def __init__(self, period: int = 14, data_dir: Path | None = None):
        """
        Initialize RSI calculator.

        Args:
            period: RSI period (default: 14)
            data_dir: Optional custom data directory
        """
        super().__init__(data_dir)
        self.period = period

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RSI using pandas-ta.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with RSI column added
        """
        if not self.validate_ohlcv(df):
            raise ValueError("DataFrame must contain OHLCV columns")

        df = df.copy()

        # Use pandas-ta to calculate RSI
        df[f"RSI_{self.period}"] = ta.rsi(df["Close"], length=self.period)

        return df


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Standalone function to calculate RSI.

    Args:
        df: DataFrame with OHLCV data
        period: RSI period

    Returns:
        DataFrame with RSI column added
    """
    rsi_calculator = RSI(period=period)
    return rsi_calculator.calculate(df)
