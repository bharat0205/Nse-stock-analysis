"""
supertrend.py
Supertrend indicator using pandas-ta.
Located at: src/nse_analyzer/indicators/technical/supertrend.py
"""

import pandas as pd
import pandas_ta as ta
from pathlib import Path
from nse_analyzer.indicators.technical.base import TechnicalIndicator


class Supertrend(TechnicalIndicator):
    """Calculate Supertrend indicator."""

    def __init__(
        self, period: int = 10, multiplier: float = 3.0, data_dir: Path | None = None
    ):
        """
        Initialize Supertrend calculator.

        Args:
            period: ATR period (default: 10)
            multiplier: ATR multiplier (default: 3.0)
            data_dir: Optional custom data directory
        """
        super().__init__(data_dir)
        self.period = period
        self.multiplier = multiplier

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Supertrend using pandas-ta.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with Supertrend columns added
        """
        if not self.validate_ohlcv(df):
            raise ValueError("DataFrame must contain OHLCV columns")

        df = df.copy()

        # Use pandas-ta to calculate Supertrend
        st_result = ta.supertrend(
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            length=self.period,
            multiplier=self.multiplier,
        )

        # pandas-ta returns columns: SUPERT_10_3.0, SUPERTd_10_3.0,
        #  SUPERTl_10_3.0, SUPERTs_10_3.0
        df["Supertrend"] = st_result[f"SUPERT_{self.period}_{self.multiplier}"]
        df["Supertrend_Direction"] = st_result[
            f"SUPERTd_{self.period}_{self.multiplier}"
        ]
        df["Supertrend_Long"] = st_result[f"SUPERTl_{self.period}_{self.multiplier}"]
        df["Supertrend_Short"] = st_result[f"SUPERTs_{self.period}_{self.multiplier}"]

        return df


def calculate_supertrend(
    df: pd.DataFrame, period: int = 10, multiplier: float = 3.0
) -> pd.DataFrame:
    """
    Standalone function to calculate Supertrend.

    Args:
        df: DataFrame with OHLCV data
        period: ATR period
        multiplier: ATR multiplier

    Returns:
        DataFrame with Supertrend columns added
    """
    st_calculator = Supertrend(period=period, multiplier=multiplier)
    return st_calculator.calculate(df)
