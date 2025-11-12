"""
macd.py
MACD indicator using pandas-ta.
Located at: src/nse_analyzer/indicators/technical/macd.py
"""

import pandas as pd
import pandas_ta as ta
from pathlib import Path
from nse_analyzer.indicators.technical.base import TechnicalIndicator


class MACD(TechnicalIndicator):
    """Calculate MACD indicator."""

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        data_dir: Path | None = None,
    ):
        """
        Initialize MACD calculator.

        Args:
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line EMA period (default: 9)
            data_dir: Optional custom data directory
        """
        super().__init__(data_dir)
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MACD using pandas-ta.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with MACD columns added
        """
        if not self.validate_ohlcv(df):
            raise ValueError("DataFrame must contain OHLCV columns")

        df = df.copy()

        # Use pandas-ta to calculate MACD
        macd_result = ta.macd(
            df["Close"], fast=self.fast, slow=self.slow, signal=self.signal
        )

        if macd_result is None or macd_result.empty:
            raise ValueError(
                "MACD calculation returned empty result. Input data may be too short."
            )

        # pandas-ta returns columns: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
        df["MACD"] = macd_result[f"MACD_{self.fast}_{self.slow}_{self.signal}"]
        df["MACD_Signal"] = macd_result[f"MACDs_{self.fast}_{self.slow}_{self.signal}"]
        df["MACD_Histogram"] = macd_result[
            f"MACDh_{self.fast}_{self.slow}_{self.signal}"
        ]

        return df


def calculate_macd(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """
    Standalone function to calculate MACD.

    Args:
        df: DataFrame with OHLCV data
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line EMA period

    Returns:
        DataFrame with MACD columns added
    """
    macd_calculator = MACD(fast=fast, slow=slow, signal=signal)
    return macd_calculator.calculate(df)
