"""
base.py
Base class for technical indicators.
Located at: src/nse_analyzer/indicators/technical/base.py
"""

from abc import ABC, abstractmethod
import pandas as pd
from pathlib import Path


class TechnicalIndicator(ABC):
    """Base class for all technical indicators using pandas-ta."""

    def __init__(self, data_dir: Path | None = None):
        """
        Initialize indicator.

        Args:
            data_dir: Optional custom data directory
        """
        self.data_dir = data_dir

    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the indicator.

        Args:
            df: DataFrame with OHLCV data (must have Date column)

        Returns:
            DataFrame with original data + indicator columns
        """
        pass

    @staticmethod
    def validate_ohlcv(df: pd.DataFrame) -> bool:
        """
        Validate that DataFrame has required OHLCV columns.

        Args:
            df: DataFrame to validate

        Returns:
            True if valid, False otherwise
        """
        required_cols = {"Open", "High", "Low", "Close", "Volume"}
        return required_cols.issubset(df.columns)

    @staticmethod
    def ensure_datetime(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
        """
        Ensure Date column is datetime type.

        Args:
            df: DataFrame to process
            date_col: Name of date column

        Returns:
            DataFrame with datetime converted Date column
        """
        df = df.copy()
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
        return df
