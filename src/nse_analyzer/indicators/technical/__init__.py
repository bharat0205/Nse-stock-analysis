"""
Technical indicators module.
Imports all indicator calculators for easy access.
Uses pandas-ta for TradingView-accurate indicators.
Located at: src/nse_analyzer/indicators/technical/__init__.py
"""

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

__all__ = [
    # RSI
    "RSI",
    "calculate_rsi",
    # MACD
    "MACD",
    "calculate_macd",
    # Supertrend
    "Supertrend",
    "calculate_supertrend",
    # Bollinger Bands
    "BollingerBands",
    "calculate_bollinger_bands",
    # Calculator
    "TechnicalIndicatorCalculator",
]
