"""
config.py
Configuration manager for loading settings and stock lists.
Located at: src/nse_analyzer/utils/config.py
"""

import yaml
from pathlib import Path
from typing import Dict, List

# Get paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"


class ConfigManager:
    """Manage configuration loading from YAML files."""

    @staticmethod
    def load_stocks(stock_group: str = "stocks") -> List[str]:
        """
        Load stock list from stocks.yaml.

        Args:
            stock_group: Key in stocks.yaml (default: 'stocks')
                        Options: 'stocks', 'nifty50', 'nifty100', 'custom'

        Returns:
            List of stock symbols (e.g., ['RELIANCE.NS', 'TCS.NS'])

        Raises:
            FileNotFoundError: If stocks.yaml doesn't exist
            KeyError: If stock_group doesn't exist in YAML
        """
        stocks_file = CONFIG_DIR / "stocks.yaml"

        if not stocks_file.exists():
            raise FileNotFoundError(
                f"stocks.yaml not found at {stocks_file}. "
                "Please create config/stocks.yaml with stock lists."
            )

        try:
            with open(stocks_file) as f:
                config = yaml.safe_load(f)

            if stock_group not in config:
                available = list(config.keys())
                raise KeyError(
                    f"Stock group '{stock_group}' not found in stocks.yaml. "
                    f"Available groups: {available}"
                )

            return config[stock_group]

        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing stocks.yaml: {e}")

    @staticmethod
    def load_indicators() -> Dict:
        """
        Load indicator settings from settings.yaml.

        Returns:
            Dict with indicator configurations
        """
        settings_file = CONFIG_DIR / "settings.yaml"

        if not settings_file.exists():
            raise FileNotFoundError(f"settings.yaml not found at {settings_file}")

        try:
            with open(settings_file) as f:
                config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing settings.yaml: {e}")

    @staticmethod
    def get_data_dir() -> Path:
        """Get project data directory path."""
        return DATA_DIR

    @staticmethod
    def get_config_dir() -> Path:
        """Get project config directory path."""
        return CONFIG_DIR

    @staticmethod
    def list_stock_groups() -> List[str]:
        """List all available stock groups."""
        stocks_file = CONFIG_DIR / "stocks.yaml"
        with open(stocks_file) as f:
            config = yaml.safe_load(f)
        return list(config.keys())


if __name__ == "__main__":
    # Test config loading
    print("Available stock groups:", ConfigManager.list_stock_groups())
    print("\nDefault stocks:", ConfigManager.load_stocks())
    print("\nNifty 50 stocks:", ConfigManager.load_stocks("nifty50"))
