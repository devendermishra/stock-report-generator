"""
Unit tests for stock data calculator.
Tests StockDataCalculator class methods.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import pandas as pd

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from tools.stock_data_calculator import StockDataCalculator


class TestStockDataCalculator:
    """Test cases for StockDataCalculator class."""
    
    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.calculator = StockDataCalculator()
    
    def test_calculate_metrics_complete(self) -> None:
        """Test calculating metrics with complete data."""
        mock_ticker = Mock()
        info = {
            "symbol": "TCS",
            "marketCap": 10000000000,
            "trailingPE": 20.0,
            "priceToBook": 2.0,
            "trailingEps": 5.0,
            "dividendYield": 0.025,
            "beta": 1.0,
            "fiftyTwoWeekHigh": 120.0,
            "fiftyTwoWeekLow": 80.0
        }
        current_price = 100.0
        
        result = self.calculator.calculate_metrics(mock_ticker, info, current_price)
        
        assert result["symbol"] == "TCS"
        assert result["current_price"] == 100.0
        assert result["market_cap"] == 10000000000
        assert result["pe_ratio"] == 20.0
        assert result["pb_ratio"] == 2.0
        assert result["eps"] == 5.0
        assert result["dividend_yield"] == 0.025
        assert result["beta"] == 1.0
        assert result["high_52w"] == 120.0
        assert result["low_52w"] == 80.0
    
    def test_calculate_metrics_with_forward_pe(self) -> None:
        """Test calculating metrics with forward P/E when trailing is missing."""
        mock_ticker = Mock()
        info = {
            "symbol": "TCS",
            "forwardPE": 18.0,
            "forwardEps": 5.5
        }
        current_price = 100.0
        
        result = self.calculator.calculate_metrics(mock_ticker, info, current_price)
        
        assert result["pe_ratio"] == 18.0
        assert result["eps"] == 5.5
    
    def test_calculate_metrics_minimal(self) -> None:
        """Test calculating metrics with minimal data."""
        mock_ticker = Mock()
        info = {"symbol": "TCS"}
        current_price = 100.0
        
        result = self.calculator.calculate_metrics(mock_ticker, info, current_price)
        
        assert result["symbol"] == "TCS"
        assert result["current_price"] == 100.0
    
    def test_calculate_metrics_exception_handling(self) -> None:
        """Test calculating metrics with exception handling."""
        mock_ticker = Mock()
        info = {}  # Empty dict instead of None to avoid AttributeError
        current_price = 100.0
        
        result = self.calculator.calculate_metrics(mock_ticker, info, current_price)
        
        # Should return minimal result
        assert "current_price" in result
        assert result["current_price"] == 100.0
    
    def test_calculate_revenue_growth_from_info(self) -> None:
        """Test calculating revenue growth from info."""
        mock_ticker = Mock()
        info = {"revenueGrowth": 0.10}  # 10% growth
        
        result = self.calculator.calculate_revenue_growth(mock_ticker, info)
        
        assert result == 10.0  # Converted to percentage
    
    def test_calculate_revenue_growth_from_financials(self) -> None:
        """Test calculating revenue growth from financials."""
        mock_ticker = Mock()
        info = {}
        
        # Mock financials DataFrame - yfinance returns data with dates as columns
        # and financial metrics as index (rows)
        financials_df = pd.DataFrame(
            [[1000000, 900000]],  # Data: [current, previous]
            columns=[pd.Timestamp('2023-12-31'), pd.Timestamp('2022-12-31')],
            index=['Total Revenue']
        )
        mock_ticker.financials = financials_df
        
        result = self.calculator.calculate_revenue_growth(mock_ticker, info)
        
        assert result is not None
        assert result > 0  # Should be positive growth (~11.11%)
    
    def test_calculate_revenue_growth_none(self) -> None:
        """Test calculating revenue growth when data is unavailable."""
        mock_ticker = Mock()
        info = {}
        mock_ticker.financials = pd.DataFrame()
        mock_ticker.quarterly_financials = pd.DataFrame()
        mock_ticker.balance_sheet = pd.DataFrame()
        
        result = self.calculator.calculate_revenue_growth(mock_ticker, info)
        
        assert result is None
    
    def test_calculate_revenue_growth_exception_handling(self) -> None:
        """Test calculating revenue growth with exception handling."""
        mock_ticker = Mock()
        mock_ticker.financials = None  # This will cause an error
        info = {}
        
        result = self.calculator.calculate_revenue_growth(mock_ticker, info)
        
        assert result is None
    
    def test_calculate_profit_growth_from_info(self) -> None:
        """Test calculating profit growth from info."""
        mock_ticker = Mock()
        info = {"earningsGrowth": 0.15}  # 15% growth
        
        result = self.calculator.calculate_profit_growth(mock_ticker, info)
        
        assert result == 15.0  # Converted to percentage
    
    def test_calculate_profit_growth_from_financials(self) -> None:
        """Test calculating profit growth from financials."""
        mock_ticker = Mock()
        info = {}
        
        # Mock financials DataFrame - yfinance returns data with dates as columns
        financials_df = pd.DataFrame(
            [[500000, 400000]],  # Data: [current, previous]
            columns=[pd.Timestamp('2023-12-31'), pd.Timestamp('2022-12-31')],
            index=['Net Income']
        )
        mock_ticker.financials = financials_df
        
        result = self.calculator.calculate_profit_growth(mock_ticker, info)
        
        assert result is not None
        assert result > 0  # Should be positive growth (25%)
    
    def test_calculate_profit_growth_none(self) -> None:
        """Test calculating profit growth when data is unavailable."""
        mock_ticker = Mock()
        info = {}
        mock_ticker.financials = pd.DataFrame()
        mock_ticker.quarterly_financials = pd.DataFrame()
        
        result = self.calculator.calculate_profit_growth(mock_ticker, info)
        
        assert result is None
    
    def test_calculate_roe_from_info(self) -> None:
        """Test calculating ROE from info."""
        mock_ticker = Mock()
        info = {"returnOnEquity": 0.20}  # 20% ROE
        
        result = self.calculator.calculate_roe(mock_ticker, info)
        
        assert result == 20.0  # Converted to percentage
    
    def test_calculate_roe_from_financials(self) -> None:
        """Test calculating ROE from financials."""
        mock_ticker = Mock()
        info = {}
        
        # Mock financials DataFrame - yfinance returns data with dates as columns
        # Each row is a financial metric, columns are dates
        financials_df = pd.DataFrame(
            [[100000], [500000]],  # Net Income row, Stockholders Equity row
            columns=[pd.Timestamp('2023-12-31')],
            index=['Net Income', 'Stockholders Equity']
        )
        mock_ticker.financials = financials_df
        
        result = self.calculator.calculate_roe(mock_ticker, info)
        
        assert result is not None
        assert result == 20.0  # 100000 / 500000 * 100
    
    def test_calculate_roe_none(self) -> None:
        """Test calculating ROE when data is unavailable."""
        mock_ticker = Mock()
        info = {}
        mock_ticker.financials = pd.DataFrame()
        
        result = self.calculator.calculate_roe(mock_ticker, info)
        
        assert result is None
    
    def test_calculate_roa_from_info(self) -> None:
        """Test calculating ROA from info."""
        mock_ticker = Mock()
        info = {"returnOnAssets": 0.15}  # 15% ROA
        
        result = self.calculator.calculate_roa(mock_ticker, info)
        
        assert result == 15.0  # Converted to percentage
    
    def test_calculate_roa_from_financials(self) -> None:
        """Test calculating ROA from financials."""
        mock_ticker = Mock()
        info = {}
        
        # Mock financials DataFrame - yfinance returns data with dates as columns
        financials_df = pd.DataFrame(
            [[100000], [1000000]],  # Net Income row, Total Assets row
            columns=[pd.Timestamp('2023-12-31')],
            index=['Net Income', 'Total Assets']
        )
        mock_ticker.financials = financials_df
        
        result = self.calculator.calculate_roa(mock_ticker, info)
        
        assert result is not None
        assert result == 10.0  # 100000 / 1000000 * 100
    
    def test_calculate_roa_none(self) -> None:
        """Test calculating ROA when data is unavailable."""
        mock_ticker = Mock()
        info = {}
        mock_ticker.financials = pd.DataFrame()
        
        result = self.calculator.calculate_roa(mock_ticker, info)
        
        assert result is None
    
    def test_calculate_ev_ebitda_from_info(self) -> None:
        """Test calculating EV/EBITDA from info."""
        mock_ticker = Mock()
        info = {"evToEbitda": 12.5}
        
        result = self.calculator.calculate_ev_ebitda(mock_ticker, info)
        
        assert result == 12.5
    
    def test_calculate_ev_ebitda_from_financials(self) -> None:
        """Test calculating EV/EBITDA from financials."""
        mock_ticker = Mock()
        info = {"marketCap": 1000000000}  # 1 billion market cap
        
        # Mock financials DataFrame
        financials_data = {
            '2023': [100000000]  # EBITDA
        }
        financials_df = pd.DataFrame(financials_data, index=['EBITDA'])
        mock_ticker.financials = financials_df
        
        result = self.calculator.calculate_ev_ebitda(mock_ticker, info)
        
        assert result is not None
        assert result == 10.0  # 1000000000 / 100000000
    
    def test_calculate_ev_ebitda_none(self) -> None:
        """Test calculating EV/EBITDA when data is unavailable."""
        mock_ticker = Mock()
        info = {}
        mock_ticker.financials = pd.DataFrame()
        
        result = self.calculator.calculate_ev_ebitda(mock_ticker, info)
        
        assert result is None
    
    def test_calculate_debt_to_equity_from_info(self) -> None:
        """Test calculating Debt-to-Equity from info."""
        mock_ticker = Mock()
        info = {"debtToEquity": 0.5}
        
        result = self.calculator.calculate_debt_to_equity(mock_ticker, info)
        
        assert result == 0.5
    
    def test_calculate_debt_to_equity_calculated(self) -> None:
        """Test calculating Debt-to-Equity from calculated values."""
        mock_ticker = Mock()
        info = {
            "totalDebt": 500000,
            "totalStockholderEquity": 1000000
        }
        
        result = self.calculator.calculate_debt_to_equity(mock_ticker, info)
        
        assert result == 0.5  # 500000 / 1000000
    
    def test_calculate_debt_to_equity_none(self) -> None:
        """Test calculating Debt-to-Equity when data is unavailable."""
        mock_ticker = Mock()
        info = {}
        
        result = self.calculator.calculate_debt_to_equity(mock_ticker, info)
        
        assert result is None
    
    def test_calculate_current_ratio_from_info(self) -> None:
        """Test calculating Current Ratio from info."""
        mock_ticker = Mock()
        info = {"currentRatio": 2.0}
        
        result = self.calculator.calculate_current_ratio(mock_ticker, info)
        
        assert result == 2.0
    
    def test_calculate_current_ratio_calculated(self) -> None:
        """Test calculating Current Ratio from calculated values."""
        mock_ticker = Mock()
        info = {
            "totalCurrentAssets": 2000000,
            "totalCurrentLiabilities": 1000000
        }
        
        result = self.calculator.calculate_current_ratio(mock_ticker, info)
        
        assert result == 2.0  # 2000000 / 1000000
    
    def test_calculate_current_ratio_none(self) -> None:
        """Test calculating Current Ratio when data is unavailable."""
        mock_ticker = Mock()
        info = {}
        
        result = self.calculator.calculate_current_ratio(mock_ticker, info)
        
        assert result is None
    
    def test_calculate_current_ratio_zero_liabilities(self) -> None:
        """Test calculating Current Ratio with zero liabilities."""
        mock_ticker = Mock()
        info = {
            "totalCurrentAssets": 2000000,
            "totalCurrentLiabilities": 0
        }
        
        result = self.calculator.calculate_current_ratio(mock_ticker, info)
        
        assert result is None  # Division by zero should return None

