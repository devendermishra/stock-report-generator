"""
Unit tests for stock data validator.
Tests StockDataValidator class methods.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from tools.stock_data_validator import StockDataValidator
from tools.stock_data_models import CompanyInfo


class TestStockDataValidator:
    """Test cases for StockDataValidator class."""
    
    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.validator = StockDataValidator()
    
    def test_validate_metrics_complete(self) -> None:
        """Test validating metrics with complete data."""
        metrics = {
            "current_price": 100.0,
            "market_cap": 10000000000,
            "high_52w": 120.0,
            "low_52w": 80.0
        }
        result = self.validator.validate_metrics(metrics)
        
        assert result.is_valid is True
        assert len(result.issues) == 0
    
    def test_validate_metrics_missing_current_price(self) -> None:
        """Test validating metrics with missing current price."""
        metrics = {
            "market_cap": 10000000000,
            "high_52w": 120.0,
            "low_52w": 80.0
        }
        result = self.validator.validate_metrics(metrics)
        
        assert result.is_valid is False
        assert "current_price" in result.issues[0].lower()
    
    def test_validate_metrics_zero_current_price(self) -> None:
        """Test validating metrics with zero current price."""
        metrics = {
            "current_price": 0,
            "market_cap": 10000000000,
            "high_52w": 120.0,
            "low_52w": 80.0
        }
        result = self.validator.validate_metrics(metrics)
        
        assert result.is_valid is False
        assert "current_price" in result.issues[0].lower()
    
    def test_validate_metrics_missing_market_cap(self) -> None:
        """Test validating metrics with missing market cap."""
        metrics = {
            "current_price": 100.0,
            "high_52w": 120.0,
            "low_52w": 80.0
        }
        result = self.validator.validate_metrics(metrics)
        
        assert result.is_valid is False
        assert "market_cap" in result.issues[0].lower()
    
    def test_validate_metrics_zero_market_cap(self) -> None:
        """Test validating metrics with zero market cap."""
        metrics = {
            "current_price": 100.0,
            "market_cap": 0,
            "high_52w": 120.0,
            "low_52w": 80.0
        }
        result = self.validator.validate_metrics(metrics)
        
        assert result.is_valid is False
        assert "market_cap" in result.issues[0].lower()
    
    def test_validate_metrics_missing_52w_range(self) -> None:
        """Test validating metrics with missing 52-week range."""
        metrics = {
            "current_price": 100.0,
            "market_cap": 10000000000
        }
        result = self.validator.validate_metrics(metrics)
        
        assert result.is_valid is False
        assert "52-week" in result.issues[0].lower()
    
    def test_validate_metrics_multiple_issues(self) -> None:
        """Test validating metrics with multiple issues."""
        metrics = {}
        result = self.validator.validate_metrics(metrics)
        
        assert result.is_valid is False
        assert len(result.issues) >= 3
    
    @patch('yfinance.Ticker')
    def test_validate_symbol_valid(self, mock_ticker_class) -> None:
        """Test validating a valid stock symbol."""
        # Mock yfinance response - need at least 5 fields to pass validation
        mock_ticker = Mock()
        mock_ticker.info = {
            'currentPrice': 100.0,
            'longName': 'Test Company',
            'marketCap': 10000000000,
            'symbol': 'TCS.NS',
            'currency': 'INR',
            'exchange': 'NSE'
        }
        mock_ticker_class.return_value = mock_ticker
        
        result = self.validator.validate_symbol("TCS")
        
        assert result is True
        mock_ticker_class.assert_called_once()
    
    @patch('yfinance.Ticker')
    def test_validate_symbol_invalid_no_data(self, mock_ticker_class) -> None:
        """Test validating an invalid stock symbol with no data."""
        # Mock yfinance response with minimal data
        mock_ticker = Mock()
        mock_ticker.info = {}
        mock_ticker_class.return_value = mock_ticker
        
        result = self.validator.validate_symbol("INVALID")
        
        assert result is False
    
    @patch('yfinance.Ticker')
    def test_validate_symbol_invalid_no_price_name_cap(self, mock_ticker_class) -> None:
        """Test validating symbol with no price, name, or market cap."""
        # Mock yfinance response without key fields
        mock_ticker = Mock()
        mock_ticker.info = {'otherField': 'value'}
        mock_ticker_class.return_value = mock_ticker
        
        result = self.validator.validate_symbol("INVALID")
        
        assert result is False
    
    @patch('yfinance.Ticker')
    def test_validate_symbol_with_ns_suffix(self, mock_ticker_class) -> None:
        """Test validating symbol that already has .NS suffix."""
        mock_ticker = Mock()
        # Need at least 5 fields to pass validation check
        mock_ticker.info = {
            'currentPrice': 100.0,
            'longName': 'Test Company',
            'marketCap': 10000000000,
            'symbol': 'TCS.NS',
            'currency': 'INR',
            'exchange': 'NSE'
        }
        mock_ticker_class.return_value = mock_ticker
        
        result = self.validator.validate_symbol("TCS.NS")
        
        assert result is True
        # Should still call with .NS suffix
        mock_ticker_class.assert_called_once()
    
    @patch('yfinance.Ticker')
    def test_validate_symbol_exception_handling(self, mock_ticker_class) -> None:
        """Test validating symbol with exception handling."""
        mock_ticker_class.side_effect = Exception("Network error")
        
        result = self.validator.validate_symbol("TCS")
        
        assert result is False
    
    @patch('yfinance.Ticker')
    def test_get_company_name_and_sector_success(self, mock_ticker_class) -> None:
        """Test getting company name and sector successfully."""
        mock_ticker = Mock()
        mock_ticker.info = {
            'longName': 'Tata Consultancy Services',
            'sector': 'Technology',
            'industry': 'IT Services',
            'marketCap': 10000000000
        }
        mock_ticker_class.return_value = mock_ticker
        
        result = self.validator.get_company_name_and_sector("TCS")
        
        assert isinstance(result, CompanyInfo)
        assert result.symbol == "TCS"
        assert result.company_name == "Tata Consultancy Services"
        assert result.sector == "Technology"
        assert result.industry == "IT Services"
    
    @patch('yfinance.Ticker')
    def test_get_company_name_and_sector_with_sector_mapping(self, mock_ticker_class) -> None:
        """Test getting company name with sector mapping."""
        mock_ticker = Mock()
        mock_ticker.info = {
            'longName': 'Tata Consultancy Services',
            'sector': '',  # Empty sector
            'industry': 'IT Services',
            'marketCap': 10000000000
        }
        mock_ticker_class.return_value = mock_ticker
        
        result = self.validator.get_company_name_and_sector("TCS")
        
        # The mapping should apply when sector is empty and symbol is in mapping
        assert result.sector == "Technology"  # Should use mapping
        assert result.company_name == "Tata Consultancy Services"
    
    @patch('yfinance.Ticker')
    def test_get_company_name_and_sector_fallback(self, mock_ticker_class) -> None:
        """Test getting company name with fallback values."""
        mock_ticker = Mock()
        mock_ticker.info = {}  # Empty info
        mock_ticker_class.return_value = mock_ticker
        
        result = self.validator.get_company_name_and_sector("UNKNOWN")
        
        assert isinstance(result, CompanyInfo)
        assert result.symbol == "UNKNOWN"
        assert result.company_name == "UNKNOWN"  # Fallback to symbol
        assert result.sector == "Unknown"  # Fallback sector
    
    @patch('yfinance.Ticker')
    def test_get_company_name_and_sector_exception_handling(self, mock_ticker_class) -> None:
        """Test getting company name with exception handling."""
        mock_ticker_class.side_effect = Exception("Network error")
        
        result = self.validator.get_company_name_and_sector("TCS")
        
        assert isinstance(result, CompanyInfo)
        assert result.symbol == "TCS"
        assert result.company_name == "TCS"  # Fallback
        assert result.sector == "Unknown"  # Fallback
    
    @patch('yfinance.Ticker')
    def test_get_company_name_and_sector_with_ns_suffix(self, mock_ticker_class) -> None:
        """Test getting company name with .NS suffix in input."""
        mock_ticker = Mock()
        mock_ticker.info = {
            'longName': 'Test Company',
            'sector': 'Technology',
            'industry': 'IT',
            'marketCap': 10000000000
        }
        mock_ticker_class.return_value = mock_ticker
        
        result = self.validator.get_company_name_and_sector("TCS.NS")
        
        assert result.symbol == "TCS.NS"
        mock_ticker_class.assert_called_once()

