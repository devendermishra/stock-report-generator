"""
Stock data validation utilities.
Contains methods for validating stock symbols and data quality.
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any
from .stock_data_models import CompanyInfo

logger = logging.getLogger(__name__)


class StockDataValidator:
    """Handles validation of stock symbols and data quality."""
    
    @dataclass
    class ValidationResult:
        is_valid: bool
        issues: List[str]
    
    def validate_metrics(self, metrics: Dict[str, Any]) -> "StockDataValidator.ValidationResult":
        """Validate presence and sanity of key metrics used by callers."""
        issues: List[str] = []
        if metrics.get("current_price") in (None, 0):
            issues.append("Missing or zero current_price")
        if metrics.get("market_cap") in (None, 0):
            issues.append("Missing market_cap")
        if metrics.get("high_52w") is None or metrics.get("low_52w") is None:
            issues.append("Missing 52-week range")
        return StockDataValidator.ValidationResult(is_valid=len(issues) == 0, issues=issues)
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a stock symbol exists and is tradeable.
        
        Args:
            symbol: Stock symbol to validate
            
        Returns:
            True if symbol is valid, False otherwise
        """
        try:
            # Clean the symbol
            clean_symbol = symbol.upper().strip()
            if not clean_symbol.endswith('.NS'):
                yf_symbol = f"{clean_symbol}.NS"
            else:
                yf_symbol = clean_symbol
                
            logger.info(f"Validating symbol: {clean_symbol}")
            
            # Try to get basic info from yfinance
            import yfinance as yf
            ticker = yf.Ticker(yf_symbol)
            info = ticker.info
            
            # Check if we got valid data
            if not info or len(info) < 5:  # Very basic info should have at least 5 fields
                logger.warning(f"No data found for {clean_symbol}")
                return False
                
            # Check for key indicators of a valid stock
            has_price = info.get('currentPrice') is not None or info.get('regularMarketPrice') is not None
            has_name = bool(info.get('longName') or info.get('shortName'))
            has_market_cap = info.get('marketCap') is not None
            
            # At least one of these should be true for a valid stock
            is_valid = has_price or has_name or has_market_cap
            
            if is_valid:
                logger.info(f"Symbol {clean_symbol} is valid")
                return True
            else:
                logger.warning(f"Symbol {clean_symbol} appears to be invalid - no price, name, or market cap data")
                return False
            
        except Exception as e:
            logger.error(f"Symbol validation failed for {clean_symbol}: {e}")
            return False
    
    def validate_symbol_with_yahoo(self, symbol: str) -> bool:
        """
        Validate symbol using Yahoo Finance API.
        
        Args:
            symbol: Stock symbol to validate
            
        Returns:
            True if symbol is valid, False otherwise
        """
        try:
            clean_symbol = symbol.upper().strip()
            logger.info(f"Validating {clean_symbol} with Yahoo Finance API")
            
            # Use Yahoo Finance validation
            return self.validate_symbol(clean_symbol)
                
        except Exception as e:
            logger.error(f"Yahoo Finance validation failed for {clean_symbol}: {e}")
            return False
    
    def get_company_name_and_sector(self, symbol: str) -> CompanyInfo:
        """
        Get company name and sector for a given symbol.
        
        Args:
            symbol: NSE stock symbol (without .NS suffix)
            
        Returns:
            CompanyInfo object with company name and sector
        """
        try:
            import yfinance as yf
            
            # First try yfinance
            yf_symbol = f"{symbol}.NS" if not symbol.endswith('.NS') else symbol
            ticker = yf.Ticker(yf_symbol)
            info = ticker.info
            
            company_name = info.get('longName', '')
            sector = info.get('sector', '')
            industry = info.get('industry', '')
            market_cap = info.get('marketCap')
            
            # Fix known sector mapping issues
            sector_mapping = {
                'TCS': 'Technology',
                'INFY': 'Technology', 
                'WIPRO': 'Technology',
                'HCLTECH': 'Technology',
                'TECHM': 'Technology',
                'MINDTREE': 'Technology',
                'LTTS': 'Technology',
                'PERSISTENT': 'Technology',
                'MPHASIS': 'Technology',
                'COFORGE': 'Technology'
            }
            
            if symbol in sector_mapping:
                sector = sector_mapping[symbol]
            
            # If yfinance doesn't have the data, use fallback values
            if not company_name or not sector:
                logger.warning(f"Insufficient data from Yahoo Finance for {symbol}")
            
            # Fallback to symbol if no company name found
            if not company_name:
                company_name = symbol
                
            # Fallback to 'Unknown' if no sector found
            if not sector:
                sector = 'Unknown'
                
            company_info = CompanyInfo(
                symbol=symbol,
                company_name=company_name,
                sector=sector,
                industry=industry,
                market_cap=market_cap
            )
            
            logger.info(f"Retrieved company info: {company_name} ({sector})")
            return company_info
            
        except Exception as e:
            logger.error(f"Failed to get company name and sector for {symbol}: {e}")
            # Return fallback info
            return CompanyInfo(
                symbol=symbol,
                company_name=symbol,
                sector='Unknown',
                industry='Unknown'
            )
