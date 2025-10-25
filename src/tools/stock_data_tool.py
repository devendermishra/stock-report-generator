"""
Stock Data Tool for retrieving stock data and metrics.
Uses Yahoo Finance API for comprehensive stock analysis.
"""

import yfinance as yf
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import time

from .stock_data_models import StockMetrics, PriceData, CompanyInfo
from .stock_data_calculator import StockDataCalculator
from .stock_data_validator import StockDataValidator
from ..exceptions import DataRetrievalError, ValidationError

logger = logging.getLogger(__name__)

class StockDataTool:
    """
    Stock Data Tool for retrieving comprehensive stock data.
    
    Provides access to real-time and historical stock data, financial metrics,
    and calculated ratios for analysis using Yahoo Finance API.
    """
    
    def __init__(self):
        """
        Initialize the Stock Data Tool.
        """
        self.calculator = StockDataCalculator()
        self.validator = StockDataValidator()
        
    def get_stock_metrics(self, symbol: str) -> Optional[StockMetrics]:
        """
        Get comprehensive stock metrics for a given symbol.
        
        Args:
            symbol: NSE stock symbol (e.g., 'RELIANCE.NS')
            
        Returns:
            StockMetrics object or None if data unavailable
        """
        try:
            # Ensure .NS suffix for NSE stocks
            if not symbol.endswith('.NS'):
                symbol = f"{symbol}.NS"
                
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Debug: Log available fields
            logger.info(f"Available fields for {symbol}: {list(info.keys())[:20]}...")  # First 20 fields
            
            # Get current price and basic info - try multiple field names
            current_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose') or 0
            market_cap = info.get('marketCap') or info.get('totalAssets') or 0
            
            # Financial ratios - try multiple field names
            pe_ratio = info.get('trailingPE') or info.get('forwardPE') or info.get('priceToEarnings') or info.get('peRatio')
            pb_ratio = info.get('priceToBook') or info.get('priceToBookRatio') or info.get('pbRatio')
            eps = info.get('trailingEps') or info.get('forwardEps') or info.get('earningsPerShare') or info.get('eps')
            dividend_yield = info.get('dividendYield') or info.get('dividendRate') or info.get('yield')
            beta = info.get('beta') or info.get('beta3Year') or info.get('beta3Y')
            
            # Debug: Log specific values
            logger.info(f"P/E ratio fields: trailingPE={info.get('trailingPE')}, forwardPE={info.get('forwardPE')}, priceToEarnings={info.get('priceToEarnings')}")
            logger.info(f"P/B ratio fields: priceToBook={info.get('priceToBook')}, priceToBookRatio={info.get('priceToBookRatio')}")
            logger.info(f"EPS fields: trailingEps={info.get('trailingEps')}, forwardEps={info.get('forwardEps')}, earningsPerShare={info.get('earningsPerShare')}")
            
            # Volume data - try multiple field names
            volume = info.get('volume') or info.get('regularMarketVolume') or 0
            avg_volume = info.get('averageVolume') or info.get('averageVolume10days') or 0
            
            # 52-week high/low - try multiple field names
            high_52w = info.get('fiftyTwoWeekHigh') or info.get('dayHigh') or 0
            low_52w = info.get('fiftyTwoWeekLow') or info.get('dayLow') or 0
            
            # Price change - try multiple field names
            change_percent = info.get('regularMarketChangePercent') or info.get('changePercent') or 0
            
            # Calculate additional financial metrics from multiple sources
            revenue_growth = self.calculator.calculate_revenue_growth(ticker, info)
            profit_growth = self.calculator.calculate_profit_growth(ticker, info)
            roe = self.calculator.calculate_roe(ticker, info)
            roa = self.calculator.calculate_roa(ticker, info)
            ev_ebitda = self.calculator.calculate_ev_ebitda(ticker, info)
            debt_to_equity = self.calculator.calculate_debt_to_equity(ticker, info)
            current_ratio = self.calculator.calculate_current_ratio(ticker, info)
            
            metrics = StockMetrics(
                symbol=symbol,
                current_price=current_price,
                market_cap=market_cap,
                pe_ratio=pe_ratio,
                pb_ratio=pb_ratio,
                eps=eps,
                dividend_yield=dividend_yield,
                beta=beta,
                volume=volume,
                avg_volume=avg_volume,
                high_52w=high_52w,
                low_52w=low_52w,
                change_percent=change_percent,
                last_updated=datetime.now(),
                # Additional financial metrics
                revenue_growth=revenue_growth,
                profit_growth=profit_growth,
                roe=roe,
                roa=roa,
                ev_ebitda=ev_ebitda,
                debt_to_equity=debt_to_equity,
                current_ratio=current_ratio,
                return_on_equity=roe,
                return_on_assets=roa
            )
            
            logger.info(f"Retrieved metrics for {symbol}")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get stock metrics for {symbol}: {e}")
            return None
            
    def get_historical_data(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d"
    ) -> List[PriceData]:
        """
        Get historical price data for a stock.
        
        Args:
            symbol: NSE stock symbol
            period: Period for data (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            List of PriceData objects
        """
        try:
            if not symbol.endswith('.NS'):
                symbol = f"{symbol}.NS"
                
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)
            
            price_data = []
            for date, row in hist.iterrows():
                price_data.append(PriceData(
                    date=date,
                    open=row['Open'],
                    high=row['High'],
                    low=row['Low'],
                    close=row['Close'],
                    volume=int(row['Volume'])
                ))
                
            logger.info(f"Retrieved {len(price_data)} historical data points for {symbol}")
            return price_data
            
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            return []
            
    def calculate_technical_indicators(
        self,
        price_data: List[PriceData]
    ) -> Dict[str, Any]:
        """
        Calculate technical indicators from price data.
        
        Args:
            price_data: List of historical price data
            
        Returns:
            Dictionary containing technical indicators
        """
        if not price_data:
            return {}
            
        try:
            # Convert to DataFrame for easier calculations
            df = pd.DataFrame([{
                'date': p.date,
                'open': p.open,
                'high': p.high,
                'low': p.low,
                'close': p.close,
                'volume': p.volume
            } for p in price_data])
            
            df.set_index('date', inplace=True)
            
            # Calculate moving averages
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['sma_200'] = df['close'].rolling(window=200).mean()
            
            # Calculate RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Calculate Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # Get latest values
            latest = df.iloc[-1]
            
            indicators = {
                'sma_20': latest.get('sma_20'),
                'sma_50': latest.get('sma_50'),
                'sma_200': latest.get('sma_200'),
                'rsi': latest.get('rsi'),
                'bb_upper': latest.get('bb_upper'),
                'bb_middle': latest.get('bb_middle'),
                'bb_lower': latest.get('bb_lower'),
                'current_price': latest['close'],
                'volume_trend': df['volume'].tail(5).mean()
            }
            
            logger.info("Calculated technical indicators")
            return indicators
            
        except Exception as e:
            logger.error(f"Failed to calculate technical indicators: {e}")
            return {}
            
    def get_sector_performance(self, sector: str) -> Dict[str, Any]:
        """
        Get sector performance data and peer comparison.
        
        Args:
            sector: Sector name (e.g., 'Banking', 'IT', 'Pharma')
            
        Returns:
            Dictionary containing sector performance metrics
        """
        try:
            # This is a simplified implementation
            # In a real scenario, you'd query a sector database or API
            
            sector_symbols = {
                'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS'],
                'IT': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS'],
                'Technology': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS'],
                'Financial Services': ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS'],
                'Pharma': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'BIOCON.NS'],
                'Auto': ['MARUTI.NS', 'TATAMOTORS.NS', 'BAJAJ-AUTO.NS', 'M&M.NS']
            }
            
            if sector not in sector_symbols:
                return {}
                
            sector_data = []
            for symbol in sector_symbols[sector]:
                metrics = self.get_stock_metrics(symbol)
                if metrics:
                    sector_data.append({
                        'symbol': symbol,
                        'price': metrics.current_price,
                        'change': metrics.change_percent,
                        'pe': metrics.pe_ratio
                    })
                    
            if not sector_data:
                return {}
                
            # Calculate sector averages
            avg_change = sum(d['change'] for d in sector_data) / len(sector_data)
            avg_pe = sum(d['pe'] for d in sector_data if d['pe']) / len([d for d in sector_data if d['pe']])
            
            return {
                'sector': sector,
                'avg_change': avg_change,
                'avg_pe': avg_pe,
                'stocks': sector_data,
                'performance': 'positive' if avg_change > 0 else 'negative'
            }
            
        except Exception as e:
            logger.error(f"Failed to get sector performance for {sector}: {e}")
            return {}
            
    def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive company information.
        
        Args:
            symbol: NSE stock symbol
            
        Returns:
            Dictionary containing company information
        """
        try:
            if not symbol.endswith('.NS'):
                symbol = f"{symbol}.NS"
                
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            company_info = {
                'name': info.get('longName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'description': info.get('longBusinessSummary', ''),
                'website': info.get('website', ''),
                'employees': info.get('fullTimeEmployees', 0),
                'city': info.get('city', ''),
                'state': info.get('state', ''),
                'country': info.get('country', ''),
                'market_cap': info.get('marketCap', 0),
                'enterprise_value': info.get('enterpriseValue', 0)
            }
            
            logger.info(f"Retrieved company info for {symbol}")
            return company_info
            
        except Exception as e:
            logger.error(f"Failed to get company info for {symbol}: {e}")
            return {}
            
    def get_company_name_and_sector(self, symbol: str) -> CompanyInfo:
        """
        Get company name and sector for a given symbol.
        
        Args:
            symbol: NSE stock symbol (without .NS suffix)
            
        Returns:
            CompanyInfo object with company name and sector
        """
        return self.validator.get_company_name_and_sector(symbol)
            
            
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a stock symbol exists and is tradeable.
        
        Args:
            symbol: Stock symbol to validate
            
        Returns:
            True if symbol is valid, False otherwise
        """
        return self.validator.validate_symbol(symbol)
            
    def validate_symbol_with_yahoo(self, symbol: str) -> bool:
        """
        Validate symbol using Yahoo Finance API.
        
        Args:
            symbol: Stock symbol to validate
            
        Returns:
            True if symbol is valid, False otherwise
        """
        return self.validator.validate_symbol_with_yahoo(symbol)
    
