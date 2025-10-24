"""
Stock Data Tool for retrieving stock data and metrics.
Uses Yahoo Finance API for comprehensive stock analysis.
"""

import yfinance as yf
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import time

logger = logging.getLogger(__name__)

@dataclass
class StockMetrics:
    """Represents key stock metrics and ratios."""
    symbol: str
    current_price: float
    market_cap: float
    pe_ratio: Optional[float]
    pb_ratio: Optional[float]
    eps: Optional[float]
    dividend_yield: Optional[float]
    beta: Optional[float]
    volume: int
    avg_volume: int
    high_52w: float
    low_52w: float
    change_percent: float
    last_updated: datetime
    # Additional financial metrics
    revenue_growth: Optional[float] = None
    profit_growth: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    ev_ebitda: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    return_on_equity: Optional[float] = None
    return_on_assets: Optional[float] = None

@dataclass
class PriceData:
    """Represents historical price data."""
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

@dataclass
class CompanyInfo:
    """Represents company information from NSE."""
    symbol: str
    company_name: str
    sector: str
    industry: str
    market_cap: Optional[float] = None
    isin: Optional[str] = None
    listing_date: Optional[str] = None

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
        pass
        
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
            revenue_growth = self._calculate_revenue_growth(ticker, info)
            profit_growth = self._calculate_profit_growth(ticker, info)
            roe = self._calculate_roe(ticker, info)
            roa = self._calculate_roa(ticker, info)
            ev_ebitda = self._calculate_ev_ebitda(ticker, info)
            debt_to_equity = self._calculate_debt_to_equity(ticker, info)
            current_ratio = self._calculate_current_ratio(ticker, info)
            
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
        try:
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
    
    def _calculate_revenue_growth(self, ticker, info: Dict[str, Any]) -> Optional[float]:
        """Calculate revenue growth from multiple sources."""
        try:
            # Try to get from info first
            revenue_growth = info.get('revenueGrowth') or info.get('revenueGrowthTTM') or info.get('revenueGrowthQuarterly')
            if revenue_growth is not None:
                return float(revenue_growth) * 100  # Convert to percentage
            
            # Try to calculate from financials
            financials = ticker.financials
            if not financials.empty:
                # Try different revenue field names
                revenue_fields = ['Total Revenue', 'Revenue', 'Net Sales', 'Sales']
                for field in revenue_fields:
                    if field in financials.index:
                        revenue_data = financials.loc[field]
                        if revenue_data is not None and len(revenue_data) >= 2:
                            current_revenue = revenue_data.iloc[0]
                            previous_revenue = revenue_data.iloc[1]
                            if previous_revenue != 0:
                                growth = ((current_revenue - previous_revenue) / abs(previous_revenue)) * 100
                                return growth
            
            # Try quarterly data
            quarterly = ticker.quarterly_financials
            if not quarterly.empty:
                revenue_fields = ['Total Revenue', 'Revenue', 'Net Sales', 'Sales']
                for field in revenue_fields:
                    if field in quarterly.index:
                        revenue_data = quarterly.loc[field]
                        if revenue_data is not None and len(revenue_data) >= 2:
                            current_revenue = revenue_data.iloc[0]
                            previous_revenue = revenue_data.iloc[1]
                            if previous_revenue != 0:
                                growth = ((current_revenue - previous_revenue) / abs(previous_revenue)) * 100
                                return growth
            
            # Try to get from balance sheet
            balance_sheet = ticker.balance_sheet
            if not balance_sheet.empty:
                revenue_fields = ['Total Revenue', 'Revenue', 'Net Sales', 'Sales']
                for field in revenue_fields:
                    if field in balance_sheet.index:
                        revenue_data = balance_sheet.loc[field]
                        if revenue_data is not None and len(revenue_data) >= 2:
                            current_revenue = revenue_data.iloc[0]
                            previous_revenue = revenue_data.iloc[1]
                            if previous_revenue != 0:
                                growth = ((current_revenue - previous_revenue) / abs(previous_revenue)) * 100
                                return growth
            
            return None
        except Exception as e:
            logger.warning(f"Error calculating revenue growth: {e}")
            return None
    
    def _calculate_profit_growth(self, ticker, info: Dict[str, Any]) -> Optional[float]:
        """Calculate profit growth from multiple sources."""
        try:
            # Try to get from info first
            profit_growth = info.get('earningsGrowth') or info.get('earningsGrowthTTM') or info.get('earningsGrowthQuarterly')
            if profit_growth is not None:
                return float(profit_growth) * 100  # Convert to percentage
            
            # Try to calculate from financials
            financials = ticker.financials
            if not financials.empty:
                # Try different profit field names
                profit_fields = ['Net Income', 'Net Income Common Stockholders', 'Net Earnings', 'Profit']
                for field in profit_fields:
                    if field in financials.index:
                        net_income_data = financials.loc[field]
                        if net_income_data is not None and len(net_income_data) >= 2:
                            current_income = net_income_data.iloc[0]
                            previous_income = net_income_data.iloc[1]
                            if previous_income != 0:
                                growth = ((current_income - previous_income) / abs(previous_income)) * 100
                                return growth
            
            # Try quarterly data
            quarterly = ticker.quarterly_financials
            if not quarterly.empty:
                profit_fields = ['Net Income', 'Net Income Common Stockholders', 'Net Earnings', 'Profit']
                for field in profit_fields:
                    if field in quarterly.index:
                        net_income_data = quarterly.loc[field]
                        if net_income_data is not None and len(net_income_data) >= 2:
                            current_income = net_income_data.iloc[0]
                            previous_income = net_income_data.iloc[1]
                            if previous_income != 0:
                                growth = ((current_income - previous_income) / abs(previous_income)) * 100
                                return growth
            
            return None
        except Exception as e:
            logger.warning(f"Error calculating profit growth: {e}")
            return None
    
    def _calculate_roe(self, ticker, info: Dict[str, Any]) -> Optional[float]:
        """Calculate Return on Equity (ROE)."""
        try:
            # Try to get from info first
            roe = info.get('returnOnEquity') or info.get('roe') or info.get('returnOnEquityTTM')
            if roe is not None:
                return float(roe) * 100  # Convert to percentage
            
            # Try to calculate from financials
            financials = ticker.financials
            if not financials.empty:
                # Try different field names for net income and equity
                net_income_fields = ['Net Income', 'Net Income Common Stockholders', 'Net Earnings', 'Profit']
                equity_fields = ['Stockholders Equity', 'Total Stockholders Equity', 'Shareholders Equity', 'Total Equity']
                
                net_income = None
                shareholders_equity = None
                
                for field in net_income_fields:
                    if field in financials.index:
                        net_income = financials.loc[field].iloc[0]
                        break
                
                for field in equity_fields:
                    if field in financials.index:
                        shareholders_equity = financials.loc[field].iloc[0]
                        break
                
                if net_income is not None and shareholders_equity is not None and shareholders_equity != 0:
                    roe = (net_income / shareholders_equity) * 100
                    return roe
            
            return None
        except Exception as e:
            logger.warning(f"Error calculating ROE: {e}")
            return None
    
    def _calculate_roa(self, ticker, info: Dict[str, Any]) -> Optional[float]:
        """Calculate Return on Assets (ROA)."""
        try:
            # Try to get from info first
            roa = info.get('returnOnAssets') or info.get('roa') or info.get('returnOnAssetsTTM')
            if roa is not None:
                return float(roa) * 100  # Convert to percentage
            
            # Try to calculate from financials
            financials = ticker.financials
            if not financials.empty:
                # Try different field names for net income and assets
                net_income_fields = ['Net Income', 'Net Income Common Stockholders', 'Net Earnings', 'Profit']
                assets_fields = ['Total Assets', 'Total Current Assets', 'Assets', 'Current Assets']
                
                net_income = None
                total_assets = None
                
                for field in net_income_fields:
                    if field in financials.index:
                        net_income = financials.loc[field].iloc[0]
                        break
                
                for field in assets_fields:
                    if field in financials.index:
                        total_assets = financials.loc[field].iloc[0]
                        break
                
                if net_income is not None and total_assets is not None and total_assets != 0:
                    roa = (net_income / total_assets) * 100
                    return roa
            
            return None
        except Exception as e:
            logger.warning(f"Error calculating ROA: {e}")
            return None

    def _calculate_ev_ebitda(self, ticker, info: Dict[str, Any]) -> Optional[float]:
        """Calculate EV/EBITDA ratio."""
        try:
            # Try to get from info first
            ev_ebitda = info.get('evToEbitda') or info.get('ev_ebitda') or info.get('evToEbitdaTTM')
            if ev_ebitda is not None:
                return float(ev_ebitda)
            
            # Try to calculate from financials
            financials = ticker.financials
            if not financials.empty:
                # Try different field names for EBITDA
                ebitda_fields = ['EBITDA', 'Earnings Before Interest Taxes Depreciation Amortization', 'EBIT', 'Operating Income']
                
                ebitda = None
                for field in ebitda_fields:
                    if field in financials.index:
                        ebitda = financials.loc[field].iloc[0]
                        break
                
                if ebitda is not None and ebitda != 0:
                    # Get market cap for EV calculation
                    market_cap = info.get('marketCap') or info.get('market_cap')
                    if market_cap is not None:
                        ev_ebitda = market_cap / ebitda
                        return ev_ebitda
            
            return None
        except Exception as e:
            logger.warning(f"Error calculating EV/EBITDA: {e}")
            return None
    
    def _calculate_debt_to_equity(self, ticker, info: Dict[str, Any]) -> Optional[float]:
        """Calculate Debt-to-Equity ratio."""
        try:
            # Try to get from info first
            debt_to_equity = info.get('debtToEquity') or info.get('debtToEquityRatio')
            if debt_to_equity is not None:
                return float(debt_to_equity)
            
            # Try to calculate from financials
            total_debt = info.get('totalDebt') or info.get('longTermDebt', 0)
            shareholders_equity = info.get('totalStockholderEquity') or info.get('shareholdersEquity')
            
            if total_debt and shareholders_equity and shareholders_equity != 0:
                return total_debt / shareholders_equity
            
            return None
        except Exception as e:
            logger.warning(f"Error calculating Debt-to-Equity: {e}")
            return None
    
    def _calculate_current_ratio(self, ticker, info: Dict[str, Any]) -> Optional[float]:
        """Calculate Current Ratio."""
        try:
            # Try to get from info first
            current_ratio = info.get('currentRatio') or info.get('currentRatioQuarterly')
            if current_ratio is not None:
                return float(current_ratio)
            
            # Try to calculate from financials
            current_assets = info.get('totalCurrentAssets') or info.get('currentAssets')
            current_liabilities = info.get('totalCurrentLiabilities') or info.get('currentLiabilities')
            
            if current_assets and current_liabilities and current_liabilities != 0:
                return current_assets / current_liabilities
            
            return None
        except Exception as e:
            logger.warning(f"Error calculating Current Ratio: {e}")
            return None
