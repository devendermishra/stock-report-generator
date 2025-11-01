"""
Stock data calculation utilities.
Contains methods for calculating financial metrics and ratios.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class StockDataCalculator:
    """Handles calculation of financial metrics and ratios."""
    
    def calculate_metrics(self, ticker, info: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """
        Aggregate a lightweight set of metrics derived from available sources.
        This complements raw `info` and recent price, and is used for validation.
        """
        try:
            metrics: Dict[str, Any] = {
                "symbol": info.get("symbol"),
                "current_price": current_price,
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE") or info.get("forwardPE"),
                "pb_ratio": info.get("priceToBook"),
                "eps": info.get("trailingEps") or info.get("forwardEps"),
                "dividend_yield": info.get("dividendYield"),
                "beta": info.get("beta"),
                "high_52w": info.get("fiftyTwoWeekHigh"),
                "low_52w": info.get("fiftyTwoWeekLow"),
            }
            return metrics
        except Exception as e:
            logger.warning(f"calculate_metrics failed: {e}")
            return {"symbol": info.get("symbol"), "current_price": current_price}
    
    def calculate_revenue_growth(self, ticker, info: Dict[str, Any]) -> Optional[float]:
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
    
    def calculate_profit_growth(self, ticker, info: Dict[str, Any]) -> Optional[float]:
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
    
    def calculate_roe(self, ticker, info: Dict[str, Any]) -> Optional[float]:
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
    
    def calculate_roa(self, ticker, info: Dict[str, Any]) -> Optional[float]:
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

    def calculate_ev_ebitda(self, ticker, info: Dict[str, Any]) -> Optional[float]:
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
    
    def calculate_debt_to_equity(self, ticker, info: Dict[str, Any]) -> Optional[float]:
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
    
    def calculate_current_ratio(self, ticker, info: Dict[str, Any]) -> Optional[float]:
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
