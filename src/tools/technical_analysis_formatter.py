"""
Technical analysis formatting utilities.
Contains methods for formatting technical analysis data into readable reports.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class TechnicalAnalysisFormatter:
    """Handles formatting of technical analysis data."""
    
    def format_technical_analysis(self, technical_data: Dict[str, Any]) -> str:
        """Format technical analysis data into readable format."""
        if not technical_data:
            return "Technical analysis indicates neutral market sentiment with mixed signals. Key technical indicators suggest a balanced outlook for the stock."
        
        # Extract indicators
        indicators = technical_data.get('indicators', {})
        trend_analysis = technical_data.get('trend_analysis', 'Neutral')
        support_resistance = technical_data.get('support_resistance', {})
        momentum = technical_data.get('momentum', 0)
        
        # Format the analysis
        analysis_parts = []
        
        # Trend Analysis
        if trend_analysis:
            trend_desc = self._get_trend_description(trend_analysis)
            analysis_parts.append(f"**Trend Analysis:** {trend_desc}")
        
        # Key Indicators
        if indicators:
            analysis_parts.append("**Key Technical Indicators:**")
            
            # Moving Averages
            sma_20 = indicators.get('sma_20')
            sma_50 = indicators.get('sma_50')
            sma_200 = indicators.get('sma_200')
            current_price = indicators.get('current_price')
            
            if all(x is not None for x in [sma_20, sma_50, sma_200, current_price]):
                analysis_parts.append(f"- **Moving Averages:** SMA 20: ₹{sma_20:.2f}, SMA 50: ₹{sma_50:.2f}, SMA 200: ₹{sma_200:.2f}")
                
                # Trend interpretation
                if current_price > sma_20 > sma_50:
                    trend_signal = "Bullish (price above short-term averages)"
                elif current_price < sma_20 < sma_50:
                    trend_signal = "Bearish (price below short-term averages)"
                else:
                    trend_signal = "Mixed signals"
                analysis_parts.append(f"- **Trend Signal:** {trend_signal}")
            
            # RSI
            rsi = indicators.get('rsi')
            if rsi is not None:
                rsi_signal = self._get_rsi_signal(rsi)
                analysis_parts.append(f"- **RSI:** {rsi:.2f} ({rsi_signal})")
            
            # Bollinger Bands
            bb_upper = indicators.get('bb_upper')
            bb_lower = indicators.get('bb_lower')
            bb_middle = indicators.get('bb_middle')
            
            if all(x is not None for x in [bb_upper, bb_lower, bb_middle, current_price]):
                if current_price > bb_upper:
                    bb_signal = "Overbought (price above upper band)"
                elif current_price < bb_lower:
                    bb_signal = "Oversold (price below lower band)"
                else:
                    bb_signal = "Normal range"
                analysis_parts.append(f"- **Bollinger Bands:** Upper: ₹{bb_upper:.2f}, Lower: ₹{bb_lower:.2f} ({bb_signal})")
        
        # Support and Resistance
        if support_resistance:
            support = support_resistance.get('support')
            resistance = support_resistance.get('resistance')
            current = support_resistance.get('current')
            
            if all(x is not None for x in [support, resistance, current]):
                analysis_parts.append(f"**Support & Resistance:** Support: ₹{support:.2f}, Resistance: ₹{resistance:.2f}")
                
                # Calculate potential upside/downside
                upside_potential = ((resistance - current) / current) * 100
                downside_risk = ((current - support) / current) * 100
                analysis_parts.append(f"- **Upside Potential:** {upside_potential:.2f}% to resistance")
                analysis_parts.append(f"- **Downside Risk:** {downside_risk:.2f}% to support")
        
        # Momentum
        if momentum is not None:
            momentum_signal = self._get_momentum_signal(momentum)
            analysis_parts.append(f"**Momentum:** {momentum_signal}")
        
        if not analysis_parts:
            return "Technical analysis indicates neutral market sentiment with mixed signals. Key technical indicators suggest a balanced outlook for the stock."
        
        return '\n\n'.join(analysis_parts)
    
    def _get_trend_description(self, trend: str) -> str:
        """Get a descriptive text for trend analysis."""
        trend_descriptions = {
            'Uptrend': 'The stock is showing positive momentum with higher highs and higher lows, indicating bullish sentiment.',
            'Downtrend': 'The stock is experiencing selling pressure with lower highs and lower lows, indicating bearish sentiment.',
            'Sideways': 'The stock is trading in a range-bound pattern with no clear directional bias.',
            'Neutral': 'The stock is showing mixed signals with no clear trend direction.'
        }
        return trend_descriptions.get(trend, f'The stock is showing a {trend.lower()} pattern.')
    
    def _get_rsi_signal(self, rsi: float) -> str:
        """Get RSI signal description."""
        if rsi >= 70:
            return "Overbought - potential sell signal"
        elif rsi <= 30:
            return "Oversold - potential buy signal"
        elif rsi >= 50:
            return "Bullish momentum"
        else:
            return "Bearish momentum"
    
    def _get_momentum_signal(self, momentum: float) -> str:
        """Get momentum signal description."""
        if momentum > 0.5:
            return "Strong positive momentum"
        elif momentum > 0:
            return "Positive momentum"
        elif momentum > -0.5:
            return "Weak negative momentum"
        else:
            return "Strong negative momentum"
