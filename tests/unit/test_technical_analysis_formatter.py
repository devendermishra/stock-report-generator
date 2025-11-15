"""
Unit tests for technical analysis formatter.
Tests TechnicalAnalysisFormatter class and format_technical_analysis function.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from tools.technical_analysis_formatter import (
    TechnicalAnalysisFormatter
)


class TestTechnicalAnalysisFormatter:
    """Test cases for TechnicalAnalysisFormatter class."""
    
    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.formatter = TechnicalAnalysisFormatter()
    
    def test_get_trend_description_uptrend(self) -> None:
        """Test getting trend description for uptrend."""
        result = self.formatter._get_trend_description("Uptrend")
        assert "positive momentum" in result.lower()
        assert "bullish" in result.lower()
    
    def test_get_trend_description_downtrend(self) -> None:
        """Test getting trend description for downtrend."""
        result = self.formatter._get_trend_description("Downtrend")
        assert "selling pressure" in result.lower()
        assert "bearish" in result.lower()
    
    def test_get_trend_description_sideways(self) -> None:
        """Test getting trend description for sideways."""
        result = self.formatter._get_trend_description("Sideways")
        assert "range-bound" in result.lower()
    
    def test_get_trend_description_neutral(self) -> None:
        """Test getting trend description for neutral."""
        result = self.formatter._get_trend_description("Neutral")
        assert "mixed signals" in result.lower()
    
    def test_get_trend_description_unknown(self) -> None:
        """Test getting trend description for unknown trend."""
        result = self.formatter._get_trend_description("Unknown")
        assert "unknown" in result.lower()
    
    def test_get_rsi_signal_overbought(self) -> None:
        """Test getting RSI signal for overbought condition."""
        result = self.formatter._get_rsi_signal(75)
        assert "Overbought" in result
        assert "sell signal" in result.lower()
    
    def test_get_rsi_signal_oversold(self) -> None:
        """Test getting RSI signal for oversold condition."""
        result = self.formatter._get_rsi_signal(25)
        assert "Oversold" in result
        assert "buy signal" in result.lower()
    
    def test_get_rsi_signal_bullish_momentum(self) -> None:
        """Test getting RSI signal for bullish momentum."""
        result = self.formatter._get_rsi_signal(60)
        assert "Bullish momentum" in result
    
    def test_get_rsi_signal_bearish_momentum(self) -> None:
        """Test getting RSI signal for bearish momentum."""
        result = self.formatter._get_rsi_signal(40)
        assert "Bearish momentum" in result
    
    def test_get_momentum_signal_strong_positive(self) -> None:
        """Test getting momentum signal for strong positive momentum."""
        result = self.formatter._get_momentum_signal(0.6)
        assert "Strong positive momentum" in result
    
    def test_get_momentum_signal_positive(self) -> None:
        """Test getting momentum signal for positive momentum."""
        result = self.formatter._get_momentum_signal(0.1)
        assert "Positive momentum" in result
    
    def test_get_momentum_signal_weak_negative(self) -> None:
        """Test getting momentum signal for weak negative momentum."""
        result = self.formatter._get_momentum_signal(-0.05)
        assert "Weak negative momentum" in result
    
    def test_get_momentum_signal_strong_negative(self) -> None:
        """Test getting momentum signal for strong negative momentum."""
        result = self.formatter._get_momentum_signal(-0.6)
        assert "Strong negative momentum" in result
    
    def test_format_technical_analysis_complete(self) -> None:
        """Test formatting complete technical analysis data."""
        technical_data = {
            "trend_analysis": "Uptrend",
            "indicators": {
                "sma_20": 100.0,
                "sma_50": 95.0,
                "sma_200": 90.0,
                "current_price": 105.0,
                "rsi": 65.0,
                "bb_upper": 110.0,
                "bb_lower": 90.0,
                "bb_middle": 100.0
            },
            "support_resistance": {
                "support": 95.0,
                "resistance": 115.0,
                "current": 105.0
            },
            "momentum": 0.2
        }
        result = self.formatter.format_technical_analysis(technical_data)
        
        assert "Trend Analysis" in result
        assert "Key Technical Indicators" in result
        assert "Moving Averages" in result
        assert "RSI" in result
        assert "Bollinger Bands" in result
        assert "Support & Resistance" in result
        assert "Momentum" in result
        assert "100.00" in result
        assert "65.00" in result
    
    def test_format_technical_analysis_bullish_trend(self) -> None:
        """Test formatting technical analysis with bullish trend."""
        technical_data = {
            "indicators": {
                "sma_20": 100.0,
                "sma_50": 95.0,
                "sma_200": 90.0,
                "current_price": 105.0
            }
        }
        result = self.formatter.format_technical_analysis(technical_data)
        
        assert "Bullish" in result or "price above" in result.lower()
    
    def test_format_technical_analysis_bearish_trend(self) -> None:
        """Test formatting technical analysis with bearish trend."""
        technical_data = {
            "indicators": {
                "sma_20": 100.0,
                "sma_50": 105.0,
                "sma_200": 110.0,
                "current_price": 95.0
            }
        }
        result = self.formatter.format_technical_analysis(technical_data)
        
        assert "Bearish" in result or "price below" in result.lower()
    
    def test_format_technical_analysis_bollinger_overbought(self) -> None:
        """Test formatting technical analysis with overbought Bollinger Bands."""
        technical_data = {
            "indicators": {
                "bb_upper": 100.0,
                "bb_lower": 90.0,
                "bb_middle": 95.0,
                "current_price": 105.0
            }
        }
        result = self.formatter.format_technical_analysis(technical_data)
        
        assert "Overbought" in result or "above upper" in result.lower()
    
    def test_format_technical_analysis_bollinger_oversold(self) -> None:
        """Test formatting technical analysis with oversold Bollinger Bands."""
        technical_data = {
            "indicators": {
                "bb_upper": 100.0,
                "bb_lower": 90.0,
                "bb_middle": 95.0,
                "current_price": 85.0
            }
        }
        result = self.formatter.format_technical_analysis(technical_data)
        
        assert "Oversold" in result or "below lower" in result.lower()
    
    def test_format_technical_analysis_support_resistance(self) -> None:
        """Test formatting technical analysis with support and resistance."""
        technical_data = {
            "support_resistance": {
                "support": 90.0,
                "resistance": 110.0,
                "current": 100.0
            }
        }
        result = self.formatter.format_technical_analysis(technical_data)
        
        assert "Support" in result
        assert "Resistance" in result
        assert "Upside Potential" in result
        assert "Downside Risk" in result
        assert "90.00" in result
        assert "110.00" in result
    
    def test_format_technical_analysis_empty(self) -> None:
        """Test formatting empty technical analysis data."""
        result = self.formatter.format_technical_analysis({})
        assert "neutral market sentiment" in result.lower()
        assert "mixed signals" in result.lower()
    
    def test_format_technical_analysis_none(self) -> None:
        """Test formatting None technical analysis data."""
        result = self.formatter.format_technical_analysis(None)
        assert "neutral market sentiment" in result.lower()
    
    def test_format_technical_analysis_partial_indicators(self) -> None:
        """Test formatting technical analysis with partial indicators."""
        technical_data = {
            "indicators": {
                "rsi": 55.0
            }
        }
        result = self.formatter.format_technical_analysis(technical_data)
        
        assert "RSI" in result
        assert "55.00" in result


class TestFormatTechnicalAnalysisFunction:
    """Test cases for format_technical_analysis function (tool wrapper)."""
    
    def test_format_technical_analysis_function_success(self) -> None:
        """Test format_technical_analysis function with valid data."""
        from tools.technical_analysis_formatter import _formatter
        
        technical_data = {
            "indicators": {
                "rsi": 60.0,
                "current_price": 100.0
            }
        }
        # Test the underlying formatter since format_technical_analysis is a LangChain tool
        result = _formatter.format_technical_analysis(technical_data)
        
        assert isinstance(result, str)
        assert "RSI" in result or "neutral" in result.lower()
    
    def test_format_technical_analysis_function_empty(self) -> None:
        """Test format_technical_analysis function with empty data."""
        from tools.technical_analysis_formatter import _formatter
        
        # Test the underlying formatter since format_technical_analysis is a LangChain tool
        result = _formatter.format_technical_analysis({})
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_format_technical_analysis_function_error_handling(self) -> None:
        """Test format_technical_analysis function error handling."""
        from tools.technical_analysis_formatter import _formatter
        
        # Test the underlying formatter since format_technical_analysis is a LangChain tool
        result = _formatter.format_technical_analysis(None)
        assert isinstance(result, str)

