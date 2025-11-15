"""
Unit tests for report formatter helper classes.
Tests FinancialFormatter, TechnicalFormatter, PeerAnalysisFormatter, and SWOTFormatter.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from tools.report_formatter_helpers import (
    FinancialFormatter,
    TechnicalFormatter,
    PeerAnalysisFormatter,
    SWOTFormatter
)


class TestFinancialFormatter:
    """Test cases for FinancialFormatter class."""
    
    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.formatter = FinancialFormatter()
    
    def test_format_market_cap_large(self) -> None:
        """Test formatting large market cap (> 1 lakh crores)."""
        market_cap = 20000000000000  # 20 lakh crores (2000000 crores)
        result = self.formatter.format_market_cap(market_cap)
        
        assert "L Cr" in result
        assert "20.00" in result
    
    def test_format_market_cap_medium(self) -> None:
        """Test formatting medium market cap (1000-100000 crores)."""
        market_cap = 50000000000  # 5 thousand crores
        result = self.formatter.format_market_cap(market_cap)
        
        assert "K Cr" in result
        assert "5.00" in result
    
    def test_format_market_cap_small(self) -> None:
        """Test formatting small market cap (< 1000 crores)."""
        market_cap = 5000000000  # 500 crores
        result = self.formatter.format_market_cap(market_cap)
        
        assert "Cr" in result
        assert "500.00" in result
    
    def test_format_market_cap_none(self) -> None:
        """Test formatting None value."""
        result = self.formatter.format_market_cap(None)
        assert result == 'N/A'
    
    def test_format_market_cap_na_string(self) -> None:
        """Test formatting 'N/A' string."""
        result = self.formatter.format_market_cap('N/A')
        assert result == 'N/A'
    
    def test_format_market_cap_string_number(self) -> None:
        """Test formatting string number."""
        result = self.formatter.format_market_cap("10000000000")
        assert "Cr" in result
    
    def test_format_market_cap_invalid(self) -> None:
        """Test formatting invalid value."""
        result = self.formatter.format_market_cap("invalid")
        assert result == 'N/A'
    
    def test_get_financial_health_indicators_undervalued(self) -> None:
        """Test financial health indicators for undervalued stock."""
        indicators = self.formatter.get_financial_health_indicators(
            pe_ratio=12, pb_ratio=0.8, eps=10.5, dividend_yield=2.5, beta=0.7
        )
        
        assert "Undervalued" in indicators
        assert "Trading Below Book" in indicators
        assert "Profitable" in indicators
        assert "Low Volatility" in indicators
    
    def test_get_financial_health_indicators_overvalued(self) -> None:
        """Test financial health indicators for overvalued stock."""
        indicators = self.formatter.get_financial_health_indicators(
            pe_ratio=30, pb_ratio=4.0, eps=5.0, dividend_yield=0.5, beta=1.5
        )
        
        assert "Overvalued" in indicators
        assert "High Book Value" in indicators
        assert "High Volatility" in indicators
    
    def test_get_financial_health_indicators_fairly_valued(self) -> None:
        """Test financial health indicators for fairly valued stock."""
        indicators = self.formatter.get_financial_health_indicators(
            pe_ratio=20, pb_ratio=2.0, eps=8.0, dividend_yield=2.0, beta=1.0
        )
        
        assert "Fairly Valued" in indicators
        assert "Market Volatility" in indicators
    
    def test_get_financial_health_indicators_negative_eps(self) -> None:
        """Test financial health indicators with negative EPS."""
        indicators = self.formatter.get_financial_health_indicators(
            pe_ratio=15, pb_ratio=1.5, eps=-2.0, dividend_yield=0, beta=1.0
        )
        
        assert "Loss Making" in indicators
    
    def test_get_financial_health_indicators_high_dividend(self) -> None:
        """Test financial health indicators with high dividend yield."""
        indicators = self.formatter.get_financial_health_indicators(
            pe_ratio=18, pb_ratio=2.0, eps=10.0, dividend_yield=4.5, beta=0.9
        )
        
        assert "High Dividend Yield" in indicators
    
    def test_get_financial_health_indicators_none_values(self) -> None:
        """Test financial health indicators with None values."""
        indicators = self.formatter.get_financial_health_indicators(
            pe_ratio=None, pb_ratio=None, eps=None, dividend_yield=None, beta=None
        )
        
        assert "not available" in indicators


class TestTechnicalFormatter:
    """Test cases for TechnicalFormatter class."""
    
    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.formatter = TechnicalFormatter()
    
    def test_get_rsi_signal_overbought(self) -> None:
        """Test RSI signal for overbought condition."""
        result = self.formatter.get_rsi_signal(75)
        assert result == "Overbought"
    
    def test_get_rsi_signal_oversold(self) -> None:
        """Test RSI signal for oversold condition."""
        result = self.formatter.get_rsi_signal(25)
        assert result == "Oversold"
    
    def test_get_rsi_signal_neutral(self) -> None:
        """Test RSI signal for neutral condition."""
        result = self.formatter.get_rsi_signal(50)
        assert result == "Neutral"
    
    def test_get_momentum_signal_strong_bullish(self) -> None:
        """Test momentum signal for strong bullish."""
        result = self.formatter.get_momentum_signal(0.15)
        assert result == "Strong Bullish"
    
    def test_get_momentum_signal_bullish(self) -> None:
        """Test momentum signal for bullish."""
        result = self.formatter.get_momentum_signal(0.05)
        assert result == "Bullish"
    
    def test_get_momentum_signal_bearish(self) -> None:
        """Test momentum signal for bearish."""
        result = self.formatter.get_momentum_signal(-0.05)
        assert result == "Bearish"
    
    def test_get_momentum_signal_strong_bearish(self) -> None:
        """Test momentum signal for strong bearish."""
        result = self.formatter.get_momentum_signal(-0.15)
        assert result == "Strong Bearish"
    
    def test_format_technical_analysis_complete(self) -> None:
        """Test formatting complete technical analysis data."""
        technical_data = {
            'sma_20': 100.0,
            'sma_50': 95.0,
            'sma_200': 90.0,
            'rsi': 55.0,
            'bb_upper': 110.0,
            'bb_lower': 90.0,
            'bb_middle': 100.0,
            'current_price': 105.0,
            'support': 95.0,
            'resistance': 115.0
        }
        result = self.formatter.format_technical_analysis(technical_data)
        
        assert "Moving Averages" in result
        assert "RSI" in result
        assert "Bollinger Bands" in result
        assert "Support & Resistance" in result
        assert "Upside Potential" in result
        assert "Downside Risk" in result
    
    def test_format_technical_analysis_empty(self) -> None:
        """Test formatting empty technical analysis data."""
        result = self.formatter.format_technical_analysis({})
        assert "not available" in result
    
    def test_format_technical_analysis_none(self) -> None:
        """Test formatting None technical analysis data."""
        result = self.formatter.format_technical_analysis(None)
        assert "not available" in result
    
    def test_format_technical_analysis_partial(self) -> None:
        """Test formatting partial technical analysis data."""
        technical_data = {
            'rsi': 65.0,
            'current_price': 100.0
        }
        result = self.formatter.format_technical_analysis(technical_data)
        
        assert "RSI" in result
        assert "65.00" in result


class TestPeerAnalysisFormatter:
    """Test cases for PeerAnalysisFormatter class."""
    
    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.formatter = PeerAnalysisFormatter()
    
    def test_format_peer_comparison_table(self) -> None:
        """Test formatting peer comparison table."""
        peer_data = {
            'peers': [
                {
                    'name': 'Company A',
                    'price': 100.0,
                    'pe_ratio': 20.0,
                    'pb_ratio': 2.0,
                    'market_cap': 10000000000,
                    'change_percent': 5.0
                },
                {
                    'name': 'Company B',
                    'price': 200.0,
                    'pe_ratio': 25.0,
                    'pb_ratio': 3.0,
                    'market_cap': 20000000000,
                    'change_percent': -2.0
                }
            ]
        }
        result = self.formatter.format_peer_comparison_table(peer_data)
        
        assert "Company" in result
        assert "Price" in result
        assert "P/E" in result
        assert "Company A" in result
        assert "Company B" in result
        assert "100.00" in result
    
    def test_format_peer_comparison_table_empty(self) -> None:
        """Test formatting empty peer comparison table."""
        result = self.formatter.format_peer_comparison_table({})
        assert "not available" in result
    
    def test_format_peer_comparison_table_no_peers(self) -> None:
        """Test formatting peer comparison table with no peers."""
        peer_data = {'peers': []}
        result = self.formatter.format_peer_comparison_table(peer_data)
        assert "No peer data available" in result
    
    def test_format_peer_performance_summary(self) -> None:
        """Test formatting peer performance summary."""
        peer_data = {
            'peers': [
                {'name': 'Company A', 'change_percent': 10.0},
                {'name': 'Company B', 'change_percent': 5.0},
                {'name': 'Company C', 'change_percent': -3.0}
            ]
        }
        result = self.formatter.format_peer_performance_summary(peer_data)
        
        assert "Total Peers Analyzed" in result
        assert "Positive Performers" in result
        assert "Average Performance" in result
        assert "Best Performer" in result
        assert "Worst Performer" in result
        assert "Company A" in result
        assert "Company C" in result
    
    def test_format_peer_performance_summary_empty(self) -> None:
        """Test formatting empty peer performance summary."""
        result = self.formatter.format_peer_performance_summary({})
        assert "not available" in result
    
    def test_format_market_cap_private(self) -> None:
        """Test private _format_market_cap method."""
        result = self.formatter._format_market_cap(10000000000)  # 1000 crores
        assert "K Cr" in result or "Cr" in result
        
        result = self.formatter._format_market_cap(10000000000000)  # 100000 crores = 1 lakh crores
        assert "L Cr" in result
        
        result = self.formatter._format_market_cap(None)
        assert result == 'N/A'


class TestSWOTFormatter:
    """Test cases for SWOTFormatter class."""
    
    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.formatter = SWOTFormatter()
    
    def test_format_swot_list(self) -> None:
        """Test formatting SWOT list."""
        items = ["Strength 1", "Strength 2", "Strength 3"]
        result = self.formatter.format_swot_list(items, "Strengths")
        
        assert "1. Strength 1" in result
        assert "2. Strength 2" in result
        assert "3. Strength 3" in result
    
    def test_format_swot_list_empty(self) -> None:
        """Test formatting empty SWOT list."""
        result = self.formatter.format_swot_list([], "Strengths")
        assert "No strengths identified" in result
    
    def test_format_swot_list_more_than_five(self) -> None:
        """Test formatting SWOT list with more than 5 items."""
        items = [f"Item {i}" for i in range(1, 8)]
        result = self.formatter.format_swot_list(items, "Strengths")
        
        assert "1. Item 1" in result
        assert "5. Item 5" in result
        assert "and 2 more strengths" in result
    
    def test_format_swot_analysis_complete(self) -> None:
        """Test formatting complete SWOT analysis."""
        swot_data = {
            'strengths': ['Strength 1', 'Strength 2'],
            'weaknesses': ['Weakness 1'],
            'opportunities': ['Opportunity 1', 'Opportunity 2'],
            'threats': ['Threat 1'],
            'summary': 'Overall SWOT summary'
        }
        result = self.formatter.format_swot_analysis(swot_data)
        
        assert "Strengths" in result
        assert "Weaknesses" in result
        assert "Opportunities" in result
        assert "Threats" in result
        assert "Summary" in result
        assert "Strength 1" in result
        assert "Overall SWOT summary" in result
    
    def test_format_swot_analysis_partial(self) -> None:
        """Test formatting partial SWOT analysis."""
        swot_data = {
            'strengths': ['Strength 1'],
            'weaknesses': []
        }
        result = self.formatter.format_swot_analysis(swot_data)
        
        assert "Strengths" in result
        assert "Strength 1" in result
    
    def test_format_swot_analysis_empty(self) -> None:
        """Test formatting empty SWOT analysis."""
        result = self.formatter.format_swot_analysis({})
        assert "not available" in result
    
    def test_format_swot_analysis_none(self) -> None:
        """Test formatting None SWOT analysis."""
        result = self.formatter.format_swot_analysis(None)
        assert "not available" in result

