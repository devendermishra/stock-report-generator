"""
Unit tests for report formatter utilities.
Tests formatting functions for market cap, recommendations, and other utilities.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from tools.report_formatter_utils import ReportFormatterUtils


class TestFormatMarketCap:
    """Test cases for format_market_cap function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.utils = ReportFormatterUtils()
    
    def test_format_large_cap(self):
        """Test formatting large market cap (> 1 lakh crores)."""
        # 2 lakh crores = 2e12
        market_cap = 20000000000000
        result = self.utils.format_market_cap(market_cap)
        
        assert "L Cr" in result
        assert "2.00" in result
    
    def test_format_medium_cap(self):
        """Test formatting medium market cap (1000-100000 crores)."""
        # 5 thousand crores = 5e10
        market_cap = 50000000000
        result = self.utils.format_market_cap(market_cap)
        
        assert "K Cr" in result
        assert "5.00" in result
    
    def test_format_small_cap(self):
        """Test formatting small market cap (< 1000 crores)."""
        # 500 crores = 5e9
        market_cap = 5000000000
        result = self.utils.format_market_cap(market_cap)
        
        assert "Cr" in result
        assert "500.00" in result
    
    def test_format_none_value(self):
        """Test formatting None value."""
        result = self.utils.format_market_cap(None)
        assert result == 'N/A'
    
    def test_format_na_string(self):
        """Test formatting 'N/A' string."""
        result = self.utils.format_market_cap('N/A')
        assert result == 'N/A'
    
    def test_format_string_number(self):
        """Test formatting string number."""
        result = self.utils.format_market_cap("10000000000")
        assert "Cr" in result
    
    def test_format_invalid_value(self):
        """Test formatting invalid value."""
        result = self.utils.format_market_cap("invalid")
        assert result == 'N/A'


class TestFormatTrendsList:
    """Test cases for format_trends_list function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.utils = ReportFormatterUtils()
    
    def test_format_trends_list(self):
        """Test formatting list of trends."""
        trends = ["Trend 1", "Trend 2", "Trend 3"]
        result = self.utils.format_trends_list(trends)
        
        assert "- Trend 1" in result
        assert "- Trend 2" in result
        assert "- Trend 3" in result
    
    def test_format_empty_trends(self):
        """Test formatting empty trends list."""
        result = self.utils.format_trends_list([])
        
        assert "Key sector trends" in result
        assert "digital transformation" in result.lower()
    
    def test_format_string_trend(self):
        """Test formatting when trends is already a string."""
        trends = "Already formatted trend text"
        result = self.utils.format_trends_list(trends)
        
        assert result == trends
    
    def test_format_trends_with_whitespace(self):
        """Test formatting trends with extra whitespace."""
        trends = ["  Trend 1  ", "  Trend 2  "]
        result = self.utils.format_trends_list(trends)
        
        assert "- Trend 1" in result
        assert "- Trend 2" in result
        assert "  " not in result


class TestFormatRiskList:
    """Test cases for format_risk_list function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.utils = ReportFormatterUtils()
    
    def test_format_sector_risks(self):
        """Test formatting sector risks."""
        risks = ["Risk 1", "Risk 2"]
        result = self.utils.format_risk_list(risks, "sector")
        
        assert "- Risk 1" in result
        assert "- Risk 2" in result
    
    def test_format_empty_risks_sector(self):
        """Test formatting empty risks with sector type."""
        result = self.utils.format_risk_list([], "sector")
        
        assert "Economic volatility" in result
        assert "Regulatory changes" in result
    
    def test_format_empty_risks_company(self):
        """Test formatting empty risks with company type."""
        result = self.utils.format_risk_list([], "company")
        
        assert "Management execution" in result
        assert "Financial performance" in result
    
    def test_format_empty_risks_market(self):
        """Test formatting empty risks with market type."""
        result = self.utils.format_risk_list([], "market")
        
        assert "Interest rate" in result
        assert "Currency exchange" in result
    
    def test_format_risks_string(self):
        """Test formatting when risks is already a string."""
        risks = "Already formatted risk text"
        result = self.utils.format_risk_list(risks, "sector")
        
        assert result == risks


class TestScoreFunctions:
    """Test cases for scoring functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.utils = ReportFormatterUtils()
    
    def test_score_sector_outlook_positive(self):
        """Test scoring positive sector outlook."""
        sector_summary = {"outlook": "positive"}
        score = self.utils._score_sector_outlook(sector_summary)
        
        assert score == 0.8
    
    def test_score_sector_outlook_neutral(self):
        """Test scoring neutral sector outlook."""
        sector_summary = {"outlook": "neutral"}
        score = self.utils._score_sector_outlook(sector_summary)
        
        assert score == 0.5
    
    def test_score_sector_outlook_negative(self):
        """Test scoring negative sector outlook."""
        sector_summary = {"outlook": "negative"}
        score = self.utils._score_sector_outlook(sector_summary)
        
        assert score == 0.2
    
    def test_score_financial_performance_good_pe(self):
        """Test scoring financial performance with good P/E ratio."""
        stock_summary = {"pe_ratio": 20}  # Good range
        score = self.utils._score_financial_performance(stock_summary)
        
        assert score == 0.7
    
    def test_score_financial_performance_low_pe(self):
        """Test scoring financial performance with low P/E ratio."""
        stock_summary = {"pe_ratio": 8}  # Undervalued
        score = self.utils._score_financial_performance(stock_summary)
        
        assert score == 0.8
    
    def test_score_financial_performance_high_pe(self):
        """Test scoring financial performance with high P/E ratio."""
        stock_summary = {"pe_ratio": 30}  # Overvalued
        score = self.utils._score_financial_performance(stock_summary)
        
        assert score == 0.3
    
    def test_score_management_quality_excellent(self):
        """Test scoring excellent management quality."""
        management_summary = {"management_rating": "excellent"}
        score = self.utils._score_management_quality(management_summary)
        
        assert score == 0.8
    
    def test_score_management_quality_average(self):
        """Test scoring average management quality."""
        management_summary = {"management_rating": "average"}
        score = self.utils._score_management_quality(management_summary)
        
        assert score == 0.5
    
    def test_score_management_quality_poor(self):
        """Test scoring poor management quality."""
        management_summary = {"management_rating": "poor"}
        score = self.utils._score_management_quality(management_summary)
        
        assert score == 0.2
