"""
Unit tests for report recommendation helper functions.
Tests scoring functions and recommendation generation.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from tools.report_recommendation_helpers import (
    score_sector_outlook,
    score_financial_performance,
    score_management_quality,
    determine_recommendation,
    create_investment_recommendation_section,
    create_risk_factors_section
)
from tools.report_formatter_models import ReportSection


class TestScoreSectorOutlook:
    """Test cases for score_sector_outlook function."""
    
    def test_score_positive_outlook(self) -> None:
        """Test scoring positive sector outlook."""
        sector_summary = {"outlook": "positive"}
        score = score_sector_outlook(sector_summary)
        assert score == 0.8
    
    def test_score_strong_outlook(self) -> None:
        """Test scoring strong sector outlook."""
        sector_summary = {"outlook": "strong"}
        score = score_sector_outlook(sector_summary)
        assert score == 0.8
    
    def test_score_bullish_outlook(self) -> None:
        """Test scoring bullish sector outlook."""
        sector_summary = {"outlook": "bullish"}
        score = score_sector_outlook(sector_summary)
        assert score == 0.8
    
    def test_score_neutral_outlook(self) -> None:
        """Test scoring neutral sector outlook."""
        sector_summary = {"outlook": "neutral"}
        score = score_sector_outlook(sector_summary)
        assert score == 0.5
    
    def test_score_stable_outlook(self) -> None:
        """Test scoring stable sector outlook."""
        sector_summary = {"outlook": "stable"}
        score = score_sector_outlook(sector_summary)
        assert score == 0.5
    
    def test_score_negative_outlook(self) -> None:
        """Test scoring negative sector outlook."""
        sector_summary = {"outlook": "negative"}
        score = score_sector_outlook(sector_summary)
        assert score == 0.2
    
    def test_score_weak_outlook(self) -> None:
        """Test scoring weak sector outlook."""
        sector_summary = {"outlook": "weak"}
        score = score_sector_outlook(sector_summary)
        assert score == 0.2
    
    def test_score_bearish_outlook(self) -> None:
        """Test scoring bearish sector outlook."""
        sector_summary = {"outlook": "bearish"}
        score = score_sector_outlook(sector_summary)
        assert score == 0.2
    
    def test_score_unknown_outlook(self) -> None:
        """Test scoring unknown sector outlook (default neutral)."""
        sector_summary = {"outlook": "unknown"}
        score = score_sector_outlook(sector_summary)
        assert score == 0.5
    
    def test_score_missing_outlook(self) -> None:
        """Test scoring missing outlook (default neutral)."""
        sector_summary = {}
        score = score_sector_outlook(sector_summary)
        assert score == 0.5


class TestScoreFinancialPerformance:
    """Test cases for score_financial_performance function."""
    
    def test_score_good_pe_ratio(self) -> None:
        """Test scoring financial performance with good P/E ratio."""
        stock_summary = {"pe_ratio": 20}  # Good range
        score = score_financial_performance(stock_summary)
        assert score == 0.7
    
    def test_score_low_pe_ratio(self) -> None:
        """Test scoring financial performance with low P/E ratio."""
        stock_summary = {"pe_ratio": 8}  # Undervalued
        score = score_financial_performance(stock_summary)
        assert score == 0.8
    
    def test_score_high_pe_ratio(self) -> None:
        """Test scoring financial performance with high P/E ratio."""
        stock_summary = {"pe_ratio": 30}  # Overvalued
        score = score_financial_performance(stock_summary)
        assert score == 0.3
    
    def test_score_pe_ratio_at_boundary_low(self) -> None:
        """Test scoring financial performance with P/E at lower boundary."""
        stock_summary = {"pe_ratio": 10}  # At boundary
        score = score_financial_performance(stock_summary)
        assert score == 0.7
    
    def test_score_pe_ratio_at_boundary_high(self) -> None:
        """Test scoring financial performance with P/E at upper boundary."""
        stock_summary = {"pe_ratio": 25}  # At boundary
        score = score_financial_performance(stock_summary)
        assert score == 0.7
    
    def test_score_missing_pe_ratio(self) -> None:
        """Test scoring financial performance with missing P/E ratio."""
        stock_summary = {}
        score = score_financial_performance(stock_summary)
        assert score == 0.5
    
    def test_score_zero_pe_ratio(self) -> None:
        """Test scoring financial performance with zero P/E ratio."""
        stock_summary = {"pe_ratio": 0}
        score = score_financial_performance(stock_summary)
        assert score == 0.5


class TestScoreManagementQuality:
    """Test cases for score_management_quality function."""
    
    def test_score_excellent_management(self) -> None:
        """Test scoring excellent management quality."""
        management_summary = {"management_rating": "excellent"}
        score = score_management_quality(management_summary)
        assert score == 0.8
    
    def test_score_strong_management(self) -> None:
        """Test scoring strong management quality."""
        management_summary = {"management_rating": "strong"}
        score = score_management_quality(management_summary)
        assert score == 0.8
    
    def test_score_good_management(self) -> None:
        """Test scoring good management quality."""
        management_summary = {"management_rating": "good"}
        score = score_management_quality(management_summary)
        assert score == 0.8
    
    def test_score_average_management(self) -> None:
        """Test scoring average management quality."""
        management_summary = {"management_rating": "average"}
        score = score_management_quality(management_summary)
        assert score == 0.5
    
    def test_score_fair_management(self) -> None:
        """Test scoring fair management quality."""
        management_summary = {"management_rating": "fair"}
        score = score_management_quality(management_summary)
        assert score == 0.5
    
    def test_score_poor_management(self) -> None:
        """Test scoring poor management quality."""
        management_summary = {"management_rating": "poor"}
        score = score_management_quality(management_summary)
        assert score == 0.2
    
    def test_score_weak_management(self) -> None:
        """Test scoring weak management quality."""
        management_summary = {"management_rating": "weak"}
        score = score_management_quality(management_summary)
        assert score == 0.2
    
    def test_score_unknown_management(self) -> None:
        """Test scoring unknown management quality (default neutral)."""
        management_summary = {"management_rating": "unknown"}
        score = score_management_quality(management_summary)
        assert score == 0.5
    
    def test_score_missing_management_rating(self) -> None:
        """Test scoring missing management rating (default neutral)."""
        management_summary = {}
        score = score_management_quality(management_summary)
        assert score == 0.5


class TestDetermineRecommendation:
    """Test cases for determine_recommendation function."""
    
    def test_determine_buy_recommendation(self) -> None:
        """Test determining BUY recommendation."""
        sector_summary = {"outlook": "positive"}
        stock_summary = {"pe_ratio": 15}
        management_summary = {"management_rating": "excellent"}
        
        recommendation = determine_recommendation(
            sector_summary,
            stock_summary,
            management_summary,
            score_sector_outlook,
            score_financial_performance,
            score_management_quality
        )
        
        assert recommendation["rating"] == "BUY"
        assert "Strong fundamentals" in recommendation["rationale"]
        assert "Attractive valuation" in recommendation["valuation"]
    
    def test_determine_hold_recommendation(self) -> None:
        """Test determining HOLD recommendation."""
        sector_summary = {"outlook": "neutral"}
        stock_summary = {"pe_ratio": 20}
        management_summary = {"management_rating": "average"}
        
        recommendation = determine_recommendation(
            sector_summary,
            stock_summary,
            management_summary,
            score_sector_outlook,
            score_financial_performance,
            score_management_quality
        )
        
        assert recommendation["rating"] == "HOLD"
        assert "Mixed signals" in recommendation["rationale"]
        assert "Fair valuation" in recommendation["valuation"]
    
    def test_determine_sell_recommendation(self) -> None:
        """Test determining SELL recommendation."""
        sector_summary = {"outlook": "negative"}
        stock_summary = {"pe_ratio": 30}
        management_summary = {"management_rating": "poor"}
        
        recommendation = determine_recommendation(
            sector_summary,
            stock_summary,
            management_summary,
            score_sector_outlook,
            score_financial_performance,
            score_management_quality
        )
        
        assert recommendation["rating"] == "SELL"
        assert "Multiple concerns" in recommendation["rationale"]
        assert "Overvalued" in recommendation["valuation"]
    
    def test_determine_recommendation_borderline_buy(self) -> None:
        """Test determining recommendation at BUY/HOLD borderline."""
        sector_summary = {"outlook": "positive"}
        stock_summary = {"pe_ratio": 20}
        management_summary = {"management_rating": "good"}
        
        recommendation = determine_recommendation(
            sector_summary,
            stock_summary,
            management_summary,
            score_sector_outlook,
            score_financial_performance,
            score_management_quality
        )
        
        assert recommendation["rating"] in ["BUY", "HOLD"]
        assert "rating" in recommendation
        assert "rationale" in recommendation
        assert "valuation" in recommendation
        assert "risk_reward" in recommendation
        assert "catalysts" in recommendation


class TestCreateInvestmentRecommendationSection:
    """Test cases for create_investment_recommendation_section function."""
    
    def test_create_buy_recommendation_section(self) -> None:
        """Test creating BUY recommendation section."""
        sector_summary = {"outlook": "positive"}
        stock_summary = {"current_price": 100, "pe_ratio": 15}
        management_summary = {"management_rating": "excellent"}
        
        def mock_determine(sector, stock, mgmt):
            return {
                "rating": "BUY",
                "rationale": "Strong fundamentals",
                "valuation": "Attractive",
                "risk_reward": "Favorable",
                "catalysts": "• Growth\n• Performance"
            }
        
        section = create_investment_recommendation_section(
            "TCS",
            sector_summary,
            stock_summary,
            management_summary,
            mock_determine
        )
        
        assert isinstance(section, ReportSection)
        assert section.title == "Investment Recommendation"
        assert "BUY" in section.content
        assert "TCS" in section.content
        assert "₹100" in section.content
        assert "For Existing Holders" in section.content
    
    def test_create_hold_recommendation_section(self) -> None:
        """Test creating HOLD recommendation section."""
        sector_summary = {"outlook": "neutral"}
        stock_summary = {"current_price": 100, "pe_ratio": 20}
        management_summary = {"management_rating": "average"}
        
        def mock_determine(sector, stock, mgmt):
            return {
                "rating": "HOLD",
                "rationale": "Mixed signals",
                "valuation": "Fair",
                "risk_reward": "Balanced",
                "catalysts": "• Stability"
            }
        
        section = create_investment_recommendation_section(
            "INFY",
            sector_summary,
            stock_summary,
            management_summary,
            mock_determine
        )
        
        assert isinstance(section, ReportSection)
        assert section.title == "Investment Recommendation"
        assert "HOLD" in section.content
        assert "Maintain Position" in section.content
    
    def test_create_sell_recommendation_section(self) -> None:
        """Test creating SELL recommendation section."""
        sector_summary = {"outlook": "negative"}
        stock_summary = {"current_price": 100, "pe_ratio": 30}
        management_summary = {"management_rating": "poor"}
        
        def mock_determine(sector, stock, mgmt):
            return {
                "rating": "SELL",
                "rationale": "Multiple concerns",
                "valuation": "Overvalued",
                "risk_reward": "Unfavorable",
                "catalysts": "• Headwinds"
            }
        
        section = create_investment_recommendation_section(
            "TEST",
            sector_summary,
            stock_summary,
            management_summary,
            mock_determine
        )
        
        assert isinstance(section, ReportSection)
        assert section.title == "Investment Recommendation"
        assert "SELL" in section.content
        assert "Consider Exiting" in section.content
    
    def test_create_recommendation_section_no_price(self) -> None:
        """Test creating recommendation section without price."""
        sector_summary = {"outlook": "positive"}
        stock_summary = {"pe_ratio": 15}
        management_summary = {"management_rating": "excellent"}
        
        def mock_determine(sector, stock, mgmt):
            return {
                "rating": "BUY",
                "rationale": "Strong",
                "valuation": "Attractive",
                "risk_reward": "Favorable",
                "catalysts": "• Growth"
            }
        
        section = create_investment_recommendation_section(
            "TCS",
            sector_summary,
            stock_summary,
            management_summary,
            mock_determine
        )
        
        assert isinstance(section, ReportSection)
        assert "N/A" in section.content or "0" in section.content


class TestCreateRiskFactorsSection:
    """Test cases for create_risk_factors_section function."""
    
    def test_create_risk_factors_section(self) -> None:
        """Test creating risk factors section."""
        sector_summary = {"risks": ["Sector risk 1", "Sector risk 2"]}
        stock_summary = {
            "risks": ["Company risk 1"],
            "market_risks": ["Market risk 1"]
        }
        management_summary = {}
        
        def mock_format_risk(risks, risk_type):
            if isinstance(risks, list):
                return "\n".join([f"- {risk}" for risk in risks])
            return "Default risks"
        
        section = create_risk_factors_section(
            sector_summary,
            stock_summary,
            management_summary,
            mock_format_risk
        )
        
        assert isinstance(section, ReportSection)
        assert section.title == "Risk Factors"
        assert "Sector Risks" in section.content
        assert "Company-Specific Risks" in section.content
        assert "Market Risks" in section.content
        assert "Regulatory Risks" in section.content
    
    def test_create_risk_factors_section_empty(self) -> None:
        """Test creating risk factors section with empty data."""
        sector_summary = {}
        stock_summary = {}
        management_summary = {}
        
        def mock_format_risk(risks, risk_type):
            return f"Default {risk_type} risks"
        
        section = create_risk_factors_section(
            sector_summary,
            stock_summary,
            management_summary,
            mock_format_risk
        )
        
        assert isinstance(section, ReportSection)
        assert "Sector Risks" in section.content
        assert "Default sector risks" in section.content


