"""
Unit tests for report section generator functions.
Tests functions that generate different sections of the stock research report.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from tools.report_section_generators import (
    create_executive_summary_section,
    create_company_overview_section,
    create_sector_analysis_section,
    create_financial_performance_section,
    create_management_discussion_section
)
from tools.report_formatter_models import ReportSection


class TestCreateExecutiveSummarySection:
    """Test cases for create_executive_summary_section function."""
    
    def test_create_executive_summary_complete(self) -> None:
        """Test creating executive summary with complete data."""
        stock_summary = {
            "company_name": "Test Company",
            "current_price": 100.0,
            "market_cap": 10000000000,
            "pe_ratio": 20.0,
            "sectors": ["Technology"]
        }
        sector_summary = {
            "sector": "Technology",
            "summary": "Sector is growing",
            "outlook": "positive",
            "avg_change": 5.0
        }
        management_summary = {
            "management_outlook": "Positive outlook",
            "summary": "Management is strong"
        }
        
        def mock_format_market_cap(market_cap):
            return "100.00 Cr"
        
        def mock_get_recommendation(stock, sector, mgmt):
            return "**BUY** - Strong fundamentals"
        
        def mock_get_management_outlook(mgmt):
            return "Positive outlook"
        
        section = create_executive_summary_section(
            "TCS",
            stock_summary,
            sector_summary,
            management_summary,
            mock_format_market_cap,
            mock_get_recommendation,
            mock_get_management_outlook
        )
        
        assert isinstance(section, ReportSection)
        assert section.title == "Executive Summary"
        assert "TCS" in section.content
        assert "Test Company" in section.content
        assert "Technology" in section.content
        assert "₹100.0" in section.content
        assert "100.00 Cr" in section.content
    
    def test_create_executive_summary_minimal(self) -> None:
        """Test creating executive summary with minimal data."""
        stock_summary = {}
        sector_summary = {}
        management_summary = {}
        
        def mock_format_market_cap(market_cap):
            return "N/A"
        
        def mock_get_recommendation(stock, sector, mgmt):
            return "**HOLD** - Neutral"
        
        def mock_get_management_outlook(mgmt):
            return "Outlook pending"
        
        section = create_executive_summary_section(
            "TEST",
            stock_summary,
            sector_summary,
            management_summary,
            mock_format_market_cap,
            mock_get_recommendation,
            mock_get_management_outlook
        )
        
        assert isinstance(section, ReportSection)
        assert "TEST" in section.content


class TestCreateCompanyOverviewSection:
    """Test cases for create_company_overview_section function."""
    
    def test_create_company_overview_complete(self) -> None:
        """Test creating company overview with complete data."""
        stock_summary = {
            "company_name": "Test Company",
            "current_price": 100.0,
            "market_cap": 10000000000,
            "pe_ratio": 20.0,
            "pb_ratio": 2.0,
            "eps": 5.0,
            "dividend_yield": 2.5,
            "beta": 1.0,
            "high_52w": 120.0,
            "low_52w": 80.0,
            "volume": 1000000,
            "avg_volume": 900000,
            "change_percent": 5.0,
            "sectors": ["Technology"]
        }
        sector_summary = {"sector": "Technology"}
        
        def mock_format_market_cap(market_cap):
            return "100.00 Cr"
        
        section = create_company_overview_section(
            "TCS",
            stock_summary,
            sector_summary,
            mock_format_market_cap
        )
        
        assert isinstance(section, ReportSection)
        assert section.title == "Company Overview"
        assert "TCS" in section.content
        assert "Test Company" in section.content
        assert "Technology" in section.content
        assert "₹100.0" in section.content
        assert "20.0" in section.content
        assert "2.0" in section.content
    
    def test_create_company_overview_minimal(self) -> None:
        """Test creating company overview with minimal data."""
        stock_summary = {}
        sector_summary = {}
        
        def mock_format_market_cap(market_cap):
            return "N/A"
        
        section = create_company_overview_section(
            "TEST",
            stock_summary,
            sector_summary,
            mock_format_market_cap
        )
        
        assert isinstance(section, ReportSection)
        assert "TEST" in section.content
    
    def test_create_company_overview_no_format_function(self) -> None:
        """Test creating company overview without format function."""
        stock_summary = {"market_cap": 10000000000}
        section = create_company_overview_section("TEST", stock_summary)
        
        assert isinstance(section, ReportSection)
        assert "10000000000" in section.content or "N/A" in section.content


class TestCreateSectorAnalysisSection:
    """Test cases for create_sector_analysis_section function."""
    
    def test_create_sector_analysis_complete(self) -> None:
        """Test creating sector analysis with complete data."""
        sector_summary = {
            "summary": "Sector overview text",
            "trends": ["Trend 1", "Trend 2"],
            "peer_comparison": {
                "sector_leader": {"symbol": "LEAD", "change": 10.0}
            },
            "regulatory_environment": "Regulatory text"
        }
        
        def mock_format_trends(trends):
            return "\n".join([f"- {t}" for t in trends])
        
        def mock_format_peer(peer_data):
            return "Peer comparison text"
        
        def mock_format_regulatory(reg_data):
            return reg_data
        
        section = create_sector_analysis_section(
            sector_summary,
            mock_format_trends,
            mock_format_peer,
            mock_format_regulatory
        )
        
        assert isinstance(section, ReportSection)
        assert section.title == "Sector Analysis"
        assert "Sector overview text" in section.content
        assert "Trend 1" in section.content
        assert "Peer comparison text" in section.content
        assert "Regulatory text" in section.content
    
    def test_create_sector_analysis_minimal(self) -> None:
        """Test creating sector analysis with minimal data."""
        sector_summary = {}
        
        def mock_format_trends(trends):
            return "Default trends"
        
        def mock_format_peer(peer_data):
            return "Default peer comparison"
        
        def mock_format_regulatory(reg_data):
            return "Default regulatory"
        
        section = create_sector_analysis_section(
            sector_summary,
            mock_format_trends,
            mock_format_peer,
            mock_format_regulatory
        )
        
        assert isinstance(section, ReportSection)
        assert "Sector Analysis" in section.title


class TestCreateFinancialPerformanceSection:
    """Test cases for create_financial_performance_section function."""
    
    def test_create_financial_performance_complete(self) -> None:
        """Test creating financial performance with complete data."""
        stock_summary = {
            "revenue_growth": 10.0,
            "profit_growth": 15.0,
            "roe": 20.0,
            "roa": 12.0,
            "pe_ratio": 20.0,
            "pb_ratio": 2.0,
            "ev_ebitda": 15.0,
            "technical_analysis": {
                "indicators": {"rsi": 55.0}
            }
        }
        
        def mock_format_technical(tech_data):
            return "Technical analysis: RSI 55"
        
        section = create_financial_performance_section(
            stock_summary,
            mock_format_technical
        )
        
        assert isinstance(section, ReportSection)
        assert section.title == "Financial Performance"
        assert "10.0" in section.content
        assert "15.0" in section.content
        assert "20.0" in section.content
        assert "Technical analysis: RSI 55" in section.content
    
    def test_create_financial_performance_minimal(self) -> None:
        """Test creating financial performance with minimal data."""
        stock_summary = {}
        
        def mock_format_technical(tech_data):
            return "Technical analysis not available"
        
        section = create_financial_performance_section(
            stock_summary,
            mock_format_technical
        )
        
        assert isinstance(section, ReportSection)
        assert "Financial Performance" in section.title


class TestCreateManagementDiscussionSection:
    """Test cases for create_management_discussion_section function."""
    
    def test_create_management_discussion_complete(self) -> None:
        """Test creating management discussion with complete data."""
        management_summary = {
            "management_outlook": "Positive outlook",
            "strategic_initiatives": ["Initiative 1", "Initiative 2"],
            "growth_opportunities": ["Opportunity 1"],
            "risk_factors": ["Risk 1", "Risk 2"]
        }
        
        section = create_management_discussion_section(management_summary)
        
        assert isinstance(section, ReportSection)
        assert section.title == "Management Discussion & Analysis"
        assert "Positive outlook" in section.content
        assert "Initiative 1" in section.content
        assert "Opportunity 1" in section.content
        assert "Risk 1" in section.content
    
    def test_create_management_discussion_string_lists(self) -> None:
        """Test creating management discussion with string lists."""
        management_summary = {
            "strategic_initiatives": "Already formatted initiatives",
            "growth_opportunities": "Already formatted opportunities",
            "risk_factors": "Already formatted risks"
        }
        
        section = create_management_discussion_section(management_summary)
        
        assert isinstance(section, ReportSection)
        assert "Already formatted initiatives" in section.content
        assert "Already formatted opportunities" in section.content
        assert "Already formatted risks" in section.content
    
    def test_create_management_discussion_minimal(self) -> None:
        """Test creating management discussion with minimal data."""
        management_summary = {}
        
        section = create_management_discussion_section(management_summary)
        
        assert isinstance(section, ReportSection)
        assert "Management Discussion" in section.title
        assert "being processed" in section.content.lower() or "pending" in section.content.lower()
    
    def test_create_management_discussion_fallback_summary(self) -> None:
        """Test creating management discussion with fallback to summary."""
        management_summary = {
            "summary": "Management summary text",
            "management_outlook": "Management discussion highlights..."
        }
        
        section = create_management_discussion_section(management_summary)
        
        assert isinstance(section, ReportSection)
        # Should use summary when outlook is default placeholder
        assert "Management summary text" in section.content or "being processed" in section.content.lower()


