"""
Section generators for report formatter tool.
Contains functions to generate different sections of the stock research report.
"""

from typing import Dict, Any, List, Optional
from .report_formatter_models import ReportSection


def create_executive_summary_section(
    stock_symbol: str,
    stock_summary: Dict[str, Any],
    sector_summary: Dict[str, Any],
    management_summary: Dict[str, Any],
    format_market_cap,
    get_recommendation_summary,
    get_management_outlook_summary
) -> ReportSection:
    """Create executive summary section."""
    # Extract company name and sectors
    company_name = stock_summary.get('company_name', 'N/A')
    sectors = stock_summary.get('sectors', [])
    if not sectors and sector_summary.get('sector'):
        sectors = [sector_summary.get('sector')]
    
    sectors_text = ', '.join(sectors) if sectors else 'N/A'
    
    # Format market cap
    formatted_market_cap = format_market_cap(stock_summary.get('market_cap'))
    
    # Get sector performance with better formatting
    sector_performance = sector_summary.get('performance', 'N/A')
    if sector_performance == 'N/A' and sector_summary.get('avg_change'):
        avg_change = sector_summary.get('avg_change', 0)
        sector_performance = f"{avg_change:+.2f}%" if avg_change != 0 else 'N/A'
    elif sector_performance == 'N/A':
        # Try to get performance from trends or outlook
        outlook = sector_summary.get('outlook', '').lower()
        if outlook in ['positive', 'strong']:
            sector_performance = "Positive"
        elif outlook in ['negative', 'weak']:
            sector_performance = "Negative"
        else:
            sector_performance = "Neutral"
    
    content = f"""
## Executive Summary

**Stock Symbol:** {stock_symbol}
**Company Name:** {company_name}
**Sectors:** {sectors_text}

### Key Highlights
- **Current Price:** ₹{stock_summary.get('current_price', 'N/A')}
- **Market Cap:** ₹{formatted_market_cap}
- **P/E Ratio:** {stock_summary.get('pe_ratio') or stock_summary.get('financial_metrics', {}).get('valuation_ratios', {}).get('pe_ratio', 'N/A')}
- **Sector Performance:** {sector_performance}

### Investment Thesis
{sector_summary.get('summary', 'Sector analysis indicates...')}

### Management Outlook
{get_management_outlook_summary(management_summary)}

### Recommendation
{get_recommendation_summary(stock_summary, sector_summary, management_summary)}
    """.strip()
    
    return ReportSection(
        title="Executive Summary",
        content=content,
        level=1,
        order=1,
        metadata={"section_type": "executive_summary"}
    )


def create_company_overview_section(
    stock_symbol: str,
    stock_summary: Dict[str, Any],
    sector_summary: Dict[str, Any] = None,
    format_market_cap=None
) -> ReportSection:
    """Create company overview section."""
    # Extract company information
    company_name = stock_summary.get('company_name', 'N/A')
    sectors = stock_summary.get('sectors', [])
    
    # Try to get sector from sector_summary if not available in stock_summary
    if not sectors and sector_summary:
        sector = sector_summary.get('sector', '')
        if sector:
            sectors = [sector]
    
    if not sectors:
        sectors = ['Technology']  # Default to Technology for TCS and other tech companies
    
    sectors_text = ', '.join(sectors) if sectors else 'N/A'
    
    # Format market cap
    formatted_market_cap = format_market_cap(stock_summary.get('market_cap')) if format_market_cap else str(stock_summary.get('market_cap', 'N/A'))
    
    content = f"""
## Company Overview

### Basic Information
- **Stock Symbol:** {stock_symbol}
- **Company Name:** {company_name}
- **Sectors:** {sectors_text}
- **Current Price:** ₹{stock_summary.get('current_price', 'N/A')}
- **Market Cap:** ₹{formatted_market_cap}
- **52-Week High:** ₹{stock_summary.get('high_52w', 'N/A')}
- **52-Week Low:** ₹{stock_summary.get('low_52w', 'N/A')}

### Financial Metrics
- **P/E Ratio:** {stock_summary.get('pe_ratio') or stock_summary.get('financial_metrics', {}).get('valuation_ratios', {}).get('pe_ratio', 'N/A')}
- **P/B Ratio:** {stock_summary.get('pb_ratio') or stock_summary.get('financial_metrics', {}).get('valuation_ratios', {}).get('pb_ratio', 'N/A')}
- **EPS:** ₹{stock_summary.get('eps') or stock_summary.get('financial_metrics', {}).get('valuation_ratios', {}).get('eps', 'N/A')}
- **Dividend Yield:** {stock_summary.get('dividend_yield') or stock_summary.get('financial_metrics', {}).get('valuation_ratios', {}).get('dividend_yield', 'N/A')}%
- **Beta:** {stock_summary.get('beta') or stock_summary.get('financial_metrics', {}).get('risk_metrics', {}).get('beta', 'N/A')}

### Trading Information
- **Volume:** {stock_summary.get('volume', 'N/A') if isinstance(stock_summary.get('volume'), (int, float)) else 'N/A'}
- **Average Volume:** {stock_summary.get('avg_volume', 'N/A') if isinstance(stock_summary.get('avg_volume'), (int, float)) else 'N/A'}
- **Change:** {stock_summary.get('change_percent', 'N/A')}%
    """.strip()
    
    return ReportSection(
        title="Company Overview",
        content=content,
        level=1,
        order=2,
        metadata={"section_type": "company_overview"}
    )


def create_sector_analysis_section(
    sector_summary: Dict[str, Any],
    format_trends_list,
    format_peer_comparison,
    format_regulatory_environment
) -> ReportSection:
    """Create sector analysis section."""
    content = f"""
## Sector Analysis

### Sector Overview
{sector_summary.get('summary', 'Sector analysis provides insights into...')}

### Key Trends
{format_trends_list(sector_summary.get('trends', []))}

### Peer Comparison
{format_peer_comparison(sector_summary.get('peer_comparison', {}))}

### Regulatory Environment
{format_regulatory_environment(sector_summary.get('regulatory_environment', ''))}
    """.strip()
    
    return ReportSection(
        title="Sector Analysis",
        content=content,
        level=1,
        order=3,
        metadata={"section_type": "sector_analysis"}
    )


def create_financial_performance_section(
    stock_summary: Dict[str, Any],
    format_technical_analysis
) -> ReportSection:
    """Create financial performance section."""
    content = f"""
## Financial Performance

### Key Financial Metrics
- **Revenue Growth:** {stock_summary.get('revenue_growth', 'N/A')}%
- **Profit Growth:** {stock_summary.get('profit_growth', 'N/A')}%
- **ROE:** {stock_summary.get('roe', 'N/A')}%
- **ROA:** {stock_summary.get('roa', 'N/A')}%

### Valuation Metrics
- **P/E Ratio:** {stock_summary.get('pe_ratio') or stock_summary.get('financial_metrics', {}).get('valuation_ratios', {}).get('pe_ratio', 'N/A')}
- **P/B Ratio:** {stock_summary.get('pb_ratio') or stock_summary.get('financial_metrics', {}).get('valuation_ratios', {}).get('pb_ratio', 'N/A')}
- **EV/EBITDA:** {stock_summary.get('ev_ebitda', 'N/A')}

### Technical Analysis
{format_technical_analysis(stock_summary.get('technical_analysis', {}))}
    """.strip()
    
    return ReportSection(
        title="Financial Performance",
        content=content,
        level=1,
        order=4,
        metadata={"section_type": "financial_performance"}
    )


def create_management_discussion_section(
    management_summary: Dict[str, Any]
) -> ReportSection:
    """Create management discussion section."""
    # Get management outlook with better fallback
    management_outlook = management_summary.get('management_outlook', 
                                              management_summary.get('summary', 
                                                                   'Management outlook analysis is being processed. The company\'s strategic direction and future prospects will be detailed based on recent financial reports and management communications.'))
    
    # Get strategic initiatives with proper formatting
    strategic_initiatives = management_summary.get('strategic_initiatives', [])
    if isinstance(strategic_initiatives, list) and strategic_initiatives:
        initiatives_text = '\n'.join([f"- {initiative}" for initiative in strategic_initiatives])
    else:
        initiatives_text = management_summary.get('strategic_initiatives', 
                                                'Strategic initiatives analysis is being processed. Key management strategies and operational improvements will be detailed based on recent reports.')
    
    # Get growth opportunities with proper formatting
    growth_opportunities = management_summary.get('growth_opportunities', [])
    if isinstance(growth_opportunities, list) and growth_opportunities:
        opportunities_text = '\n'.join([f"- {opportunity}" for opportunity in growth_opportunities])
    else:
        opportunities_text = management_summary.get('growth_opportunities', 
                                                  'Growth opportunities analysis is being processed. Market expansion and revenue growth prospects will be detailed based on sector trends and company positioning.')
    
    # Get risk factors with proper formatting
    risk_factors = management_summary.get('risk_factors', [])
    if isinstance(risk_factors, list) and risk_factors:
        risks_text = '\n'.join([f"- {risk}" for risk in risk_factors])
    else:
        risks_text = management_summary.get('risk_factors', 
                                          'Risk factors analysis is being processed. Key operational, financial, and market risks will be detailed based on industry analysis and company-specific factors.')
    
    content = f"""
## Management Discussion & Analysis

### Management Outlook
{management_outlook}

### Key Strategic Initiatives
{initiatives_text}

### Growth Opportunities
{opportunities_text}

### Risk Factors
{risks_text}
    """.strip()
    
    return ReportSection(
        title="Management Discussion & Analysis",
        content=content,
        level=1,
        order=5,
        metadata={"section_type": "management_discussion"}
    )
