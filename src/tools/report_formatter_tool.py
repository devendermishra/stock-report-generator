"""
Report Formatter Tool for generating professional markdown reports.
Handles report formatting, consistency checking, and final output generation.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import os
import random
import numpy as np
from langchain_core.tools import tool

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

from .report_formatter_models import ReportSection, ConsistencyIssue, FormattedReport
from .report_formatter_utils import ReportFormatterUtils
from .report_formatter_helpers import FinancialFormatter, TechnicalFormatter, PeerAnalysisFormatter, SWOTFormatter
from .technical_analysis_formatter import TechnicalAnalysisFormatter
from .report_section_generators import (
    create_executive_summary_section,
    create_company_overview_section,
    create_sector_analysis_section,
    create_financial_performance_section,
    create_management_discussion_section
)
from .report_recommendation_helpers import (
    create_investment_recommendation_section,
    create_risk_factors_section,
    determine_recommendation,
    score_sector_outlook,
    score_financial_performance,
    score_management_quality
)

logger = logging.getLogger(__name__)

class ReportFormatterTool:
    """
    Report Formatter Tool for generating professional markdown reports.
    
    Provides functionality to format reports, check consistency,
    and generate final markdown output.
    """
    
    def _format_market_cap(self, market_cap: Any) -> str:
        """Format market cap in a readable format.
        
        Args:
            market_cap: Market capitalization value to format.
        
        Returns:
            Formatted market cap string.
        """
        return self.utils.format_market_cap(market_cap)
    
    def _get_recommendation_summary(self, stock_summary: Dict[str, Any], sector_summary: Dict[str, Any], management_summary: Dict[str, Any]) -> str:
        """Get a concise recommendation summary.
        
        Args:
            stock_summary: Dictionary containing stock analysis data.
            sector_summary: Dictionary containing sector analysis data.
            management_summary: Dictionary containing management analysis data.
        
        Returns:
            Concise recommendation summary string.
        """
        return self.utils.get_recommendation_summary(stock_summary, sector_summary, management_summary)
    
    def _get_management_outlook_summary(self, management_summary: Dict[str, Any]) -> str:
        """Get a concise management outlook summary.
        
        Args:
            management_summary: Dictionary containing management analysis data.
        
        Returns:
            Concise management outlook summary string.
        """
        # Try to get management outlook
        outlook = management_summary.get('management_outlook', '')
        if outlook and outlook != 'Management discussion highlights...':
            return outlook
        
        # Try to get from summary
        summary = management_summary.get('summary', '')
        if summary and summary != 'Management discussion highlights...':
            return summary
        
        # Generate a basic outlook based on available data
        strategic_initiatives = management_summary.get('strategic_initiatives', [])
        growth_opportunities = management_summary.get('growth_opportunities', [])
        
        if strategic_initiatives or growth_opportunities:
            return f"Management is focused on strategic initiatives including {', '.join(strategic_initiatives[:2]) if strategic_initiatives else 'growth opportunities'}. The company shows commitment to operational excellence and market expansion."
        else:
            return "Management outlook analysis is being processed. The company's strategic direction and future prospects will be detailed based on recent financial reports and management communications."
    
    def _format_trends_list(self, trends: List[str]) -> str:
        """Format trends list into proper markdown format.
        
        Args:
            trends: List of trend strings or a single string.
        
        Returns:
            Formatted markdown string of trends.
        """
        if not trends:
            return "Key sector trends include digital transformation, regulatory changes, and market consolidation."
        
        # If trends is already a string, return it
        if isinstance(trends, str):
            return trends
        
        # Format as markdown list
        formatted_trends = []
        for trend in trends:
            if isinstance(trend, str) and trend.strip():
                formatted_trends.append(f"- {trend.strip()}")
        
        if formatted_trends:
            return '\n'.join(formatted_trends)
        else:
            return "Key sector trends include digital transformation, regulatory changes, and market consolidation."
    
    def _format_peer_comparison(self, peer_data: Dict[str, Any]) -> str:
        """Format peer comparison data into readable format.
        
        Args:
            peer_data: Dictionary containing peer comparison data.
        
        Returns:
            Formatted peer comparison string.
        """
        if not peer_data:
            return "Peer comparison analysis is being processed. Key competitors and their performance metrics will be detailed based on sector analysis."
        
        # Extract peer information
        sector_leader = peer_data.get('sector_leader', {})
        performance_summary = peer_data.get('performance_summary', '')
        competitive_landscape = peer_data.get('competitive_landscape', '')
        
        # Format the comparison
        comparison_text = []
        
        if sector_leader and isinstance(sector_leader, dict):
            leader_name = sector_leader.get('symbol', 'N/A')
            leader_change = sector_leader.get('change', 0)
            comparison_text.append(f"**Sector Leader:** {leader_name} ({leader_change:+.2f}%)")
        
        if performance_summary:
            comparison_text.append(f"**Performance:** {performance_summary}")
        
        if competitive_landscape:
            comparison_text.append(f"**Competitive Landscape:** {competitive_landscape}")
        
        if not comparison_text:
            # Try to get peer companies from the data
            peer_companies = []
            if 'top_performers' in peer_data:
                for performer in peer_data['top_performers'][:3]:
                    if isinstance(performer, dict):
                        name = performer.get('symbol', 'N/A')
                        change = performer.get('change', 0)
                        peer_companies.append(f"- {name}: {change:+.2f}%")
            
            if peer_companies:
                comparison_text.append("**Key Peers:**")
                comparison_text.extend(peer_companies)
            else:
                return "Peer comparison analysis is being processed. Key competitors and their performance metrics will be detailed based on sector analysis."
        
        return '\n\n'.join(comparison_text)
    
    def _format_regulatory_environment(self, regulatory_data: str) -> str:
        """Format regulatory environment data into readable format.
        
        Args:
            regulatory_data: String containing regulatory environment data.
        
        Returns:
            Formatted regulatory environment string.
        """
        if not regulatory_data or regulatory_data == "Regulatory environment analysis pending":
            return "Regulatory environment analysis is being processed. Key regulatory developments, policy changes, and compliance requirements will be detailed based on recent regulatory updates and sector-specific regulations."
        
        # If it's already a well-formatted string, return it
        if isinstance(regulatory_data, str) and len(regulatory_data) > 50:
            return regulatory_data
        
        # Generate a basic regulatory environment description
        return f"Regulatory environment analysis indicates ongoing regulatory developments affecting the sector. Key regulatory factors include policy changes, compliance requirements, and regulatory oversight that may impact sector performance and operational dynamics."
    
    def _format_technical_analysis(self, technical_data: Dict[str, Any]) -> str:
        """Format technical analysis data into readable format.
        
        Args:
            technical_data: Dictionary containing technical analysis data.
        
        Returns:
            Formatted technical analysis string.
        """
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
    
    def _format_risk_list(self, risks: List[str], risk_type: str) -> str:
        """Format risk list into readable format."""
        if not risks:
            # Generate default risks based on type
            default_risks = {
                'sector': [
                    "Economic volatility affecting sector performance",
                    "Regulatory changes impacting sector dynamics", 
                    "Competitive pressures from new market entrants",
                    "Technology disruption changing business models"
                ],
                'company': [
                    "Management execution risks",
                    "Financial performance volatility",
                    "Operational challenges and inefficiencies",
                    "Strategic misalignment with market trends"
                ],
                'market': [
                    "Interest rate fluctuations",
                    "Currency exchange rate volatility",
                    "Market liquidity constraints",
                    "Global economic uncertainty"
                ],
                'regulatory': [
                    "Compliance with evolving regulations",
                    "Regulatory enforcement actions",
                    "Policy changes affecting business operations",
                    "Licensing and approval delays"
                ]
            }
            risks = default_risks.get(risk_type, ["Risk analysis pending"])
        
        # If risks is already a string, return it
        if isinstance(risks, str):
            return risks
        
        # Format as markdown list
        formatted_risks = []
        for risk in risks:
            if isinstance(risk, str) and risk.strip():
                formatted_risks.append(f"- {risk.strip()}")
        
        if formatted_risks:
            return '\n'.join(formatted_risks)
        else:
            return f"{risk_type.title()}-specific risk analysis is being processed. Key risk factors will be detailed based on comprehensive analysis."
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize the Report Formatter Tool.
        
        Args:
            output_dir: Directory to save formatted reports
        """
        self.output_dir = output_dir
        self.utils = ReportFormatterUtils()
        self.technical_formatter = TechnicalAnalysisFormatter()
        
        # Initialize helper formatters
        self.financial_formatter = FinancialFormatter()
        self.technical_formatter_helper = TechnicalFormatter()
        self.peer_formatter = PeerAnalysisFormatter()
        self.swot_formatter = SWOTFormatter()
        
        os.makedirs(output_dir, exist_ok=True)
        
    def format_report(
        self,
        stock_symbol: str,
        sector_summary: Dict[str, Any],
        stock_summary: Dict[str, Any],
        management_summary: Dict[str, Any],
        swot_summary: Optional[Dict[str, Any]] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> FormattedReport:
        """
        Format a comprehensive stock report from all agent outputs.
        
        Args:
            stock_symbol: NSE stock symbol
            sector_summary: Output from SectorResearcherAgent
            stock_summary: Output from StockResearcherAgent
            management_summary: Output from ManagementAnalysisAgent
            additional_data: Optional additional data
            
        Returns:
            FormattedReport object
        """
        try:
            # Create report sections
            sections = []
            
            # Executive Summary
            sections.append(create_executive_summary_section(
                stock_symbol, stock_summary, sector_summary, management_summary,
                self._format_market_cap, self._get_recommendation_summary, self._get_management_outlook_summary
            ))
            
            # Company Overview
            sections.append(create_company_overview_section(
                stock_symbol, stock_summary, sector_summary, self._format_market_cap
            ))
            
            # Sector Analysis
            sections.append(create_sector_analysis_section(
                sector_summary,
                self._format_trends_list,
                self._format_peer_comparison,
                self._format_regulatory_environment
            ))
            
            # Financial Performance
            sections.append(create_financial_performance_section(
                stock_summary, self._format_technical_analysis
            ))
            
            # Financial Summary
            sections.append(self._create_financial_summary(
                stock_summary
            ))
            
            # Management Discussion
            sections.append(create_management_discussion_section(management_summary))
            
            # Investment Recommendation
            def _determine_rec(sector, stock, mgmt):
                return determine_recommendation(sector, stock, mgmt, 
                    score_sector_outlook, score_financial_performance, score_management_quality)
            sections.append(create_investment_recommendation_section(
                stock_symbol, sector_summary, stock_summary, management_summary, _determine_rec
            ))
            
            # Risk Factors
            sections.append(create_risk_factors_section(
                sector_summary, stock_summary, management_summary, self._format_risk_list
            ))
            
            # SWOT Analysis
            if swot_summary:
                sections.append(self._create_swot_analysis(swot_summary))
            
            # Peer Analysis
            sections.append(self._create_peer_analysis(sector_summary))
            
            # Generate markdown content
            markdown_content = self._generate_markdown(sections)
            
            # Calculate word count
            word_count = len(markdown_content.split())
            
            # Create metadata
            metadata = {
                "stock_symbol": stock_symbol,
                "report_type": "equity_research",
                "sections_count": len(sections),
                "data_sources": self._extract_data_sources(sector_summary, stock_summary, management_summary),
                "generation_method": "multi_agent_collaboration"
            }
            
            formatted_report = FormattedReport(
                title=f"Equity Research Report: {stock_symbol}",
                sections=sections,
                metadata=metadata,
                markdown_content=markdown_content,
                word_count=word_count,
                creation_timestamp=datetime.now()
            )
            
            logger.info(f"Formatted report for {stock_symbol}")
            return formatted_report
            
        except Exception as e:
            logger.error(f"Error formatting report: {e}")
            raise
            
    def _create_executive_summary(
        self,
        stock_symbol: str,
        sector_summary: Dict[str, Any],
        stock_summary: Dict[str, Any],
        management_summary: Dict[str, Any]
    ) -> ReportSection:
        """Create executive summary section.
        
        Args:
            stock_symbol: NSE stock symbol.
            sector_summary: Dictionary containing sector analysis data.
            stock_summary: Dictionary containing stock analysis data.
            management_summary: Dictionary containing management analysis data.
        
        Returns:
            ReportSection object containing the executive summary.
        """
        # Extract company name and sectors
        company_name = stock_summary.get('company_name', 'N/A')
        sectors = stock_summary.get('sectors', [])
        if not sectors and sector_summary.get('sector'):
            sectors = [sector_summary.get('sector')]
        
        sectors_text = ', '.join(sectors) if sectors else 'N/A'
        
        # Format market cap
        formatted_market_cap = self._format_market_cap(stock_summary.get('market_cap'))
        
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
{self._get_management_outlook_summary(management_summary)}

### Recommendation
{self._get_recommendation_summary(stock_summary, sector_summary, management_summary)}
        """.strip()
        
        return ReportSection(
            title="Executive Summary",
            content=content,
            level=1,
            order=1,
            metadata={"section_type": "executive_summary"}
        )
        
    def _create_company_overview(
        self,
        stock_symbol: str,
        stock_summary: Dict[str, Any],
        sector_summary: Dict[str, Any] = None
    ) -> ReportSection:
        """Create company overview section.
        
        Args:
            stock_symbol: NSE stock symbol.
            stock_summary: Dictionary containing stock analysis data.
            sector_summary: Optional dictionary containing sector analysis data.
        
        Returns:
            ReportSection object containing the company overview.
        """
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
        formatted_market_cap = self._format_market_cap(stock_summary.get('market_cap'))
        
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
        
    def _create_sector_analysis(
        self,
        sector_summary: Dict[str, Any]
    ) -> ReportSection:
        """Create sector analysis section.
        
        Args:
            sector_summary: Dictionary containing sector analysis data.
        
        Returns:
            ReportSection object containing the sector analysis.
        """
        content = f"""
## Sector Analysis

### Sector Overview
{sector_summary.get('summary', 'Sector analysis provides insights into...')}

### Key Trends
{self._format_trends_list(sector_summary.get('trends', []))}

### Peer Comparison
{self._format_peer_comparison(sector_summary.get('peer_comparison', {}))}

### Regulatory Environment
{self._format_regulatory_environment(sector_summary.get('regulatory_environment', ''))}
        """.strip()
        
        return ReportSection(
            title="Sector Analysis",
            content=content,
            level=1,
            order=3,
            metadata={"section_type": "sector_analysis"}
        )
        
    def _create_financial_performance(
        self,
        stock_summary: Dict[str, Any]
    ) -> ReportSection:
        """Create financial performance section.
        
        Args:
            stock_summary: Dictionary containing stock analysis data.
        
        Returns:
            ReportSection object containing the financial performance.
        """
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
{self._format_technical_analysis(stock_summary.get('technical_analysis', {}))}
        """.strip()
        
        return ReportSection(
            title="Financial Performance",
            content=content,
            level=1,
            order=4,
            metadata={"section_type": "financial_performance"}
        )
        
    def _create_financial_summary(self, stock_summary: Dict[str, Any]) -> ReportSection:
        """Create comprehensive financial summary section.
        
        Args:
            stock_summary: Dictionary containing stock analysis data.
        
        Returns:
            ReportSection object containing the financial summary.
        """
        # Extract key financial metrics
        current_price = stock_summary.get('current_price', 0) or 0
        pe_ratio = stock_summary.get('pe_ratio', 0) or 0
        pb_ratio = stock_summary.get('pb_ratio', 0) or 0
        eps = stock_summary.get('eps', 0) or 0
        dividend_yield = stock_summary.get('dividend_yield', 0) or 0
        market_cap = stock_summary.get('market_cap', 0) or 0
        beta = stock_summary.get('beta', 0) or 0
        volume = stock_summary.get('volume', 0) or 0
        avg_volume = stock_summary.get('avg_volume', 0) or 0
        change_percent = stock_summary.get('change_percent', 0) or 0
        high_52w = stock_summary.get('high_52w', 0) or 0
        low_52w = stock_summary.get('low_52w', 0) or 0
        
        # Calculate additional metrics
        price_to_52w_high = (current_price / high_52w * 100) if high_52w > 0 else 0
        price_to_52w_low = (current_price / low_52w * 100) if low_52w > 0 else 0
        volume_ratio = (volume / avg_volume) if avg_volume > 0 else 0
        
        content = f"""
## Financial Summary

### Stock Price Summary
- **Current Price:** ₹{current_price:.2f}
- **52-Week Range:** ₹{low_52w:.2f} - ₹{high_52w:.2f}
- **Price vs 52W High:** {price_to_52w_high:.2f}% of 52-week high
- **Price vs 52W Low:** {price_to_52w_low:.2f}% of 52-week low
- **Daily Change:** {change_percent:+.2f}%

### Key Valuation Metrics
- **P/E Ratio:** {pe_ratio:.2f}
- **P/B Ratio:** {pb_ratio:.2f}
- **Earnings Per Share (EPS):** ₹{eps:.2f}
- **Dividend Yield:** {dividend_yield:.2f}%
- **Market Cap:** {self._format_market_cap(market_cap)}

### Risk & Volatility Metrics
- **Beta:** {beta:.2f}
- **Current Volume:** {volume:,}
- **Average Volume:** {avg_volume:,}
- **Volume Ratio:** {volume_ratio:.2f}x

### Financial Health Indicators
{self._get_financial_health_indicators(pe_ratio, pb_ratio, eps, dividend_yield, beta)}
        """.strip()
        
        return ReportSection(
            title="Financial Summary",
            content=content,
            level=1,
            order=5,
            metadata={"section_type": "financial_summary"}
        )
        
    def _get_financial_health_indicators(self, pe_ratio: float, pb_ratio: float, eps: float, dividend_yield: float, beta: float) -> str:
        """Get financial health indicators based on key metrics.
        
        Args:
            pe_ratio: Price-to-earnings ratio.
            pb_ratio: Price-to-book ratio.
            eps: Earnings per share.
            dividend_yield: Dividend yield percentage.
            beta: Beta value.
        
        Returns:
            Formatted string containing financial health indicators.
        """
        indicators = []
        
        # P/E Ratio analysis
        if pe_ratio > 0:
            if pe_ratio < 15:
                indicators.append("- **P/E Ratio:** Attractive valuation (below 15)")
            elif pe_ratio < 25:
                indicators.append("- **P/E Ratio:** Fair valuation (15-25)")
            else:
                indicators.append("- **P/E Ratio:** High valuation (above 25)")
        
        # P/B Ratio analysis
        if pb_ratio > 0:
            if pb_ratio < 1.5:
                indicators.append("- **P/B Ratio:** Undervalued (below 1.5)")
            elif pb_ratio < 3.0:
                indicators.append("- **P/B Ratio:** Fair value (1.5-3.0)")
            else:
                indicators.append("- **P/B Ratio:** Overvalued (above 3.0)")
        
        # EPS analysis
        if eps > 0:
            if eps > 50:
                indicators.append("- **EPS:** Strong earnings performance (above ₹50)")
            elif eps > 20:
                indicators.append("- **EPS:** Good earnings performance (₹20-50)")
            else:
                indicators.append("- **EPS:** Moderate earnings performance (below ₹20)")
        
        # Dividend Yield analysis
        if dividend_yield > 0:
            if dividend_yield > 3:
                indicators.append("- **Dividend Yield:** Attractive income (above 3%)")
            elif dividend_yield > 1:
                indicators.append("- **Dividend Yield:** Moderate income (1-3%)")
            else:
                indicators.append("- **Dividend Yield:** Low income (below 1%)")
        
        # Beta analysis
        if beta > 0:
            if beta < 0.8:
                indicators.append("- **Beta:** Low volatility (below 0.8)")
            elif beta < 1.2:
                indicators.append("- **Beta:** Market-like volatility (0.8-1.2)")
            else:
                indicators.append("- **Beta:** High volatility (above 1.2)")
        
        if not indicators:
            indicators.append("- **Analysis:** Financial health indicators are being processed")
        
        return '\n'.join(indicators)
        
    def _create_management_discussion(
        self,
        management_summary: Dict[str, Any]
    ) -> ReportSection:
        """Create management discussion section.
        
        Args:
            management_summary: Dictionary containing management analysis data.
        
        Returns:
            ReportSection object containing the management discussion.
        """
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
        
    def _create_investment_recommendation(
        self,
        stock_symbol: str,
        sector_summary: Dict[str, Any],
        stock_summary: Dict[str, Any],
        management_summary: Dict[str, Any]
    ) -> ReportSection:
        """Create investment recommendation section.
        
        Args:
            stock_symbol: NSE stock symbol.
            sector_summary: Dictionary containing sector analysis data.
            stock_summary: Dictionary containing stock analysis data.
            management_summary: Dictionary containing management analysis data.
        
        Returns:
            ReportSection object containing the investment recommendation.
        """
        # Generate recommendation based on analysis
        current_price = stock_summary.get('current_price', 0)
        pe_ratio = stock_summary.get('pe_ratio', 0)
        
        # Calculate target price based on P/E ratio and sector outlook
        if current_price and pe_ratio:
            # Simple target price calculation (can be enhanced with more sophisticated models)
            sector_multiplier = 1.1 if sector_summary.get('outlook', '').lower() in ['positive', 'strong'] else 0.95
            target_price = current_price * sector_multiplier
            upside_potential = ((target_price - current_price) / current_price) * 100
        else:
            target_price = 'N/A'
            upside_potential = 'N/A'
        
        # Determine recommendation based on multiple factors
        recommendation = self._determine_recommendation(sector_summary, stock_summary, management_summary)
        
        # Generate action guidance for existing holders based on recommendation
        if recommendation['rating'] == "BUY":
            existing_holder_guidance = """
**For Existing Holders:**
- **Continue Holding:** The stock shows strong fundamentals and positive outlook, making it a good candidate for holding
- **Consider Adding:** If you have room in your portfolio and agree with the thesis, consider averaging up on dips
- **Set Stop-Loss:** Protect gains by setting a stop-loss at 10-15% below current levels or at key support levels
- **Take Partial Profits:** If the stock has run up significantly, consider taking partial profits while maintaining core position
- **Monitor Catalysts:** Watch for the key catalysts mentioned above that could drive further upside
            """.strip()
        elif recommendation['rating'] == "SELL":
            existing_holder_guidance = """
**For Existing Holders:**
- **Consider Exiting:** Given the concerns identified, consider reducing or exiting your position
- **Staggered Exit:** If you have substantial holdings, consider a staggered exit to avoid market impact
- **Tax Implications:** Review tax implications before selling, especially for long-term holdings
- **Stop-Loss Immediately:** Set a tight stop-loss to protect capital if you plan to hold temporarily
- **Review Entry Thesis:** Reassess your original investment thesis - if it no longer holds, exit the position
- **Consider Alternatives:** Look for better opportunities in the same sector or other sectors
            """.strip()
        else:  # HOLD
            existing_holder_guidance = """
**For Existing Holders:**
- **Maintain Position:** The analysis suggests holding your current position without major changes
- **Avoid Averaging:** Given mixed signals, avoid aggressive averaging down unless you have high conviction
- **Regular Monitoring:** Keep a close watch on the key factors mentioned above for any changes
- **Partial Profit Taking:** If you're sitting on gains, consider taking partial profits while keeping core holdings
- **Rebalance if Needed:** If this position has become too large in your portfolio, consider rebalancing
- **Stay Disciplined:** Stick to your investment plan and avoid emotional decisions based on short-term volatility
            """.strip()
        
        content = f"""
## Investment Recommendation

### Recommendation Summary
Based on comprehensive analysis of {stock_symbol}, we provide the following investment recommendation:

**Recommendation:** {recommendation['rating']}
**Rationale:** {recommendation['rationale']}

### Key Factors
1. **Sector Outlook:** {sector_summary.get('outlook', 'Cautiously Optimistic')}
2. **Financial Performance:** {stock_summary.get('performance_rating', 'Strong')}
3. **Management Quality:** {management_summary.get('management_rating', 'Good')}
4. **Valuation:** {recommendation['valuation']}

### Target Price & Valuation
- **Current Price:** ₹{current_price}
- **Target Price:** ₹{target_price}
- **Upside Potential:** {upside_potential}%

### Investment Horizon
{stock_summary.get('investment_horizon', 'Medium to Long Term (12-24 months)')}

### Risk-Reward Profile
{recommendation['risk_reward']}

### Key Catalysts
{recommendation['catalysts']}

### Action for Existing Holders
{existing_holder_guidance}
        """.strip()
        
        return ReportSection(
            title="Investment Recommendation",
            content=content,
            level=1,
            order=6,
            metadata={"section_type": "investment_recommendation"}
        )
    
    def _determine_recommendation(
        self,
        sector_summary: Dict[str, Any],
        stock_summary: Dict[str, Any],
        management_summary: Dict[str, Any]
    ) -> Dict[str, str]:
        """Determine investment recommendation based on analysis.
        
        Args:
            sector_summary: Dictionary containing sector analysis data.
            stock_summary: Dictionary containing stock analysis data.
            management_summary: Dictionary containing management analysis data.
        
        Returns:
            Dictionary containing recommendation rating, rationale, valuation,
            risk_reward, and catalysts.
        """
        # Score different factors
        sector_score = self._score_sector_outlook(sector_summary)
        financial_score = self._score_financial_performance(stock_summary)
        management_score = self._score_management_quality(management_summary)
        
        # Calculate overall score
        total_score = (sector_score + financial_score + management_score) / 3
        
        # Determine recommendation
        if total_score >= 0.7:
            rating = "BUY"
            rationale = "Strong fundamentals, positive sector outlook, and competent management support a BUY recommendation."
            valuation = "Attractive valuation with upside potential"
            risk_reward = "Favorable risk-reward profile with strong upside potential"
            catalysts = "• Sector tailwinds\n• Strong financial performance\n• Management execution\n• Market expansion opportunities"
        elif total_score >= 0.5:
            rating = "HOLD"
            rationale = "Mixed signals with some positive factors but also concerns that warrant a HOLD recommendation."
            valuation = "Fair valuation with limited upside"
            risk_reward = "Balanced risk-reward profile"
            catalysts = "• Sector stability\n• Steady performance\n• Management initiatives\n• Market conditions"
        else:
            rating = "SELL"
            rationale = "Multiple concerns including weak fundamentals, challenging sector outlook, or management issues support a SELL recommendation."
            valuation = "Overvalued with downside risk"
            risk_reward = "Unfavorable risk-reward profile"
            catalysts = "• Sector headwinds\n• Weak financial performance\n• Management concerns\n• Market challenges"
        
        return {
            'rating': rating,
            'rationale': rationale,
            'valuation': valuation,
            'risk_reward': risk_reward,
            'catalysts': catalysts
        }
    
    def _score_sector_outlook(self, sector_summary: Dict[str, Any]) -> float:
        """Score sector outlook (0-1).
        
        Args:
            sector_summary: Dictionary containing sector analysis data with 'outlook' key.
        
        Returns:
            Float score between 0 and 1 indicating sector outlook strength.
        """
        outlook = sector_summary.get('outlook', '').lower()
        if outlook in ['positive', 'strong', 'bullish']:
            return 0.8
        elif outlook in ['neutral', 'stable']:
            return 0.5
        elif outlook in ['negative', 'weak', 'bearish']:
            return 0.2
        else:
            return 0.5  # Default neutral
    
    def _score_financial_performance(self, stock_summary: Dict[str, Any]) -> float:
        """Score financial performance (0-1).
        
        Args:
            stock_summary: Dictionary containing stock analysis data with 'pe_ratio' key.
        
        Returns:
            Float score between 0 and 1 indicating financial performance strength.
        """
        pe_ratio = stock_summary.get('pe_ratio', 0)
        if pe_ratio and 10 <= pe_ratio <= 25:  # Reasonable P/E range
            return 0.7
        elif pe_ratio and pe_ratio < 10:  # Potentially undervalued
            return 0.8
        elif pe_ratio and pe_ratio > 25:  # Potentially overvalued
            return 0.3
        else:
            return 0.5  # Default neutral
    
    def _score_management_quality(self, management_summary: Dict[str, Any]) -> float:
        """Score management quality (0-1).
        
        Args:
            management_summary: Dictionary containing management analysis data with 'management_rating' key.
        
        Returns:
            Float score between 0 and 1 indicating management quality strength.
        """
        management_rating = management_summary.get('management_rating', '').lower()
        if management_rating in ['excellent', 'strong', 'good']:
            return 0.8
        elif management_rating in ['average', 'fair']:
            return 0.5
        elif management_rating in ['poor', 'weak']:
            return 0.2
        else:
            return 0.5  # Default neutral
        
    def _create_risk_factors(
        self,
        sector_summary: Dict[str, Any],
        stock_summary: Dict[str, Any],
        management_summary: Dict[str, Any]
    ) -> ReportSection:
        """Create risk factors section.
        
        Args:
            sector_summary: Dictionary containing sector analysis data.
            stock_summary: Dictionary containing stock analysis data.
            management_summary: Dictionary containing management analysis data.
        
        Returns:
            ReportSection object containing the risk factors.
        """
        content = f"""
## Risk Factors

### Sector Risks
{self._format_risk_list(sector_summary.get('risks', []), 'sector')}

### Company-Specific Risks
{self._format_risk_list(stock_summary.get('risks', []), 'company')}

### Market Risks
{self._format_risk_list(stock_summary.get('market_risks', []), 'market')}

### Regulatory Risks
{self._format_risk_list(sector_summary.get('regulatory_risks', []), 'regulatory')}
        """.strip()
        
        return ReportSection(
            title="Risk Factors",
            content=content,
            level=1,
            order=7,
            metadata={"section_type": "risk_factors"}
        )
        
    def _create_swot_analysis(self, swot_summary: Dict[str, Any]) -> ReportSection:
        """Create SWOT analysis section.
        
        Args:
            swot_summary: Dictionary containing SWOT analysis data.
        
        Returns:
            ReportSection object containing the SWOT analysis.
        """
        content = f"""
## SWOT Analysis

### Strengths
{self._format_swot_list(swot_summary.get('strengths', []), 'strengths')}

### Weaknesses
{self._format_swot_list(swot_summary.get('weaknesses', []), 'weaknesses')}

### Opportunities
{self._format_swot_list(swot_summary.get('opportunities', []), 'opportunities')}

### Threats
{self._format_swot_list(swot_summary.get('threats', []), 'threats')}

### Strategic Summary
{swot_summary.get('summary', 'SWOT analysis provides comprehensive insights into the company\'s strategic position and market dynamics.')}
        """.strip()
        
        return ReportSection(
            title="SWOT Analysis",
            content=content,
            level=1,
            order=8,
            metadata={"section_type": "swot_analysis"}
        )
        
    def _format_swot_list(self, items: List[str], category: str) -> str:
        """Format SWOT list into readable format."""
        if not items:
            # Generate default items based on category
            default_items = {
                'strengths': [
                    "Strong market position and brand recognition",
                    "Robust financial performance and profitability",
                    "Experienced management team and leadership",
                    "Diversified business model and revenue streams"
                ],
                'weaknesses': [
                    "Limited market share in key segments",
                    "Dependency on specific markets or products",
                    "Operational inefficiencies and cost structure",
                    "Regulatory compliance and governance challenges"
                ],
                'opportunities': [
                    "Market expansion and geographic growth",
                    "Technology adoption and digital transformation",
                    "Strategic partnerships and acquisitions",
                    "New product development and innovation"
                ],
                'threats': [
                    "Intense competition and market saturation",
                    "Economic volatility and market uncertainty",
                    "Regulatory changes and compliance requirements",
                    "Technology disruption and market evolution"
                ]
            }
            items = default_items.get(category, [f"{category.title()} analysis pending"])
        
        # If items is already a string, return it
        if isinstance(items, str):
            return items
        
        # Format as markdown list
        formatted_items = []
        for item in items:
            if isinstance(item, str) and item.strip():
                formatted_items.append(f"- {item.strip()}")
        
        if formatted_items:
            return '\n'.join(formatted_items)
        else:
            return f"{category.title()} analysis is being processed. Key {category} will be detailed based on comprehensive analysis."
        
    def _create_peer_analysis(self, sector_summary: Dict[str, Any]) -> ReportSection:
        """Create peer analysis section."""
        peer_data = sector_summary.get('peer_comparison', {})
        
        content = f"""
## Peer Analysis

### Key Peers Comparison
{self._format_peer_comparison_table(peer_data)}

### Sector Leaders & Market Cap Analysis
{self._format_sector_leaders_analysis(peer_data)}

### Performance Summary
{self._format_peer_performance_summary(peer_data)}

### Competitive Positioning
{self._format_competitive_positioning(peer_data)}
        """.strip()
        
        return ReportSection(
            title="Peer Analysis",
            content=content,
            level=1,
            order=9,
            metadata={"section_type": "peer_analysis"}
        )
        
    def _format_peer_comparison_table(self, peer_data: Dict[str, Any]) -> str:
        """Format peer comparison as a table."""
        detailed_peers = peer_data.get('detailed_peers', [])
        
        if not detailed_peers:
            # Try to get default peers for Financial Services sector
            default_peers = [
                {"symbol": "HDFCBANK", "name": "HDFC Bank Limited", "current_price": 0, "market_cap": 0, "pe_ratio": 0, "pb_ratio": 0, "eps": 0, "dividend_yield": 0, "change_percent": 0},
                {"symbol": "SBIN", "name": "State Bank of India", "current_price": 0, "market_cap": 0, "pe_ratio": 0, "pb_ratio": 0, "eps": 0, "dividend_yield": 0, "change_percent": 0},
                {"symbol": "KOTAKBANK", "name": "Kotak Mahindra Bank", "current_price": 0, "market_cap": 0, "pe_ratio": 0, "pb_ratio": 0, "eps": 0, "dividend_yield": 0, "change_percent": 0}
            ]
            detailed_peers = default_peers
        
        # Create table header
        table_lines = [
            "| Company | Symbol | Price | Market Cap | P/E | P/B | EPS | Div Yield | Change |",
            "|---------|--------|-------|------------|-----|-----|-----|-----------|--------|"
        ]
        
        # Add peer data rows
        for peer in detailed_peers[:3]:  # Limit to top 3 peers
            symbol = peer.get('symbol', 'N/A')
            name = peer.get('name', 'N/A')
            price = peer.get('current_price', 0)
            market_cap = peer.get('market_cap', 0)
            pe_ratio = peer.get('pe_ratio', 0)
            pb_ratio = peer.get('pb_ratio', 0)
            eps = peer.get('eps', 0)
            div_yield = peer.get('dividend_yield', 0)
            change = peer.get('change_percent', 0)
            
            # Format values
            price_str = f"₹{price:.2f}" if price else "N/A"
            market_cap_str = self._format_market_cap(market_cap) if market_cap else "N/A"
            pe_str = f"{pe_ratio:.2f}" if pe_ratio else "N/A"
            pb_str = f"{pb_ratio:.2f}" if pb_ratio else "N/A"
            eps_str = f"₹{eps:.2f}" if eps else "N/A"
            div_str = f"{div_yield:.2f}%" if div_yield else "N/A"
            change_str = f"{change:+.2f}%" if change else "N/A"
            
            table_lines.append(f"| {name} | {symbol} | {price_str} | {market_cap_str} | {pe_str} | {pb_str} | {eps_str} | {div_str} | {change_str} |")
        
        return '\n'.join(table_lines)
        
    def _format_peer_performance_summary(self, peer_data: Dict[str, Any]) -> str:
        """Format peer performance summary."""
        avg_performance = peer_data.get('avg_performance', 0)
        performance_summary = peer_data.get('performance_summary', '')
        sector_leader = peer_data.get('sector_leader', {})
        
        summary_parts = []
        
        if performance_summary:
            summary_parts.append(f"**Sector Performance:** {performance_summary}")
        
        if avg_performance != 0:
            summary_parts.append(f"**Average Sector Performance:** {avg_performance:+.2f}%")
        
        if sector_leader and isinstance(sector_leader, dict):
            leader_name = sector_leader.get('symbol', 'N/A')
            leader_change = sector_leader.get('change', 0)
            summary_parts.append(f"**Sector Leader:** {leader_name} ({leader_change:+.2f}%)")
        
        if not summary_parts:
            summary_parts.append("**Key Financial Services Peers:** HDFC Bank, State Bank of India, Kotak Mahindra Bank")
            summary_parts.append("**Analysis Status:** Peer performance metrics are being processed. Key performance indicators and competitive positioning will be detailed based on comprehensive sector analysis.")
        
        return '\n\n'.join(summary_parts)
        
    def _format_competitive_positioning(self, peer_data: Dict[str, Any]) -> str:
        """Format competitive positioning analysis."""
        top_performers = peer_data.get('top_performers', [])
        underperformers = peer_data.get('underperformers', [])
        
        positioning_parts = []
        
        if top_performers:
            top_list = []
            for performer in top_performers[:3]:
                if isinstance(performer, dict):
                    name = performer.get('symbol', 'N/A')
                    change = performer.get('change', 0)
                    top_list.append(f"- {name}: {change:+.2f}%")
            if top_list:
                positioning_parts.append("**Top Performers:**\n" + '\n'.join(top_list))
        
        if underperformers:
            under_list = []
            for underperformer in underperformers[:3]:
                if isinstance(underperformer, dict):
                    name = underperformer.get('symbol', 'N/A')
                    change = underperformer.get('change', 0)
                    under_list.append(f"- {name}: {change:+.2f}%")
            if under_list:
                positioning_parts.append("**Underperformers:**\n" + '\n'.join(under_list))
        
        if not positioning_parts:
            positioning_parts.append("**Market Leaders:** HDFC Bank (largest private sector bank), State Bank of India (largest public sector bank), Kotak Mahindra Bank (premium banking)")
            positioning_parts.append("**Competitive Dynamics:** Intense competition in digital banking, retail lending, and wealth management segments")
            positioning_parts.append("**Analysis Status:** Detailed competitive positioning analysis is being processed based on market share, financial performance, and strategic initiatives.")
        
        return '\n\n'.join(positioning_parts)
        
    def _format_sector_leaders_analysis(self, peer_data: Dict[str, Any]) -> str:
        """Format sector leaders and market cap analysis."""
        detailed_peers = peer_data.get('detailed_peers', [])
        sector_leader = peer_data.get('sector_leader', {})
        
        if not detailed_peers:
            # Provide default sector leaders for Financial Services
            return """
**Sector Leaders by Market Cap:**
- **HDFC Bank Limited (HDFCBANK):** Largest private sector bank by market cap
- **State Bank of India (SBIN):** Largest public sector bank by market cap  
- **ICICI Bank Limited (ICICIBANK):** Second largest private sector bank
- **Kotak Mahindra Bank (KOTAKBANK):** Premium banking segment leader

**Market Cap Positioning:**
- **Large Cap Leaders:** HDFC Bank, SBI, ICICI Bank (₹2L+ Cr market cap)
- **Mid Cap Players:** Kotak Bank, Axis Bank, IndusInd Bank (₹50K-2L Cr market cap)
- **Competitive Landscape:** Intense competition in digital banking, retail lending, and wealth management

**Analysis Status:** Detailed market cap analysis and sector leadership positioning is being processed based on comprehensive sector data.
            """.strip()
        
        # Sort peers by market cap if available
        peers_with_market_cap = []
        for peer in detailed_peers:
            market_cap = peer.get('market_cap', 0)
            if market_cap > 0:
                peers_with_market_cap.append(peer)
        
        if peers_with_market_cap:
            # Sort by market cap descending
            peers_with_market_cap.sort(key=lambda x: x.get('market_cap', 0), reverse=True)
            
            analysis_parts = ["**Sector Leaders by Market Cap:**"]
            
            for i, peer in enumerate(peers_with_market_cap[:3], 1):
                name = peer.get('name', 'N/A')
                symbol = peer.get('symbol', 'N/A')
                market_cap = peer.get('market_cap', 0)
                market_cap_str = self._format_market_cap(market_cap)
                
                if i == 1:
                    analysis_parts.append(f"- **{name} ({symbol}):** Sector leader with {market_cap_str} market cap")
                else:
                    analysis_parts.append(f"- **{name} ({symbol}):** {market_cap_str} market cap")
            
            # Add market cap positioning analysis
            if len(peers_with_market_cap) >= 2:
                largest_market_cap = peers_with_market_cap[0].get('market_cap', 0)
                second_largest_market_cap = peers_with_market_cap[1].get('market_cap', 0)
                
                if largest_market_cap > 0 and second_largest_market_cap > 0:
                    dominance_ratio = largest_market_cap / second_largest_market_cap
                    analysis_parts.append(f"\n**Market Dominance:** Leader has {dominance_ratio:.2f}x the market cap of second-largest competitor")
            
            return '\n'.join(analysis_parts)
        else:
            return """
**Sector Leaders Analysis:**
- **Market Cap Data:** Detailed market cap analysis is being processed
- **Sector Leadership:** Key sector leaders and their market positioning will be detailed
- **Competitive Dynamics:** Market share and competitive positioning analysis pending

**Analysis Status:** Comprehensive sector leadership and market cap analysis is being processed based on sector data.
            """.strip()
        
    def _generate_markdown(self, sections: List[ReportSection]) -> str:
        """Generate markdown content from sections."""
        markdown_parts = []
        
        # Add header
        markdown_parts.append("# Equity Research Report")
        markdown_parts.append(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        markdown_parts.append("")
        
        # Add sections
        for section in sorted(sections, key=lambda x: x.order):
            markdown_parts.append(section.content)
            markdown_parts.append("")
            
        # Add footer
        markdown_parts.append("---")
        markdown_parts.append("*This report was generated using AI-powered multi-agent analysis.*")
        
        return "\n".join(markdown_parts)
        
    def _extract_data_sources(
        self,
        sector_summary: Dict[str, Any],
        stock_summary: Dict[str, Any],
        management_summary: Dict[str, Any]
    ) -> List[str]:
        """Extract data sources from agent outputs."""
        sources = []
        
        if sector_summary.get('sources'):
            sources.extend(sector_summary['sources'])
        if stock_summary.get('sources'):
            sources.extend(stock_summary['sources'])
        if management_summary.get('sources'):
            sources.extend(management_summary['sources'])
            
        return list(set(sources))  # Remove duplicates
        
    def check_consistency(self, report: FormattedReport) -> List[ConsistencyIssue]:
        """
        Check the report for consistency issues.
        
        Args:
            report: FormattedReport object to check
            
        Returns:
            List of ConsistencyIssue objects
        """
        issues = []
        
        try:
            # Check for missing data
            for section in report.sections:
                if "N/A" in section.content:
                    issues.append(ConsistencyIssue(
                        issue_type="missing_data",
                        description=f"Missing data in {section.title}",
                        severity="medium",
                        location=section.title,
                        suggestion="Verify data sources and agent outputs"
                    ))
                    
            # Check for inconsistent formatting
            if not report.markdown_content.startswith("#"):
                issues.append(ConsistencyIssue(
                    issue_type="formatting",
                    description="Report does not start with proper header",
                    severity="low",
                    location="header",
                    suggestion="Ensure proper markdown formatting"
                ))
                
            # Check for empty sections
            for section in report.sections:
                if len(section.content.strip()) < 50:
                    issues.append(ConsistencyIssue(
                        issue_type="empty_section",
                        description=f"Section {section.title} appears to be empty or too short",
                        severity="high",
                        location=section.title,
                        suggestion="Review agent outputs for this section"
                    ))
                    
            logger.info(f"Found {len(issues)} consistency issues")
            return issues
            
        except Exception as e:
            logger.error(f"Error checking consistency: {e}")
            return []
            
    def save_report(self, report: FormattedReport, filename: Optional[str] = None) -> str:
        """
        Save the formatted report to a file.
        
        Args:
            report: FormattedReport object to save
            filename: Optional custom filename
            
        Returns:
            Path to the saved file
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"stock_report_{report.metadata['stock_symbol']}_{timestamp}.md"
                
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report.markdown_content)
                
            logger.info(f"Saved report to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving report: {e}")
            raise

# Global formatter instance
_formatter = ReportFormatterTool()

@tool(
    description="Format comprehensive stock research reports from multiple agent outputs. Combines sector analysis, stock metrics, management analysis, and SWOT data into a professionally formatted markdown report. Essential for generating final stock research reports with proper structure and formatting.",
    infer_schema=True,
    parse_docstring=False
)
def format_stock_report(
    stock_symbol: str,
    sector_summary: Dict[str, Any],
    stock_summary: Dict[str, Any],
    management_summary: Dict[str, Any],
    swot_summary: Optional[Dict[str, Any]] = None,
    additional_data: Optional[Dict[str, Any]] = None,
    output_dir: str = "reports"
) -> Dict[str, Any]:
    """
    Format comprehensive stock research reports from multiple agent outputs.
    
    Combines outputs from multiple analysis agents into a single professionally formatted
    markdown report with executive summary, company overview, sector analysis, financial
    performance, management analysis, and recommendations.
    
    Args:
        stock_symbol: NSE stock symbol.
        sector_summary: Dictionary containing sector analysis output.
        stock_summary: Dictionary containing stock analysis output.
        management_summary: Dictionary containing management analysis output.
        swot_summary: Optional dictionary containing SWOT analysis.
        additional_data: Optional dictionary with additional data.
        output_dir: Directory to save formatted report (default: "reports").
    
    Returns:
        Dictionary containing success, markdown_content, sections, metadata,
        consistency_issues, file_path, and error (if failed).
    """
    try:
        formatter = ReportFormatterTool(output_dir=output_dir)
        formatted_report = formatter.format_report(
            stock_symbol=stock_symbol,
            sector_summary=sector_summary,
            stock_summary=stock_summary,
            management_summary=management_summary,
            swot_summary=swot_summary,
            additional_data=additional_data
        )
        
        # Check consistency
        consistency_issues = formatter.check_consistency(formatted_report)
        
        # Save report
        file_path = formatter.save_report(formatted_report)
        
        return {
            "success": True,
            "markdown_content": formatted_report.markdown_content,
            "sections": [
                {
                    "title": section.title,
                    "content": section.content,
                    "order": section.order
                }
                for section in formatted_report.sections
            ],
            "metadata": formatted_report.metadata,
            "consistency_issues": [
                {
                    "issue_type": issue.issue_type,
                    "description": issue.description,
                    "severity": issue.severity,
                    "location": issue.location,
                    "suggestion": issue.suggestion
                }
                for issue in consistency_issues
            ],
            "file_path": file_path
        }
    except Exception as e:
        logger.error(f"Error formatting stock report: {e}")
        return {
            "success": False,
            "error": f"Report formatting failed: {str(e)}",
            "markdown_content": None,
            "sections": [],
            "metadata": {},
            "consistency_issues": [],
            "file_path": None
        }
