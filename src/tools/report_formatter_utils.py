"""
Utility functions for report formatting.
Contains helper methods for formatting various report components.
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class ReportFormatterUtils:
    """Utility class for report formatting operations."""
    
    def format_market_cap(self, market_cap: Any) -> str:
        """Format market cap in a readable format."""
        if not market_cap or market_cap == 'N/A':
            return 'N/A'
        
        try:
            # Convert to float if it's a string
            if isinstance(market_cap, str):
                market_cap = float(market_cap)
            
            # Convert to crores (divide by 1e7)
            market_cap_cr = market_cap / 1e7
            
            if market_cap_cr >= 100000:  # 1 lakh crores
                return f"{market_cap_cr/100000:.2f}L Cr"
            elif market_cap_cr >= 1000:  # 1 thousand crores
                return f"{market_cap_cr/1000:.2f}K Cr"
            else:
                return f"{market_cap_cr:.2f} Cr"
        except (ValueError, TypeError):
            return 'N/A'
    
    def get_recommendation_summary(self, stock_summary: Dict[str, Any], sector_summary: Dict[str, Any], management_summary: Dict[str, Any]) -> str:
        """Get a concise recommendation summary."""
        # Determine recommendation from analysis
        recommendation_data = self._determine_recommendation(sector_summary, stock_summary, management_summary)
        return f"**{recommendation_data['rating']}** - {recommendation_data['rationale']}"
    
    def get_management_outlook_summary(self, management_summary: Dict[str, Any]) -> str:
        """Get a concise management outlook summary."""
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
    
    def format_trends_list(self, trends: List[str]) -> str:
        """Format trends list into proper markdown format."""
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
    
    def format_peer_comparison(self, peer_data: Dict[str, Any]) -> str:
        """Format peer comparison data into readable format."""
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
    
    def format_regulatory_environment(self, regulatory_data: str) -> str:
        """Format regulatory environment data into readable format."""
        if not regulatory_data or regulatory_data == "Regulatory environment analysis pending":
            return "Regulatory environment analysis is being processed. Key regulatory developments, policy changes, and compliance requirements will be detailed based on recent regulatory updates and sector-specific regulations."
        
        # If it's already a well-formatted string, return it
        if isinstance(regulatory_data, str) and len(regulatory_data) > 50:
            return regulatory_data
        
        # Generate a basic regulatory environment description
        return f"Regulatory environment analysis indicates ongoing regulatory developments affecting the sector. Key regulatory factors include policy changes, compliance requirements, and regulatory oversight that may impact sector performance and operational dynamics."
    
    def format_risk_list(self, risks: List[str], risk_type: str) -> str:
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
    
    def _determine_recommendation(
        self,
        sector_summary: Dict[str, Any],
        stock_summary: Dict[str, Any],
        management_summary: Dict[str, Any]
    ) -> Dict[str, str]:
        """Determine investment recommendation based on analysis."""
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
        """Score sector outlook (0-1)."""
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
        """Score financial performance (0-1)."""
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
        """Score management quality (0-1)."""
        management_rating = management_summary.get('management_rating', '').lower()
        if management_rating in ['excellent', 'strong', 'good']:
            return 0.8
        elif management_rating in ['average', 'fair']:
            return 0.5
        elif management_rating in ['poor', 'weak']:
            return 0.2
        else:
            return 0.5  # Default neutral
