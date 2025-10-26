"""
Report Formatter Helper Classes.
Contains specialized helper classes for different aspects of report formatting.
"""

import logging
from typing import Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ReportSection:
    """Represents a section of a formatted report."""
    title: str
    content: str
    level: int
    order: int
    metadata: Dict[str, Any]

class FinancialFormatter:
    """Handles financial data formatting."""
    
    def format_market_cap(self, market_cap: Any) -> str:
        """Format market cap in a readable format."""
        if not market_cap or market_cap == 'N/A':
            return 'N/A'
        
        try:
            if isinstance(market_cap, str):
                market_cap = float(market_cap)
            
            market_cap_cr = market_cap / 1e7
            
            if market_cap_cr >= 100000:
                return f"{market_cap_cr/100000:.2f}L Cr"
            elif market_cap_cr >= 1000:
                return f"{market_cap_cr/1000:.2f}K Cr"
            else:
                return f"{market_cap_cr:.2f} Cr"
        except (ValueError, TypeError):
            return 'N/A'
    
    def get_financial_health_indicators(self, pe_ratio: float, pb_ratio: float, eps: float, dividend_yield: float, beta: float) -> str:
        """Get financial health indicators."""
        indicators = []
        
        # P/E Ratio analysis
        if pe_ratio:
            if pe_ratio < 15:
                indicators.append("✅ **Undervalued** (P/E < 15)")
            elif pe_ratio > 25:
                indicators.append("⚠️ **Overvalued** (P/E > 25)")
            else:
                indicators.append("✅ **Fairly Valued** (P/E 15-25)")
        
        # P/B Ratio analysis
        if pb_ratio:
            if pb_ratio < 1:
                indicators.append("✅ **Trading Below Book** (P/B < 1)")
            elif pb_ratio > 3:
                indicators.append("⚠️ **High Book Value** (P/B > 3)")
        
        # EPS analysis
        if eps:
            if eps > 0:
                indicators.append("✅ **Profitable** (Positive EPS)")
            else:
                indicators.append("❌ **Loss Making** (Negative EPS)")
        
        # Dividend analysis
        if dividend_yield:
            if dividend_yield > 3:
                indicators.append("✅ **High Dividend Yield** (>3%)")
            elif dividend_yield > 1:
                indicators.append("✅ **Moderate Dividend** (1-3%)")
        
        # Beta analysis
        if beta:
            if beta < 0.8:
                indicators.append("✅ **Low Volatility** (Beta < 0.8)")
            elif beta > 1.2:
                indicators.append("⚠️ **High Volatility** (Beta > 1.2)")
            else:
                indicators.append("✅ **Market Volatility** (Beta 0.8-1.2)")
        
        return "\n".join(indicators) if indicators else "Financial health indicators not available"

class TechnicalFormatter:
    """Handles technical analysis formatting."""
    
    def get_rsi_signal(self, rsi: float) -> str:
        """Get RSI signal description."""
        if rsi > 70:
            return "Overbought"
        elif rsi < 30:
            return "Oversold"
        else:
            return "Neutral"
    
    def get_momentum_signal(self, momentum: float) -> str:
        """Get momentum signal description."""
        if momentum > 0.1:
            return "Strong Bullish"
        elif momentum > 0:
            return "Bullish"
        elif momentum > -0.1:
            return "Bearish"
        else:
            return "Strong Bearish"
    
    def format_technical_analysis(self, technical_data: Dict[str, Any]) -> str:
        """Format technical analysis data."""
        if not technical_data:
            return "Technical analysis data not available"
        
        analysis_parts = []
        
        # Moving Averages
        sma_20 = technical_data.get('sma_20')
        sma_50 = technical_data.get('sma_50')
        sma_200 = technical_data.get('sma_200')
        
        if all(x is not None for x in [sma_20, sma_50, sma_200]):
            analysis_parts.append(f"- **Moving Averages:** SMA 20: ₹{sma_20:.2f}, SMA 50: ₹{sma_50:.2f}, SMA 200: ₹{sma_200:.2f}")
        
        # RSI
        rsi = technical_data.get('rsi')
        if rsi is not None:
            rsi_signal = self.get_rsi_signal(rsi)
            analysis_parts.append(f"- **RSI:** {rsi:.2f} ({rsi_signal})")
        
        # Bollinger Bands
        bb_upper = technical_data.get('bb_upper')
        bb_lower = technical_data.get('bb_lower')
        bb_middle = technical_data.get('bb_middle')
        
        if all(x is not None for x in [bb_upper, bb_lower, bb_middle]):
            current = technical_data.get('current_price', 0)
            if current > bb_upper:
                bb_signal = "Above Upper Band"
            elif current < bb_lower:
                bb_signal = "Below Lower Band"
            else:
                bb_signal = "Within Bands"
            analysis_parts.append(f"- **Bollinger Bands:** Upper: ₹{bb_upper:.2f}, Lower: ₹{bb_lower:.2f} ({bb_signal})")
        
        # Support and Resistance
        support = technical_data.get('support')
        resistance = technical_data.get('resistance')
        current = technical_data.get('current_price')
        
        if all(x is not None for x in [support, resistance, current]):
            analysis_parts.append(f"**Support & Resistance:** Support: ₹{support:.2f}, Resistance: ₹{resistance:.2f}")
            
            # Calculate potential upside/downside
            upside_potential = ((resistance - current) / current) * 100
            downside_risk = ((current - support) / current) * 100
            analysis_parts.append(f"- **Upside Potential:** {upside_potential:.2f}% to resistance")
            analysis_parts.append(f"- **Downside Risk:** {downside_risk:.2f}% to support")
        
        return "\n".join(analysis_parts) if analysis_parts else "Technical indicators not available"

class PeerAnalysisFormatter:
    """Handles peer analysis formatting."""
    
    def format_peer_comparison_table(self, peer_data: Dict[str, Any]) -> str:
        """Format peer comparison table."""
        if not peer_data or 'peers' not in peer_data:
            return "Peer comparison data not available"
        
        peers = peer_data['peers']
        if not peers:
            return "No peer data available"
        
        # Create table header
        table_lines = [
            "| Company | Price | P/E | P/B | Market Cap | Change |",
            "|---------|-------|-----|-----|------------|--------|"
        ]
        
        for peer in peers[:5]:  # Top 5 peers
            name = peer.get('name', 'N/A')[:20]  # Truncate long names
            price = f"₹{peer.get('price', 0):.2f}" if peer.get('price') else "N/A"
            pe = f"{peer.get('pe_ratio', 0):.2f}" if peer.get('pe_ratio') else "N/A"
            pb = f"{peer.get('pb_ratio', 0):.2f}" if peer.get('pb_ratio') else "N/A"
            market_cap = self._format_market_cap(peer.get('market_cap'))
            change = f"{peer.get('change_percent', 0):+.2f}%" if peer.get('change_percent') else "N/A"
            
            table_lines.append(f"| {name} | {price} | {pe} | {pb} | {market_cap} | {change} |")
        
        return "\n".join(table_lines)
    
    def _format_market_cap(self, market_cap: Any) -> str:
        """Format market cap for display."""
        if not market_cap or market_cap == 'N/A':
            return 'N/A'
        
        try:
            if isinstance(market_cap, str):
                market_cap = float(market_cap)
            
            market_cap_cr = market_cap / 1e7
            
            if market_cap_cr >= 100000:
                return f"{market_cap_cr/100000:.2f}L Cr"
            elif market_cap_cr >= 1000:
                return f"{market_cap_cr/1000:.2f}K Cr"
            else:
                return f"{market_cap_cr:.2f} Cr"
        except (ValueError, TypeError):
            return 'N/A'
    
    def format_peer_performance_summary(self, peer_data: Dict[str, Any]) -> str:
        """Format peer performance summary."""
        if not peer_data or 'peers' not in peer_data:
            return "Peer performance data not available"
        
        peers = peer_data['peers']
        if not peers:
            return "No peer performance data available"
        
        # Calculate performance metrics
        total_peers = len(peers)
        positive_performers = sum(1 for peer in peers if peer.get('change_percent', 0) > 0)
        avg_change = sum(peer.get('change_percent', 0) for peer in peers) / total_peers if total_peers > 0 else 0
        
        summary = f"""
**Performance Summary:**
- **Total Peers Analyzed:** {total_peers}
- **Positive Performers:** {positive_performers} ({positive_performers/total_peers*100:.1f}%)
- **Average Performance:** {avg_change:+.2f}%
- **Best Performer:** {max(peers, key=lambda x: x.get('change_percent', 0)).get('name', 'N/A')} ({max(peers, key=lambda x: x.get('change_percent', 0)).get('change_percent', 0):+.2f}%)
- **Worst Performer:** {min(peers, key=lambda x: x.get('change_percent', 0)).get('name', 'N/A')} ({min(peers, key=lambda x: x.get('change_percent', 0)).get('change_percent', 0):+.2f}%)
        """.strip()
        
        return summary

class SWOTFormatter:
    """Handles SWOT analysis formatting."""
    
    def format_swot_list(self, items: List[str], category: str) -> str:
        """Format SWOT items as a list."""
        if not items:
            return f"No {category.lower()} identified"
        
        formatted_items = []
        for i, item in enumerate(items[:5], 1):  # Limit to top 5 items
            formatted_items.append(f"{i}. {item}")
        
        if len(items) > 5:
            formatted_items.append(f"... and {len(items) - 5} more {category.lower()}")
        
        return "\n".join(formatted_items)
    
    def format_swot_analysis(self, swot_data: Dict[str, Any]) -> str:
        """Format complete SWOT analysis."""
        if not swot_data:
            return "SWOT analysis not available"
        
        sections = []
        
        # Strengths
        strengths = swot_data.get('strengths', [])
        if strengths:
            sections.append(f"**Strengths:**\n{self.format_swot_list(strengths, 'Strengths')}")
        
        # Weaknesses
        weaknesses = swot_data.get('weaknesses', [])
        if weaknesses:
            sections.append(f"**Weaknesses:**\n{self.format_swot_list(weaknesses, 'Weaknesses')}")
        
        # Opportunities
        opportunities = swot_data.get('opportunities', [])
        if opportunities:
            sections.append(f"**Opportunities:**\n{self.format_swot_list(opportunities, 'Opportunities')}")
        
        # Threats
        threats = swot_data.get('threats', [])
        if threats:
            sections.append(f"**Threats:**\n{self.format_swot_list(threats, 'Threats')}")
        
        # Summary
        summary = swot_data.get('summary', '')
        if summary:
            sections.append(f"**Summary:**\n{summary}")
        
        return "\n\n".join(sections) if sections else "SWOT analysis data not available"
