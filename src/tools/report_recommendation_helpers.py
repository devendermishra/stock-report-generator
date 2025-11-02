"""
Recommendation and risk helpers for report formatter tool.
Contains functions to generate recommendation and risk analysis sections.
"""

from typing import Dict, Any
from .report_formatter_models import ReportSection


def create_investment_recommendation_section(
    stock_symbol: str,
    sector_summary: Dict[str, Any],
    stock_summary: Dict[str, Any],
    management_summary: Dict[str, Any],
    determine_recommendation
) -> ReportSection:
    """Create investment recommendation section."""
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
    recommendation = determine_recommendation(sector_summary, stock_summary, management_summary)
    
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


def create_risk_factors_section(
    sector_summary: Dict[str, Any],
    stock_summary: Dict[str, Any],
    management_summary: Dict[str, Any],
    format_risk_list
) -> ReportSection:
    """Create risk factors section."""
    content = f"""
## Risk Factors

### Sector Risks
{format_risk_list(sector_summary.get('risks', []), 'sector')}

### Company-Specific Risks
{format_risk_list(stock_summary.get('risks', []), 'company')}

### Market Risks
{format_risk_list(stock_summary.get('market_risks', []), 'market')}

### Regulatory Risks
{format_risk_list(sector_summary.get('regulatory_risks', []), 'regulatory')}
    """.strip()
    
    return ReportSection(
        title="Risk Factors",
        content=content,
        level=1,
        order=7,
        metadata={"section_type": "risk_factors"}
    )


def determine_recommendation(
    sector_summary: Dict[str, Any],
    stock_summary: Dict[str, Any],
    management_summary: Dict[str, Any],
    score_sector_outlook,
    score_financial_performance,
    score_management_quality
) -> Dict[str, str]:
    """Determine investment recommendation based on analysis."""
    # Score different factors
    sector_score = score_sector_outlook(sector_summary)
    financial_score = score_financial_performance(stock_summary)
    management_score = score_management_quality(management_summary)
    
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


def score_sector_outlook(sector_summary: Dict[str, Any]) -> float:
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


def score_financial_performance(stock_summary: Dict[str, Any]) -> float:
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


def score_management_quality(management_summary: Dict[str, Any]) -> float:
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
