"""
Helper functions for AI Analysis Agent.

Contains utility functions extracted from AIAnalysisAgent to satisfy
code quality criteria that agent files should be focused on the agent class only.
"""

import logging
from typing import Dict, Any
import json
import re

try:
    from ..config import Config
except ImportError:
    from config import Config

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


def extract_research_data_for_tools(research_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract research data and map to tool names for reuse.
    
    Maps research data to the tool names that would fetch the same data,
    so the agent can reuse it instead of calling tools.
    
    Handles both AIResearchAgent output (with gathered_data) and ResearchAgent output (structured).
    """
    extracted = {}
    
    # Check if this is from AIResearchAgent with gathered_data structure
    if "ai_iterations" in research_data or "gathered_data" in research_data:
        gathered_data = research_data.get("gathered_data", {})
        
        # Extract directly from gathered_data if available (tool names as keys)
        if isinstance(gathered_data, dict):
            if "get_stock_metrics" in gathered_data:
                extracted["get_stock_metrics"] = gathered_data["get_stock_metrics"]
            if "get_company_info" in gathered_data:
                extracted["get_company_info"] = gathered_data["get_company_info"]
            if "search_company_news" in gathered_data:
                extracted["search_company_news"] = gathered_data["search_company_news"]
            if "search_market_trends" in gathered_data:
                extracted["search_market_trends"] = gathered_data["search_market_trends"]
    
    # Handle structured format from both AIResearchAgent and ResearchAgent
    company_data = research_data.get("company_data", {})
    
    if isinstance(company_data, dict):
        # Check for nested structure (both stock_metrics and company_info)
        stock_metrics = company_data.get("stock_metrics")
        company_info = company_data.get("company_info")
        
        if stock_metrics or company_info:
            if stock_metrics and "get_stock_metrics" not in extracted:
                extracted["get_stock_metrics"] = stock_metrics
            if company_info and "get_company_info" not in extracted:
                extracted["get_company_info"] = company_info
        else:
            # Check if company_data itself is flat data
            if any(key in company_data for key in ["current_price", "market_cap", "pe_ratio", "pb_ratio"]):
                if "get_stock_metrics" not in extracted:
                    extracted["get_stock_metrics"] = company_data
            elif any(key in company_data for key in ["company_name", "name", "sector", "industry", "business"]):
                if "get_company_info" not in extracted:
                    extracted["get_company_info"] = company_data
    
    # Extract company news (search_company_news)
    news_data = research_data.get("news_data", {})
    if news_data and "search_company_news" not in extracted:
        extracted["search_company_news"] = news_data
    
    # Extract market trends (search_market_trends)
    market_trends = research_data.get("market_trends", {})
    if market_trends:
        extracted["search_market_trends"] = market_trends
    
    # Also check sector_data for trends (ResearchAgent structure)
    sector_data = research_data.get("sector_data", {})
    if sector_data:
        trends = sector_data.get("trends", {})
        if trends and "search_market_trends" not in extracted:
            extracted["search_market_trends"] = trends
    
    return extracted


def summarize_available_data(gathered_data: Dict[str, Any], tool_map: Dict[str, Any]) -> str:
    """
    Summarize available data for the LLM prompt.
    
    Args:
        gathered_data: Dictionary mapping tool names to available data
        tool_map: Dictionary mapping tool names to tool objects
        
    Returns:
        String summary of available data with details
    """
    if not gathered_data:
        return "No data available from research. You may need to call tools to gather data."
    
    summary_parts = []
    
    # Iterate through gathered_data dynamically
    for tool_name, tool_data in gathered_data.items():
        if not tool_data:
            continue
        
        # Get tool description from tool_map if available
        tool_obj = tool_map.get(tool_name)
        tool_description = ""
        if tool_obj:
            try:
                tool_description = getattr(tool_obj, 'description', '') or ""
            except:
                pass
        
        summary_parts.append(f"\n✓ {tool_name}:")
        
        # Extract details generically from tool_data structure
        details = []
        
        if isinstance(tool_data, dict):
            # Extract company info fields
            company_info_fields = {
                "company_name": "Company",
                "name": "Company",
                "sector": "Sector",
                "industry": "Industry",
                "business": "Business",
                "description": "Description"
            }
            
            # Extract stock metrics
            metrics_fields = {
                "current_price": ("Price", "₹{:.2f}"),
                "market_cap": ("Market Cap", "₹{:,.0f}"),
                "pe_ratio": ("P/E Ratio", "{:.2f}"),
                "pb_ratio": ("P/B Ratio", "{:.2f}"),
                "dividend_yield": ("Dividend Yield", "{:.2%}"),
                "eps": ("EPS", "₹{:.2f}"),
                "volume": ("Volume", "{:,}"),
                "52_week_high": ("52W High", "₹{:.2f}"),
                "52_week_low": ("52W Low", "₹{:.2f}")
            }
            
            # Check for company info fields
            for field, label in company_info_fields.items():
                if field in tool_data and tool_data[field]:
                    value = tool_data[field]
                    if field == "description" and isinstance(value, str) and len(value) > 50:
                        details.append(f"{label}: {value[:50]}...")
                    else:
                        details.append(f"{label}: {value}")
            
            # Check for metrics fields
            for field, (label, fmt) in metrics_fields.items():
                if field in tool_data and tool_data[field] is not None:
                    try:
                        value = float(tool_data[field])
                        details.append(f"{label}: {fmt.format(value)}")
                    except (ValueError, TypeError):
                        pass
            
            # Check for list/array fields
            if "articles" in tool_data:
                articles = tool_data["articles"]
                if isinstance(articles, list):
                    details.append(f"{len(articles)} news articles available")
            
            if "results" in tool_data:
                results = tool_data["results"]
                if isinstance(results, list):
                    details.append(f"{len(results)} results available")
            
            # If no specific details extracted, show data type and keys
            if not details:
                keys = list(tool_data.keys())[:5]
                details.append(f"Data available with fields: {', '.join(keys)}")
                if len(tool_data) > 5:
                    details[-1] += f" (+{len(tool_data) - 5} more)"
        
        elif isinstance(tool_data, (list, tuple)):
            details.append(f"{len(tool_data)} items available")
        
        # Add extracted details
        if details:
            summary_parts.append(f"  - {', '.join(details[:3])}")
            if len(details) > 3:
                summary_parts.append(f"  - ... and {len(details) - 3} more fields")
        
        # Add tool description if no details were extracted
        elif tool_description:
            summary_parts.append(f"  - {tool_description[:100]}")
    
    if len(summary_parts) == 1:  # Only header, no data
        return "No data available from research. You may need to call tools to gather data."
    
    return "\n".join(summary_parts)


def calculate_financial_ratios(stock_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate financial ratios."""
    ratios = {}
    ratios["pe_ratio"] = stock_metrics.get("pe_ratio")
    ratios["pb_ratio"] = stock_metrics.get("pb_ratio")
    ratios["dividend_yield"] = stock_metrics.get("dividend_yield")
    ratios["beta"] = stock_metrics.get("beta")
    
    current_price = stock_metrics.get("current_price", 0)
    market_cap = stock_metrics.get("market_cap", 0)
    
    if current_price and market_cap:
        ratios["market_cap_category"] = categorize_market_cap(market_cap)
    
    return ratios


async def analyze_financial_health(
    stock_metrics: Dict[str, Any], 
    ratios: Dict[str, Any], 
    openai_client: AsyncOpenAI
) -> Dict[str, Any]:
    """Analyze financial health using LLM."""
    try:
        key_metrics = {
            "current_price": stock_metrics.get("current_price"),
            "market_cap": stock_metrics.get("market_cap"),
            "volume": stock_metrics.get("volume"),
            "52_week_high": stock_metrics.get("52_week_high"),
            "52_week_low": stock_metrics.get("52_week_low"),
            "beta": stock_metrics.get("beta"),
            "dividend_yield": stock_metrics.get("dividend_yield"),
            "eps": stock_metrics.get("eps")
        }
        
        prompt = f"""Analyze financial health from stock metrics and ratios. Return JSON:

<stock_metrics>
{json.dumps(key_metrics, indent=2, default=str)}
</stock_metrics>

<ratios>
{json.dumps(ratios, indent=2, default=str)}
</ratios>

Output format:
{{
    "health_score": <0-100 integer>,
    "health_factors": ["factor1", "factor2"],
    "overall_assessment": "<brief assessment>"
}}"""
        
        response = await openai_client.chat.completions.create(
            model=Config.DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": "You are a financial analyst. Analyze financial health and return structured JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        response_text = response.choices[0].message.content
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        
        if json_match:
            analysis_data = json.loads(json_match.group(0))
            return {
                "health_score": analysis_data.get("health_score", 0),
                "health_factors": analysis_data.get("health_factors", []),
                "overall_assessment": analysis_data.get("overall_assessment", "Unable to assess")
            }
        else:
            return {
                "health_score": 50,
                "health_factors": ["Analysis completed"],
                "overall_assessment": "Financial health assessment completed"
            }
    except Exception as e:
        logger.error(f"Financial health analysis failed: {e}")
        return {"health_score": 0, "health_factors": [], "overall_assessment": "Unable to assess"}


def calculate_technical_indicators(stock_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate technical indicators."""
    indicators = {}
    
    current_price = stock_metrics.get("current_price", 0)
    high_52w = stock_metrics.get("52_week_high", 0)
    low_52w = stock_metrics.get("52_week_low", 0)
    
    if current_price and high_52w and low_52w:
        range_position = (current_price - low_52w) / (high_52w - low_52w) if high_52w != low_52w else 0.5
        indicators["range_position"] = range_position
        
        if range_position > 0.8:
            indicators["trend"] = "Near 52-week high"
        elif range_position < 0.2:
            indicators["trend"] = "Near 52-week low"
        else:
            indicators["trend"] = "Mid-range"
        
        upside_potential = ((high_52w - current_price) / current_price) * 100
        downside_risk = ((current_price - low_52w) / current_price) * 100
        indicators["upside_potential"] = upside_potential
        indicators["downside_risk"] = downside_risk
    
    volume = stock_metrics.get("volume", 0)
    avg_volume = stock_metrics.get("avg_volume", 0)
    
    if volume and avg_volume:
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        indicators["volume_ratio"] = volume_ratio
        
        if volume_ratio > 1.5:
            indicators["volume_trend"] = "High volume"
        elif volume_ratio < 0.5:
            indicators["volume_trend"] = "Low volume"
        else:
            indicators["volume_trend"] = "Normal volume"
    
    return indicators


def calculate_valuation_metrics(stock_metrics: Dict[str, Any], sector: str) -> Dict[str, Any]:
    """Calculate valuation metrics."""
    metrics = {}
    
    pe_ratio = stock_metrics.get("pe_ratio")
    pb_ratio = stock_metrics.get("pb_ratio")
    market_cap = stock_metrics.get("market_cap", 0)
    
    if pe_ratio:
        metrics["pe_ratio"] = pe_ratio
        if pe_ratio < 15:
            metrics["pe_assessment"] = "Undervalued"
        elif pe_ratio < 25:
            metrics["pe_assessment"] = "Fairly valued"
        else:
            metrics["pe_assessment"] = "Overvalued"
    
    if pb_ratio:
        metrics["pb_ratio"] = pb_ratio
        if pb_ratio < 1:
            metrics["pb_assessment"] = "Trading below book value"
        elif pb_ratio < 3:
            metrics["pb_assessment"] = "Reasonable price-to-book"
        else:
            metrics["pb_assessment"] = "High price-to-book"
    
    if market_cap:
        metrics["market_cap"] = market_cap
        metrics["market_cap_category"] = categorize_market_cap(market_cap)
    
    return metrics


async def calculate_target_price_llm(
    stock_metrics: Dict[str, Any], 
    valuation_metrics: Dict[str, Any],
    sector: str,
    market_trends: Dict[str, Any],
    openai_client: AsyncOpenAI
) -> Dict[str, Any]:
    """
    Calculate target price using LLM analysis.
    
    Args:
        stock_metrics: Current stock metrics (price, ratios, etc.)
        valuation_metrics: Calculated valuation metrics
        sector: Sector name
        market_trends: Market trends data
        openai_client: OpenAI client for API calls
        
    Returns:
        Dictionary with target_price, recommendation, reasoning, etc.
    """
    try:
        current_price = stock_metrics.get("current_price", 0)
        if not current_price:
            return {"target_price": 0, "method": "insufficient_data", "error": "No current price available"}
        
        # Prepare prompt with all relevant data
        market_cap = stock_metrics.get('market_cap')
        market_cap_str = f"₹{market_cap:,.0f}" if market_cap else 'N/A'
        
        dividend_yield = stock_metrics.get('dividend_yield')
        dividend_yield_str = f"{dividend_yield * 100:.2f}%" if dividend_yield else "0.00%"
        
        prompt = f"""You are a financial analyst calculating a target price for a stock.

STOCK METRICS:
- Current Price: ₹{current_price:.2f}
- P/E Ratio: {stock_metrics.get('pe_ratio', 'N/A')}
- P/B Ratio: {stock_metrics.get('pb_ratio', 'N/A')}
- Market Cap: {market_cap_str}
- EPS: ₹{stock_metrics.get('eps', 'N/A')}
- Dividend Yield: {dividend_yield_str}
- 52 Week High: ₹{stock_metrics.get('52_week_high', 'N/A')}
- 52 Week Low: ₹{stock_metrics.get('52_week_low', 'N/A')}

VALUATION METRICS:
{format_valuation_metrics_for_llm(valuation_metrics)}

SECTOR: {sector}

MARKET TRENDS:
{format_market_trends_for_llm(market_trends)}

Analyze this stock and provide:
1. Target Price: A realistic target price in ₹ (Indian Rupees) based on valuation analysis
2. Recommendation: One of BUY, HOLD, or SELL
3. Upside Potential: Percentage upside/downside from current price
4. Time Horizon: Suggested time horizon for the target (e.g., "12 months", "6 months")
5. Reasoning: Brief explanation (2-3 sentences) of your analysis and how you arrived at the target price

Consider:
- Valuation metrics (P/E, P/B ratios compared to sector/industry averages)
- Growth prospects and market trends
- Current price relative to 52-week range
- Risk factors
- Market sentiment

Respond in JSON format:
{{
    "target_price": <numeric_value>,
    "current_price": {current_price:.2f},
    "upside_potential": <percentage>,
    "recommendation": "<BUY|HOLD|SELL>",
    "time_horizon": "<time period>",
    "reasoning": "<brief explanation>"
}}"""

        # Call LLM
        response = await openai_client.chat.completions.create(
            model=Config.DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert financial analyst specializing in stock valuation. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        # Parse response
        response_content = response.choices[0].message.content
        result = json.loads(response_content)
        
        # Validate and format result
        target_price = float(result.get("target_price", current_price))
        recommendation = result.get("recommendation", "HOLD").upper()
        if recommendation not in ["BUY", "HOLD", "SELL"]:
            recommendation = "HOLD"
        
        upside_potential = result.get("upside_potential", 0)
        if not upside_potential:
            # Calculate if not provided
            upside_potential = ((target_price - current_price) / current_price) * 100
        
        return {
            "target_price": round(target_price, 2),
            "current_price": current_price,
            "upside_potential": round(float(upside_potential), 2),
            "recommendation": recommendation,
            "time_horizon": result.get("time_horizon", "12 months"),
            "reasoning": result.get("reasoning", ""),
            "method": "llm_analysis"
        }
        
    except Exception as e:
        logger.error(f"LLM target price calculation failed: {e}")
        # Fallback to simple calculation
        current_price = stock_metrics.get("current_price", 0)
        pe_ratio = stock_metrics.get("pe_ratio")
        
        if not current_price or not pe_ratio:
            return {"target_price": current_price, "method": "fallback_insufficient_data", "error": str(e)}
        
        # Simple fallback
        if pe_ratio < 15:
            target_price = current_price * 1.2
            recommendation = "BUY"
        elif pe_ratio < 25:
            target_price = current_price * 1.1
            recommendation = "HOLD"
        else:
            target_price = current_price * 0.9
            recommendation = "SELL"
        
        return {
            "target_price": round(target_price, 2),
            "current_price": current_price,
            "upside_potential": round(((target_price - current_price) / current_price) * 100, 2),
            "recommendation": recommendation,
            "method": "fallback_pe_ratio_based",
            "error": str(e)
        }


def format_valuation_metrics_for_llm(valuation_metrics: Dict[str, Any]) -> str:
    """Format valuation metrics for LLM prompt."""
    if not valuation_metrics:
        return "No valuation metrics available"
    
    lines = []
    for key, value in valuation_metrics.items():
        if value is not None:
            if isinstance(value, (int, float)):
                lines.append(f"- {key.replace('_', ' ').title()}: {value}")
            else:
                lines.append(f"- {key.replace('_', ' ').title()}: {value}")
    
    return "\n".join(lines) if lines else "No valuation metrics available"


def format_market_trends_for_llm(market_trends: Dict[str, Any]) -> str:
    """Format market trends for LLM prompt."""
    if not market_trends:
        return "No market trends data available"
    
    # Extract key information from market trends
    if isinstance(market_trends, dict):
        if "results" in market_trends and isinstance(market_trends["results"], list):
            # Take first few results
            results = market_trends["results"][:3]
            lines = ["Recent market trends:"]
            for i, result in enumerate(results, 1):
                if isinstance(result, dict):
                    title = result.get("title", "Trend")[:100]
                    snippet = result.get("snippet", "")[:200]
                    lines.append(f"{i}. {title}")
                    if snippet:
                        lines.append(f"   {snippet}")
            return "\n".join(lines)
        else:
            # Try to extract any text content
            return str(market_trends)[:500]
    
    return "No market trends data available"


def categorize_market_cap(market_cap: float) -> str:
    """Categorize market cap."""
    if market_cap >= 100000000000:  # >= 100k crores
        return "Large Cap"
    elif market_cap >= 20000000000:  # >= 20k crores
        return "Mid Cap"
    else:
        return "Small Cap"


def format_valuation_metrics_for_llm(valuation_metrics: Dict[str, Any]) -> str:
    """Format valuation metrics for LLM prompt."""
    if not valuation_metrics:
        return "No valuation metrics available"
    
    lines = []
    for key, value in valuation_metrics.items():
        if value is not None:
            if isinstance(value, (int, float)):
                lines.append(f"- {key.replace('_', ' ').title()}: {value}")
            else:
                lines.append(f"- {key.replace('_', ' ').title()}: {value}")
    
    return "\n".join(lines) if lines else "No valuation metrics available"


def format_market_trends_for_llm(market_trends: Dict[str, Any]) -> str:
    """Format market trends for LLM prompt."""
    if not market_trends:
        return "No market trends data available"
    
    # Extract key information from market trends
    if isinstance(market_trends, dict):
        if "results" in market_trends and isinstance(market_trends["results"], list):
            # Take first few results
            results = market_trends["results"][:3]
            lines = ["Recent market trends:"]
            for i, result in enumerate(results, 1):
                if isinstance(result, dict):
                    title = result.get("title", "Trend")[:100]
                    snippet = result.get("snippet", "")[:200]
                    lines.append(f"{i}. {title}")
                    if snippet:
                        lines.append(f"   {snippet}")
            return "\n".join(lines)
        else:
            # Try to extract any text content
            return str(market_trends)[:500]
    
    return "No market trends data available"



async def perform_comprehensive_analysis(
    stock_symbol: str,
    company_name: str,
    sector: str,
    gathered_data: Dict[str, Any],
    research_data: Dict[str, Any],
    openai_client: AsyncOpenAI
) -> Dict[str, Any]:
    """
    Perform comprehensive analysis on gathered data.
    
    Args:
        stock_symbol: Stock symbol
        company_name: Company name
        sector: Sector name
        gathered_data: Data gathered by agent loop
        research_data: Data from research agent
        openai_client: OpenAI client for API calls
        
    Returns:
        Dictionary with all analysis results
    """
    results = {
        "financial": {},
        "management": {},
        "technical": {},
        "valuation": {}
    }
    
    try:
        # Get stock metrics (from gathered data or research data)
        stock_metrics = gathered_data.get("get_stock_metrics")
        if not stock_metrics:
            company_data = research_data.get("company_data", {})
            stock_metrics = company_data.get("stock_metrics")
        
        # Get company info
        company_info = gathered_data.get("get_company_info")
        if not company_info:
            company_data = research_data.get("company_data", {})
            company_info = company_data.get("company_info")
        
        # Get news data
        news_data = gathered_data.get("search_company_news")
        if not news_data:
            news_data = research_data.get("news_data", {})
        
        # Get market trends
        market_trends = gathered_data.get("search_market_trends")
        if not market_trends:
            sector_data = research_data.get("sector_data", {})
            market_trends = sector_data.get("trends", {})
        
        # 1. Financial Analysis
        if stock_metrics:
            results["financial"] = await perform_financial_analysis(stock_metrics, openai_client)
        
        # 2. Management Analysis
        if company_info or news_data:
            results["management"] = await perform_management_analysis(company_info, news_data, openai_client)
        
        # 3. Technical Analysis
        if stock_metrics:
            results["technical"] = perform_technical_analysis(stock_metrics)
        
        # 4. Valuation Analysis
        if stock_metrics:
            results["valuation"] = await perform_valuation_analysis(stock_metrics, sector, market_trends, openai_client)
        
    except Exception as e:
        logger.error(f"Comprehensive analysis failed: {e}")
        results["error"] = str(e)
    
    return results


async def perform_financial_analysis(stock_metrics: Dict[str, Any], openai_client: AsyncOpenAI) -> Dict[str, Any]:
    """Perform financial analysis."""
    try:
        # Calculate financial ratios
        financial_ratios = calculate_financial_ratios(stock_metrics)
        
        # Analyze financial health using LLM
        financial_health = await analyze_financial_health(stock_metrics, financial_ratios, openai_client)
        
        return {
            "stock_metrics": stock_metrics,
            "financial_ratios": financial_ratios,
            "financial_health": financial_health,
            "analysis_type": "comprehensive_financial"
        }
    except Exception as e:
        logger.error(f"Financial analysis failed: {e}")
        return {"error": str(e)}


async def perform_management_analysis(
    company_info: Dict[str, Any], 
    news_data: Dict[str, Any],
    openai_client: AsyncOpenAI
) -> Dict[str, Any]:
    """Perform management analysis."""
    try:
        # Extract key information
        analysis_input = {
            "company_info": company_info or {},
            "recent_news": news_data.get("articles", [])[:5] if news_data else []
        }
        
        # Use LLM for management assessment
        prompt = f"""Analyze management effectiveness and governance for this company.

Company Info: {json.dumps(company_info, default=str)[:1000] if company_info else "Limited information available"}
Recent News: {json.dumps(analysis_input["recent_news"], default=str)[:1000]}

Provide assessment in JSON format:
{{
    "management_score": <0-100 integer>,
    "key_strengths": ["strength1", "strength2"],
    "key_concerns": ["concern1", "concern2"],
    "overall_assessment": "<brief assessment>"
}}"""
        
        response = await openai_client.chat.completions.create(
            model=Config.DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": "You are a management analyst. Analyze management effectiveness and return structured JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        response_text = response.choices[0].message.content
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        
        if json_match:
            analysis_data = json.loads(json_match.group(0))
            return {
                "management_score": analysis_data.get("management_score", 50),
                "key_strengths": analysis_data.get("key_strengths", []),
                "key_concerns": analysis_data.get("key_concerns", []),
                "overall_assessment": analysis_data.get("overall_assessment", "Limited assessment"),
                "analysis_type": "management_analysis"
            }
        else:
            return {
                "management_score": 50,
                "key_strengths": [],
                "key_concerns": [],
                "overall_assessment": "Assessment completed",
                "analysis_type": "management_analysis"
            }
    except Exception as e:
        logger.error(f"Management analysis failed: {e}")
        return {"error": str(e), "analysis_type": "management_analysis"}


def perform_technical_analysis(stock_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Perform technical analysis."""
    try:
        # Calculate technical indicators
        technical_indicators = calculate_technical_indicators(stock_metrics)
        
        # Import here to avoid circular imports
        try:
            try:
                from ..tools.technical_analysis_formatter import format_technical_analysis
            except ImportError:
                from tools.technical_analysis_formatter import format_technical_analysis
            technical_summary = format_technical_analysis.invoke({"technical_data": technical_indicators})
        except:
            technical_summary = "Technical analysis completed"
        
        return {
            "stock_metrics": stock_metrics,
            "technical_indicators": technical_indicators,
            "technical_summary": technical_summary,
            "analysis_type": "technical_analysis"
        }
    except Exception as e:
        logger.error(f"Technical analysis failed: {e}")
        return {"error": str(e), "analysis_type": "technical_analysis"}


async def perform_valuation_analysis(
    stock_metrics: Dict[str, Any], 
    sector: str, 
    market_trends: Dict[str, Any],
    openai_client: AsyncOpenAI
) -> Dict[str, Any]:
    """Perform valuation analysis."""
    try:
        # Calculate valuation metrics
        valuation_metrics = calculate_valuation_metrics(stock_metrics, sector)
        
        # Calculate target price using LLM
        target_price = await calculate_target_price_llm(stock_metrics, valuation_metrics, sector, market_trends, openai_client)
        
        return {
            "stock_metrics": stock_metrics,
            "market_trends": market_trends,
            "valuation_metrics": valuation_metrics,
            "target_price": target_price,
            "analysis_type": "valuation_analysis"
        }
    except Exception as e:
        logger.error(f"Valuation analysis failed: {e}")
        return {"error": str(e), "analysis_type": "valuation_analysis"}
