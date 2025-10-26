"""
LangChain Tools Registry for Stock Report Generator.
Exports all tools as LangChain-compatible tools that can be used by LLMs.
"""

from typing import List
from langchain_core.tools import Tool

# Import all the @tool decorated functions
from .stock_data_tool import get_stock_metrics, get_company_info, validate_symbol
from .web_search_tool import search_sector_news, search_company_news, search_market_trends
from .summarizer_tool import summarize_text, extract_insights, initialize_summarizer

# Initialize summarizer (this should be called with proper API key)
def initialize_tools(openai_api_key: str):
    """Initialize all tools with required API keys."""
    initialize_summarizer(openai_api_key)

# Export all tools as a list for easy use with LangChain agents
def get_all_tools() -> List[Tool]:
    """
    Get all available LangChain tools.
    
    Returns:
        List of LangChain Tool objects
    """
    return [
        # Stock Data Tools
        get_stock_metrics,
        get_company_info,
        validate_symbol,
        
        # Web Search Tools
        search_sector_news,
        search_company_news,
        search_market_trends,
        
        # Summarizer Tools
        summarize_text,
        extract_insights,
    ]

# Individual tool groups for specific use cases
def get_stock_data_tools() -> List[Tool]:
    """Get stock data related tools."""
    return [get_stock_metrics, get_company_info, validate_symbol]

def get_web_search_tools() -> List[Tool]:
    """Get web search related tools."""
    return [search_sector_news, search_company_news, search_market_trends]

def get_summarizer_tools() -> List[Tool]:
    """Get summarizer related tools."""
    return [summarize_text, extract_insights]

# Tool descriptions for agent configuration
TOOL_DESCRIPTIONS = {
    "get_stock_metrics": "Get comprehensive stock metrics including price, ratios, and financial data",
    "get_company_info": "Get detailed company information including sector, industry, and business description",
    "validate_symbol": "Validate if a stock symbol exists and is accessible",
    "search_sector_news": "Search for sector-specific news and market trends",
    "search_company_news": "Search for company-specific news and announcements",
    "search_market_trends": "Search for general market trends and analysis",
    "summarize_text": "Summarize text content with focus on specific areas",
    "extract_insights": "Extract insights and key information from text"
}
