"""
Example demonstrating enhanced LangChain tools with detailed annotations.
Shows how the improved tool descriptions help LLMs understand and use tools better.
"""

from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from ..tools.langchain_tools import get_all_tools, initialize_tools

def create_enhanced_stock_agent(openai_api_key: str):
    """
    Create a LangChain agent with enhanced tool descriptions.
    
    Args:
        openai_api_key: OpenAI API key
        
    Returns:
        Configured LangChain agent with enhanced tools
    """
    # Initialize tools
    initialize_tools(openai_api_key)
    
    # Get all available tools (now with enhanced descriptions)
    tools = get_all_tools()
    
    # Create the LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        api_key=openai_api_key
    )
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert financial analyst with access to comprehensive tools for:

STOCK DATA TOOLS:
- get_stock_metrics: Get comprehensive stock metrics including current price, market cap, P/E ratio, volume, and other financial indicators for NSE stocks
- get_company_info: Get detailed company information including business description, sector, industry, website, employee count, and location for NSE stocks
- validate_symbol: Validate if a stock symbol exists and is accessible on NSE

WEB SEARCH TOOLS:
- search_sector_news: Search for sector-specific news and market trends from Indian financial markets
- search_company_news: Search for company-specific news, announcements, earnings reports, and corporate updates
- search_market_trends: Search for general market trends, economic analysis, and financial market insights

SUMMARIZER TOOLS:
- summarize_text: Summarize financial documents, reports, and text content with AI-powered analysis
- extract_insights: Extract key insights, metrics, and structured information from financial documents

Use these tools strategically to provide comprehensive analysis. Always explain your reasoning and cite sources."""),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create the agent
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10
    )
    
    return agent_executor

def demonstrate_tool_descriptions():
    """Demonstrate how enhanced tool descriptions help LLMs."""
    
    print("Enhanced LangChain Tool Descriptions")
    print("=" * 50)
    
    # Example of how the enhanced descriptions help
    enhanced_descriptions = {
        "get_stock_metrics": {
            "name": "get_stock_metrics",
            "description": "Get comprehensive stock metrics including current price, market cap, P/E ratio, volume, and other financial indicators for NSE stocks. Automatically adds .NS suffix if not provided.",
            "parameters": {
                "symbol": "NSE stock symbol (e.g., 'RELIANCE' or 'RELIANCE.NS')"
            }
        },
        "search_sector_news": {
            "name": "search_sector_news", 
            "description": "Search for sector-specific news and market trends from Indian financial markets. Returns recent news articles, analysis, and market insights for a given sector. Useful for understanding sector performance and trends.",
            "parameters": {
                "sector": "Sector name (e.g., 'Banking', 'IT', 'Pharmaceuticals')",
                "days_back": "Number of days to look back for news (default: 7)",
                "include_analysis": "Whether to include analysis and opinion pieces (default: True)"
            }
        },
        "summarize_text": {
            "name": "summarize_text",
            "description": "Summarize financial documents, reports, and text content with AI-powered analysis. Extracts key points, sentiment, and insights. Perfect for processing earnings reports, analyst notes, and financial documents.",
            "parameters": {
                "text": "Text to summarize",
                "max_length": "Maximum length of summary (default: 500)",
                "focus_areas": "Optional list of focus areas (e.g., ['financial', 'strategic'])"
            }
        }
    }
    
    for tool_name, tool_info in enhanced_descriptions.items():
        print(f"\n🔧 {tool_name.upper()}")
        print(f"Description: {tool_info['description']}")
        print("Parameters:")
        for param, desc in tool_info['parameters'].items():
            print(f"  - {param}: {desc}")
    
    print("\n" + "=" * 50)
    print("Benefits of Enhanced Tool Descriptions:")
    print("✅ Better LLM Understanding - Clear descriptions help LLMs choose the right tool")
    print("✅ Improved Parameter Usage - Detailed parameter descriptions reduce errors")
    print("✅ Context-Aware Selection - LLMs can better match tools to user needs")
    print("✅ Error Reduction - Clear expectations reduce tool misuse")
    print("✅ Better User Experience - More accurate and relevant responses")

def example_queries():
    """Show example queries that benefit from enhanced tool descriptions."""
    
    example_queries = [
        {
            "query": "Get stock metrics for Reliance Industries and analyze its financial health",
            "expected_tools": ["validate_symbol", "get_stock_metrics", "get_company_info"],
            "reasoning": "LLM will first validate the symbol, then get comprehensive metrics and company info"
        },
        {
            "query": "Find recent news about the banking sector in India",
            "expected_tools": ["search_sector_news"],
            "reasoning": "LLM understands this is sector-specific news search"
        },
        {
            "query": "Summarize the latest quarterly results for TCS and extract key insights",
            "expected_tools": ["search_company_news", "summarize_text", "extract_insights"],
            "reasoning": "LLM will search for TCS news, then summarize and extract insights"
        },
        {
            "query": "Compare the performance of HDFC Bank and ICICI Bank",
            "expected_tools": ["get_stock_metrics", "search_company_news"],
            "reasoning": "LLM will get metrics for both banks and search for comparative news"
        }
    ]
    
    print("\nExample Queries and Expected Tool Usage:")
    print("=" * 50)
    
    for i, example in enumerate(example_queries, 1):
        print(f"\n{i}. Query: {example['query']}")
        print(f"   Expected Tools: {', '.join(example['expected_tools'])}")
        print(f"   Reasoning: {example['reasoning']}")

if __name__ == "__main__":
    demonstrate_tool_descriptions()
    example_queries()
