"""
Simple LangGraph-based Dynamic Agent for LLM-driven tool execution.
Uses the new LangChain 1.0+ API with create_agent.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio

# Import LangChain components
try:
    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.tools import tool
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"LangChain not available: {e}")
    LANGCHAIN_AVAILABLE = False

try:
    from ..graph.context_manager_mcp import MCPContextManager, ContextType
    from ..tools.stock_data_tool import StockDataTool
    from ..tools.web_search_tool import WebSearchTool
    from ..tools.summarizer_tool import SummarizerTool
except ImportError:
    # Fall back to absolute imports (when run as script)
    from graph.context_manager_mcp import MCPContextManager, ContextType
    from tools.stock_data_tool import StockDataTool
    from tools.web_search_tool import WebSearchTool
    from tools.summarizer_tool import SummarizerTool

logger = logging.getLogger(__name__)

class LangGraphDynamicAgentSimple:
    """
    Simple LangGraph-based Dynamic Agent using the new LangChain 1.0+ API.
    """
    
    def __init__(
        self,
        agent_id: str,
        mcp_context: MCPContextManager,
        stock_data_tool: StockDataTool,
        web_search_tool: WebSearchTool,
        summarizer_tool: SummarizerTool,
        openai_api_key: str,
        model: str = "gpt-4o-mini"
    ):
        """
        Initialize the LangGraph Dynamic Agent.
        
        Args:
            agent_id: Unique identifier for this agent
            mcp_context: MCP context manager for shared memory
            stock_data_tool: Stock data tool instance
            web_search_tool: Web search tool instance
            summarizer_tool: Summarizer tool instance
            openai_api_key: OpenAI API key
            model: OpenAI model to use
        """
        self.agent_id = agent_id
        self.mcp_context = mcp_context
        self.stock_data_tool = stock_data_tool
        self.web_search_tool = web_search_tool
        self.summarizer_tool = summarizer_tool
        self.openai_api_key = openai_api_key
        self.model = model

        # Initialize the LLM
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model=model,
            temperature=0.1
        )
        
        # Create LangChain tools
        self.langchain_tools = self._create_langchain_tools()
        
        # Create the agent
        self.agent = self._create_agent()
        
        logger.info(f"LangGraph Dynamic Agent {agent_id} initialized successfully")
    
    def _create_langchain_tools(self) -> List:
        """Create LangChain-compatible tools."""
        tools = []
        
        # Stock data tool
        @tool
        def get_stock_data(symbol: str) -> str:
            """Get comprehensive stock data for a given symbol.
            
            Returns detailed financial metrics including:
            - Current price, market cap, P/E ratio, P/B ratio
            - EPS, dividend yield, beta, volume data
            - 52-week high/low, price change percentage
            - Revenue growth, profit growth, debt-to-equity ratio
            - EV/EBITDA and other key financial ratios
            
            Args:
                symbol: Stock symbol (e.g., 'CIPLA', 'TCS', 'RELIANCE')
            """
            try:
                result = self.stock_data_tool.get_stock_metrics(symbol)
                return f"Stock data for {symbol}: {result}"
            except Exception as e:
                return f"Error getting stock data for {symbol}: {str(e)}"
        
        # Company info tool
        @tool
        def get_company_info(symbol: str) -> str:
            """Get comprehensive company information for a given symbol.
            
            Returns detailed company profile including:
            - Company name, sector, industry classification
            - Business description and operations overview
            - Website, employee count, headquarters location
            - Market cap, enterprise value
            - Key business segments and product portfolio
            
            Args:
                symbol: Stock symbol (e.g., 'CIPLA', 'TCS', 'RELIANCE')
            """
            try:
                result = self.stock_data_tool.get_company_info(symbol)
                return f"Company info for {symbol}: {result}"
            except Exception as e:
                return f"Error getting company info for {symbol}: {str(e)}"
        
        # Web search tool
        @tool
        def search_web(query: str) -> str:
            """Search the web for recent market information and news.
            
            Searches trusted financial sources for:
            - Recent news and developments about companies
            - Market trends and analysis
            - Industry reports and insights
            - Regulatory updates and announcements
            - Analyst opinions and recommendations
            
            Sources include NSE, LiveMint, Financial Express, and other reputable financial websites.
            
            Args:
                query: Search query (e.g., 'CIPLA recent news', 'pharmaceutical sector trends')
            """
            try:
                # Use the market trends search method
                result = self.web_search_tool.search_market_trends(query)
                return f"Web search results for '{query}': {result}"
            except Exception as e:
                return f"Error searching web for '{query}': {str(e)}"
        
        # Summarizer tool
        @tool
        def summarize_text(text: str) -> str:
            """Summarize and condense large amounts of text into key insights.
            
            Uses AI-powered summarization to extract:
            - Key points and main arguments
            - Important financial metrics and trends
            - Critical insights and conclusions
            - Actionable recommendations
            
            Useful for processing long reports, news articles, or analysis documents.
            
            Args:
                text: The text content to summarize (can be long articles, reports, etc.)
            """
            try:
                result = self.summarizer_tool.summarize(text)
                return f"Summary: {result}"
            except Exception as e:
                return f"Error summarizing text: {str(e)}"
        
        # Peer comparison tool
        @tool
        def get_peer_comparison(symbol: str) -> str:
            """Get comprehensive peer comparison data for a stock symbol.
            
            Automatically identifies the company's sector and compares with industry peers:
            - Identifies key competitors in the same sector
            - Compares financial metrics (P/E, P/B, growth rates)
            - Analyzes market positioning and competitive advantages
            - Provides relative valuation insights
            - Highlights industry trends and benchmarks
            
            Args:
                symbol: Stock symbol to analyze (e.g., 'CIPLA', 'TCS', 'RELIANCE')
            """
            try:
                # Get company info to find sector
                company_info = self.stock_data_tool.get_company_info(symbol)
                sector = getattr(company_info, 'sector', 'Unknown')
                
                # Search for peer companies in the same sector
                peer_query = f"{sector} sector companies stock comparison {symbol} peers"
                peer_results = self.web_search_tool.search_market_trends(peer_query)
                
                return f"Peer comparison for {symbol} in {sector} sector: {peer_results}"
            except Exception as e:
                return f"Error getting peer comparison for {symbol}: {str(e)}"
        
        tools.extend([get_stock_data, get_company_info, search_web, summarize_text, get_peer_comparison])
        return tools
    
    def _create_agent(self):
        """Create the LangChain agent."""
        system_prompt = """You are an expert financial analyst. Your job is to analyze stocks comprehensively and provide structured investment analysis.

Available tools with detailed capabilities:

1. **get_stock_data(symbol)**: Get comprehensive financial metrics
   - Current price, market cap, P/E, P/B ratios, EPS, dividend yield
   - Volume data, 52-week high/low, price changes
   - Growth metrics (revenue, profit), debt ratios, EV/EBITDA

2. **get_company_info(symbol)**: Get detailed company profile
   - Company name, sector, industry, business description
   - Operations overview, employee count, headquarters
   - Market cap, enterprise value, business segments

3. **search_web(query)**: Search trusted financial sources
   - Recent news, market trends, industry reports
   - Regulatory updates, analyst opinions
   - Sources: NSE, LiveMint, Financial Express, etc.

4. **summarize_text(text)**: AI-powered text summarization
   - Extract key points from long documents
   - Identify critical insights and recommendations
   - Process reports, articles, analysis documents

5. **get_peer_comparison(symbol)**: Industry peer analysis
   - Automatic sector identification and competitor analysis
   - Financial metrics comparison (P/E, P/B, growth rates)
   - Market positioning and competitive advantages

When analyzing a stock, follow this structured approach:

1. **Data Collection**: Get stock data, company info, and recent news
2. **Financial Analysis**: Analyze P/E, P/B, ROE, debt ratios, growth metrics
3. **Technical Analysis**: Review price trends, support/resistance levels
4. **Market Sentiment**: Assess recent news, analyst opinions, market conditions
5. **Peer Comparison**: Compare with industry peers on key metrics
6. **Investment Rating**: Provide BUY/HOLD/SELL recommendation with confidence score

**IMPORTANT**: Structure your final analysis with these exact sections:
- Investment Rating: [BUY/HOLD/SELL]
- Confidence Score: [1-10 scale]
- Market Sentiment: [BULLISH/NEUTRAL/BEARISH]
- Target Price: [specific price or range]
- Key Strengths: [bullet points]
- Key Risks: [bullet points]
- Peer Comparison: [comparison with 2-3 industry peers]

Always be thorough and provide detailed, structured analysis."""
        
        try:
            agent = create_agent(
                model=self.llm,
                tools=self.langchain_tools,
                system_prompt=system_prompt
            )
            return agent
        except Exception as e:
            logger.error(f"Error creating agent: {e}")
            raise
    
    async def analyze_stock(self, stock_symbol: str) -> Dict[str, Any]:
        """
        Analyze a stock using the LangGraph agent.
        
        Args:
            stock_symbol: Stock symbol to analyze
            
        Returns:
            Analysis results
        """
        try:
            logger.info(f"Starting LangGraph analysis for {stock_symbol}")
            
            # Prepare the input
            inputs = {
                "messages": [
                    {
                        "role": "user", 
                        "content": f"Analyze the stock {stock_symbol} comprehensively. Get stock data, company info, search for recent news, and provide investment recommendations."
                    }
                ]
            }
            
            # Run the agent
            result = self.agent.invoke(inputs)
            
            # Extract the final message
            final_message = result.get("messages", [])[-1] if result.get("messages") else None
            analysis_text = final_message.content if final_message else "No analysis generated"
            
            # Parse structured data from analysis
            structured_data = self._parse_structured_analysis(analysis_text)
            logger.info(f"Structured data extracted: {structured_data}")

            # Store results in MCP context
            self._store_analysis_results(stock_symbol, analysis_text, result)
            
            logger.info(f"Completed LangGraph analysis for {stock_symbol}")
            return {
                "success": True,
                "stock_symbol": stock_symbol,
                "analysis": analysis_text,
                "structured_data": structured_data,
                "full_result": result,
                "agent_id": self.agent_id,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in LangGraph analysis for {stock_symbol}: {e}")
            return {
                "success": False,
                "stock_symbol": stock_symbol,
                "error": str(e),
                "agent_id": self.agent_id,
                "timestamp": datetime.now().isoformat()
            }
    
    def _parse_structured_analysis(self, analysis_text: str) -> Dict[str, Any]:
        """Parse structured data from analysis text using simple text search."""
        
        structured_data = {
            "investment_rating": "N/A",
            "confidence_score": "N/A", 
            "market_sentiment": "N/A",
            "target_price": "N/A",
            "key_strengths": [],
            "key_risks": [],
            "peer_comparison": "N/A"
        }
        
        try:
            # Convert to lowercase for easier searching
            text_lower = analysis_text.lower()
            logger.debug(f"Parsing analysis text, length: {len(analysis_text)}")
            
            # Extract Investment Rating
            if "investment rating:" in text_lower:
                # Find the line with investment rating
                lines = analysis_text.split('\n')
                self._populate_field(lines, "investment rating:", ["buy", "hold", "sell"],
                                     "investment_rating", structured_data)

            # Extract Confidence Score
            if "confidence score:" in text_lower:
                lines = analysis_text.split('\n')
                for line in lines:
                    if "confidence score:" in line.lower():
                        # Extract number from the line
                        import re
                        numbers = re.findall(r'\d+', line)
                        if numbers:
                            structured_data["confidence_score"] = numbers[0]
                            logger.debug(f"Set confidence_score to: {structured_data['confidence_score']}")
                        break
            
            # Extract Market Sentiment
            if "market sentiment:" in text_lower:
                lines = analysis_text.split('\n')
                self._populate_field(lines, "market sentiment:", ["bullish", "bearish", "neutral"],
                                     "market_sentiment", structured_data)
            
            # Extract Target Price
            if "target price:" in text_lower:
                lines = analysis_text.split('\n')
                for line in lines:
                    if "target price:" in line.lower():
                        # Extract price range
                        import re
                        price_match = re.search(r'₹?\d+(?:\.\d+)?(?:\s*-\s*₹?\d+(?:\.\d+)?)?', line)
                        if price_match:
                            structured_data["target_price"] = price_match.group(0)
                        break
            
            # Extract Key Strengths
            if "key strengths:" in text_lower:
                # Find the section and extract bullet points
                start_idx = text_lower.find("key strengths:")
                if start_idx != -1:
                    # Look for the next section
                    end_markers = ["key risks:", "peer comparison:", "conclusion:"]
                    end_idx = len(analysis_text)
                    for marker in end_markers:
                        marker_idx = text_lower.find(marker, start_idx)
                        if marker_idx != -1 and marker_idx < end_idx:
                            end_idx = marker_idx
                    
                    strengths_section = analysis_text[start_idx:end_idx]
                    # Extract bullet points
                    import re
                    bullets = re.findall(r'[-•]\s*(.+?)(?=\n|$)', strengths_section)
                    structured_data["key_strengths"] = [b.strip() for b in bullets if b.strip()]
            
            # Extract Key Risks
            if "key risks:" in text_lower:
                start_idx = text_lower.find("key risks:")
                if start_idx != -1:
                    end_markers = ["peer comparison:", "conclusion:", "investment rating:"]
                    end_idx = len(analysis_text)
                    for marker in end_markers:
                        marker_idx = text_lower.find(marker, start_idx)
                        if marker_idx != -1 and marker_idx < end_idx:
                            end_idx = marker_idx
                    
                    risks_section = analysis_text[start_idx:end_idx]
                    import re
                    bullets = re.findall(r'[-•]\s*(.+?)(?=\n|$)', risks_section)
                    structured_data["key_risks"] = [b.strip() for b in bullets if b.strip()]
            
            # Extract Peer Comparison
            if "peer comparison:" in text_lower:
                start_idx = text_lower.find("peer comparison:")
                if start_idx != -1:
                    end_markers = ["conclusion:", "investment rating:", "key strengths:"]
                    end_idx = len(analysis_text)
                    for marker in end_markers:
                        marker_idx = text_lower.find(marker, start_idx)
                        if marker_idx != -1 and marker_idx < end_idx:
                            end_idx = marker_idx
                    
                    peer_section = analysis_text[start_idx:end_idx]
                    structured_data["peer_comparison"] = peer_section.replace("Peer Comparison:", "").strip()
            
        except Exception as e:
            logger.warning(f"Error parsing structured analysis: {e}")
        
        return structured_data

    def _populate_field(self, lines: list[str],
                        pattern: str, possible_vals: list[str],
                        key: str,
                        structured_data: dict[str, str | list[Any]]):
        for line in lines:
            if pattern in line.lower():
                for x in possible_vals:
                    if x in line.lower():
                        structured_data[key] = x.upper()
                        logger.debug(f"Set {key} to: {structured_data[key]}")
                break

    def _store_analysis_results(self, stock_symbol: str, analysis: str, full_result: Dict[str, Any]):
        """Store analysis results in MCP context."""
        try:
            self.mcp_context.store_context(
                context_id=f"langgraph_analysis_{stock_symbol}_{int(datetime.now().timestamp())}",
                context_type=ContextType.STOCK_SUMMARY,
                data={
                    "symbol": stock_symbol,
                    "analysis": analysis,
                    "full_result": full_result,
                    "agent_id": self.agent_id,
                    "timestamp": datetime.now().isoformat()
                },
                agent_id=self.agent_id
            )
        except Exception as e:
            logger.error(f"Error storing analysis results: {e}")
    
    async def generate_recommendations(self, stock_symbol: str) -> Dict[str, Any]:
        """
        Generate investment recommendations using the LangGraph agent.
        
        Args:
            stock_symbol: Stock symbol to generate recommendations for
            
        Returns:
            Recommendation results
        """
        try:
            logger.info(f"Generating LangGraph recommendations for {stock_symbol}")
            
            # Prepare the input
            inputs = {
                "messages": [
                    {
                        "role": "user", 
                        "content": f"Based on the analysis of {stock_symbol}, provide specific investment recommendations including buy/hold/sell rating, target price, and reasoning."
                    }
                ]
            }
            
            # Run the agent
            result = self.agent.invoke(inputs)
            
            # Extract the final message
            final_message = result.get("messages", [])[-1] if result.get("messages") else None
            recommendations = final_message.content if final_message else "No recommendations generated"
            
            logger.info(f"Completed LangGraph recommendations for {stock_symbol}")
            
            return {
                "success": True,
                "stock_symbol": stock_symbol,
                "recommendations": recommendations,
                "full_result": result,
                "agent_id": self.agent_id,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating LangGraph recommendations for {stock_symbol}: {e}")
            return {
                "success": False,
                "stock_symbol": stock_symbol,
                "error": str(e),
                "agent_id": self.agent_id,
                "timestamp": datetime.now().isoformat()
            }
