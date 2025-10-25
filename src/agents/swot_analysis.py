"""
SWOT Analysis Agent for analyzing strengths, weaknesses, opportunities, and threats.
Specializes in comprehensive SWOT analysis for investment decision making.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import openai

try:
    # Try relative imports first (when run as module)
    from ..tools.stock_data_tool import StockDataTool
    from ..tools.web_search_tool import WebSearchTool
    from ..tools.openai_logger import openai_logger
    from ..graph.context_manager_mcp import MCPContextManager, ContextType
except ImportError:
    # Fall back to absolute imports (when run as script)
    from tools.stock_data_tool import StockDataTool
    from tools.web_search_tool import WebSearchTool
    from tools.openai_logger import openai_logger
    from graph.context_manager_mcp import MCPContextManager, ContextType

logger = logging.getLogger(__name__)

@dataclass
class SWOTAnalysis:
    """Represents the output of SWOT analysis."""
    company_name: str
    strengths: List[str]
    weaknesses: List[str]
    opportunities: List[str]
    threats: List[str]
    summary: str
    confidence_score: float

class SWOTAnalysisAgent:
    """
    SWOT Analysis Agent for analyzing strengths, weaknesses, opportunities, and threats.
    
    This agent specializes in:
    - Financial strength analysis
    - Competitive positioning assessment
    - Market opportunity identification
    - Threat analysis and risk assessment
    - Strategic positioning evaluation
    """
    
    def __init__(
        self,
        agent_id: str,
        mcp_context: MCPContextManager,
        stock_data_tool: StockDataTool,
        web_search_tool: WebSearchTool,
        openai_api_key: str
    ):
        """
        Initialize the SWOT Analysis Agent.
        
        Args:
            agent_id: Unique identifier for this agent
            mcp_context: MCP context manager for shared memory
            stock_data_tool: Stock data tool for financial metrics
            web_search_tool: Web search tool for market research
            openai_api_key: OpenAI API key for reasoning
        """
        self.agent_id = agent_id
        self.mcp_context = mcp_context
        self.stock_data_tool = stock_data_tool
        self.web_search_tool = web_search_tool
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        
    def analyze_swot(
        self,
        stock_symbol: str,
        company_name: str,
        sector: str
    ) -> SWOTAnalysis:
        """
        Perform comprehensive SWOT analysis.
        
        Args:
            stock_symbol: NSE stock symbol
            company_name: Full company name
            sector: Business sector
            
        Returns:
            SWOTAnalysis object
        """
        try:
            logger.info(f"Starting SWOT analysis for {stock_symbol}")
            
            # Step 1: Gather financial data
            financial_data = self._gather_financial_data(stock_symbol)
            
            # Step 2: Research market position
            market_data = self._research_market_position(company_name, sector, stock_symbol)
            
            # Step 3: Analyze competitive landscape
            competitive_data = self._analyze_competitive_landscape(company_name, sector)
            
            # Step 4: Identify opportunities and threats
            market_insights = self._identify_market_insights(company_name, sector)
            
            # Step 5: Synthesize SWOT analysis
            swot_analysis = self._synthesize_swot_analysis(
                company_name, financial_data, market_data, 
                competitive_data, market_insights
            )
            
            # Step 6: Store results in MCP context
            self._store_analysis_results(swot_analysis)
            
            logger.info(f"Completed SWOT analysis for {stock_symbol}")
            return swot_analysis
            
        except Exception as e:
            logger.error(f"Error in SWOT analysis: {e}")
            return self._create_fallback_analysis(company_name)
            
    def _gather_financial_data(self, stock_symbol: str) -> Dict[str, Any]:
        """Gather financial data for SWOT analysis."""
        try:
            # Get stock metrics
            stock_metrics = self.stock_data_tool.get_stock_metrics(stock_symbol)
            
            if not stock_metrics:
                return {}
                
            # Get historical data for trend analysis
            historical_data = self.stock_data_tool.get_historical_data(stock_symbol, period="1y")
            
            # Calculate financial ratios and trends
            financial_data = {
                "market_cap": stock_metrics.market_cap,
                "pe_ratio": stock_metrics.pe_ratio,
                "pb_ratio": stock_metrics.pb_ratio,
                "eps": stock_metrics.eps,
                "dividend_yield": stock_metrics.dividend_yield,
                "beta": stock_metrics.beta,
                "current_price": stock_metrics.current_price,
                "volume": stock_metrics.volume,
                "price_trend": self._analyze_price_trend(historical_data),
                "volatility": self._calculate_volatility(historical_data)
            }
            
            logger.info("Gathered financial data for SWOT analysis")
            return financial_data
            
        except Exception as e:
            logger.error(f"Error gathering financial data: {e}")
            return {}
            
    def _research_market_position(self, company_name: str, sector: str, stock_symbol: str) -> Dict[str, Any]:
        """Research company's market position."""
        try:
            # Search for company-specific news and analysis
            company_news = self.web_search_tool.search_company_news(company_name, stock_symbol)
            
            # Search for sector analysis
            sector_news = self.web_search_tool.search_sector_news(sector)
            
            market_data = {
                "company_news_count": len(company_news),
                "sector_news_count": len(sector_news),
                "recent_developments": [news.title for news in company_news[:5]],
                "sector_trends": [news.title for news in sector_news[:5]]
            }
            
            logger.info(f"Researched market position for {company_name}")
            return market_data
            
        except Exception as e:
            logger.error(f"Error researching market position: {e}")
            return {}
            
    def _analyze_competitive_landscape(self, company_name: str, sector: str) -> Dict[str, Any]:
        """Analyze competitive landscape."""
        try:
            # Search for competitive analysis using sector news
            competitive_results = self.web_search_tool.search_sector_news(f"{sector} competitive analysis {company_name}")
            
            competitive_data = {
                "competitor_analysis": [result.title for result in competitive_results],
                "market_share_info": len(competitive_results),
                "competitive_position": "Analysis pending"
            }
            
            logger.info(f"Analyzed competitive landscape for {company_name}")
            return competitive_data
            
        except Exception as e:
            logger.error(f"Error analyzing competitive landscape: {e}")
            return {}
            
    def _identify_market_insights(self, company_name: str, sector: str) -> Dict[str, Any]:
        """Identify market opportunities and threats."""
        try:
            # Search for opportunities using sector news
            opportunity_results = self.web_search_tool.search_sector_news(f"{sector} growth opportunities {company_name}")
            
            # Search for threats using sector news
            threat_results = self.web_search_tool.search_sector_news(f"{sector} challenges risks {company_name}")
            
            market_insights = {
                "opportunities": [result.title for result in opportunity_results],
                "threats": [result.title for result in threat_results],
                "market_outlook": "Analysis pending"
            }
            
            logger.info(f"Identified market insights for {company_name}")
            return market_insights
            
        except Exception as e:
            logger.error(f"Error identifying market insights: {e}")
            return {}
            
    def _synthesize_swot_analysis(
        self,
        company_name: str,
        financial_data: Dict[str, Any],
        market_data: Dict[str, Any],
        competitive_data: Dict[str, Any],
        market_insights: Dict[str, Any]
    ) -> SWOTAnalysis:
        """Synthesize all data into comprehensive SWOT analysis."""
        try:
            # Prepare context for AI analysis
            context = {
                "company_name": company_name,
                "financial_metrics": financial_data,
                "market_position": market_data,
                "competitive_landscape": competitive_data,
                "market_insights": market_insights
            }
            
            # Create prompt for AI analysis
            prompt = f"""
            Perform a comprehensive SWOT analysis for {company_name} based on the following data:
            
            Financial Metrics: {financial_data}
            Market Position: {market_data}
            Competitive Landscape: {competitive_data}
            Market Insights: {market_insights}
            
            Provide analysis in the following JSON format:
            {{
                "strengths": ["Strength 1", "Strength 2", "Strength 3", "Strength 4"],
                "weaknesses": ["Weakness 1", "Weakness 2", "Weakness 3", "Weakness 4"],
                "opportunities": ["Opportunity 1", "Opportunity 2", "Opportunity 3", "Opportunity 4"],
                "threats": ["Threat 1", "Threat 2", "Threat 3", "Threat 4"],
                "summary": "Comprehensive SWOT summary highlighting key strategic insights",
                "confidence_score": 0.85
            }}
            """
            
            # Call OpenAI for analysis with logging
            import time
            start_time = time.time()
            
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a senior business analyst specializing in SWOT analysis for investment decisions."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.1
                )
                
                duration_ms = int((time.time() - start_time) * 1000)
                analysis_text = response.choices[0].message.content
                
                # Log the OpenAI completion
                openai_logger.log_chat_completion(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a senior business analyst specializing in SWOT analysis for investment decisions."},
                        {"role": "user", "content": prompt}
                    ],
                    response=analysis_text,
                    usage=response.usage.__dict__ if response.usage else None,
                    duration_ms=duration_ms,
                    agent_name="SWOTAnalysisAgent"
                )
                
            except Exception as api_error:
                openai_logger.log_error(api_error, "gpt-4o-mini", "SWOTAnalysisAgent")
                raise api_error
            
            # Parse JSON response
            import json
            import re
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if json_match:
                analysis_data = json.loads(json_match.group())
            else:
                # Fallback if JSON parsing fails
                analysis_data = self._create_fallback_swot_data(company_name)
                
            # Create SWOTAnalysis object
            analysis = SWOTAnalysis(
                company_name=company_name,
                strengths=analysis_data.get("strengths", []),
                weaknesses=analysis_data.get("weaknesses", []),
                opportunities=analysis_data.get("opportunities", []),
                threats=analysis_data.get("threats", []),
                summary=analysis_data.get("summary", f"SWOT analysis for {company_name}"),
                confidence_score=analysis_data.get("confidence_score", 0.7)
            )
            
            logger.info(f"Synthesized SWOT analysis for {company_name}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error synthesizing SWOT analysis: {e}")
            return self._create_fallback_analysis(company_name)
            
    def _analyze_price_trend(self, historical_data: List[Any]) -> str:
        """Analyze price trend from historical data."""
        try:
            if len(historical_data) < 2:
                return "Insufficient data"
                
            # Simple trend analysis
            recent_prices = [data.close for data in historical_data[-20:]]
            if len(recent_prices) < 2:
                return "Insufficient data"
                
            if recent_prices[-1] > recent_prices[0]:
                return "Uptrend"
            elif recent_prices[-1] < recent_prices[0]:
                return "Downtrend"
            else:
                return "Sideways"
                
        except Exception as e:
            logger.error(f"Error analyzing price trend: {e}")
            return "Analysis pending"
            
    def _calculate_volatility(self, historical_data: List[Any]) -> float:
        """Calculate price volatility."""
        try:
            if len(historical_data) < 2:
                return 0.0
                
            prices = [data.close for data in historical_data[-30:]]
            if len(prices) < 2:
                return 0.0
                
            # Simple volatility calculation
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            volatility = sum(abs(r) for r in returns) / len(returns)
            
            return volatility
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.0
            
    def _create_fallback_swot_data(self, company_name: str) -> Dict[str, Any]:
        """Create fallback SWOT data."""
        return {
            "strengths": [
                "Established market presence",
                "Strong financial position",
                "Experienced management team",
                "Diversified business model"
            ],
            "weaknesses": [
                "Limited market share growth",
                "Dependency on specific markets",
                "Operational inefficiencies",
                "Regulatory compliance challenges"
            ],
            "opportunities": [
                "Market expansion potential",
                "Technology adoption opportunities",
                "Strategic partnerships",
                "New product development"
            ],
            "threats": [
                "Intense competition",
                "Economic volatility",
                "Regulatory changes",
                "Market disruption"
            ],
            "summary": f"SWOT analysis for {company_name} indicates a balanced position with growth potential.",
            "confidence_score": 0.6
        }
        
    def _create_fallback_analysis(self, company_name: str) -> SWOTAnalysis:
        """Create fallback SWOT analysis."""
        fallback_data = self._create_fallback_swot_data(company_name)
        
        return SWOTAnalysis(
            company_name=company_name,
            strengths=fallback_data["strengths"],
            weaknesses=fallback_data["weaknesses"],
            opportunities=fallback_data["opportunities"],
            threats=fallback_data["threats"],
            summary=fallback_data["summary"],
            confidence_score=fallback_data["confidence_score"]
        )
        
    def _store_analysis_results(self, analysis: SWOTAnalysis) -> None:
        """Store analysis results in MCP context."""
        try:
            analysis_data = {
                "company_name": analysis.company_name,
                "strengths": analysis.strengths,
                "weaknesses": analysis.weaknesses,
                "opportunities": analysis.opportunities,
                "threats": analysis.threats,
                "summary": analysis.summary,
                "confidence_score": analysis.confidence_score,
                "timestamp": datetime.now().isoformat()
            }
            
            self.mcp_context.store_context(
                context_id=f"swot_analysis_{analysis.company_name.replace(' ', '_')}",
                context_type=ContextType.SWOT_SUMMARY,
                data=analysis_data,
                agent_id=self.agent_id,
                metadata={"analysis_type": "swot_analysis"}
            )
            
            logger.info(f"Stored SWOT analysis results for {analysis.company_name}")
            
        except Exception as e:
            logger.error(f"Error storing analysis results: {e}")

