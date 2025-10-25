"""
Sector Researcher Agent for analyzing sector trends and peer comparison.
Specializes in sector-specific research and macro-economic analysis.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import openai

try:
    # Try relative imports first (when run as module)
    from ..tools.web_search_tool import WebSearchTool
    from ..tools.stock_data_tool import StockDataTool
    from ..tools.openai_logger import openai_logger
    from ..graph.context_manager_mcp import MCPContextManager, ContextType
except ImportError:
    # Fall back to absolute imports (when run as script)
    from tools.web_search_tool import WebSearchTool
    from tools.stock_data_tool import StockDataTool
    from tools.openai_logger import openai_logger
    from graph.context_manager_mcp import MCPContextManager, ContextType

logger = logging.getLogger(__name__)

@dataclass
class SectorAnalysis:
    """Represents the output of sector analysis."""
    sector_name: str
    summary: str
    trends: List[str]
    peer_comparison: Dict[str, Any]
    regulatory_environment: str
    outlook: str
    risks: List[str]
    opportunities: List[str]
    confidence_score: float

class SectorResearcherAgent:
    """
    Sector Researcher Agent for analyzing sector trends and peer comparison.
    
    This agent specializes in:
    - Sector trend analysis
    - Peer company comparison
    - Regulatory environment assessment
    - Macro-economic factor analysis
    - Sector outlook and risks
    """
    
    def __init__(
        self,
        agent_id: str,
        mcp_context: MCPContextManager,
        web_search_tool: WebSearchTool,
        stock_data_tool: StockDataTool,
        openai_api_key: str
    ):
        """
        Initialize the Sector Researcher Agent.
        
        Args:
            agent_id: Unique identifier for this agent
            mcp_context: MCP context manager for shared memory
            web_search_tool: Web search tool for news and trends
            stock_data_tool: Stock data tool for sector performance
            openai_api_key: OpenAI API key for reasoning
        """
        self.agent_id = agent_id
        self.mcp_context = mcp_context
        self.web_search_tool = web_search_tool
        self.stock_data_tool = stock_data_tool
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        
    def analyze_sector(
        self,
        stock_symbol: str,
        company_name: str,
        sector: str
    ) -> SectorAnalysis:
        """
        Perform comprehensive sector analysis.
        
        Args:
            stock_symbol: NSE stock symbol
            company_name: Full company name
            sector: Sector name
            
        Returns:
            SectorAnalysis object
        """
        try:
            logger.info(f"Starting sector analysis for {company_name} in {sector} sector")
            
            # Step 1: Gather sector news and trends
            sector_news = self._gather_sector_news(sector)
            
            # Step 2: Analyze sector performance
            sector_performance = self._analyze_sector_performance(sector)
            
            # Step 3: Research regulatory environment
            regulatory_news = self._research_regulatory_environment(sector)
            
            # Step 4: Get peer comparison data
            peer_analysis = self._analyze_peer_companies(sector)
            
            # Step 5: Synthesize findings using AI reasoning
            analysis = self._synthesize_sector_analysis(
                sector, sector_news, sector_performance, 
                regulatory_news, peer_analysis
            )
            
            # Step 6: Store results in MCP context
            self._store_analysis_results(analysis)
            
            logger.info(f"Completed sector analysis for {sector}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in sector analysis: {e}")
            return self._create_fallback_analysis(sector)
            
    def _gather_sector_news(self, sector: str) -> List[Dict[str, Any]]:
        """Gather recent sector news and trends."""
        try:
            # Search for sector-specific news
            news_results = self.web_search_tool.search_sector_news(
                sector=sector,
                days_back=7,
                include_analysis=True
            )
            
            # Search for regulatory news
            regulatory_results = self.web_search_tool.search_regulatory_news(sector)
            
            # Combine and process results
            all_news = []
            for result in news_results + regulatory_results:
                all_news.append({
                    "title": result.title,
                    "content": result.content,
                    "url": result.url,
                    "published_date": result.published_date,
                    "relevance_score": result.relevance_score
                })
                
            logger.info(f"Gathered {len(all_news)} news items for {sector} sector")
            return all_news
            
        except Exception as e:
            logger.error(f"Error gathering sector news: {e}")
            return []
            
    def _analyze_sector_performance(self, sector: str) -> Dict[str, Any]:
        """Analyze sector performance metrics."""
        try:
            # Get sector performance data
            performance_data = self.stock_data_tool.get_sector_performance(sector)
            
            if not performance_data:
                logger.warning(f"No performance data available for {sector}")
                return {}
                
            return {
                "avg_change": performance_data.get("avg_change", 0),
                "avg_pe": performance_data.get("avg_pe", 0),
                "performance": performance_data.get("performance", "neutral"),
                "stocks": performance_data.get("stocks", [])
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sector performance: {e}")
            return {}
            
    def _research_regulatory_environment(self, sector: str) -> List[Dict[str, Any]]:
        """Research regulatory environment for the sector."""
        try:
            # Search for regulatory news
            regulatory_results = self.web_search_tool.search_regulatory_news(sector)
            
            regulatory_info = []
            for result in regulatory_results:
                regulatory_info.append({
                    "title": result.title,
                    "content": result.content,
                    "url": result.url,
                    "published_date": result.published_date
                })
                
            logger.info(f"Found {len(regulatory_info)} regulatory items for {sector}")
            return regulatory_info
            
        except Exception as e:
            logger.error(f"Error researching regulatory environment: {e}")
            return []
            
    def _analyze_peer_companies(self, sector: str) -> Dict[str, Any]:
        """Analyze peer companies in the sector."""
        try:
            # Get sector performance data which includes peer stocks
            sector_data = self.stock_data_tool.get_sector_performance(sector)
            
            if not sector_data or "stocks" not in sector_data:
                return self._get_default_peer_analysis(sector)
                
            stocks = sector_data["stocks"]
            
            # Get detailed metrics for top 3 peers
            detailed_peers = []
            for stock in stocks[:3]:
                try:
                    # Get detailed metrics for each peer
                    peer_metrics = self.stock_data_tool.get_stock_metrics(stock.get('symbol', ''))
                    if peer_metrics:
                        # Get company name from stock data tool
                        company_name, _ = self.stock_data_tool.get_company_name_and_sector(stock.get('symbol', ''))
                        detailed_peers.append({
                            'symbol': stock.get('symbol', 'N/A'),
                            'name': company_name or stock.get('name', 'N/A'),
                            'current_price': peer_metrics.current_price,
                            'market_cap': peer_metrics.market_cap,
                            'pe_ratio': peer_metrics.pe_ratio,
                            'pb_ratio': peer_metrics.pb_ratio,
                            'eps': peer_metrics.eps,
                            'dividend_yield': peer_metrics.dividend_yield,
                            'change_percent': stock.get('change', 0),
                            'volume': peer_metrics.volume
                        })
                except Exception as e:
                    logger.warning(f"Could not get detailed metrics for {stock.get('symbol', '')}: {e}")
                    # Add basic info if detailed metrics fail
                    detailed_peers.append({
                        'symbol': stock.get('symbol', 'N/A'),
                        'name': stock.get('name', 'N/A'),
                        'change_percent': stock.get('change', 0),
                        'current_price': stock.get('price', 0)
                    })
            
            # Analyze peer performance
            peer_analysis = {
                "peer_count": len(stocks),
                "avg_performance": sum(stock.get("change", 0) for stock in stocks) / len(stocks) if stocks else 0,
                "top_performers": sorted(stocks, key=lambda x: x.get("change", 0), reverse=True)[:3],
                "underperformers": sorted(stocks, key=lambda x: x.get("change", 0))[:3],
                "sector_leader": max(stocks, key=lambda x: x.get("change", 0)) if stocks else None,
                "detailed_peers": detailed_peers[:3],  # Top 3 peers with detailed metrics
                "performance_summary": f"Sector average performance: {sum(stock.get('change', 0) for stock in stocks) / len(stocks):.2f}%" if stocks else "Performance data unavailable"
            }
            
            logger.info(f"Analyzed {len(stocks)} peer companies in {sector} with detailed metrics for {len(detailed_peers)} peers")
            return peer_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing peer companies: {e}")
            return self._get_default_peer_analysis(sector)
            
    def _get_default_peer_analysis(self, sector: str) -> Dict[str, Any]:
        """Get default peer analysis when sector data is unavailable."""
        # Define common peers by sector
        sector_peers = {
            "Financial Services": [
                {"symbol": "HDFCBANK", "name": "HDFC Bank Limited"},
                {"symbol": "SBIN", "name": "State Bank of India"},
                {"symbol": "KOTAKBANK", "name": "Kotak Mahindra Bank"}
            ],
            "Technology": [
                {"symbol": "TCS", "name": "Tata Consultancy Services"},
                {"symbol": "INFY", "name": "Infosys Limited"},
                {"symbol": "WIPRO", "name": "Wipro Limited"}
            ],
            "Energy": [
                {"symbol": "RELIANCE", "name": "Reliance Industries Limited"},
                {"symbol": "ONGC", "name": "Oil and Natural Gas Corporation"},
                {"symbol": "IOC", "name": "Indian Oil Corporation"}
            ]
        }
        
        default_peers = sector_peers.get(sector, [
            {"symbol": "N/A", "name": "Peer analysis pending"},
            {"symbol": "N/A", "name": "Peer analysis pending"},
            {"symbol": "N/A", "name": "Peer analysis pending"}
        ])
        
        return {
            "peer_count": len(default_peers),
            "avg_performance": 0.0,
            "top_performers": default_peers,
            "underperformers": default_peers,
            "sector_leader": default_peers[0] if default_peers else None,
            "detailed_peers": default_peers,
            "performance_summary": f"Peer analysis for {sector} sector is being processed. Key competitors and their performance metrics will be detailed based on sector analysis."
        }
            
    def _synthesize_sector_analysis(
        self,
        sector: str,
        sector_news: List[Dict[str, Any]],
        sector_performance: Dict[str, Any],
        regulatory_news: List[Dict[str, Any]],
        peer_analysis: Dict[str, Any]
    ) -> SectorAnalysis:
        """Synthesize all sector analysis data using AI reasoning."""
        try:
            # Prepare context for AI analysis
            context = {
                "sector": sector,
                "news_count": len(sector_news),
                "performance": sector_performance,
                "regulatory_count": len(regulatory_news),
                "peer_count": peer_analysis.get("peer_count", 0)
            }
            
            # Create prompt for AI analysis
            prompt = f"""
            Analyze the following sector data for {sector} sector and provide comprehensive insights:
            
            Sector Performance: {sector_performance}
            Peer Analysis: {peer_analysis}
            News Count: {len(sector_news)} recent articles
            Regulatory Items: {len(regulatory_news)} regulatory updates
            
            Provide analysis in the following JSON format:
            {{
                "summary": "Comprehensive sector overview",
                "trends": ["Trend 1", "Trend 2", "Trend 3"],
                "peer_comparison": {{
                    "sector_leader": "Company name",
                    "performance_summary": "Overall sector performance",
                    "competitive_landscape": "Competitive analysis"
                }},
                "regulatory_environment": "Regulatory outlook and impact",
                "outlook": "Sector outlook for next 6-12 months",
                "risks": ["Risk 1", "Risk 2", "Risk 3"],
                "opportunities": ["Opportunity 1", "Opportunity 2"],
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
                        {"role": "system", "content": "You are a senior sector analyst with expertise in Indian equity markets."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1500,
                    temperature=0.1
                )
                
                duration_ms = int((time.time() - start_time) * 1000)
                analysis_text = response.choices[0].message.content
                
                # Log the OpenAI completion
                openai_logger.log_chat_completion(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a senior sector analyst with expertise in Indian equity markets."},
                        {"role": "user", "content": prompt}
                    ],
                    response=analysis_text,
                    usage=response.usage.__dict__ if response.usage else None,
                    duration_ms=duration_ms,
                    agent_name="SectorResearcherAgent"
                )
                
            except Exception as api_error:
                openai_logger.log_error(api_error, "gpt-4o-mini", "SectorResearcherAgent")
                raise api_error
            
            # Parse response
            
            # Extract JSON from response
            import json
            import re
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if json_match:
                analysis_data = json.loads(json_match.group())
            else:
                # Fallback if JSON parsing fails
                analysis_data = self._create_fallback_analysis_data(sector)
                
            # Create SectorAnalysis object
            analysis = SectorAnalysis(
                sector_name=sector,
                summary=analysis_data.get("summary", f"Analysis of {sector} sector"),
                trends=analysis_data.get("trends", []),
                peer_comparison=peer_analysis,  # Use the actual peer analysis data
                regulatory_environment=analysis_data.get("regulatory_environment", "Regulatory environment analysis pending"),
                outlook=analysis_data.get("outlook", "Sector outlook analysis pending"),
                risks=analysis_data.get("risks", []),
                opportunities=analysis_data.get("opportunities", []),
                confidence_score=analysis_data.get("confidence_score", 0.7)
            )
            
            logger.info(f"Synthesized sector analysis for {sector}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error synthesizing sector analysis: {e}")
            return self._create_fallback_analysis(sector)
            
    def _store_analysis_results(self, analysis: SectorAnalysis) -> None:
        """Store analysis results in MCP context."""
        try:
            analysis_data = {
                "sector": analysis.sector_name,
                "sector_name": analysis.sector_name,
                "summary": analysis.summary,
                "trends": analysis.trends,
                "peer_comparison": analysis.peer_comparison,
                "regulatory_environment": analysis.regulatory_environment,
                "outlook": analysis.outlook,
                "risks": analysis.risks,
                "opportunities": analysis.opportunities,
                "confidence_score": analysis.confidence_score,
                "timestamp": datetime.now().isoformat()
            }
            
            self.mcp_context.store_context(
                context_id=f"sector_analysis_{analysis.sector_name}",
                context_type=ContextType.SECTOR_SUMMARY,
                data=analysis_data,
                agent_id=self.agent_id,
                metadata={"analysis_type": "sector_research"}
            )
            
            logger.info(f"Stored sector analysis results for {analysis.sector_name}")
            
        except Exception as e:
            logger.error(f"Error storing analysis results: {e}")
            
    def _create_fallback_analysis(self, sector: str) -> SectorAnalysis:
        """Create fallback analysis when main analysis fails."""
        return SectorAnalysis(
            sector_name=sector,
            summary=f"Basic analysis of {sector} sector. Detailed analysis unavailable due to data limitations.",
            trends=["Sector analysis pending", "Trend identification in progress"],
            peer_comparison={"status": "Analysis pending"},
            regulatory_environment="Regulatory analysis pending",
            outlook="Sector outlook analysis pending",
            risks=["Analysis pending"],
            opportunities=["Analysis pending"],
            confidence_score=0.3
        )
        
    def _create_fallback_analysis_data(self, sector: str) -> Dict[str, Any]:
        """Create fallback analysis data."""
        return {
            "summary": f"Basic analysis of {sector} sector",
            "trends": ["Analysis pending"],
            "peer_comparison": {"status": "Analysis pending"},
            "regulatory_environment": "Analysis pending",
            "outlook": "Analysis pending",
            "risks": ["Analysis pending"],
            "opportunities": ["Analysis pending"],
            "confidence_score": 0.3
        }
