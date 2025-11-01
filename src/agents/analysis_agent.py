"""
Analysis Agent for performing financial, management, and technical analysis.
This agent autonomously selects and uses tools to perform comprehensive analysis.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
import json
import re

try:
    # Try relative imports first (when run as module)
    from .base_agent import BaseAgent, AgentState
    from ..tools.stock_data_tool import get_stock_metrics, get_company_info
    from ..tools.technical_analysis_formatter import TechnicalAnalysisFormatter
    from ..tools.stock_data_calculator import StockDataCalculator
    from ..tools.web_search_tool import search_company_news, search_market_trends
    from ..config import Config
except ImportError:
    # Fall back to absolute imports (when run as script)
    from agents.base_agent import BaseAgent, AgentState
    from tools.stock_data_tool import get_stock_metrics, get_company_info
    from tools.technical_analysis_formatter import TechnicalAnalysisFormatter
    from tools.stock_data_calculator import StockDataCalculator
    from tools.web_search_tool import search_company_news, search_market_trends
    from config import Config

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

class AnalysisAgent(BaseAgent):
    """
    Analysis Agent responsible for comprehensive financial and technical analysis.
    
    Tasks:
    - Financial statement analysis and ratio interpretation
    - Management analysis and governance assessment
    - Technical analysis with indicators
    - Valuation analysis and target price calculation
    """
    
    def __init__(self, agent_id: str, openai_api_key: str):
        """
        Initialize the Analysis Agent.
        
        Args:
            agent_id: Unique identifier for the agent
            openai_api_key: OpenAI API key for LLM calls
        """
        # Define available tools
        available_tools = [
            get_stock_metrics,
            get_company_info,
            search_company_news,
            search_market_trends
        ]
        
        super().__init__(agent_id, openai_api_key, available_tools)
        
        # Initialize OpenAI client
        self.openai_client = AsyncOpenAI(api_key=openai_api_key)
        
        # Initialize analysis tools
        self.technical_formatter = TechnicalAnalysisFormatter()
        self.stock_calculator = StockDataCalculator()
        
    async def execute_task(
        self,
        stock_symbol: str,
        company_name: str,
        sector: str,
        context: Dict[str, Any]
    ) -> AgentState:
        """
        Execute analysis tasks to perform comprehensive financial and technical analysis.
        
        Args:
            stock_symbol: NSE stock symbol
            company_name: Full company name
            sector: Sector name
            context: Current context from previous agents
            
        Returns:
            AgentState with analysis results
        """
        start_time = datetime.now()
        self.logger.info(f"Starting analysis for {stock_symbol} ({company_name})")
        
        # Initialize state
        state = AgentState(
            agent_id=self.agent_id,
            stock_symbol=stock_symbol,
            company_name=company_name,
            sector=sector,
            current_task="comprehensive_analysis",
            context=context,
            results={},
            tools_used=[],
            confidence_score=0.0,
            errors=[],
            start_time=start_time
        )
        
        try:
            # Get research results from context to reuse data
            research_data = context.get("research_agent_results", {})
            
            # Execute analysis tasks in parallel
            analysis_tasks = [
                self._perform_financial_analysis(stock_symbol, company_name, research_data),
                self._perform_management_analysis(stock_symbol, company_name, research_data),
                self._perform_technical_analysis(stock_symbol, research_data),
                self._perform_valuation_analysis(stock_symbol, company_name, sector, research_data)
            ]
            
            # Execute all analysis tasks in parallel
            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Process results
            financial_analysis, management_analysis, technical_analysis, valuation_analysis = results
            
            # Handle exceptions
            if isinstance(financial_analysis, Exception):
                state.errors.append(f"Financial analysis failed: {str(financial_analysis)}")
                financial_analysis = {}
            else:
                state.tools_used.extend(financial_analysis.get("tools_used", []))
            
            if isinstance(management_analysis, Exception):
                state.errors.append(f"Management analysis failed: {str(management_analysis)}")
                management_analysis = {}
            else:
                state.tools_used.extend(management_analysis.get("tools_used", []))
            
            if isinstance(technical_analysis, Exception):
                state.errors.append(f"Technical analysis failed: {str(technical_analysis)}")
                technical_analysis = {}
            else:
                state.tools_used.extend(technical_analysis.get("tools_used", []))
            
            if isinstance(valuation_analysis, Exception):
                state.errors.append(f"Valuation analysis failed: {str(valuation_analysis)}")
                valuation_analysis = {}
            else:
                state.tools_used.extend(valuation_analysis.get("tools_used", []))
            
            # Compile results
            state.results = {
                "financial_analysis": financial_analysis.get("data", {}),
                "management_analysis": management_analysis.get("data", {}),
                "technical_analysis": technical_analysis.get("data", {}),
                "valuation_analysis": valuation_analysis.get("data", {}),
                "analysis_summary": self._generate_analysis_summary(
                    financial_analysis.get("data", {}),
                    management_analysis.get("data", {}),
                    technical_analysis.get("data", {}),
                    valuation_analysis.get("data", {})
                )
            }
            
            # Calculate confidence score
            state.confidence_score = self.calculate_confidence_score(
                state.results, state.tools_used, state.errors
            )
            
            # Update context
            state.context = self.update_context(context, state.results, self.agent_id)
            
            state.end_time = datetime.now()
            duration = (state.end_time - state.start_time).total_seconds()
            self.log_execution(state, duration)
            
            return state
            
        except Exception as e:
            self.logger.error(f"Analysis execution failed: {e}")
            state.errors.append(f"Analysis execution failed: {str(e)}")
            state.end_time = datetime.now()
            state.confidence_score = 0.0
            return state
    
    async def _perform_financial_analysis(self, stock_symbol: str, company_name: str, research_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform comprehensive financial analysis."""
        try:
            self.logger.info(f"Performing financial analysis for {stock_symbol}")
            tools_used = []
            research_data = research_data or {}
            
            # Check if get_stock_metrics was already executed by ResearchAgent
            company_data = research_data.get("company_data", {})
            stock_metrics = company_data.get("stock_metrics")
            
            if stock_metrics:
                self.logger.info("Reusing stock_metrics from ResearchAgent")
            else:
                # Get detailed stock metrics
                stock_metrics = get_stock_metrics.invoke({"symbol": stock_symbol})
                tools_used.append("get_stock_metrics")
            
            # Calculate additional financial ratios
            financial_ratios = self._calculate_financial_ratios(stock_metrics)
            
            # Analyze financial health
            financial_health = await self._analyze_financial_health(stock_metrics, financial_ratios)
            
            return {
                "data": {
                    "stock_metrics": stock_metrics,
                    "financial_ratios": financial_ratios,
                    "financial_health": financial_health,
                    "analysis_type": "comprehensive_financial"
                },
                "tools_used": tools_used
            }
            
        except Exception as e:
            self.logger.error(f"Financial analysis failed: {e}")
            return {"data": {}, "tools_used": [], "error": str(e)}
    
    async def _perform_management_analysis(self, stock_symbol: str, company_name: str, research_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform management and governance analysis."""
        try:
            self.logger.info(f"Performing management analysis for {stock_symbol}")
            tools_used = []
            research_data = research_data or {}
            
            # Check if get_company_info was already executed by ResearchAgent
            company_data = research_data.get("company_data", {})
            company_info = company_data.get("company_info")
            
            if company_info:
                self.logger.info("Reusing company_info from ResearchAgent")
            else:
                # Get company information
                company_info = get_company_info.invoke({"symbol": stock_symbol})
                tools_used.append("get_company_info")
            
            # Check if search_company_news was already executed by ResearchAgent
            news_data = research_data.get("news_data", {})
            management_news = news_data.get("company_news")
            
            if management_news:
                self.logger.info("Reusing company_news from ResearchAgent")
            else:
                # Search for management-related news
                management_news = search_company_news.invoke({
                    "company_name": company_name,
                    "stock_symbol": stock_symbol,
                    "days_back": 90
                })
                tools_used.append("search_company_news")
            
            # Analyze management effectiveness
            management_assessment = self._assess_management_effectiveness(company_info, management_news)
            
            return {
                "data": {
                    "company_info": company_info,
                    "management_news": management_news,
                    "management_assessment": management_assessment,
                    "analysis_type": "management_governance"
                },
                "tools_used": tools_used
            }
            
        except Exception as e:
            self.logger.error(f"Management analysis failed: {e}")
            return {"data": {}, "tools_used": [], "error": str(e)}
    
    async def _perform_technical_analysis(self, stock_symbol: str, research_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform technical analysis with indicators."""
        try:
            self.logger.info(f"Performing technical analysis for {stock_symbol}")
            tools_used = []
            research_data = research_data or {}
            
            # Check if get_stock_metrics was already executed by ResearchAgent
            company_data = research_data.get("company_data", {})
            stock_metrics = company_data.get("stock_metrics")
            
            if stock_metrics:
                self.logger.info("Reusing stock_metrics from ResearchAgent for technical analysis")
            else:
                # Get stock metrics for technical analysis
                stock_metrics = get_stock_metrics.invoke({"symbol": stock_symbol})
                tools_used.append("get_stock_metrics")
            
            # Calculate technical indicators
            technical_indicators = self._calculate_technical_indicators(stock_metrics)
            
            # Generate technical analysis summary
            technical_summary = self.technical_formatter.format_technical_analysis(technical_indicators)
            
            return {
                "data": {
                    "stock_metrics": stock_metrics,
                    "technical_indicators": technical_indicators,
                    "technical_summary": technical_summary,
                    "analysis_type": "technical_analysis"
                },
                "tools_used": tools_used
            }
            
        except Exception as e:
            self.logger.error(f"Technical analysis failed: {e}")
            return {"data": {}, "tools_used": [], "error": str(e)}
    
    async def _perform_valuation_analysis(self, stock_symbol: str, company_name: str, sector: str, research_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform valuation analysis and target price calculation."""
        try:
            self.logger.info(f"Performing valuation analysis for {stock_symbol}")
            tools_used = []
            research_data = research_data or {}
            
            # Check if get_stock_metrics was already executed by ResearchAgent
            company_data = research_data.get("company_data", {})
            stock_metrics = company_data.get("stock_metrics")
            
            if stock_metrics:
                self.logger.info("Reusing stock_metrics from ResearchAgent for valuation analysis")
            else:
                # Get stock metrics
                stock_metrics = get_stock_metrics.invoke({"symbol": stock_symbol})
                tools_used.append("get_stock_metrics")
            
            # Check if search_market_trends was already executed by ResearchAgent
            sector_data = research_data.get("sector_data", {})
            valuation_trends = sector_data.get("market_trends")
            
            if valuation_trends:
                self.logger.info("Reusing market_trends from ResearchAgent")
            else:
                # Search for sector valuation trends
                valuation_trends = search_market_trends.invoke({
                    "query": f"{sector} sector valuation trends P/E ratios India",
                    "max_results": 10
                })
                tools_used.append("search_market_trends")
            
            # Calculate valuation metrics
            valuation_metrics = self._calculate_valuation_metrics(stock_metrics, sector)
            
            # Generate target price
            target_price = self._calculate_target_price(stock_metrics, valuation_metrics)
            
            return {
                "data": {
                    "stock_metrics": stock_metrics,
                    "valuation_trends": valuation_trends,
                    "valuation_metrics": valuation_metrics,
                    "target_price": target_price,
                    "analysis_type": "valuation_analysis"
                },
                "tools_used": tools_used
            }
            
        except Exception as e:
            self.logger.error(f"Valuation analysis failed: {e}")
            return {"data": {}, "tools_used": [], "error": str(e)}
    
    def _calculate_financial_ratios(self, stock_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate additional financial ratios."""
        try:
            ratios = {}
            
            # Basic ratios from stock metrics
            ratios["pe_ratio"] = stock_metrics.get("pe_ratio")
            ratios["pb_ratio"] = stock_metrics.get("pb_ratio")
            ratios["dividend_yield"] = stock_metrics.get("dividend_yield")
            ratios["beta"] = stock_metrics.get("beta")
            
            # Calculate additional ratios if data available
            current_price = stock_metrics.get("current_price", 0)
            market_cap = stock_metrics.get("market_cap", 0)
            
            if current_price and market_cap:
                ratios["price_to_sales"] = market_cap / (market_cap * 0.1) if market_cap > 0 else None  # Simplified
                ratios["market_cap_category"] = self._categorize_market_cap(market_cap)
            
            return ratios
            
        except Exception as e:
            self.logger.error(f"Financial ratios calculation failed: {e}")
            return {}
    
    async def _analyze_financial_health(self, stock_metrics: Dict[str, Any], ratios: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall financial health using LLM."""
        try:
            # Extract key stock metrics for analysis
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
            
            # Build prompt with both stock metrics and ratios
            prompt = f"""Analyze financial health from stock metrics and ratios. Return JSON:

<stock_metrics>
{json.dumps(key_metrics, indent=2, default=str)}
</stock_metrics>

<ratios>
{json.dumps(ratios, indent=2, default=str)}
</ratios>

**Instructions:**
- Consider both stock metrics (price, market cap, volume, 52-week range, beta, dividend yield, EPS) and financial ratios (P/E, P/B)
- Assess financial health holistically using all available metrics
- Consider valuation relative to 52-week range
- Factor in market cap category and trading volume trends

Output format:
{{
    "health_score": <0-100 integer>,
    "health_factors": ["factor1", "factor2"],
    "overall_assessment": "<brief assessment>"
}}
"""
            
            response = await self.openai_client.chat.completions.create(
                model=Config.DEFAULT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial analyst. Analyze financial health using both stock metrics and financial ratios, and return structured JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            # Parse response
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
                # Fallback to simple assessment
                self.logger.warning("Failed to parse LLM response, using fallback")
                return {
                    "health_score": 50,
                    "health_factors": ["Analysis completed"],
                    "overall_assessment": "Financial health assessment completed"
                }
            
        except Exception as e:
            self.logger.error(f"Financial health analysis failed: {e}")
            return {"health_score": 0, "health_factors": [], "overall_assessment": "Unable to assess"}
    
    def _assess_management_effectiveness(self, company_info: Dict[str, Any], management_news: Dict[str, Any]) -> Dict[str, Any]:
        """Assess management effectiveness and governance."""
        try:
            assessment = {
                "governance_score": 0,
                "key_factors": [],
                "recent_developments": []
            }
            
            # Analyze company information
            if company_info.get("employees"):
                employee_count = company_info["employees"]
                if employee_count > 10000:
                    assessment["key_factors"].append("Large organization with established management")
                    assessment["governance_score"] += 20
                elif employee_count > 1000:
                    assessment["key_factors"].append("Mid-size organization with growing management")
                    assessment["governance_score"] += 15
                else:
                    assessment["key_factors"].append("Small organization with lean management")
                    assessment["governance_score"] += 10
            
            # Analyze recent news for management developments
            if management_news.get("results"):
                news_count = len(management_news["results"])
                assessment["recent_developments"].append(f"{news_count} recent news articles analyzed")
                
                # Look for positive/negative keywords in news
                positive_keywords = ["growth", "expansion", "profit", "revenue", "acquisition", "partnership"]
                negative_keywords = ["loss", "decline", "layoff", "scandal", "investigation", "fine"]
                
                positive_count = 0
                negative_count = 0
                
                for result in management_news["results"][:5]:  # Analyze top 5 results
                    title = result.get("title", "").lower()
                    content = result.get("content", "").lower()
                    
                    for keyword in positive_keywords:
                        if keyword in title or keyword in content:
                            positive_count += 1
                    
                    for keyword in negative_keywords:
                        if keyword in title or keyword in content:
                            negative_count += 1
                
                if positive_count > negative_count:
                    assessment["key_factors"].append("Positive management developments")
                    assessment["governance_score"] += 15
                elif negative_count > positive_count:
                    assessment["key_factors"].append("Some management concerns")
                    assessment["governance_score"] += 5
                else:
                    assessment["key_factors"].append("Neutral management developments")
                    assessment["governance_score"] += 10
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Management effectiveness assessment failed: {e}")
            return {"governance_score": 0, "key_factors": [], "recent_developments": []}
    
    def _calculate_technical_indicators(self, stock_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate technical indicators."""
        try:
            indicators = {}
            
            current_price = stock_metrics.get("current_price", 0)
            high_52w = stock_metrics.get("52_week_high", 0)
            low_52w = stock_metrics.get("52_week_low", 0)
            
            if current_price and high_52w and low_52w:
                # Calculate position within 52-week range
                range_position = (current_price - low_52w) / (high_52w - low_52w) if high_52w != low_52w else 0.5
                indicators["range_position"] = range_position
                
                # Determine trend based on position
                if range_position > 0.8:
                    indicators["trend"] = "Near 52-week high"
                elif range_position < 0.2:
                    indicators["trend"] = "Near 52-week low"
                else:
                    indicators["trend"] = "Mid-range"
                
                # Calculate potential upside/downside
                upside_potential = ((high_52w - current_price) / current_price) * 100
                downside_risk = ((current_price - low_52w) / current_price) * 100
                
                indicators["upside_potential"] = upside_potential
                indicators["downside_risk"] = downside_risk
            
            # Volume analysis
            volume = stock_metrics.get("volume", 0)
            avg_volume = stock_metrics.get("avg_volume", 0)
            
            if volume and avg_volume:
                volume_ratio = volume / avg_volume if avg_volume > 0 else 1
                indicators["volume_ratio"] = volume_ratio
                
                if volume_ratio > 1.5:
                    indicators["volume_trend"] = "High volume (increased interest)"
                elif volume_ratio < 0.5:
                    indicators["volume_trend"] = "Low volume (reduced interest)"
                else:
                    indicators["volume_trend"] = "Normal volume"
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Technical indicators calculation failed: {e}")
            return {}
    
    def _calculate_valuation_metrics(self, stock_metrics: Dict[str, Any], sector: str) -> Dict[str, Any]:
        """Calculate valuation metrics."""
        try:
            metrics = {}
            
            pe_ratio = stock_metrics.get("pe_ratio")
            pb_ratio = stock_metrics.get("pb_ratio")
            current_price = stock_metrics.get("current_price", 0)
            market_cap = stock_metrics.get("market_cap", 0)
            
            if pe_ratio:
                metrics["pe_ratio"] = pe_ratio
                
                # PE ratio assessment
                if pe_ratio < 15:
                    metrics["pe_assessment"] = "Undervalued (low P/E)"
                elif pe_ratio < 25:
                    metrics["pe_assessment"] = "Fairly valued (reasonable P/E)"
                else:
                    metrics["pe_assessment"] = "Overvalued (high P/E)"
            
            if pb_ratio:
                metrics["pb_ratio"] = pb_ratio
                
                # PB ratio assessment
                if pb_ratio < 1:
                    metrics["pb_assessment"] = "Trading below book value"
                elif pb_ratio < 3:
                    metrics["pb_assessment"] = "Reasonable price-to-book"
                else:
                    metrics["pb_assessment"] = "High price-to-book"
            
            # Market cap analysis
            if market_cap:
                metrics["market_cap"] = market_cap
                metrics["market_cap_category"] = self._categorize_market_cap(market_cap)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Valuation metrics calculation failed: {e}")
            return {}
    
    def _calculate_target_price(self, stock_metrics: Dict[str, Any], valuation_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate target price based on analysis."""
        try:
            current_price = stock_metrics.get("current_price", 0)
            pe_ratio = stock_metrics.get("pe_ratio")
            
            if not current_price or not pe_ratio:
                return {"target_price": current_price, "method": "insufficient_data"}
            
            # Simple target price calculation based on P/E ratio
            if pe_ratio < 15:
                # Undervalued - potential upside
                target_price = current_price * 1.2
                recommendation = "BUY"
            elif pe_ratio < 25:
                # Fairly valued - moderate upside
                target_price = current_price * 1.1
                recommendation = "HOLD"
            else:
                # Overvalued - potential downside
                target_price = current_price * 0.9
                recommendation = "SELL"
            
            return {
                "target_price": round(target_price, 2),
                "current_price": current_price,
                "upside_potential": round(((target_price - current_price) / current_price) * 100, 2),
                "recommendation": recommendation,
                "method": "pe_ratio_based"
            }
            
        except Exception as e:
            self.logger.error(f"Target price calculation failed: {e}")
            return {"target_price": stock_metrics.get("current_price", 0), "method": "error"}
    
    def _categorize_market_cap(self, market_cap: float) -> str:
        """Categorize market cap."""
        if market_cap >= 100000000000:  # >= 100k crores
            return "Large Cap"
        elif market_cap >= 20000000000:  # >= 20k crores
            return "Mid Cap"
        else:
            return "Small Cap"
    
    def _get_health_assessment(self, health_score: int) -> str:
        """Get health assessment based on score."""
        if health_score >= 80:
            return "Excellent financial health"
        elif health_score >= 60:
            return "Good financial health"
        elif health_score >= 40:
            return "Fair financial health"
        else:
            return "Poor financial health"
    
    def _generate_analysis_summary(
        self,
        financial_analysis: Dict[str, Any],
        management_analysis: Dict[str, Any],
        technical_analysis: Dict[str, Any],
        valuation_analysis: Dict[str, Any]
    ) -> str:
        """Generate comprehensive analysis summary."""
        try:
            summary_parts = []
            
            # Financial analysis summary
            if financial_analysis.get("financial_health"):
                health = financial_analysis["financial_health"]
                summary_parts.append(
                    f"**Financial Health:** {health.get('overall_assessment', 'N/A')} "
                    f"(Score: {health.get('health_score', 0)}/100)"
                )
            
            # Management analysis summary
            if management_analysis.get("management_assessment"):
                mgmt = management_analysis["management_assessment"]
                summary_parts.append(
                    f"**Management Assessment:** Governance score {mgmt.get('governance_score', 0)}/100"
                )
            
            # Technical analysis summary
            if technical_analysis.get("technical_indicators"):
                tech = technical_analysis["technical_indicators"]
                summary_parts.append(
                    f"**Technical Analysis:** {tech.get('trend', 'N/A')} with "
                    f"{tech.get('volume_trend', 'normal volume')}"
                )
            
            # Valuation summary
            if valuation_analysis.get("target_price"):
                target = valuation_analysis["target_price"]
                summary_parts.append(
                    f"**Valuation:** Target price ₹{target.get('target_price', 'N/A')} "
                    f"({target.get('recommendation', 'N/A')} recommendation)"
                )
            
            return "\n\n".join(summary_parts) if summary_parts else "Analysis completed successfully."
            
        except Exception as e:
            self.logger.error(f"Analysis summary generation failed: {e}")
            return "Analysis completed with some limitations."
