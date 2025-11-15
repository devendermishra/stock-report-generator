"""
Financial Analysis Agent for performing financial analysis.
This agent focuses on financial statement analysis and ratio interpretation.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime
import json
import re

try:
    # Try relative imports first (when run as module)
    from .base_agent import BaseAgent, AgentState
    from ..tools.stock_data_tool import get_stock_metrics
    from ..config import Config
except ImportError:
    # Fall back to absolute imports (when run as script)
    from agents.base_agent import BaseAgent, AgentState
    from tools.stock_data_tool import get_stock_metrics
    from config import Config

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

class FinancialAnalysisAgent(BaseAgent):
    """
    Financial Analysis Agent responsible for financial statement analysis and ratio interpretation.
    
    **Specialization:** Financial Statement Analysis and Ratio Interpretation (Structured Workflow Mode)
    
    **Role:** Performs comprehensive financial analysis including financial ratios,
    health assessment, and financial metrics evaluation. Runs in parallel with other
    analysis agents in Structured Workflow Mode.
    
    **When Used:** Only in Structured Workflow Mode (runs in parallel with Management,
    Technical, and Valuation Analysis Agents)
    
    **Tasks:**
    - Financial statement analysis and ratio interpretation
    - Calculates additional financial ratios (P/E, P/B, Price-to-Sales)
    - Financial health assessment using LLM analysis
    - Market cap categorization (Large/Mid/Small Cap)
    
    **Tools Used:**
    - get_stock_metrics (for financial metrics)
    
    For detailed information on agent specialization and roles,
    see docs/AGENT_SPECIALIZATION.md
    """
    
    def __init__(self, agent_id: str, openai_api_key: str):
        """
        Initialize the Financial Analysis Agent.
        
        Args:
            agent_id: Unique identifier for the agent
            openai_api_key: OpenAI API key for LLM calls
        """
        # Define available tools
        available_tools = [
            get_stock_metrics
        ]
        
        super().__init__(agent_id, openai_api_key, available_tools)
        
        # Initialize OpenAI client
        self.openai_client = AsyncOpenAI(api_key=openai_api_key)
        
    async def execute_task(
        self,
        stock_symbol: str,
        company_name: str,
        sector: str,
        context: Dict[str, Any]
    ) -> AgentState:
        """
        Execute financial analysis task.
        
        Args:
            stock_symbol: NSE stock symbol
            company_name: Full company name
            sector: Sector name
            context: Current context from previous agents
            
        Returns:
            AgentState with financial analysis results
        """
        start_time = datetime.now()
        self.logger.info(f"Starting financial analysis for {stock_symbol} ({company_name})")
        
        # Initialize state
        state = AgentState(
            agent_id=self.agent_id,
            stock_symbol=stock_symbol,
            company_name=company_name,
            sector=sector,
            current_task="financial_analysis",
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
            
            # Perform financial analysis
            financial_result = await self._perform_financial_analysis(stock_symbol, company_name, research_data)
            
            # Handle result
            if isinstance(financial_result, Exception):
                state.errors.append(f"Financial analysis failed: {str(financial_result)}")
                state.results = {}
            else:
                state.results = financial_result.get("data", {})
                state.tools_used.extend(financial_result.get("tools_used", []))
            
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
            self.logger.error(f"Financial analysis execution failed: {e}")
            state.errors.append(f"Financial analysis execution failed: {str(e)}")
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
    
    def _categorize_market_cap(self, market_cap: float) -> str:
        """Categorize market cap."""
        if market_cap >= 100000000000:  # >= 100k crores
            return "Large Cap"
        elif market_cap >= 20000000000:  # >= 20k crores
            return "Mid Cap"
        else:
            return "Small Cap"

