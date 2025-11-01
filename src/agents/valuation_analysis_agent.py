"""
Valuation Analysis Agent for performing valuation analysis.
This agent focuses on valuation metrics and target price calculation.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

try:
    # Try relative imports first (when run as module)
    from .base_agent import BaseAgent, AgentState
    from ..tools.stock_data_tool import get_stock_metrics
    from ..tools.web_search_tool import search_market_trends
except ImportError:
    # Fall back to absolute imports (when run as script)
    from agents.base_agent import BaseAgent, AgentState
    from tools.stock_data_tool import get_stock_metrics
    from tools.web_search_tool import search_market_trends

logger = logging.getLogger(__name__)

class ValuationAnalysisAgent(BaseAgent):
    """
    Valuation Analysis Agent responsible for valuation analysis and target price calculation.
    
    Tasks:
    - Valuation analysis and target price calculation
    - Valuation metrics assessment
    """
    
    def __init__(self, agent_id: str, openai_api_key: str):
        """
        Initialize the Valuation Analysis Agent.
        
        Args:
            agent_id: Unique identifier for the agent
            openai_api_key: OpenAI API key for LLM calls
        """
        # Define available tools
        available_tools = [
            get_stock_metrics,
            search_market_trends
        ]
        
        super().__init__(agent_id, openai_api_key, available_tools)
        
    async def execute_task(
        self,
        stock_symbol: str,
        company_name: str,
        sector: str,
        context: Dict[str, Any]
    ) -> AgentState:
        """
        Execute valuation analysis task.
        
        Args:
            stock_symbol: NSE stock symbol
            company_name: Full company name
            sector: Sector name
            context: Current context from previous agents
            
        Returns:
            AgentState with valuation analysis results
        """
        start_time = datetime.now()
        self.logger.info(f"Starting valuation analysis for {stock_symbol} ({company_name})")
        
        # Initialize state
        state = AgentState(
            agent_id=self.agent_id,
            stock_symbol=stock_symbol,
            company_name=company_name,
            sector=sector,
            current_task="valuation_analysis",
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
            
            # Perform valuation analysis
            valuation_result = await self._perform_valuation_analysis(stock_symbol, sector, research_data)
            
            # Handle result
            if isinstance(valuation_result, Exception):
                state.errors.append(f"Valuation analysis failed: {str(valuation_result)}")
                state.results = {}
            else:
                state.results = valuation_result.get("data", {})
                state.tools_used.extend(valuation_result.get("tools_used", []))
            
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
            self.logger.error(f"Valuation analysis execution failed: {e}")
            state.errors.append(f"Valuation analysis execution failed: {str(e)}")
            state.end_time = datetime.now()
            state.confidence_score = 0.0
            return state
    
    async def _perform_valuation_analysis(self, stock_symbol: str, sector: str, research_data: Dict[str, Any] = None) -> Dict[str, Any]:
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

