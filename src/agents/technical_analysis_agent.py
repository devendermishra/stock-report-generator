"""
Technical Analysis Agent for performing technical analysis.
This agent focuses on technical indicators and price analysis.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

try:
    from .base_agent import BaseAgent, AgentState
    from ..tools.stock_data_tool import get_stock_metrics
    from ..tools.technical_analysis_formatter import TechnicalAnalysisFormatter
except ImportError:
    from agents.base_agent import BaseAgent, AgentState
    from tools.stock_data_tool import get_stock_metrics
    from tools.technical_analysis_formatter import TechnicalAnalysisFormatter

logger = logging.getLogger(__name__)

class TechnicalAnalysisAgent(BaseAgent):
    """
    Technical Analysis Agent responsible for technical analysis with indicators.
    
    Tasks:
    - Technical analysis with indicators
    - Price trend analysis
    - Volume analysis
    """
    
    def __init__(self, agent_id: str, openai_api_key: str):
        """
        Initialize the Technical Analysis Agent.
        
        Args:
            agent_id: Unique identifier for the agent
            openai_api_key: OpenAI API key for LLM calls
        """
        # Define available tools
        available_tools = [
            get_stock_metrics
        ]
        
        super().__init__(agent_id, openai_api_key, available_tools)
        
        # Initialize analysis tools
        self.technical_formatter = TechnicalAnalysisFormatter()
        
    async def execute_task(
        self,
        stock_symbol: str,
        company_name: str,
        sector: str,
        context: Dict[str, Any]
    ) -> AgentState:
        """
        Execute technical analysis task.
        
        Args:
            stock_symbol: NSE stock symbol
            company_name: Full company name
            sector: Sector name
            context: Current context from previous agents
            
        Returns:
            AgentState with technical analysis results
        """
        start_time = datetime.now()
        self.logger.info(f"Starting technical analysis for {stock_symbol} ({company_name})")
        
        # Initialize state
        state = AgentState(
            agent_id=self.agent_id,
            stock_symbol=stock_symbol,
            company_name=company_name,
            sector=sector,
            current_task="technical_analysis",
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
            
            # Perform technical analysis
            technical_result = await self._perform_technical_analysis(stock_symbol, research_data)
            
            # Handle result
            if isinstance(technical_result, Exception):
                state.errors.append(f"Technical analysis failed: {str(technical_result)}")
                state.results = {}
            else:
                state.results = technical_result.get("data", {})
                state.tools_used.extend(technical_result.get("tools_used", []))
            
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
            self.logger.error(f"Technical analysis execution failed: {e}")
            state.errors.append(f"Technical analysis execution failed: {str(e)}")
            state.end_time = datetime.now()
            state.confidence_score = 0.0
            return state
    
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

