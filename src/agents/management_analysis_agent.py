"""
Management Analysis Agent for performing management and governance analysis.
This agent focuses on management effectiveness and governance assessment.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

try:
    # Try relative imports first (when run as module)
    from .base_agent import BaseAgent, AgentState
    from ..tools.stock_data_tool import get_company_info
    from ..tools.web_search_tool import search_company_news
except ImportError:
    # Fall back to absolute imports (when run as script)
    from agents.base_agent import BaseAgent, AgentState
    from tools.stock_data_tool import get_company_info
    from tools.web_search_tool import search_company_news

logger = logging.getLogger(__name__)

class ManagementAnalysisAgent(BaseAgent):
    """
    Management Analysis Agent responsible for management and governance analysis.
    
    Tasks:
    - Management analysis and governance assessment
    - Management effectiveness evaluation
    """
    
    def __init__(self, agent_id: str, openai_api_key: str):
        """
        Initialize the Management Analysis Agent.
        
        Args:
            agent_id: Unique identifier for the agent
            openai_api_key: OpenAI API key for LLM calls
        """
        # Define available tools
        available_tools = [
            get_company_info,
            search_company_news
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
        Execute management analysis task.
        
        Args:
            stock_symbol: NSE stock symbol
            company_name: Full company name
            sector: Sector name
            context: Current context from previous agents
            
        Returns:
            AgentState with management analysis results
        """
        start_time = datetime.now()
        self.logger.info(f"Starting management analysis for {stock_symbol} ({company_name})")
        
        # Initialize state
        state = AgentState(
            agent_id=self.agent_id,
            stock_symbol=stock_symbol,
            company_name=company_name,
            sector=sector,
            current_task="management_analysis",
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
            
            # Perform management analysis
            management_result = await self._perform_management_analysis(stock_symbol, company_name, research_data)
            
            # Handle result
            if isinstance(management_result, Exception):
                state.errors.append(f"Management analysis failed: {str(management_result)}")
                state.results = {}
            else:
                state.results = management_result.get("data", {})
                state.tools_used.extend(management_result.get("tools_used", []))
            
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
            self.logger.error(f"Management analysis execution failed: {e}")
            state.errors.append(f"Management analysis execution failed: {str(e)}")
            state.end_time = datetime.now()
            state.confidence_score = 0.0
            return state
    
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

