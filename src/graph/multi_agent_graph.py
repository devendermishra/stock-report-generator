"""
LangGraph-based Multi-Agent System for Autonomous Stock Research Report Generation.
This orchestrator allows agents to autonomously select and use tools based on context.
"""

import logging
from typing import Dict, Any, List, Optional, cast, Annotated

from pydantic import BaseModel
from typing_extensions import TypedDict
from datetime import datetime
import asyncio

from langgraph.graph import StateGraph, END
from langchain_core.runnables import Runnable, RunnableLambda
from langgraph.graph.state import CompiledStateGraph

try:
    # Try relative imports first (when run as module)
    from ..agents import (
        ResearchAgent, ResearchPlannerAgent, ReportAgent,
        FinancialAnalysisAgent, ManagementAnalysisAgent,
        TechnicalAnalysisAgent, ValuationAnalysisAgent
    )
    from ..config import Config
except ImportError:
    # Fall back to absolute imports (when run as script)
    from agents import (
        ResearchAgent, ResearchPlannerAgent, ReportAgent,
        FinancialAnalysisAgent, ManagementAnalysisAgent,
        TechnicalAnalysisAgent, ValuationAnalysisAgent
    )
    from config import Config

logger = logging.getLogger(__name__)

def reduce_errors(left: List[str], right: List[str]) -> List[str]:
    """Reducer function to merge error lists from concurrent nodes."""
    if left is None:
        left = []
    if right is None:
        right = []
    # Combine lists and remove duplicates while preserving order
    combined = left + right
    seen = set()
    result = []
    for item in combined:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

class MultiAgentState(BaseModel):
    """State for the multi-agent workflow."""
    stock_symbol: str
    company_name: str
    sector: str
    current_agent: str
    research_plan_results: Optional[Dict[str, Any]] = None
    research_results: Optional[Dict[str, Any]] = None
    analysis_results: Optional[Dict[str, Any]] = None
    financial_analysis_results: Optional[Dict[str, Any]] = None
    management_analysis_results: Optional[Dict[str, Any]] = None
    technical_analysis_results: Optional[Dict[str, Any]] = None
    valuation_analysis_results: Optional[Dict[str, Any]] = None
    report_results: Optional[Dict[str, Any]] = None
    final_report: Optional[str] = None
    pdf_path: Optional[str] = None
    errors: Annotated[List[str], reduce_errors] = []
    start_time: datetime
    end_time: Optional[datetime] = None

class MultiAgentOrchestrator:
    """
    LangGraph-based orchestrator for the multi-agent stock research system.
    
    This orchestrator manages multiple autonomous agents:
    1. ResearchPlannerAgent - Creates structured research plan with ordered tool calls
    2. ResearchAgent - Gathers company and sector data based on the plan
    3. FinancialAnalysisAgent - Performs financial analysis
    4. ManagementAnalysisAgent - Performs management and governance analysis
    5. TechnicalAnalysisAgent - Performs technical analysis
    6. ValuationAnalysisAgent - Performs valuation analysis
    7. ReportAgent - Synthesizes data into comprehensive reports
    """
    
    def __init__(self, openai_api_key: str):
        """
        Initialize the multi-agent orchestrator.
        
        Args:
            openai_api_key: OpenAI API key for LLM calls
        """
        self.openai_api_key = openai_api_key
        
        # Initialize agents
        self.research_planner_agent = ResearchPlannerAgent("research_planner_agent", openai_api_key)
        self.research_agent = ResearchAgent("research_agent", openai_api_key)
        self.financial_analysis_agent = FinancialAnalysisAgent("financial_analysis_agent", openai_api_key)
        self.management_analysis_agent = ManagementAnalysisAgent("management_analysis_agent", openai_api_key)
        self.technical_analysis_agent = TechnicalAnalysisAgent("technical_analysis_agent", openai_api_key)
        self.valuation_analysis_agent = ValuationAnalysisAgent("valuation_analysis_agent", openai_api_key)
        self.report_agent = ReportAgent("report_agent", openai_api_key)
        
        # Build the graph
        self.graph = self._build_graph()
        
    def _build_graph(self) -> CompiledStateGraph[Any, Any, Any, Any]:
        """Build the LangGraph workflow with parallel analysis agents."""
        # Create the state graph
        workflow = StateGraph(MultiAgentState)
        
        # Add agent nodes (cast for type-checker; async callables are supported at runtime)
        planner_node = cast(Runnable[MultiAgentState, MultiAgentState],
                             RunnableLambda(self._research_planner_agent_node))
        research_node = cast(Runnable[MultiAgentState, MultiAgentState],
                             RunnableLambda(self._research_agent_node))
        financial_analysis_node = cast(Runnable[MultiAgentState, MultiAgentState],
                                      RunnableLambda(self._financial_analysis_agent_node))
        management_analysis_node = cast(Runnable[MultiAgentState, MultiAgentState],
                                        RunnableLambda(self._management_analysis_agent_node))
        technical_analysis_node = cast(Runnable[MultiAgentState, MultiAgentState],
                                       RunnableLambda(self._technical_analysis_agent_node))
        valuation_analysis_node = cast(Runnable[MultiAgentState, MultiAgentState],
                                       RunnableLambda(self._valuation_analysis_agent_node))
        report_node = cast(Runnable[MultiAgentState, MultiAgentState],
                           RunnableLambda(self._report_agent_node))

        workflow.add_node("research_planner_agent", planner_node)
        workflow.add_node("research_agent", research_node)
        workflow.add_node("financial_analysis_agent", financial_analysis_node)
        workflow.add_node("management_analysis_agent", management_analysis_node)
        workflow.add_node("technical_analysis_agent", technical_analysis_node)
        workflow.add_node("valuation_analysis_agent", valuation_analysis_node)
        workflow.add_node("report_agent", report_node)
        
        # Add conditional edges for ResearchPlanner -> ResearchAgent flow
        workflow.add_conditional_edges(
            "research_planner_agent",
            self._should_continue_after_planning,
            {
                "continue": "research_agent",
                "error": END
            }
        )
        
        # Add 4 parallel unconditional edges from research_agent to all analysis agents
        # These will execute in parallel. Each analysis node will check for errors in state.
        workflow.add_edge("research_agent", "financial_analysis_agent")
        workflow.add_edge("research_agent", "management_analysis_agent")
        workflow.add_edge("research_agent", "technical_analysis_agent")
        workflow.add_edge("research_agent", "valuation_analysis_agent")
        
        # Each analysis agent node has an edge to report_agent
        # All analysis agents route to report_agent regardless of errors
        # This ensures report_agent waits for ALL analysis agents to complete
        # LangGraph automatically waits for all incoming edges before executing a node
        workflow.add_edge("financial_analysis_agent", "report_agent")
        workflow.add_edge("management_analysis_agent", "report_agent")
        workflow.add_edge("technical_analysis_agent", "report_agent")
        workflow.add_edge("valuation_analysis_agent", "report_agent")
        
        workflow.add_edge("report_agent", END)
        
        # Set entry point to research planner
        workflow.set_entry_point("research_planner_agent")
        
        # Compile the graph
        return workflow.compile()
    
    async def run_workflow(
        self,
        stock_symbol: str,
        company_name: str,
        sector: str
    ) -> Dict[str, Any]:
        """
        Run the complete multi-agent workflow.
        
        Args:
            stock_symbol: NSE stock symbol
            company_name: Full company name
            sector: Sector name
            
        Returns:
            Dictionary containing workflow results
        """
        try:
            logger.info(f"Starting multi-agent workflow for {stock_symbol}")
            
            # Create initial state
            initial_state = MultiAgentState(
                stock_symbol=stock_symbol,
                company_name=company_name,
                sector=sector,
                current_agent="research_planner_agent",
                research_plan_results=None,
                research_results=None,
                analysis_results=None,
                financial_analysis_results=None,
                management_analysis_results=None,
                technical_analysis_results=None,
                valuation_analysis_results=None,
                report_results=None,
                final_report=None,
                pdf_path=None,
                errors=[],
                start_time=datetime.now(),
                end_time=None
            )
            
            # Run the workflow
            final_state = await self.graph.ainvoke(initial_state)
            
            # Calculate duration
            if final_state.get("start_time") and final_state.get("end_time"):
                duration = (final_state["end_time"] - final_state["start_time"]).total_seconds()
            else:
                duration = 0
            
            # Prepare results
            results = {
                "stock_symbol": stock_symbol,
                "company_name": company_name,
                "sector": sector,
                "workflow_status": "completed" if not final_state.get("errors") else "completed_with_errors",
                "start_time": final_state.get("start_time").isoformat() if final_state.get("start_time") else None,
                "end_time": final_state.get("end_time").isoformat() if final_state.get("end_time") else None,
                "duration_seconds": duration,
                "errors": final_state.get("errors", []),
                "research_plan_results": final_state.get("research_plan_results"),
                "research_results": final_state.get("research_results"),
                "analysis_results": final_state.get("analysis_results"),
                "financial_analysis_results": final_state.get("financial_analysis_results"),
                "management_analysis_results": final_state.get("management_analysis_results"),
                "technical_analysis_results": final_state.get("technical_analysis_results"),
                "valuation_analysis_results": final_state.get("valuation_analysis_results"),
                "report_results": final_state.get("report_results"),
                "final_report": final_state.get("final_report"),
                "pdf_path": final_state.get("pdf_path")
            }
            
            logger.info(f"Completed multi-agent workflow for {stock_symbol} in {duration:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Multi-agent workflow failed: {e}")
            return {
                "stock_symbol": stock_symbol,
                "company_name": company_name,
                "sector": sector,
                "workflow_status": "failed",
                "error": str(e),
                "errors": [str(e)]
            }
    
    async def _research_planner_agent_node(self, state: MultiAgentState) -> dict:
        """Execute research planner agent."""
        try:
            logger.info(f"Executing research planner agent for {state.stock_symbol}")
            
            # Get available tools from research agent
            available_tools = self.research_agent.available_tools
            
            # Prepare context with country and available tools
            context = {
                "country": "India",  # Default to India for NSE stocks
                "available_tools": available_tools
            }
            
            # Convert state to dict for agent execution
            state_dict = {
                "stock_symbol": state.stock_symbol,
                "company_name": state.company_name,
                "sector": state.sector,
                "context": context
            }
            
            # Execute research planner agent using partial state update method
            partial_update = await self.research_planner_agent.execute_task_partial(state_dict)
            
            # Extract results and update state
            plan_results = partial_update.get("results", {})
            agent_errors = partial_update.get("errors", [])
            
            # Return state changes only (do not mutate input state)
            # Note: Return only new errors; reducer will merge with existing errors
            logger.info(f"Research planner agent completed for {state.stock_symbol}")
            return {
                "research_plan_results": plan_results,
                "current_agent": "research_agent",
                "errors": agent_errors if agent_errors else []
            }
            
        except Exception as e:
            logger.error(f"Research planner agent failed: {e}")
            return {
                "errors": [f"Research planner agent failed: {str(e)}"],
                "current_agent": "error"
            }
    
    async def _research_agent_node(self, state: MultiAgentState) -> dict:
        """Execute research agent."""
        try:
            logger.info(f"Executing research agent for {state.stock_symbol}")
            
            # Prepare context with research plan if available
            context = {}
            
            # Include research plan in context for research agent to use
            if state.research_plan_results:
                context["research_planner_agent_results"] = state.research_plan_results
                logger.info("Including research plan for research agent")
            
            # Convert state to dict for agent execution
            state_dict = {
                "stock_symbol": state.stock_symbol,
                "company_name": state.company_name,
                "sector": state.sector,
                "context": context
            }
            
            # Execute research agent using partial state update method
            partial_update = await self.research_agent.execute_task_partial(state_dict)
            
            # Extract results and update state
            research_results = partial_update.get("results", {})
            agent_errors = partial_update.get("errors", [])
            
            # Return state changes only (do not mutate input state)
            # Note: Return only new errors; reducer will merge with existing errors
            logger.info(f"Research agent completed for {state.stock_symbol}")
            return {
                "research_results": research_results,
                "current_agent": "analysis_agents",
                "errors": agent_errors if agent_errors else []
            }
            
        except Exception as e:
            logger.error(f"Research agent failed: {e}")
            return {
                "errors": [f"Research agent failed: {str(e)}"],
                "current_agent": "error"
            }
    
    async def _financial_analysis_agent_node(self, state: MultiAgentState) -> dict:
        """Execute financial analysis agent."""
        # Check for errors before executing
        if state.current_agent == "error" or (state.errors and len(state.errors) > 0):
            logger.warning(f"Skipping financial analysis agent due to errors in state for {state.stock_symbol}")
            # Return empty results to indicate this agent was skipped (but still "completed")
            return {"financial_analysis_results": {}}
        
        try:
            logger.info(f"Executing financial analysis agent for {state.stock_symbol}")
            
            # Prepare context with research results
            context = {
                "research_agent_results": state.research_results or {}
            }
            
            # Convert state to dict for agent execution
            state_dict = {
                "stock_symbol": state.stock_symbol,
                "company_name": state.company_name,
                "sector": state.sector,
                "context": context
            }
            
            # Execute financial analysis agent
            partial_update = await self.financial_analysis_agent.execute_task_partial(state_dict)
            
            # Extract results
            financial_results = partial_update.get("results", {})
            agent_errors = partial_update.get("errors", [])
            
            # Return state changes
            # Note: Return only new errors; reducer will merge with existing errors
            logger.info(f"Financial analysis agent completed for {state.stock_symbol}")
            return {
                "financial_analysis_results": financial_results,
                "errors": agent_errors if agent_errors else []
            }
            
        except Exception as e:
            logger.error(f"Financial analysis agent failed: {e}")
            return {
                "errors": [f"Financial analysis agent failed: {str(e)}"]
            }
    
    async def _management_analysis_agent_node(self, state: MultiAgentState) -> dict:
        """Execute management analysis agent."""
        # Check for errors before executing
        if state.current_agent == "error" or (state.errors and len(state.errors) > 0):
            logger.warning(f"Skipping management analysis agent due to errors in state for {state.stock_symbol}")
            # Return empty results to indicate this agent was skipped (but still "completed")
            return {"management_analysis_results": {}}
        
        try:
            logger.info(f"Executing management analysis agent for {state.stock_symbol}")
            
            # Prepare context with research results
            context = {
                "research_agent_results": state.research_results or {}
            }
            
            # Convert state to dict for agent execution
            state_dict = {
                "stock_symbol": state.stock_symbol,
                "company_name": state.company_name,
                "sector": state.sector,
                "context": context
            }
            
            # Execute management analysis agent
            partial_update = await self.management_analysis_agent.execute_task_partial(state_dict)
            
            # Extract results
            management_results = partial_update.get("results", {})
            agent_errors = partial_update.get("errors", [])
            
            # Return state changes
            # Note: Return only new errors; reducer will merge with existing errors
            logger.info(f"Management analysis agent completed for {state.stock_symbol}")
            return {
                "management_analysis_results": management_results,
                "errors": agent_errors if agent_errors else []
            }
            
        except Exception as e:
            logger.error(f"Management analysis agent failed: {e}")
            return {
                "errors": [f"Management analysis agent failed: {str(e)}"]
            }
    
    async def _technical_analysis_agent_node(self, state: MultiAgentState) -> dict:
        """Execute technical analysis agent."""
        # Check for errors before executing
        if state.current_agent == "error" or (state.errors and len(state.errors) > 0):
            logger.warning(f"Skipping technical analysis agent due to errors in state for {state.stock_symbol}")
            # Return empty results to indicate this agent was skipped (but still "completed")
            return {"technical_analysis_results": {}}
        
        try:
            logger.info(f"Executing technical analysis agent for {state.stock_symbol}")
            
            # Prepare context with research results
            context = {
                "research_agent_results": state.research_results or {}
            }
            
            # Convert state to dict for agent execution
            state_dict = {
                "stock_symbol": state.stock_symbol,
                "company_name": state.company_name,
                "sector": state.sector,
                "context": context
            }
            
            # Execute technical analysis agent
            partial_update = await self.technical_analysis_agent.execute_task_partial(state_dict)
            
            # Extract results
            technical_results = partial_update.get("results", {})
            agent_errors = partial_update.get("errors", [])
            
            # Return state changes
            # Note: Return only new errors; reducer will merge with existing errors
            logger.info(f"Technical analysis agent completed for {state.stock_symbol}")
            return {
                "technical_analysis_results": technical_results,
                "errors": agent_errors if agent_errors else []
            }
            
        except Exception as e:
            logger.error(f"Technical analysis agent failed: {e}")
            return {
                "errors": [f"Technical analysis agent failed: {str(e)}"]
            }
    
    async def _valuation_analysis_agent_node(self, state: MultiAgentState) -> dict:
        """Execute valuation analysis agent."""
        # Check for errors before executing
        if state.current_agent == "error" or (state.errors and len(state.errors) > 0):
            logger.warning(f"Skipping valuation analysis agent due to errors in state for {state.stock_symbol}")
            # Return empty results to indicate this agent was skipped (but still "completed")
            return {"valuation_analysis_results": {}}
        
        try:
            logger.info(f"Executing valuation analysis agent for {state.stock_symbol}")
            
            # Prepare context with research results
            context = {
                "research_agent_results": state.research_results or {}
            }
            
            # Convert state to dict for agent execution
            state_dict = {
                "stock_symbol": state.stock_symbol,
                "company_name": state.company_name,
                "sector": state.sector,
                "context": context
            }
            
            # Execute valuation analysis agent
            partial_update = await self.valuation_analysis_agent.execute_task_partial(state_dict)
            
            # Extract results
            valuation_results = partial_update.get("results", {})
            agent_errors = partial_update.get("errors", [])
            
            # Return state changes
            # Note: Return only new errors; reducer will merge with existing errors
            logger.info(f"Valuation analysis agent completed for {state.stock_symbol}")
            return {
                "valuation_analysis_results": valuation_results,
                "errors": agent_errors if agent_errors else []
            }
            
        except Exception as e:
            logger.error(f"Valuation analysis agent failed: {e}")
            return {
                "errors": [f"Valuation analysis agent failed: {str(e)}"]
            }
    
    async def _report_agent_node(self, state: MultiAgentState) -> dict:
        """
        Execute report agent.
        
        This node executes only after ALL 4 analysis agents have completed.
        LangGraph automatically waits for all incoming edges before executing a node.
        """
        try:
            logger.info(f"Executing report agent for {state.stock_symbol}")
            
            # Verify that all analysis agents have completed
            # Check which analysis results are available
            analysis_completion = {
                "financial": state.financial_analysis_results is not None,
                "management": state.management_analysis_results is not None,
                "technical": state.technical_analysis_results is not None,
                "valuation": state.valuation_analysis_results is not None
            }
            
            completed_count = sum(1 for v in analysis_completion.values() if v)
            logger.info(f"Report agent received results from {completed_count}/4 analysis agents")
            
            # Combine all analysis results into analysis_results for backward compatibility
            # Include empty dicts for any missing results
            combined_analysis_results = {
                "financial_analysis": state.financial_analysis_results or {},
                "management_analysis": state.management_analysis_results or {},
                "technical_analysis": state.technical_analysis_results or {},
                "valuation_analysis": state.valuation_analysis_results or {}
            }
            
            # Prepare context with research and all analysis results
            context = {
                "research_agent_results": state.research_results or {},
                "analysis_agent_results": combined_analysis_results,
                "financial_analysis_agent_results": state.financial_analysis_results or {},
                "management_analysis_agent_results": state.management_analysis_results or {},
                "technical_analysis_agent_results": state.technical_analysis_results or {},
                "valuation_analysis_agent_results": state.valuation_analysis_results or {}
            }
            
            # Convert state to dict for agent execution
            state_dict = {
                "stock_symbol": state.stock_symbol,
                "company_name": state.company_name,
                "sector": state.sector,
                "context": context
            }
            
            # Execute report agent using partial state update method
            partial_update = await self.report_agent.execute_task_partial(state_dict)
            
            # Extract results and update state
            report_results = partial_update.get("results", {})
            agent_errors = partial_update.get("errors", [])
            
            # Return state changes only (do not mutate input state)
            # Note: Return only new errors; reducer will merge with existing errors
            logger.info(f"Report agent completed for {state.stock_symbol}")
            return {
                "analysis_results": combined_analysis_results,  # Update combined results in state
                "report_results": report_results,
                "final_report": report_results.get("final_report", ""),
                "pdf_path": report_results.get("pdf_path", ""),
                "current_agent": "completed",
                "end_time": datetime.now(),
                "errors": agent_errors if agent_errors else []
            }
            
        except Exception as e:
            logger.error(f"Report agent failed: {e}")
            return {
                "errors": [f"Report agent failed: {str(e)}"],
                "current_agent": "error",
                "end_time": datetime.now()
            }
    
    def _should_continue_after_planning(self, state: MultiAgentState) -> str:
        """Determine if workflow should continue after planning."""
        if state.current_agent == "error" or len(state.errors) > 0:
            return "error"
        return "continue"
    
    def _should_continue_after_research(self, state: MultiAgentState) -> str:
        """Determine if workflow should continue after research."""
        if state.current_agent == "error" or len(state.errors) > 0:
            return "error"
        return "continue"
    
    def _should_continue_after_analysis(self, state: MultiAgentState) -> str:
        """Determine if workflow should continue after analysis."""
        if state.current_agent == "error" or len(state.errors) > 0:
            return "error"
        return "continue"
    
    def get_workflow_status(self, stock_symbol: str) -> Dict[str, Any]:
        """
        Get the current status of a workflow.
        
        Args:
            stock_symbol: Stock symbol to check status for
            
        Returns:
            Dictionary containing workflow status
        """
        return {
            "stock_symbol": stock_symbol,
            "workflow_status": "not_tracked",
            "message": "Workflow status tracking not implemented in this version"
        }
    
    def export_workflow_data(self, stock_symbol: str) -> Dict[str, Any]:
        """
        Export workflow data for a stock symbol.
        
        Args:
            stock_symbol: Stock symbol to export data for
            
        Returns:
            Dictionary containing workflow data
        """
        return {
            "stock_symbol": stock_symbol,
            "export_timestamp": datetime.now().isoformat(),
            "message": "Workflow data export not implemented in this version"
        }
