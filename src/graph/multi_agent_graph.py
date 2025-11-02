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
    from ..agents.ai_research_agent import AIResearchAgent
    from ..agents.ai_analysis_agent import AIAnalysisAgent
    from ..agents.ai_report_agent import AIReportAgent
    from ..config import Config
except ImportError:
    # Fall back to absolute imports (when run as script)
    from agents import (
        ResearchAgent, ResearchPlannerAgent, ReportAgent,
        FinancialAnalysisAgent, ManagementAnalysisAgent,
        TechnicalAnalysisAgent, ValuationAnalysisAgent
    )
    from agents.ai_research_agent import AIResearchAgent
    from agents.ai_analysis_agent import AIAnalysisAgent
    from agents.ai_report_agent import AIReportAgent
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
    
    def __init__(self, openai_api_key: str, use_ai_research: bool = True, use_ai_analysis: bool = True):
        """
        Initialize the multi-agent orchestrator.
        
        Args:
            openai_api_key: OpenAI API key for LLM calls
            use_ai_research: If True, use AIResearchAgent (iterative LLM-based research)
                            If False, use ResearchPlannerAgent + ResearchAgent (structured workflow)
            use_ai_analysis: If True, use AIAnalysisAgent (iterative LLM-based analysis)
                            If False, use separate Financial, Management, Technical, Valuation agents
        """
        self.openai_api_key = openai_api_key
        self.use_ai_research = use_ai_research
        self.use_ai_analysis = use_ai_analysis
        
        # Initialize agents conditionally
        if use_ai_research:
            # AI-based iterative research agent
            self.ai_research_agent = AIResearchAgent("ai_research_agent", openai_api_key)
            self.research_planner_agent = None
            self.research_agent = None
        else:
            # Traditional structured workflow agents
            self.research_planner_agent = ResearchPlannerAgent("research_planner_agent", openai_api_key)
            self.research_agent = ResearchAgent("research_agent", openai_api_key)
            self.ai_research_agent = None
        
        # Initialize analysis agents conditionally
        if use_ai_analysis:
            # AI-based iterative analysis agent (replaces all 4 analysis agents)
            self.ai_analysis_agent = AIAnalysisAgent("ai_analysis_agent", openai_api_key)
            self.financial_analysis_agent = None
            self.management_analysis_agent = None
            self.technical_analysis_agent = None
            self.valuation_analysis_agent = None
        else:
            # Traditional separate analysis agents
            self.financial_analysis_agent = FinancialAnalysisAgent("financial_analysis_agent", openai_api_key)
            self.management_analysis_agent = ManagementAnalysisAgent("management_analysis_agent", openai_api_key)
            self.technical_analysis_agent = TechnicalAnalysisAgent("technical_analysis_agent", openai_api_key)
            self.valuation_analysis_agent = ValuationAnalysisAgent("valuation_analysis_agent", openai_api_key)
            self.ai_analysis_agent = None
        
        # Initialize report agent conditionally
        if use_ai_research or use_ai_analysis:
            # Use AI Report Agent when using AI for research or analysis
            self.report_agent = AIReportAgent("ai_report_agent", openai_api_key)
        else:
            # Use traditional Report Agent for traditional workflow
            self.report_agent = ReportAgent("report_agent", openai_api_key)
        
        # Build the graph
        self.graph = self._build_graph()
        
    def _build_graph(self) -> CompiledStateGraph[Any, Any, Any, Any]:
        """Build the LangGraph workflow with parallel analysis agents."""
        # Create the state graph
        workflow = StateGraph(MultiAgentState)
        
        logger.info(f"Building graph with use_ai_research={self.use_ai_research}, use_ai_analysis={self.use_ai_analysis}")
        
        # Phase 1: Node Addition - Add all nodes based on flags
        self._add_nodes(workflow)
        
        # Phase 2: Edge Addition - Add all edges based on flags (nodes assumed to exist)
        self._add_edges(workflow)
        
        # Compile the graph
        return workflow.compile()
    
    def _add_nodes(self, workflow: StateGraph) -> None:
        """Phase 1: Add all nodes to the graph based on flags."""
        # Report node is always needed
        report_node = cast(Runnable[MultiAgentState, MultiAgentState],
                           RunnableLambda(self._report_agent_node))
        workflow.add_node("report_agent", report_node)
        
        # Add research nodes based on flag
        if self.use_ai_research:
            logger.info("Adding AI Research Agent node")
            ai_research_node = cast(Runnable[MultiAgentState, MultiAgentState],
                                    RunnableLambda(self._ai_research_agent_node))
            workflow.add_node("ai_research_agent", ai_research_node)
        else:
            logger.info("Adding Research Planner and Research Agent nodes")
            planner_node = cast(Runnable[MultiAgentState, MultiAgentState],
                                RunnableLambda(self._research_planner_agent_node))
            research_node = cast(Runnable[MultiAgentState, MultiAgentState],
                                RunnableLambda(self._research_agent_node))
            workflow.add_node("research_planner_agent", planner_node)
            workflow.add_node("research_agent", research_node)
        
        # Add analysis nodes based on flag
        if self.use_ai_analysis:
            logger.info("Adding AI Analysis Agent node")
            ai_analysis_node = cast(Runnable[MultiAgentState, MultiAgentState],
                                    RunnableLambda(self._ai_analysis_agent_node))
            workflow.add_node("ai_analysis_agent", ai_analysis_node)
        else:
            logger.info("Adding separate analysis agent nodes")
            financial_analysis_node = cast(Runnable[MultiAgentState, MultiAgentState],
                                          RunnableLambda(self._financial_analysis_agent_node))
            management_analysis_node = cast(Runnable[MultiAgentState, MultiAgentState],
                                            RunnableLambda(self._management_analysis_agent_node))
            technical_analysis_node = cast(Runnable[MultiAgentState, MultiAgentState],
                                           RunnableLambda(self._technical_analysis_agent_node))
            valuation_analysis_node = cast(Runnable[MultiAgentState, MultiAgentState],
                                           RunnableLambda(self._valuation_analysis_agent_node))
            workflow.add_node("financial_analysis_agent", financial_analysis_node)
            workflow.add_node("management_analysis_agent", management_analysis_node)
            workflow.add_node("technical_analysis_agent", technical_analysis_node)
            workflow.add_node("valuation_analysis_agent", valuation_analysis_node)
    
    def _add_edges(self, workflow: StateGraph) -> None:
        """Phase 2: Add all edges to the graph based on flags (nodes assumed to exist)."""
        # Determine entry point and research -> analysis edges
        if self.use_ai_research:
            # AI Research Agent is entry point
            workflow.set_entry_point("ai_research_agent")
            
            if self.use_ai_analysis:
                # AI Research -> AI Analysis
                workflow.add_edge("ai_research_agent", "ai_analysis_agent")
            else:
                # AI Research -> All Analysis Agents (parallel)
                workflow.add_edge("ai_research_agent", "financial_analysis_agent")
                workflow.add_edge("ai_research_agent", "management_analysis_agent")
                workflow.add_edge("ai_research_agent", "technical_analysis_agent")
                workflow.add_edge("ai_research_agent", "valuation_analysis_agent")
        else:
            # Research Planner is entry point
            workflow.set_entry_point("research_planner_agent")
            
            # Research Planner -> Research Agent (conditional)
            workflow.add_conditional_edges(
                "research_planner_agent",
                self._should_continue_after_planning,
                {
                    "continue": "research_agent",
                    "error": END
                }
            )
            
            if self.use_ai_analysis:
                # Research -> AI Analysis
                workflow.add_edge("research_agent", "ai_analysis_agent")
            else:
                # Research -> All Analysis Agents (parallel)
                workflow.add_edge("research_agent", "financial_analysis_agent")
                workflow.add_edge("research_agent", "management_analysis_agent")
                workflow.add_edge("research_agent", "technical_analysis_agent")
                workflow.add_edge("research_agent", "valuation_analysis_agent")
        
        # Analysis -> Report edges
        if self.use_ai_analysis:
            # AI Analysis -> Report
            workflow.add_edge("ai_analysis_agent", "report_agent")
        else:
            # All Analysis Agents -> Report (parallel, LangGraph waits for all)
            workflow.add_edge("financial_analysis_agent", "report_agent")
            workflow.add_edge("management_analysis_agent", "report_agent")
            workflow.add_edge("technical_analysis_agent", "report_agent")
            workflow.add_edge("valuation_analysis_agent", "report_agent")
        
        # Report -> END
        workflow.add_edge("report_agent", END)
    
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
            
            # Create initial state with correct entry agent
            entry_agent = "ai_research_agent" if self.use_ai_research else "research_planner_agent"
            initial_state = MultiAgentState(
                stock_symbol=stock_symbol,
                company_name=company_name,
                sector=sector,
                current_agent=entry_agent,
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
    
    async def _ai_research_agent_node(self, state: MultiAgentState) -> dict:
        """Execute AI research agent (iterative LLM-based research)."""
        try:
            logger.info(f"Executing AI research agent for {state.stock_symbol}")
            
            # Prepare context
            context = {}
            
            # Convert state to dict for agent execution
            state_dict = {
                "stock_symbol": state.stock_symbol,
                "company_name": state.company_name,
                "sector": state.sector,
                "context": context
            }
            
            # Execute AI research agent using partial state update method
            partial_update = await self.ai_research_agent.execute_task_partial(state_dict)
            
            # Extract results and update state
            research_results = partial_update.get("results", {})
            agent_errors = partial_update.get("errors", [])
            
            # Return state changes only
            logger.info(f"AI research agent completed for {state.stock_symbol}")
            return {
                "research_results": research_results,
                "current_agent": "analysis_agents",
                "errors": agent_errors if agent_errors else []
            }
            
        except Exception as e:
            logger.error(f"AI research agent failed: {e}")
            return {
                "errors": [f"AI research agent failed: {str(e)}"],
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
    
    async def _ai_analysis_agent_node(self, state: MultiAgentState) -> dict:
        """Execute AI analysis agent (iterative LLM-based comprehensive analysis)."""
        try:
            logger.info(f"Executing AI analysis agent for {state.stock_symbol}")
            
            # Verify AI analysis agent is available
            if not hasattr(self, 'ai_analysis_agent') or self.ai_analysis_agent is None:
                error_msg = "AI analysis agent not initialized. Check use_ai_analysis flag."
                logger.error(error_msg)
                return {
                    "errors": [error_msg],
                    "current_agent": "error"
                }
            
            # Prepare context with research results
            context = {}
            
            # Include research results in context
            # Support both traditional research agent and AI research agent results
            if state.research_results:
                context["research_agent_results"] = state.research_results
                logger.info("Including research results for AI analysis agent")
            
            # AI research agent results are also stored in research_results, so we're covered
            
            # Convert state to dict for agent execution
            state_dict = {
                "stock_symbol": state.stock_symbol,
                "company_name": state.company_name,
                "sector": state.sector,
                "context": context
            }
            
            # Execute AI analysis agent using partial state update method
            partial_update = await self.ai_analysis_agent.execute_task_partial(state_dict)
            
            # Extract results and update state
            # AI analysis agent returns results structured with all analysis types
            analysis_results = partial_update.get("results", {})
            agent_errors = partial_update.get("errors", [])
            
            # Map AI analysis results to state format
            # AI analysis agent returns: financial_analysis, management_analysis, technical_analysis, valuation_analysis
            return {
                "financial_analysis_results": analysis_results.get("financial_analysis", {}),
                "management_analysis_results": analysis_results.get("management_analysis", {}),
                "technical_analysis_results": analysis_results.get("technical_analysis", {}),
                "valuation_analysis_results": analysis_results.get("valuation_analysis", {}),
                "analysis_results": analysis_results,  # Keep full results for report agent
                "current_agent": "report_agent",
                "errors": agent_errors if agent_errors else []
            }
            
        except Exception as e:
            logger.error(f"AI analysis agent failed: {e}")
            return {
                "errors": [f"AI analysis agent failed: {str(e)}"],
                "current_agent": "error"
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
