"""
LangGraph orchestration flow for Stock Report Generator.
Defines the agent workflow and coordination logic.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import asyncio
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

try:
    # Try relative imports first (when run as module)
    from .context_manager_mcp import MCPContextManager, ContextType
    from ..agents.sector_researcher import SectorResearcherAgent
    from ..agents.stock_researcher import StockResearcherAgent
    from ..agents.management_analysis import ManagementAnalysisAgent
    from ..agents.swot_analysis import SWOTAnalysisAgent
    from ..agents.report_reviewer import ReportReviewerAgent
    from ..tools.web_search_tool import WebSearchTool
    from ..tools.stock_data_tool import StockDataTool
    from ..tools.report_fetcher_tool import ReportFetcherTool
    from ..tools.pdf_parser_tool import PDFParserTool
    from ..tools.summarizer_tool import SummarizerTool
    from ..tools.report_formatter_tool import ReportFormatterTool
except ImportError:
    # Fall back to absolute imports (when run as script)
    from context_manager_mcp import MCPContextManager, ContextType
    from agents.sector_researcher import SectorResearcherAgent
    from agents.stock_researcher import StockResearcherAgent
    from agents.management_analysis import ManagementAnalysisAgent
    from agents.swot_analysis import SWOTAnalysisAgent
    from agents.report_reviewer import ReportReviewerAgent
    from tools.web_search_tool import WebSearchTool
    from tools.stock_data_tool import StockDataTool
    from tools.report_fetcher_tool import ReportFetcherTool
    from tools.pdf_parser_tool import PDFParserTool
    from tools.summarizer_tool import SummarizerTool
    from tools.report_formatter_tool import ReportFormatterTool

logger = logging.getLogger(__name__)

@dataclass
class WorkflowState:
    """Represents the state of the workflow."""
    stock_symbol: str
    company_name: str
    sector: str
    current_step: str
    sector_analysis: Optional[Dict[str, Any]] = None
    stock_analysis: Optional[Dict[str, Any]] = None
    management_analysis: Optional[Dict[str, Any]] = None
    swot_analysis: Optional[Dict[str, Any]] = None
    final_report: Optional[Dict[str, Any]] = None
    errors: List[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []

class StockReportGraph:
    """
    LangGraph orchestration for Stock Report Generator.
    
    Manages the workflow of multiple agents to generate comprehensive
    equity research reports.
    """
    
    def __init__(
        self,
        mcp_context: MCPContextManager,
        web_search_tool: WebSearchTool,
        stock_data_tool: StockDataTool,
        report_fetcher_tool: ReportFetcherTool,
        pdf_parser_tool: PDFParserTool,
        summarizer_tool: SummarizerTool,
        report_formatter_tool: ReportFormatterTool,
        openai_api_key: str
    ):
        """
        Initialize the Stock Report Graph.
        
        Args:
            mcp_context: MCP context manager
            web_search_tool: Web search tool
            stock_data_tool: Stock data tool
            report_fetcher_tool: Report fetcher tool
            pdf_parser_tool: PDF parser tool
            summarizer_tool: Summarizer tool
            report_formatter_tool: Report formatter tool
            openai_api_key: OpenAI API key
        """
        self.mcp_context = mcp_context
        self.openai_api_key = openai_api_key
        
        # Initialize agents
        self.sector_researcher = SectorResearcherAgent(
            agent_id="sector_researcher",
            mcp_context=mcp_context,
            web_search_tool=web_search_tool,
            stock_data_tool=stock_data_tool,
            openai_api_key=openai_api_key
        )
        
        self.stock_researcher = StockResearcherAgent(
            agent_id="stock_researcher",
            mcp_context=mcp_context,
            stock_data_tool=stock_data_tool,
            openai_api_key=openai_api_key
        )
        
        self.management_analysis = ManagementAnalysisAgent(
            agent_id="management_analysis",
            mcp_context=mcp_context,
            report_fetcher_tool=report_fetcher_tool,
            pdf_parser_tool=pdf_parser_tool,
            summarizer_tool=summarizer_tool,
            openai_api_key=openai_api_key
        )
        
        self.swot_analysis = SWOTAnalysisAgent(
            agent_id="swot_analysis",
            mcp_context=mcp_context,
            stock_data_tool=stock_data_tool,
            web_search_tool=web_search_tool,
            openai_api_key=openai_api_key
        )
        
        self.report_reviewer = ReportReviewerAgent(
            agent_id="report_reviewer",
            mcp_context=mcp_context,
            report_formatter_tool=report_formatter_tool,
            openai_api_key=openai_api_key
        )
        
        # Build the graph
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Create the state graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes for each agent
        workflow.add_node("sector_researcher", self._sector_research_node)
        workflow.add_node("stock_researcher", self._stock_research_node)
        workflow.add_node("management_analysis", self._management_analysis_node)
        workflow.add_node("swot_analysis", self._swot_analysis_node)
        workflow.add_node("report_reviewer", self._report_reviewer_node)
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "sector_researcher",
            self._should_continue_after_sector,
            {
                "continue": "stock_researcher",
                "error": END
            }
        )
        
        workflow.add_conditional_edges(
            "stock_researcher",
            self._should_continue_after_stock,
            {
                "continue": "management_analysis",
                "error": END
            }
        )
        
        workflow.add_conditional_edges(
            "management_analysis",
            self._should_continue_after_management,
            {
                "continue": "swot_analysis",
                "error": END
            }
        )
        
        workflow.add_conditional_edges(
            "swot_analysis",
            self._should_continue_after_swot,
            {
                "continue": "report_reviewer",
                "error": END
            }
        )
        
        workflow.add_edge("report_reviewer", END)
        
        # Set entry point
        workflow.set_entry_point("sector_researcher")
        
        # Compile the graph
        return workflow.compile()
        
    async def run_workflow(
        self,
        stock_symbol: str,
        company_name: str,
        sector: str
    ) -> Dict[str, Any]:
        """
        Run the complete workflow for generating a stock report.
        
        Args:
            stock_symbol: NSE stock symbol
            company_name: Full company name
            sector: Sector name
            
        Returns:
            Dictionary containing the final report and workflow results
        """
        try:
            logger.info(f"Starting workflow for {stock_symbol}")
            
            # Create initial state
            initial_state = WorkflowState(
                stock_symbol=stock_symbol,
                company_name=company_name,
                sector=sector,
                current_step="sector_researcher",
                start_time=datetime.now()
            )
            
            # Start MCP session
            session_id = f"stock_report_{stock_symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.mcp_context.start_session(session_id)
            
            # Run the workflow
            final_state = await self.graph.ainvoke(initial_state)
            
            # Complete the workflow - handle both dict and WorkflowState objects
            if hasattr(final_state, 'end_time'):
                final_state.end_time = datetime.now()
            else:
                # If it's a dict, create a proper WorkflowState object
                if isinstance(final_state, dict):
                    final_state = WorkflowState(
                        stock_symbol=final_state.get('stock_symbol', stock_symbol),
                        company_name=final_state.get('company_name', company_name),
                        sector=final_state.get('sector', sector),
                        current_step=final_state.get('current_step', 'completed'),
                        sector_analysis=final_state.get('sector_analysis'),
                        stock_analysis=final_state.get('stock_analysis'),
                        management_analysis=final_state.get('management_analysis'),
                        final_report=final_state.get('final_report'),
                        errors=final_state.get('errors', []),
                        start_time=final_state.get('start_time', initial_state.start_time),
                        end_time=datetime.now()
                    )
                else:
                    # Try to set end_time if it's an object with __dict__
                    try:
                        final_state.end_time = datetime.now()
                    except:
                        # Create a new WorkflowState if we can't modify the existing object
                        final_state = WorkflowState(
                            stock_symbol=stock_symbol,
                            company_name=company_name,
                            sector=sector,
                            current_step='completed',
                            start_time=initial_state.start_time,
                            end_time=datetime.now()
                        )
            
            # Get final report from MCP context
            final_report_data = self.mcp_context.get_latest_context(ContextType.FINAL_REPORT)
            
            # Prepare results
            results = {
                "stock_symbol": stock_symbol,
                "company_name": company_name,
                "sector": sector,
                "workflow_status": "completed",
                "start_time": final_state.start_time.isoformat() if final_state.start_time else None,
                "end_time": final_state.end_time.isoformat() if final_state.end_time else None,
                "duration_seconds": (final_state.end_time - final_state.start_time).total_seconds() if final_state.start_time and final_state.end_time else None,
                "errors": final_state.errors,
                "final_report": final_report_data["data"] if final_report_data else None,
                "context_summary": self.mcp_context.get_context_summary()
            }
            
            logger.info(f"Completed workflow for {stock_symbol}")
            return results
            
        except Exception as e:
            logger.error(f"Error running workflow: {e}")
            return {
                "stock_symbol": stock_symbol,
                "company_name": company_name,
                "sector": sector,
                "workflow_status": "failed",
                "error": str(e),
                "errors": [str(e)]
            }
            
    def _sector_research_node(self, state: WorkflowState) -> WorkflowState:
        """Execute sector research analysis."""
        try:
            logger.info(f"Executing sector research for {state.stock_symbol}")
            
            # Perform sector analysis
            sector_analysis = self.sector_researcher.analyze_sector(
                stock_symbol=state.stock_symbol,
                company_name=state.company_name,
                sector=state.sector
            )
            
            # Update state
            state.sector_analysis = {
                "sector_name": sector_analysis.sector_name,
                "summary": sector_analysis.summary,
                "trends": sector_analysis.trends,
                "peer_comparison": sector_analysis.peer_comparison,
                "regulatory_environment": sector_analysis.regulatory_environment,
                "outlook": sector_analysis.outlook,
                "risks": sector_analysis.risks,
                "opportunities": sector_analysis.opportunities,
                "confidence_score": sector_analysis.confidence_score
            }
            
            state.current_step = "stock_researcher"
            logger.info(f"Completed sector research for {state.stock_symbol}")
            
        except Exception as e:
            logger.error(f"Error in sector research: {e}")
            state.errors.append(f"Sector research failed: {str(e)}")
            state.current_step = "error"
            
        return state
        
    def _stock_research_node(self, state: WorkflowState) -> WorkflowState:
        """Execute stock research analysis."""
        try:
            logger.info(f"Executing stock research for {state.stock_symbol}")
            
            # Perform stock analysis
            stock_analysis = self.stock_researcher.analyze_stock(
                stock_symbol=state.stock_symbol,
                company_name=state.company_name
            )
            
            # Update state
            state.stock_analysis = {
                "symbol": stock_analysis.symbol,
                "current_price": stock_analysis.current_price,
                "market_cap": stock_analysis.market_cap,
                "financial_metrics": stock_analysis.financial_metrics,
                "technical_analysis": stock_analysis.technical_analysis,
                "valuation_metrics": stock_analysis.valuation_metrics,
                "performance_summary": stock_analysis.performance_summary,
                "investment_rating": stock_analysis.investment_rating,
                "target_price": stock_analysis.target_price,
                "confidence_score": stock_analysis.confidence_score
            }
            
            state.current_step = "management_analysis"
            logger.info(f"Completed stock research for {state.stock_symbol}")
            
        except Exception as e:
            logger.error(f"Error in stock research: {e}")
            state.errors.append(f"Stock research failed: {str(e)}")
            state.current_step = "error"
            
        return state
        
    def _management_analysis_node(self, state: WorkflowState) -> WorkflowState:
        """Execute management analysis."""
        try:
            logger.info(f"Executing management analysis for {state.stock_symbol}")
            
            # Perform management analysis
            management_analysis = self.management_analysis.analyze_management(
                stock_symbol=state.stock_symbol,
                company_name=state.company_name
            )
            
            # Update state
            state.management_analysis = {
                "company_name": management_analysis.company_name,
                "summary": management_analysis.summary,
                "strategic_initiatives": management_analysis.strategic_initiatives,
                "growth_opportunities": management_analysis.growth_opportunities,
                "risk_factors": management_analysis.risk_factors,
                "management_outlook": management_analysis.management_outlook,
                "key_insights": management_analysis.key_insights,
                "financial_highlights": management_analysis.financial_highlights,
                "confidence_score": management_analysis.confidence_score
            }
            
            state.current_step = "report_reviewer"
            logger.info(f"Completed management analysis for {state.stock_symbol}")
            
        except Exception as e:
            logger.error(f"Error in management analysis: {e}")
            state.errors.append(f"Management analysis failed: {str(e)}")
            state.current_step = "error"
            
        return state
        
    def _swot_analysis_node(self, state: WorkflowState) -> WorkflowState:
        """Execute SWOT analysis."""
        try:
            logger.info(f"Executing SWOT analysis for {state.stock_symbol}")
            
            # Perform SWOT analysis
            swot_analysis = self.swot_analysis.analyze_swot(
                stock_symbol=state.stock_symbol,
                company_name=state.company_name,
                sector=state.sector
            )
            
            # Update state
            state.swot_analysis = {
                "company_name": swot_analysis.company_name,
                "strengths": swot_analysis.strengths,
                "weaknesses": swot_analysis.weaknesses,
                "opportunities": swot_analysis.opportunities,
                "threats": swot_analysis.threats,
                "summary": swot_analysis.summary,
                "confidence_score": swot_analysis.confidence_score
            }
            
            state.current_step = "report_reviewer"
            logger.info(f"Completed SWOT analysis for {state.stock_symbol}")
            
        except Exception as e:
            logger.error(f"Error in SWOT analysis: {e}")
            state.errors.append(f"SWOT analysis failed: {str(e)}")
            state.current_step = "error"
            
        return state
        
    def _report_reviewer_node(self, state: WorkflowState) -> WorkflowState:
        """Execute final report generation."""
        try:
            logger.info(f"Executing final report generation for {state.stock_symbol}")
            
            # Create final report
            final_report = self.report_reviewer.create_final_report(
                stock_symbol=state.stock_symbol,
                company_name=state.company_name
            )
            
            # Update state
            state.final_report = {
                "stock_symbol": final_report.stock_symbol,
                "company_name": final_report.company_name,
                "report_content": final_report.report_content,
                "sections": final_report.sections,
                "data_sources": final_report.data_sources,
                "confidence_score": final_report.confidence_score,
                "consistency_issues": [issue.__dict__ for issue in final_report.consistency_issues],
                "recommendations": final_report.recommendations,
                "created_at": final_report.created_at.isoformat()
            }
            
            state.current_step = "completed"
            logger.info(f"Completed final report generation for {state.stock_symbol}")
            
        except Exception as e:
            logger.error(f"Error in final report generation: {e}")
            state.errors.append(f"Final report generation failed: {str(e)}")
            state.current_step = "error"
            
        return state
        
    def _should_continue_after_sector(self, state: WorkflowState) -> str:
        """Determine if workflow should continue after sector research."""
        if state.current_step == "error" or len(state.errors) > 0:
            return "error"
        return "continue"
        
    def _should_continue_after_stock(self, state: WorkflowState) -> str:
        """Determine if workflow should continue after stock research."""
        if state.current_step == "error" or len(state.errors) > 0:
            return "error"
        return "continue"
        
    def _should_continue_after_management(self, state: WorkflowState) -> str:
        """Determine if workflow should continue after management analysis."""
        if state.current_step == "error" or len(state.errors) > 0:
            return "error"
        return "continue"
        
    def _should_continue_after_swot(self, state: WorkflowState) -> str:
        """Determine if workflow should continue after SWOT analysis."""
        if state.current_step == "error" or len(state.errors) > 0:
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
        try:
            # Get context summary
            context_summary = self.mcp_context.get_context_summary()
            
            # Check for final report
            final_report = self.mcp_context.get_latest_context(ContextType.FINAL_REPORT)
            
            status = {
                "stock_symbol": stock_symbol,
                "workflow_status": "completed" if final_report else "in_progress",
                "context_entries": context_summary.get("total_entries", 0),
                "agent_outputs": {
                    "sector_summary": bool(self.mcp_context.get_latest_context(ContextType.SECTOR_SUMMARY)),
                    "stock_summary": bool(self.mcp_context.get_latest_context(ContextType.STOCK_SUMMARY)),
                    "management_summary": bool(self.mcp_context.get_latest_context(ContextType.MANAGEMENT_SUMMARY)),
                    "final_report": bool(final_report)
                },
                "context_summary": context_summary
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting workflow status: {e}")
            return {
                "stock_symbol": stock_symbol,
                "workflow_status": "error",
                "error": str(e)
            }
            
    def export_workflow_data(self, stock_symbol: str) -> Dict[str, Any]:
        """
        Export all workflow data for a stock symbol.
        
        Args:
            stock_symbol: Stock symbol to export data for
            
        Returns:
            Dictionary containing all workflow data
        """
        try:
            # Export MCP context
            context_data = self.mcp_context.export_context()
            
            # Get all agent outputs
            agent_outputs = {
                "sector_summary": self.mcp_context.get_latest_context(ContextType.SECTOR_SUMMARY),
                "stock_summary": self.mcp_context.get_latest_context(ContextType.STOCK_SUMMARY),
                "management_summary": self.mcp_context.get_latest_context(ContextType.MANAGEMENT_SUMMARY),
                "final_report": self.mcp_context.get_latest_context(ContextType.FINAL_REPORT)
            }
            
            export_data = {
                "stock_symbol": stock_symbol,
                "export_timestamp": datetime.now().isoformat(),
                "context_data": context_data,
                "agent_outputs": agent_outputs,
                "workflow_metadata": {
                    "graph_type": "StockReportGraph",
                    "agents": ["sector_researcher", "stock_researcher", "management_analysis", "report_reviewer"],
                    "context_types": [t.value for t in ContextType]
                }
            }
            
            logger.info(f"Exported workflow data for {stock_symbol}")
            return export_data
            
        except Exception as e:
            logger.error(f"Error exporting workflow data: {e}")
            return {
                "stock_symbol": stock_symbol,
                "export_timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
