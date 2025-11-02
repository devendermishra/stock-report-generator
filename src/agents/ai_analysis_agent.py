"""
AI Analysis Agent - True LangGraph Agent with Iterative Tool Selection.

This agent implements the true LangGraph agent pattern for comprehensive analysis:
1. LLM decides which analysis to perform and what data is needed
2. Executes tools to gather required data
3. Performs analysis on gathered data
4. Decides if more analysis or data is needed
5. Repeats until comprehensive analysis is complete

This is an alternative to the separate Financial, Management, Technical, and Valuation Analysis Agents.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import re

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, ToolMessage
except ImportError:
    ChatOpenAI = None
    HumanMessage = None
    ToolMessage = None

try:
    from .base_agent import BaseAgent, AgentState
    from ..tools.stock_data_tool import get_stock_metrics, get_company_info
    from ..tools.web_search_tool import search_company_news, search_market_trends
    from ..tools.technical_analysis_formatter import format_technical_analysis
    from ..config import Config
except ImportError:
    from agents.base_agent import BaseAgent, AgentState
    from tools.stock_data_tool import get_stock_metrics, get_company_info
    from tools.web_search_tool import search_company_news, search_market_trends
    from tools.technical_analysis_formatter import format_technical_analysis
    from config import Config

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class AIAnalysisAgent(BaseAgent):
    """
    AI Analysis Agent with iterative tool selection for comprehensive analysis.
    
    This agent uses an LLM in a loop to:
    1. Analyze what data is needed for different analysis types
    2. Decide which tools to use to gather that data
    3. Execute tools and gather results
    4. Perform financial, management, technical, and valuation analysis
    5. Decide if more data or analysis is needed
    6. Stop when comprehensive analysis is complete
    
    Implements the true LangGraph agent pattern as an alternative to
    separate Financial, Management, Technical, and Valuation Analysis Agents.
    """
    
    def __init__(self, agent_id: str, openai_api_key: str):
        """
        Initialize the AI Analysis Agent.
        
        Args:
            agent_id: Unique identifier for the agent
            openai_api_key: OpenAI API key for LLM calls
        """
        # Define available tools
        available_tools = [
            get_stock_metrics,
            get_company_info,
            search_company_news,
            search_market_trends,
        ]
        
        super().__init__(agent_id, openai_api_key, available_tools)
        
        # Initialize LLM with tool bindings
        if ChatOpenAI is None:
            raise ImportError("langchain_openai is required for AIAnalysisAgent. Install with: pip install langchain-openai")
        
        self.llm = ChatOpenAI(
            model=Config.DEFAULT_MODEL,
            temperature=0.1,
            api_key=openai_api_key
        )
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(available_tools)
        
        # Tool name to function mapping (use actual tool names from @tool decorators)
        self.tool_map = {tool.name: tool for tool in available_tools}
        
        # Initialize OpenAI client for analysis prompts
        self.openai_client = AsyncOpenAI(api_key=openai_api_key)
        
        # Maximum iterations to prevent infinite loops
        self.max_iterations = 12
    
    async def execute_task(
        self,
        stock_symbol: str,
        company_name: str,
        sector: str,
        context: Dict[str, Any]
    ) -> AgentState:
        """
        Execute comprehensive analysis using true agent pattern with iterative tool selection.
        
        Args:
            stock_symbol: NSE stock symbol
            company_name: Full company name
            sector: Sector name
            context: Current context from previous agents (research data)
            
        Returns:
            AgentState with comprehensive analysis results
        """
        start_time = datetime.now()
        
        # Initialize state
        state = AgentState(
            agent_id=self.agent_id,
            stock_symbol=stock_symbol,
            company_name=company_name,
            sector=sector,
            current_task="ai_comprehensive_analysis",
            context=context,
            results={},
            tools_used=[],
            confidence_score=0.0,
            errors=[],
            start_time=start_time
        )
        
        try:
            # Get research results from context
            research_data = context.get("research_agent_results", {}) or context.get("ai_research_agent_results", {})
            
            # Log research data structure for debugging
            self.logger.debug(f"Research data keys: {list(research_data.keys()) if research_data else 'None'}")
            if research_data:
                company_data = research_data.get("company_data", {})
                if company_data:
                    self.logger.debug(f"Company data keys: {list(company_data.keys()) if isinstance(company_data, dict) else 'Not a dict'}")
                    if isinstance(company_data, dict):
                        self.logger.debug(f"Has stock_metrics: {'stock_metrics' in company_data}")
                        self.logger.debug(f"Has company_info: {'company_info' in company_data}")
                
                gathered_data_raw = research_data.get("gathered_data", {})
                if gathered_data_raw:
                    self.logger.debug(f"Gathered data keys: {list(gathered_data_raw.keys()) if isinstance(gathered_data_raw, dict) else 'Not a dict'}")
            
            # Pre-populate gathered_data from research_data to avoid redundant tool calls
            gathered_data = self._extract_research_data_for_tools(research_data)
            
            # Log what was extracted
            self.logger.info(f"Extracted data from research: {list(gathered_data.keys())}")
            if "get_stock_metrics" in gathered_data:
                self.logger.info(f"Stock metrics available: {bool(gathered_data['get_stock_metrics'])}")
            if "get_company_info" in gathered_data:
                self.logger.info(f"Company info available: {bool(gathered_data['get_company_info'])}")
            
            # Build initial task description with available data summary
            available_data_summary = self._summarize_available_data(gathered_data)
            
            # Determine which tools should be skipped based on available data
            skip_tools = []
            if "get_stock_metrics" in gathered_data and gathered_data["get_stock_metrics"]:
                skip_tools.append("get_stock_metrics")
            if "get_company_info" in gathered_data and gathered_data["get_company_info"]:
                skip_tools.append("get_company_info")
            if "search_company_news" in gathered_data and gathered_data["search_company_news"]:
                skip_tools.append("search_company_news")
            if "search_market_trends" in gathered_data and gathered_data["search_market_trends"]:
                skip_tools.append("search_market_trends")
            
            skip_instructions = ""
            if skip_tools:
                skip_list = ", ".join(skip_tools)
                skip_instructions = (f"\n\n⚠️ CRITICAL - DO NOT CALL THESE TOOLS (data already available): {skip_list}\n"
                                     f"You MUST skip calling these tools since the data is already available above."
                                     f"Proceed directly to analysis using the available data.")
            
            task_description = f"""Perform comprehensive analysis for {company_name} ({stock_symbol}) in the {sector} sector.

Your goal is to conduct:
1. FINANCIAL ANALYSIS: Analyze stock metrics, financial ratios, and financial health
2. MANAGEMENT ANALYSIS: Analyze company management, governance, and leadership effectiveness
3. TECHNICAL ANALYSIS: Analyze price trends, technical indicators, and trading patterns
4. VALUATION ANALYSIS: Analyze valuation metrics, target price, and investment recommendation

AVAILABLE DATA FROM RESEARCH (SKIP TOOLS):
<available>
{available_data_summary}
</available>
{skip_instructions}

Available tools (only use if data is NOT available above):
- get_stock_metrics: Get current stock metrics and financial data
- get_company_info: Get company information and business details
- search_company_news: Search for company-specific news
- search_market_trends: Search for market trends and analysis

Workflow:
1. Check if needed data is already available from research (listed above)
2. ONLY call tools if data is MISSING and you cannot proceed without it
3. If data is available, SKIP the tool call and use the available data directly
4. Perform each analysis type using available data
5. Once all analyses are complete, respond with: "Analysis complete. I have completed comprehensive financial, management, technical, and valuation analysis."

CRITICAL RULES:
- DO NOT call get_stock_metrics if it appears in the AVAILABLE DATA above - SKIP IT
- DO NOT call get_company_info if it appears in the AVAILABLE DATA above - SKIP IT
- DO NOT call search_company_news if it appears in the AVAILABLE DATA above - SKIP IT
- DO NOT call search_market_trends if it appears in the AVAILABLE DATA above - SKIP IT
- If a tool is listed as "DO NOT CALL", you MUST skip it and proceed with analysis using available data
- Reuse available data and proceed directly to analysis"""
            
            # Execute iterative agent loop (gathered_data is pre-populated)
            agent_results = await self._agent_loop(task_description, stock_symbol, company_name, sector, research_data, gathered_data)
            
            # Process results - structure them similar to individual analysis agents
            # Merge pre-populated data with any additional data gathered
            final_gathered_data = gathered_data.copy()
            final_gathered_data.update(agent_results.get("gathered_data", {}))
            
            # Perform analysis on gathered data
            analysis_results = await self._perform_comprehensive_analysis(
                stock_symbol, company_name, sector, final_gathered_data, research_data
            )
            
            # Combine gathered data with analysis results
            state.results = {
                "financial_analysis": analysis_results.get("financial", {}),
                "management_analysis": analysis_results.get("management", {}),
                "technical_analysis": analysis_results.get("technical", {}),
                "valuation_analysis": analysis_results.get("valuation", {}),
                "gathered_data": final_gathered_data,
                "ai_iterations": agent_results.get("iterations", []),
                "final_summary": agent_results.get("final_summary", "")
            }
            
            state.tools_used = agent_results.get("tools_executed", [])
            state.confidence_score = agent_results.get("confidence", 0.8)
            
            # Add any errors
            if agent_results.get("errors"):
                state.errors.extend(agent_results["errors"])
            
            state.end_time = datetime.now()
            
            self.logger.info(
                f"AI Analysis Agent completed analysis for {stock_symbol} using {len(state.tools_used)} tools "
                f"in {len(agent_results.get('iterations', []))} iterations"
            )
            
            return state
            
        except Exception as e:
            self.logger.error(f"AI Analysis Agent execution failed: {e}")
            state.errors.append(f"Agent execution failed: {str(e)}")
            state.confidence_score = 0.0
            state.end_time = datetime.now()
            return state
    
    async def _agent_loop(
        self,
        task_description: str,
        stock_symbol: str,
        company_name: str,
        sector: str,
        research_data: Dict[str, Any],
        pre_populated_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute the iterative agent loop.
        
        Args:
            task_description: Task description for the agent
            stock_symbol: Stock symbol
            company_name: Company name
            sector: Sector name
            research_data: Data from research agent
            pre_populated_data: Data already available from research (to avoid redundant tool calls)
            
        Returns:
            Dictionary with gathered data, tools executed, and confidence
        """
        if HumanMessage is None or ToolMessage is None:
            raise ImportError("langchain_core.messages is required")
        
        messages = [HumanMessage(content=task_description)]
        gathered_data = pre_populated_data.copy()  # Start with pre-populated data
        tools_executed = []
        iterations = []
        errors = []
        
        # Add initial message about available data if any
        if pre_populated_data:
            available_tools = list(pre_populated_data.keys())
            initial_info = f"The following data is already available and ready for analysis:\n"
            for tool_name in available_tools:
                initial_info += f"- {tool_name}: Data available\n"
            messages.append(HumanMessage(content=initial_info))
        
        for iteration in range(self.max_iterations):
            try:
                # Step 1: LLM decides next action
                response = await self.llm_with_tools.ainvoke(messages)
                messages.append(response)
                
                # Check if LLM wants to finish
                if not response.tool_calls:
                    # No more tool calls - agent thinks it's done
                    final_summary = response.content or "Analysis completed."
                    self.logger.info(f"Agent decided to finish after {iteration + 1} iterations")
                    
                    return {
                        "gathered_data": gathered_data,
                        "tools_executed": tools_executed,
                        "confidence": min(1.0, 0.5 + (len(tools_executed) * 0.05)),
                        "iterations": iterations,
                        "final_summary": final_summary,
                        "errors": errors
                    }
                
                # Step 2: Execute tool calls
                tool_results = []
                for tool_call in response.tool_calls:
                    # Extract tool call information
                    if isinstance(tool_call, dict):
                        tool_name = tool_call.get("name", "")
                        tool_args = tool_call.get("args", {})
                        tool_call_id = tool_call.get("id", f"call_{iteration}_{len(tool_results)}")
                    else:
                        tool_name = getattr(tool_call, "name", "")
                        tool_args = getattr(tool_call, "args", {})
                        tool_call_id = getattr(tool_call, "id", f"call_{iteration}_{len(tool_results)}")
                    
                    # Update args with context
                    if "symbol" in tool_args and (not tool_args["symbol"] or tool_args["symbol"] == "None"):
                        tool_args["symbol"] = stock_symbol
                    if "company_name" in tool_args and (not tool_args["company_name"] or tool_args["company_name"] == "None"):
                        tool_args["company_name"] = company_name
                    if "sector" in tool_args and (not tool_args["sector"] or tool_args["sector"] == "None"):
                        tool_args["sector"] = sector
                    
                    # Check if data is already available from research
                    if tool_name in gathered_data and gathered_data[tool_name]:
                        self.logger.info(f"Iteration {iteration + 1}: Reusing data from research for {tool_name} (skipping tool call)")
                        result = gathered_data[tool_name]
                        
                        # Create tool message with existing data
                        result_str = json.dumps(result, default=str)
                        if len(result_str) > 2000:
                            result_str = result_str[:2000] + "... (truncated)"
                        
                        tool_message = ToolMessage(
                            content=f"Using existing data from research: {result_str}",
                            tool_call_id=tool_call_id
                        )
                        tool_results.append(tool_message)
                        
                        iterations.append({
                            "iteration": iteration + 1,
                            "tool": tool_name,
                            "args": tool_args,
                            "success": True,
                            "reused_from_research": True
                        })
                        continue
                    
                    self.logger.info(f"Iteration {iteration + 1}: Executing {tool_name} with args {tool_args}")
                    
                    try:
                        # Execute tool
                        tool_obj = self.tool_map.get(tool_name)
                        if not tool_obj:
                            error_msg = f"Tool {tool_name} not found"
                            errors.append(error_msg)
                            tool_results.append(ToolMessage(
                                content=f"Error: {error_msg}",
                                tool_call_id=tool_call_id
                            ))
                            continue
                        
                        # Call tool
                        try:
                            result = await tool_obj.ainvoke(tool_args)
                        except AttributeError:
                            result = tool_obj.invoke(tool_args)
                        
                        # Store results
                        gathered_data[tool_name] = result
                        tools_executed.append(tool_name)
                        
                        # Create tool message (truncate very long results)
                        result_str = json.dumps(result, default=str)
                        if len(result_str) > 2000:
                            result_str = result_str[:2000] + "... (truncated)"
                        
                        tool_message = ToolMessage(
                            content=result_str,
                            tool_call_id=tool_call_id
                        )
                        tool_results.append(tool_message)
                        
                        iterations.append({
                            "iteration": iteration + 1,
                            "tool": tool_name,
                            "args": tool_args,
                            "success": True
                        })
                        
                    except Exception as e:
                        error_msg = f"Tool {tool_name} execution failed: {str(e)}"
                        self.logger.error(error_msg)
                        errors.append(error_msg)
                        
                        tool_results.append(ToolMessage(
                            content=f"Error: {error_msg}",
                            tool_call_id=tool_call_id
                        ))
                        
                        iterations.append({
                            "iteration": iteration + 1,
                            "tool": tool_name,
                            "args": tool_args,
                            "success": False,
                            "error": str(e)
                        })
                
                # Step 3: Add tool results to messages for next iteration
                messages.extend(tool_results)
                
            except Exception as e:
                error_msg = f"Agent loop iteration {iteration + 1} failed: {str(e)}"
                self.logger.error(error_msg)
                errors.append(error_msg)
                break
        
        # If we hit max iterations, return what we have
        self.logger.warning(f"Agent loop reached max iterations ({self.max_iterations})")
        
        return {
            "gathered_data": gathered_data,
            "tools_executed": tools_executed,
            "confidence": min(1.0, 0.3 + (len(tools_executed) * 0.05)),
            "iterations": iterations,
            "errors": errors,
            "max_iterations_reached": True
        }
    
    def _extract_research_data_for_tools(self, research_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract research data and map to tool names for reuse.
        
        Maps research data to the tool names that would fetch the same data,
        so the agent can reuse it instead of calling tools.
        
        Handles both AIResearchAgent output (with gathered_data) and ResearchAgent output (structured).
        """
        extracted = {}
        
        # Check if this is from AIResearchAgent with gathered_data structure
        # AIResearchAgent stores tool results directly in gathered_data with tool names as keys
        if "ai_iterations" in research_data or "gathered_data" in research_data:
            # AIResearchAgent structure: gathered_data may be in the state.results
            # But actually, state.results has the structured format, not gathered_data
            # Let's check if we have direct tool results
            gathered_data = research_data.get("gathered_data", {})
            
            # Extract directly from gathered_data if available (tool names as keys)
            if isinstance(gathered_data, dict):
                if "get_stock_metrics" in gathered_data:
                    extracted["get_stock_metrics"] = gathered_data["get_stock_metrics"]
                if "get_company_info" in gathered_data:
                    extracted["get_company_info"] = gathered_data["get_company_info"]
                if "search_company_news" in gathered_data:
                    extracted["search_company_news"] = gathered_data["search_company_news"]
                if "search_market_trends" in gathered_data:
                    extracted["search_market_trends"] = gathered_data["search_market_trends"]
        
        # Handle structured format from both AIResearchAgent and ResearchAgent
        company_data = research_data.get("company_data", {})
        
        if isinstance(company_data, dict):
            # FIRST: Check for nested structure (both stock_metrics and company_info) - this is the current AIResearchAgent format
            stock_metrics = company_data.get("stock_metrics")
            company_info = company_data.get("company_info")
            
            if stock_metrics or company_info:
                # This is nested structure - extract both
                if stock_metrics and "get_stock_metrics" not in extracted:
                    extracted["get_stock_metrics"] = stock_metrics
                    self.logger.debug("Extracted stock_metrics from nested company_data structure")
                if company_info and "get_company_info" not in extracted:
                    extracted["get_company_info"] = company_info
                    self.logger.debug("Extracted company_info from nested company_data structure")
            else:
                # No nested structure - check if company_data itself is flat data
                # Check if it looks like stock_metrics (has metrics fields)
                if any(key in company_data for key in ["current_price", "market_cap", "pe_ratio", "pb_ratio"]):
                    if "get_stock_metrics" not in extracted:
                        extracted["get_stock_metrics"] = company_data
                        self.logger.debug("Extracted stock_metrics from flat company_data")
                # Check if it looks like company_info (has company info fields)
                elif any(key in company_data for key in ["company_name", "name", "sector", "industry", "business"]):
                    if "get_company_info" not in extracted:
                        extracted["get_company_info"] = company_data
                        self.logger.debug("Extracted company_info from flat company_data")
        
        # Extract company news (search_company_news)
        news_data = research_data.get("news_data", {})
        if news_data and "search_company_news" not in extracted:
            extracted["search_company_news"] = news_data
        
        # Extract market trends (search_market_trends)
        # AIResearchAgent stores it directly as "market_trends" key
        market_trends = research_data.get("market_trends", {})
        if market_trends:
            extracted["search_market_trends"] = market_trends
        
        # Also check sector_data for trends (ResearchAgent structure)
        sector_data = research_data.get("sector_data", {})
        if sector_data:
            trends = sector_data.get("trends", {})
            if trends and "search_market_trends" not in extracted:
                extracted["search_market_trends"] = trends
        
        return extracted
    
    def _summarize_available_data(self, gathered_data: Dict[str, Any]) -> str:
        """
        Summarize available data for the LLM prompt.
        
        Generically extracts tool names from gathered_data and summarizes
        what data is available, including details about metrics and company info.
        Uses tool descriptions from tool_map dynamically.
        
        Args:
            gathered_data: Dictionary mapping tool names to available data
            
        Returns:
            String summary of available data with details
        """
        if not gathered_data:
            return "No data available from research. You may need to call tools to gather data."
        
        summary_parts = []

        # Get all available tool names from tool_map for reference
        available_tool_names = set(self.tool_map.keys())
        
        # Iterate through gathered_data dynamically
        for tool_name, tool_data in gathered_data.items():
            if not tool_data:
                continue
            
            # Get tool description from tool_map if available
            tool_obj = self.tool_map.get(tool_name)
            tool_description = ""
            if tool_obj:
                try:
                    tool_description = getattr(tool_obj, 'description', '') or ""
                except:
                    pass
            
            summary_parts.append(f"\n✓ {tool_name}:")
            
            # Extract details generically from tool_data structure
            details = []
            
            if isinstance(tool_data, dict):
                # Extract company info fields (generic field names)
                company_info_fields = {
                    "company_name": "Company",
                    "name": "Company",
                    "sector": "Sector",
                    "industry": "Industry",
                    "business": "Business",
                    "description": "Description"
                }
                
                # Extract stock metrics (generic numeric fields)
                metrics_fields = {
                    "current_price": ("Price", "₹{:.2f}"),
                    "market_cap": ("Market Cap", "₹{:,.0f}"),
                    "pe_ratio": ("P/E Ratio", "{:.2f}"),
                    "pb_ratio": ("P/B Ratio", "{:.2f}"),
                    "dividend_yield": ("Dividend Yield", "{:.2%}"),
                    "eps": ("EPS", "₹{:.2f}"),
                    "volume": ("Volume", "{:,}"),
                    "52_week_high": ("52W High", "₹{:.2f}"),
                    "52_week_low": ("52W Low", "₹{:.2f}")
                }
                
                # Check for company info fields
                for field, label in company_info_fields.items():
                    if field in tool_data and tool_data[field]:
                        value = tool_data[field]
                        if field == "description" and isinstance(value, str) and len(value) > 50:
                            details.append(f"{label}: {value[:50]}...")
                        else:
                            details.append(f"{label}: {value}")
                
                # Check for metrics fields
                for field, (label, fmt) in metrics_fields.items():
                    if field in tool_data and tool_data[field] is not None:
                        try:
                            value = float(tool_data[field])
                            details.append(f"{label}: {fmt.format(value)}")
                        except (ValueError, TypeError):
                            pass
                
                # Check for list/array fields (like articles, results)
                if "articles" in tool_data:
                    articles = tool_data["articles"]
                    if isinstance(articles, list):
                        details.append(f"{len(articles)} news articles available")
                
                if "results" in tool_data:
                    results = tool_data["results"]
                    if isinstance(results, list):
                        details.append(f"{len(results)} results available")
                
                # If no specific details extracted, show data type and keys
                if not details:
                    keys = list(tool_data.keys())[:5]  # Show first 5 keys
                    details.append(f"Data available with fields: {', '.join(keys)}")
                    if len(tool_data) > 5:
                        details[-1] += f" (+{len(tool_data) - 5} more)"
            
            elif isinstance(tool_data, (list, tuple)):
                details.append(f"{len(tool_data)} items available")
            
            # Add extracted details
            if details:
                summary_parts.append(f"  - {', '.join(details[:3])}")  # Limit to 3 details per tool
                if len(details) > 3:
                    summary_parts.append(f"  - ... and {len(details) - 3} more fields")
            
            # Add tool description if no details were extracted
            elif tool_description:
                summary_parts.append(f"  - {tool_description[:100]}")
        
        if len(summary_parts) == 1:  # Only header, no data
            return "No data available from research. You may need to call tools to gather data."
        
        return "\n".join(summary_parts)
    
    async def _perform_comprehensive_analysis(
        self,
        stock_symbol: str,
        company_name: str,
        sector: str,
        gathered_data: Dict[str, Any],
        research_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on gathered data.
        
        Args:
            stock_symbol: Stock symbol
            company_name: Company name
            sector: Sector name
            gathered_data: Data gathered by agent loop
            research_data: Data from research agent
            
        Returns:
            Dictionary with all analysis results
        """
        results = {
            "financial": {},
            "management": {},
            "technical": {},
            "valuation": {}
        }
        
        try:
            # Get stock metrics (from gathered data or research data)
            stock_metrics = gathered_data.get("get_stock_metrics")
            if not stock_metrics:
                company_data = research_data.get("company_data", {})
                stock_metrics = company_data.get("stock_metrics")
            
            # Get company info
            company_info = gathered_data.get("get_company_info")
            if not company_info:
                company_data = research_data.get("company_data", {})
                company_info = company_data.get("company_info")
            
            # Get news data
            news_data = gathered_data.get("search_company_news")
            if not news_data:
                news_data = research_data.get("news_data", {})
            
            # Get market trends
            market_trends = gathered_data.get("search_market_trends")
            if not market_trends:
                sector_data = research_data.get("sector_data", {})
                market_trends = sector_data.get("trends", {})
            
            # 1. Financial Analysis
            if stock_metrics:
                results["financial"] = await self._perform_financial_analysis(stock_metrics)
            
            # 2. Management Analysis
            if company_info or news_data:
                results["management"] = await self._perform_management_analysis(company_info, news_data)
            
            # 3. Technical Analysis
            if stock_metrics:
                results["technical"] = self._perform_technical_analysis(stock_metrics)
            
            # 4. Valuation Analysis
            if stock_metrics:
                results["valuation"] = await self._perform_valuation_analysis(stock_metrics, sector, market_trends)
            
        except Exception as e:
            self.logger.error(f"Comprehensive analysis failed: {e}")
            results["error"] = str(e)
        
        return results
    
    async def _perform_financial_analysis(self, stock_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Perform financial analysis."""
        try:
            # Calculate financial ratios
            financial_ratios = self._calculate_financial_ratios(stock_metrics)
            
            # Analyze financial health using LLM
            financial_health = await self._analyze_financial_health(stock_metrics, financial_ratios)
            
            return {
                "stock_metrics": stock_metrics,
                "financial_ratios": financial_ratios,
                "financial_health": financial_health,
                "analysis_type": "comprehensive_financial"
            }
        except Exception as e:
            self.logger.error(f"Financial analysis failed: {e}")
            return {"error": str(e)}
    
    async def _perform_management_analysis(self, company_info: Dict[str, Any], news_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform management analysis."""
        try:
            # Extract key information
            analysis_input = {
                "company_info": company_info or {},
                "recent_news": news_data.get("articles", [])[:5] if news_data else []
            }
            
            # Use LLM for management assessment
            prompt = f"""Analyze management effectiveness and governance for this company.

Company Info: {json.dumps(company_info, default=str)[:1000] if company_info else "Limited information available"}
Recent News: {json.dumps(analysis_input["recent_news"], default=str)[:1000]}

Provide assessment in JSON format:
{{
    "management_score": <0-100 integer>,
    "key_strengths": ["strength1", "strength2"],
    "key_concerns": ["concern1", "concern2"],
    "overall_assessment": "<brief assessment>"
}}"""
            
            response = await self.openai_client.chat.completions.create(
                model=Config.DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a management analyst. Analyze management effectiveness and return structured JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            response_text = response.choices[0].message.content
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            if json_match:
                analysis_data = json.loads(json_match.group(0))
                return {
                    "management_score": analysis_data.get("management_score", 50),
                    "key_strengths": analysis_data.get("key_strengths", []),
                    "key_concerns": analysis_data.get("key_concerns", []),
                    "overall_assessment": analysis_data.get("overall_assessment", "Limited assessment"),
                    "analysis_type": "management_analysis"
                }
            else:
                return {
                    "management_score": 50,
                    "key_strengths": [],
                    "key_concerns": [],
                    "overall_assessment": "Assessment completed",
                    "analysis_type": "management_analysis"
                }
        except Exception as e:
            self.logger.error(f"Management analysis failed: {e}")
            return {"error": str(e), "analysis_type": "management_analysis"}
    
    def _perform_technical_analysis(self, stock_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Perform technical analysis."""
        try:
            # Calculate technical indicators
            technical_indicators = self._calculate_technical_indicators(stock_metrics)
            
            # Format technical analysis
            try:
                technical_summary = format_technical_analysis.invoke({"technical_data": technical_indicators})
            except:
                technical_summary = "Technical analysis completed"
            
            return {
                "stock_metrics": stock_metrics,
                "technical_indicators": technical_indicators,
                "technical_summary": technical_summary,
                "analysis_type": "technical_analysis"
            }
        except Exception as e:
            self.logger.error(f"Technical analysis failed: {e}")
            return {"error": str(e), "analysis_type": "technical_analysis"}
    
    async def _perform_valuation_analysis(self, stock_metrics: Dict[str, Any], sector: str, market_trends: Dict[str, Any]) -> Dict[str, Any]:
        """Perform valuation analysis."""
        try:
            # Calculate valuation metrics
            valuation_metrics = self._calculate_valuation_metrics(stock_metrics, sector)
            
            # Calculate target price using LLM
            target_price = await self._calculate_target_price_llm(stock_metrics, valuation_metrics, sector, market_trends)
            
            return {
                "stock_metrics": stock_metrics,
                "market_trends": market_trends,
                "valuation_metrics": valuation_metrics,
                "target_price": target_price,
                "analysis_type": "valuation_analysis"
            }
        except Exception as e:
            self.logger.error(f"Valuation analysis failed: {e}")
            return {"error": str(e), "analysis_type": "valuation_analysis"}
    
    def _calculate_financial_ratios(self, stock_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate financial ratios."""
        ratios = {}
        ratios["pe_ratio"] = stock_metrics.get("pe_ratio")
        ratios["pb_ratio"] = stock_metrics.get("pb_ratio")
        ratios["dividend_yield"] = stock_metrics.get("dividend_yield")
        ratios["beta"] = stock_metrics.get("beta")
        
        current_price = stock_metrics.get("current_price", 0)
        market_cap = stock_metrics.get("market_cap", 0)
        
        if current_price and market_cap:
            ratios["market_cap_category"] = self._categorize_market_cap(market_cap)
        
        return ratios
    
    async def _analyze_financial_health(self, stock_metrics: Dict[str, Any], ratios: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze financial health using LLM."""
        try:
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
            
            prompt = f"""Analyze financial health from stock metrics and ratios. Return JSON:

<stock_metrics>
{json.dumps(key_metrics, indent=2, default=str)}
</stock_metrics>

<ratios>
{json.dumps(ratios, indent=2, default=str)}
</ratios>

Output format:
{{
    "health_score": <0-100 integer>,
    "health_factors": ["factor1", "factor2"],
    "overall_assessment": "<brief assessment>"
}}"""
            
            response = await self.openai_client.chat.completions.create(
                model=Config.DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a financial analyst. Analyze financial health and return structured JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
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
                return {
                    "health_score": 50,
                    "health_factors": ["Analysis completed"],
                    "overall_assessment": "Financial health assessment completed"
                }
        except Exception as e:
            self.logger.error(f"Financial health analysis failed: {e}")
            return {"health_score": 0, "health_factors": [], "overall_assessment": "Unable to assess"}
    
    def _calculate_technical_indicators(self, stock_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate technical indicators."""
        indicators = {}
        
        current_price = stock_metrics.get("current_price", 0)
        high_52w = stock_metrics.get("52_week_high", 0)
        low_52w = stock_metrics.get("52_week_low", 0)
        
        if current_price and high_52w and low_52w:
            range_position = (current_price - low_52w) / (high_52w - low_52w) if high_52w != low_52w else 0.5
            indicators["range_position"] = range_position
            
            if range_position > 0.8:
                indicators["trend"] = "Near 52-week high"
            elif range_position < 0.2:
                indicators["trend"] = "Near 52-week low"
            else:
                indicators["trend"] = "Mid-range"
            
            upside_potential = ((high_52w - current_price) / current_price) * 100
            downside_risk = ((current_price - low_52w) / current_price) * 100
            indicators["upside_potential"] = upside_potential
            indicators["downside_risk"] = downside_risk
        
        volume = stock_metrics.get("volume", 0)
        avg_volume = stock_metrics.get("avg_volume", 0)
        
        if volume and avg_volume:
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1
            indicators["volume_ratio"] = volume_ratio
            
            if volume_ratio > 1.5:
                indicators["volume_trend"] = "High volume"
            elif volume_ratio < 0.5:
                indicators["volume_trend"] = "Low volume"
            else:
                indicators["volume_trend"] = "Normal volume"
        
        return indicators
    
    def _calculate_valuation_metrics(self, stock_metrics: Dict[str, Any], sector: str) -> Dict[str, Any]:
        """Calculate valuation metrics."""
        metrics = {}
        
        pe_ratio = stock_metrics.get("pe_ratio")
        pb_ratio = stock_metrics.get("pb_ratio")
        market_cap = stock_metrics.get("market_cap", 0)
        
        if pe_ratio:
            metrics["pe_ratio"] = pe_ratio
            if pe_ratio < 15:
                metrics["pe_assessment"] = "Undervalued"
            elif pe_ratio < 25:
                metrics["pe_assessment"] = "Fairly valued"
            else:
                metrics["pe_assessment"] = "Overvalued"
        
        if pb_ratio:
            metrics["pb_ratio"] = pb_ratio
            if pb_ratio < 1:
                metrics["pb_assessment"] = "Trading below book value"
            elif pb_ratio < 3:
                metrics["pb_assessment"] = "Reasonable price-to-book"
            else:
                metrics["pb_assessment"] = "High price-to-book"
        
        if market_cap:
            metrics["market_cap"] = market_cap
            metrics["market_cap_category"] = self._categorize_market_cap(market_cap)
        
        return metrics
    
    async def _calculate_target_price_llm(
        self, 
        stock_metrics: Dict[str, Any], 
        valuation_metrics: Dict[str, Any],
        sector: str,
        market_trends: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate target price using LLM analysis.
        
        Args:
            stock_metrics: Current stock metrics (price, ratios, etc.)
            valuation_metrics: Calculated valuation metrics
            sector: Sector name
            market_trends: Market trends data
            
        Returns:
            Dictionary with target_price, recommendation, reasoning, etc.
        """
        try:
            current_price = stock_metrics.get("current_price", 0)
            if not current_price:
                return {"target_price": 0, "method": "insufficient_data", "error": "No current price available"}
            
            # Prepare prompt with all relevant data
            market_cap = stock_metrics.get('market_cap')
            market_cap_str = f"₹{market_cap:,.0f}" if market_cap else 'N/A'
            
            dividend_yield = stock_metrics.get('dividend_yield')
            dividend_yield_str = f"{dividend_yield * 100:.2f}%" if dividend_yield else "0.00%"
            
            prompt = f"""You are a financial analyst calculating a target price for a stock.

STOCK METRICS:
- Current Price: ₹{current_price:.2f}
- P/E Ratio: {stock_metrics.get('pe_ratio', 'N/A')}
- P/B Ratio: {stock_metrics.get('pb_ratio', 'N/A')}
- Market Cap: {market_cap_str}
- EPS: ₹{stock_metrics.get('eps', 'N/A')}
- Dividend Yield: {dividend_yield_str}
- 52 Week High: ₹{stock_metrics.get('52_week_high', 'N/A')}
- 52 Week Low: ₹{stock_metrics.get('52_week_low', 'N/A')}

VALUATION METRICS:
{self._format_valuation_metrics_for_llm(valuation_metrics)}

SECTOR: {sector}

MARKET TRENDS:
{self._format_market_trends_for_llm(market_trends)}

Analyze this stock and provide:
1. Target Price: A realistic target price in ₹ (Indian Rupees) based on valuation analysis
2. Recommendation: One of BUY, HOLD, or SELL
3. Upside Potential: Percentage upside/downside from current price
4. Time Horizon: Suggested time horizon for the target (e.g., "12 months", "6 months")
5. Reasoning: Brief explanation (2-3 sentences) of your analysis and how you arrived at the target price

Consider:
- Valuation metrics (P/E, P/B ratios compared to sector/industry averages)
- Growth prospects and market trends
- Current price relative to 52-week range
- Risk factors
- Market sentiment

Respond in JSON format:
{{
    "target_price": <numeric_value>,
    "current_price": {current_price:.2f},
    "upside_potential": <percentage>,
    "recommendation": "<BUY|HOLD|SELL>",
    "time_horizon": "<time period>",
    "reasoning": "<brief explanation>"
}}"""

            # Call LLM
            response = await self.openai_client.chat.completions.create(
                model=Config.DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert financial analyst specializing in stock valuation. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            response_content = response.choices[0].message.content
            import json
            result = json.loads(response_content)
            
            # Validate and format result
            target_price = float(result.get("target_price", current_price))
            recommendation = result.get("recommendation", "HOLD").upper()
            if recommendation not in ["BUY", "HOLD", "SELL"]:
                recommendation = "HOLD"
            
            upside_potential = result.get("upside_potential", 0)
            if not upside_potential:
                # Calculate if not provided
                upside_potential = ((target_price - current_price) / current_price) * 100
            
            return {
                "target_price": round(target_price, 2),
                "current_price": current_price,
                "upside_potential": round(float(upside_potential), 2),
                "recommendation": recommendation,
                "time_horizon": result.get("time_horizon", "12 months"),
                "reasoning": result.get("reasoning", ""),
                "method": "llm_analysis"
            }
            
        except Exception as e:
            self.logger.error(f"LLM target price calculation failed: {e}")
            # Fallback to simple calculation
            current_price = stock_metrics.get("current_price", 0)
            pe_ratio = stock_metrics.get("pe_ratio")
            
            if not current_price or not pe_ratio:
                return {"target_price": current_price, "method": "fallback_insufficient_data", "error": str(e)}
            
            # Simple fallback
            if pe_ratio < 15:
                target_price = current_price * 1.2
                recommendation = "BUY"
            elif pe_ratio < 25:
                target_price = current_price * 1.1
                recommendation = "HOLD"
            else:
                target_price = current_price * 0.9
                recommendation = "SELL"
            
            return {
                "target_price": round(target_price, 2),
                "current_price": current_price,
                "upside_potential": round(((target_price - current_price) / current_price) * 100, 2),
                "recommendation": recommendation,
                "method": "fallback_pe_ratio_based",
                "error": str(e)
            }
    
    def _format_valuation_metrics_for_llm(self, valuation_metrics: Dict[str, Any]) -> str:
        """Format valuation metrics for LLM prompt."""
        if not valuation_metrics:
            return "No valuation metrics available"
        
        lines = []
        for key, value in valuation_metrics.items():
            if value is not None:
                if isinstance(value, (int, float)):
                    if key in ["pe_ratio", "pb_ratio", "market_cap"]:
                        lines.append(f"- {key.replace('_', ' ').title()}: {value}")
                    else:
                        lines.append(f"- {key.replace('_', ' ').title()}: {value}")
                else:
                    lines.append(f"- {key.replace('_', ' ').title()}: {value}")
        
        return "\n".join(lines) if lines else "No valuation metrics available"
    
    def _format_market_trends_for_llm(self, market_trends: Dict[str, Any]) -> str:
        """Format market trends for LLM prompt."""
        if not market_trends:
            return "No market trends data available"
        
        # Extract key information from market trends
        if isinstance(market_trends, dict):
            if "results" in market_trends and isinstance(market_trends["results"], list):
                # Take first few results
                results = market_trends["results"][:3]
                lines = ["Recent market trends:"]
                for i, result in enumerate(results, 1):
                    if isinstance(result, dict):
                        title = result.get("title", "Trend")[:100]
                        snippet = result.get("snippet", "")[:200]
                        lines.append(f"{i}. {title}")
                        if snippet:
                            lines.append(f"   {snippet}")
                return "\n".join(lines)
            else:
                # Try to extract any text content
                return str(market_trends)[:500]
        
        return "No market trends data available"
    
    def _categorize_market_cap(self, market_cap: float) -> str:
        """Categorize market cap."""
        if market_cap >= 100000000000:  # >= 100k crores
            return "Large Cap"
        elif market_cap >= 20000000000:  # >= 20k crores
            return "Mid Cap"
        else:
            return "Small Cap"

