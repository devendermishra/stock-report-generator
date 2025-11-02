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
    from .ai_analysis_helpers import (
        extract_research_data_for_tools,
        summarize_available_data,
        perform_comprehensive_analysis
    )
    from ..tools.stock_data_tool import get_stock_metrics, get_company_info
    from ..tools.web_search_tool import search_company_news, search_market_trends
    from ..tools.technical_analysis_formatter import format_technical_analysis
    from ..config import Config
except ImportError:
    from agents.base_agent import BaseAgent, AgentState
    from agents.ai_analysis_helpers import (
        extract_research_data_for_tools,
        summarize_available_data,
        perform_comprehensive_analysis
    )
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
            gathered_data = extract_research_data_for_tools(research_data)
            
            # Log what was extracted
            self.logger.info(f"Extracted data from research: {list(gathered_data.keys())}")
            if "get_stock_metrics" in gathered_data:
                self.logger.info(f"Stock metrics available: {bool(gathered_data['get_stock_metrics'])}")
            if "get_company_info" in gathered_data:
                self.logger.info(f"Company info available: {bool(gathered_data['get_company_info'])}")
            
            # Build initial task description with available data summary
            available_data_summary = summarize_available_data(gathered_data, self.tool_map)
            
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
            analysis_results = await perform_comprehensive_analysis(
                stock_symbol, company_name, sector, final_gathered_data, research_data, self.openai_client
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
    

