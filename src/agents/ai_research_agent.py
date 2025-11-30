"""
AI Research Agent - True LangGraph Agent with Iterative Tool Selection.

This agent implements the true LangGraph agent pattern with iterative decision-making:
1. LLM decides which tool to use next
2. Executes the tool
3. Observes the result
4. Decides next action based on observation
5. Repeats until goal achieved

This is an alternative to ResearchPlannerAgent + ResearchAgent workflow.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import asyncio

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
    from ..tools.web_search_tool import search_sector_news, search_company_news, search_market_trends
    from ..tools.generic_web_search_tool import search_web_generic
    from ..config import Config
    from ..utils.metrics import record_llm_request
    from ..tools.openai_logger import openai_logger
except ImportError:
    from agents.base_agent import BaseAgent, AgentState
    from tools.stock_data_tool import get_stock_metrics, get_company_info
    from tools.web_search_tool import search_sector_news, search_company_news, search_market_trends
    from tools.generic_web_search_tool import search_web_generic
    from config import Config
    from utils.metrics import record_llm_request
    from tools.openai_logger import openai_logger

logger = logging.getLogger(__name__)


class AIResearchAgent(BaseAgent):
    """
    AI Research Agent with iterative tool selection using LLM decision-making.
    
    **Specialization:** Adaptive Iterative Research (AI-Powered Iterative Mode)
    
    **Role:** Uses iterative LLM-based decision-making to dynamically gather research
    data based on emerging insights. Implements the true LangGraph agent pattern
    with autonomous tool selection.
    
    **When Used:** Only in AI-Powered Iterative Mode (replaces ResearchPlannerAgent + ResearchAgent)
    
    **Execution Pattern:**
    This agent uses an LLM in a loop to:
    1. Analyze current state and gathered information
    2. Decide which tool to use next
    3. Execute the tool
    4. Observe results
    5. Decide if goal is achieved or continue with next tool
    
    **Key Features:**
    - Dynamic tool selection based on current state
    - Self-correcting (can gather missing data if initial results incomplete)
    - Efficient (skips unnecessary tools if goal already achieved)
    - Adaptive (adjusts research depth based on data availability)
    
    **Tools Used:**
    - get_stock_metrics, get_company_info
    - search_sector_news, search_company_news, search_market_trends
    - search_web_generic
    
    **Max Iterations:** 5 (configurable)
    
    For detailed information on agent specialization and roles,
    see docs/AGENT_SPECIALIZATION.md
    """
    
    def __init__(self, agent_id: str, openai_api_key: str):
        """
        Initialize the AI Research Agent.
        
        Args:
            agent_id: Unique identifier for the agent
            openai_api_key: OpenAI API key for LLM calls
        """
        # Define available tools
        # Note: validate_symbol is excluded since symbol is always validated before agent execution
        available_tools = [
            get_stock_metrics,
            get_company_info,
            search_sector_news,
            search_company_news,
            search_market_trends,
            search_web_generic
        ]
        
        super().__init__(agent_id, openai_api_key, available_tools)
        
        # Initialize LLM with tool bindings
        if ChatOpenAI is None:
            raise ImportError("langchain_openai is required for AIResearchAgent. Install with: pip install langchain-openai")
        
        self.llm = ChatOpenAI(
            model=Config.DEFAULT_MODEL,
            temperature=0.1,
            api_key=openai_api_key
        )
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(available_tools)
        
        # Tool name to function mapping (use actual tool names from @tool decorators)
        self.tool_map = {tool.name: tool for tool in available_tools}
        
        # Maximum iterations to prevent infinite loops
        self.max_iterations = 5
    
    async def execute_task(
        self,
        stock_symbol: str,
        company_name: str,
        sector: str,
        context: Dict[str, Any]
    ) -> AgentState:
        """
        Execute research using true agent pattern with iterative tool selection.
        
        Args:
            stock_symbol: NSE stock symbol
            company_name: Full company name
            sector: Sector name
            context: Current context from previous agents
            
        Returns:
            AgentState with research results gathered iteratively
        """
        start_time = datetime.now()
        
        # Initialize state
        state = AgentState(
            agent_id=self.agent_id,
            stock_symbol=stock_symbol,
            company_name=company_name,
            sector=sector,
            current_task="ai_iterative_research",
            context=context,
            results={},
            tools_used=[],
            confidence_score=0.0,
            errors=[],
            start_time=start_time
        )
        
        try:
            # Build initial task description
            task_description = f"""Research {company_name} ({stock_symbol}) in the {sector} sector.
            
Your goal is to gather comprehensive research data including:
1. Stock metrics and company fundamentals (use get_stock_metrics)
2. Company information and business details (use get_company_info)
3. Sector trends and market outlook (use search_sector_news)
4. Company-specific news and developments (use search_company_news)
5. Market trends and analysis (use search_market_trends or search_web_generic)

Note: The stock symbol has already been validated. You do not need to validate it.

EFFICIENCY TIP: You can call multiple independent tools at once to gather data faster. For example:
- Call get_stock_metrics AND get_company_info together in one turn
- Call search_sector_news AND search_company_news together if you need both

After tool executions, analyze the results and decide:
- If you have enough information → respond with "Research complete. I have gathered sufficient data."
- If you need more data → select and execute the next appropriate tool(s) - you can call multiple tools at once

IMPORTANT: 
- Call multiple independent tools simultaneously when possible to reduce iterations
- After gathering enough data, respond with a summary instead of calling more tools"""
            
            # Execute iterative agent loop
            agent_results = await self._agent_loop(task_description, stock_symbol, company_name, sector)
            
            # Process results - structure them similar to ResearchAgent output
            gathered_data = agent_results.get("gathered_data", {})
            
            # Store company_data in structured format with both stock_metrics and company_info
            # This allows downstream agents to access both pieces of data
            company_data = {}
            stock_metrics = gathered_data.get("get_stock_metrics")
            company_info = gathered_data.get("get_company_info")
            
            if stock_metrics:
                company_data["stock_metrics"] = stock_metrics
            if company_info:
                company_data["company_info"] = company_info
            
            # If we only have one type of data, also store it at the top level for compatibility
            if not company_data:
                # Fallback: use whichever is available
                company_data = stock_metrics or company_info or {}
            
            state.results = {
                "company_data": company_data,
                "sector_data": {
                    "sector_name": sector,
                    "trends": gathered_data.get("search_sector_news", {}),
                    "news_data": gathered_data.get("search_company_news", {})
                },
                "peer_data": {},  # AI agent doesn't gather peer data directly
                "news_data": gathered_data.get("search_company_news", {}),
                "market_trends": gathered_data.get("search_market_trends", {}),
                "ai_iterations": agent_results.get("iterations", []),
                "final_summary": agent_results.get("final_summary", ""),
                # Also preserve the original gathered_data for easier extraction
                "gathered_data": gathered_data
            }
            
            state.tools_used = agent_results.get("tools_executed", [])
            state.confidence_score = agent_results.get("confidence", 0.8)
            
            # Add any errors
            if agent_results.get("errors"):
                state.errors.extend(agent_results["errors"])
            
            state.end_time = datetime.now()
            
            self.logger.info(
                f"AI Research Agent completed research for {stock_symbol} using {len(state.tools_used)} tools "
                f"in {len(agent_results.get('iterations', []))} iterations"
            )
            
            return state
            
        except Exception as e:
            self.logger.error(f"AI Research Agent execution failed: {e}")
            state.errors.append(f"Agent execution failed: {str(e)}")
            state.confidence_score = 0.0
            state.end_time = datetime.now()
            return state
    
    async def _agent_loop(
        self,
        task_description: str,
        stock_symbol: str,
        company_name: str,
        sector: str
    ) -> Dict[str, Any]:
        """
        Execute the iterative agent loop.
        
        This implements the true agent pattern:
        1. LLM decides action (tool call or finish)
        2. Execute tool if called
        3. Observe results
        4. Update state
        5. Repeat until goal achieved
        
        Args:
            task_description: Task description for the agent
            stock_symbol: Stock symbol
            company_name: Company name
            sector: Sector name
            
        Returns:
            Dictionary with gathered data, tools executed, and confidence
        """
        if HumanMessage is None or ToolMessage is None:
            raise ImportError("langchain_core.messages is required")
        
        messages = [HumanMessage(content=task_description)]
        gathered_data = {}
        tools_executed = []
        iterations = []
        errors = []
        
        for iteration in range(self.max_iterations):
            try:
                # Step 1: LLM decides next action
                # Convert LangChain messages to OpenAI format for logging
                openai_messages = []
                for msg in messages:
                    if hasattr(msg, 'content'):
                        # Determine role based on message type
                        msg_class = msg.__class__.__name__ if hasattr(msg, '__class__') else ''
                        if 'Human' in msg_class:
                            role = 'user'
                        elif 'AIMessage' in msg_class or 'Assistant' in msg_class:
                            role = 'assistant'
                        elif 'Tool' in msg_class:
                            role = 'tool'
                        else:
                            role = 'user'  # Default
                        openai_messages.append({"role": role, "content": msg.content})
                
                # Record start time for metrics
                import time
                llm_start_time = time.time()
                
                response = await self.llm_with_tools.ainvoke(messages)
                
                # Calculate duration for metrics
                llm_duration = time.time() - llm_start_time
                
                # Record metrics for LangChain LLM call
                try:
                    # Try to extract usage info from response if available
                    request_tokens = None
                    response_tokens = None
                    if hasattr(response, 'response_metadata') and response.response_metadata:
                        usage = response.response_metadata.get('token_usage', {})
                        if usage:
                            request_tokens = usage.get('prompt_tokens')
                            response_tokens = usage.get('completion_tokens')
                    
                    record_llm_request(
                        model=Config.DEFAULT_MODEL,
                        agent_name="AIResearchAgent",
                        request_tokens=request_tokens,
                        response_tokens=response_tokens,
                        duration_seconds=llm_duration,
                        success=True
                    )
                except Exception as metrics_error:
                    self.logger.warning(f"Failed to record metrics: {metrics_error}")
                
                # Log prompt and response
                try:
                    response_content = response.content if hasattr(response, 'content') else str(response)
                    openai_logger.log_chat_completion(
                        model=Config.DEFAULT_MODEL,
                        messages=openai_messages,
                        response=response_content,
                        agent_name="AIResearchAgent"
                    )
                except Exception as log_error:
                    self.logger.warning(f"Failed to log prompt/response: {log_error}")
                
                messages.append(response)
                
                # Check if LLM wants to finish
                if not response.tool_calls:
                    # No more tool calls - agent thinks it's done
                    final_summary = response.content or "Research completed."
                    self.logger.info(f"Agent decided to finish after {iteration + 1} iterations")
                    
                    return {
                        "gathered_data": gathered_data,
                        "tools_executed": tools_executed,
                        "confidence": min(1.0, 0.5 + (len(tools_executed) * 0.1)),
                        "iterations": iterations,
                        "final_summary": final_summary,
                        "errors": errors
                    }
                
                # Step 2: Execute tool calls (execute multiple tools in parallel)
                tool_calls_data = []
                for tool_call in response.tool_calls:
                    # Extract tool call information
                    if isinstance(tool_call, dict):
                        tool_name = tool_call.get("name", "")
                        tool_args = tool_call.get("args", {})
                        tool_call_id = tool_call.get("id", f"call_{iteration}_{len(tool_calls_data)}")
                    else:
                        tool_name = getattr(tool_call, "name", "")
                        tool_args = getattr(tool_call, "args", {})
                        tool_call_id = getattr(tool_call, "id", f"call_{iteration}_{len(tool_calls_data)}")
                    
                    # Update args with context
                    if "symbol" in tool_args and (not tool_args["symbol"] or tool_args["symbol"] == "None"):
                        tool_args["symbol"] = stock_symbol
                    if "company_name" in tool_args and (not tool_args["company_name"] or tool_args["company_name"] == "None"):
                        tool_args["company_name"] = company_name
                    if "sector" in tool_args and (not tool_args["sector"] or tool_args["sector"] == "None"):
                        tool_args["sector"] = sector
                    
                    tool_calls_data.append({
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "tool_call_id": tool_call_id
                    })
                
                # Log tools to be executed
                tool_names = [tc["tool_name"] for tc in tool_calls_data]
                self.logger.info(f"Iteration {iteration + 1}: Executing {len(tool_calls_data)} tool(s): {', '.join(tool_names)}")
                
                # Execute all tools in parallel
                async def execute_single_tool(tc_data: Dict[str, Any]) -> Dict[str, Any]:
                    """Execute a single tool and return result."""
                    tool_name = tc_data["tool_name"]
                    tool_args = tc_data["tool_args"]
                    tool_call_id = tc_data["tool_call_id"]
                    
                    try:
                        tool_obj = self.tool_map.get(tool_name)
                        if not tool_obj:
                            return {
                                "tool_call_id": tool_call_id,
                                "success": False,
                                "error": f"Tool {tool_name} not found",
                                "tool_name": tool_name
                            }
                        
                        # Call tool (LangChain tools support both sync and async)
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
                        
                        return {
                            "tool_call_id": tool_call_id,
                            "success": True,
                            "result": result,
                            "tool_message": ToolMessage(content=result_str, tool_call_id=tool_call_id),
                            "tool_name": tool_name,
                            "tool_args": tool_args
                        }
                        
                    except Exception as e:
                        error_msg = f"Tool {tool_name} execution failed: {str(e)}"
                        self.logger.error(error_msg)
                        errors.append(error_msg)
                        
                        return {
                            "tool_call_id": tool_call_id,
                            "success": False,
                            "error": error_msg,
                            "tool_message": ToolMessage(content=f"Error: {error_msg}", tool_call_id=tool_call_id),
                            "tool_name": tool_name,
                            "tool_args": tool_args
                        }
                
                # Execute all tools in parallel using asyncio.gather
                tool_results_data = await asyncio.gather(*[execute_single_tool(tc) for tc in tool_calls_data])
                
                # Process results and create tool messages
                tool_results = []
                for result_data in tool_results_data:
                    tool_results.append(result_data["tool_message"])
                    
                    iterations.append({
                        "iteration": iteration + 1,
                        "tool": result_data["tool_name"],
                        "args": result_data["tool_args"],
                        "success": result_data["success"],
                        "error": result_data.get("error")
                    })
                
                # Step 3: Add tool results to messages for next iteration
                messages.extend(tool_results)
                
            except Exception as e:
                # Record failed LLM request metrics if this was an LLM call failure
                try:
                    if 'llm_start_time' in locals():
                        llm_duration = time.time() - llm_start_time
                        record_llm_request(
                            model=Config.DEFAULT_MODEL,
                            agent_name="AIResearchAgent",
                            request_tokens=None,
                            response_tokens=None,
                            duration_seconds=llm_duration,
                            success=False
                        )
                except Exception:
                    pass  # Don't fail on metrics errors
                
                error_msg = f"Agent loop iteration {iteration + 1} failed: {str(e)}"
                self.logger.error(error_msg)
                errors.append(error_msg)
                break
        
        # If we hit max iterations, return what we have
        self.logger.warning(f"Agent loop reached max iterations ({self.max_iterations})")
        
        return {
            "gathered_data": gathered_data,
            "tools_executed": tools_executed,
            "confidence": min(1.0, 0.3 + (len(tools_executed) * 0.1)),
            "iterations": iterations,
            "errors": errors,
            "max_iterations_reached": True
        }
