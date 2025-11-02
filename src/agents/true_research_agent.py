"""
True LangGraph Agent for Research - Iterative Tool Selection with LLM Decision-Making.

This agent implements the true LangGraph agent pattern:
1. LLM decides which tool to use next
2. Executes the tool
3. Observes the result
4. Decides next action based on observation
5. Repeats until goal achieved

This demonstrates how to convert a workflow node into a true agent.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
    from langchain_core.tools import StructuredTool
except ImportError:
    # Fallback if langchain packages not available
    ChatOpenAI = None
    ChatPromptTemplate = None
    MessagesPlaceholder = None
    HumanMessage = None
    AIMessage = None
    ToolMessage = None
    StructuredTool = None

try:
    from .base_agent import BaseAgent, AgentState
    from ..tools.stock_data_tool import get_stock_metrics, get_company_info, validate_symbol
    from ..tools.web_search_tool import search_sector_news, search_company_news, search_market_trends
    from ..tools.generic_web_search_tool import generic_web_search_tool, search_web_generic
    from ..config import Config
except ImportError:
    from agents.base_agent import BaseAgent, AgentState
    from tools.stock_data_tool import get_stock_metrics, get_company_info, validate_symbol
    from tools.web_search_tool import search_sector_news, search_company_news, search_market_trends
    from tools.generic_web_search_tool import generic_web_search_tool, search_web_generic
    from config import Config

logger = logging.getLogger(__name__)


class TrueResearchAgent(BaseAgent):
    """
    True LangGraph Agent for Research with iterative tool selection.
    
    This agent uses an LLM in a loop to:
    1. Analyze current state and gathered information
    2. Decide which tool to use next
    3. Execute the tool
    4. Observe results
    5. Decide if goal is achieved or continue with next tool
    
    Implements the true agent pattern from LangGraph.
    """
    
    def __init__(self, agent_id: str, openai_api_key: str):
        """
        Initialize the True Research Agent.
        
        Args:
            agent_id: Unique identifier for the agent
            openai_api_key: OpenAI API key for LLM calls
        """
        # Define available tools
        available_tools = [
            get_stock_metrics,
            get_company_info,
            validate_symbol,
            search_sector_news,
            search_company_news,
            search_market_trends,
            search_web_generic
        ]
        
        super().__init__(agent_id, openai_api_key, available_tools)
        
        # Initialize LLM with tool bindings
        if ChatOpenAI is None:
            raise ImportError("langchain_openai is required for TrueResearchAgent. Install with: pip install langchain-openai")
        
        self.llm = ChatOpenAI(
            model=Config.DEFAULT_MODEL,
            temperature=0.1,
            api_key=openai_api_key
        )
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(available_tools)
        
        # Build agent prompt (optional, using messages directly for simplicity)
        if ChatPromptTemplate is not None:
            self.agent_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a financial research agent responsible for gathering comprehensive 
research data about stocks. Your goal is to gather complete information about:
1. Company fundamentals (stock metrics, company info)
2. Sector analysis (sector trends, market news)
3. Company-specific news and developments
4. Market trends and outlook

You have access to various tools. Use them strategically:
- Start with basic validation and stock metrics
- Then gather company information
- Search for sector and company news
- Explore market trends if needed

After each tool execution, analyze the results and decide:
- If you have enough information → respond with final summary
- If you need more data → select and execute the next appropriate tool

IMPORTANT: Only call tools when you actually need the information. 
Don't call tools you've already used unless you need updated information."""),
                MessagesPlaceholder(variable_name="messages"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])
        else:
            self.agent_prompt = None
        
        # Tool name to function mapping
        self.tool_map = {
            "get_stock_metrics": get_stock_metrics,
            "get_company_info": get_company_info,
            "validate_symbol": validate_symbol,
            "search_sector_news": search_sector_news,
            "search_company_news": search_company_news,
            "search_market_trends": search_market_trends,
            "search_web_generic": search_web_generic
        }
        
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
            current_task="iterative_research",
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
1. Stock metrics and company fundamentals
2. Company information and business details  
3. Sector trends and market outlook
4. Company-specific news and developments
5. Market trends and analysis

Available context: {json.dumps(context, indent=2)[:500]}..."""
            
            # Execute iterative agent loop
            agent_results = await self._agent_loop(task_description, stock_symbol, company_name, sector)
            
            # Process results
            state.results = agent_results.get("gathered_data", {})
            state.tools_used = agent_results.get("tools_executed", [])
            state.confidence_score = agent_results.get("confidence", 0.8)
            
            # Add any errors
            if agent_results.get("errors"):
                state.errors.extend(agent_results["errors"])
            
            state.end_time = datetime.now()
            
            self.logger.info(
                f"True agent completed research for {stock_symbol} using {len(state.tools_used)} tools "
                f"in {len(agent_results.get('iterations', []))} iterations"
            )
            
            return state
            
        except Exception as e:
            self.logger.error(f"True agent execution failed: {e}")
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
        if HumanMessage is None:
            raise ImportError("langchain_core.messages is required")
        
        messages = [HumanMessage(content=task_description)]
        gathered_data = {}
        tools_executed = []
        iterations = []
        errors = []
        
        for iteration in range(self.max_iterations):
            try:
                # Step 1: LLM decides next action
                response = await self.llm_with_tools.ainvoke(messages)
                messages.append(response)
                
                # Check if LLM wants to finish
                if not response.tool_calls:
                    # No more tool calls - agent thinks it's done
                    final_summary = response.content
                    self.logger.info(f"Agent decided to finish after {iteration + 1} iterations")
                    
                    return {
                        "gathered_data": gathered_data,
                        "tools_executed": tools_executed,
                        "confidence": min(1.0, 0.5 + (len(tools_executed) * 0.1)),
                        "iterations": iterations,
                        "final_summary": final_summary,
                        "errors": errors
                    }
                
                # Step 2: Execute tool calls
                tool_results = []
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    
                    # Update args with context
                    if "symbol" in tool_args and not tool_args["symbol"]:
                        tool_args["symbol"] = stock_symbol
                    if "company_name" in tool_args and not tool_args["company_name"]:
                        tool_args["company_name"] = company_name
                    if "sector" in tool_args and not tool_args["sector"]:
                        tool_args["sector"] = sector
                    
                    self.logger.info(f"Iteration {iteration + 1}: Executing {tool_name} with args {tool_args}")
                    
                    try:
                        # Execute tool
                        tool_func = self.tool_map.get(tool_name)
                        if not tool_func:
                            error_msg = f"Tool {tool_name} not found"
                            errors.append(error_msg)
                            tool_results.append(ToolMessage(
                                content=f"Error: {error_msg}",
                                tool_call_id=tool_call["id"]
                            ))
                            continue
                        
                        # Call tool
                        if isinstance(tool_func, StructuredTool):
                            result = await tool_func.ainvoke(tool_args)
                        else:
                            # Handle regular function tools
                            import inspect
                            if inspect.iscoroutinefunction(tool_func):
                                result = await tool_func(**tool_args)
                            else:
                                result = tool_func(**tool_args)
                        
                        # Store results
                        gathered_data[tool_name] = result
                        tools_executed.append(tool_name)
                        
                        # Create tool message
                        if ToolMessage is None:
                            raise ImportError("langchain_core.messages is required")
                        
                        tool_message = ToolMessage(
                            content=json.dumps(result, default=str)[:2000],  # Truncate very long results
                            tool_call_id=tool_call["id"]
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
                            tool_call_id=tool_call["id"]
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
                
                # Step 4: Check if we have enough information
                # LLM will decide in next iteration based on gathered data
                
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
            "confidence": min(1.0, 0.3 + (len(tools_executed) * 0.1)),
            "iterations": iterations,
            "errors": errors,
            "max_iterations_reached": True
        }
    
    async def _should_continue(
        self,
        gathered_data: Dict[str, Any],
        required_keys: List[str]
    ) -> bool:
        """
        Determine if agent should continue based on gathered data.
        
        Args:
            gathered_data: Currently gathered data
            required_keys: Keys that should be present
            
        Returns:
            True if should continue, False if enough data gathered
        """
        # Check if we have required data
        missing_keys = [key for key in required_keys if key not in gathered_data]
        
        if not missing_keys:
            return False
        
        # Check if we have at least some data
        if len(gathered_data) < 2:
            return True
        
        # If we have most data, might be enough
        return len(missing_keys) > len(required_keys) / 2

