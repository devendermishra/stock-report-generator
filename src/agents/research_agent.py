"""
Research Agent for gathering company information, sector overview, and peer data.
This agent autonomously selects and uses tools to collect comprehensive research data.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio

try:
    # Try relative imports first (when run as module)
    from .base_agent import BaseAgent, AgentState
    from ..tools.stock_data_tool import get_stock_metrics, get_company_info, validate_symbol
    from ..tools.web_search_tool import search_sector_news, search_company_news, search_market_trends
    from ..tools.generic_web_search_tool import generic_web_search, search_web_generic
except ImportError:
    # Fall back to absolute imports (when run as script)
    from agents.base_agent import BaseAgent, AgentState
    from tools.stock_data_tool import get_stock_metrics, get_company_info, validate_symbol
    from tools.web_search_tool import search_sector_news, search_company_news, search_market_trends
    from tools.generic_web_search_tool import generic_web_search, search_web_generic

logger = logging.getLogger(__name__)

class ResearchAgent(BaseAgent):
    """
    Research Agent responsible for gathering comprehensive research data.
    
    **Specialization:** Comprehensive Data Gathering (Structured Workflow Mode)
    
    **Role:** Executes the research plan created by ResearchPlannerAgent to gather
    all necessary data for analysis. This agent follows a structured, plan-driven
    approach rather than iterative decision-making.
    
    **When Used:** Only in Structured Workflow Mode (after ResearchPlannerAgent)
    
    **Tasks:**
    - Executes tool calls in the order specified by research plan
    - Company information and background
    - Sector overview and trends
    - Peer analysis and comparison
    - Market news and developments
    
    **Tools Used:**
    - get_stock_metrics, get_company_info, validate_symbol
    - search_sector_news, search_company_news, search_market_trends
    - generic_web_search, search_web_generic
    
    For detailed information on agent specialization and roles,
    see docs/AGENT_SPECIALIZATION.md
    """
    
    def __init__(self, agent_id: str, openai_api_key: str):
        """
        Initialize the Research Agent.
        
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
            generic_web_search,
            search_web_generic
        ]
        
        super().__init__(agent_id, openai_api_key, available_tools)
        
        # Initialize tool mapping - maps tool names to execution methods
        self.tool_mapping = {
            "get_stock_metrics": self._execute_stock_metrics_tool,
            "get_company_info": self._execute_company_info_tool,
            "validate_symbol": self._execute_validate_symbol_tool,
            "search_sector_news": self._execute_sector_news_tool,
            "search_company_news": self._execute_company_news_tool,
            "search_market_trends": self._execute_market_trends_tool,
            "get_peer_data": self._execute_peer_data_tool,
            "generic_web_search": self._execute_generic_web_search_tool,
            "search_web_generic": self._execute_generic_web_search_tool
        }
        
    async def execute_task(
        self,
        stock_symbol: str,
        company_name: str,
        sector: str,
        context: Dict[str, Any]
    ) -> AgentState:
        """
        Execute research tasks to gather comprehensive company and sector data.
        Can perform targeted research based on critique feedback.
        
        Args:
            stock_symbol: NSE stock symbol
            company_name: Full company name
            sector: Sector name
            context: Current context from previous agents (may include critique feedback)
            
        Returns:
            AgentState with research results
        """
        start_time = datetime.now()
        
        # Check for research plan from ResearchPlannerAgent
        planner_results = context.get("research_planner_agent_results", {})
        research_plan = planner_results.get("research_plan", {})
        tool_calls = research_plan.get("tool_calls", [])
        
        # Check if this is a follow-up research based on critique feedback (legacy support)
        critique_feedback = context.get("research_critique_agent_results", {})
        is_follow_up = bool(critique_feedback)
        has_plan = bool(tool_calls)
        
        if has_plan:
            self.logger.info(f"Starting research for {stock_symbol} ({company_name}) using research plan with {len(tool_calls)} tools")
            task_name = "planned_research"
        elif is_follow_up:
            self.logger.info(f"Starting follow-up research for {stock_symbol} ({company_name}) based on critique feedback")
            task_name = "targeted_research"
        else:
            self.logger.info(f"Starting initial research for {stock_symbol} ({company_name})")
            task_name = "comprehensive_research"
        
        # Initialize state
        state = AgentState(
            agent_id=self.agent_id,
            stock_symbol=stock_symbol,
            company_name=company_name,
            sector=sector,
            current_task=task_name,
            context=context,
            results={},
            tools_used=[],  # Ensure this is always a list
            confidence_score=0.0,
            errors=[],
            start_time=start_time
        )
        
        # Ensure tools_used is always a list
        if not isinstance(state.tools_used, list):
            self.logger.warning(f"tools_used was not a list, converting from {type(state.tools_used)}")
            state.tools_used = []
        
        try:
            # Determine research strategy
            if has_plan:
                # Execute research according to the plan
                research_tasks = await self._execute_research_plan(
                    stock_symbol, company_name, sector, tool_calls
                )
            elif is_follow_up:
                # Execute targeted research based on critique feedback
                research_tasks = await self._get_targeted_research_tasks(
                    stock_symbol, company_name, sector, critique_feedback
                )
            else:
                # Execute comprehensive research for initial pass
                research_tasks = [
                    self._gather_company_data(stock_symbol, company_name),
                    self._gather_sector_data(sector),
                    self._gather_peer_data(stock_symbol, sector),
                    self._gather_market_news(stock_symbol, company_name, sector)
                ]
            
            # Execute research tasks in parallel
            results = await asyncio.gather(*research_tasks, return_exceptions=True)
            
            # Process results
            if has_plan:
                # For planned research, process results from plan execution
                processed_results = self._process_planned_results(results, tool_calls)
            elif is_follow_up:
                # For targeted research, we get a list of results
                processed_results = self._process_targeted_results(results, critique_feedback)
            else:
                # For comprehensive research, we get 4 specific results
                company_data, sector_data, peer_data, news_data = results
                processed_results = {
                    "company_data": company_data,
                    "sector_data": sector_data,
                    "peer_data": peer_data,
                    "news_data": news_data
                }
            
            # Handle exceptions and compile results
            if has_plan:
                # For planned research, use processed results
                # Check if peer data is missing and gather it as fallback
                if not processed_results.get("peer_data") or not processed_results.get("peer_data", {}).get("peers"):
                    self.logger.info("Peer data missing after planned research, gathering peer data as fallback")
                    try:
                        peer_data_result = await self._gather_peer_data(stock_symbol, sector)
                        if isinstance(peer_data_result, dict) and peer_data_result.get("data"):
                            processed_results["peer_data"] = peer_data_result["data"]
                            if isinstance(peer_data_result.get("tools_used"), list):
                                state.tools_used.extend(peer_data_result["tools_used"])
                    except Exception as e:
                        self.logger.warning(f"Failed to gather peer data as fallback: {e}")
                        # Initialize peer_data structure if gathering fails
                        if not processed_results.get("peer_data"):
                            processed_results["peer_data"] = {"peers": {}, "sector": sector, "peer_count": 0}
                
                # Ensure sector_data has sector_name
                if processed_results.get("sector_data") and not processed_results["sector_data"].get("sector_name"):
                    processed_results["sector_data"]["sector_name"] = sector
                
                state.results = processed_results
                # Extract tools used from all results
                for result in results:
                    if not isinstance(result, Exception) and isinstance(result, dict):
                        tools_used = result.get("tools_used", [])
                        if isinstance(tools_used, list):
                            state.tools_used.extend(tools_used)
                        else:
                            self.logger.warning(f"Invalid tools_used type in result: {type(tools_used)}")
            elif is_follow_up:
                # For targeted research, use processed results
                state.results = processed_results
                # Extract tools used from all results
                for result in results:
                    if not isinstance(result, Exception) and isinstance(result, dict):
                        tools_used = result.get("tools_used", [])
                        if isinstance(tools_used, list):
                            state.tools_used.extend(tools_used)
                        else:
                            self.logger.warning(f"Invalid tools_used type in result: {type(tools_used)}")
            else:
                # For comprehensive research, handle each result type
                company_data, sector_data, peer_data, news_data = results
                
                if isinstance(company_data, Exception):
                    state.errors.append(f"Company data gathering failed: {str(company_data)}")
                    company_data = {}
                else:
                    tools_used = company_data.get("tools_used", [])
                    if isinstance(tools_used, list):
                        state.tools_used.extend(tools_used)
                    else:
                        self.logger.warning(f"Invalid tools_used type in company_data: {type(tools_used)}")
                
                if isinstance(sector_data, Exception):
                    state.errors.append(f"Sector data gathering failed: {str(sector_data)}")
                    sector_data = {}
                else:
                    tools_used = sector_data.get("tools_used", [])
                    if isinstance(tools_used, list):
                        state.tools_used.extend(tools_used)
                    else:
                        self.logger.warning(f"Invalid tools_used type in sector_data: {type(tools_used)}")
                
                if isinstance(peer_data, Exception):
                    state.errors.append(f"Peer data gathering failed: {str(peer_data)}")
                    peer_data = {}
                else:
                    tools_used = peer_data.get("tools_used", [])
                    if isinstance(tools_used, list):
                        state.tools_used.extend(tools_used)
                    else:
                        self.logger.warning(f"Invalid tools_used type in peer_data: {type(tools_used)}")
                
                if isinstance(news_data, Exception):
                    state.errors.append(f"News data gathering failed: {str(news_data)}")
                    news_data = {}
                else:
                    tools_used = news_data.get("tools_used", [])
                    if isinstance(tools_used, list):
                        state.tools_used.extend(tools_used)
                    else:
                        self.logger.warning(f"Invalid tools_used type in news_data: {type(tools_used)}")
                
                # Compile results
                state.results = {
                    "company_data": company_data.get("data", {}),
                    "sector_data": sector_data.get("data", {}),
                    "peer_data": peer_data.get("data", {}),
                    "news_data": news_data.get("data", {}),
                    "research_summary": self._generate_research_summary(
                        company_data.get("data", {}),
                        sector_data.get("data", {}),
                        peer_data.get("data", {}),
                        news_data.get("data", {})
                    )
                }
            
            # Attach aggregated tools used into results for downstream agents
            try:
                unique_tools_used = []
                seen = set()
                for t in state.tools_used:
                    if isinstance(t, str) and t not in seen:
                        seen.add(t)
                        unique_tools_used.append(t)
                # Store in results so critique can avoid suggesting duplicates
                state.results["tools_used"] = unique_tools_used
            except Exception as e:
                self.logger.warning(f"Failed to attach tools_used to results: {e}")

            # Calculate confidence score
            try:
                state.confidence_score = self.calculate_confidence_score(
                    state.results, state.tools_used, state.errors
                )
            except Exception as e:
                self.logger.error(f"Failed to calculate confidence score: {e}")
                state.confidence_score = 0.0
            
            # Update context
            state.context = self.update_context(context, state.results, self.agent_id)
            
            state.end_time = datetime.now()
            duration = (state.end_time - state.start_time).total_seconds()
            self.log_execution(state, duration)
            
            return state
            
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            self.logger.error(f"Research execution failed: {e}")
            self.logger.error(f"Stack trace:\n{error_traceback}")
            state.errors.append(f"Research execution failed: {str(e)}")
            state.end_time = datetime.now()
            state.confidence_score = 0.0
            return state
    
    async def _execute_research_plan(
        self,
        stock_symbol: str,
        company_name: str,
        sector: str,
        tool_calls: List[Dict[str, Any]]
    ) -> List[Any]:
        """
        Execute research plan by running tools in the specified order.
        
        Args:
            stock_symbol: NSE stock symbol
            company_name: Full company name
            sector: Sector name
            tool_calls: List of tool calls from research plan (with order, tool_name, parameters)
            
        Returns:
            List of research tasks (coroutines) to execute
        """
        tasks = []
        
        self.logger.info(f"Executing research plan with {len(tool_calls)} tool calls")
        
        # Sort tool calls by order to ensure correct execution sequence
        sorted_tool_calls = sorted(tool_calls, key=lambda x: x.get("order", 0))
        
        for tool_call in sorted_tool_calls:
            tool_name = tool_call.get("tool_name", "")
            parameters = tool_call.get("parameters", {})
            order = tool_call.get("order", 0)
            
            self.logger.info(f"Queuing tool {order}: {tool_name} with parameters {parameters}")
            
            # Create a task that will execute this tool
            tasks.append(
                self._execute_planned_tool(
                    tool_name, parameters, stock_symbol, company_name, sector, order
                )
            )
        
        return tasks
    
    async def _execute_planned_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        stock_symbol: str,
        company_name: str,
        sector: str,
        order: int
    ) -> Dict[str, Any]:
        """
        Execute a single tool from the research plan.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Parameters for the tool
            stock_symbol: NSE stock symbol
            company_name: Full company name
            sector: Sector name
            order: Order number from plan
            
        Returns:
            Dictionary containing tool execution results
        """
        self.logger.info(f"Executing planned tool [{order}]: {tool_name}")
        
        try:
            if tool_name in self.tool_mapping:
                result = await self.tool_mapping[tool_name](parameters, stock_symbol, company_name, sector)
                result["tool_name"] = tool_name
                result["order"] = order
                result["parameters"] = parameters
                return result
            else:
                self.logger.warning(f"Unknown tool name in plan: {tool_name}")
                return {
                    "tool_name": tool_name,
                    "order": order,
                    "data": {},
                    "tools_used": [],
                    "error": f"Tool not found: {tool_name}",
                    "parameters": parameters
                }
                
        except Exception as e:
            self.logger.error(f"Failed to execute planned tool {tool_name}: {e}")
            return {
                "tool_name": tool_name,
                "order": order,
                "data": {},
                "tools_used": [],
                "error": str(e),
                "parameters": parameters
            }
    
    async def _execute_validate_symbol_tool(self, parameters: Dict[str, Any], stock_symbol: str, company_name: str, sector: str) -> Dict[str, Any]:
        """Execute validate_symbol tool."""
        try:
            tool_params = {
                "symbol": parameters.get("symbol", stock_symbol)
            }
            result = validate_symbol.invoke(tool_params)
            return {
                "data": {"validation": result},
                "tools_used": ["validate_symbol"]
            }
        except Exception as e:
            return {"data": {}, "tools_used": [], "error": str(e)}
    
    def _process_planned_results(
        self,
        results: List[Any],
        tool_calls: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Process results from planned research execution.
        
        Args:
            results: List of results from tool executions
            tool_calls: Original tool calls from plan
            
        Returns:
            Processed results dictionary
        """
        try:
            processed_results = {
                "company_data": {},
                "sector_data": {},
                "peer_data": {},
                "news_data": {},
                "planned_research": True,
                "tools_executed": []
            }
            
            # Process each result and categorize it
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.warning(f"Planned tool execution {i} failed: {result}")
                    continue
                
                if not isinstance(result, dict):
                    continue
                
                # Track executed tools
                tool_name = result.get("tool_name", "unknown")
                order = result.get("order", i + 1)
                success = not bool(result.get("error"))
                
                processed_results["tools_executed"].append({
                    "tool_name": tool_name,
                    "order": order,
                    "success": success,
                    "error": result.get("error")
                })
                
                data = result.get("data", {})
                tools_used = result.get("tools_used", [])
                
                # Categorize based on data content
                try:
                    if "company_info" in data or "stock_metrics" in data or "validation" in data:
                        if processed_results["company_data"]:
                            self._safe_merge_dict(processed_results["company_data"], data)
                        else:
                            processed_results["company_data"] = data.copy() if isinstance(data, dict) else {}
                    elif "sector_news" in data or "market_trends" in data:
                        if processed_results["sector_data"]:
                            self._safe_merge_dict(processed_results["sector_data"], data)
                        else:
                            processed_results["sector_data"] = data.copy() if isinstance(data, dict) else {}
                    elif "peers" in data:
                        # peer_data structure should have: peers, sector, peer_count
                        if isinstance(data, dict):
                            processed_results["peer_data"] = data.copy()
                        else:
                            processed_results["peer_data"] = {}
                    elif "company_news" in data or "generic_search_results" in data:
                        if processed_results["news_data"]:
                            self._safe_merge_dict(processed_results["news_data"], data)
                        else:
                            processed_results["news_data"] = data.copy() if isinstance(data, dict) else {}
                except Exception as e:
                    self.logger.error(f"Failed to categorize planned research data: {e}")
            
            # Ensure peer_data has proper structure with sector
            if not processed_results.get("peer_data") or not processed_results["peer_data"].get("peers"):
                self.logger.info("Peer data missing or empty in planned results, initializing structure")
                if not processed_results.get("peer_data"):
                    processed_results["peer_data"] = {}
                # Ensure sector is set even if peers are empty
                if "sector" not in processed_results["peer_data"]:
                    processed_results["peer_data"]["sector"] = "N/A"
            
            # Ensure sector_data has proper structure
            if not processed_results.get("sector_data"):
                processed_results["sector_data"] = {}
            if "sector_name" not in processed_results["sector_data"]:
                processed_results["sector_data"]["sector_name"] = "N/A"
            
            # Generate research summary
            processed_results["research_summary"] = self._generate_planned_research_summary(
                processed_results, tool_calls
            )
            
            return processed_results
            
        except Exception as e:
            self.logger.error(f"Failed to process planned results: {e}")
            return {
                "company_data": {},
                "sector_data": {},
                "peer_data": {},
                "news_data": {},
                "planned_research": True,
                "tools_executed": [],
                "research_summary": "Error processing planned research results",
                "error": str(e)
            }
    
    def _generate_planned_research_summary(
        self,
        results: Dict[str, Any],
        tool_calls: List[Dict[str, Any]]
    ) -> str:
        """Generate a summary for planned research execution."""
        try:
            summary_parts = []
            
            summary_parts.append(
                f"**Planned Research:** Research executed according to structured plan with {len(tool_calls)} tool calls."
            )
            
            tools_executed = results.get("tools_executed", [])
            if tools_executed:
                successful_tools = [tool for tool in tools_executed if tool.get("success")]
                failed_tools = [tool for tool in tools_executed if not tool.get("success")]
                
                summary_parts.append(
                    f"**Execution Results:** {len(tools_executed)} tools executed "
                    f"({len(successful_tools)} successful, {len(failed_tools)} failed)"
                )
                
                if successful_tools:
                    successful_names = [tool.get("tool_name", "unknown") for tool in successful_tools]
                    summary_parts.append(
                        f"**Successful Tools:** {', '.join(successful_names)}"
                    )
            
            # Add summary of what was gathered
            data_sections = ["company_data", "sector_data", "peer_data", "news_data"]
            gathered_sections = [section for section in data_sections if results.get(section)]
            
            if gathered_sections:
                summary_parts.append(
                    f"**Data Gathered:** Successfully collected {len(gathered_sections)} types of data: "
                    f"{', '.join(gathered_sections)}"
                )
            
            return "\n\n".join(summary_parts) if summary_parts else "Planned research completed successfully."
            
        except Exception as e:
            self.logger.error(f"Planned research summary generation failed: {e}")
            return "Planned research completed with some limitations."
    
    async def _gather_company_data(self, stock_symbol: str, company_name: str) -> Dict[str, Any]:
        """Gather comprehensive company data."""
        try:
            self.logger.info(f"Gathering company data for {stock_symbol}")
            tools_used = []
            
            # Validate symbol first
            validation_result = validate_symbol.invoke({"symbol": stock_symbol})
            if not validation_result.get("valid", False):
                return {"data": {}, "tools_used": [], "error": "Invalid symbol"}
            
            tools_used.append("validate_symbol")
            
            # Get company info
            company_info = get_company_info.invoke({"symbol": stock_symbol})
            tools_used.append("get_company_info")
            
            # Get stock metrics
            stock_metrics = get_stock_metrics.invoke({"symbol": stock_symbol})
            tools_used.append("get_stock_metrics")
            
            return {
                "data": {
                    "company_info": company_info,
                    "stock_metrics": stock_metrics,
                    "validation": validation_result
                },
                "tools_used": tools_used
            }
            
        except Exception as e:
            self.logger.error(f"Company data gathering failed: {e}")
            return {"data": {}, "tools_used": [], "error": str(e)}
    
    async def _gather_sector_data(self, sector: str) -> Dict[str, Any]:
        """Gather sector overview and trends."""
        try:
            self.logger.info(f"Gathering sector data for {sector}")
            tools_used = []
            
            # Search for sector news and trends
            sector_news = search_sector_news.invoke({
                "sector": sector,
                "days_back": 30,
                "include_analysis": True
            })
            tools_used.append("search_sector_news")
            
            # Search for market trends
            market_trends = search_market_trends.invoke({
                "query": f"{sector} sector trends India market outlook",
                "max_results": 10
            })
            tools_used.append("search_market_trends")
            
            return {
                "data": {
                    "sector_news": sector_news,
                    "market_trends": market_trends,
                    "sector_name": sector
                },
                "tools_used": tools_used
            }
            
        except Exception as e:
            self.logger.error(f"Sector data gathering failed: {e}")
            return {"data": {}, "tools_used": [], "error": str(e)}
    
    async def _gather_peer_data(self, stock_symbol: str, sector: str) -> Dict[str, Any]:
        """Gather peer company data for comparison. Excludes current stock and ensures 4+ peers."""
        try:
            self.logger.info(f"Gathering peer data for {stock_symbol} in {sector}")
            tools_used = []
            
            # Define peer companies based on sector
            peer_symbols = self._get_peer_symbols(sector)
            
            # Remove current stock from peer list if present
            if stock_symbol in peer_symbols:
                peer_symbols.remove(stock_symbol)
                self.logger.info(f"Excluded current stock {stock_symbol} from peer list")
            
            if not peer_symbols:
                self.logger.warning(f"No peer symbols found for sector {sector}")
                return {
                    "data": {
                        "peers": {},
                        "sector": sector,
                        "peer_count": 0
                    },
                    "tools_used": []
                }
            
            # Gather data for all available peers (need at least 4 for comparison)
            peer_data = {}
            for peer_symbol in peer_symbols:
                try:
                    # Get peer company info
                    peer_info = get_company_info.invoke({"symbol": peer_symbol})
                    
                    # Get peer stock metrics
                    peer_metrics = get_stock_metrics.invoke({"symbol": peer_symbol})
                    
                    # Validate that we got valid data
                    if peer_info and peer_metrics:
                        peer_data[peer_symbol] = {
                            "company_info": peer_info,
                            "stock_metrics": peer_metrics
                        }
                    else:
                        self.logger.warning(f"Incomplete data for peer {peer_symbol}, skipping")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to get data for peer {peer_symbol}: {e}")
                    continue
            
            tools_used.extend(["get_company_info", "get_stock_metrics"])
            
            peer_count = len(peer_data)
            if peer_count < 4:
                self.logger.warning(f"Only gathered {peer_count} peers, need at least 4 for proper comparison")
            
            return {
                "data": {
                    "peers": peer_data,
                    "sector": sector,
                    "peer_count": peer_count
                },
                "tools_used": tools_used
            }
            
        except Exception as e:
            self.logger.error(f"Peer data gathering failed: {e}")
            return {"data": {}, "tools_used": [], "error": str(e)}
    
    async def _gather_market_news(self, stock_symbol: str, company_name: str, sector: str) -> Dict[str, Any]:
        """Gather recent market news and developments."""
        try:
            self.logger.info(f"Gathering market news for {stock_symbol}")
            tools_used = []
            
            # Search for company-specific news
            company_news = search_company_news.invoke({
                "company_name": company_name,
                "stock_symbol": stock_symbol,
                "days_back": 14
            })
            tools_used.append("search_company_news")
            
            # Search for sector news
            sector_news = search_sector_news.invoke({
                "sector": sector,
                "days_back": 14,
                "include_analysis": True
            })
            tools_used.append("search_sector_news")
            
            return {
                "data": {
                    "company_news": company_news,
                    "sector_news": sector_news,
                    "news_period": "14_days"
                },
                "tools_used": tools_used
            }
            
        except Exception as e:
            self.logger.error(f"Market news gathering failed: {e}")
            return {"data": {}, "tools_used": [], "error": str(e)}
    
    def _get_peer_symbols(self, sector: str) -> List[str]:
        """Get peer company symbols based on sector."""
        # Normalize sector name for matching
        sector_normalized = sector.strip()
        
        sector_peers = {
            "Banking": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK"],
            "IT": ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM"],
            "Technology": ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM"],
            "Financial Services": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK"],
            "Pharmaceuticals": ["SUNPHARMA", "DRREDDY", "CIPLA", "BIOCON", "LUPIN"],
            "Auto": ["MARUTI", "TATAMOTORS", "BAJAJ-AUTO", "M&M", "HEROMOTOCO"],
            "Energy": ["RELIANCE", "ONGC", "IOC", "BPCL", "HPCL"],
            "Telecommunications": ["BHARTIARTL", "VODAFONEIDEA", "RCOM"],
            "FMCG": ["HINDUNILVR", "ITC", "NESTLEIND", "DABUR", "MARICO"],
            "Consumer Defensive": ["HINDUNILVR", "ITC", "NESTLEIND", "DABUR", "MARICO"],
            "Consumer Goods": ["HINDUNILVR", "ITC", "NESTLEIND", "DABUR", "MARICO"],
            "Metals": ["TATASTEEL", "JSWSTEEL", "SAIL", "HINDALCO", "NMDC"]
        }
        
        # Try exact match first
        if sector_normalized in sector_peers:
            return sector_peers[sector_normalized]
        
        # Try case-insensitive match
        sector_lower = sector_normalized.lower()
        for key, value in sector_peers.items():
            if key.lower() == sector_lower:
                return value
        
        # Try partial match (e.g., "Consumer Defensive" contains "Consumer")
        for key, value in sector_peers.items():
            if sector_lower in key.lower() or key.lower() in sector_lower:
                self.logger.info(f"Matched sector '{sector}' to '{key}' via partial match")
                return value
        
        # If no match found, return empty list
        self.logger.warning(f"No peer symbols found for sector: {sector}")
        return []
    
    def _generate_research_summary(
        self,
        company_data: Dict[str, Any],
        sector_data: Dict[str, Any],
        peer_data: Dict[str, Any],
        news_data: Dict[str, Any]
    ) -> str:
        """Generate a comprehensive research summary."""
        try:
            summary_parts = []
            
            # Company overview
            if company_data.get("company_info"):
                company_info = company_data["company_info"]
                summary_parts.append(
                    f"**Company Overview:** {company_info.get('company_name', 'N/A')} "
                    f"({company_info.get('symbol', 'N/A')}) operates in the "
                    f"{company_info.get('sector', 'N/A')} sector with "
                    f"{company_info.get('employees', 'N/A')} employees."
                )
            
            # Financial highlights
            if company_data.get("stock_metrics"):
                metrics = company_data["stock_metrics"]
                summary_parts.append(
                    f"**Financial Highlights:** Current price ₹{metrics.get('current_price', 'N/A')}, "
                    f"Market cap ₹{metrics.get('market_cap', 'N/A'):,}, "
                    f"P/E ratio {metrics.get('pe_ratio', 'N/A')}, "
                    f"52-week range ₹{metrics.get('52_week_low', 'N/A')} - ₹{metrics.get('52_week_high', 'N/A')}."
                )
            
            # Sector trends
            if sector_data.get("sector_news"):
                news_count = len(sector_data["sector_news"].get("results", []))
                summary_parts.append(
                    f"**Sector Analysis:** {news_count} recent news articles analyzed "
                    f"for {sector_data.get('sector_name', 'the sector')} trends and developments."
                )
            
            # Peer comparison
            if peer_data.get("peers"):
                peer_count = peer_data.get("peer_count", 0)
                summary_parts.append(
                    f"**Peer Analysis:** Compared against {peer_count} peer companies "
                    f"in the same sector for relative performance assessment."
                )
            
            # News summary
            if news_data.get("company_news"):
                company_news_count = len(news_data["company_news"].get("results", []))
                summary_parts.append(
                    f"**Recent Developments:** {company_news_count} company-specific news articles "
                    f"analyzed for recent developments and announcements."
                )
            
            return "\n\n".join(summary_parts) if summary_parts else "Research data collected successfully."
            
        except Exception as e:
            self.logger.error(f"Research summary generation failed: {e}")
            return "Research data collected with some limitations."
    
    async def _get_targeted_research_tasks(
        self,
        stock_symbol: str,
        company_name: str,
        sector: str,
        critique_feedback: Dict[str, Any]
    ) -> List[Any]:
        """
        Get targeted research tasks based on critique feedback with actionable tools.
        
        Args:
            stock_symbol: NSE stock symbol
            company_name: Full company name
            sector: Sector name
            critique_feedback: Feedback from ResearchCritique agent
            
        Returns:
            List of research tasks to execute
        """
        tasks = []
        actionable_tools = critique_feedback.get("actionable_tools", [])
        missing_data = critique_feedback.get("missing_data", [])
        recommendations = critique_feedback.get("recommendations", [])
        
        self.logger.info(f"Planning targeted research based on critique feedback")
        self.logger.info(f"Actionable tools: {len(actionable_tools)}")
        self.logger.info(f"Missing data: {missing_data}")
        self.logger.info(f"Recommendations: {recommendations}")
        
        # Process actionable tools first (highest priority)
        if actionable_tools:
            self.logger.info(f"Processing {len(actionable_tools)} actionable tools from critique feedback")
            # Sort by priority (high first)
            sorted_tools = sorted(actionable_tools, key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x.get("priority", "medium"), 1))
            
            for i, tool_call in enumerate(sorted_tools, 1):
                self.logger.info(f"Queuing tool {i}: {tool_call.get('tool_name', 'unknown')} with parameters {tool_call.get('parameters', {})}")
                # Queue the tool execution coroutine for asyncio.gather
                tasks.append(
                    self._execute_actionable_tool(
                        tool_call, stock_symbol, company_name, sector
                    )
                )
        else:
            self.logger.info("No actionable tools found in critique feedback")
        
        # Fallback to keyword-based analysis if no actionable tools
        if not tasks:
            self.logger.info("No actionable tools found, using keyword-based analysis")
            tasks = self._get_keyword_based_tasks(
                stock_symbol, company_name, sector, missing_data
            )
        
        # If still no tasks, do comprehensive research
        if not tasks:
            self.logger.info("No specific areas identified, performing comprehensive research")
            tasks = [
                self._gather_company_data(stock_symbol, company_name),
                self._gather_sector_data(sector),
                self._gather_peer_data(stock_symbol, sector),
                self._gather_market_news(stock_symbol, company_name, sector)
            ]
        
        return tasks
    
    async def _execute_actionable_tool(
        self,
        tool_call: Dict[str, Any],
        stock_symbol: str,
        company_name: str,
        sector: str
    ) -> Dict[str, Any]:
        """
        Execute an actionable tool call directly.
        
        Args:
            tool_call: Tool call specification from critique
            stock_symbol: NSE stock symbol
            company_name: Full company name
            sector: Sector name
            
        Returns:
            Dictionary containing tool execution results
        """
        tool_name = tool_call.get("tool_name", "")
        parameters = tool_call.get("parameters", {})
        reason = tool_call.get("reason", "Data improvement needed")
        priority = tool_call.get("priority", "medium")
        
        self.logger.info(f"Executing actionable tool: {tool_name} with parameters: {parameters}")
        
        try:
            if tool_name in self.tool_mapping:
                result = await self.tool_mapping[tool_name](parameters, stock_symbol, company_name, sector)
                result["tool_name"] = tool_name
                result["reason"] = reason
                result["priority"] = priority
                result["parameters"] = parameters
                result["is_available_tool"] = True
                return result
            else:
                self.logger.warning(f"Unknown tool name: {tool_name} - adding to future tool recommendations")
                return {
                    "tool_name": tool_name,
                    "data": {},
                    "tools_used": [],
                    "error": f"Tool not available: {tool_name}",
                    "reason": reason,
                    "priority": priority,
                    "parameters": parameters,
                    "is_available_tool": False,
                    "is_future_tool": True
                }
                
        except Exception as e:
            self.logger.error(f"Failed to execute tool {tool_name}: {e}")
            return {
                "tool_name": tool_name,
                "data": {},
                "tools_used": [],
                "error": str(e),
                "reason": reason,
                "priority": priority,
                "parameters": parameters
            }
    
    async def _execute_stock_metrics_tool(self, parameters: Dict[str, Any], stock_symbol: str, company_name: str, sector: str) -> Dict[str, Any]:
        """Execute get_stock_metrics tool."""
        try:
            # Ensure required parameters are present
            tool_params = {
                "symbol": parameters.get("symbol", stock_symbol)
            }
            result = get_stock_metrics.invoke(tool_params)
            return {
                "data": {"stock_metrics": result},
                "tools_used": ["get_stock_metrics"]
            }
        except Exception as e:
            return {"data": {}, "tools_used": [], "error": str(e)}
    
    async def _execute_company_info_tool(self, parameters: Dict[str, Any], stock_symbol: str, company_name: str, sector: str) -> Dict[str, Any]:
        """Execute get_company_info tool."""
        try:
            # Ensure required parameters are present
            tool_params = {
                "symbol": parameters.get("symbol", stock_symbol)
            }
            result = get_company_info.invoke(tool_params)
            return {
                "data": {"company_info": result},
                "tools_used": ["get_company_info"]
            }
        except Exception as e:
            return {"data": {}, "tools_used": [], "error": str(e)}
    
    async def _execute_sector_news_tool(self, parameters: Dict[str, Any], stock_symbol: str, company_name: str, sector: str) -> Dict[str, Any]:
        """Execute search_sector_news tool."""
        try:
            # Ensure required parameters are present
            tool_params = {
                "sector": parameters.get("sector", sector),
                "days_back": parameters.get("days_back", 30),
                "include_analysis": parameters.get("include_analysis", True)
            }
            result = search_sector_news.invoke(tool_params)
            return {
                "data": {"sector_news": result},
                "tools_used": ["search_sector_news"]
            }
        except Exception as e:
            return {"data": {}, "tools_used": [], "error": str(e)}
    
    async def _execute_company_news_tool(self, parameters: Dict[str, Any], stock_symbol: str, company_name: str, sector: str) -> Dict[str, Any]:
        """Execute search_company_news tool."""
        try:
            # Ensure required parameters are present
            tool_params = {
                "company_name": parameters.get("company_name", company_name),
                "stock_symbol": parameters.get("stock_symbol", stock_symbol),
                "days_back": parameters.get("days_back", 14)
            }
            result = search_company_news.invoke(tool_params)
            return {
                "data": {"company_news": result},
                "tools_used": ["search_company_news"]
            }
        except Exception as e:
            return {"data": {}, "tools_used": [], "error": str(e)}
    
    async def _execute_market_trends_tool(self, parameters: Dict[str, Any], stock_symbol: str, company_name: str, sector: str) -> Dict[str, Any]:
        """Execute search_market_trends tool."""
        try:
            # Ensure required parameters are present
            tool_params = {
                "query": parameters.get("query", f"{sector} sector trends India market outlook"),
                "max_results": parameters.get("max_results", 10)
            }
            result = search_market_trends.invoke(tool_params)
            return {
                "data": {"market_trends": result},
                "tools_used": ["search_market_trends"]
            }
        except Exception as e:
            return {"data": {}, "tools_used": [], "error": str(e)}
    
    async def _execute_peer_data_tool(self, parameters: Dict[str, Any], stock_symbol: str, company_name: str, sector: str) -> Dict[str, Any]:
        """Execute get_peer_data tool."""
        try:
            # Use the existing _gather_peer_data method
            result = await self._gather_peer_data(stock_symbol, sector)
            return {
                "data": result.get("data", {}),
                "tools_used": result.get("tools_used", [])
            }
        except Exception as e:
            return {"data": {}, "tools_used": [], "error": str(e)}
    
    async def _execute_generic_web_search_tool(self, parameters: Dict[str, Any], stock_symbol: str, company_name: str, sector: str) -> Dict[str, Any]:
        """Execute generic_web_search tool."""
        try:
            # Ensure required parameters are present
            tool_params = {
                "query": parameters.get("query", f"{company_name} {stock_symbol} {sector}"),
                "search_type": parameters.get("search_type", "web"),
                "max_results": parameters.get("max_results", 10),
                "filters": parameters.get("filters", {}),
                "format_output": parameters.get("format_output", True),
                "include_metadata": parameters.get("include_metadata", True),
                "language": parameters.get("language", "en"),
                "region": parameters.get("region", "us"),
                "time_range": parameters.get("time_range", "all"),
                "site_filter": parameters.get("site_filter", ""),
                "file_type": parameters.get("file_type", "")
            }
            
            result = generic_web_search.invoke(tool_params)
            return {
                "data": {"generic_search_results": result},
                "tools_used": ["generic_web_search"]
            }
        except Exception as e:
            return {"data": {}, "tools_used": [], "error": str(e)}
    
    def _get_keyword_based_tasks(
        self,
        stock_symbol: str,
        company_name: str,
        sector: str,
        missing_data: List[str]
    ) -> List[Any]:
        """
        Get research tasks based on keyword analysis of missing data.
        
        Args:
            stock_symbol: NSE stock symbol
            company_name: Full company name
            sector: Sector name
            missing_data: List of missing data points
            
        Returns:
            List of research tasks
        """
        tasks = []
        
        # Determine which research areas need attention
        needs_company_data = any(keyword in str(missing_data).lower() for keyword in [
            "company", "financial", "metrics", "price", "market cap", "pe ratio"
        ])
        needs_sector_data = any(keyword in str(missing_data).lower() for keyword in [
            "sector", "industry", "trends", "outlook"
        ])
        needs_peer_data = any(keyword in str(missing_data).lower() for keyword in [
            "peer", "competitor", "comparison", "benchmark"
        ])
        needs_news_data = any(keyword in str(missing_data).lower() for keyword in [
            "news", "recent", "developments", "announcements"
        ])
        
        # Add targeted tasks based on what's missing
        if needs_company_data:
            tasks.append(self._gather_company_data(stock_symbol, company_name))
            self.logger.info("Added company data gathering task")
        
        if needs_sector_data:
            tasks.append(self._gather_sector_data(sector))
            self.logger.info("Added sector data gathering task")
        
        if needs_peer_data:
            tasks.append(self._gather_peer_data(stock_symbol, sector))
            self.logger.info("Added peer data gathering task")
        
        if needs_news_data:
            tasks.append(self._gather_market_news(stock_symbol, company_name, sector))
            self.logger.info("Added news data gathering task")
        
        return tasks
    
    def _process_targeted_results(
        self,
        results: List[Any],
        critique_feedback: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process results from targeted research tasks (now direct tool executions).
        
        Args:
            results: List of results from direct tool executions
            critique_feedback: Original critique feedback
            
        Returns:
            Processed results dictionary
        """
        try:
            processed_results = {
                "company_data": {},
                "sector_data": {},
                "peer_data": {},
                "news_data": {},
                "targeted_research": True,
                "critique_feedback": critique_feedback,
                "actionable_tools_executed": [],
                "future_tool_recommendations": []
            }
            
            # Process each result and categorize it
            for result in results:
                if isinstance(result, Exception):
                    self.logger.warning(f"Research task failed: {result}")
                    continue
                
                if not isinstance(result, dict):
                    continue
                
                # Track executed actionable tools and future tool recommendations
                if result.get("tool_name"):
                    tool_info = {
                        "tool_name": str(result.get("tool_name", "")),
                        "priority": str(result.get("priority", "medium")),
                        "reason": str(result.get("reason", "")),
                        "success": not bool(result.get("error")),
                        "is_available_tool": result.get("is_available_tool", True)
                    }
                    
                    # Categorize tools based on availability
                    if result.get("is_future_tool", False):
                        # Add to future tool recommendations
                        future_tool_info = {
                            "tool_name": str(result.get("tool_name", "")),
                            "priority": str(result.get("priority", "medium")),
                            "reason": str(result.get("reason", "")),
                            "parameters": result.get("parameters", {}),
                            "suggested_implementation": self._generate_tool_implementation_suggestion(
                                result.get("tool_name", ""), 
                                result.get("parameters", {})
                            )
                        }
                        processed_results["future_tool_recommendations"].append(future_tool_info)
                        self.logger.info(f"Added {result.get('tool_name')} to future tool recommendations")
                    else:
                        # Add to executed tools
                        processed_results["actionable_tools_executed"].append(tool_info)
                
                data = result.get("data", {})
                tools_used = result.get("tools_used", [])
                
                # Categorize based on data content
                try:
                    if "company_info" in data or "stock_metrics" in data:
                        # Merge with existing company_data safely
                        if processed_results["company_data"]:
                            self._safe_merge_dict(processed_results["company_data"], data)
                        else:
                            processed_results["company_data"] = data.copy() if isinstance(data, dict) else {}
                    elif "sector_news" in data or "market_trends" in data:
                        # Merge with existing sector_data safely
                        if processed_results["sector_data"]:
                            self._safe_merge_dict(processed_results["sector_data"], data)
                        else:
                            processed_results["sector_data"] = data.copy() if isinstance(data, dict) else {}
                    elif "peers" in data:
                        processed_results["peer_data"] = data.copy() if isinstance(data, dict) else {}
                    elif "company_news" in data:
                        # Merge with existing news_data safely
                        if processed_results["news_data"]:
                            self._safe_merge_dict(processed_results["news_data"], data)
                        else:
                            processed_results["news_data"] = data.copy() if isinstance(data, dict) else {}
                except Exception as e:
                    self.logger.error(f"Failed to categorize data: {e}")
                    self.logger.error(f"Data type: {type(data)}, Data content: {data}")
                    # Continue processing other results
        
            # Generate targeted research summary
            processed_results["research_summary"] = self._generate_targeted_research_summary(
                processed_results, critique_feedback
            )
            
            return processed_results
            
        except Exception as e:
            self.logger.error(f"Failed to process targeted results: {e}")
            self.logger.error(f"Results type: {type(results)}, Results content: {results}")
            # Return minimal results to prevent complete failure
            return {
                "company_data": {},
                "sector_data": {},
                "peer_data": {},
                "news_data": {},
                "targeted_research": True,
                "critique_feedback": critique_feedback,
                "actionable_tools_executed": [],
                "research_summary": "Error processing results",
                "error": str(e)
            }
    
    def _safe_merge_dict(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Safely merge source dictionary into target dictionary.
        Handles nested dictionaries and prevents unhashable type errors.
        
        Args:
            target: Target dictionary to merge into
            source: Source dictionary to merge from
        """
        try:
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    # Recursively merge nested dictionaries
                    self._safe_merge_dict(target[key], value)
                else:
                    # Replace or add the value
                    target[key] = value
        except Exception as e:
            self.logger.error(f"Failed to merge dictionaries: {e}")
            # Fallback to simple update
            try:
                target.update(source)
            except Exception as e2:
                self.logger.error(f"Fallback merge also failed: {e2}")
    
    def _generate_tool_implementation_suggestion(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """
        Generate implementation suggestions for future tools.
        
        Args:
            tool_name: Name of the tool that needs to be implemented
            parameters: Parameters that the tool was called with
            
        Returns:
            Implementation suggestion string
        """
        try:
            # Analyze tool name and parameters to suggest implementation
            suggestions = []
            
            # Common tool patterns
            if "news" in tool_name.lower():
                suggestions.append("Consider implementing a news search tool using web scraping or news APIs")
                suggestions.append("Use existing search_sector_news or search_company_news as reference")
            elif "data" in tool_name.lower():
                suggestions.append("Consider implementing a data retrieval tool using existing stock data tools as reference")
                suggestions.append("Use get_stock_metrics or get_company_info as implementation patterns")
            elif "analysis" in tool_name.lower():
                suggestions.append("Consider implementing an analysis tool using financial calculation libraries")
                suggestions.append("Use existing analysis methods in AnalysisAgent as reference")
            elif "peer" in tool_name.lower():
                suggestions.append("Consider implementing a peer analysis tool using sector-based company grouping")
                suggestions.append("Use existing _gather_peer_data method as reference")
            else:
                suggestions.append("Consider implementing this tool based on the required functionality")
                suggestions.append("Use existing tools in the research agent as implementation patterns")
            
            # Add parameter-specific suggestions
            if parameters:
                param_suggestions = []
                for param, value in parameters.items():
                    if isinstance(value, str) and len(value) > 10:
                        param_suggestions.append(f"Parameter '{param}' expects string input")
                    elif isinstance(value, (int, float)):
                        param_suggestions.append(f"Parameter '{param}' expects numeric input")
                    elif isinstance(value, bool):
                        param_suggestions.append(f"Parameter '{param}' expects boolean input")
                
                if param_suggestions:
                    suggestions.extend(param_suggestions)
            
            return " | ".join(suggestions)
            
        except Exception as e:
            self.logger.error(f"Failed to generate tool implementation suggestion: {e}")
            return f"Tool '{tool_name}' needs to be implemented with parameters: {parameters}"

    def _generate_targeted_research_summary(
        self,
        results: Dict[str, Any],
        critique_feedback: Dict[str, Any]
    ) -> str:
        """Generate a summary for targeted research based on critique feedback."""
        try:
            summary_parts = []
            
            # Add context about the targeted research
            summary_parts.append(
                f"**Targeted Research:** This research was conducted based on critique feedback "
                f"to address specific data gaps and quality issues."
            )
            
            # Add information about actionable tools executed
            executed_tools = results.get("actionable_tools_executed", [])
            if executed_tools:
                successful_tools = [tool for tool in executed_tools if tool.get("success")]
                failed_tools = [tool for tool in executed_tools if not tool.get("success")]
                
                summary_parts.append(
                    f"**Actionable Tools Executed:** {len(executed_tools)} tools executed "
                    f"({len(successful_tools)} successful, {len(failed_tools)} failed)"
                )
                
                if successful_tools:
                    successful_names = [tool.get("tool_name", "unknown") for tool in successful_tools]
                    summary_parts.append(
                        f"**Successful Tools:** {', '.join(successful_names)}"
                    )
                
                if failed_tools:
                    failed_names = [tool.get("tool_name", "unknown") for tool in failed_tools]
                    summary_parts.append(
                        f"**Failed Tools:** {', '.join(failed_names)}"
                    )
                
                # Add high priority tools
                high_priority_tools = [tool for tool in executed_tools if tool.get("priority") == "high"]
                if high_priority_tools:
                    high_priority_names = [tool.get("tool_name", "unknown") for tool in high_priority_tools]
                    summary_parts.append(
                        f"**High Priority Actions:** {', '.join(high_priority_names)}"
                    )
            
            # Add information about what was addressed
            missing_data = critique_feedback.get("missing_data", [])
            if missing_data:
                summary_parts.append(
                    f"**Addressed Issues:** Focused on gathering missing data: {', '.join(missing_data[:5])}"
                )
            
            # Add recommendations that were followed
            recommendations = critique_feedback.get("recommendations", [])
            if recommendations:
                summary_parts.append(
                    f"**Recommendations Followed:** {', '.join(recommendations[:3])}"
                )
            
            # Add information about future tool recommendations
            future_tools = results.get("future_tool_recommendations", [])
            if future_tools:
                future_tool_names = [tool.get("tool_name", "unknown") for tool in future_tools]
                summary_parts.append(
                    f"**Future Tool Recommendations:** {len(future_tools)} tools identified for future implementation: "
                    f"{', '.join(future_tool_names)}"
                )
                
                # Add high priority future tools
                high_priority_future = [tool for tool in future_tools if tool.get("priority") == "high"]
                if high_priority_future:
                    high_priority_future_names = [tool.get("tool_name", "unknown") for tool in high_priority_future]
                    summary_parts.append(
                        f"**High Priority Future Tools:** {', '.join(high_priority_future_names)}"
                    )
            
            # Add data quality improvements
            completeness_score = critique_feedback.get("completeness_score", 0.0)
            quality_score = critique_feedback.get("quality_score", 0.0)
            summary_parts.append(
                f"**Quality Improvement:** Previous scores - Completeness: {completeness_score:.2f}, "
                f"Quality: {quality_score:.2f}. Additional data gathered to improve these scores."
            )
            
            # Add summary of what was gathered
            data_sections = ["company_data", "sector_data", "peer_data", "news_data"]
            gathered_sections = [section for section in data_sections if results.get(section)]
            
            if gathered_sections:
                summary_parts.append(
                    f"**Data Gathered:** Successfully collected {len(gathered_sections)} types of data: "
                    f"{', '.join(gathered_sections)}"
                )
            
            return "\n\n".join(summary_parts) if summary_parts else "Targeted research completed successfully."
            
        except Exception as e:
            self.logger.error(f"Targeted research summary generation failed: {e}")
            return "Targeted research completed with some limitations."
