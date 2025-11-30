"""
Research Planner Agent for creating structured research plans.
This agent uses OpenAI to generate an ordered sequence of tool calls
needed to gather comprehensive research data for stock analysis.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

try:
    # Try relative imports first (when run as module)
    from .base_agent import BaseAgent, AgentState
    from ..config import Config
except ImportError:
    # Fall back to absolute imports (when run as script)
    from agents.base_agent import BaseAgent, AgentState
    from config import Config

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

class ResearchPlannerAgent(BaseAgent):
    """
    Research Planner Agent responsible for creating structured research plans.
    
    Tasks:
    - Analyze company, sector, and country context
    - Review available tools
    - Generate an ordered research plan with specific tool calls
    """
    
    def __init__(self, agent_id: str, openai_api_key: str):
        """
        Initialize the Research Planner Agent.
        
        Args:
            agent_id: Unique identifier for the agent
            openai_api_key: OpenAI API key for LLM calls
        """
        # Research planner doesn't need external tools, only OpenAI
        super().__init__(agent_id, openai_api_key, [])
        
        # Initialize OpenAI client
        self.openai_client = AsyncOpenAI(api_key=openai_api_key)
        
    async def execute_task(
        self,
        stock_symbol: str,
        company_name: str,
        sector: str,
        context: Dict[str, Any]
    ) -> AgentState:
        """
        Execute research planning to create an ordered tool execution plan.
        
        Args:
            stock_symbol: Stock symbol
            company_name: Full company name
            sector: Sector name
            context: Current context (may include country and available_tools)
            
        Returns:
            AgentState with research plan results
        """
        start_time = datetime.now()
        self.logger.info(f"Starting research planning for {company_name} ({stock_symbol})")
        
        # Initialize state
        state = AgentState(
            agent_id=self.agent_id,
            stock_symbol=stock_symbol,
            company_name=company_name,
            sector=sector,
            current_task="research_planning",
            context=context,
            results={},
            tools_used=[],
            confidence_score=0.0,
            errors=[],
            start_time=start_time
        )
        
        try:
            # Extract country and available tools from context
            country = context.get("country", "India")
            available_tools = context.get("available_tools", [])
            
            # Exclude validate_symbol from available_tools (already executed before planning)
            filtered_tools = []
            for tool in available_tools:
                tool_name = None
                if hasattr(tool, 'name'):
                    tool_name = tool.name
                elif hasattr(tool, '__name__'):
                    tool_name = tool.__name__
                elif isinstance(tool, dict):
                    tool_name = tool.get("tool_name") or tool.get("name")
                
                if tool_name != "validate_symbol":
                    filtered_tools.append(tool)
            
            available_tools = filtered_tools
            
            # Generate research plan
            research_plan = await self._generate_research_plan(
                stock_symbol, company_name, sector, country, available_tools
            )
            
            # Update state with plan results
            state.results = research_plan
            state.confidence_score = research_plan.get("confidence_score", 0.8)
            
            # Update context with plan results
            state.context = self.update_context(context, state.results, self.agent_id)
            
            state.end_time = datetime.now()
            duration = (state.end_time - state.start_time).total_seconds()
            self.log_execution(state, duration)
            
            return state
            
        except Exception as e:
            self.logger.error(f"Research planning execution failed: {e}")
            state.errors.append(f"Research planning execution failed: {str(e)}")
            state.end_time = datetime.now()
            state.confidence_score = 0.0
            state.results = {
                "planning_status": "failed",
                "error": f"Planning failed: {str(e)}",
                "research_plan": {
                    "tool_calls": [],
                    "total_steps": 0
                }
            }
            return state
    
    async def _generate_research_plan(
        self,
        stock_symbol: str,
        company_name: str,
        sector: str,
        country: str,
        available_tools: List[Any]
    ) -> Dict[str, Any]:
        """
        Use OpenAI to generate a structured research plan with ordered tool calls.
        
        Args:
            stock_symbol: Stock symbol
            company_name: Full company name
            sector: Sector name
            country: Country name
            available_tools: List of available tool objects or tool descriptions
            
        Returns:
            Dictionary containing research plan with ordered tool calls
        """
        try:
            # Prepare the planning prompt
            planning_prompt = self._build_planning_prompt(
                stock_symbol, company_name, sector, country, available_tools
            )
            
            # Call OpenAI for planning with logging
            try:
                from ..tools.openai_call_wrapper import logged_async_chat_completion
            except ImportError:
                from tools.openai_call_wrapper import logged_async_chat_completion
            
            response = await logged_async_chat_completion(
                client=self.openai_client,
                model=Config.DEFAULT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert financial research strategist. Your job is to create comprehensive, ordered research plans for stock analysis. Generate specific, executable tool call sequences."
                    },
                    {
                        "role": "user",
                        "content": planning_prompt
                    }
                ],
                temperature=0.2,
                max_tokens=3000,
                agent_name="ResearchPlannerAgent"
            )
            
            # Parse the response
            plan_text = response.choices[0].message.content
            plan_data = self._parse_plan_response(plan_text)
            
            # Validate and enhance the plan
            validated_plan = self._validate_research_plan(plan_data, available_tools)
            
            # Log plan details
            tool_calls = validated_plan.get("tool_calls", [])
            self.logger.info(f"Generated research plan with {len(tool_calls)} tool calls")
            for i, tool_call in enumerate(tool_calls, 1):
                self.logger.info(f"Step {i}: {tool_call.get('tool_name', 'unknown')} (order: {tool_call.get('order', i)})")
            
            return {
                "planning_status": "completed",
                "plan_text": plan_text,
                "research_plan": validated_plan,
                "confidence_score": validated_plan.get("confidence_score", 0.8)
            }
            
        except Exception as e:
            self.logger.error(f"OpenAI planning failed: {e}")
            return {
                "planning_status": "failed",
                "error": f"Planning analysis failed: {str(e)}",
                "research_plan": {
                    "tool_calls": [],
                    "total_steps": 0,
                    "rationale": "Failed to generate plan"
                },
                "confidence_score": 0.0
            }
    
    def _build_planning_prompt(
        self,
        stock_symbol: str,
        company_name: str,
        sector: str,
        country: str,
        available_tools: List[Any]
    ) -> str:
        """
        Build the planning prompt for OpenAI.
        
        Args:
            stock_symbol: Stock symbol
            company_name: Full company name
            sector: Sector name
            country: Country name
            available_tools: List of available tools
            
        Returns:
            Formatted prompt string
        """
        # Extract tool information
        tools_info = self._extract_tools_info(available_tools)
        
        prompt = f"""Create a research plan for {company_name} ({stock_symbol}) in {sector} sector, {country}.

<available_tools>
{json.dumps(tools_info, indent=2)}
</available_tools>

Generate an ordered sequence of tool calls to gather:
- Company fundamentals (info, metrics, business overview)
- Market data (stock metrics, price data)
- Sector analysis (trends, positioning)
- News & sentiment (company and sector news)
- Peer comparison data

Output JSON format:
{{
    "tool_calls": [
        {{
            "order": <integer starting from 1>,
            "tool_name": "<exact tool name from available_tools>",
            "parameters": {{
                "<param_name>": "<param_value>"
            }}
        }}
    ]
}}

Requirements:
- Use ONLY tools from <available_tools> with exact tool names (case-sensitive)
- Order logically: prerequisites first, respect dependencies
- Include all essential tools for comprehensive research
- Parameters: symbol="{stock_symbol}", company="{company_name}", sector="{sector}", country="{country}"
- For news: use days_back=7 for recent news
- Construct relevant search queries based on company/sector context
"""
        return prompt
    
    def _extract_tools_info(self, available_tools: List[Any]) -> List[Dict[str, Any]]:
        """
        Extract essential information from available tools for the prompt.
        Only includes: tool_name, description, and clean parameter schema.
        Excludes validate_symbol as it's already executed before planning.
        
        Args:
            available_tools: List of tool objects or dictionaries
            
        Returns:
            List of tool information dictionaries with only essential fields
        """
        tools_info = []
        
        for tool in available_tools:
            # Skip validate_symbol as it's already executed before planning
            tool_name = None
            if hasattr(tool, 'name'):
                tool_name = tool.name
            elif hasattr(tool, '__name__'):
                tool_name = tool.__name__
            elif isinstance(tool, dict):
                tool_name = tool.get("tool_name") or tool.get("name")
            
            if tool_name == "validate_symbol":
                continue
            
            tool_info = {}
            
            # Extract tool name
            if hasattr(tool, 'name'):
                tool_info["tool_name"] = tool.name
            elif hasattr(tool, '__name__'):
                tool_info["tool_name"] = tool.__name__
            elif isinstance(tool, dict):
                tool_info["tool_name"] = tool.get("tool_name") or tool.get("name", "unknown")
            else:
                tool_str = str(tool)
                tool_info["tool_name"] = tool_str.split('.')[-1] if '.' in tool_str else tool_str
            
            # Extract description
            if hasattr(tool, 'description'):
                tool_info["description"] = tool.description
            elif hasattr(tool, '__doc__') and tool.__doc__:
                # Extract first line of docstring as description
                doc_lines = tool.__doc__.strip().split('\n')
                tool_info["description"] = doc_lines[0] if doc_lines else ''
            elif isinstance(tool, dict):
                tool_info["description"] = tool.get("description", "")
            else:
                tool_info["description"] = ""
            
            # Extract clean parameter schema (only names and types, no implementation details)
            params = {}
            
            # Handle LangChain tools - extract from args_schema if available
            if hasattr(tool, 'args_schema') and tool.args_schema:
                try:
                    schema = tool.args_schema
                    if hasattr(schema, 'schema') or hasattr(schema, 'model_fields'):
                        # Pydantic model schema
                        if hasattr(schema, 'model_fields'):
                            for field_name, field_info in schema.model_fields.items():
                                # Extract type annotation
                                annotation = field_info.annotation if hasattr(field_info, 'annotation') else 'Any'
                                type_str = str(annotation).replace('typing.', '').replace('typing_extensions.', '')
                                # Simplify Optional types
                                if 'Optional' in type_str or 'Union' in type_str:
                                    # Extract base type from Optional/Union
                                    base_type = type_str.split('[')[-1].split(',')[0].replace(']', '').strip()
                                    params[field_name] = base_type if base_type else 'Any'
                                else:
                                    params[field_name] = type_str
                except Exception as e:
                    self.logger.debug(f"Could not extract schema from args_schema: {e}")
            
            # Fallback to function annotations if no args_schema or if args_schema didn't work
            if not params and hasattr(tool, '__annotations__'):
                for param, param_type in tool.__annotations__.items():
                    if param != 'return':
                        type_str = str(param_type)
                        # Skip Annotated types with internal schemas (they contain implementation details)
                        if 'Annotated' in type_str and ('ArgsSchema' in type_str or 'SkipValidation' in type_str):
                            continue
                        # Simplify type strings
                        type_str = type_str.replace('typing.', '').replace('typing_extensions.', '')
                        if 'Optional' in type_str:
                            # Extract base type from Optional[Type]
                            base_type = type_str.split('[')[-1].split(',')[0].replace(']', '').replace('?', '').strip()
                            params[param] = base_type if base_type else 'Any'
                        elif 'Union' in type_str and 'None' in type_str:
                            # Handle Union[Type, None] as Optional
                            parts = type_str.replace('Union[', '').replace(']', '').split(',')
                            base_type = [p.strip() for p in parts if 'None' not in p][0] if parts else 'Any'
                            params[param] = base_type
                        else:
                            params[param] = type_str
            
            # Handle dict-based tools (check separately from function annotations)
            if not params and isinstance(tool, dict) and "parameters" in tool:
                tool_params = tool.get("parameters", {})
                if isinstance(tool_params, dict):
                    for param_name, param_info in tool_params.items():
                        if isinstance(param_info, dict):
                            param_type = param_info.get("type", param_info.get("annotation", "Any"))
                            params[param_name] = str(param_type).replace('typing.', '')
                        else:
                            params[param_name] = str(param_info).replace('typing.', '')
            
            tool_info["parameters"] = params
            tools_info.append(tool_info)
        
        return tools_info
    
    def _parse_plan_response(self, plan_text: str) -> Dict[str, Any]:
        """Parse the OpenAI response to extract structured plan data."""
        try:
            # Try to extract JSON from the response
            import re
            
            # Look for JSON in the response
            json_match = re.search(r'\{.*\}', plan_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed_data = json.loads(json_str)
                return parsed_data
            else:
                # Fallback: create a basic plan structure
                self.logger.warning("Could not find JSON in plan response, using fallback")
                return self._parse_text_plan(plan_text)
                
        except Exception as e:
            self.logger.warning(f"Failed to parse plan response: {e}")
            return self._parse_text_plan(plan_text)
    
    def _parse_text_plan(self, plan_text: str) -> Dict[str, Any]:
        """Parse text-based plan response as fallback."""
        return {
            "tool_calls": []
        }
    
    def _validate_research_plan(
        self,
        plan_data: Dict[str, Any],
        available_tools: List[Any]
    ) -> Dict[str, Any]:
        """
        Validate and enhance the research plan.
        
        Args:
            plan_data: Parsed plan data from OpenAI
            available_tools: List of available tools
            
        Returns:
            Validated and enhanced plan
        """
        # Extract tool names from available tools (excluding validate_symbol)
        available_tool_names = set()
        for tool in available_tools:
            tool_name = None
            if hasattr(tool, 'name'):
                tool_name = tool.name
            elif hasattr(tool, '__name__'):
                tool_name = tool.__name__
            elif isinstance(tool, dict) and "tool_name" in tool:
                tool_name = tool["tool_name"]
            
            # Skip validate_symbol as it's already executed before planning
            if tool_name and tool_name != "validate_symbol":
                available_tool_names.add(tool_name)
        
        # Validate tool calls
        tool_calls = plan_data.get("tool_calls", [])
        validated_calls = []
        
        for i, tool_call in enumerate(tool_calls, 1):
            if not isinstance(tool_call, dict):
                continue
            
            tool_name = tool_call.get("tool_name")
            # Skip validate_symbol as it's already executed before planning
            if tool_name == "validate_symbol":
                self.logger.info(f"Skipping validate_symbol (already executed): {tool_name}")
                continue
            
            if not tool_name or tool_name not in available_tool_names:
                self.logger.warning(f"Skipping invalid tool: {tool_name}")
                continue
            
            # Ensure order is set
            if "order" not in tool_call:
                tool_call["order"] = i
            
            # Ensure required fields
            validated_call = {
                "order": tool_call.get("order", i),
                "tool_name": tool_name,
                "parameters": tool_call.get("parameters", {})
            }
            
            validated_calls.append(validated_call)
        
        # Sort by order
        validated_calls.sort(key=lambda x: x.get("order", 0))
        
        # Update order numbers to be sequential
        for i, call in enumerate(validated_calls, 1):
            call["order"] = i
        
        # Calculate confidence score based on number of validated calls
        confidence_score = 0.8 if len(validated_calls) > 0 else 0.0
        
        return {
            "tool_calls": validated_calls,
            "confidence_score": confidence_score
        }
    
    def get_research_plan(self, state: AgentState) -> Dict[str, Any]:
        """
        Get the research plan from agent state.
        
        Args:
            state: Current agent state
            
        Returns:
            Dictionary containing the research plan
        """
        results = state.results
        return results.get("research_plan", {
            "tool_calls": []
        })

