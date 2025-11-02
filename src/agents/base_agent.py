"""
Base agent class for the multi-agent stock research system.
Provides common functionality and interface for all agents.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import json

logger = logging.getLogger(__name__)

@dataclass
class AgentState:
    """Represents the state of an agent during execution."""
    agent_id: str
    stock_symbol: str
    company_name: str
    sector: str
    current_task: str
    context: Dict[str, Any]
    results: Dict[str, Any]
    tools_used: List[str]
    confidence_score: float
    errors: List[str]
    start_time: datetime
    end_time: Optional[datetime] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    def to_partial_update(self) -> Dict[str, Any]:
        """Convert AgentState to partial state update for LangGraph."""
        return {
            "agent_id": self.agent_id,
            "current_task": self.current_task,
            "context": self.context,
            "results": self.results,
            "tools_used": self.tools_used,
            "confidence_score": self.confidence_score,
            "errors": self.errors,
            "end_time": self.end_time
        }

class BaseAgent(ABC):
    """
    Base class for all agents in the multi-agent system.
    
    Each agent is responsible for a specific aspect of stock research
    and can autonomously select and use tools to complete its tasks.
    """
    
    def __init__(
        self,
        agent_id: str,
        openai_api_key: str,
        available_tools: List[Any]
    ):
        """
        Initialize the base agent.
        
        Args:
            agent_id: Unique identifier for the agent
            openai_api_key: OpenAI API key for LLM calls
            available_tools: List of available tools the agent can use
        """
        self.agent_id = agent_id
        self.openai_api_key = openai_api_key
        self.available_tools = available_tools
        self.logger = logging.getLogger(f"{__name__}.{agent_id}")
        
    @abstractmethod
    async def execute_task(
        self,
        stock_symbol: str,
        company_name: str,
        sector: str,
        context: Dict[str, Any]
    ) -> AgentState:
        """
        Execute the agent's primary task.
        
        Args:
            stock_symbol: NSE stock symbol
            company_name: Full company name
            sector: Sector name
            context: Current context from previous agents
            
        Returns:
            AgentState with results and context updates
        """
        pass
    
    async def execute_task_partial(
        self,
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute the agent's primary task and return partial state update.
        This method follows LangGraph best practices by working with existing state
        and returning only the changes.
        
        Args:
            state: Current workflow state from LangGraph
            
        Returns:
            Dictionary containing only the state changes
        """
        try:
            # Extract parameters from state
            stock_symbol = state.get("stock_symbol", "")
            company_name = state.get("company_name", "")
            sector = state.get("sector", "")
            context = state.get("context", {})
            
            # Execute the task using the existing method
            agent_state = await self.execute_task(
                stock_symbol=stock_symbol,
                company_name=company_name,
                sector=sector,
                context=context
            )
            
            # Return only the partial update
            return agent_state.to_partial_update()
            
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            return {
                "errors": [f"{self.agent_id} execution failed: {str(e)}"],
                "confidence_score": 0.0
            }

    def update_context(
        self,
        current_context: Dict[str, Any],
        new_results: Dict[str, Any],
        agent_id: str
    ) -> Dict[str, Any]:
        """
        Update the shared context with new results.
        
        Args:
            current_context: Current shared context
            new_results: New results from this agent
            agent_id: ID of the agent providing results
            
        Returns:
            Updated context
        """
        updated_context = current_context.copy()
        updated_context[f"{agent_id}_results"] = new_results
        updated_context["last_updated_by"] = agent_id
        updated_context["last_updated_at"] = datetime.now().isoformat()
        
        return updated_context
    
    def calculate_confidence_score(
        self,
        results: Dict[str, Any],
        tools_used: List[str],
        errors: List[str]
    ) -> float:
        """
        Calculate confidence score based on results quality.
        
        Args:
            results: Results from agent execution
            tools_used: List of tools used
            errors: List of errors encountered
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        base_score = 0.5
        
        # Increase score for successful tool usage
        if tools_used:
            base_score += min(0.3, len(tools_used) * 0.1)
        
        # Increase score for comprehensive results
        if results:
            result_completeness = len([v for v in results.values() if v is not None])
            base_score += min(0.2, result_completeness * 0.05)
        
        # Decrease score for errors
        if errors:
            base_score -= min(0.3, len(errors) * 0.1)
        
        return max(0.0, min(1.0, base_score))
    
    def log_execution(
        self,
        state: AgentState,
        duration: float
    ) -> None:
        """
        Log agent execution details.
        
        Args:
            state: Final agent state
            duration: Execution duration in seconds
        """
        self.logger.info(
            f"Agent {self.agent_id} completed task '{state.current_task}' "
            f"for {state.stock_symbol} in {duration:.2f}s "
            f"(confidence: {state.confidence_score:.2f}, tools: {len(state.tools_used)})"
        )
        
        if state.errors:
            self.logger.warning(f"Agent {self.agent_id} encountered {len(state.errors)} errors")
    
    def serialize_state(self, state: AgentState) -> Dict[str, Any]:
        """
        Serialize agent state to dictionary.
        
        Args:
            state: Agent state to serialize
            
        Returns:
            Serialized state dictionary
        """
        return {
            "agent_id": state.agent_id,
            "stock_symbol": state.stock_symbol,
            "company_name": state.company_name,
            "sector": state.sector,
            "current_task": state.current_task,
            "context": state.context,
            "results": state.results,
            "tools_used": state.tools_used,
            "confidence_score": state.confidence_score,
            "errors": state.errors,
            "start_time": state.start_time.isoformat(),
            "end_time": state.end_time.isoformat() if state.end_time else None
        }
    
    def deserialize_state(self, data: Dict[str, Any]) -> AgentState:
        """
        Deserialize dictionary to agent state.
        
        Args:
            data: Serialized state data
            
        Returns:
            AgentState object
        """
        return AgentState(
            agent_id=data["agent_id"],
            stock_symbol=data["stock_symbol"],
            company_name=data["company_name"],
            sector=data["sector"],
            current_task=data["current_task"],
            context=data["context"],
            results=data["results"],
            tools_used=data["tools_used"],
            confidence_score=data["confidence_score"],
            errors=data["errors"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]) if data["end_time"] else None
        )
