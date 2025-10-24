"""
MCP (Model Context Protocol) Context Manager
Manages shared memory and context between agents in the LangGraph workflow.
"""

from typing import Dict, Any, List, Optional
import json
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class ContextType(Enum):
    """Types of context data that can be stored."""
    SECTOR_SUMMARY = "sector_summary"
    STOCK_SUMMARY = "stock_summary"
    MANAGEMENT_SUMMARY = "management_summary"
    SWOT_SUMMARY = "swot_summary"
    FINAL_REPORT = "final_report"
    RAW_DATA = "raw_data"
    TOOL_OUTPUT = "tool_output"

@dataclass
class ContextEntry:
    """Represents a single context entry in the MCP store."""
    id: str
    type: ContextType
    data: Dict[str, Any]
    agent_id: str
    timestamp: datetime
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type.value,
            "data": self.data,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

class MCPContextManager:
    """
    Model Context Protocol Context Manager.
    
    Manages shared memory and context between agents in the LangGraph workflow.
    Provides methods for storing, retrieving, and managing context data.
    """
    
    def __init__(self, max_context_size: int = 10000):
        """
        Initialize the MCP Context Manager.
        
        Args:
            max_context_size: Maximum number of context entries to store
        """
        self.max_context_size = max_context_size
        self.context_store: Dict[str, ContextEntry] = {}
        self.context_history: List[str] = []
        self.current_session_id = None
        
    def start_session(self, session_id: str) -> None:
        """Start a new MCP session."""
        self.current_session_id = session_id
        logger.info(f"Started MCP session: {session_id}")
        
    def store_context(
        self,
        context_id: str,
        context_type: ContextType,
        data: Dict[str, Any],
        agent_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Store context data in the MCP store.
        
        Args:
            context_id: Unique identifier for the context entry
            context_type: Type of context data
            data: The actual data to store
            agent_id: ID of the agent storing the data
            metadata: Optional metadata about the context entry
        """
        if metadata is None:
            metadata = {}
            
        entry = ContextEntry(
            id=context_id,
            type=context_type,
            data=data,
            agent_id=agent_id,
            timestamp=datetime.now(),
            metadata=metadata
        )
        
        self.context_store[context_id] = entry
        self.context_history.append(context_id)
        
        # Maintain max context size
        if len(self.context_store) > self.max_context_size:
            oldest_id = self.context_history.pop(0)
            if oldest_id in self.context_store:
                del self.context_store[oldest_id]
                
        logger.info(f"Stored context: {context_id} by {agent_id}")
        
    def retrieve_context(self, context_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve context data by ID.
        
        Args:
            context_id: ID of the context entry to retrieve
            
        Returns:
            Context data if found, None otherwise
        """
        if context_id in self.context_store:
            entry = self.context_store[context_id]
            logger.info(f"Retrieved context: {context_id}")
            return entry.to_dict()
        return None
        
    def get_context_by_type(self, context_type: ContextType) -> List[Dict[str, Any]]:
        """
        Get all context entries of a specific type.
        
        Args:
            context_type: Type of context to retrieve
            
        Returns:
            List of context entries of the specified type
        """
        entries = []
        for entry in self.context_store.values():
            if entry.type == context_type:
                entries.append(entry.to_dict())
        return entries
        
    def get_context_by_agent(self, agent_id: str) -> List[Dict[str, Any]]:
        """
        Get all context entries created by a specific agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            List of context entries created by the agent
        """
        entries = []
        for entry in self.context_store.values():
            if entry.agent_id == agent_id:
                entries.append(entry.to_dict())
        return entries
        
    def get_latest_context(self, context_type: ContextType) -> Optional[Dict[str, Any]]:
        """
        Get the latest context entry of a specific type.
        
        Args:
            context_type: Type of context to retrieve
            
        Returns:
            Latest context entry of the specified type, or None
        """
        latest_entry = None
        latest_timestamp = None
        
        for entry in self.context_store.values():
            if entry.type == context_type:
                if latest_timestamp is None or entry.timestamp > latest_timestamp:
                    latest_entry = entry
                    latest_timestamp = entry.timestamp
                    
        return latest_entry.to_dict() if latest_entry else None
        
    def clear_context(self) -> None:
        """Clear all context data."""
        self.context_store.clear()
        self.context_history.clear()
        logger.info("Cleared all context data")
        
    def get_context_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current context state.
        
        Returns:
            Dictionary containing context summary
        """
        type_counts = {}
        agent_counts = {}
        
        for entry in self.context_store.values():
            type_counts[entry.type.value] = type_counts.get(entry.type.value, 0) + 1
            agent_counts[entry.agent_id] = agent_counts.get(entry.agent_id, 0) + 1
            
        return {
            "total_entries": len(self.context_store),
            "session_id": self.current_session_id,
            "type_counts": type_counts,
            "agent_counts": agent_counts,
            "context_types": [t.value for t in ContextType]
        }
        
    def export_context(self) -> Dict[str, Any]:
        """
        Export all context data for persistence or debugging.
        
        Returns:
            Dictionary containing all context data
        """
        return {
            "session_id": self.current_session_id,
            "context_entries": [entry.to_dict() for entry in self.context_store.values()],
            "context_history": self.context_history,
            "summary": self.get_context_summary()
        }
        
    def import_context(self, context_data: Dict[str, Any]) -> None:
        """
        Import context data from a previous session.
        
        Args:
            context_data: Dictionary containing context data to import
        """
        self.clear_context()
        
        if "session_id" in context_data:
            self.current_session_id = context_data["session_id"]
            
        if "context_entries" in context_data:
            for entry_data in context_data["context_entries"]:
                entry = ContextEntry(
                    id=entry_data["id"],
                    type=ContextType(entry_data["type"]),
                    data=entry_data["data"],
                    agent_id=entry_data["agent_id"],
                    timestamp=datetime.fromisoformat(entry_data["timestamp"]),
                    metadata=entry_data["metadata"]
                )
                self.context_store[entry.id] = entry
                
        if "context_history" in context_data:
            self.context_history = context_data["context_history"]
            
        logger.info(f"Imported context data from session: {self.current_session_id}")
