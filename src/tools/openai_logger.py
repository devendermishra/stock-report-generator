"""
OpenAI Chat Logging Utility.
Provides structured logging for OpenAI chat completions with separate prompt/output logging.
"""

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass

# Import session manager and logging config
try:
    from ..utils.session_manager import get_session_id, get_session_context
    from ..utils.logging_config import get_prompt_logger, get_output_logger
except ImportError:
    from utils.session_manager import get_session_id, get_session_context
    from utils.logging_config import get_prompt_logger, get_output_logger

logger = logging.getLogger(__name__)

@dataclass
class ChatLogEntry:
    """Represents a single chat completion log entry."""
    timestamp: datetime
    model: str
    messages: List[Dict[str, str]]
    response: str
    tokens_used: Optional[int] = None
    cost_estimate: Optional[float] = None
    duration_ms: Optional[int] = None
    session_id: Optional[str] = None
    agent_name: Optional[str] = None
    stock_symbol: Optional[str] = None

class OpenAILogger:
    """
    Logger for OpenAI chat completions.
    
    Provides structured logging for OpenAI API calls including:
    - Request/response logging
    - Token usage tracking
    - Cost estimation
    - Performance metrics
    """
    
    def __init__(self, log_level: str = "INFO"):
        """
        Initialize the OpenAI logger.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.logger = logging.getLogger(f"{__name__}.OpenAILogger")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Token costs (approximate, as of 2024)
        self.token_costs = {
            "gpt-4o": {"input": 0.005, "output": 0.015},  # per 1K tokens
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
        }
    
    def log_chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        response: str,
        usage: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[int] = None,
        agent_name: Optional[str] = None
    ) -> ChatLogEntry:
        """
        Log a chat completion request and response with separate prompt/output logging.
        
        Args:
            model: Model used for completion
            messages: List of messages in the conversation
            response: Response content
            usage: Token usage information
            duration_ms: Request duration in milliseconds
            agent_name: Name of the agent making the request
            
        Returns:
            ChatLogEntry object
        """
        try:
            # Get session ID and context
            session_id = get_session_id()
            context = get_session_context()
            stock_symbol = context.get('stock_symbol')
            
            # Calculate token usage and cost
            tokens_used = None
            cost_estimate = None
            
            if usage:
                tokens_used = usage.get('total_tokens', 0)
                if model in self.token_costs:
                    input_tokens = usage.get('prompt_tokens', 0)
                    output_tokens = usage.get('completion_tokens', 0)
                    
                    input_cost = (input_tokens / 1000) * self.token_costs[model]['input']
                    output_cost = (output_tokens / 1000) * self.token_costs[model]['output']
                    cost_estimate = input_cost + output_cost
            
            # Create log entry
            log_entry = ChatLogEntry(
                timestamp=datetime.now(),
                model=model,
                messages=messages,
                response=response,
                tokens_used=tokens_used,
                cost_estimate=cost_estimate,
                duration_ms=duration_ms,
                session_id=session_id,
                agent_name=agent_name,
                stock_symbol=stock_symbol
            )
            
            # Log prompts separately
            self._log_prompt(log_entry, agent_name)
            
            # Log outputs separately
            self._log_output(log_entry, agent_name)
            
            # Log the completion details (summary)
            self._log_completion_details(log_entry, agent_name)
            
            return log_entry
            
        except Exception as e:
            self.logger.error(f"Error logging chat completion: {e}")
            return ChatLogEntry(
                timestamp=datetime.now(),
                model=model,
                messages=messages,
                response=response
            )
    
    def _log_prompt(self, entry: ChatLogEntry, agent_name: Optional[str] = None):
        """Log prompt separately to prompts.log in simple format."""
        try:
            prompt_logger = get_prompt_logger()
            
            # Check if logger has handlers
            if not prompt_logger.handlers:
                self.logger.warning("Prompt logger has no handlers. Logging setup may not be complete.")
                return
            
            agent = agent_name or entry.agent_name or "Unknown"
            session = entry.session_id or "N/A"
            symbol = entry.stock_symbol or "N/A"
            
            # Simple, readable format - write all at once to ensure it's flushed
            log_content = (
                f"\n{'='*80}\n"
                f"TIMESTAMP: {entry.timestamp.isoformat()}\n"
                f"SESSION ID: {session}\n"
                f"STOCK SYMBOL: {symbol}\n"
                f"AGENT: {agent}\n"
                f"MODEL: {entry.model}\n"
                f"MESSAGE COUNT: {len(entry.messages)}\n"
                f"{'-'*80}\n"
                f"MESSAGES:\n"
            )
            
            # Add each message
            for i, msg in enumerate(entry.messages, 1):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                log_content += f"\n--- Message {i} ({role.upper()}) ---\n{content}\n"
            
            log_content += f"{'='*80}\n"
            
            # Write to logger and flush
            prompt_logger.info(log_content)
            # Force flush all handlers
            for handler in prompt_logger.handlers:
                try:
                    handler.flush()
                except Exception as flush_error:
                    self.logger.warning(f"Failed to flush handler: {flush_error}")
            
        except Exception as e:
            self.logger.error(f"Error logging prompt: {e}", exc_info=True)
    
    def _log_output(self, entry: ChatLogEntry, agent_name: Optional[str] = None):
        """Log output separately to outputs.log in simple format."""
        try:
            output_logger = get_output_logger()
            
            # Check if logger has handlers
            if not output_logger.handlers:
                self.logger.warning("Output logger has no handlers. Logging setup may not be complete.")
                return
            
            agent = agent_name or entry.agent_name or "Unknown"
            session = entry.session_id or "N/A"
            symbol = entry.stock_symbol or "N/A"
            
            # Simple, readable format - write all at once to ensure it's flushed
            log_content = (
                f"\n{'='*80}\n"
                f"TIMESTAMP: {entry.timestamp.isoformat()}\n"
                f"SESSION ID: {session}\n"
                f"STOCK SYMBOL: {symbol}\n"
                f"AGENT: {agent}\n"
                f"MODEL: {entry.model}\n"
            )
            
            if entry.tokens_used:
                log_content += f"TOKENS USED: {entry.tokens_used}\n"
            if entry.cost_estimate:
                log_content += f"ESTIMATED COST: ${entry.cost_estimate:.4f}\n"
            if entry.duration_ms:
                log_content += f"DURATION: {entry.duration_ms}ms\n"
            
            log_content += (
                f"{'-'*80}\n"
                f"RESPONSE:\n{entry.response}\n"
                f"{'='*80}\n"
            )
            
            # Write to logger and flush
            output_logger.info(log_content)
            # Force flush all handlers
            for handler in output_logger.handlers:
                try:
                    handler.flush()
                except Exception as flush_error:
                    self.logger.warning(f"Failed to flush handler: {flush_error}")
            
        except Exception as e:
            self.logger.error(f"Error logging output: {e}", exc_info=True)
    
    def _log_completion_details(self, entry: ChatLogEntry, agent_name: Optional[str] = None):
        """Log summary completion information to main log."""
        try:
            # Basic completion log
            agent_info = f" [{agent_name}]" if agent_name else ""
            session_info = f" [Session: {entry.session_id}]" if entry.session_id else ""
            self.logger.info(f"OpenAI Chat Completion{agent_info}{session_info}: {entry.model}")
            
            # Log usage and cost information
            if entry.tokens_used:
                self.logger.info(f"Tokens used: {entry.tokens_used}")
                
            if entry.cost_estimate:
                self.logger.info(f"Estimated cost: ${entry.cost_estimate:.4f}")
                
            if entry.duration_ms:
                self.logger.info(f"Duration: {entry.duration_ms}ms")
            
            # Log performance metrics
            if entry.tokens_used and entry.duration_ms:
                tokens_per_second = (entry.tokens_used / entry.duration_ms) * 1000
                self.logger.debug(f"Performance: {tokens_per_second:.2f} tokens/second")
                
        except Exception as e:
            self.logger.error(f"Error logging completion details: {e}")
    
    def log_error(self, error: Exception, model: str, agent_name: Optional[str] = None):
        """Log OpenAI API errors."""
        agent_info = f" [{agent_name}]" if agent_name else ""
        self.logger.error(f"OpenAI API Error{agent_info} ({model}): {str(error)}")
    
    def get_usage_summary(self, entries: List[ChatLogEntry]) -> Dict[str, Any]:
        """
        Get usage summary from log entries.
        
        Args:
            entries: List of ChatLogEntry objects
            
        Returns:
            Dictionary with usage summary
        """
        try:
            total_tokens = sum(entry.tokens_used for entry in entries if entry.tokens_used)
            total_cost = sum(entry.cost_estimate for entry in entries if entry.cost_estimate)
            total_duration = sum(entry.duration_ms for entry in entries if entry.duration_ms)
            
            model_usage = {}
            for entry in entries:
                model = entry.model
                if model not in model_usage:
                    model_usage[model] = {
                        'count': 0,
                        'tokens': 0,
                        'cost': 0.0,
                        'duration': 0
                    }
                
                model_usage[model]['count'] += 1
                if entry.tokens_used:
                    model_usage[model]['tokens'] += entry.tokens_used
                if entry.cost_estimate:
                    model_usage[model]['cost'] += entry.cost_estimate
                if entry.duration_ms:
                    model_usage[model]['duration'] += entry.duration_ms
            
            return {
                'total_requests': len(entries),
                'total_tokens': total_tokens,
                'total_cost': total_cost,
                'total_duration_ms': total_duration,
                'model_usage': model_usage,
                'average_tokens_per_request': total_tokens / len(entries) if entries else 0,
                'average_cost_per_request': total_cost / len(entries) if entries else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating usage summary: {e}")
            return {}

# Global logger instance
openai_logger = OpenAILogger()
