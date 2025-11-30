"""
Enhanced Logging Configuration with MDC Support.

Provides structured logging with session ID tracking and separate
log files for prompts and outputs.
"""

import logging
import sys
import os
from contextvars import copy_context
from typing import Optional, Dict, Any
from datetime import datetime

# Import session manager
try:
    from ..utils.session_manager import get_session_id, get_session_context
except ImportError:
    from utils.session_manager import get_session_id, get_session_context


class MDCFilter(logging.Filter):
    """
    Logging filter that adds MDC (Mapped Diagnostic Context) information
    including session ID to log records.
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Add MDC information to log record.
        
        Args:
            record: Log record to filter
            
        Returns:
            True to include the record
        """
        # Add session ID to record
        record.session_id = get_session_id() or "N/A"
        
        # Add session context to record
        context = get_session_context()
        record.stock_symbol = context.get('stock_symbol', 'N/A')
        record.agent_name = context.get('agent_name', 'N/A')
        
        return True


class PromptOutputFilter(logging.Filter):
    """
    Filter to separate prompt and output logs.
    """
    
    def __init__(self, log_type: str):
        """
        Initialize filter.
        
        Args:
            log_type: 'prompt' or 'output'
        """
        super().__init__()
        self.log_type = log_type
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter based on log type.
        
        Args:
            record: Log record to filter
            
        Returns:
            True if record matches this filter's type or if log_type is not set
        """
        record_log_type = getattr(record, 'log_type', None)
        # If log_type is not set, allow through (for backward compatibility)
        # If log_type matches, allow through
        return record_log_type is None or record_log_type == self.log_type


def setup_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
    include_session_id: bool = True,
    combine_prompts_and_outputs: bool = True
) -> None:
    """
    Set up enhanced logging with MDC support and separate log files.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        include_session_id: Whether to include session ID in logs
        combine_prompts_and_outputs: If True, combine prompts and outputs into prompts.log
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Base log format with MDC
    if include_session_id:
        log_format = '%(asctime)s - [%(session_id)s] - [%(stock_symbol)s] - [%(agent_name)s] - %(name)s - %(levelname)s - %(message)s'
    else:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create MDC filter
    mdc_filter = MDCFilter()
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Main log file handler
    main_log_file = os.path.join(log_dir, 'stock_report_generator.log')
    main_file_handler = logging.FileHandler(main_log_file)
    main_file_handler.setLevel(getattr(logging, log_level.upper()))
    main_file_handler.setFormatter(logging.Formatter(log_format))
    main_file_handler.addFilter(mdc_filter)
    root_logger.addHandler(main_file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(logging.Formatter(log_format))
    console_handler.addFilter(mdc_filter)
    root_logger.addHandler(console_handler)
    
    if combine_prompts_and_outputs:
        # Combined prompt and output log file - simple format, always write
        combined_log_file = os.path.join(log_dir, 'prompts.log')
        combined_handler = logging.FileHandler(combined_log_file, mode='a', encoding='utf-8')
        combined_handler.setLevel(logging.DEBUG)  # Accept all levels
        # Simple format for combined logs
        combined_format = '%(message)s'
        combined_handler.setFormatter(logging.Formatter(combined_format))
        
        # Create combined logger for prompts and outputs
        prompt_logger = logging.getLogger('prompts')
        prompt_logger.handlers.clear()
        prompt_logger.addHandler(combined_handler)
        prompt_logger.setLevel(logging.DEBUG)
        prompt_logger.propagate = False  # Don't propagate to root logger
        
        # Use the same logger for outputs
        output_logger = logging.getLogger('outputs')
        output_logger.handlers.clear()
        output_logger.addHandler(combined_handler)
        output_logger.setLevel(logging.DEBUG)
        output_logger.propagate = False  # Don't propagate to root logger
    else:
        # Separate prompt log file - simple format, always write
        prompt_log_file = os.path.join(log_dir, 'prompts.log')
        prompt_handler = logging.FileHandler(prompt_log_file, mode='a', encoding='utf-8')
        prompt_handler.setLevel(logging.DEBUG)  # Accept all levels
        # Simple format for prompts
        prompt_format = '%(message)s'
        prompt_handler.setFormatter(logging.Formatter(prompt_format))
        
        # Create separate logger for prompts
        prompt_logger = logging.getLogger('prompts')
        # Remove any existing handlers
        prompt_logger.handlers.clear()
        prompt_logger.addHandler(prompt_handler)
        prompt_logger.setLevel(logging.DEBUG)
        prompt_logger.propagate = False  # Don't propagate to root logger
        
        # Separate output log file - simple format, always write
        output_log_file = os.path.join(log_dir, 'outputs.log')
        output_handler = logging.FileHandler(output_log_file, mode='a', encoding='utf-8')
        output_handler.setLevel(logging.DEBUG)  # Accept all levels
        # Simple format for outputs
        output_format = '%(message)s'
        output_handler.setFormatter(logging.Formatter(output_format))
        
        # Create separate logger for outputs
        output_logger = logging.getLogger('outputs')
        # Remove any existing handlers
        output_logger.handlers.clear()
        output_logger.addHandler(output_handler)
        output_logger.setLevel(logging.DEBUG)
        output_logger.propagate = False  # Don't propagate to root logger


def get_prompt_logger():
    """
    Get logger for prompts.
    
    Returns:
        Logger instance for prompts
    """
    logger = logging.getLogger('prompts')
    # If logger has no handlers, try to set up logging again
    if not logger.handlers:
        # Re-setup logging if handlers are missing
        # Get config value for combining logs
        try:
            from ..config import Config
            combine_logs = Config.COMBINE_PROMPTS_AND_OUTPUTS
        except ImportError:
            try:
                from config import Config
                combine_logs = Config.COMBINE_PROMPTS_AND_OUTPUTS
            except ImportError:
                combine_logs = True  # Default to enabled
        setup_logging(combine_prompts_and_outputs=combine_logs)
    return logger


def get_output_logger():
    """
    Get logger for outputs.
    
    Returns:
        Logger instance for outputs
    """
    logger = logging.getLogger('outputs')
    # If logger has no handlers, try to set up logging again
    if not logger.handlers:
        # Re-setup logging if handlers are missing
        # Get config value for combining logs
        try:
            from ..config import Config
            combine_logs = Config.COMBINE_PROMPTS_AND_OUTPUTS
        except ImportError:
            try:
                from config import Config
                combine_logs = Config.COMBINE_PROMPTS_AND_OUTPUTS
            except ImportError:
                combine_logs = True  # Default to enabled
        setup_logging(combine_prompts_and_outputs=combine_logs)
    return logger

