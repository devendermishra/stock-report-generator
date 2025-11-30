"""
OpenAI Call Wrapper for Automatic Logging.

Wraps OpenAI API calls to automatically log prompts and outputs
to separate log files.
"""

import time
import logging
from typing import Dict, Any, List, Optional
from openai import OpenAI, AsyncOpenAI

# Import logger
try:
    from .openai_logger import openai_logger
except ImportError:
    from tools.openai_logger import openai_logger

# Import metrics
try:
    from ..utils.metrics import record_llm_request_from_response
except ImportError:
    try:
        from utils.metrics import record_llm_request_from_response
    except ImportError:
        record_llm_request_from_response = None

logger = logging.getLogger(__name__)


def logged_chat_completion(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    agent_name: Optional[str] = None,
    **kwargs
):
    """
    Wrapper for OpenAI chat.completions.create that automatically logs prompts and outputs.
    
    Args:
        client: OpenAI client instance
        model: Model name
        messages: List of message dictionaries
        agent_name: Optional agent name for logging
        **kwargs: Additional arguments for chat.completions.create
    
    Returns:
        OpenAI chat completion response
    """
    start_time = time.time()
    
    try:
        # Make the API call
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        
        duration_ms = int((time.time() - start_time) * 1000)
        duration_seconds = (time.time() - start_time)
        response_content = ""
        
        if hasattr(response, 'choices') and len(response.choices) > 0:
            response_content = response.choices[0].message.content
        
        # Record metrics
        if record_llm_request_from_response:
            try:
                record_llm_request_from_response(
                    model=model,
                    response=response,
                    agent_name=agent_name,
                    duration_seconds=duration_seconds,
                    success=True
                )
            except Exception as metrics_error:
                logger.warning(f"Failed to record metrics: {metrics_error}")
        
        # Log the completion
        try:
            openai_logger.log_chat_completion(
                model=model,
                messages=messages,
                response=response_content,
                usage=response.usage.__dict__ if hasattr(response, 'usage') and response.usage else None,
                duration_ms=duration_ms,
                agent_name=agent_name
            )
        except Exception as log_error:
            logger.warning(f"Failed to log completion: {log_error}")
        
        return response
        
    except Exception as e:
        # Record failed request metrics
        if record_llm_request_from_response:
            try:
                duration_seconds = time.time() - start_time
                record_llm_request_from_response(
                    model=model,
                    response=None,
                    agent_name=agent_name,
                    duration_seconds=duration_seconds,
                    success=False
                )
            except Exception:
                pass  # Don't fail on metrics errors
        
        logger.error(f"OpenAI API call failed: {e}")
        raise


async def logged_async_chat_completion(
    client: AsyncOpenAI,
    model: str,
    messages: List[Dict[str, str]],
    agent_name: Optional[str] = None,
    **kwargs
):
    """
    Wrapper for OpenAI async chat.completions.create that automatically logs prompts and outputs.
    
    Args:
        client: AsyncOpenAI client instance
        model: Model name
        messages: List of message dictionaries
        agent_name: Optional agent name for logging
        **kwargs: Additional arguments for chat.completions.create
    
    Returns:
        OpenAI async chat completion response
    """
    start_time = time.time()
    
    try:
        # Make the API call
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        
        duration_ms = int((time.time() - start_time) * 1000)
        duration_seconds = (time.time() - start_time)
        response_content = ""
        
        if hasattr(response, 'choices') and len(response.choices) > 0:
            response_content = response.choices[0].message.content
        
        # Record metrics
        if record_llm_request_from_response:
            try:
                record_llm_request_from_response(
                    model=model,
                    response=response,
                    agent_name=agent_name,
                    duration_seconds=duration_seconds,
                    success=True
                )
            except Exception as metrics_error:
                logger.warning(f"Failed to record metrics: {metrics_error}")
        
        # Log the completion
        try:
            openai_logger.log_chat_completion(
                model=model,
                messages=messages,
                response=response_content,
                usage=response.usage.__dict__ if hasattr(response, 'usage') and response.usage else None,
                duration_ms=duration_ms,
                agent_name=agent_name
            )
        except Exception as log_error:
            logger.warning(f"Failed to log completion: {log_error}")
        
        return response
        
    except Exception as e:
        # Record failed request metrics
        if record_llm_request_from_response:
            try:
                duration_seconds = time.time() - start_time
                record_llm_request_from_response(
                    model=model,
                    response=None,
                    agent_name=agent_name,
                    duration_seconds=duration_seconds,
                    success=False
                )
            except Exception:
                pass  # Don't fail on metrics errors
        
        logger.error(f"OpenAI async API call failed: {e}")
        raise


