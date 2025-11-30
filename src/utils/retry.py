"""
Retry utility for LLM calls and tool invocations.

Provides configurable retry decorators with exponential backoff
for handling transient failures in API calls and tool executions.
"""

import asyncio
import logging
import time
from functools import wraps
from typing import Callable, TypeVar, Any, Optional, Union, Tuple, List, Dict
from openai import APIError, RateLimitError, APIConnectionError, APITimeoutError

try:
    from ..config import Config
except ImportError:
    from config import Config

logger = logging.getLogger(__name__)

# Type variables for generic function typing
T = TypeVar('T')
R = TypeVar('R')

# Retryable exceptions for OpenAI API
OPENAI_RETRYABLE_EXCEPTIONS = (
    APIError,
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    ConnectionError,
    TimeoutError,
)

# Retryable exceptions for general API calls
GENERAL_RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OSError,
)


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retryable_exceptions: Tuple[Exception, ...] = GENERAL_RETRYABLE_EXCEPTIONS,
    on_retry: Optional[Callable[[Exception, int], None]] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for retrying function calls with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds before first retry (default: 1.0)
        max_delay: Maximum delay in seconds between retries (default: 60.0)
        exponential_base: Base for exponential backoff calculation (default: 2.0)
        retryable_exceptions: Tuple of exception types that should trigger retries
        on_retry: Optional callback function called on each retry (exception, attempt_number)

    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e

                    if attempt < max_retries:
                        # Calculate delay with exponential backoff
                        delay = min(
                            initial_delay * (exponential_base ** attempt),
                            max_delay
                        )

                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {str(e)}. "
                            f"Retrying in {delay:.2f} seconds..."
                        )

                        # Call retry callback if provided
                        if on_retry:
                            try:
                                on_retry(e, attempt + 1)
                            except Exception as callback_error:
                                logger.warning(f"Retry callback failed: {callback_error}")

                        time.sleep(delay)
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_retries + 1} attempts: {str(e)}"
                        )
                        raise
                except Exception as e:
                    # Non-retryable exception - raise immediately
                    logger.error(f"{func.__name__} failed with non-retryable exception: {str(e)}")
                    raise

            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
            raise RuntimeError(f"{func.__name__} failed unexpectedly")

        return wrapper
    return decorator


def async_retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retryable_exceptions: Tuple[Exception, ...] = GENERAL_RETRYABLE_EXCEPTIONS,
    on_retry: Optional[Callable[[Exception, int], None]] = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator for retrying async function calls with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds before first retry (default: 1.0)
        max_delay: Maximum delay in seconds between retries (default: 60.0)
        exponential_base: Base for exponential backoff calculation (default: 2.0)
        retryable_exceptions: Tuple of exception types that should trigger retries
        on_retry: Optional callback function called on each retry (exception, attempt_number)

    Returns:
        Decorated async function with retry logic
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e

                    if attempt < max_retries:
                        # Calculate delay with exponential backoff
                        delay = min(
                            initial_delay * (exponential_base ** attempt),
                            max_delay
                        )

                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {str(e)}. "
                            f"Retrying in {delay:.2f} seconds..."
                        )

                        # Call retry callback if provided
                        if on_retry:
                            try:
                                on_retry(e, attempt + 1)
                            except Exception as callback_error:
                                logger.warning(f"Retry callback failed: {callback_error}")

                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_retries + 1} attempts: {str(e)}"
                        )
                        raise
                except Exception as e:
                    # Non-retryable exception - raise immediately
                    logger.error(f"{func.__name__} failed with non-retryable exception: {str(e)}")
                    raise

            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
            raise RuntimeError(f"{func.__name__} failed unexpectedly")

        return wrapper
    return decorator


def retry_llm_call(
    max_retries: Optional[int] = None,
    initial_delay: Optional[float] = None,
    max_delay: Optional[float] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator specifically for LLM API calls with OpenAI-specific retry logic.

    Args:
        max_retries: Maximum number of retry attempts (uses Config if None)
        initial_delay: Initial delay in seconds (uses Config if None)
        max_delay: Maximum delay in seconds (uses Config if None)

    Returns:
        Decorated function with LLM-specific retry logic
    """
    max_retries = max_retries or Config.LLM_RETRY_MAX_ATTEMPTS
    initial_delay = initial_delay or Config.LLM_RETRY_INITIAL_DELAY
    max_delay = max_delay or Config.LLM_RETRY_MAX_DELAY

    return retry_with_backoff(
        max_retries=max_retries,
        initial_delay=initial_delay,
        max_delay=max_delay,
        retryable_exceptions=OPENAI_RETRYABLE_EXCEPTIONS
    )


def async_retry_llm_call(
    max_retries: Optional[int] = None,
    initial_delay: Optional[float] = None,
    max_delay: Optional[float] = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator specifically for async LLM API calls with OpenAI-specific retry logic.

    Args:
        max_retries: Maximum number of retry attempts (uses Config if None)
        initial_delay: Initial delay in seconds (uses Config if None)
        max_delay: Maximum delay in seconds (uses Config if None)

    Returns:
        Decorated async function with LLM-specific retry logic
    """
    max_retries = max_retries or Config.LLM_RETRY_MAX_ATTEMPTS
    initial_delay = initial_delay or Config.LLM_RETRY_INITIAL_DELAY
    max_delay = max_delay or Config.LLM_RETRY_MAX_DELAY

    return async_retry_with_backoff(
        max_retries=max_retries,
        initial_delay=initial_delay,
        max_delay=max_delay,
        retryable_exceptions=OPENAI_RETRYABLE_EXCEPTIONS
    )


def retry_tool_call(
    max_retries: Optional[int] = None,
    initial_delay: Optional[float] = None,
    max_delay: Optional[float] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator specifically for tool invocations with retry logic.

    Args:
        max_retries: Maximum number of retry attempts (uses Config if None)
        initial_delay: Initial delay in seconds (uses Config if None)
        max_delay: Maximum delay in seconds (uses Config if None)

    Returns:
        Decorated function with tool-specific retry logic
    """
    max_retries = max_retries or Config.TOOL_RETRY_MAX_ATTEMPTS
    initial_delay = initial_delay or Config.TOOL_RETRY_INITIAL_DELAY
    max_delay = max_delay or Config.TOOL_RETRY_MAX_DELAY

    return retry_with_backoff(
        max_retries=max_retries,
        initial_delay=initial_delay,
        max_delay=max_delay,
        retryable_exceptions=GENERAL_RETRYABLE_EXCEPTIONS
    )


async def invoke_tool_with_retry(
    tool: Any,
    args: Dict[str, Any],
    max_retries: Optional[int] = None,
    is_async: bool = True
) -> Any:
    """
    Invoke a tool (sync or async) with retry logic.

    Args:
        tool: Tool object with invoke or ainvoke method
        args: Arguments to pass to the tool
        max_retries: Maximum number of retry attempts (uses Config if None)
        is_async: Whether to use async invocation (default: True)

    Returns:
        Tool execution result

    Raises:
        Exception: If tool execution fails after all retries
    """
    max_retries = max_retries or Config.TOOL_RETRY_MAX_ATTEMPTS
    initial_delay = Config.TOOL_RETRY_INITIAL_DELAY
    max_delay = Config.TOOL_RETRY_MAX_DELAY
    exponential_base = 2.0

    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            if is_async:
                if hasattr(tool, 'ainvoke'):
                    return await tool.ainvoke(args)
                elif hasattr(tool, 'invoke'):
                    # Fallback to sync if async not available
                    return tool.invoke(args)
                else:
                    raise AttributeError(f"Tool {type(tool).__name__} has no invoke or ainvoke method")
            else:
                if hasattr(tool, 'invoke'):
                    return tool.invoke(args)
                elif hasattr(tool, 'ainvoke'):
                    # Fallback to async if sync not available
                    return await tool.ainvoke(args)
                else:
                    raise AttributeError(f"Tool {type(tool).__name__} has no invoke or ainvoke method")
        except GENERAL_RETRYABLE_EXCEPTIONS as e:
            last_exception = e

            if attempt < max_retries:
                delay = min(
                    initial_delay * (exponential_base ** attempt),
                    max_delay
                )

                logger.warning(
                    f"Tool {getattr(tool, 'name', type(tool).__name__)} invocation failed "
                    f"(attempt {attempt + 1}/{max_retries + 1}): {str(e)}. "
                    f"Retrying in {delay:.2f} seconds..."
                )

                await asyncio.sleep(delay) if is_async else time.sleep(delay)
            else:
                logger.error(
                    f"Tool {getattr(tool, 'name', type(tool).__name__)} invocation failed "
                    f"after {max_retries + 1} attempts: {str(e)}"
                )
                raise
        except Exception as e:
            # Non-retryable exception - raise immediately
            logger.error(
                f"Tool {getattr(tool, 'name', type(tool).__name__)} invocation failed "
                f"with non-retryable exception: {str(e)}"
            )
            raise

    if last_exception:
        raise last_exception
    raise RuntimeError("Tool invocation failed unexpectedly")


async def call_llm_with_retry(
    llm_with_tools: Any,
    messages: List[Any],
    max_retries: Optional[int] = None,
    initial_delay: Optional[float] = None,
    max_delay: Optional[float] = None
) -> Any:
    """
    Call LangChain LLM with retry logic.

    This is a convenience function that wraps the common pattern of calling
    llm_with_tools.ainvoke() with retry logic.

    Args:
        llm_with_tools: LangChain LLM object with tools bound (e.g., llm.bind_tools())
        messages: List of messages to send to the LLM
        max_retries: Maximum number of retry attempts (uses Config if None)
        initial_delay: Initial delay in seconds (uses Config if None)
        max_delay: Maximum delay in seconds (uses Config if None)

    Returns:
        LLM response

    Raises:
        Exception: If LLM call fails after all retries
    """
    max_retries = max_retries or Config.LLM_RETRY_MAX_ATTEMPTS
    initial_delay = initial_delay or Config.LLM_RETRY_INITIAL_DELAY
    max_delay = max_delay or Config.LLM_RETRY_MAX_DELAY

    @async_retry_with_backoff(
        max_retries=max_retries,
        initial_delay=initial_delay,
        max_delay=max_delay,
        retryable_exceptions=OPENAI_RETRYABLE_EXCEPTIONS
    )
    async def _call_llm(msgs):
        return await llm_with_tools.ainvoke(msgs)

    return await _call_llm(messages)

