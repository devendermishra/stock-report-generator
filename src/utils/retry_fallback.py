"""
Fallback implementations for retry utilities when the main retry module is not available.

This module provides no-op implementations that allow code to work even if
the retry module has import issues or dependencies are missing.
"""


async def invoke_tool_with_retry(tool, args, max_retries=None, is_async=True):
    """
    Fallback implementation of invoke_tool_with_retry without retry logic.
    
    Args:
        tool: Tool object with invoke or ainvoke method
        args: Arguments to pass to the tool
        max_retries: Ignored in fallback mode
        is_async: Whether to use async invocation
    
    Returns:
        Tool execution result
    """
    if is_async:
        try:
            return await tool.ainvoke(args)
        except AttributeError:
            return tool.invoke(args)
    else:
        try:
            return tool.invoke(args)
        except AttributeError:
            return await tool.ainvoke(args)


async def call_llm_with_retry(llm_with_tools, messages, max_retries=None, initial_delay=None, max_delay=None):
    """
    Fallback implementation of call_llm_with_retry without retry logic.
    
    Args:
        llm_with_tools: LangChain LLM object with tools bound
        messages: List of messages to send to the LLM
        max_retries: Ignored in fallback mode
        initial_delay: Ignored in fallback mode
        max_delay: Ignored in fallback mode
    
    Returns:
        LLM response
    """
    return await llm_with_tools.ainvoke(messages)

