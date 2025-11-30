"""
LLM Guardrails Wrapper for OpenAI API Calls.

Provides guardrails validation for LLM calls without Giskard dependency.
Wraps OpenAI API calls with input/output validation and safety checks.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from openai import OpenAI, AsyncOpenAI

try:
    from ..utils.session_manager import get_session_id, get_session_context
    from ..tools.openai_logger import openai_logger
    from ..utils.metrics import record_llm_request_from_response
except ImportError:
    from utils.session_manager import get_session_id, get_session_context
    from tools.openai_logger import openai_logger
    try:
        from utils.metrics import record_llm_request_from_response
    except ImportError:
        record_llm_request_from_response = None

logger = logging.getLogger(__name__)

# Guardrails imports
try:
    from .guardrails import (
        initialize_guardrails,
        get_guardrails,
        GuardrailResult
    )
    GUARDRAILS_AVAILABLE = True
except ImportError:
    GUARDRAILS_AVAILABLE = False
    initialize_guardrails = None
    get_guardrails = None
    GuardrailResult = None
    logger.warning("Guardrails module not available")


class LLMGuardrailsWrapper:
    """
    Wrapper class for LLM calls with guardrails validation.
    
    Provides input/output validation and safety checks for LLM API calls.
    """
    
    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini", enable_guardrails: bool = True):
        """
        Initialize LLM Guardrails Wrapper.
        
        Args:
            api_key: OpenAI API key
            model_name: Model name to use
            enable_guardrails: Whether to enable guardrails validation
        """
        self.api_key = api_key
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
        self.enable_guardrails = enable_guardrails
        self.guardrails = None
        self._initialize_guardrails()
    
    def _initialize_guardrails(self):
        """Initialize guardrails if enabled."""
        if not self.enable_guardrails:
            logger.info("Guardrails disabled")
            return
        
        if not GUARDRAILS_AVAILABLE:
            logger.warning("Guardrails not available. LLM calls will proceed without guardrails validation.")
            return
        
        try:
            self.guardrails = initialize_guardrails(enable_guardrails_ai=True)
            logger.info("Guardrails initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize guardrails: {e}. Continuing without guardrails.")
            self.guardrails = None
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1000,
        expected_output_format: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Make a chat completion call with guardrails validation.
        
        Args:
            messages: List of message dictionaries
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            expected_output_format: Expected output format (e.g., "json", "markdown")
            **kwargs: Additional arguments for OpenAI API
        
        Returns:
            OpenAI chat completion response
        
        Raises:
            ValueError: If guardrails validation fails
        """
        try:
            # Step 1: Validate input with guardrails
            input_guardrail_results = None
            if self.guardrails:
                is_valid, checks = self.guardrails.validate_input(messages)
                
                # Store input guardrail results for logging
                input_guardrail_results = {
                    'is_valid': is_valid,
                    'failed_checks': [c.message for c in checks if c.status == GuardrailResult.FAIL],
                    'warnings': [c.message for c in checks if c.status == GuardrailResult.WARNING],
                    'passed_checks': [c.message for c in checks if c.status == GuardrailResult.PASS]
                }
                
                if not is_valid:
                    failed_checks = [c for c in checks if c.status == GuardrailResult.FAIL]
                    error_msg = f"Input validation failed: {', '.join([c.message for c in failed_checks])}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                # Log warnings if any
                warnings = [c for c in checks if c.status == GuardrailResult.WARNING]
                for warning in warnings:
                    logger.warning(f"Guardrail input warning: {warning.message}")
                
                # Log input validation results
                if input_guardrail_results:
                    logger.debug(f"Input guardrail validation: {'PASSED' if is_valid else 'FAILED'} - "
                               f"Passed: {len(input_guardrail_results['passed_checks'])}, "
                               f"Warnings: {len(input_guardrail_results['warnings'])}, "
                               f"Failed: {len(input_guardrail_results['failed_checks'])}")
            
            # Step 2: Add session ID to system message if present
            session_id = get_session_id()
            if session_id:
                # Add session ID to the first system message, or create one
                messages_with_session = messages.copy()
                system_message_found = False
                for msg in messages_with_session:
                    if msg.get('role') == 'system':
                        # Append session ID to system message
                        current_content = msg.get('content', '')
                        if 'Session ID:' not in current_content:
                            msg['content'] = f"{current_content}\n\nSession ID: {session_id}"
                        system_message_found = True
                        break
                
                # If no system message, add one with session ID
                if not system_message_found:
                    messages_with_session.insert(0, {
                        'role': 'system',
                        'content': f'Session ID: {session_id}'
                    })
            else:
                messages_with_session = messages
            
            # Step 3: Make the actual API call
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages_with_session,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            duration_ms = int((time.time() - start_time) * 1000)
            duration_seconds = (time.time() - start_time)
            
            # Extract response content
            response_content = ""
            if hasattr(response, 'choices') and len(response.choices) > 0:
                response_content = response.choices[0].message.content
            
            # Record metrics (before guardrails validation to capture all requests)
            if record_llm_request_from_response:
                try:
                    record_llm_request_from_response(
                        model=self.model_name,
                        response=response,
                        agent_name=agent_name,
                        duration_seconds=duration_seconds,
                        success=True
                    )
                except Exception as metrics_error:
                    logger.warning(f"Failed to record metrics: {metrics_error}")
            
            # Step 4: Validate output with guardrails
            guardrail_results = None
            if self.guardrails and hasattr(response, 'choices') and len(response.choices) > 0:
                output_text = response.choices[0].message.content
                is_valid, checks = self.guardrails.validate_output(
                    output_text,
                    expected_format=expected_output_format
                )
                
                # Store guardrail results for logging
                guardrail_results = {
                    'is_valid': is_valid,
                    'failed_checks': [c.message for c in checks if c.status == GuardrailResult.FAIL],
                    'warnings': [c.message for c in checks if c.status == GuardrailResult.WARNING],
                    'passed_checks': [c.message for c in checks if c.status == GuardrailResult.PASS]
                }
                
                if not is_valid:
                    failed_checks = [c for c in checks if c.status == GuardrailResult.FAIL]
                    error_msg = f"Output validation failed: {', '.join([c.message for c in failed_checks])}"
                    logger.error(error_msg)
                    # Note: We still return the response but log the failure
                    # You can choose to raise an exception here if needed
                
                # Log warnings if any
                warnings = [c for c in checks if c.status == GuardrailResult.WARNING]
                for warning in warnings:
                    logger.warning(f"Guardrail output warning: {warning.message}")
            
            # Log the completion (this will log prompts and outputs separately)
            # Get agent name from session context
            agent_name = None
            try:
                context = get_session_context()
                agent_name = context.get('agent_name')
            except Exception:
                pass
            
            try:
                # Add guardrail information to response if available
                if guardrail_results:
                    guardrail_info = f"\n\n[GUARDRAIL VALIDATION]\n"
                    guardrail_info += f"Status: {'PASSED' if guardrail_results['is_valid'] else 'FAILED'}\n"
                    if guardrail_results['failed_checks']:
                        guardrail_info += f"Failed Checks: {', '.join(guardrail_results['failed_checks'])}\n"
                    if guardrail_results['warnings']:
                        guardrail_info += f"Warnings: {', '.join(guardrail_results['warnings'])}\n"
                    if guardrail_results['passed_checks']:
                        guardrail_info += f"Passed Checks: {len(guardrail_results['passed_checks'])} checks passed\n"
                    
                    # Log guardrail info to main logger
                    logger.info(f"Guardrail validation results: {guardrail_info}")
                
                openai_logger.log_chat_completion(
                    model=self.model_name,
                    messages=messages_with_session,
                    response=response_content,
                    usage=response.usage.__dict__ if hasattr(response, 'usage') and response.usage else None,
                    duration_ms=duration_ms,
                    agent_name=agent_name
                )
            except Exception as log_error:
                logger.warning(f"Failed to log completion: {log_error}", exc_info=True)
            
            return response
            
        except ValueError:
            # Re-raise validation errors
            raise
        except Exception as e:
            # Record failed request metrics
            if record_llm_request_from_response:
                try:
                    duration_seconds = time.time() - start_time
                    record_llm_request_from_response(
                        model=self.model_name,
                        response=None,
                        agent_name=agent_name,
                        duration_seconds=duration_seconds,
                        success=False
                    )
                except Exception:
                    pass  # Don't fail on metrics errors
            
            logger.error(f"Error in chat completion: {e}")
            raise
    
    async def async_chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1000,
        expected_output_format: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Make an async chat completion call with guardrails validation.
        
        Args:
            messages: List of message dictionaries
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            expected_output_format: Expected output format (e.g., "json", "markdown")
            **kwargs: Additional arguments for OpenAI API
        
        Returns:
            OpenAI async chat completion response
        
        Raises:
            ValueError: If guardrails validation fails
        """
        try:
            # Step 1: Validate input with guardrails
            input_guardrail_results = None
            if self.guardrails:
                is_valid, checks = self.guardrails.validate_input(messages)
                
                # Store input guardrail results for logging
                input_guardrail_results = {
                    'is_valid': is_valid,
                    'failed_checks': [c.message for c in checks if c.status == GuardrailResult.FAIL],
                    'warnings': [c.message for c in checks if c.status == GuardrailResult.WARNING],
                    'passed_checks': [c.message for c in checks if c.status == GuardrailResult.PASS]
                }
                
                if not is_valid:
                    failed_checks = [c for c in checks if c.status == GuardrailResult.FAIL]
                    error_msg = f"Input validation failed: {', '.join([c.message for c in failed_checks])}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                # Log warnings if any
                warnings = [c for c in checks if c.status == GuardrailResult.WARNING]
                for warning in warnings:
                    logger.warning(f"Guardrail input warning: {warning.message}")
                
                # Log input validation results
                if input_guardrail_results:
                    logger.debug(f"Input guardrail validation: {'PASSED' if is_valid else 'FAILED'} - "
                               f"Passed: {len(input_guardrail_results['passed_checks'])}, "
                               f"Warnings: {len(input_guardrail_results['warnings'])}, "
                               f"Failed: {len(input_guardrail_results['failed_checks'])}")
            
            # Step 2: Add session ID to system message if present
            session_id = get_session_id()
            if session_id:
                # Add session ID to the first system message, or create one
                messages_with_session = messages.copy()
                system_message_found = False
                for msg in messages_with_session:
                    if msg.get('role') == 'system':
                        # Append session ID to system message
                        current_content = msg.get('content', '')
                        if 'Session ID:' not in current_content:
                            msg['content'] = f"{current_content}\n\nSession ID: {session_id}"
                        system_message_found = True
                        break
                
                # If no system message, add one with session ID
                if not system_message_found:
                    messages_with_session.insert(0, {
                        'role': 'system',
                        'content': f'Session ID: {session_id}'
                    })
            else:
                messages_with_session = messages
            
            # Step 3: Make the actual API call
            start_time = time.time()
            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=messages_with_session,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            duration_ms = int((time.time() - start_time) * 1000)
            duration_seconds = (time.time() - start_time)
            
            # Extract response content
            response_content = ""
            if hasattr(response, 'choices') and len(response.choices) > 0:
                response_content = response.choices[0].message.content
            
            # Record metrics (before guardrails validation to capture all requests)
            if record_llm_request_from_response:
                try:
                    record_llm_request_from_response(
                        model=self.model_name,
                        response=response,
                        agent_name=agent_name,
                        duration_seconds=duration_seconds,
                        success=True
                    )
                except Exception as metrics_error:
                    logger.warning(f"Failed to record metrics: {metrics_error}")
            
            # Step 4: Validate output with guardrails
            guardrail_results = None
            if self.guardrails and hasattr(response, 'choices') and len(response.choices) > 0:
                output_text = response.choices[0].message.content
                is_valid, checks = self.guardrails.validate_output(
                    output_text,
                    expected_format=expected_output_format
                )
                
                # Store guardrail results for logging
                guardrail_results = {
                    'is_valid': is_valid,
                    'failed_checks': [c.message for c in checks if c.status == GuardrailResult.FAIL],
                    'warnings': [c.message for c in checks if c.status == GuardrailResult.WARNING],
                    'passed_checks': [c.message for c in checks if c.status == GuardrailResult.PASS]
                }
                
                if not is_valid:
                    failed_checks = [c for c in checks if c.status == GuardrailResult.FAIL]
                    error_msg = f"Output validation failed: {', '.join([c.message for c in failed_checks])}"
                    logger.error(error_msg)
                    # Note: We still return the response but log the failure
                    # You can choose to raise an exception here if needed
                
                # Log warnings if any
                warnings = [c for c in checks if c.status == GuardrailResult.WARNING]
                for warning in warnings:
                    logger.warning(f"Guardrail output warning: {warning.message}")
            
            # Log the completion (this will log prompts and outputs separately)
            # Get agent name from session context
            agent_name = None
            try:
                context = get_session_context()
                agent_name = context.get('agent_name')
            except Exception:
                pass
            
            try:
                # Add guardrail information to response if available
                if guardrail_results:
                    guardrail_info = f"\n\n[GUARDRAIL VALIDATION]\n"
                    guardrail_info += f"Status: {'PASSED' if guardrail_results['is_valid'] else 'FAILED'}\n"
                    if guardrail_results['failed_checks']:
                        guardrail_info += f"Failed Checks: {', '.join(guardrail_results['failed_checks'])}\n"
                    if guardrail_results['warnings']:
                        guardrail_info += f"Warnings: {', '.join(guardrail_results['warnings'])}\n"
                    if guardrail_results['passed_checks']:
                        guardrail_info += f"Passed Checks: {len(guardrail_results['passed_checks'])} checks passed\n"
                    
                    # Log guardrail info to main logger
                    logger.info(f"Guardrail validation results: {guardrail_info}")
                
                openai_logger.log_chat_completion(
                    model=self.model_name,
                    messages=messages_with_session,
                    response=response_content,
                    usage=response.usage.__dict__ if hasattr(response, 'usage') and response.usage else None,
                    duration_ms=duration_ms,
                    agent_name=agent_name
                )
            except Exception as log_error:
                logger.warning(f"Failed to log completion: {log_error}", exc_info=True)
            
            return response
            
        except ValueError:
            # Re-raise validation errors
            raise
        except Exception as e:
            # Record failed request metrics
            if record_llm_request_from_response:
                try:
                    duration_seconds = time.time() - start_time
                    record_llm_request_from_response(
                        model=self.model_name,
                        response=None,
                        agent_name=agent_name,
                        duration_seconds=duration_seconds,
                        success=False
                    )
                except Exception:
                    pass  # Don't fail on metrics errors
            
            logger.error(f"Error in async chat completion: {e}")
            raise


# Global wrapper instance
_llm_wrapper: Optional[LLMGuardrailsWrapper] = None


def initialize_llm_guardrails(api_key: str, model_name: str = "gpt-4o-mini", enable_guardrails: bool = True):
    """
    Initialize global LLM guardrails wrapper.
    
    Args:
        api_key: OpenAI API key
        model_name: Model name to use
        enable_guardrails: Whether to enable guardrails validation
    """
    global _llm_wrapper
    _llm_wrapper = LLMGuardrailsWrapper(api_key, model_name, enable_guardrails=enable_guardrails)
    logger.info("LLM guardrails wrapper initialized" if enable_guardrails else "LLM wrapper initialized")


def get_llm_wrapper() -> Optional[LLMGuardrailsWrapper]:
    """
    Get the global LLM guardrails wrapper instance.
    
    Returns:
        LLMGuardrailsWrapper instance or None if not initialized
    """
    return _llm_wrapper

