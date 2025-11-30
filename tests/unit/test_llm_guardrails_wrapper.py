"""
Unit tests for LLM guardrails wrapper.
Tests LLM calls with guardrails validation.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from tools.llm_guardrails_wrapper import (
    LLMGuardrailsWrapper,
    initialize_llm_guardrails,
    get_llm_wrapper
)


class TestLLMGuardrailsWrapper:
    """Test cases for LLMGuardrailsWrapper class."""
    
    def setup_method(self) -> None:
        """Set up test fixtures."""
        with patch('tools.llm_guardrails_wrapper.GUARDRAILS_AVAILABLE', True):
            with patch('tools.llm_guardrails_wrapper.initialize_guardrails') as mock_init:
                mock_guardrails = MagicMock()
                mock_init.return_value = mock_guardrails
                self.wrapper = LLMGuardrailsWrapper(
                    api_key="test-key",
                    model_name="gpt-4o-mini",
                    enable_guardrails=True
                )
    
    def test_initialization(self) -> None:
        """Test wrapper initialization."""
        assert self.wrapper is not None
        assert self.wrapper.api_key == "test-key"
        assert self.wrapper.model_name == "gpt-4o-mini"
        assert self.wrapper.enable_guardrails is True
    
    @patch('tools.llm_guardrails_wrapper.get_session_id')
    @patch('tools.llm_guardrails_wrapper.get_session_context')
    @patch('tools.llm_guardrails_wrapper.record_llm_request_from_response')
    @patch('tools.llm_guardrails_wrapper.openai_logger.log_chat_completion')
    def test_chat_completion_success(self, mock_log_chat, mock_metrics, 
                                     mock_get_context, mock_get_session) -> None:
        """Test successful chat completion with guardrails."""
        mock_get_session.return_value = "test-session"
        mock_get_context.return_value = {}
        
        # Mock guardrails validation
        self.wrapper.guardrails = MagicMock()
        self.wrapper.guardrails.validate_input.return_value = (True, [])
        self.wrapper.guardrails.validate_output.return_value = (True, [])
        
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Test response"
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock()
        mock_response.usage.__dict__ = {"prompt_tokens": 10, "completion_tokens": 20}
        
        # Mock the client's chat.completions.create method
        self.wrapper.client = MagicMock()
        self.wrapper.client.chat.completions.create.return_value = mock_response
        
        messages = [{"role": "user", "content": "Hello"}]
        response = self.wrapper.chat_completion(messages)
        
        assert response == mock_response
        self.wrapper.guardrails.validate_input.assert_called_once()
        self.wrapper.guardrails.validate_output.assert_called_once()
        # Logger may fail internally, but function should still work
        # Just verify the response is correct
    
    def test_chat_completion_input_validation_fails(self) -> None:
        """Test chat completion when input validation fails."""
        # Mock guardrails validation failure
        self.wrapper.guardrails = MagicMock()
        from tools.guardrails import GuardrailResult, GuardrailCheck
        failed_check = GuardrailCheck(
            name="test",
            status=GuardrailResult.FAIL,
            message="Input validation failed"
        )
        self.wrapper.guardrails.validate_input.return_value = (False, [failed_check])
        
        messages = [{"role": "user", "content": "Malicious input"}]
        
        with pytest.raises(ValueError, match="Input validation failed"):
            self.wrapper.chat_completion(messages)
    
    @patch('tools.llm_guardrails_wrapper.get_session_id')
    @patch('tools.llm_guardrails_wrapper.get_session_context')
    @patch('tools.llm_guardrails_wrapper.record_llm_request_from_response')
    @patch('tools.llm_guardrails_wrapper.openai_logger')
    def test_chat_completion_output_validation_fails(self, mock_logger, mock_metrics,
                                                     mock_get_context, mock_get_session) -> None:
        """Test chat completion when output validation fails."""
        mock_get_session.return_value = None
        mock_get_context.return_value = {}
        
        # Mock guardrails validation - input passes, output fails
        self.wrapper.guardrails = MagicMock()
        from tools.guardrails import GuardrailResult, GuardrailCheck
        self.wrapper.guardrails.validate_input.return_value = (True, [])
        failed_check = GuardrailCheck(
            name="output_validation",
            status=GuardrailResult.FAIL,
            message="Output validation failed"
        )
        self.wrapper.guardrails.validate_output.return_value = (False, [failed_check])
        
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Invalid output"
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        
        # Mock the client's chat.completions.create method
        self.wrapper.client = MagicMock()
        self.wrapper.client.chat.completions.create.return_value = mock_response
        
        messages = [{"role": "user", "content": "Hello"}]
        
        # Should still return response but log failure
        response = self.wrapper.chat_completion(messages)
        
        assert response == mock_response
        # Should log the validation failure
        assert self.wrapper.guardrails.validate_output.called
    
    @patch('tools.llm_guardrails_wrapper.get_session_id')
    def test_chat_completion_adds_session_id(self, mock_get_session) -> None:
        """Test that session ID is added to messages."""
        mock_get_session.return_value = "test-session-123"
        
        self.wrapper.guardrails = MagicMock()
        self.wrapper.guardrails.validate_input.return_value = (True, [])
        self.wrapper.guardrails.validate_output.return_value = (True, [])
        
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Response"
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        
        # Mock the client
        self.wrapper.client = MagicMock()
        self.wrapper.client.chat.completions.create.return_value = mock_response
        
        messages = [{"role": "user", "content": "Hello"}]
        self.wrapper.chat_completion(messages)
        
        # Verify session ID was added to messages
        call_args = self.wrapper.client.chat.completions.create.call_args
        call_messages = call_args[1]['messages']
        
        # Should have system message with session ID
        system_messages = [m for m in call_messages if m.get('role') == 'system']
        assert len(system_messages) > 0
        assert "test-session-123" in system_messages[0]['content']
    
    @patch('tools.llm_guardrails_wrapper.get_session_id')
    def test_chat_completion_no_session_id(self, mock_get_session) -> None:
        """Test chat completion when no session ID is set."""
        mock_get_session.return_value = None
        
        self.wrapper.guardrails = MagicMock()
        self.wrapper.guardrails.validate_input.return_value = (True, [])
        self.wrapper.guardrails.validate_output.return_value = (True, [])
        
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Response"
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        
        # Mock the client
        self.wrapper.client = MagicMock()
        self.wrapper.client.chat.completions.create.return_value = mock_response
        
        messages = [{"role": "user", "content": "Hello"}]
        self.wrapper.chat_completion(messages)
        
        # Messages should not have session ID added (no system message)
        call_args = self.wrapper.client.chat.completions.create.call_args
        call_messages = call_args[1]['messages']
        # Should have original message (no system message added)
        assert len(call_messages) >= 1
    
    def test_chat_completion_guardrails_disabled(self) -> None:
        """Test chat completion when guardrails are disabled."""
        wrapper = LLMGuardrailsWrapper(
            api_key="test-key",
            model_name="gpt-4o-mini",
            enable_guardrails=False
        )
        
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Response"
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        
        # Mock the client
        wrapper.client = MagicMock()
        wrapper.client.chat.completions.create.return_value = mock_response
        
        messages = [{"role": "user", "content": "Hello"}]
        response = wrapper.chat_completion(messages)
        
        assert response == mock_response
        # Guardrails should not be called
        assert wrapper.guardrails is None
    
    @pytest.mark.asyncio
    @patch('tools.llm_guardrails_wrapper.get_session_id')
    @patch('tools.llm_guardrails_wrapper.get_session_context')
    @patch('tools.llm_guardrails_wrapper.record_llm_request_from_response')
    @patch('tools.llm_guardrails_wrapper.openai_logger')
    async def test_async_chat_completion_success(self, mock_logger, mock_metrics,
                                                  mock_get_context, mock_get_session) -> None:
        """Test successful async chat completion with guardrails."""
        mock_get_session.return_value = "test-session"
        mock_get_context.return_value = {}
        
        # Mock guardrails validation
        self.wrapper.guardrails = MagicMock()
        self.wrapper.guardrails.validate_input.return_value = (True, [])
        self.wrapper.guardrails.validate_output.return_value = (True, [])
        
        # Mock OpenAI async response
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Test response"
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock()
        mock_response.usage.__dict__ = {"prompt_tokens": 10, "completion_tokens": 20}
        
        self.wrapper.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        messages = [{"role": "user", "content": "Hello"}]
        response = await self.wrapper.async_chat_completion(messages)
        
        assert response == mock_response
        self.wrapper.guardrails.validate_input.assert_called_once()
        self.wrapper.guardrails.validate_output.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_async_chat_completion_input_validation_fails(self) -> None:
        """Test async chat completion when input validation fails."""
        # Mock guardrails validation failure
        self.wrapper.guardrails = MagicMock()
        from tools.guardrails import GuardrailResult, GuardrailCheck
        failed_check = GuardrailCheck(
            name="test",
            status=GuardrailResult.FAIL,
            message="Input validation failed"
        )
        self.wrapper.guardrails.validate_input.return_value = (False, [failed_check])
        
        messages = [{"role": "user", "content": "Malicious input"}]
        
        with pytest.raises(ValueError, match="Input validation failed"):
            await self.wrapper.async_chat_completion(messages)
    
    def test_chat_completion_with_kwargs(self) -> None:
        """Test chat completion with additional kwargs."""
        self.wrapper.guardrails = MagicMock()
        self.wrapper.guardrails.validate_input.return_value = (True, [])
        self.wrapper.guardrails.validate_output.return_value = (True, [])
        
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Response"
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        
        # Mock the client
        self.wrapper.client = MagicMock()
        self.wrapper.client.chat.completions.create.return_value = mock_response
        
        messages = [{"role": "user", "content": "Hello"}]
        response = self.wrapper.chat_completion(
            messages,
            temperature=0.7,
            max_tokens=200
        )
        
        assert response == mock_response
        # Verify kwargs were passed
        call_kwargs = self.wrapper.client.chat.completions.create.call_args[1]
        assert call_kwargs['temperature'] == 0.7
        assert call_kwargs['max_tokens'] == 200
    
    def test_chat_completion_api_error(self) -> None:
        """Test chat completion when API call fails."""
        self.wrapper.guardrails = MagicMock()
        self.wrapper.guardrails.validate_input.return_value = (True, [])
        
        # Mock the client
        self.wrapper.client = MagicMock()
        self.wrapper.client.chat.completions.create.side_effect = Exception("API Error")
        
        messages = [{"role": "user", "content": "Hello"}]
        
        with pytest.raises(Exception, match="API Error"):
            self.wrapper.chat_completion(messages)


class TestGuardrailsWrapperFunctions:
    """Test cases for guardrails wrapper module functions."""
    
    def test_initialize_llm_guardrails(self) -> None:
        """Test initializing LLM guardrails wrapper."""
        with patch('tools.llm_guardrails_wrapper.GUARDRAILS_AVAILABLE', True):
            initialize_llm_guardrails(
                api_key="test-key",
                model_name="gpt-4o-mini",
                enable_guardrails=True
            )
            
            wrapper = get_llm_wrapper()
            assert wrapper is not None
            assert isinstance(wrapper, LLMGuardrailsWrapper)
            assert wrapper.api_key == "test-key"
            assert wrapper.model_name == "gpt-4o-mini"
    
    def test_get_llm_wrapper_not_initialized(self) -> None:
        """Test getting wrapper when not initialized."""
        # Reset global
        import tools.llm_guardrails_wrapper as wrapper_module
        wrapper_module._llm_wrapper = None
        
        wrapper = get_llm_wrapper()
        assert wrapper is None
    
    def test_get_llm_wrapper_singleton(self) -> None:
        """Test that get_llm_wrapper returns singleton."""
        with patch('tools.llm_guardrails_wrapper.GUARDRAILS_AVAILABLE', True):
            initialize_llm_guardrails(
                api_key="test-key",
                model_name="gpt-4o-mini"
            )
            
            wrapper1 = get_llm_wrapper()
            wrapper2 = get_llm_wrapper()
            assert wrapper1 is wrapper2

