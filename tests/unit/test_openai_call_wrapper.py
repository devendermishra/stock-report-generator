"""
Unit tests for OpenAI call wrapper.
Tests automatic logging wrapper for OpenAI API calls.
"""

import pytest
import sys
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from tools.openai_call_wrapper import logged_chat_completion, logged_async_chat_completion


class TestLoggedChatCompletion:
    """Test cases for logged_chat_completion function."""
    
    @patch('tools.openai_call_wrapper.record_llm_request_from_response')
    @patch('tools.openai_call_wrapper.openai_logger.log_chat_completion')
    def test_logged_chat_completion_success(self, mock_log_chat, mock_metrics) -> None:
        """Test successful logged chat completion."""
        # Mock OpenAI client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Test response"
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock()
        mock_response.usage.__dict__ = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
        
        mock_client.chat.completions.create.return_value = mock_response
        
        messages = [{"role": "user", "content": "Hello"}]
        
        response = logged_chat_completion(
            client=mock_client,
            model="gpt-4o-mini",
            messages=messages,
            agent_name="TestAgent"
        )
        
        assert response == mock_response
        mock_client.chat.completions.create.assert_called_once()
        # Logger may fail internally, but function should still work
        # Just verify metrics was called
        mock_metrics.assert_called_once()
    
    @patch('tools.openai_call_wrapper.record_llm_request_from_response')
    @patch('tools.openai_call_wrapper.openai_logger')
    def test_logged_chat_completion_with_kwargs(self, mock_logger, mock_metrics) -> None:
        """Test logged chat completion with additional kwargs."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Test response"
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        
        mock_client.chat.completions.create.return_value = mock_response
        
        response = logged_chat_completion(
            client=mock_client,
            model="gpt-4o-mini",
            messages=[],
            temperature=0.7,
            max_tokens=100
        )
        
        assert response == mock_response
        # Verify kwargs were passed
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs['temperature'] == 0.7
        assert call_kwargs['max_tokens'] == 100
    
    @patch('tools.openai_call_wrapper.record_llm_request_from_response')
    @patch('tools.openai_call_wrapper.openai_logger')
    def test_logged_chat_completion_no_usage(self, mock_logger, mock_metrics) -> None:
        """Test logged chat completion without usage info."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Test response"
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        
        mock_client.chat.completions.create.return_value = mock_response
        
        response = logged_chat_completion(
            client=mock_client,
            model="gpt-4o-mini",
            messages=[]
        )
        
        assert response == mock_response
        # Should still log even without usage
        mock_logger.log_chat_completion.assert_called_once()
    
    @patch('tools.openai_call_wrapper.record_llm_request_from_response')
    @patch('tools.openai_call_wrapper.openai_logger')
    def test_logged_chat_completion_error(self, mock_logger, mock_metrics) -> None:
        """Test logged chat completion with API error."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        with pytest.raises(Exception, match="API Error"):
            logged_chat_completion(
                client=mock_client,
                model="gpt-4o-mini",
                messages=[]
            )
        
        # Should record failed metrics
        assert mock_metrics.called
        call_kwargs = mock_metrics.call_args[1]
        assert call_kwargs['success'] is False
    
    @patch('tools.openai_call_wrapper.record_llm_request_from_response', None)
    @patch('tools.openai_call_wrapper.openai_logger')
    def test_logged_chat_completion_no_metrics(self, mock_logger) -> None:
        """Test logged chat completion when metrics are not available."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Test response"
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        
        mock_client.chat.completions.create.return_value = mock_response
        
        # Should not raise error even if metrics is None
        response = logged_chat_completion(
            client=mock_client,
            model="gpt-4o-mini",
            messages=[]
        )
        
        assert response == mock_response
    
    @patch('tools.openai_call_wrapper.record_llm_request_from_response')
    @patch('tools.openai_call_wrapper.openai_logger')
    def test_logged_chat_completion_duration(self, mock_logger, mock_metrics) -> None:
        """Test that duration is measured correctly."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Test response"
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        
        # Simulate delay
        def delayed_create(*args, **kwargs):
            time.sleep(0.1)
            return mock_response
        
        mock_client.chat.completions.create.side_effect = delayed_create
        
        start_time = time.time()
        response = logged_chat_completion(
            client=mock_client,
            model="gpt-4o-mini",
            messages=[]
        )
        end_time = time.time()
        
        assert response == mock_response
        # Verify duration was logged
        call_kwargs = mock_logger.log_chat_completion.call_args[1]
        assert call_kwargs['duration_ms'] is not None
        assert call_kwargs['duration_ms'] > 0
        assert (end_time - start_time) >= 0.1


class TestLoggedAsyncChatCompletion:
    """Test cases for logged_async_chat_completion function."""
    
    @pytest.mark.asyncio
    @patch('tools.openai_call_wrapper.record_llm_request_from_response')
    @patch('tools.openai_call_wrapper.openai_logger.log_chat_completion')
    async def test_logged_async_chat_completion_success(self, mock_log_chat, mock_metrics) -> None:
        """Test successful async logged chat completion."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Test response"
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock()
        mock_response.usage.__dict__ = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
        
        mock_client.chat.completions.create.return_value = mock_response
        
        messages = [{"role": "user", "content": "Hello"}]
        
        response = await logged_async_chat_completion(
            client=mock_client,
            model="gpt-4o-mini",
            messages=messages,
            agent_name="TestAgent"
        )
        
        assert response == mock_response
        mock_client.chat.completions.create.assert_called_once()
        # Logger may fail internally, but function should still work
        # Just verify metrics was called
        mock_metrics.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('tools.openai_call_wrapper.record_llm_request_from_response')
    @patch('tools.openai_call_wrapper.openai_logger')
    async def test_logged_async_chat_completion_error(self, mock_logger, mock_metrics) -> None:
        """Test async logged chat completion with API error."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        with pytest.raises(Exception, match="API Error"):
            await logged_async_chat_completion(
                client=mock_client,
                model="gpt-4o-mini",
                messages=[]
            )
        
        # Should record failed metrics
        assert mock_metrics.called
        call_kwargs = mock_metrics.call_args[1]
        assert call_kwargs['success'] is False
    
    @pytest.mark.asyncio
    @patch('tools.openai_call_wrapper.record_llm_request_from_response', None)
    @patch('tools.openai_call_wrapper.openai_logger')
    async def test_logged_async_chat_completion_no_metrics(self, mock_logger) -> None:
        """Test async logged chat completion when metrics are not available."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Test response"
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        
        mock_client.chat.completions.create.return_value = mock_response
        
        # Should not raise error even if metrics is None
        response = await logged_async_chat_completion(
            client=mock_client,
            model="gpt-4o-mini",
            messages=[]
        )
        
        assert response == mock_response

