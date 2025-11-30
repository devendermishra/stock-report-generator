"""
Unit tests for OpenAI logger.
Tests OpenAI chat completion logging functionality.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from tools.openai_logger import OpenAILogger, ChatLogEntry


class TestChatLogEntry:
    """Test cases for ChatLogEntry dataclass."""
    
    def test_chat_log_entry_creation(self) -> None:
        """Test creating a ChatLogEntry."""
        entry = ChatLogEntry(
            timestamp=datetime.now(),
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            response="Hi there",
            tokens_used=100,
            cost_estimate=0.01,
            duration_ms=500,
            session_id="test-session",
            agent_name="TestAgent",
            stock_symbol="TCS"
        )
        
        assert entry.model == "gpt-4o-mini"
        assert entry.response == "Hi there"
        assert entry.tokens_used == 100
        assert entry.cost_estimate == 0.01
        assert entry.session_id == "test-session"
        assert entry.agent_name == "TestAgent"
        assert entry.stock_symbol == "TCS"


class TestOpenAILogger:
    """Test cases for OpenAILogger class."""
    
    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.logger = OpenAILogger()
    
    def test_initialization(self) -> None:
        """Test logger initialization."""
        assert self.logger is not None
        assert self.logger.token_costs is not None
        assert "gpt-4o-mini" in self.logger.token_costs
    
    def test_token_costs(self) -> None:
        """Test token cost configuration."""
        assert "gpt-4o" in self.logger.token_costs
        assert "gpt-4o-mini" in self.logger.token_costs
        assert "gpt-3.5-turbo" in self.logger.token_costs
        
        # Check cost structure
        for model, costs in self.logger.token_costs.items():
            assert "input" in costs
            assert "output" in costs
    
    @patch('tools.openai_logger.get_session_id')
    @patch('tools.openai_logger.get_session_context')
    @patch('tools.openai_logger.get_prompt_logger')
    @patch('tools.openai_logger.get_output_logger')
    def test_log_chat_completion(self, mock_output_logger, mock_prompt_logger, 
                                  mock_get_context, mock_get_session) -> None:
        """Test logging a chat completion."""
        mock_get_session.return_value = "test-session-123"
        mock_get_context.return_value = {"stock_symbol": "TCS"}
        
        mock_prompt_log = MagicMock()
        mock_prompt_log.handlers = [MagicMock()]
        mock_prompt_logger.return_value = mock_prompt_log
        
        mock_output_log = MagicMock()
        mock_output_log.handlers = [MagicMock()]
        mock_output_logger.return_value = mock_output_log
        
        messages = [{"role": "user", "content": "What is TCS?"}]
        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
        
        entry = self.logger.log_chat_completion(
            model="gpt-4o-mini",
            messages=messages,
            response="TCS is a company",
            usage=usage,
            duration_ms=500,
            agent_name="TestAgent"
        )
        
        assert entry is not None
        assert entry.model == "gpt-4o-mini"
        assert entry.session_id == "test-session-123"
        assert entry.stock_symbol == "TCS"
        assert entry.agent_name == "TestAgent"
        assert entry.tokens_used == 30
        assert entry.cost_estimate is not None
        assert entry.cost_estimate > 0
    
    @patch('tools.openai_logger.get_session_id')
    @patch('tools.openai_logger.get_session_context')
    @patch('tools.openai_logger.get_prompt_logger')
    @patch('tools.openai_logger.get_output_logger')
    def test_log_chat_completion_no_usage(self, mock_output_logger, mock_prompt_logger,
                                           mock_get_context, mock_get_session) -> None:
        """Test logging without usage information."""
        mock_get_session.return_value = None
        mock_get_context.return_value = {}
        
        mock_prompt_log = MagicMock()
        mock_prompt_log.handlers = [MagicMock()]
        mock_prompt_logger.return_value = mock_prompt_log
        
        mock_output_log = MagicMock()
        mock_output_log.handlers = [MagicMock()]
        mock_output_logger.return_value = mock_output_log
        
        entry = self.logger.log_chat_completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            response="Hi",
            usage=None,
            duration_ms=None,
            agent_name=None
        )
        
        assert entry is not None
        assert entry.tokens_used is None
        assert entry.cost_estimate is None
    
    @patch('tools.openai_logger.get_session_id')
    @patch('tools.openai_logger.get_session_context')
    @patch('tools.openai_logger.get_prompt_logger')
    @patch('tools.openai_logger.get_output_logger')
    def test_log_chat_completion_error_handling(self, mock_output_logger, mock_prompt_logger,
                                                 mock_get_context, mock_get_session) -> None:
        """Test error handling in log_chat_completion."""
        mock_get_session.side_effect = Exception("Session error")
        
        # Should not raise, but return a basic entry
        entry = self.logger.log_chat_completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            response="Hi"
        )
        
        assert entry is not None
        assert entry.model == "gpt-4o-mini"
    
    @patch('tools.openai_logger.get_prompt_logger')
    def test_log_prompt(self, mock_prompt_logger) -> None:
        """Test logging prompt separately."""
        mock_prompt_log = MagicMock()
        mock_handler = MagicMock()
        mock_prompt_log.handlers = [mock_handler]
        mock_prompt_logger.return_value = mock_prompt_log
        
        entry = ChatLogEntry(
            timestamp=datetime.now(),
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Test"}],
            response="Response",
            session_id="test-session",
            agent_name="TestAgent",
            stock_symbol="TCS"
        )
        
        self.logger._log_prompt(entry, "TestAgent")
        
        # Verify logger was called
        assert mock_prompt_log.info.called
    
    @patch('tools.openai_logger.get_output_logger')
    def test_log_output(self, mock_output_logger) -> None:
        """Test logging output separately."""
        mock_output_log = MagicMock()
        mock_handler = MagicMock()
        mock_output_log.handlers = [mock_handler]
        mock_output_logger.return_value = mock_output_log
        
        entry = ChatLogEntry(
            timestamp=datetime.now(),
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Test"}],
            response="Response",
            tokens_used=100,
            cost_estimate=0.01,
            duration_ms=500,
            session_id="test-session",
            agent_name="TestAgent",
            stock_symbol="TCS"
        )
        
        self.logger._log_output(entry, "TestAgent")
        
        # Verify logger was called
        assert mock_output_log.info.called
    
    def test_log_error(self) -> None:
        """Test logging errors."""
        with patch.object(self.logger.logger, 'error') as mock_error:
            error = ValueError("Test error")
            self.logger.log_error(error, "gpt-4o-mini", "TestAgent")
            
            mock_error.assert_called_once()
            assert "Test error" in str(mock_error.call_args)
    
    def test_get_usage_summary(self) -> None:
        """Test getting usage summary from log entries."""
        entries = [
            ChatLogEntry(
                timestamp=datetime.now(),
                model="gpt-4o-mini",
                messages=[],
                response="Response 1",
                tokens_used=100,
                cost_estimate=0.01,
                duration_ms=500
            ),
            ChatLogEntry(
                timestamp=datetime.now(),
                model="gpt-4o-mini",
                messages=[],
                response="Response 2",
                tokens_used=200,
                cost_estimate=0.02,
                duration_ms=1000
            ),
            ChatLogEntry(
                timestamp=datetime.now(),
                model="gpt-4o",
                messages=[],
                response="Response 3",
                tokens_used=150,
                cost_estimate=0.05,
                duration_ms=800
            )
        ]
        
        summary = self.logger.get_usage_summary(entries)
        
        assert summary is not None
        assert summary['total_requests'] == 3
        assert summary['total_tokens'] == 450
        assert summary['total_cost'] == 0.08
        assert summary['total_duration_ms'] == 2300
        assert 'gpt-4o-mini' in summary['model_usage']
        assert 'gpt-4o' in summary['model_usage']
        assert summary['model_usage']['gpt-4o-mini']['count'] == 2
        assert summary['model_usage']['gpt-4o']['count'] == 1
    
    def test_get_usage_summary_empty(self) -> None:
        """Test getting usage summary from empty list."""
        summary = self.logger.get_usage_summary([])
        
        assert summary is not None
        assert summary['total_requests'] == 0
        assert summary['total_tokens'] == 0
        assert summary['total_cost'] == 0
    
    def test_get_usage_summary_error_handling(self) -> None:
        """Test error handling in get_usage_summary."""
        # Invalid entry
        entries = [None]  # type: ignore
        
        with patch.object(self.logger.logger, 'error'):
            summary = self.logger.get_usage_summary(entries)
            assert summary == {}
    
    def test_cost_calculation(self) -> None:
        """Test cost calculation for different models."""
        usage = {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "total_tokens": 1500
        }
        
        # Test gpt-4o-mini
        entry = self.logger.log_chat_completion(
            model="gpt-4o-mini",
            messages=[],
            response="Test",
            usage=usage
        )
        
        # Cost should be calculated
        assert entry.cost_estimate is not None
        assert entry.cost_estimate > 0
        
        # Test gpt-4o
        entry2 = self.logger.log_chat_completion(
            model="gpt-4o",
            messages=[],
            response="Test",
            usage=usage
        )
        
        # gpt-4o should have higher cost
        assert entry2.cost_estimate > entry.cost_estimate

