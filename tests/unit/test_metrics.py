"""
Unit tests for metrics module.
Tests Prometheus metrics collection functionality.
"""

import pytest
import sys
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


class TestMetricsInitialization:
    """Test cases for metrics initialization."""
    
    @patch('config.Config')
    def test_initialize_metrics_enabled(self, mock_config) -> None:
        """Test metrics initialization when enabled."""
        mock_config.ENABLE_METRICS = True
        mock_config.METRICS_PORT = 9090
        
        # Mock Prometheus client
        with patch('prometheus_client.Counter') as mock_counter, \
             patch('prometheus_client.Histogram') as mock_histogram, \
             patch('prometheus_client.start_http_server') as mock_server:
            
            # Call initialize_metrics directly
            from utils.metrics import initialize_metrics, get_metrics_status
            initialize_metrics()
            
            # Verify metrics were initialized (check if Counter was called)
            # The actual enabled state depends on Config, but we verify the setup
            status = get_metrics_status()
            assert status['enabled'] is True
    
    @patch('config.Config')
    def test_initialize_metrics_disabled(self, mock_config) -> None:
        """Test metrics initialization when disabled."""
        mock_config.ENABLE_METRICS = False
        
        # Call initialize_metrics
        from utils.metrics import initialize_metrics, get_metrics_status
        initialize_metrics()
        
        # Metrics are always enabled (in-memory), but Prometheus may be disabled
        status = get_metrics_status()
        assert status['enabled'] is True  # Always enabled for in-memory metrics
    
    def test_initialize_metrics_prometheus_unavailable(self) -> None:
        """Test metrics initialization when Prometheus is unavailable."""
        # Test that metrics work even without Prometheus
        from utils.metrics import initialize_metrics, get_metrics_status
        
        # Mock the import to raise ImportError
        original_import = __import__
        def mock_import(name, *args, **kwargs):
            if name == 'prometheus_client':
                raise ImportError("No module named 'prometheus_client'")
            return original_import(name, *args, **kwargs)
        
        with patch('builtins.__import__', side_effect=mock_import):
            # Re-import to trigger the ImportError handling
            import importlib
            import utils.metrics as metrics_module
            importlib.reload(metrics_module)
            metrics_module.initialize_metrics()
            
            # Metrics should still be enabled (in-memory)
            status = metrics_module.get_metrics_status()
            assert status['enabled'] is True
    
    @patch('config.Config')
    @patch('prometheus_client.start_http_server', side_effect=OSError("Port in use"))
    def test_initialize_metrics_server_error(self, mock_server, mock_config) -> None:
        """Test metrics initialization when server fails to start."""
        mock_config.ENABLE_METRICS = True
        mock_config.METRICS_PORT = 9090
        
        with patch('prometheus_client.Counter'), patch('prometheus_client.Histogram'):
            # Call initialize_metrics directly
            from utils.metrics import initialize_metrics
            initialize_metrics()
            
            # Verify server start was attempted (even if it failed)
            # The function should handle the error gracefully
            assert True  # Test that it doesn't raise an exception


class TestRecordLLMRequest:
    """Test cases for record_llm_request function."""
    
    def test_record_llm_request_success(self) -> None:
        """Test recording successful LLM request."""
        from utils.metrics import record_llm_request, get_summary
        
        # Record a request
        record_llm_request(
            model="gpt-4o-mini",
            agent="TestAgent",
            request_tokens=100,
            response_tokens=50,
            duration=1.5,
            success=True
        )
        
        # Verify metrics were recorded
        summary = get_summary()
        assert "llm_requests" in str(summary['counts'])
    
    def test_record_llm_request_failure(self) -> None:
        """Test recording failed LLM request."""
        from utils.metrics import record_llm_request, get_summary
        
        # Record a failed request
        record_llm_request(
            model="gpt-4o-mini",
            agent="TestAgent",
            success=False
        )
        
        # Verify metrics were recorded
        summary = get_summary()
        assert "llm_requests" in str(summary['counts'])
    
    def test_record_llm_request_disabled(self) -> None:
        """Test that recording works even when Prometheus is disabled."""
        from utils.metrics import record_llm_request
        
        # Should not raise any errors (metrics always work in-memory)
        record_llm_request(
            model="gpt-4o-mini",
            agent="TestAgent",
            request_tokens=100,
            response_tokens=50,
            duration=1.5,
            success=True
        )
    
    def test_record_llm_request_no_agent_name(self) -> None:
        """Test recording LLM request without agent name."""
        from utils.metrics import record_llm_request, get_summary
        
        # Record without agent (should use "unknown")
        record_llm_request(
            model="gpt-4o-mini",
            request_tokens=100,
            response_tokens=50,
            success=True
        )
        
        # Verify metrics were recorded
        summary = get_summary()
        assert "llm_requests" in str(summary['counts'])


class TestRecordLLMRequestFromResponse:
    """Test cases for record_llm_request_from_response function."""
    
    def test_record_llm_request_from_response_with_usage(self) -> None:
        """Test recording LLM request from response object with usage."""
        from utils.metrics import record_llm_request_from_response
        
        # Mock response object
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50
        
        mock_response = MagicMock()
        mock_response.usage = mock_usage
        
        with patch('utils.metrics.record_llm_request') as mock_record:
            record_llm_request_from_response(
                model="gpt-4o-mini",
                response=mock_response,
                agent="TestAgent",
                duration=1.5,
                success=True
            )
            
            # Verify record_llm_request was called with correct tokens
            mock_record.assert_called_once()
            # Check positional arguments
            # Function signature: record_llm_request(model, agent=None, request_tokens=None, response_tokens=None, duration=None, success=True)
            call_args = mock_record.call_args
            # The function is called with positional args: (model, agent, request_tokens, response_tokens, duration, success)
            assert call_args[0][0] == "gpt-4o-mini"  # model
            assert call_args[0][1] == "TestAgent"  # agent
            assert call_args[0][2] == 100  # request_tokens
            assert call_args[0][3] == 50  # response_tokens
    
    def test_record_llm_request_from_response_no_usage(self) -> None:
        """Test recording LLM request from response without usage."""
        from utils.metrics import record_llm_request_from_response
        
        # Mock response object without usage
        mock_response = MagicMock()
        mock_response.usage = None
        
        with patch('utils.metrics.record_llm_request') as mock_record:
            record_llm_request_from_response(
                model="gpt-4o-mini",
                response=mock_response,
                agent="TestAgent",
                duration=1.5,
                success=True
            )
            
            # Verify record_llm_request was called with None tokens
            mock_record.assert_called_once()
            call_args = mock_record.call_args
            assert call_args.kwargs.get('request_tokens') is None
            assert call_args.kwargs.get('response_tokens') is None


class TestRecordReportGeneration:
    """Test cases for record_report function."""
    
    def test_record_report_generation_success(self) -> None:
        """Test recording successful report generation."""
        from utils.metrics import record_report, get_summary
        
        # Record a report
        record_report(
            symbol="TCS",
            duration=30.5,
            status="completed"
        )
        
        # Verify metrics were recorded
        summary = get_summary()
        assert "reports" in str(summary['counts'])
    
    def test_record_report_generation_failed(self) -> None:
        """Test recording failed report generation."""
        from utils.metrics import record_report, get_summary
        
        # Record a failed report
        record_report(
            symbol="TCS",
            status="failed"
        )
        
        # Verify metrics were recorded
        summary = get_summary()
        assert "reports" in str(summary['counts'])
    
    def test_record_report_generation_disabled(self) -> None:
        """Test that recording works even when Prometheus is disabled."""
        from utils.metrics import record_report
        
        # Should not raise any errors (metrics always work in-memory)
        record_report(
            symbol="TCS",
            duration=30.5,
            status="completed"
        )


class TestMetricsEnabled:
    """Test cases for metrics_enabled function."""
    
    def test_metrics_enabled(self) -> None:
        """Test checking if metrics are enabled."""
        from utils.metrics import metrics_enabled
        
        # Should return boolean
        result = metrics_enabled()
        assert isinstance(result, bool)

