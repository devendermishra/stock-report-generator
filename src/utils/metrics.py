"""
Prometheus Metrics Module for Stock Report Generator.

Provides metrics collection for:
- LLM token counts (request/response)
- LLM request counts (total/success/failed)
- LLM request duration
- Report generation counts and duration

Metrics are config-controlled and disabled by default.
"""

import logging
import time
from typing import Optional, Dict, Any
from functools import wraps

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("prometheus_client not available. Metrics will be disabled.")

try:
    from ..config import Config
except ImportError:
    from config import Config

logger = logging.getLogger(__name__)

# Metrics instances (initialized only if enabled)
_llm_token_count_request: Optional[Counter] = None
_llm_token_count_response: Optional[Counter] = None
_llm_requests_total: Optional[Counter] = None
_llm_requests_success: Optional[Counter] = None
_llm_requests_failed: Optional[Counter] = None
_llm_request_duration: Optional[Histogram] = None
_report_generation_count: Optional[Counter] = None
_report_generation_duration: Optional[Histogram] = None

_metrics_enabled = False
_metrics_server_started = False


def initialize_metrics():
    """Initialize Prometheus metrics if enabled in config."""
    global _llm_token_count_request, _llm_token_count_response
    global _llm_requests_total, _llm_requests_success, _llm_requests_failed
    global _llm_request_duration, _report_generation_count, _report_generation_duration
    global _metrics_enabled, _metrics_server_started
    
    if not Config.ENABLE_METRICS:
        logger.info("Metrics are disabled in configuration")
        return
    
    if not PROMETHEUS_AVAILABLE:
        logger.warning("Prometheus client not available. Metrics will be disabled.")
        return
    
    try:
        # LLM Token Count Metrics
        _llm_token_count_request = Counter(
            'llm_token_count_request',
            'Total number of tokens in LLM requests',
            ['model', 'agent_name']
        )
        
        _llm_token_count_response = Counter(
            'llm_token_count_response',
            'Total number of tokens in LLM responses',
            ['model', 'agent_name']
        )
        
        # LLM Request Count Metrics
        _llm_requests_total = Counter(
            'llm_requests_total',
            'Total number of LLM requests',
            ['model', 'agent_name']
        )
        
        _llm_requests_success = Counter(
            'llm_requests_success',
            'Total number of successful LLM requests',
            ['model', 'agent_name']
        )
        
        _llm_requests_failed = Counter(
            'llm_requests_failed',
            'Total number of failed LLM requests',
            ['model', 'agent_name']
        )
        
        # LLM Request Duration
        _llm_request_duration = Histogram(
            'llm_request_duration_seconds',
            'Time taken for LLM requests in seconds',
            ['model', 'agent_name'],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
        )
        
        # Report Generation Metrics
        _report_generation_count = Counter(
            'report_generation_count',
            'Total number of reports generated',
            ['stock_symbol', 'status']
        )
        
        _report_generation_duration = Histogram(
            'report_generation_duration_seconds',
            'Time taken to generate reports in seconds',
            ['stock_symbol'],
            buckets=(10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0)
        )
        
        _metrics_enabled = True
        logger.info("Prometheus metrics initialized successfully")
        
        # Start metrics HTTP server if not already started
        if not _metrics_server_started:
            try:
                start_http_server(Config.METRICS_PORT)
                _metrics_server_started = True
                logger.info(f"Prometheus metrics server started on port {Config.METRICS_PORT}")
            except OSError as e:
                logger.warning(f"Failed to start metrics server on port {Config.METRICS_PORT}: {e}")
                logger.warning("Metrics will still be collected but not exposed via HTTP")
        
    except Exception as e:
        logger.error(f"Failed to initialize metrics: {e}")
        _metrics_enabled = False


def record_llm_request(
    model: str,
    agent_name: Optional[str] = None,
    request_tokens: Optional[int] = None,
    response_tokens: Optional[int] = None,
    duration_seconds: Optional[float] = None,
    success: bool = True
):
    """
    Record LLM request metrics.
    
    Args:
        model: Model name (e.g., 'gpt-4o-mini')
        agent_name: Name of the agent making the request (optional)
        request_tokens: Number of tokens in the request
        response_tokens: Number of tokens in the response
        duration_seconds: Duration of the request in seconds
        success: Whether the request was successful
    """
    if not _metrics_enabled:
        return
    
    agent_label = agent_name or "unknown"
    
    try:
        # Increment total requests
        if _llm_requests_total:
            _llm_requests_total.labels(model=model, agent_name=agent_label).inc()
        
        # Increment success/failed counters
        if success:
            if _llm_requests_success:
                _llm_requests_success.labels(model=model, agent_name=agent_label).inc()
        else:
            if _llm_requests_failed:
                _llm_requests_failed.labels(model=model, agent_name=agent_label).inc()
        
        # Record token counts
        if request_tokens is not None and _llm_token_count_request:
            _llm_token_count_request.labels(model=model, agent_name=agent_label).inc(request_tokens)
        
        if response_tokens is not None and _llm_token_count_response:
            _llm_token_count_response.labels(model=model, agent_name=agent_label).inc(response_tokens)
        
        # Record duration
        if duration_seconds is not None and _llm_request_duration:
            _llm_request_duration.labels(model=model, agent_name=agent_label).observe(duration_seconds)
            
    except Exception as e:
        logger.warning(f"Failed to record LLM metrics: {e}")


def record_llm_request_from_response(
    model: str,
    response: Any,
    agent_name: Optional[str] = None,
    duration_seconds: Optional[float] = None,
    success: bool = True
):
    """
    Record LLM request metrics from an OpenAI response object.
    
    Args:
        model: Model name
        response: OpenAI response object (with usage attribute)
        agent_name: Name of the agent making the request
        duration_seconds: Duration of the request in seconds
        success: Whether the request was successful
    """
    request_tokens = None
    response_tokens = None
    
    # Extract token counts from response if available
    if hasattr(response, 'usage') and response.usage:
        if hasattr(response.usage, 'prompt_tokens'):
            request_tokens = response.usage.prompt_tokens
        if hasattr(response.usage, 'completion_tokens'):
            response_tokens = response.usage.completion_tokens
    
    record_llm_request(
        model=model,
        agent_name=agent_name,
        request_tokens=request_tokens,
        response_tokens=response_tokens,
        duration_seconds=duration_seconds,
        success=success
    )


def record_report_generation(
    stock_symbol: str,
    duration_seconds: Optional[float] = None,
    status: str = "completed"
):
    """
    Record report generation metrics.
    
    Args:
        stock_symbol: Stock symbol for the report
        duration_seconds: Duration of report generation in seconds
        status: Status of report generation ('completed', 'failed', etc.)
    """
    if not _metrics_enabled:
        return
    
    try:
        # Increment report count
        if _report_generation_count:
            _report_generation_count.labels(stock_symbol=stock_symbol, status=status).inc()
        
        # Record duration
        if duration_seconds is not None and _report_generation_duration:
            _report_generation_duration.labels(stock_symbol=stock_symbol).observe(duration_seconds)
            
    except Exception as e:
        logger.warning(f"Failed to record report generation metrics: {e}")


def metrics_enabled() -> bool:
    """Check if metrics are enabled."""
    return _metrics_enabled


# Initialize metrics on module import
initialize_metrics()

