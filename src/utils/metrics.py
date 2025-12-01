"""
Simple Metrics Collection Module.

Always collects metrics in-memory. Optionally exports to Prometheus if available.
"""

import logging
from typing import Optional, Dict, Any
from collections import defaultdict
from threading import Lock

logger = logging.getLogger(__name__)

# Simple in-memory storage
_lock = Lock()
_counts = defaultdict(int)  # metric_name -> count
_durations = defaultdict(list)  # metric_name -> [durations]

# Optional Prometheus
_prometheus_metrics = {}
_prometheus_enabled = False
_init_lock = Lock()  # Lock for initialization


def initialize_prometheus_metrics():
    """Initialize Prometheus metrics if enabled in config."""
    global _prometheus_metrics, _prometheus_enabled
    
    # Thread-safe check to prevent duplicate initialization
    with _init_lock:
        if _prometheus_enabled:
            logger.debug("Prometheus metrics already initialized")
            return
        
        try:
            from prometheus_client import Counter, Histogram, start_http_server
            from ..config import Config
        except ImportError:
            try:
                from config import Config
            except ImportError:
                logger.debug("Config not available for Prometheus initialization")
                return
            logger.debug("prometheus_client not available")
            return
        
        if not Config.ENABLE_METRICS:
            logger.debug("Prometheus metrics disabled by configuration")
            return
        
        # Initialize metrics objects first
        try:
            _prometheus_metrics = {
                'llm_requests': Counter('llm_requests_total', 'Total LLM requests', ['model', 'agent']),
                'llm_tokens': Counter('llm_tokens_total', 'Total LLM tokens', ['model', 'agent', 'type']),
                'llm_duration': Histogram('llm_duration_seconds', 'LLM request duration', ['model', 'agent']),
                'reports': Counter('reports_total', 'Reports generated', ['symbol', 'status']),
                'validations': Counter('validations_total', 'Symbol validations', ['symbol', 'status']),
                'errors': Counter('errors_total', 'Errors', ['type', 'location']),
            }
        except Exception as e:
            logger.debug(f"Failed to create Prometheus metrics: {e}")
            return
        
        # Try to start HTTP server
        try:
            start_http_server(Config.METRICS_PORT, addr='')
            _prometheus_enabled = True
            logger.info(f"Prometheus metrics enabled on port {Config.METRICS_PORT}")
        except OSError as e:
            if "Address already in use" in str(e):
                # Port already in use - metrics are still enabled and will be collected
                # They can be scraped from the existing server if it's serving the same registry
                _prometheus_enabled = True
                logger.debug(f"Prometheus port {Config.METRICS_PORT} already in use. Metrics enabled (may be served by existing process).")
            else:
                logger.debug(f"Failed to start Prometheus server: {e}")
        except Exception as e:
            logger.debug(f"Prometheus server startup failed: {e}")


def _record(name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
    """Internal: record a metric."""
    with _lock:
        _counts[name] += value
        if labels:
            key = f"{name}_{'_'.join(f'{k}={v}' for k, v in sorted(labels.items()))}"
            _counts[key] += value

    # Prometheus
    if _prometheus_enabled and name in _prometheus_metrics:
        try:
            if labels:
                _prometheus_metrics[name].labels(**labels).inc(value)
            else:
                _prometheus_metrics[name].inc(value)
        except Exception:
            pass


def _record_duration(name: str, seconds: float, labels: Optional[Dict[str, str]] = None):
    """Internal: record a duration."""
    with _lock:
        _durations[name].append(seconds)
        if len(_durations[name]) > 1000:
            _durations[name] = _durations[name][-1000:]

    # Prometheus
    if _prometheus_enabled and name in _prometheus_metrics:
        try:
            if labels:
                _prometheus_metrics[name].labels(**labels).observe(seconds)
            else:
                _prometheus_metrics[name].observe(seconds)
        except Exception:
            pass


# Public API - simple functions
def record_llm_request(model: str, agent: Optional[str] = None,
                     request_tokens: Optional[int] = None,
                     response_tokens: Optional[int] = None,
                     duration: Optional[float] = None,
                     success: bool = True):
    """Record LLM request."""
    labels = {"model": model, "agent": agent or "unknown"}
    _record("llm_requests", labels=labels)
    if not success:
        _record("llm_errors", labels=labels)
    if request_tokens:
        _record("llm_tokens", value=request_tokens, labels={**labels, "type": "request"})
    if response_tokens:
        _record("llm_tokens", value=response_tokens, labels={**labels, "type": "response"})
    if duration:
        _record_duration("llm_duration", duration, labels=labels)


def record_llm_request_from_response(model: str, response: Any,
                                    agent: Optional[str] = None,
                                    duration: Optional[float] = None,
                                    success: bool = True):
    """Record LLM request from response object."""
    request_tokens = None
    response_tokens = None
    if hasattr(response, 'usage') and response.usage:
        request_tokens = getattr(response.usage, 'prompt_tokens', None)
        response_tokens = getattr(response.usage, 'completion_tokens', None)
    record_llm_request(model, agent, request_tokens, response_tokens, duration, success)


def record_report(symbol: str, duration: Optional[float] = None, status: str = "completed"):
    """Record report generation."""
    labels = {"symbol": symbol, "status": status}
    _record("reports", labels=labels)
    if duration:
        _record_duration("report_duration", duration, labels={"symbol": symbol})


def record_validation(symbol: str, valid: bool, error: Optional[str] = None):
    """Record symbol validation."""
    status = "valid" if valid else "invalid"
    labels = {"symbol": symbol, "status": status}
    _record("validations", labels=labels)
    if not valid:
        error_type = error or "unknown"
        _record("errors", labels={"type": "validation", "location": error_type})


def record_error(error_type: str, location: str = "unknown"):
    """Record an error."""
    _record("errors", labels={"type": error_type, "location": location})


def get_summary() -> Dict[str, Any]:
    """Get metrics summary."""
    with _lock:
        avg_durations = {
            name: sum(durs) / len(durs) if durs else 0
            for name, durs in _durations.items()
        }
        return {
            "counts": dict(_counts),
            "avg_durations": avg_durations,
            "prometheus_enabled": _prometheus_enabled,
        }


def metrics_enabled() -> bool:
    """Always True."""
    return True


def initialize_metrics():
    """Initialize metrics (calls Prometheus initialization)."""
    initialize_prometheus_metrics()


def get_metrics_status() -> Dict[str, Any]:
    """Get status."""
    return {
        "enabled": True,
        "prometheus_enabled": _prometheus_enabled,
    }
