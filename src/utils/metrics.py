"""
Simple Metrics Collection Module.

Always collects metrics in-memory. Optionally exports to Prometheus if available.
Supports multiprocessing mode for Uvicorn workers.
"""

import logging
import os
from typing import Optional, Dict, Any
from collections import defaultdict
from threading import Lock

logger = logging.getLogger(__name__)

# Simple in-memory storage
_lock = Lock()
_counts = defaultdict(int)  # metric_name -> count
_durations = defaultdict(list)  # metric_name -> [durations]

# Prometheus metrics - defined at module level for multiprocessing support
_prometheus_metrics = {}
_prometheus_enabled = False
_prometheus_multiprocess = False
_init_lock = Lock()  # Lock for initialization

# Try to import Prometheus client and enable multiprocess mode if needed
try:
    from prometheus_client import Counter, Histogram, start_http_server, REGISTRY, CollectorRegistry, generate_latest
    from prometheus_client.multiprocess import MultiProcessCollector
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = None
    Histogram = None
    start_http_server = None
    REGISTRY = None
    CollectorRegistry = None
    generate_latest = None
    MultiProcessCollector = None


def _enable_multiprocess_mode():
    """Enable Prometheus multiprocess mode for Uvicorn workers."""
    global _prometheus_multiprocess
    
    if not PROMETHEUS_AVAILABLE:
        return False
    
    try:
        from prometheus_client import values
        from prometheus_client.multiprocess import mark_process_dead
        
        # Set multiprocess mode
        prometheus_multiproc_dir = os.environ.get('PROMETHEUS_MULTIPROC_DIR')
        if not prometheus_multiproc_dir:
            # Default to a directory in the project
            prometheus_multiproc_dir = os.path.join(os.getcwd(), 'prometheus_multiproc_dir')
            os.environ['PROMETHEUS_MULTIPROC_DIR'] = prometheus_multiproc_dir
        
        # Create directory if it doesn't exist
        if not os.path.exists(prometheus_multiproc_dir):
            os.makedirs(prometheus_multiproc_dir, exist_ok=True)
        
        # Clear any stale files
        for f in os.listdir(prometheus_multiproc_dir):
            if f.endswith('.db'):
                try:
                    os.remove(os.path.join(prometheus_multiproc_dir, f))
                except Exception:
                    pass
        
        # Set multiprocess mode
        values.ValueClass = values.MultiProcessValue()
        _prometheus_multiprocess = True
        logger.info(f"Prometheus multiprocess mode enabled. Using directory: {prometheus_multiproc_dir}")
        return True
    except Exception as e:
        logger.warning(f"Failed to enable Prometheus multiprocess mode: {e}")
        return False


def _create_prometheus_metrics():
    """Create Prometheus metrics at module level."""
    global _prometheus_metrics
    
    if not PROMETHEUS_AVAILABLE:
        return
    
    try:
        # Create metrics at module level - these will be shared across workers in multiprocess mode
        _prometheus_metrics = {
            'llm_requests': Counter('llm_requests_total', 'Total LLM requests', ['model', 'agent']),
            'llm_errors': Counter('llm_errors_total', 'Total LLM errors', ['model', 'agent']),
            'llm_tokens': Counter('llm_tokens_total', 'Total LLM tokens', ['model', 'agent', 'type']),
            'llm_duration': Histogram('llm_duration_seconds', 'LLM request duration', ['model', 'agent']),
            'reports': Counter('reports_total', 'Reports generated', ['symbol', 'status']),
            'report_duration': Histogram('report_duration_seconds', 'Report generation duration', ['symbol']),
            'validations': Counter('validations_total', 'Symbol validations', ['symbol', 'status']),
            'errors': Counter('errors_total', 'Errors', ['type', 'location']),
        }
        logger.debug(f"Created {len(_prometheus_metrics)} Prometheus metrics at module level")
    except Exception as e:
        logger.error(f"Failed to create Prometheus metrics: {e}", exc_info=True)


def initialize_prometheus_metrics():
    """Initialize Prometheus metrics if enabled in config."""
    global _prometheus_metrics, _prometheus_enabled

    # Thread-safe check to prevent duplicate initialization
    with _init_lock:
        if _prometheus_enabled:
            logger.debug("Prometheus metrics already initialized")
            return

        if not PROMETHEUS_AVAILABLE:
            logger.debug("prometheus_client not available")
            return

        try:
            from ..config import Config
        except ImportError:
            try:
                from config import Config
            except ImportError:
                logger.debug("Config not available for Prometheus initialization")
                return

        if not Config.ENABLE_METRICS:
            logger.debug("Prometheus metrics disabled by configuration")
            return

        # Enable multiprocess mode if not already enabled
        if not _prometheus_multiprocess:
            _enable_multiprocess_mode()

        # Create metrics at module level (only once, even if called multiple times)
        if not _prometheus_metrics:
            _create_prometheus_metrics()

        # Note: For multiprocess mode with Uvicorn workers, we don't start a separate HTTP server
        # Instead, metrics should be exposed via FastAPI endpoint (see api.py)
        # The HTTP server is only started in single-process mode
        try:
            if _prometheus_multiprocess:
                # In multiprocess mode, don't start HTTP server here
                # Metrics will be exposed via FastAPI /metrics endpoint
                logger.info(f"Prometheus metrics enabled (multiprocess mode). Expose via FastAPI /metrics endpoint.")
            else:
                # Single process mode - start HTTP server
                start_http_server(Config.METRICS_PORT, addr='')
                logger.info(f"Prometheus metrics enabled on port {Config.METRICS_PORT}")
            
            _prometheus_enabled = True
        except OSError as e:
            if "Address already in use" in str(e):
                _prometheus_enabled = True
                logger.debug(f"Prometheus port {Config.METRICS_PORT} already in use. Metrics enabled (may be served by existing process).")
            else:
                logger.debug(f"Failed to start Prometheus server: {e}")
        except Exception as e:
            logger.error(f"Prometheus server startup failed: {e}", exc_info=True)


# Create metrics at module import time if Prometheus is available
if PROMETHEUS_AVAILABLE:
    try:
        # Check if metrics should be enabled
        try:
            from ..config import Config
        except ImportError:
            try:
                from config import Config
            except ImportError:
                Config = None
        
        if Config and Config.ENABLE_METRICS:
            # Enable multiprocess mode
            _enable_multiprocess_mode()
            # Create metrics at module level
            _create_prometheus_metrics()
    except Exception:
        pass  # Will be initialized later via initialize_prometheus_metrics()


def _record(name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
    """Internal: record a metric."""
    with _lock:
        _counts[name] += value
        if labels:
            key = f"{name}_{'_'.join(f'{k}={v}' for k, v in sorted(labels.items()))}"
            _counts[key] += value

    # Prometheus - record to Prometheus metrics
    if _prometheus_enabled:
        if name in _prometheus_metrics:
            try:
                metric_obj = _prometheus_metrics[name]
                if labels:
                    # Ensure all label keys are present
                    metric_obj.labels(**labels).inc(value)
                else:
                    metric_obj.inc(value)
            except Exception as e:
                logger.warning(f"Failed to record Prometheus metric {name} with labels {labels}: {e}", exc_info=True)


def _record_duration(name: str, seconds: float, labels: Optional[Dict[str, str]] = None):
    """Internal: record a duration."""
    with _lock:
        _durations[name].append(seconds)
        if len(_durations[name]) > 1000:
            _durations[name] = _durations[name][-1000:]

    # Prometheus - record to Prometheus metrics
    if _prometheus_enabled:
        if name in _prometheus_metrics:
            try:
                metric_obj = _prometheus_metrics[name]
                if labels:
                    # Ensure all label keys are present
                    metric_obj.labels(**labels).observe(seconds)
                else:
                    metric_obj.observe(seconds)
            except Exception as e:
                logger.warning(f"Failed to record Prometheus duration metric {name} with labels {labels}: {e}", exc_info=True)


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
            "prometheus_multiprocess": _prometheus_multiprocess,
        }


def metrics_enabled() -> bool:
    """Always True."""
    return True


def initialize_metrics():
    """Initialize metrics (calls Prometheus initialization)."""
    initialize_prometheus_metrics()


def get_metrics_status() -> Dict[str, Any]:
    """Get status."""
    multiproc_dir = os.environ.get('PROMETHEUS_MULTIPROC_DIR', 'not set')
    return {
        "enabled": True,
        "prometheus_enabled": _prometheus_enabled,
        "prometheus_multiprocess": _prometheus_multiprocess,
        "prometheus_multiproc_dir": multiproc_dir,
    }
