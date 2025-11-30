"""
Circuit breaker implementation for API failure handling.

Tracks failures in a sliding time window and opens the circuit
after a threshold of failures is reached.
"""

import time
import threading
import logging
from typing import Optional
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation, allowing requests
    OPEN = "open"      # Circuit is open, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service has recovered


class CircuitBreaker:
    """
    Circuit breaker that opens after a specified number of failures
    within a time window.
    
    Args:
        failure_threshold: Number of failures to trigger circuit open (default: 3)
        time_window_seconds: Time window in seconds to track failures (default: 120)
        recovery_timeout_seconds: Time to wait before attempting recovery (default: 60)
    """
    
    def __init__(
        self,
        failure_threshold: int = 3,
        time_window_seconds: int = 120,
        recovery_timeout_seconds: int = 60
    ):
        self.failure_threshold = failure_threshold
        self.time_window_seconds = time_window_seconds
        self.recovery_timeout_seconds = recovery_timeout_seconds
        
        self._state = CircuitState.CLOSED
        self._failure_timestamps: deque = deque()
        self._last_failure_time: Optional[float] = None
        self._lock = threading.Lock()
        
        logger.info(
            f"Circuit breaker initialized: threshold={failure_threshold}, "
            f"window={time_window_seconds}s, recovery={recovery_timeout_seconds}s"
        )
    
    def _clean_old_failures(self):
        """Remove failure timestamps outside the time window."""
        current_time = time.time()
        cutoff_time = current_time - self.time_window_seconds
        
        while self._failure_timestamps and self._failure_timestamps[0] < cutoff_time:
            self._failure_timestamps.popleft()
    
    def _should_open_circuit(self) -> bool:
        """Check if circuit should be opened based on failure count."""
        self._clean_old_failures()
        return len(self._failure_timestamps) >= self.failure_threshold
    
    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self._last_failure_time is None:
            return False
        
        current_time = time.time()
        return (current_time - self._last_failure_time) >= self.recovery_timeout_seconds
    
    def record_failure(self):
        """Record a failure and update circuit state."""
        with self._lock:
            current_time = time.time()
            self._failure_timestamps.append(current_time)
            self._last_failure_time = current_time
            
            # Clean old failures
            self._clean_old_failures()
            
            # Check if we should open the circuit
            if self._state == CircuitState.CLOSED and self._should_open_circuit():
                self._state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker opened: {len(self._failure_timestamps)} failures "
                    f"in the last {self.time_window_seconds} seconds"
                )
            elif self._state == CircuitState.HALF_OPEN:
                # If we fail during half-open, go back to open
                self._state = CircuitState.OPEN
                logger.warning("Circuit breaker failed during half-open state, reopening")
    
    def record_success(self):
        """Record a success and update circuit state."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                # Success during half-open means service recovered
                self._state = CircuitState.CLOSED
                self._failure_timestamps.clear()
                self._last_failure_time = None
                logger.info("Circuit breaker closed: service recovered")
            elif self._state == CircuitState.CLOSED:
                # Clean old failures on success
                self._clean_old_failures()
    
    def is_open(self) -> bool:
        """Check if circuit is currently open."""
        with self._lock:
            # Check if we should transition from OPEN to HALF_OPEN
            if self._state == CircuitState.OPEN and self._should_attempt_recovery():
                self._state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker entering half-open state: testing recovery")
            
            return self._state == CircuitState.OPEN
    
    def is_half_open(self) -> bool:
        """Check if circuit is in half-open state."""
        with self._lock:
            # Check if we should transition from OPEN to HALF_OPEN
            if self._state == CircuitState.OPEN and self._should_attempt_recovery():
                self._state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker entering half-open state: testing recovery")
            
            return self._state == CircuitState.HALF_OPEN
    
    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            # Check if we should transition from OPEN to HALF_OPEN
            if self._state == CircuitState.OPEN and self._should_attempt_recovery():
                self._state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker entering half-open state: testing recovery")
            
            return self._state
    
    def get_failure_count(self) -> int:
        """Get current failure count within the time window."""
        with self._lock:
            self._clean_old_failures()
            return len(self._failure_timestamps)
    
    def reset(self):
        """Manually reset the circuit breaker to closed state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_timestamps.clear()
            self._last_failure_time = None
            logger.info("Circuit breaker manually reset")


# Global circuit breaker instance for API endpoints (lazy initialization)
_api_circuit_breaker: Optional[CircuitBreaker] = None
_breaker_lock = threading.Lock()


def get_api_circuit_breaker(
    failure_threshold: Optional[int] = None,
    time_window_seconds: Optional[int] = None,
    recovery_timeout_seconds: Optional[int] = None
) -> CircuitBreaker:
    """
    Get the global API circuit breaker instance.
    
    If the circuit breaker hasn't been initialized, it will be created with
    the provided parameters or Config defaults.
    
    Args:
        failure_threshold: Number of failures to trigger circuit open (uses Config if None)
        time_window_seconds: Time window in seconds (uses Config if None)
        recovery_timeout_seconds: Recovery timeout in seconds (uses Config if None)
    
    Returns:
        CircuitBreaker instance
    """
    global _api_circuit_breaker
    
    if _api_circuit_breaker is None:
        with _breaker_lock:
            # Double-check pattern
            if _api_circuit_breaker is None:
                # Try to import Config, fall back to defaults if not available
                try:
                    import sys
                    import os
                    # Add src to path if needed
                    src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "src")
                    if src_path not in sys.path:
                        sys.path.insert(0, src_path)
                    
                    try:
                        from src.config import Config
                    except ImportError:
                        from config import Config
                    
                    _failure_threshold = failure_threshold or Config.CIRCUIT_BREAKER_FAILURE_THRESHOLD
                    _time_window = time_window_seconds or Config.CIRCUIT_BREAKER_TIME_WINDOW_SECONDS
                    _recovery_timeout = recovery_timeout_seconds or Config.CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS
                except (ImportError, AttributeError):
                    # Fall back to defaults if Config is not available
                    _failure_threshold = failure_threshold or 3
                    _time_window = time_window_seconds or 120
                    _recovery_timeout = recovery_timeout_seconds or 60
                    logger.warning(
                        "Config not available, using default circuit breaker settings: "
                        f"threshold={_failure_threshold}, window={_time_window}s, recovery={_recovery_timeout}s"
                    )
                
                _api_circuit_breaker = CircuitBreaker(
                    failure_threshold=_failure_threshold,
                    time_window_seconds=_time_window,
                    recovery_timeout_seconds=_recovery_timeout
                )
    
    return _api_circuit_breaker

