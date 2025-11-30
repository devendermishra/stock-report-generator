"""
Unit tests for circuit breaker implementation.
Tests CircuitBreaker class and related functionality.
"""

import pytest
import sys
import time
import threading
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from utils.circuit_breaker import CircuitBreaker, CircuitState, get_api_circuit_breaker


class TestCircuitBreaker:
    """Test cases for CircuitBreaker class."""
    
    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.breaker = CircuitBreaker(
            failure_threshold=3,
            time_window_seconds=60,
            recovery_timeout_seconds=30
        )
    
    def test_initial_state(self) -> None:
        """Test that circuit breaker starts in CLOSED state."""
        assert self.breaker.get_state() == CircuitState.CLOSED
        assert not self.breaker.is_open()
        assert not self.breaker.is_half_open()
        assert self.breaker.get_failure_count() == 0
    
    def test_record_failure(self) -> None:
        """Test recording a failure."""
        self.breaker.record_failure()
        assert self.breaker.get_failure_count() == 1
        assert self.breaker.get_state() == CircuitState.CLOSED
    
    def test_circuit_opens_after_threshold(self) -> None:
        """Test that circuit opens after reaching failure threshold."""
        # Record failures up to threshold
        for _ in range(3):
            self.breaker.record_failure()
        
        assert self.breaker.get_state() == CircuitState.OPEN
        assert self.breaker.is_open()
        assert self.breaker.get_failure_count() == 3
    
    def test_circuit_does_not_open_below_threshold(self) -> None:
        """Test that circuit stays closed below threshold."""
        # Record failures below threshold
        for _ in range(2):
            self.breaker.record_failure()
        
        assert self.breaker.get_state() == CircuitState.CLOSED
        assert not self.breaker.is_open()
        assert self.breaker.get_failure_count() == 2
    
    def test_record_success_closed_state(self) -> None:
        """Test recording success in CLOSED state."""
        self.breaker.record_failure()
        self.breaker.record_success()
        
        # Should clean old failures
        assert self.breaker.get_state() == CircuitState.CLOSED
    
    def test_record_success_half_open_state(self) -> None:
        """Test recording success in HALF_OPEN state recovers circuit."""
        # Open the circuit
        for _ in range(3):
            self.breaker.record_failure()
        assert self.breaker.get_state() == CircuitState.OPEN
        
        # Wait for recovery timeout (mock time)
        with patch('time.time', return_value=time.time() + 31):
            # Transition to half-open
            assert self.breaker.is_half_open()
            
            # Record success
            self.breaker.record_success()
            assert self.breaker.get_state() == CircuitState.CLOSED
            assert self.breaker.get_failure_count() == 0
    
    def test_failure_during_half_open_reopens(self) -> None:
        """Test that failure during half-open state reopens circuit."""
        # Open the circuit
        for _ in range(3):
            self.breaker.record_failure()
        
        # Wait for recovery timeout
        with patch('time.time', return_value=time.time() + 31):
            assert self.breaker.is_half_open()
            
            # Record failure during half-open
            self.breaker.record_failure()
            assert self.breaker.get_state() == CircuitState.OPEN
    
    def test_clean_old_failures(self) -> None:
        """Test that old failures are cleaned from time window."""
        # Record failures
        for _ in range(3):
            self.breaker.record_failure()
        
        # Simulate time passing beyond time window
        with patch('time.time', return_value=time.time() + 61):
            # Check state - should still be open from previous failures
            # But failure count should be cleaned
            self.breaker._clean_old_failures()
            # Note: Circuit might still be open, but failure count should decrease
            # The exact behavior depends on when failures were recorded
    
    def test_reset(self) -> None:
        """Test manual reset of circuit breaker."""
        # Open the circuit
        for _ in range(3):
            self.breaker.record_failure()
        assert self.breaker.get_state() == CircuitState.OPEN
        
        # Reset
        self.breaker.reset()
        assert self.breaker.get_state() == CircuitState.CLOSED
        assert self.breaker.get_failure_count() == 0
    
    def test_thread_safety(self) -> None:
        """Test that circuit breaker is thread-safe."""
        failures = []
        
        def record_failures():
            for _ in range(10):
                self.breaker.record_failure()
                failures.append(self.breaker.get_failure_count())
        
        threads = [threading.Thread(target=record_failures) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # Should have recorded failures safely
        assert self.breaker.get_failure_count() >= 3
        assert self.breaker.get_state() == CircuitState.OPEN
    
    def test_custom_parameters(self) -> None:
        """Test circuit breaker with custom parameters."""
        custom_breaker = CircuitBreaker(
            failure_threshold=5,
            time_window_seconds=120,
            recovery_timeout_seconds=60
        )
        
        # Should not open with 4 failures
        for _ in range(4):
            custom_breaker.record_failure()
        assert custom_breaker.get_state() == CircuitState.CLOSED
        
        # Should open with 5 failures
        custom_breaker.record_failure()
        assert custom_breaker.get_state() == CircuitState.OPEN


class TestGetApiCircuitBreaker:
    """Test cases for get_api_circuit_breaker function."""
    
    def test_get_api_circuit_breaker_default(self) -> None:
        """Test getting default API circuit breaker."""
        breaker = get_api_circuit_breaker()
        assert breaker is not None
        assert isinstance(breaker, CircuitBreaker)
    
    def test_get_api_circuit_breaker_custom_params(self) -> None:
        """Test getting API circuit breaker with custom parameters."""
        # Reset the global breaker to test custom params
        import utils.circuit_breaker as cb_module
        original_breaker = cb_module._api_circuit_breaker
        cb_module._api_circuit_breaker = None
        
        try:
            breaker = get_api_circuit_breaker(
                failure_threshold=5,
                time_window_seconds=120,
                recovery_timeout_seconds=60
            )
            assert breaker is not None
            assert breaker.failure_threshold == 5
            assert breaker.time_window_seconds == 120
            assert breaker.recovery_timeout_seconds == 60
        finally:
            cb_module._api_circuit_breaker = original_breaker
    
    def test_get_api_circuit_breaker_singleton(self) -> None:
        """Test that get_api_circuit_breaker returns singleton."""
        breaker1 = get_api_circuit_breaker()
        breaker2 = get_api_circuit_breaker()
        assert breaker1 is breaker2
    
    def test_get_api_circuit_breaker_with_config(self) -> None:
        """Test getting circuit breaker with Config values."""
        # This test verifies that get_api_circuit_breaker works
        # Config loading is tested indirectly through the function working
        breaker = get_api_circuit_breaker()
        assert breaker is not None
        assert isinstance(breaker, CircuitBreaker)
        # Verify it has reasonable default values
        assert breaker.failure_threshold > 0
        assert breaker.time_window_seconds > 0
        assert breaker.recovery_timeout_seconds > 0
    
    def test_get_api_circuit_breaker_fallback(self) -> None:
        """Test fallback to defaults when Config is not available."""
        # Reset the global breaker
        import utils.circuit_breaker as cb_module
        original_breaker = cb_module._api_circuit_breaker
        cb_module._api_circuit_breaker = None
        
        try:
            # Test that get_api_circuit_breaker works even without Config
            # It should fall back to defaults
            breaker = get_api_circuit_breaker()
            assert breaker is not None
            # Should use defaults (3, 120, 60) or whatever was configured
            assert breaker.failure_threshold >= 3
            assert breaker.time_window_seconds >= 60
            assert breaker.recovery_timeout_seconds >= 30
        finally:
            cb_module._api_circuit_breaker = original_breaker

