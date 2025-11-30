"""
Session Management for Request Tracking.

Provides session ID generation and management for tracking invocations
across the application. Session IDs are included in logs and prompts.
"""

import uuid
import contextvars
from typing import Optional, Any
from datetime import datetime

# Context variable for session ID (Python's equivalent to MDC)
session_id_context: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'session_id', default=None
)

# Context variable for additional context (stock symbol, agent name, etc.)
session_context: contextvars.ContextVar[dict] = contextvars.ContextVar(
    'session_context', default={}
)


def generate_session_id() -> str:
    """
    Generate a unique session ID.
    
    Returns:
        Unique session ID string (UUID-based)
    """
    return str(uuid.uuid4())


def set_session_id(session_id: str):
    """
    Set the current session ID in context.
    
    Args:
        session_id: Session ID to set
    """
    session_id_context.set(session_id)


def get_session_id() -> Optional[str]:
    """
    Get the current session ID from context.
    
    Returns:
        Current session ID or None if not set
    """
    return session_id_context.get()


def set_session_context(key: str, value: Any):
    """
    Set additional context information for the session.
    
    Args:
        key: Context key
        value: Context value
    """
    ctx = session_context.get({})
    ctx[key] = value
    session_context.set(ctx)


def get_session_context() -> dict:
    """
    Get all session context information.
    
    Returns:
        Dictionary of context information
    """
    return session_context.get({})


def clear_session():
    """
    Clear the current session ID and context.
    """
    session_id_context.set(None)
    session_context.set({})


class SessionContext:
    """
    Context manager for session ID management.
    
    Usage:
        with SessionContext() as session_id:
            # All code here will have session_id in context
            pass
    """
    
    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize session context.
        
        Args:
            session_id: Optional session ID (will generate if not provided)
        """
        self.session_id = session_id or generate_session_id()
        self.old_session_id = None
        self.old_context = None
    
    def __enter__(self):
        """Enter context - set session ID."""
        self.old_session_id = get_session_id()
        self.old_context = get_session_context().copy()
        set_session_id(self.session_id)
        return self.session_id
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - restore previous session ID."""
        if self.old_session_id:
            set_session_id(self.old_session_id)
        else:
            clear_session()
        if self.old_context:
            session_context.set(self.old_context)
        return False

