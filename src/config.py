"""
Configuration module for Stock Report Generator.
Handles API keys, environment variables, and system settings.
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the Stock Report Generator."""
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    # Yahoo Finance API Configuration (no API key required)
    YAHOO_FINANCE_BASE_URL = "https://query1.finance.yahoo.com"
    
    # Model Configuration
    DEFAULT_MODEL = "gpt-4o-mini"
    MAX_TOKENS = 4000
    TEMPERATURE = 0.1
    
    # MCP Configuration
    MCP_SERVER_NAME = "stock_report_mcp"
    MCP_CONTEXT_SIZE = 10000
    
    # File Paths
    OUTPUT_DIR = "reports"
    TEMP_DIR = "temp"
    
    # Rate Limiting
    MAX_REQUESTS_PER_MINUTE = 60
    REQUEST_DELAY = 1.0  # seconds
    
    # API Rate Limiting
    API_RATE_LIMIT_PER_MINUTE = int(os.getenv("API_RATE_LIMIT_PER_MINUTE", "2"))  # Default: 2 requests per minute
    
    # Circuit Breaker Configuration
    CIRCUIT_BREAKER_FAILURE_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "3"))  # Default: 3 failures
    CIRCUIT_BREAKER_TIME_WINDOW_SECONDS = int(os.getenv("CIRCUIT_BREAKER_TIME_WINDOW_SECONDS", "120"))  # Default: 2 minutes (120 seconds)
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS = int(os.getenv("CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS", "60"))  # Default: 1 minute recovery timeout
    
    # Logging Configuration
    COMBINE_PROMPTS_AND_OUTPUTS = os.getenv("COMBINE_PROMPTS_AND_OUTPUTS", "true").lower() == "true"  # Default: enabled
    
    # Metrics Configuration
    ENABLE_METRICS = os.getenv("ENABLE_METRICS", "false").lower() == "true"  # Default: disabled
    METRICS_PORT = int(os.getenv("METRICS_PORT", "8000"))  # Port for Prometheus metrics endpoint
    
    # Retry Configuration
    LLM_RETRY_MAX_ATTEMPTS = int(os.getenv("LLM_RETRY_MAX_ATTEMPTS", "3"))  # Default: 3 retries
    LLM_RETRY_INITIAL_DELAY = float(os.getenv("LLM_RETRY_INITIAL_DELAY", "1.0"))  # Default: 1 second
    LLM_RETRY_MAX_DELAY = float(os.getenv("LLM_RETRY_MAX_DELAY", "60.0"))  # Default: 60 seconds
    
    TOOL_RETRY_MAX_ATTEMPTS = int(os.getenv("TOOL_RETRY_MAX_ATTEMPTS", "3"))  # Default: 3 retries
    TOOL_RETRY_INITIAL_DELAY = float(os.getenv("TOOL_RETRY_INITIAL_DELAY", "1.0"))  # Default: 1 second
    TOOL_RETRY_MAX_DELAY = float(os.getenv("TOOL_RETRY_MAX_DELAY", "30.0"))  # Default: 30 seconds
    
    @classmethod
    def validate_config(cls) -> Dict[str, bool]:
        """Validate that all required configuration is present."""
        validation = {
            "openai_key": bool(cls.OPENAI_API_KEY),
        }
        return validation
    
    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """Get model configuration for LLM calls."""
        return {
            "model": cls.DEFAULT_MODEL,
            "max_tokens": cls.MAX_TOKENS,
            "temperature": cls.TEMPERATURE,
        }
