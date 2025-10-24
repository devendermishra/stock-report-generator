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
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    # Note: TAVILY_API_KEY removed - now using DuckDuckGo search (free, no API key required)
    
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
    
    @classmethod
    def validate_config(cls) -> Dict[str, bool]:
        """Validate that all required configuration is present."""
        validation = {
            "openai_key": bool(cls.OPENAI_API_KEY),
            "anthropic_key": bool(cls.ANTHROPIC_API_KEY),
            # Note: tavily_key removed - DuckDuckGo search doesn't require API key
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
