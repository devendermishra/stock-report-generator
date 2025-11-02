"""
Response parsers for summarizer tool.
Contains functions to parse structured responses from OpenAI API.
"""

import json
import re
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def parse_summary_response(response_text: str) -> Dict[str, Any]:
    """Parse the structured response from the summarization API."""
    try:
        # Try to extract JSON from the response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            return json.loads(json_str)
        else:
            # Fallback: return basic structure
            return {
                "summary": response_text,
                "key_points": [],
                "sentiment": "neutral",
                "confidence": 0.7
            }
    except Exception as e:
        logger.warning(f"Could not parse structured response: {e}")
        return {
            "summary": response_text,
            "key_points": [],
            "sentiment": "neutral",
            "confidence": 0.7
        }


def parse_insight_response(response_text: str) -> Dict[str, Any]:
    """Parse the structured response from the insight extraction API."""
    try:
        # Try to extract JSON from the response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            return json.loads(json_str)
        else:
            # Fallback: return basic structure
            return {
                "insights": [],
                "categories": {},
                "sentiment_analysis": {"overall_sentiment": "neutral"},
                "key_metrics": {}
            }
    except Exception as e:
        logger.warning(f"Could not parse structured insight response: {e}")
        return {
            "insights": [],
            "categories": {},
            "sentiment_analysis": {"overall_sentiment": "neutral"},
            "key_metrics": {}
        }


def parse_json_response(response_text: str, default: Dict[str, Any] = None) -> Dict[str, Any]:
    """Generic JSON parser that extracts JSON from text response."""
    if default is None:
        default = {}
    
    try:
        # Try to extract JSON from the response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            return json.loads(json_str)
        else:
            return default
    except Exception as e:
        logger.warning(f"Could not parse JSON response: {e}")
        return default
