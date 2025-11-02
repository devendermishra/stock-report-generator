"""
Unit tests for summarizer parsers.
Tests the JSON parsing functionality for OpenAI API responses.
"""

import pytest
import json
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from tools.summarizer_parsers import (
    parse_summary_response,
    parse_insight_response,
    parse_json_response
)


class TestParseSummaryResponse:
    """Test cases for parse_summary_response function."""
    
    def test_parse_valid_json_response(self):
        """Test parsing a valid JSON response."""
        response = '{"summary": "Test summary", "key_points": ["point1", "point2"], "sentiment": "positive", "confidence": 0.9}'
        result = parse_summary_response(response)
        
        assert result["summary"] == "Test summary"
        assert result["key_points"] == ["point1", "point2"]
        assert result["sentiment"] == "positive"
        assert result["confidence"] == 0.9
    
    def test_parse_json_with_extra_text(self):
        """Test parsing JSON that has extra text around it."""
        response = 'Here is the result: {"summary": "Extra text summary", "key_points": ["point1"], "sentiment": "neutral", "confidence": 0.8} and more text'
        result = parse_summary_response(response)
        
        assert result["summary"] == "Extra text summary"
        assert result["key_points"] == ["point1"]
        assert result["sentiment"] == "neutral"
        assert result["confidence"] == 0.8
    
    def test_parse_non_json_response(self):
        """Test parsing a non-JSON response (fallback behavior)."""
        response = "This is just plain text without JSON"
        result = parse_summary_response(response)
        
        assert result["summary"] == response
        assert result["key_points"] == []
        assert result["sentiment"] == "neutral"
        assert result["confidence"] == 0.7
    
    def test_parse_multiline_json(self):
        """Test parsing multiline JSON response."""
        response = """{
            "summary": "Multi-line summary",
            "key_points": ["point1", "point2", "point3"],
            "sentiment": "negative",
            "confidence": 0.85
        }"""
        result = parse_summary_response(response)
        
        assert result["summary"] == "Multi-line summary"
        assert len(result["key_points"]) == 3
        assert result["sentiment"] == "negative"
    
    def test_parse_invalid_json_structure(self):
        """Test parsing invalid JSON structure (fallback behavior)."""
        response = '{"invalid": json structure}'
        result = parse_summary_response(response)
        
        # Should fallback to default structure
        assert "summary" in result
        assert result["sentiment"] == "neutral"


class TestParseInsightResponse:
    """Test cases for parse_insight_response function."""
    
    def test_parse_valid_insight_json(self):
        """Test parsing a valid insight extraction JSON."""
        response = json.dumps({
            "insights": ["Insight 1", "Insight 2"],
            "categories": {
                "financial": ["metric1"],
                "strategic": ["strategy1"]
            },
            "sentiment_analysis": {
                "sentiment": "positive",
                "confidence": 0.9
            },
            "key_metrics": {
                "revenue": "100M"
            }
        })
        result = parse_insight_response(response)
        
        assert len(result["insights"]) == 2
        assert "financial" in result["categories"]
        assert result["sentiment_analysis"]["sentiment"] == "positive"
        assert result["key_metrics"]["revenue"] == "100M"
    
    def test_parse_insight_with_extra_text(self):
        """Test parsing insight JSON with surrounding text."""
        response = 'The analysis shows: {"insights": ["Test"], "categories": {}, "sentiment_analysis": {"overall_sentiment": "neutral"}, "key_metrics": {}}'
        result = parse_insight_response(response)
        
        assert result["insights"] == ["Test"]
        assert result["sentiment_analysis"]["overall_sentiment"] == "neutral"
    
    def test_parse_non_json_insight(self):
        """Test parsing non-JSON insight response (fallback)."""
        response = "Just plain text insight"
        result = parse_insight_response(response)
        
        assert result["insights"] == []
        assert result["categories"] == {}
        assert result["sentiment_analysis"]["overall_sentiment"] == "neutral"
        assert result["key_metrics"] == {}


class TestParseJsonResponse:
    """Test cases for parse_json_response generic function."""
    
    def test_parse_valid_json_with_default(self):
        """Test parsing valid JSON with custom default."""
        response = '{"key": "value"}'
        default = {"default": "data"}
        result = parse_json_response(response, default)
        
        assert result["key"] == "value"
    
    def test_parse_invalid_json_with_default(self):
        """Test parsing invalid JSON falls back to default."""
        response = "not json"
        default = {"error": "parsing failed"}
        result = parse_json_response(response, default)
        
        assert result == default
    
    def test_parse_json_no_default(self):
        """Test parsing JSON without default (should return empty dict)."""
        response = "not json"
        result = parse_json_response(response)
        
        assert result == {}
    
    def test_parse_nested_json(self):
        """Test parsing nested JSON structures."""
        response = json.dumps({
            "level1": {
                "level2": {
                    "level3": "deep value"
                }
            }
        })
        result = parse_json_response(response)
        
        assert result["level1"]["level2"]["level3"] == "deep value"
