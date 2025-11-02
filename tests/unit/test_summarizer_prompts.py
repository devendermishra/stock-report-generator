"""
Unit tests for summarizer prompts.
Tests the prompt building functionality.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from tools.summarizer_prompts import (
    create_summarization_prompt,
    create_insight_extraction_prompt,
    create_document_chunks_prompt,
    create_insight_categorization_prompt
)


class TestCreateSummarizationPrompt:
    """Test cases for create_summarization_prompt function."""
    
    def test_basic_prompt(self):
        """Test creating a basic summarization prompt."""
        text = "Sample financial text to summarize"
        max_length = 500
        prompt = create_summarization_prompt(text, max_length)
        
        assert "Sample financial text to summarize" in prompt
        assert "500 words" in prompt
        assert "summary" in prompt.lower()
        assert "key_points" in prompt.lower()
        assert "sentiment" in prompt.lower()
    
    def test_prompt_with_focus_areas(self):
        """Test prompt creation with focus areas."""
        text = "Test text"
        max_length = 300
        focus_areas = ["financial", "strategic"]
        prompt = create_summarization_prompt(text, max_length, focus_areas)
        
        assert "financial" in prompt
        assert "strategic" in prompt
        assert "Focus on these areas" in prompt
    
    def test_prompt_without_focus_areas(self):
        """Test prompt creation without focus areas."""
        text = "Test text"
        max_length = 200
        prompt = create_summarization_prompt(text, max_length, None)
        
        assert text in prompt
        assert "Focus on these areas" not in prompt
    
    def test_prompt_includes_json_format(self):
        """Test that prompt includes JSON format specification."""
        prompt = create_summarization_prompt("test", 100)
        
        assert "JSON format" in prompt or "json" in prompt.lower()
        assert "summary" in prompt.lower()
        assert "key_points" in prompt.lower()


class TestCreateInsightExtractionPrompt:
    """Test cases for create_insight_extraction_prompt function."""
    
    def test_basic_insight_prompt(self):
        """Test creating a basic insight extraction prompt."""
        text = "Financial analysis text"
        prompt = create_insight_extraction_prompt(text)
        
        assert text in prompt
        assert "insights" in prompt.lower()
        assert "categories" in prompt.lower()
        assert "sentiment_analysis" in prompt.lower()
        assert "key_metrics" in prompt.lower()
    
    def test_insight_prompt_with_categories(self):
        """Test prompt with specific categories."""
        text = "Test text"
        categories = ["financial", "strategic", "operational"]
        prompt = create_insight_extraction_prompt(text, categories)
        
        assert "financial" in prompt
        assert "strategic" in prompt
        assert "operational" in prompt
        assert "Focus on these categories" in prompt
    
    def test_insight_prompt_json_structure(self):
        """Test that insight prompt includes correct JSON structure."""
        prompt = create_insight_extraction_prompt("test")
        
        assert "insights" in prompt
        assert "categories" in prompt
        assert "sentiment_analysis" in prompt
        assert "key_metrics" in prompt


class TestCreateDocumentChunksPrompt:
    """Test cases for create_document_chunks_prompt function."""
    
    def test_document_chunks_prompt(self):
        """Test creating prompt for document chunks."""
        combined_text = "Chunk 1\n\n---\n\nChunk 2"
        max_length = 1000
        prompt = create_document_chunks_prompt(combined_text, max_length)
        
        assert combined_text in prompt
        assert "financial document chunks" in prompt.lower()
        assert "financial_metrics" in prompt.lower()
        assert "strategic_insights" in prompt.lower()
    
    def test_prompt_includes_required_fields(self):
        """Test that prompt includes all required JSON fields."""
        prompt = create_document_chunks_prompt("test chunks", 500)
        
        assert "summary" in prompt.lower()
        assert "key_points" in prompt.lower()
        assert "financial_metrics" in prompt.lower()
        assert "sentiment" in prompt.lower()
        assert "confidence" in prompt.lower()


class TestCreateInsightCategorizationPrompt:
    """Test cases for create_insight_categorization_prompt function."""
    
    def test_categorization_prompt(self):
        """Test creating insight categorization prompt."""
        text = "Analysis text"
        categories = ["financial_performance", "strategic_initiatives", "market_outlook"]
        prompt = create_insight_categorization_prompt(text, categories)
        
        assert text in prompt
        assert "financial_performance" in prompt
        assert "strategic_initiatives" in prompt
        assert "market_outlook" in prompt
    
    def test_prompt_includes_category_structure(self):
        """Test that prompt includes category structure in JSON format."""
        prompt = create_insight_categorization_prompt(
            "test",
            ["financial_performance", "risk_factors"]
        )
        
        assert "financial_performance" in prompt
        assert "risk_factors" in prompt
        assert "categories" in prompt
        assert "insights" in prompt
        assert "sentiment_analysis" in prompt
        assert "key_metrics" in prompt
    
    def test_multiple_categories(self):
        """Test prompt with multiple categories."""
        categories = [
            "financial_performance",
            "strategic_initiatives",
            "market_outlook",
            "risk_factors",
            "growth_opportunities"
        ]
        prompt = create_insight_categorization_prompt("text", categories)
        
        for category in categories:
            assert category in prompt
