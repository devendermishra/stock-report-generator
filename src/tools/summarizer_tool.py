"""
Summarizer Tool for text summarization and insight extraction.
Provides AI-powered summarization capabilities for financial documents.
"""

import openai
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json
from langchain_core.tools import tool

try:
    # Try relative imports first (when run as module)
    from ..tools.openai_logger import openai_logger
    from ..tools.llm_guardrails_wrapper import (
        initialize_llm_guardrails,
        get_llm_wrapper
    )
    from ..tools.summarizer_prompts import (
        create_summarization_prompt,
        create_insight_extraction_prompt,
        create_document_chunks_prompt,
        create_insight_categorization_prompt
    )
    from ..tools.summarizer_parsers import (
        parse_summary_response,
        parse_insight_response
    )
    from ..config import Config
except ImportError:
    # Fall back to absolute imports (when run as script)
    from tools.openai_logger import openai_logger
    from tools.llm_guardrails_wrapper import (
        initialize_llm_guardrails,
        get_llm_wrapper
    )
    from tools.summarizer_prompts import (
        create_summarization_prompt,
        create_insight_extraction_prompt,
        create_document_chunks_prompt,
        create_insight_categorization_prompt
    )
    from tools.summarizer_parsers import (
        parse_summary_response,
        parse_insight_response
    )
    from config import Config

logger = logging.getLogger(__name__)

@dataclass
class SummaryResult:
    """Represents a summarization result."""
    original_text: str
    summary: str
    key_points: List[str]
    sentiment: str
    confidence: float
    word_count: int
    summary_ratio: float

@dataclass
class InsightExtraction:
    """Represents extracted insights from text."""
    insights: List[str]
    categories: Dict[str, List[str]]
    sentiment_analysis: Dict[str, Any]
    key_metrics: Dict[str, Any]

# Global configuration
_api_key = None
_model = Config.DEFAULT_MODEL

def initialize_summarizer(api_key: str, model: str = None):
    """Initialize the summarizer with API key and model."""
    global _api_key, _model
    _api_key = api_key
    _model = model or Config.DEFAULT_MODEL
    openai.api_key = api_key
    # Initialize guardrails for LLM call validation
    try:
        initialize_llm_guardrails(api_key, _model)
        logger.info("Guardrails initialized for LLM call validation")
    except Exception as e:
        logger.warning(f"Failed to initialize guardrails: {e}. Continuing without guardrails validation.")

@tool(
    description="Summarize financial documents, reports, and text content with AI-powered analysis. Extracts key points, sentiment, and insights. Perfect for processing earnings reports, analyst notes, and financial documents.",
    infer_schema=True,
    parse_docstring=False
)
def summarize_text(text: str, max_length: int = 500, focus_areas: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Summarize financial documents, reports, and text content with AI-powered analysis.
    
    Creates concise summaries of financial documents, extracting key points, sentiment analysis,
    and structured insights. Useful for processing earnings reports, analyst notes, and
    management discussions.
    
    Args:
        text: The text content to summarize.
        max_length: Maximum word count for the summary (default: 500).
        focus_areas: Optional list of focus areas to emphasize.
    
    Returns:
        Dictionary containing original_text, summary, key_points, sentiment, confidence,
        word_count, and summary_ratio. Returns dictionary with 'error' key if summarization fails.
    """
    try:
        if not _api_key:
            return {"error": "Summarizer not initialized. Call initialize_summarizer() first."}
        
        if not text.strip():
            return {
                "original_text": text,
                "summary": "",
                "key_points": [],
                "sentiment": "neutral",
                "confidence": 0.0,
                "word_count": 0,
                "summary_ratio": 0.0,
                "error": "Empty text provided"
            }
        
        # Prepare the prompt
        prompt = create_summarization_prompt(text, max_length, focus_areas)
        
        # Call OpenAI API with guardrails validation and logging
        import time
        start_time = time.time()
        
        try:
            # Use guardrails wrapper if available, otherwise fallback to direct call
            llm_wrapper = get_llm_wrapper()
            if llm_wrapper:
                response = llm_wrapper.chat_completion(
                    messages=[
                        {"role": "system", "content": "You are a financial analyst expert at summarizing complex documents."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_length * 2,  # Allow for structured output
                    temperature=0.1
                )
            else:
                # Fallback to direct OpenAI call with logging
                from .openai_call_wrapper import logged_chat_completion
                from openai import OpenAI as OpenAIClient
                client = OpenAIClient(api_key=_api_key)
                response = logged_chat_completion(
                    client=client,
                    model=_model,
                    messages=[
                        {"role": "system", "content": "You are a financial analyst expert at summarizing complex documents."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_length * 2,  # Allow for structured output
                    temperature=0.1,
                    agent_name="SummarizerTool"
                )
            
            duration_ms = int((time.time() - start_time) * 1000)
            result_text = response.choices[0].message.content
            
        except Exception as api_error:
            logger.error(f"OpenAI API error: {api_error}")
            return {
                "original_text": text,
                "summary": "",
                "key_points": [],
                "sentiment": "neutral",
                "confidence": 0.0,
                "word_count": len(text.split()),
                "summary_ratio": 0.0,
                "error": f"API error: {str(api_error)}"
            }
        
        # Parse the structured response
        try:
            result_data = json.loads(result_text)
        except json.JSONDecodeError:
            # Fallback to simple text parsing
            result_data = {
                "summary": result_text,
                "key_points": [],
                "sentiment": "neutral",
                "confidence": 0.8
            }
        
        # Calculate metrics
        word_count = len(text.split())
        summary_word_count = len(result_data.get("summary", "").split())
        summary_ratio = summary_word_count / word_count if word_count > 0 else 0
        
        return {
            "original_text": text,
            "summary": result_data.get("summary", ""),
            "key_points": result_data.get("key_points", []),
            "sentiment": result_data.get("sentiment", "neutral"),
            "confidence": result_data.get("confidence", 0.8),
            "word_count": word_count,
            "summary_ratio": summary_ratio
        }
        
    except Exception as e:
        logger.error(f"Error in text summarization: {e}")
        return {
            "original_text": text,
            "summary": "",
            "key_points": [],
            "sentiment": "neutral",
            "confidence": 0.0,
            "word_count": len(text.split()) if text else 0,
            "summary_ratio": 0.0,
            "error": f"Summarization failed: {str(e)}"
        }

@tool(
    description="Extract key insights, metrics, and structured information from financial documents and text. Categorizes insights by type (financial, strategic, operational) and provides sentiment analysis. Ideal for analyzing earnings calls, reports, and financial statements.",
    infer_schema=True,
    parse_docstring=False
)
def extract_insights(text: str, categories: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Extract key insights, metrics, and structured information from financial documents.
    
    Performs deep analysis of financial text to extract actionable insights, categorize them
    by type, and provide comprehensive sentiment analysis. Designed for earnings call transcripts,
    annual reports, and financial statements.
    
    Args:
        text: The text content to analyze. Should contain financial or business-related information.
        categories: Optional list of categories to focus on.
    
    Returns:
        Dictionary containing insights (list), categories (dict), sentiment_analysis (dict),
        and key_metrics (dict). Returns dictionary with 'error' key if extraction fails.
    """
    try:
        if not _api_key:
            return {"error": "Summarizer not initialized. Call initialize_summarizer() first."}
        
        if not text.strip():
            return {
                "insights": [],
                "categories": {},
                "sentiment_analysis": {},
                "key_metrics": {},
                "error": "Empty text provided"
            }
        
        # Prepare the prompt
        prompt = create_insight_extraction_prompt(text, categories)
        
        # Call OpenAI API with guardrails validation
        try:
            # Use guardrails wrapper if available, otherwise fallback to direct call
            llm_wrapper = get_llm_wrapper()
            if llm_wrapper:
                response = llm_wrapper.chat_completion(
                    messages=[
                        {"role": "system", "content": "You are a financial analyst expert at extracting insights from documents."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.1
                )
            else:
                # Fallback to direct OpenAI call with logging
                from .openai_call_wrapper import logged_chat_completion
                from openai import OpenAI as OpenAIClient
                client = OpenAIClient(api_key=_api_key)
                response = logged_chat_completion(
                    client=client,
                    model=_model,
                    messages=[
                        {"role": "system", "content": "You are a financial analyst expert at extracting insights from documents."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.1,
                    agent_name="SummarizerTool"
                )
            
            result_text = response.choices[0].message.content
            
        except Exception as api_error:
            logger.error(f"OpenAI API error: {api_error}")
            return {
                "insights": [],
                "categories": {},
                "sentiment_analysis": {},
                "key_metrics": {},
                "error": f"API error: {str(api_error)}"
            }
        
        # Parse the structured response
        try:
            result_data = json.loads(result_text)
        except json.JSONDecodeError:
            # Fallback to simple parsing
            result_data = {
                "insights": [result_text],
                "categories": {},
                "sentiment_analysis": {"sentiment": "neutral", "confidence": 0.5},
                "key_metrics": {}
            }
        
        return {
            "insights": result_data.get("insights", []),
            "categories": result_data.get("categories", {}),
            "sentiment_analysis": result_data.get("sentiment_analysis", {}),
            "key_metrics": result_data.get("key_metrics", {})
        }
        
    except Exception as e:
        logger.error(f"Error in insight extraction: {e}")
        return {
            "insights": [],
            "categories": {},
            "sentiment_analysis": {},
            "key_metrics": {},
            "error": f"Insight extraction failed: {str(e)}"
        }


class SummarizerTool:
    """
    Summarizer Tool for text summarization and insight extraction.
    
    Provides AI-powered summarization capabilities for financial documents,
    management discussions, and other text content.
    """
    
    def __init__(self, api_key: str, model: str = None):
        """
        Initialize the Summarizer Tool.
        
        Args:
            api_key: OpenAI API key
            model: Model to use for summarization (defaults to Config.DEFAULT_MODEL)
        """
        self.api_key = api_key
        self.model = model or Config.DEFAULT_MODEL
        openai.api_key = api_key
        # Initialize guardrails for LLM call validation
        try:
            initialize_llm_guardrails(api_key, self.model)
            logger.info("Guardrails initialized for LLM call validation")
        except Exception as e:
            logger.warning(f"Failed to initialize guardrails: {e}. Continuing without guardrails validation.")
        
    def summarize_text(
        self,
        text: str,
        max_length: int = 500,
        focus_areas: Optional[List[str]] = None
    ) -> SummaryResult:
        """
        Summarize a given text with focus on specific areas.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            focus_areas: Optional list of focus areas (e.g., ['financial', 'strategic'])
            
        Returns:
            SummaryResult object
        """
        try:
            if not text.strip():
                return SummaryResult(
                    original_text=text,
                    summary="",
                    key_points=[],
                    sentiment="neutral",
                    confidence=0.0,
                    word_count=0,
                    summary_ratio=0.0
                )
                
            # Prepare the prompt
            prompt = create_summarization_prompt(text, max_length, focus_areas)
            
            # Call OpenAI API with guardrails validation and logging
            import time
            start_time = time.time()
            
            try:
                # Use guardrails wrapper if available, otherwise fallback to direct call
                llm_wrapper = get_llm_wrapper()
                if llm_wrapper:
                    response = llm_wrapper.chat_completion(
                        messages=[
                            {"role": "system", "content": "You are a financial analyst expert at summarizing complex documents."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=max_length * 2,  # Allow for structured output
                        temperature=0.1
                    )
                else:
                    # Fallback to direct OpenAI call with logging
                    from .openai_call_wrapper import logged_chat_completion
                    from openai import OpenAI as OpenAIClient
                    client = OpenAIClient(api_key=self.api_key)
                    response = logged_chat_completion(
                        client=client,
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a financial analyst expert at summarizing complex documents."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=max_length * 2,  # Allow for structured output
                        temperature=0.1,
                        agent_name="SummarizerTool"
                    )
                
                duration_ms = int((time.time() - start_time) * 1000)
                result_text = response.choices[0].message.content
                
            except Exception as api_error:
                openai_logger.log_error(api_error, self.model, "SummarizerTool")
                raise api_error
            
            # Parse the structured response
            summary_data = parse_summary_response(result_text)
            
            # Calculate metrics
            word_count = len(text.split())
            summary_word_count = len(summary_data.get('summary', '').split())
            summary_ratio = summary_word_count / word_count if word_count > 0 else 0
            
            return SummaryResult(
                original_text=text,
                summary=summary_data.get('summary', ''),
                key_points=summary_data.get('key_points', []),
                sentiment=summary_data.get('sentiment', 'neutral'),
                confidence=summary_data.get('confidence', 0.8),
                word_count=word_count,
                summary_ratio=summary_ratio
            )
            
        except Exception as e:
            logger.error(f"Error summarizing text: {e}")
            return SummaryResult(
                original_text=text,
                summary="Error: Could not summarize text",
                key_points=[],
                sentiment="neutral",
                confidence=0.0,
                word_count=len(text.split()),
                summary_ratio=0.0
            )
            
    def summarize_document_chunks(
        self,
        chunks: List[str],
        max_summary_length: int = 1000
    ) -> SummaryResult:
        """
        Summarize multiple document chunks into a cohesive summary.
        
        Args:
            chunks: List of text chunks to summarize
            max_summary_length: Maximum length of final summary
            
        Returns:
            SummaryResult object
        """
        try:
            if not chunks:
                return SummaryResult(
                    original_text="",
                    summary="",
                    key_points=[],
                    sentiment="neutral",
                    confidence=0.0,
                    word_count=0,
                    summary_ratio=0.0
                )
                
            # Combine chunks with separators
            combined_text = "\n\n---\n\n".join(chunks)
            
            # Create prompt for multi-chunk summarization
            prompt = create_document_chunks_prompt(combined_text, max_summary_length)
            
            # Use guardrails wrapper if available, otherwise fallback to direct call
            llm_wrapper = get_llm_wrapper()
            if llm_wrapper:
                response = llm_wrapper.chat_completion(
                    messages=[
                        {"role": "system", "content": "You are a senior financial analyst expert at analyzing and summarizing complex financial documents."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_summary_length * 2,
                    temperature=0.1
                )
            else:
                # Fallback to direct OpenAI call with logging
                from .openai_call_wrapper import logged_chat_completion
                from openai import OpenAI as OpenAIClient
                client = OpenAIClient(api_key=self.api_key)
                response = logged_chat_completion(
                    client=client,
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a senior financial analyst expert at analyzing and summarizing complex financial documents."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_summary_length * 2,
                    temperature=0.1,
                    agent_name="SummarizerTool"
                )
            
            result_text = response.choices[0].message.content
            
            # Parse the structured response
            summary_data = parse_summary_response(result_text)
            
            # Calculate metrics
            total_word_count = sum(len(chunk.split()) for chunk in chunks)
            summary_word_count = len(summary_data.get('summary', '').split())
            summary_ratio = summary_word_count / total_word_count if total_word_count > 0 else 0
            
            return SummaryResult(
                original_text=combined_text,
                summary=summary_data.get('summary', ''),
                key_points=summary_data.get('key_points', []),
                sentiment=summary_data.get('sentiment', 'neutral'),
                confidence=summary_data.get('confidence', 0.8),
                word_count=total_word_count,
                summary_ratio=summary_ratio
            )
            
        except Exception as e:
            logger.error(f"Error summarizing document chunks: {e}")
            return SummaryResult(
                original_text="\n".join(chunks),
                summary="Error: Could not summarize document chunks",
                key_points=[],
                sentiment="neutral",
                confidence=0.0,
                word_count=sum(len(chunk.split()) for chunk in chunks),
                summary_ratio=0.0
            )
            
    def extract_insights(
        self,
        text: str,
        insight_categories: Optional[List[str]] = None
    ) -> InsightExtraction:
        """
        Extract insights and categorize them from text.
        
        Args:
            text: Text to analyze
            insight_categories: Optional list of categories to focus on
            
        Returns:
            InsightExtraction object
        """
        try:
            if not text.strip():
                return InsightExtraction(
                    insights=[],
                    categories={},
                    sentiment_analysis={},
                    key_metrics={}
                )
                
            # Default categories if none provided
            if insight_categories is None:
                insight_categories = [
                    "financial_performance",
                    "strategic_initiatives",
                    "market_outlook",
                    "risk_factors",
                    "growth_opportunities"
                ]
                
            prompt = create_insight_categorization_prompt(text, insight_categories)
            
            # Use guardrails wrapper if available, otherwise fallback to direct call
            llm_wrapper = get_llm_wrapper()
            if llm_wrapper:
                response = llm_wrapper.chat_completion(
                    messages=[
                        {"role": "system", "content": "You are a financial analyst expert at extracting insights from financial documents."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1500,
                    temperature=0.1
                )
            else:
                # Fallback to direct OpenAI call with logging
                from .openai_call_wrapper import logged_chat_completion
                from openai import OpenAI as OpenAIClient
                client = OpenAIClient(api_key=self.api_key)
                response = logged_chat_completion(
                    client=client,
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a financial analyst expert at extracting insights from financial documents."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1500,
                    temperature=0.1,
                    agent_name="SummarizerTool"
                )
            
            result_text = response.choices[0].message.content
            
            # Parse the structured response
            insight_data = parse_insight_response(result_text)
            
            return InsightExtraction(
                insights=insight_data.get('insights', []),
                categories=insight_data.get('categories', {}),
                sentiment_analysis=insight_data.get('sentiment_analysis', {}),
                key_metrics=insight_data.get('key_metrics', {})
            )
            
        except Exception as e:
            logger.error(f"Error extracting insights: {e}")
            return InsightExtraction(
                insights=[],
                categories={},
                sentiment_analysis={},
                key_metrics={}
            )
            
            
    def validate_api_key(self) -> bool:
        """
        Validate the OpenAI API key.
        
        Returns:
            True if API key is valid, False otherwise
        """
        try:
            # Use guardrails wrapper if available, otherwise fallback to direct call
            llm_wrapper = get_llm_wrapper()
            if llm_wrapper:
                response = llm_wrapper.chat_completion(
                    messages=[{"role": "user", "content": "Test"}],
                    max_tokens=5
                )
            else:
                # Fallback to direct OpenAI call with logging
                from .openai_call_wrapper import logged_chat_completion
                from openai import OpenAI as OpenAIClient
                client = OpenAIClient(api_key=self.api_key)
                response = logged_chat_completion(
                    client=client,
                    model=self.model,
                    messages=[{"role": "user", "content": "Test"}],
                    max_tokens=5,
                    agent_name="SummarizerTool"
                )
            return True
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return False
