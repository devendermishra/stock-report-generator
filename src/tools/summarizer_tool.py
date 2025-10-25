"""
Summarizer Tool for text summarization and insight extraction.
Provides AI-powered summarization capabilities for financial documents.
"""

import openai
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import json

try:
    # Try relative imports first (when run as module)
    from ..tools.openai_logger import openai_logger
except ImportError:
    # Fall back to absolute imports (when run as script)
    from tools.openai_logger import openai_logger

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

class SummarizerTool:
    """
    Summarizer Tool for text summarization and insight extraction.
    
    Provides AI-powered summarization capabilities for financial documents,
    management discussions, and other text content.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        """
        Initialize the Summarizer Tool.
        
        Args:
            api_key: OpenAI API key
            model: Model to use for summarization
        """
        self.api_key = api_key
        self.model = model
        openai.api_key = api_key
        
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
            prompt = self._create_summarization_prompt(text, max_length, focus_areas)
            
            # Call OpenAI API with logging
            import time
            start_time = time.time()
            
            try:
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a financial analyst expert at summarizing complex documents."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_length * 2,  # Allow for structured output
                    temperature=0.1
                )
                
                duration_ms = int((time.time() - start_time) * 1000)
                result_text = response.choices[0].message.content
                
                # Log the OpenAI completion
                openai_logger.log_chat_completion(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a financial analyst expert at summarizing complex documents."},
                        {"role": "user", "content": prompt}
                    ],
                    response=result_text,
                    usage=response.usage.__dict__ if response.usage else None,
                    duration_ms=duration_ms,
                    agent_name="SummarizerTool"
                )
                
            except Exception as api_error:
                openai_logger.log_error(api_error, self.model, "SummarizerTool")
                raise api_error
            
            # Parse the structured response
            summary_data = self._parse_summary_response(result_text)
            
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
            prompt = f"""
            Please analyze and summarize the following financial document chunks. 
            Focus on key financial metrics, strategic insights, and management outlook.
            
            Document chunks:
            {combined_text}
            
            Provide a comprehensive summary in the following JSON format:
            {{
                "summary": "Main summary of the document",
                "key_points": ["Point 1", "Point 2", "Point 3"],
                "financial_metrics": {{"revenue": "value", "profit": "value"}},
                "strategic_insights": ["Insight 1", "Insight 2"],
                "sentiment": "positive/negative/neutral",
                "confidence": 0.85
            }}
            """
            
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a senior financial analyst expert at analyzing and summarizing complex financial documents."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_summary_length * 2,
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content
            
            # Parse the structured response
            summary_data = self._parse_summary_response(result_text)
            
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
                
            prompt = f"""
            Analyze the following financial text and extract key insights. 
            Categorize insights into the following categories: {', '.join(insight_categories)}
            
            Text: {text}
            
            Provide your analysis in the following JSON format:
            {{
                "insights": ["Insight 1", "Insight 2", "Insight 3"],
                "categories": {{
                    "financial_performance": ["Financial insight 1", "Financial insight 2"],
                    "strategic_initiatives": ["Strategic insight 1"],
                    "market_outlook": ["Market insight 1"],
                    "risk_factors": ["Risk insight 1"],
                    "growth_opportunities": ["Growth insight 1"]
                }},
                "sentiment_analysis": {{
                    "overall_sentiment": "positive/negative/neutral",
                    "confidence": 0.85,
                    "key_sentiment_indicators": ["Indicator 1", "Indicator 2"]
                }},
                "key_metrics": {{
                    "revenue_growth": "X%",
                    "profit_margin": "Y%",
                    "market_share": "Z%"
                }}
            }}
            """
            
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial analyst expert at extracting insights from financial documents."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content
            
            # Parse the structured response
            insight_data = self._parse_insight_response(result_text)
            
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
            
    def _create_summarization_prompt(
        self,
        text: str,
        max_length: int,
        focus_areas: Optional[List[str]]
    ) -> str:
        """Create a prompt for text summarization."""
        focus_instruction = ""
        if focus_areas:
            focus_instruction = f" Focus particularly on: {', '.join(focus_areas)}."
            
        return f"""
        Please summarize the following financial text in no more than {max_length} words.{focus_instruction}
        
        Text: {text}
        
        Provide your summary in the following JSON format:
        {{
            "summary": "Main summary of the text",
            "key_points": ["Key point 1", "Key point 2", "Key point 3"],
            "sentiment": "positive/negative/neutral",
            "confidence": 0.85
        }}
        """
        
    def _parse_summary_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the structured response from the summarization API."""
        try:
            # Try to extract JSON from the response
            import re
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
            
    def _parse_insight_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the structured response from the insight extraction API."""
        try:
            # Try to extract JSON from the response
            import re
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
            
    def validate_api_key(self) -> bool:
        """
        Validate the OpenAI API key.
        
        Returns:
            True if API key is valid, False otherwise
        """
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5
            )
            return True
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return False
