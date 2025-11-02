"""
Prompt builders for summarizer tool.
Contains functions to create prompts for summarization and insight extraction.
"""

from typing import List, Optional


def create_summarization_prompt(text: str, max_length: int, focus_areas: Optional[List[str]] = None) -> str:
    """Create a prompt for text summarization."""
    focus_instruction = ""
    if focus_areas:
        focus_instruction = f" Focus on these areas: {', '.join(focus_areas)}."
    
    return f"""
Please summarize the following text in approximately {max_length} words.{focus_instruction}

Text to summarize:
{text}

Please provide your response in the following JSON format:
{{
    "summary": "Your summary here",
    "key_points": ["point1", "point2", "point3"],
    "sentiment": "positive/negative/neutral",
    "confidence": 0.8
}}
"""


def create_insight_extraction_prompt(text: str, categories: Optional[List[str]] = None) -> str:
    """Create a prompt for insight extraction."""
    category_instruction = ""
    if categories:
        category_instruction = f" Focus on these categories: {', '.join(categories)}."
    
    return f"""
Please extract insights and key information from the following text.{category_instruction}

Text to analyze:
{text}

Please provide your response in the following JSON format:
{{
    "insights": ["insight1", "insight2", "insight3"],
    "categories": {{
        "financial": ["metric1", "metric2"],
        "strategic": ["strategy1", "strategy2"]
    }},
    "sentiment_analysis": {{
        "sentiment": "positive/negative/neutral",
        "confidence": 0.8
    }},
    "key_metrics": {{
        "metric_name": "value"
    }}
}}
"""


def create_document_chunks_prompt(combined_text: str, max_summary_length: int) -> str:
    """Create a prompt for multi-chunk summarization."""
    return f"""
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


def create_insight_categorization_prompt(text: str, insight_categories: List[str]) -> str:
    """Create a prompt for categorizing insights."""
    return f"""
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
