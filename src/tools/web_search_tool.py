"""
Web Search Tool for fetching sector news and trends.
Uses DuckDuckGo search as a free and open-source alternative to Tavily API.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import time
import re

try:
    from ddgs import DDGS
except ImportError:
    DDGS = None
    print("Warning: ddgs not installed. Install with: pip install ddgs")

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Represents a single search result."""
    title: str
    url: str
    content: str
    published_date: Optional[str] = None
    relevance_score: Optional[float] = None

class WebSearchTool:
    """
    Web Search Tool for fetching sector news and market trends.
    
    Uses DuckDuckGo search as a free and open-source alternative to Tavily API.
    Provides similar functionality without requiring API keys or rate limits.
    """
    
    def __init__(self, max_results: int = 10):
        """
        Initialize the Web Search Tool.
        
        Args:
            max_results: Maximum number of results to return
        """
        self.max_results = max_results
        self.ddgs = DDGS() if DDGS else None
        
        if not self.ddgs:
            raise ImportError("ddgs is required. Install with: pip install ddgs")
        
    def search_sector_news(
        self,
        sector: str,
        days_back: int = 7,
        include_analysis: bool = True
    ) -> List[SearchResult]:
        """
        Search for sector-specific news and trends.
        
        Args:
            sector: Sector name (e.g., "Banking", "IT", "Pharmaceuticals")
            days_back: Number of days to look back for news
            include_analysis: Whether to include analysis and opinion pieces
            
        Returns:
            List of SearchResult objects
        """
        query = f"{sector} sector news India NSE market trends"
        
        if include_analysis:
            query += " analysis outlook"
            
        return self._perform_search(query, days_back)
        
    def search_company_news(
        self,
        company_name: str,
        stock_symbol: str,
        days_back: int = 7
    ) -> List[SearchResult]:
        """
        Search for company-specific news and announcements.
        
        Args:
            company_name: Full company name
            stock_symbol: NSE stock symbol
            days_back: Number of days to look back for news
            
        Returns:
            List of SearchResult objects
        """
        query = f"{company_name} {stock_symbol} NSE news announcements results"
        return self._perform_search(query, days_back)
        
    def search_market_trends(
        self,
        timeframe: str = "monthly"
    ) -> List[SearchResult]:
        """
        Search for general market trends and analysis.
        
        Args:
            timeframe: Timeframe for trends (daily, weekly, monthly)
            
        Returns:
            List of SearchResult objects
        """
        query = f"Indian stock market trends {timeframe} NSE BSE analysis"
        return self._perform_search(query, 30)
        
    def search_regulatory_news(
        self,
        sector: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search for regulatory news affecting the market or specific sector.
        
        Args:
            sector: Optional sector to focus on
            
        Returns:
            List of SearchResult objects
        """
        if sector:
            query = f"SEBI RBI regulatory news {sector} sector India"
        else:
            query = "SEBI RBI regulatory news Indian stock market"
            
        return self._perform_search(query, 14)
        
    def _perform_search(
        self,
        query: str,
        days_back: int
    ) -> List[SearchResult]:
        """
        Perform the actual search using DuckDuckGo.
        
        Args:
            query: Search query
            days_back: Number of days to look back
            
        Returns:
            List of SearchResult objects
        """
        try:
            # Add time-based filtering to query
            if days_back <= 7:
                query += " site:moneycontrol.com OR site:economic times OR site:business-standard.com"
            elif days_back <= 30:
                query += " site:nseindia.com OR site:livemint.com OR site:financialexpress.com"
            
            # Perform search with DuckDuckGo
            search_results = list(self.ddgs.text(
                query,
                max_results=self.max_results,
                safesearch='moderate'
            ))
            
            results = []
            for item in search_results:
                # Extract content from the body text
                content = item.get('body', '')
                
                # Try to extract published date from content or URL
                published_date = self._extract_date_from_content(content)
                
                # Calculate a simple relevance score based on query terms
                relevance_score = self._calculate_relevance_score(query, item.get('title', ''), content)
                
                result = SearchResult(
                    title=item.get('title', ''),
                    url=item.get('href', ''),
                    content=content,
                    published_date=published_date,
                    relevance_score=relevance_score
                )
                results.append(result)
                
            # Sort by relevance score (higher is better)
            results.sort(key=lambda x: x.relevance_score or 0, reverse=True)
            
            logger.info(f"Found {len(results)} results for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
            
    def _extract_date_from_content(self, content: str) -> Optional[str]:
        """
        Extract date from content text.
        
        Args:
            content: Content text to extract date from
            
        Returns:
            Extracted date string or None
        """
        # Look for common date patterns
        date_patterns = [
            r'\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})\b',
            r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2}),?\s+(\d{4})\b',
            r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b',
            r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return None
        
    def _calculate_relevance_score(self, query: str, title: str, content: str) -> float:
        """
        Calculate a simple relevance score based on query term matches.
        
        Args:
            query: Original search query
            title: Result title
            content: Result content
            
        Returns:
            Relevance score between 0 and 1
        """
        query_terms = set(query.lower().split())
        title_terms = set(title.lower().split())
        content_terms = set(content.lower().split())
        
        # Count matches in title (weighted higher)
        title_matches = len(query_terms.intersection(title_terms))
        content_matches = len(query_terms.intersection(content_terms))
        
        # Calculate score (title matches weighted 2x)
        total_score = (title_matches * 2) + content_matches
        max_possible = len(query_terms) * 3  # title + content + some buffer
        
        return min(total_score / max_possible, 1.0) if max_possible > 0 else 0.0
        
    def get_trending_topics(self, sector: str) -> List[str]:
        """
        Get trending topics for a specific sector.
        
        Args:
            sector: Sector name
            
        Returns:
            List of trending topic strings
        """
        query = f"trending topics {sector} sector India 2024"
        results = self._perform_search(query, 3)
        
        # Extract trending topics from results
        topics = []
        for result in results[:5]:  # Top 5 results
            # Simple keyword extraction (in a real implementation, you'd use NLP)
            words = result.title.lower().split()
            sector_keywords = [word for word in words if len(word) > 4]
            topics.extend(sector_keywords[:3])
            
        return list(set(topics))  # Remove duplicates
        
    def validate_connection(self) -> bool:
        """
        Validate that DuckDuckGo search is working.
        
        Returns:
            True if search is working, False otherwise
        """
        try:
            test_results = self.ddgs.text("test", max_results=1)
            return len(list(test_results)) > 0
            
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            return False