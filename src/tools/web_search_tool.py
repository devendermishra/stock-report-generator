"""
Web Search Tool for fetching sector news and trends.
Uses DuckDuckGo search as a free and open-source alternative to Tavily API.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import re
from langchain_core.tools import tool

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

# Initialize DDGS
_ddgs = DDGS() if DDGS else None
_max_results = 10

@tool(
    description="Search for sector-specific news and market trends from Indian financial markets. Returns recent news articles, analysis, and market insights for a given sector. Useful for understanding sector performance and trends.",
    infer_schema=True,
    parse_docstring=False
)
def search_sector_news(sector: str, days_back: int = 7, include_analysis: bool = True) -> Dict[str, Any]:
    """
    Search for sector-specific news and market trends from Indian financial markets.
    
    Retrieves recent news articles, analysis, and market insights for a specified sector.
    
    Args:
        sector: Sector name to search for.
        days_back: Number of days to look back for news (default: 7).
        include_analysis: Whether to include analysis and opinion pieces (default: True).
    
    Returns:
        Dictionary containing query, sector, days_back, time_keywords, results (list),
        total_results, and search_timestamp. Returns dictionary with 'error' key if search fails.
    """
    try:
        if not _ddgs:
            return {"error": "DuckDuckGo search not available. Install ddgs package."}
        
        # Build query with time-based keywords
        time_keywords = _get_time_keywords(days_back)
        query = f"{sector} sector news India NSE market trends {time_keywords}"
        
        if include_analysis:
            query += " analysis outlook"
            
        logger.info(f"Searching for sector news: {query}")
        
        # Perform search with retry logic
        search_results = _search_with_retry(query, max_retries=3)
        
        if not search_results:
            return {"error": "No search results found"}
        
        # Process results and filter by date if possible
        results = []
        filtered_results = _filter_results_by_date(search_results, days_back)
        
        for result in filtered_results[:_max_results]:
            search_result = SearchResult(
                title=result.get('title', ''),
                url=result.get('href', ''),
                content=result.get('body', ''),
                published_date=result.get('date', None),
                relevance_score=None
            )
            results.append(search_result.__dict__)
        
        return {
            "query": query,
            "sector": sector,
            "days_back": days_back,
            "time_keywords": time_keywords,
            "results": results,
            "total_results": len(results),
            "search_timestamp": logger.info("Search completed successfully")
        }
        
    except Exception as e:
        logger.error(f"Error in sector news search: {e}")
        return {"error": f"Search failed: {str(e)}"}

@tool(
    description="Search for company-specific news, announcements, earnings reports, and corporate updates from Indian financial markets. Returns recent news articles and press releases for a specific company and stock symbol.",
    infer_schema=True,
    parse_docstring=False
)
def search_company_news(company_name: str, stock_symbol: str, days_back: int = 7) -> Dict[str, Any]:
    """
    Search for company-specific news, announcements, and corporate updates.
    
    Retrieves recent news articles, press releases, and earnings announcements for a company.
    
    Args:
        company_name: Full company name.
        stock_symbol: NSE stock symbol.
        days_back: Number of days to look back for news (default: 7).
    
    Returns:
        Dictionary containing query, company_name, stock_symbol, days_back, time_keywords,
        results (list), total_results, and search_timestamp. Returns dictionary with 'error' key if search fails.
    """
    try:
        if not _ddgs:
            return {"error": "DuckDuckGo search not available. Install ddgs package."}
        
        # Build query with time-based keywords
        time_keywords = _get_time_keywords(days_back)
        query = f"{company_name} {stock_symbol} news India NSE announcements {time_keywords}"
        
        logger.info(f"Searching for company news: {query}")
        
        # Perform search with retry logic
        search_results = _search_with_retry(query, max_retries=3)
        
        if not search_results:
            return {"error": "No search results found"}
        
        # Process results and filter by date if possible
        results = []
        filtered_results = _filter_results_by_date(search_results, days_back)
        
        for result in filtered_results[:_max_results]:
            search_result = SearchResult(
                title=result.get('title', ''),
                url=result.get('href', ''),
                content=result.get('body', ''),
                published_date=result.get('date', None),
                relevance_score=None
            )
            results.append(search_result.__dict__)
        
        return {
            "query": query,
            "company_name": company_name,
            "stock_symbol": stock_symbol,
            "days_back": days_back,
            "time_keywords": time_keywords,
            "results": results,
            "total_results": len(results),
            "search_timestamp": logger.info("Search completed successfully")
        }
        
    except Exception as e:
        logger.error(f"Error in company news search: {e}")
        return {"error": f"Search failed: {str(e)}"}

@tool(
    description="Search for general market trends, economic analysis, and financial market insights from Indian markets. Use this for broad market research, economic indicators, and trend analysis.",
    infer_schema=True,
    parse_docstring=False
)
def search_market_trends(query: str, max_results: int = 10) -> Dict[str, Any]:
    """
    Search for general market trends, economic analysis, and financial market insights.
    
    Performs web searches for broad market research, economic indicators, and trend analysis.
    
    Args:
        query: Search query string for market trends.
        max_results: Maximum number of results to return (default: 10).
    
    Returns:
        Dictionary containing query, original_query, results (list), total_results, and search_timestamp.
        Returns dictionary with 'error' key if search fails.
    """
    try:
        if not _ddgs:
            return {"error": "DuckDuckGo search not available. Install ddgs package."}
        
        # Enhance query for market trends
        enhanced_query = f"{query} India NSE market analysis trends"
        
        logger.info(f"Searching for market trends: {enhanced_query}")
        
        # Perform search with retry logic
        search_results = _search_with_retry(enhanced_query, max_retries=3)
        
        if not search_results:
            return {"error": "No search results found"}
        
        # Process results
        results = []
        for result in search_results[:max_results]:
            search_result = SearchResult(
                title=result.get('title', ''),
                url=result.get('href', ''),
                content=result.get('body', ''),
                published_date=result.get('date', None),
                relevance_score=None
            )
            results.append(search_result.__dict__)
        
        return {
            "query": enhanced_query,
            "original_query": query,
            "results": results,
            "total_results": len(results),
            "search_timestamp": logger.info("Search completed successfully")
        }
        
    except Exception as e:
        logger.error(f"Error in market trends search: {e}")
        return {"error": f"Search failed: {str(e)}"}

def _get_time_keywords(days_back: int) -> str:
    """
    Convert days_back parameter into search-friendly time keywords.
    
    Args:
        days_back: Number of days to look back
        
    Returns:
        String with time-based search keywords
    """
    if days_back <= 1:
        return "today yesterday"
    elif days_back <= 3:
        return "last 3 days recent"
    elif days_back <= 7:
        return "last week recent"
    elif days_back <= 14:
        return "last 2 weeks recent"
    elif days_back <= 30:
        return "last month recent"
    elif days_back <= 90:
        return "last 3 months recent"
    else:
        return "recent latest"

def _filter_results_by_date(search_results: List[Dict[str, Any]], days_back: int) -> List[Dict[str, Any]]:
    """
    Filter search results by date when possible.
    
    Args:
        search_results: List of search results
        days_back: Number of days to look back
        
    Returns:
        Filtered list of search results
    """
    try:
        from datetime import datetime, timedelta
        
        # Calculate cutoff date
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        filtered_results = []
        for result in search_results:
            # Try to parse the date if available
            result_date = result.get('date')
            if result_date:
                try:
                    # Try different date formats
                    parsed_date = None
                    for date_format in ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%d/%m/%Y', '%m/%d/%Y']:
                        try:
                            parsed_date = datetime.strptime(str(result_date), date_format)
                            break
                        except ValueError:
                            continue
                    
                    # If we can parse the date, check if it's within the range
                    if parsed_date and parsed_date >= cutoff_date:
                        filtered_results.append(result)
                    elif parsed_date is None:
                        # If we can't parse the date, include it anyway
                        filtered_results.append(result)
                except Exception:
                    # If date parsing fails, include the result anyway
                    filtered_results.append(result)
            else:
                # If no date available, include the result
                filtered_results.append(result)
        
        # If no results after filtering, return original results
        return filtered_results if filtered_results else search_results
        
    except Exception as e:
        logger.warning(f"Date filtering failed: {e}. Returning all results.")
        return search_results

def _search_with_retry(query: str, max_retries: int = 3) -> List[Dict[str, Any]]:
    """
    Perform search with retry logic to handle DDGS engine errors.
    
    Args:
        query: Search query
        max_retries: Maximum number of retry attempts
        
    Returns:
        List of search results
    """
    for attempt in range(max_retries):
        try:
            logger.info(f"Search attempt {attempt + 1}/{max_retries} for query: {query}")
            
            # Try different engines if available
            engines_to_try = ['mullvad_google', 'bing', 'duckduckgo']
            
            for engine in engines_to_try:
                try:
                    logger.info(f"Trying search engine: {engine}")
                    search_results = list(_ddgs.text(
                        query,
                        max_results=_max_results,
                        safesearch='moderate',
                        backend=engine
                    ))
                    logger.info(f"Successfully got {len(search_results)} results using {engine}")
                    return search_results
                    
                except Exception as engine_error:
                    logger.warning(f"Engine {engine} failed: {engine_error}")
                    continue
            
            # If all engines fail, try default
            logger.info("Trying default search engine")
            search_results = list(_ddgs.text(
                query,
                max_results=_max_results,
                safesearch='moderate'
            ))
            return search_results
            
        except Exception as e:
            logger.warning(f"Search attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                logger.error(f"All search attempts failed for query: {query}")
                return []
            continue
    
    return []

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
            
            logger.info(f"Performing search with query: {query}")
            
            # Perform search with DuckDuckGo with retry logic
            search_results = self._search_with_retry(query)
            
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
    
    def _search_with_retry(self, query: str, max_retries: int = 3) -> List[Dict[str, Any]]:
        """
        Perform search with retry logic to handle DDGS engine errors.
        
        Args:
            query: Search query
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of search results
        """
        for attempt in range(max_retries):
            try:
                logger.info(f"Search attempt {attempt + 1}/{max_retries} for query: {query}")
                
                # Try different engines if available
                engines_to_try = ['mullvad_google', 'bing', 'duckduckgo']
                
                for engine in engines_to_try:
                    try:
                        logger.info(f"Trying search engine: {engine}")
                        search_results = list(self.ddgs.text(
                            query,
                            max_results=self.max_results,
                            safesearch='moderate',
                            backend=engine
                        ))
                        logger.info(f"Successfully got {len(search_results)} results using {engine}")
                        return search_results
                        
                    except Exception as engine_error:
                        logger.warning(f"Engine {engine} failed: {engine_error}")
                        continue
                
                # If all engines fail, try default
                logger.info("Trying default search engine")
                search_results = list(self.ddgs.text(
                    query,
                    max_results=self.max_results,
                    safesearch='moderate'
                ))
                return search_results
                
            except Exception as e:
                logger.warning(f"Search attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    # Wait before retry
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    logger.error(f"All search attempts failed for query: {query}")
                    return []
        
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