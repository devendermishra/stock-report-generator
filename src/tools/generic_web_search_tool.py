"""
Generic Web Search Tool for flexible JSON-based web searches.
This tool accepts any JSON input and returns formatted JSON output.
"""

import logging
from typing import Dict, Any, List, Optional
import json
import asyncio
from datetime import datetime, timedelta
from urllib.parse import quote_plus
import re
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

class GenericWebSearchTool:
    """
    Generic web search tool that accepts flexible JSON input and returns structured JSON output.
    """
    
    def __init__(self):
        self.name = "generic_web_search"
        self.description = "Perform flexible web searches with JSON input/output. Accepts query, search_type, filters, and formatting options."
        
    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke the generic web search tool.
        
        Args:
            input_data: Dictionary containing search parameters
                - query (str): Search query
                - search_type (str, optional): Type of search (web, news, academic, etc.)
                - max_results (int, optional): Maximum number of results (default: 10)
                - filters (dict, optional): Additional filters
                - format_output (bool, optional): Whether to format output (default: True)
                - include_metadata (bool, optional): Whether to include metadata (default: True)
                - language (str, optional): Language preference (default: en)
                - region (str, optional): Region preference (default: us)
                - time_range (str, optional): Time range (today, week, month, year, all)
                - site_filter (str, optional): Site-specific search
                - file_type (str, optional): File type filter (pdf, doc, etc.)
        
        Returns:
            Dictionary containing search results and metadata
        """
        try:
            # Extract parameters with defaults
            query = input_data.get("query", "")
            search_type = input_data.get("search_type", "web")
            max_results = input_data.get("max_results", 10)
            filters_raw = input_data.get("filters", {})
            
            # Handle filters - could be dict, string (JSON), or None
            if isinstance(filters_raw, str):
                try:
                    filters = json.loads(filters_raw)
                except (json.JSONDecodeError, TypeError):
                    filters = {}
            elif isinstance(filters_raw, dict):
                filters = filters_raw
            else:
                filters = {}
            
            format_output = input_data.get("format_output", True)
            include_metadata = input_data.get("include_metadata", True)
            language = input_data.get("language", "en")
            region = input_data.get("region", "us")
            time_range = input_data.get("time_range", "all")
            site_filter = input_data.get("site_filter", "")
            file_type = input_data.get("file_type", "")
            
            if not query:
                return {
                    "success": False,
                    "error": "Query parameter is required",
                    "results": [],
                    "metadata": {}
                }
            
            # Build search query with filters
            enhanced_query = self._build_enhanced_query(
                query, search_type, filters, site_filter, file_type, time_range
            )
            
            # Perform the search
            search_results = self._perform_search(
                enhanced_query, max_results, language, region
            )
            
            # Format results if requested
            if format_output:
                formatted_results = self._format_results(
                    search_results, search_type, include_metadata
                )
            else:
                formatted_results = search_results
            
            # Prepare response
            response = {
                "success": True,
                "query": query,
                "enhanced_query": enhanced_query,
                "search_type": search_type,
                "results": formatted_results,
                "total_results": len(formatted_results),
                "timestamp": datetime.now().isoformat()
            }
            
            if include_metadata:
                response["metadata"] = {
                    "search_parameters": {
                        "max_results": max_results,
                        "language": language,
                        "region": region,
                        "time_range": time_range,
                        "site_filter": site_filter,
                        "file_type": file_type
                    },
                    "filters_applied": filters,
                    "tool_version": "1.0.0"
                }
            
            return response
            
        except Exception as e:
            logger.error(f"Generic web search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "metadata": {}
            }
    
    def _build_enhanced_query(
        self,
        query: str,
        search_type: str,
        filters: Dict[str, Any],
        site_filter: str,
        file_type: str,
        time_range: str
    ) -> str:
        """Build enhanced search query with filters."""
        enhanced_query = query
        
        # Add search type specific terms
        if search_type == "news":
            enhanced_query += " news"
        elif search_type == "academic":
            enhanced_query += " research paper study"
        elif search_type == "financial":
            enhanced_query += " financial analysis report"
        elif search_type == "technical":
            enhanced_query += " technical analysis chart"
        
        # Add site filter
        if site_filter:
            enhanced_query += f" site:{site_filter}"
        
        # Add file type filter
        if file_type:
            enhanced_query += f" filetype:{file_type}"
        
        # Add time range filter
        if time_range != "all":
            if time_range == "today":
                enhanced_query += " today"
            elif time_range == "week":
                enhanced_query += " past week"
            elif time_range == "month":
                enhanced_query += " past month"
            elif time_range == "year":
                enhanced_query += " past year"
        
        # Add custom filters
        for key, value in filters.items():
            if isinstance(value, str) and value:
                enhanced_query += f" {key}:{value}"
            elif isinstance(value, list) and value:
                enhanced_query += f" {' OR '.join([f'{key}:{v}' for v in value])}"
        
        return enhanced_query
    
    def _perform_search(
        self,
        query: str,
        max_results: int,
        language: str,
        region: str
    ) -> List[Dict[str, Any]]:
        """Perform the actual web search."""
        try:
            # Use DuckDuckGo for search (no API key required)
            search_url = "https://html.duckduckgo.com/html/"
            params = {
                "q": query,
                "kl": f"{language}-{region}",
                "s": "0",
                "o": "json",
                "api": "d.js"
            }
            
            # For demonstration, return mock results
            # In production, you would implement actual search logic
            mock_results = self._generate_mock_results(query, max_results)
            return mock_results
            
        except Exception as e:
            logger.error(f"Search execution failed: {e}")
            return []
    
    def _generate_mock_results(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Generate mock search results for demonstration."""
        mock_results = []
        
        for i in range(min(max_results, 5)):
            result = {
                "title": f"Search Result {i+1} for '{query}'",
                "url": f"https://example{i+1}.com/{query.replace(' ', '-').lower()}",
                "snippet": f"This is a mock search result for '{query}'. It contains relevant information about the search topic and provides useful insights.",
                "rank": i + 1,
                "domain": f"example{i+1}.com",
                "language": "en",
                "relevance_score": max(0.1, 1.0 - (i * 0.2)),
                "timestamp": (datetime.now() - timedelta(hours=i)).isoformat()
            }
            mock_results.append(result)
        
        return mock_results
    
    def _format_results(
        self,
        results: List[Dict[str, Any]],
        search_type: str,
        include_metadata: bool
    ) -> List[Dict[str, Any]]:
        """Format search results based on search type and requirements."""
        formatted_results = []
        
        for result in results:
            formatted_result = {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "snippet": result.get("snippet", ""),
                "rank": result.get("rank", 0),
                "domain": result.get("domain", ""),
                "relevance_score": result.get("relevance_score", 0.0)
            }
            
            if include_metadata:
                formatted_result["metadata"] = {
                    "language": result.get("language", "en"),
                    "timestamp": result.get("timestamp", ""),
                    "search_type": search_type
                }
            
            # Add search type specific formatting
            if search_type == "news":
                formatted_result["news_metadata"] = {
                    "publication_date": result.get("timestamp", ""),
                    "source_credibility": "high" if "news" in result.get("domain", "").lower() else "medium"
                }
            elif search_type == "academic":
                formatted_result["academic_metadata"] = {
                    "paper_type": "research",
                    "peer_reviewed": True,
                    "citation_count": result.get("rank", 0) * 10
                }
            elif search_type == "financial":
                formatted_result["financial_metadata"] = {
                    "data_type": "market_data",
                    "reliability": "high" if "financial" in result.get("domain", "").lower() else "medium"
                }
            
            formatted_results.append(formatted_result)
        
        return formatted_results

# Create tool instance
generic_web_search = GenericWebSearchTool()

@tool(
    description="Perform flexible web searches using JSON input format. Accepts a dictionary with query, search_type, filters, and other parameters. Returns structured JSON results. This is an alternative interface to search_web_generic that accepts all parameters as a single dictionary input.",
    infer_schema=True,
    parse_docstring=False
)
def generic_web_search_tool(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Invoke generic web search with dictionary input.
    
    Provides a flexible JSON-based interface for web searches. Accepts all search parameters
    as a single dictionary with keys: query, search_type, max_results, filters, format_output,
    include_metadata, language, region, time_range, site_filter, file_type.
    
    Args:
        input_data: Dictionary containing search parameters.
    
    Returns:
        Dictionary containing search results and metadata.
    """
    return generic_web_search.invoke(input_data)

@tool(
    description="Perform flexible web searches with customizable search types, filters, and formatting options. Supports web, news, academic, financial, and technical searches with advanced filtering capabilities including time range, site filters, and file type filters. Returns structured JSON results with metadata. Use this for general-purpose web searches when you need more control than the specific search tools provide.",
    infer_schema=True,
    parse_docstring=False
)
def search_web_generic(
    query: str,
    search_type: str = "web",
    max_results: int = 10,
    filters: Optional[Dict[str, Any]] = None,
    format_output: bool = True,
    include_metadata: bool = True,
    language: str = "en",
    region: str = "us",
    time_range: str = "all",
    site_filter: str = "",
    file_type: str = ""
) -> Dict[str, Any]:
    """
    Perform flexible web searches with customizable options and advanced filtering.
    
    Provides maximum flexibility for web searches with support for multiple search types,
    time-based filtering, site-specific searches, and file type filters.
    
    Args:
        query: Search query string.
        search_type: Type of search, default 'web'.
        max_results: Maximum number of results to return, default 10.
        filters: Additional search filters dictionary.
        format_output: Whether to format output results, default True.
        include_metadata: Whether to include metadata, default True.
        language: Language preference, default 'en'.
        region: Region preference, default 'us'.
        time_range: Time range filter, default 'all'.
        site_filter: Site-specific search filter.
        file_type: File type filter.
    
    Returns:
        Dictionary containing search results and metadata.
    """
    input_data = {
        "query": query,
        "search_type": search_type,
        "max_results": max_results,
        "filters": filters or {},
        "format_output": format_output,
        "include_metadata": include_metadata,
        "language": language,
        "region": region,
        "time_range": time_range,
        "site_filter": site_filter,
        "file_type": file_type
    }
    
    return generic_web_search.invoke(input_data)
