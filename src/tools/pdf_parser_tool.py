"""
PDF Parser Tool for extracting and processing text from financial documents.
Handles PDF parsing, text extraction, and content chunking for analysis.
"""

import PyPDF2
import fitz  # PyMuPDF
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import re
import os
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class TextChunk:
    """Represents a chunk of extracted text."""
    content: str
    page_number: int
    chunk_index: int
    metadata: Dict[str, Any]

@dataclass
class ParsedDocument:
    """Represents a parsed PDF document."""
    title: str
    total_pages: int
    chunks: List[TextChunk]
    metadata: Dict[str, Any]
    extraction_timestamp: datetime

class PDFParserTool:
    """
    PDF Parser Tool for extracting and processing text from financial documents.
    
    Provides functionality to parse PDFs, extract text, chunk content,
    and prepare it for analysis by other agents.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the PDF Parser Tool.
        
        Args:
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def parse_pdf(self, file_path: str) -> Optional[ParsedDocument]:
        """
        Parse a PDF file and extract text content.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            ParsedDocument object or None if parsing fails
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"PDF file not found: {file_path}")
                return None
            
            # Check if file is too small (likely placeholder)
            file_size = os.path.getsize(file_path)
            if file_size < 1000:  # Less than 1KB is likely a placeholder
                logger.warning(f"PDF file {file_path} is very small ({file_size} bytes), likely a placeholder")
                return self._create_placeholder_document(file_path)
                
            # Try PyMuPDF first (better for complex PDFs)
            try:
                return self._parse_with_pymupdf(file_path)
            except Exception as e:
                logger.warning(f"PyMuPDF parsing failed, trying PyPDF2: {e}")
                return self._parse_with_pypdf2(file_path)
                
        except Exception as e:
            logger.error(f"Failed to parse PDF {file_path}: {e}")
            return None
    
    def _create_placeholder_document(self, file_path: str) -> ParsedDocument:
        """Create a placeholder document for small/placeholder files."""
        try:
            # Read the placeholder content
            with open(file_path, 'r', encoding='utf-8') as f:
                placeholder_content = f.read().strip()
            
            # Create a single chunk with the placeholder content
            chunk = TextChunk(
                content=placeholder_content,
                page_number=1,
                chunk_index=0,
                metadata={
                    "is_placeholder": True,
                    "original_file": file_path
                }
            )
            
            return ParsedDocument(
                title=os.path.basename(file_path),
                total_pages=1,
                chunks=[chunk],
                metadata={
                    "parser": "placeholder",
                    "file_size": os.path.getsize(file_path),
                    "extraction_method": "placeholder_content",
                    "is_placeholder": True
                },
                extraction_timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Error creating placeholder document: {e}")
            return None
            
    def _parse_with_pymupdf(self, file_path: str) -> ParsedDocument:
        """Parse PDF using PyMuPDF (fitz)."""
        doc = fitz.open(file_path)
        title = doc.metadata.get('title', os.path.basename(file_path))
        total_pages = len(doc)
        
        chunks = []
        chunk_index = 0
        
        for page_num in range(total_pages):
            page = doc.load_page(page_num)
            text = page.get_text()
            
            if text.strip():
                # Split text into chunks
                page_chunks = self._split_text_into_chunks(text, page_num, chunk_index)
                chunks.extend(page_chunks)
                chunk_index += len(page_chunks)
                
        doc.close()
        
        return ParsedDocument(
            title=title,
            total_pages=total_pages,
            chunks=chunks,
            metadata={
                "parser": "PyMuPDF",
                "file_size": os.path.getsize(file_path),
                "extraction_method": "full_text"
            },
            extraction_timestamp=datetime.now()
        )
        
    def _parse_with_pypdf2(self, file_path: str) -> ParsedDocument:
        """Parse PDF using PyPDF2 (fallback)."""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            title = pdf_reader.metadata.get('/Title', os.path.basename(file_path)) if pdf_reader.metadata else os.path.basename(file_path)
            total_pages = len(pdf_reader.pages)
            
            chunks = []
            chunk_index = 0
            
            for page_num in range(total_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                
                if text.strip():
                    # Split text into chunks
                    page_chunks = self._split_text_into_chunks(text, page_num, chunk_index)
                    chunks.extend(page_chunks)
                    chunk_index += len(page_chunks)
                    
        return ParsedDocument(
            title=title,
            total_pages=total_pages,
            chunks=chunks,
            metadata={
                "parser": "PyPDF2",
                "file_size": os.path.getsize(file_path),
                "extraction_method": "full_text"
            },
            extraction_timestamp=datetime.now()
        )
        
    def _split_text_into_chunks(
        self,
        text: str,
        page_number: int,
        start_chunk_index: int
    ) -> List[TextChunk]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to split
            page_number: Page number
            start_chunk_index: Starting chunk index
            
        Returns:
            List of TextChunk objects
        """
        chunks = []
        text = text.strip()
        
        if len(text) <= self.chunk_size:
            # Text fits in one chunk
            chunks.append(TextChunk(
                content=text,
                page_number=page_number,
                chunk_index=start_chunk_index,
                metadata={
                    "chunk_type": "full_page",
                    "word_count": len(text.split()),
                    "char_count": len(text)
                }
            ))
        else:
            # Split into multiple chunks with overlap
            start = 0
            chunk_index = start_chunk_index
            
            while start < len(text):
                end = start + self.chunk_size
                
                # Try to break at sentence boundary
                if end < len(text):
                    # Look for sentence endings within the last 100 characters
                    search_start = max(end - 100, start)
                    sentence_end = text.rfind('.', search_start, end)
                    if sentence_end > start:
                        end = sentence_end + 1
                        
                chunk_text = text[start:end].strip()
                
                if chunk_text:
                    chunks.append(TextChunk(
                        content=chunk_text,
                        page_number=page_number,
                        chunk_index=chunk_index,
                        metadata={
                            "chunk_type": "partial",
                            "word_count": len(chunk_text.split()),
                            "char_count": len(chunk_text),
                            "start_pos": start,
                            "end_pos": end
                        }
                    ))
                    chunk_index += 1
                    
                # Move start position with overlap
                start = end - self.chunk_overlap
                if start >= len(text):
                    break
                    
        return chunks
        
    def extract_financial_metrics(self, document: ParsedDocument) -> Dict[str, Any]:
        """
        Extract financial metrics from a parsed document.
        
        Args:
            document: ParsedDocument object
            
        Returns:
            Dictionary containing extracted financial metrics
        """
        try:
            metrics = {
                "revenue": [],
                "profit": [],
                "eps": [],
                "pe_ratio": [],
                "market_cap": [],
                "debt": [],
                "cash": []
            }
            
            # Patterns for common financial metrics
            patterns = {
                "revenue": [
                    r"revenue[:\s]*₹?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:crore|cr|million|billion|bn)",
                    r"total income[:\s]*₹?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:crore|cr|million|billion|bn)",
                    r"sales[:\s]*₹?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:crore|cr|million|billion|bn)"
                ],
                "profit": [
                    r"profit[:\s]*₹?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:crore|cr|million|billion|bn)",
                    r"net profit[:\s]*₹?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:crore|cr|million|billion|bn)",
                    r"pbt[:\s]*₹?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:crore|cr|million|billion|bn)"
                ],
                "eps": [
                    r"eps[:\s]*₹?\s*([0-9,]+(?:\.[0-9]+)?)",
                    r"earnings per share[:\s]*₹?\s*([0-9,]+(?:\.[0-9]+)?)"
                ],
                "pe_ratio": [
                    r"p/e[:\s]*([0-9,]+(?:\.[0-9]+)?)",
                    r"pe ratio[:\s]*([0-9,]+(?:\.[0-9]+)?)"
                ]
            }
            
            # Search through all chunks
            for chunk in document.chunks:
                text = chunk.content.lower()
                
                for metric_type, pattern_list in patterns.items():
                    for pattern in pattern_list:
                        matches = re.findall(pattern, text, re.IGNORECASE)
                        for match in matches:
                            try:
                                # Clean and convert the number
                                clean_match = match.replace(',', '')
                                if '.' in clean_match:
                                    value = float(clean_match)
                                else:
                                    value = int(clean_match)
                                metrics[metric_type].append({
                                    "value": value,
                                    "page": chunk.page_number,
                                    "context": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
                                })
                            except ValueError:
                                continue
                                
            # Remove duplicates and sort
            for metric_type in metrics:
                if metrics[metric_type]:
                    # Remove duplicates based on value
                    seen = set()
                    unique_metrics = []
                    for metric in metrics[metric_type]:
                        if metric["value"] not in seen:
                            seen.add(metric["value"])
                            unique_metrics.append(metric)
                    metrics[metric_type] = unique_metrics
                    
            logger.info(f"Extracted financial metrics from {document.title}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error extracting financial metrics: {e}")
            return {}
            
    def extract_management_insights(self, document: ParsedDocument) -> List[str]:
        """
        Extract management insights and key statements.
        
        Args:
            document: ParsedDocument object
            
        Returns:
            List of key insights
        """
        try:
            insights = []
            
            # Keywords that indicate management insights
            insight_keywords = [
                "management discussion",
                "outlook",
                "strategy",
                "growth",
                "expansion",
                "investment",
                "future plans",
                "guidance",
                "challenges",
                "opportunities"
            ]
            
            for chunk in document.chunks:
                text = chunk.content.lower()
                
                # Check if chunk contains insight keywords
                for keyword in insight_keywords:
                    if keyword in text:
                        # Extract sentences containing the keyword
                        sentences = re.split(r'[.!?]+', chunk.content)
                        for sentence in sentences:
                            if keyword in sentence.lower() and len(sentence.strip()) > 50:
                                insights.append(sentence.strip())
                                
            # Remove duplicates and limit to top insights
            unique_insights = list(set(insights))
            return unique_insights[:10]  # Return top 10 insights
            
        except Exception as e:
            logger.error(f"Error extracting management insights: {e}")
            return []
            
    def get_document_summary(self, document: ParsedDocument) -> Dict[str, Any]:
        """
        Get a summary of the parsed document.
        
        Args:
            document: ParsedDocument object
            
        Returns:
            Dictionary containing document summary
        """
        try:
            total_chunks = len(document.chunks)
            total_text_length = sum(len(chunk.content) for chunk in document.chunks)
            total_words = sum(chunk.metadata.get("word_count", 0) for chunk in document.chunks)
            
            # Get page distribution
            page_distribution = {}
            for chunk in document.chunks:
                page = chunk.page_number
                page_distribution[page] = page_distribution.get(page, 0) + 1
                
            return {
                "title": document.title,
                "total_pages": document.total_pages,
                "total_chunks": total_chunks,
                "total_text_length": total_text_length,
                "total_words": total_words,
                "avg_chunk_size": total_text_length / total_chunks if total_chunks > 0 else 0,
                "page_distribution": page_distribution,
                "extraction_timestamp": document.extraction_timestamp.isoformat(),
                "parser_metadata": document.metadata
            }
            
        except Exception as e:
            logger.error(f"Error creating document summary: {e}")
            return {}
            
    def search_in_document(
        self,
        document: ParsedDocument,
        search_terms: List[str],
        case_sensitive: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search for specific terms in the document.
        
        Args:
            document: ParsedDocument object
            search_terms: List of terms to search for
            case_sensitive: Whether search should be case sensitive
            
        Returns:
            List of search results
        """
        try:
            results = []
            
            for chunk in document.chunks:
                text = chunk.content if case_sensitive else chunk.content.lower()
                
                for term in search_terms:
                    search_term = term if case_sensitive else term.lower()
                    
                    if search_term in text:
                        # Find all occurrences
                        start = 0
                        while True:
                            pos = text.find(search_term, start)
                            if pos == -1:
                                break
                                
                            # Extract context around the match
                            context_start = max(0, pos - 100)
                            context_end = min(len(chunk.content), pos + len(term) + 100)
                            context = chunk.content[context_start:context_end]
                            
                            results.append({
                                "term": term,
                                "page": chunk.page_number,
                                "chunk_index": chunk.chunk_index,
                                "position": pos,
                                "context": context,
                                "full_chunk": chunk.content
                            })
                            
                            start = pos + 1
                            
            return results
            
        except Exception as e:
            logger.error(f"Error searching in document: {e}")
            return []
