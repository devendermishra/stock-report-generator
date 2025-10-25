"""
Data models for report formatting operations.
Contains dataclasses and type definitions for report-related data structures.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional


@dataclass
class ReportSection:
    """Represents a section of the final report."""
    title: str
    content: str
    level: int
    order: int
    metadata: Dict[str, Any]


@dataclass
class ConsistencyIssue:
    """Represents a consistency issue found in the report."""
    issue_type: str
    description: str
    severity: str
    location: str
    suggestion: str


@dataclass
class FormattedReport:
    """Represents a formatted report."""
    title: str
    sections: List[ReportSection]
    metadata: Dict[str, Any]
    markdown_content: str
    word_count: int
    creation_timestamp: datetime
