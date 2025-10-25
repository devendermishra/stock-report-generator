"""
Custom exception classes for the Stock Report Generator.
Provides specific error handling for different components.
"""


class StockReportGeneratorError(Exception):
    """Base exception for Stock Report Generator."""
    pass


class DataRetrievalError(StockReportGeneratorError):
    """Raised when data retrieval fails."""
    pass


class AnalysisError(StockReportGeneratorError):
    """Raised when analysis fails."""
    pass


class ValidationError(StockReportGeneratorError):
    """Raised when validation fails."""
    pass


class ConfigurationError(StockReportGeneratorError):
    """Raised when configuration is invalid."""
    pass


class AgentError(StockReportGeneratorError):
    """Raised when agent operations fail."""
    pass


class ToolError(StockReportGeneratorError):
    """Raised when tool operations fail."""
    pass


class WorkflowError(StockReportGeneratorError):
    """Raised when workflow operations fail."""
    pass


class ReportGenerationError(StockReportGeneratorError):
    """Raised when report generation fails."""
    pass


class ContextError(StockReportGeneratorError):
    """Raised when context management fails."""
    pass
