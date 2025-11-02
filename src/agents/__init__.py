"""
Multi-agent system for autonomous stock research report generation.
"""

try:
    # Try relative imports first (when run as module)
    from .base_agent import BaseAgent
    from .research_agent import ResearchAgent
    from .research_planner_agent import ResearchPlannerAgent
    from .financial_analysis_agent import FinancialAnalysisAgent
    from .management_analysis_agent import ManagementAnalysisAgent
    from .technical_analysis_agent import TechnicalAnalysisAgent
    from .valuation_analysis_agent import ValuationAnalysisAgent
    from .report_agent import ReportAgent
except ImportError:
    # Fall back to absolute imports (when run as script)
    from base_agent import BaseAgent
    from research_agent import ResearchAgent
    from research_planner_agent import ResearchPlannerAgent
    from financial_analysis_agent import FinancialAnalysisAgent
    from management_analysis_agent import ManagementAnalysisAgent
    from technical_analysis_agent import TechnicalAnalysisAgent
    from valuation_analysis_agent import ValuationAnalysisAgent
    from report_agent import ReportAgent

__all__ = [
    "BaseAgent",
    "ResearchAgent",
    "ResearchPlannerAgent", 
    "AnalysisAgent",
    "FinancialAnalysisAgent",
    "ManagementAnalysisAgent",
    "TechnicalAnalysisAgent",
    "ValuationAnalysisAgent",
    "ReportAgent"
]
