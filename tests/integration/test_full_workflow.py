"""
Integration tests for the full stock report generation workflow.

This test suite verifies the end-to-end functionality of the multi-agent system,
including agent collaboration, tool usage, and report generation.
"""

import pytest
import sys
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from main import StockReportGenerator
from config import Config


class TestFullWorkflow:
    """Integration tests for the complete stock report generation workflow."""
    
    @pytest.fixture
    def mock_openai_api_key(self):
        """Provide a mock OpenAI API key for testing."""
        return "test-openai-api-key-12345"
    
    @pytest.fixture
    def mock_stock_data(self):
        """Provide mock stock data for testing."""
        return {
            "current_price": 2500.0,
            "market_cap": 1000000000000,
            "high_52w": 2800.0,
            "low_52w": 2200.0,
            "pe_ratio": 25.5,
            "pb_ratio": 3.2,
            "eps": 98.0,
            "dividend_yield": 1.5,
            "beta": 1.1,
            "volume": 1000000,
            "avg_volume": 1200000,
            "change_percent": 2.5
        }
    
    @pytest.fixture
    def mock_company_info(self):
        """Provide mock company information for testing."""
        return {
            "company_name": "Test Company Limited",
            "short_name": "TESTCOMP",
            "sector": "Technology",
            "industry": "Software",
            "description": "A test technology company",
            "website": "https://testcompany.com",
            "employees": 5000
        }
    
    @pytest.fixture
    def mock_research_data(self):
        """Provide mock research agent results."""
        return {
            "company_data": {
                "stock_metrics": {
                    "current_price": 2500.0,
                    "market_cap": 1000000000000,
                    "pe_ratio": 25.5
                },
                "company_info": {
                    "company_name": "Test Company Limited",
                    "sector": "Technology"
                }
            },
            "gathered_data": {
                "get_stock_metrics": {
                    "current_price": 2500.0,
                    "market_cap": 1000000000000
                },
                "get_company_info": {
                    "company_name": "Test Company Limited",
                    "sector": "Technology"
                }
            },
            "sector_analysis": {
                "sector": "Technology",
                "summary": "Technology sector is growing",
                "trends": ["Cloud computing", "AI adoption"]
            }
        }
    
    @pytest.fixture
    def mock_analysis_results(self):
        """Provide mock analysis agent results."""
        return {
            "financial_analysis": {
                "revenue_growth": 15.5,
                "profit_growth": 20.0,
                "roe": 18.5,
                "roa": 12.0,
                "summary": "Strong financial performance"
            },
            "management_analysis": {
                "management_outlook": "Positive outlook",
                "strategic_initiatives": ["Digital transformation", "Market expansion"],
                "summary": "Strong management team"
            },
            "technical_analysis": {
                "trend": "Bullish",
                "support_level": 2400.0,
                "resistance_level": 2600.0,
                "summary": "Positive technical indicators"
            },
            "valuation_analysis": {
                "target_price": 2700.0,
                "recommendation": "BUY",
                "summary": "Undervalued with growth potential"
            }
        }
    
    @pytest.fixture
    def mock_llm_response(self):
        """Provide mock LLM response for AI agents."""
        mock_response = MagicMock()
        mock_response.content = "This is a comprehensive analysis of the stock."
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "This is a comprehensive analysis of the stock."
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        return mock_response
    
    def _mock_tool_calls(self, tool_name: str, return_value: Any):
        """Helper to mock tool calls."""
        mock_tool = MagicMock()
        mock_tool.name = tool_name
        mock_tool.invoke = Mock(return_value=return_value)
        mock_tool.ainvoke = AsyncMock(return_value=return_value)
        return mock_tool
    
    @pytest.mark.asyncio
    @patch('main.MultiAgentOrchestrator')
    @patch('main.tool_validate_symbol')
    @patch('main.tool_get_company_info')
    async def test_workflow_ai_mode_success(
        self,
        mock_get_company_info,
        mock_validate_symbol,
        mock_orchestrator_class,
        mock_openai_api_key,
        mock_stock_data,
        mock_company_info,
        mock_research_data,
        mock_analysis_results
    ):
        """Test successful workflow execution in AI mode."""
        # Setup mocks
        mock_validate_symbol.invoke.return_value = {
            "valid": True,
            "company_name": "Test Company Limited",
            "sector": "Technology"
        }
        mock_get_company_info.invoke.return_value = mock_company_info
        
        # Mock orchestrator
        mock_orchestrator = MagicMock()
        mock_orchestrator.run_workflow = AsyncMock(return_value={
            "workflow_status": "completed",
            "stock_symbol": "TESTCOMP",
            "company_name": "Test Company Limited",
            "sector": "Technology",
            "final_report": "# Stock Report\n\n## Executive Summary\n\nTest report content.",
            "pdf_path": "reports/TESTCOMP_report.pdf",
            "duration_seconds": 30.5,
            "errors": []
        })
        mock_orchestrator_class.return_value = mock_orchestrator
        
        # Initialize generator
        generator = StockReportGenerator(
            openai_api_key=mock_openai_api_key,
            use_ai_research=True,
            use_ai_analysis=True,
            skip_pdf=True
        )
        
        # Generate report
        results = await generator.generate_report(
            stock_symbol="TESTCOMP",
            company_name="Test Company Limited",
            sector="Technology"
        )
        
        # Assertions
        assert results["workflow_status"] == "completed"
        assert results["stock_symbol"] == "TESTCOMP"
        assert results["company_name"] == "Test Company Limited"
        assert results["sector"] == "Technology"
        assert "final_report" in results
        assert "session_id" in results
        assert len(results.get("errors", [])) == 0
        
        # Verify orchestrator was called correctly
        mock_orchestrator.run_workflow.assert_called_once()
        call_args = mock_orchestrator.run_workflow.call_args
        assert call_args.kwargs["stock_symbol"] == "TESTCOMP"
        assert call_args.kwargs["company_name"] == "Test Company Limited"
        assert call_args.kwargs["sector"] == "Technology"
        assert call_args.kwargs["skip_pdf"] is True
    
    @pytest.mark.asyncio
    @patch('main.MultiAgentOrchestrator')
    @patch('main.tool_validate_symbol')
    @patch('main.tool_get_company_info')
    async def test_workflow_structured_mode_success(
        self,
        mock_get_company_info,
        mock_validate_symbol,
        mock_orchestrator_class,
        mock_openai_api_key,
        mock_company_info
    ):
        """Test successful workflow execution in structured mode."""
        # Setup mocks
        mock_validate_symbol.invoke.return_value = {
            "valid": True,
            "company_name": "Test Company Limited",
            "sector": "Technology"
        }
        mock_get_company_info.invoke.return_value = mock_company_info
        
        # Mock orchestrator
        mock_orchestrator = MagicMock()
        mock_orchestrator.run_workflow = AsyncMock(return_value={
            "workflow_status": "completed",
            "stock_symbol": "TESTCOMP",
            "company_name": "Test Company Limited",
            "sector": "Technology",
            "final_report": "# Stock Report\n\n## Executive Summary\n\nTest report content.",
            "pdf_path": "reports/TESTCOMP_report.pdf",
            "duration_seconds": 45.2,
            "errors": []
        })
        mock_orchestrator_class.return_value = mock_orchestrator
        
        # Initialize generator in structured mode
        generator = StockReportGenerator(
            openai_api_key=mock_openai_api_key,
            use_ai_research=False,
            use_ai_analysis=False,
            skip_pdf=True
        )
        
        # Generate report
        results = await generator.generate_report(
            stock_symbol="TESTCOMP"
        )
        
        # Assertions
        assert results["workflow_status"] == "completed"
        assert results["stock_symbol"] == "TESTCOMP"
        assert "final_report" in results
        assert "session_id" in results
    
    @pytest.mark.asyncio
    @patch('main.MultiAgentOrchestrator')
    @patch('main.tool_validate_symbol')
    async def test_workflow_invalid_symbol(
        self,
        mock_validate_symbol,
        mock_orchestrator_class,
        mock_openai_api_key
    ):
        """Test workflow with invalid stock symbol."""
        # Setup mock to return invalid symbol
        mock_validate_symbol.invoke.return_value = {
            "valid": False,
            "error": "Symbol not found on NSE"
        }
        
        # Initialize generator
        generator = StockReportGenerator(
            openai_api_key=mock_openai_api_key,
            skip_pdf=True
        )
        
        # Attempt to generate report
        results = await generator.generate_report(
            stock_symbol="INVALID"
        )
        
        # Assertions
        assert results["workflow_status"] == "failed"
        assert "error" in results
        assert "Symbol not found" in results["error"] or "not found" in results["error"].lower()
        
        # Verify orchestrator was not called
        mock_orchestrator_class.assert_called_once()
        mock_orchestrator = mock_orchestrator_class.return_value
        mock_orchestrator.run_workflow.assert_not_called()
    
    @pytest.mark.asyncio
    @patch('main.MultiAgentOrchestrator')
    @patch('main.tool_validate_symbol')
    @patch('main.tool_get_company_info')
    async def test_workflow_orchestrator_error(
        self,
        mock_get_company_info,
        mock_validate_symbol,
        mock_orchestrator_class,
        mock_openai_api_key,
        mock_company_info
    ):
        """Test workflow when orchestrator raises an error."""
        # Setup mocks
        mock_validate_symbol.invoke.return_value = {
            "valid": True,
            "company_name": "Test Company Limited",
            "sector": "Technology"
        }
        mock_get_company_info.invoke.return_value = mock_company_info
        
        # Mock orchestrator to raise error
        mock_orchestrator = MagicMock()
        mock_orchestrator.run_workflow = AsyncMock(side_effect=Exception("Orchestrator error"))
        mock_orchestrator_class.return_value = mock_orchestrator
        
        # Initialize generator
        generator = StockReportGenerator(
            openai_api_key=mock_openai_api_key,
            skip_pdf=True
        )
        
        # Generate report
        results = await generator.generate_report(
            stock_symbol="TESTCOMP"
        )
        
        # Assertions
        assert results["workflow_status"] == "failed"
        assert "error" in results
        assert "Orchestrator error" in results["error"]
        assert "session_id" in results
    
    @pytest.mark.asyncio
    @patch('main.MultiAgentOrchestrator')
    @patch('main.tool_validate_symbol')
    @patch('main.tool_get_company_info')
    async def test_workflow_with_partial_errors(
        self,
        mock_get_company_info,
        mock_validate_symbol,
        mock_orchestrator_class,
        mock_openai_api_key,
        mock_company_info
    ):
        """Test workflow that completes but with some errors."""
        # Setup mocks
        mock_validate_symbol.invoke.return_value = {
            "valid": True,
            "company_name": "Test Company Limited",
            "sector": "Technology"
        }
        mock_get_company_info.invoke.return_value = mock_company_info
        
        # Mock orchestrator with partial errors
        mock_orchestrator = MagicMock()
        mock_orchestrator.run_workflow = AsyncMock(return_value={
            "workflow_status": "completed",
            "stock_symbol": "TESTCOMP",
            "company_name": "Test Company Limited",
            "sector": "Technology",
            "final_report": "# Stock Report\n\n## Executive Summary\n\nTest report content.",
            "pdf_path": None,  # PDF generation failed
            "duration_seconds": 30.5,
            "errors": ["PDF generation failed: Permission denied"]
        })
        mock_orchestrator_class.return_value = mock_orchestrator
        
        # Initialize generator
        generator = StockReportGenerator(
            openai_api_key=mock_openai_api_key,
            skip_pdf=True
        )
        
        # Generate report
        results = await generator.generate_report(
            stock_symbol="TESTCOMP"
        )
        
        # Assertions
        assert results["workflow_status"] == "completed"
        assert "final_report" in results
        assert len(results.get("errors", [])) > 0
        assert "PDF generation failed" in results["errors"][0]
    
    @pytest.mark.asyncio
    @patch('main.MultiAgentOrchestrator')
    @patch('main.tool_validate_symbol')
    @patch('main.tool_get_company_info')
    async def test_workflow_auto_populate_company_info(
        self,
        mock_get_company_info,
        mock_validate_symbol,
        mock_orchestrator_class,
        mock_openai_api_key,
        mock_company_info
    ):
        """Test workflow auto-populates company name and sector when not provided."""
        # Setup mocks
        mock_validate_symbol.invoke.return_value = {
            "valid": True,
            "company_name": "Test Company Limited",
            "sector": "Technology"
        }
        mock_get_company_info.invoke.return_value = mock_company_info
        
        # Mock orchestrator
        mock_orchestrator = MagicMock()
        mock_orchestrator.run_workflow = AsyncMock(return_value={
            "workflow_status": "completed",
            "stock_symbol": "TESTCOMP",
            "company_name": "Test Company Limited",
            "sector": "Technology",
            "final_report": "# Stock Report\n\nTest content.",
            "duration_seconds": 25.0,
            "errors": []
        })
        mock_orchestrator_class.return_value = mock_orchestrator
        
        # Initialize generator
        generator = StockReportGenerator(
            openai_api_key=mock_openai_api_key,
            skip_pdf=True
        )
        
        # Generate report without company_name and sector
        results = await generator.generate_report(
            stock_symbol="TESTCOMP"
        )
        
        # Assertions
        assert results["workflow_status"] == "completed"
        assert results["company_name"] == "Test Company Limited"
        assert results["sector"] == "Technology"
        
        # Verify tool was called to get company info
        mock_get_company_info.invoke.assert_called_once()
        call_args = mock_get_company_info.invoke.call_args
        # invoke is called with a dict as first positional argument: invoke({"symbol": "TESTCOMP"})
        assert call_args[0][0]["symbol"] == "TESTCOMP"
    
    @pytest.mark.asyncio
    @patch('main.MultiAgentOrchestrator')
    @patch('main.tool_validate_symbol')
    async def test_workflow_symbol_normalization(
        self,
        mock_validate_symbol,
        mock_orchestrator_class,
        mock_openai_api_key
    ):
        """Test workflow normalizes stock symbol (removes .NS suffix)."""
        # Setup mocks
        mock_validate_symbol.invoke.return_value = {
            "valid": True,
            "company_name": "Test Company Limited",
            "sector": "Technology"
        }
        
        # Mock orchestrator
        mock_orchestrator = MagicMock()
        mock_orchestrator.run_workflow = AsyncMock(return_value={
            "workflow_status": "completed",
            "stock_symbol": "TESTCOMP",
            "final_report": "# Stock Report\n\nTest content.",
            "duration_seconds": 20.0,
            "errors": []
        })
        mock_orchestrator_class.return_value = mock_orchestrator
        
        # Initialize generator
        generator = StockReportGenerator(
            openai_api_key=mock_openai_api_key,
            skip_pdf=True
        )
        
        # Generate report with .NS suffix
        results = await generator.generate_report(
            stock_symbol="TESTCOMP.NS"
        )
        
        # Assertions
        assert results["workflow_status"] == "completed"
        
        # Verify validation was called with normalized symbol
        mock_validate_symbol.invoke.assert_called_once()
        call_args = mock_validate_symbol.invoke.call_args
        # invoke is called with a dict as first positional argument: invoke({"symbol": "TESTCOMP"})
        assert call_args[0][0]["symbol"] == "TESTCOMP"  # .NS should be removed
    
    @patch('main.MultiAgentOrchestrator')
    @patch('main.tool_validate_symbol')
    @patch('main.tool_get_company_info')
    def test_generate_report_sync(
        self,
        mock_get_company_info,
        mock_validate_symbol,
        mock_orchestrator_class,
        mock_openai_api_key,
        mock_company_info
    ):
        """Test synchronous wrapper for generate_report."""
        # Setup mocks
        mock_validate_symbol.invoke.return_value = {
            "valid": True,
            "company_name": "Test Company Limited",
            "sector": "Technology"
        }
        mock_get_company_info.invoke.return_value = mock_company_info
        
        # Mock orchestrator
        mock_orchestrator = MagicMock()
        # Create a future that will be resolved by asyncio.run
        async def mock_workflow(*args, **kwargs):
            return {
                "workflow_status": "completed",
                "stock_symbol": "TESTCOMP",
                "company_name": "Test Company Limited",
                "sector": "Technology",
                "final_report": "# Test Report",
                "duration_seconds": 20.0,
                "errors": []
            }
        mock_orchestrator.run_workflow = AsyncMock(side_effect=mock_workflow)
        mock_orchestrator_class.return_value = mock_orchestrator
        
        # Initialize generator
        generator = StockReportGenerator(
            openai_api_key=mock_openai_api_key,
            skip_pdf=True
        )
        
        # Test sync wrapper - use asyncio.run which will work in non-async context
        results = generator.generate_report_sync(
            stock_symbol="TESTCOMP"
        )
        
        # Assertions
        assert results["workflow_status"] == "completed"
        assert results["stock_symbol"] == "TESTCOMP"
        assert "final_report" in results

