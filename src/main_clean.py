"""
Main entry point for Stock Report Generator.
Provides a command-line interface for generating equity research reports using LangGraph.

Copyright (c) 2025 Stock Report Generator. All rights reserved.
"""

import asyncio
import logging
import argparse
import sys
from typing import Dict, Any, Optional
from datetime import datetime
import json
import os

try:
    # Try relative imports first (when run as module)
    from .config import Config
    from .graph.context_manager_mcp import MCPContextManager
    from .graph.stock_report_graph import StockReportGraph
    from .tools.web_search_tool import WebSearchTool
    from .tools.stock_data_tool import StockDataTool
    from .tools.report_fetcher_tool import ReportFetcherTool
    from .tools.pdf_parser_tool import PDFParserTool
    from .tools.summarizer_tool import SummarizerTool
    from .tools.report_formatter_tool import ReportFormatterTool
    from .tools.pdf_generator_tool import PDFGeneratorTool
    # LangGraph imports
    from .agents.langgraph_dynamic_agent_simple import LangGraphDynamicAgentSimple as LangGraphDynamicAgent

except ImportError:
    # Fall back to absolute imports (when run as script)
    from config import Config
    from graph.context_manager_mcp import MCPContextManager
    from graph.stock_report_graph import StockReportGraph
    from tools.web_search_tool import WebSearchTool
    from tools.stock_data_tool import StockDataTool
    from tools.report_fetcher_tool import ReportFetcherTool
    from tools.pdf_parser_tool import PDFParserTool
    from tools.summarizer_tool import SummarizerTool
    from tools.report_formatter_tool import ReportFormatterTool
    from tools.pdf_generator_tool import PDFGeneratorTool
    # LangGraph imports
    from agents.langgraph_dynamic_agent_simple import LangGraphDynamicAgentSimple as LangGraphDynamicAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('stock_report_generator.log')
    ]
)

logger = logging.getLogger(__name__)

class StockReportGenerator:
    """
    Main class for generating comprehensive stock reports using LangGraph.
    """
    
    def __init__(self):
        """Initialize the Stock Report Generator."""
        self.config = Config()
        self.mcp_context = None
        self.workflow_graph = None
        self.tools = {}
        self.langgraph_agent = None

    def initialize(self) -> bool:
        """
        Initialize the system components.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing Stock Report Generator...")
            
            # Validate configuration
            config_validation = self.config.validate_config()
            if not all(config_validation.values()):
                logger.error("Configuration validation failed:")
                for key, value in config_validation.items():
                    if not value:
                        logger.error(f"  - {key}: Missing or invalid")
                return False
                
            # Initialize MCP context
            self.mcp_context = MCPContextManager(
                max_context_size=self.config.MCP_CONTEXT_SIZE
            )
            
            # Initialize tools
            self._initialize_tools()
            
            # Initialize workflow graph
            self._initialize_workflow()
            
            # Initialize LangGraph agent system
            self._initialize_langgraph_agent_system()
            
            logger.info("Stock Report Generator initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Stock Report Generator: {e}")
            return False
            
    def _initialize_tools(self) -> None:
        """Initialize all tools."""
        try:
            # Web Search Tool (DuckDuckGo - no API key required)
            self.tools['web_search'] = WebSearchTool(
                max_results=10
            )
            
            # Stock Data Tool (Yahoo Finance - no API key required)
            self.tools['stock_data'] = StockDataTool()
            
            # Report Fetcher Tool
            self.tools['report_fetcher'] = ReportFetcherTool()
            
            # PDF Parser Tool
            self.tools['pdf_parser'] = PDFParserTool()
            
            # Summarizer Tool
            self.tools['summarizer'] = SummarizerTool(
                openai_api_key=self.config.OPENAI_API_KEY,
                model=self.config.DEFAULT_MODEL
            )
            
            # Report Formatter Tool
            self.tools['report_formatter'] = ReportFormatterTool()
            
            # PDF Generator Tool
            self.tools['pdf_generator'] = PDFGeneratorTool()
            
            logger.info("All tools initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing tools: {e}")
            raise
    
    def _initialize_workflow(self) -> None:
        """Initialize the workflow graph."""
        try:
            self.workflow_graph = StockReportGraph(
                mcp_context=self.mcp_context,
                tools=self.tools
            )
            logger.info("Workflow graph initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing workflow graph: {e}")
            raise
    
    def _initialize_langgraph_agent_system(self) -> None:
        """Initialize the LangGraph agent system."""
        try:
            # Initialize LangGraph agent
            self.langgraph_agent = LangGraphDynamicAgent(
                agent_id="langgraph_stock_analyzer",
                mcp_context=self.mcp_context,
                stock_data_tool=self.tools['stock_data'],
                web_search_tool=self.tools['web_search'],
                summarizer_tool=self.tools['summarizer'],
                openai_api_key=self.config.OPENAI_API_KEY,
                model=self.config.DEFAULT_MODEL
            )
            
            logger.info("LangGraph agent system initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing LangGraph agent system: {e}")
            # Don't raise - LangGraph system is optional
            self.langgraph_agent = None
            
    async def generate_report(self, stock_symbol: str) -> Dict[str, Any]:
        """
        Generate a comprehensive stock report using traditional workflow.
        
        Args:
            stock_symbol: NSE stock symbol (e.g., 'RELIANCE')
            
        Returns:
            Dictionary containing the generated report and metadata
        """
        try:
            logger.info(f"Starting traditional report generation for {stock_symbol}")
            
            # Execute the workflow
            result = await self.workflow_graph.execute_workflow(stock_symbol)
            
            if result.get("success", False):
                logger.info(f"Traditional report generated successfully for {stock_symbol}")
                return {
                    "success": True,
                    "stock_symbol": stock_symbol,
                    "report_type": "traditional",
                    "generated_at": datetime.now().isoformat(),
                    **result
                }
            else:
                logger.error(f"Traditional report generation failed for {stock_symbol}")
                return {
                    "success": False,
                    "error": result.get("error", "Unknown error"),
                    "stock_symbol": stock_symbol
                }
                
        except Exception as e:
            logger.error(f"Error generating traditional report for {stock_symbol}: {e}")
            return {
                "success": False,
                "error": str(e),
                "stock_symbol": stock_symbol
            }
    
    async def generate_langgraph_report(self, stock_symbol: str) -> Dict[str, Any]:
        """
        Generate a stock report using the LangGraph agent system.
        
        Args:
            stock_symbol: NSE stock symbol (e.g., 'RELIANCE')
            
        Returns:
            Dictionary containing the generated report and metadata
        """
        try:
            if not self.langgraph_agent:
                return {
                    "success": False,
                    "error": "LangGraph agent system not initialized",
                    "stock_symbol": stock_symbol
                }
            
            logger.info(f"Starting LangGraph report generation for {stock_symbol}")
            
            # Get company name and sector automatically
            logger.info(f"Fetching company information for {stock_symbol}")
            try:
                company_info = self.tools['stock_data'].get_company_name_and_sector(stock_symbol)
                company_name = company_info.company_name
                sector = company_info.sector
            except Exception as e:
                logger.warning(f"Could not get company info: {e}")
                company_name = stock_symbol
                sector = "Unknown"
            
            logger.info(f"Found: {company_name} in {sector} sector")
            
            # Perform LangGraph analysis
            analysis_result = await self.langgraph_agent.analyze_stock(stock_symbol)
            
            if not analysis_result.get("success", False):
                return {
                    "success": False,
                    "error": f"LangGraph analysis failed: {analysis_result.get('error', 'Unknown error')}",
                    "stock_symbol": stock_symbol
                }
            
            # Generate recommendations
            recommendations_result = await self.langgraph_agent.generate_recommendations(stock_symbol)
            
            if not recommendations_result.get("success", False):
                logger.warning(f"LangGraph recommendations failed: {recommendations_result.get('error', 'Unknown error')}")
                # Continue with analysis results only
            
            # Format the report
            report_data = {
                "stock_symbol": stock_symbol,
                "company_name": company_name,
                "sector": sector,
                "analysis": analysis_result.get("analysis", ""),
                "recommendations": recommendations_result.get("recommendations", ""),
                "agent_id": self.langgraph_agent.agent_id,
                "report_type": "langgraph",
                "generated_at": datetime.now().isoformat()
            }
            
            # Generate markdown report
            markdown_content = self._format_langgraph_report(report_data)
            
            # Save markdown report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            markdown_filename = f"reports/langgraph_stock_report_{stock_symbol}_{timestamp}.md"
            os.makedirs(os.path.dirname(markdown_filename), exist_ok=True)
            
            with open(markdown_filename, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            logger.info(f"LangGraph report saved to {markdown_filename}")
            
            # Generate PDF report
            pdf_filename = f"reports/stock_report_{stock_symbol}_{timestamp}.pdf"
            pdf_result = await self.tools['pdf_generator'].generate_pdf(
                markdown_content=markdown_content,
                output_path=pdf_filename,
                title=f"LangGraph Stock Analysis Report - {stock_symbol}"
            )
            
            if pdf_result.get("success", False):
                logger.info(f"LangGraph PDF report generated: {pdf_filename}")
            else:
                logger.warning(f"Failed to generate PDF: {pdf_result.get('error', 'Unknown error')}")
            
            return {
                "success": True,
                "stock_symbol": stock_symbol,
                "company_name": company_name,
                "sector": sector,
                "report_type": "langgraph",
                "markdown_file": markdown_filename,
                "pdf_file": pdf_filename if pdf_result.get("success", False) else None,
                "analysis": analysis_result.get("analysis", ""),
                "recommendations": recommendations_result.get("recommendations", ""),
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating LangGraph report for {stock_symbol}: {e}")
            return {
                "success": False,
                "error": str(e),
                "stock_symbol": stock_symbol
            }
    
    def _format_langgraph_report(self, report_data: Dict[str, Any]) -> str:
        """Format LangGraph report data into markdown."""
        return f"""# LangGraph Stock Analysis Report

## 📊 Stock Information
- **Symbol**: {report_data['stock_symbol']}
- **Company**: {report_data['company_name']}
- **Sector**: {report_data['sector']}
- **Analysis Type**: LangGraph LLM-Driven Analysis
- **Generated**: {report_data['generated_at']}

## 🔍 Analysis
{report_data['analysis']}

## 🎯 Recommendations
{report_data['recommendations']}

---
*Report generated by LangGraph Stock Analysis System*
"""
    
    def export_workflow_data(self, stock_symbol: str) -> Dict[str, Any]:
        """
        Export workflow data for analysis.
        
        Args:
            stock_symbol: Stock symbol
            
        Returns:
            Export result
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_filename = f"exports/workflow_data_{stock_symbol}_{timestamp}.json"
            
            # Create exports directory if it doesn't exist
            os.makedirs(os.path.dirname(export_filename), exist_ok=True)
            
            # Get workflow data
            workflow_data = {
                "stock_symbol": stock_symbol,
                "timestamp": timestamp,
                "config": {
                    "openai_api_key": "***REDACTED***",
                    "default_model": self.config.DEFAULT_MODEL,
                    "mcp_context_size": self.config.MCP_CONTEXT_SIZE
                },
                "tools": list(self.tools.keys()),
                "langgraph_available": self.langgraph_agent is not None
            }
            
            with open(export_filename, 'w', encoding='utf-8') as f:
                json.dump(workflow_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Workflow data exported to {export_filename}")
            
            return {
                "success": True,
                "export_file": export_filename,
                "data": workflow_data
            }
            
        except Exception as e:
            logger.error(f"Error exporting workflow data: {e}")
            return {
                "success": False,
                "error": str(e)
            }

async def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Stock Report Generator for NSE using LangGraph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --symbol RELIANCE
  python main.py --symbol TCS
  python main.py --symbol HDFCBANK
  python main.py --symbol INFY --output-dir my_reports
  python main.py --symbol RELIANCE --langgraph
  python main.py --symbol TCS --langgraph --verbose
        """
    )
    
    parser.add_argument(
        "--symbol",
        required=True,
        help="NSE stock symbol (e.g., RELIANCE, TCS, HDFCBANK) - company name and sector will be fetched automatically"
    )
    
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Output directory for generated reports (default: reports)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--export-data",
        action="store_true",
        help="Export workflow data after generation"
    )

    parser.add_argument(
        "--langgraph",
        action="store_true",
        help="Use LangGraph framework for intelligent tool execution (default)"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create generator instance
    generator = StockReportGenerator()
    
    if not generator.initialize():
        logger.error("Failed to initialize Stock Report Generator")
        sys.exit(1)
        
    try:
        # Generate the report
        if args.langgraph:
            logger.info(f"Generating LangGraph report for {args.symbol}")
            result = await generator.generate_langgraph_report(
                stock_symbol=args.symbol.upper()
            )
        else:
            logger.info(f"Generating traditional report for {args.symbol}")
            result = await generator.generate_report(
                stock_symbol=args.symbol.upper()
            )
        
        if result["success"]:
            report_type = result.get('report_type', 'traditional')
            print(f"\n✅ Report generated successfully!")
            print(f"📊 Stock: {result['stock_symbol']}")
            print(f"🏢 Company: {result.get('company_name', 'N/A')}")
            print(f"🏭 Sector: {result.get('sector', 'N/A')}")
            if report_type == 'langgraph':
                print(f"🤖 Report Type: LangGraph LLM-Driven")
            else:
                print(f"🤖 Report Type: Traditional Fixed-Sequence")
            
            # Show file paths
            if 'markdown_file' in result:
                print(f"📄 Markdown report: {result['markdown_file']}")
            if 'pdf_file' in result and result['pdf_file']:
                print(f"📋 PDF report: {result['pdf_file']}")
            if 'report_path' in result:
                print(f"📄 Report: {result['report_path']}")
            
            print(f"⏱️  Generated at: {result['generated_at']}")
            
            # Show additional info for LangGraph reports
            if report_type == 'langgraph':
                print(f"🎯 Investment Rating: N/A")
                print(f"🎯 Confidence Score: N/A")
                print(f"📊 Market Sentiment: N/A")
            
            # Export data if requested
            if args.export_data:
                export_result = generator.export_workflow_data(args.symbol.upper())
                if export_result.get("success", False):
                    print(f"📊 Workflow data exported: {export_result['export_file']}")
                else:
                    print(f"❌ Failed to export workflow data: {export_result.get('error', 'Unknown error')}")
            
        else:
            print(f"\n❌ Report generation failed!")
            print(f"📊 Stock: {result['stock_symbol']}")
            print(f"🚨 Error: {result.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Report generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

