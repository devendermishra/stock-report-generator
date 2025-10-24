"""
Main entry point for Stock Report Generator.
Provides a command-line interface for generating equity research reports.
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_report_generator.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class StockReportGenerator:
    """
    Main class for the Stock Report Generator system.
    
    Orchestrates the multi-agent workflow to generate comprehensive
    equity research reports for NSE stocks.
    """
    
    def __init__(self):
        """Initialize the Stock Report Generator."""
        self.config = Config()
        self.mcp_context = None
        self.workflow_graph = None
        self.tools = {}
        
        # Initialize the system
        if not self.initialize():
            raise RuntimeError("Failed to initialize Stock Report Generator")
        
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
            
            # Stock Data Tool
            self.tools['stock_data'] = StockDataTool()
            
            # Report Fetcher Tool
            self.tools['report_fetcher'] = ReportFetcherTool(
                download_dir=self.config.TEMP_DIR
            )
            
            # PDF Parser Tool
            self.tools['pdf_parser'] = PDFParserTool(
                chunk_size=1000,
                chunk_overlap=200
            )
            
            # Summarizer Tool
            self.tools['summarizer'] = SummarizerTool(
                api_key=self.config.OPENAI_API_KEY,
                model=self.config.DEFAULT_MODEL
            )
            
            # Report Formatter Tool
            self.tools['report_formatter'] = ReportFormatterTool(
                output_dir=self.config.OUTPUT_DIR
            )
            
            # PDF Generator Tool
            self.tools['pdf_generator'] = PDFGeneratorTool(
                output_dir=self.config.OUTPUT_DIR
            )
            
            logger.info("All tools initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing tools: {e}")
            raise
            
    def _initialize_workflow(self) -> None:
        """Initialize the workflow graph."""
        try:
            self.workflow_graph = StockReportGraph(
                mcp_context=self.mcp_context,
                web_search_tool=self.tools['web_search'],
                stock_data_tool=self.tools['stock_data'],
                report_fetcher_tool=self.tools['report_fetcher'],
                pdf_parser_tool=self.tools['pdf_parser'],
                summarizer_tool=self.tools['summarizer'],
                report_formatter_tool=self.tools['report_formatter'],
                openai_api_key=self.config.OPENAI_API_KEY
            )
            
            logger.info("Workflow graph initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing workflow: {e}")
            raise
            
    async def generate_report(self, stock_symbol: str) -> Dict[str, Any]:
        """
        Generate a comprehensive stock report.
        
        Args:
            stock_symbol: NSE stock symbol (e.g., 'RELIANCE')
            
        Returns:
            Dictionary containing the generated report and metadata
        """
        try:
            logger.info(f"Starting report generation for {stock_symbol}")
            
            # Validate symbol exists
            logger.info(f"Validating symbol {stock_symbol}...")
            if not self.tools['stock_data'].validate_symbol_with_yahoo(stock_symbol):
                return {
                    "success": False,
                    "error": f"Stock symbol '{stock_symbol}' not found. Please check the symbol and try again.",
                    "stock_symbol": stock_symbol
                }
            
            # Get company name and sector automatically
            logger.info(f"Fetching company information for {stock_symbol}")
            company_info = self.tools['stock_data'].get_company_name_and_sector(stock_symbol)
            
            company_name = company_info.company_name
            sector = company_info.sector
            
            logger.info(f"Found: {company_name} in {sector} sector")
                
            # Run the workflow
            workflow_results = await self.workflow_graph.run_workflow(
                stock_symbol=stock_symbol,
                company_name=company_name,
                sector=sector
            )
            
            # Check if workflow completed successfully
            if workflow_results.get("workflow_status") == "completed":
                # Save the report
                report_path = self._save_report(workflow_results)
                
                return {
                    "success": True,
                    "stock_symbol": stock_symbol,
                    "company_name": company_name,
                    "sector": sector,
                    "report_path": report_path,
                    "workflow_results": workflow_results,
                    "generated_at": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": "Workflow failed to complete",
                    "workflow_results": workflow_results,
                    "stock_symbol": stock_symbol
                }
                
        except Exception as e:
            logger.error(f"Error generating report for {stock_symbol}: {e}")
            return {
                "success": False,
                "error": str(e),
                "stock_symbol": stock_symbol
            }
            
    def _validate_symbol(self, stock_symbol: str) -> bool:
        """Validate stock symbol parameter."""
        if not stock_symbol or not stock_symbol.strip():
            logger.error("Stock symbol is required")
            return False
            
        # Validate stock symbol format
        if not stock_symbol.isupper() or len(stock_symbol) < 2:
            logger.warning(f"Stock symbol '{stock_symbol}' may not be in correct format")
            
        return True
        
    def _save_report(self, workflow_results: Dict[str, Any]) -> Optional[str]:
        """Save the generated report to markdown and PDF files."""
        try:
            final_report = workflow_results.get("final_report")
            if not final_report:
                logger.warning("No final report found in workflow results")
                return None
                
            # Create filename
            stock_symbol = workflow_results.get("stock_symbol", "UNKNOWN")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stock_report_{stock_symbol}_{timestamp}.md"
            
            # Save markdown report
            report_path = os.path.join(self.config.OUTPUT_DIR, filename)
            os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
            
            report_content = final_report.get("report_content", "")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
                
            logger.info(f"Markdown report saved to {report_path}")
            
            # Generate PDF version
            try:
                pdf_path = self.tools['pdf_generator'].generate_pdf(
                    markdown_content=report_content,
                    stock_symbol=stock_symbol
                )
                logger.info(f"PDF report generated: {pdf_path}")
            except Exception as e:
                logger.error(f"Failed to generate PDF: {e}")
                # Continue with markdown report even if PDF generation fails
            
            return report_path
            
        except Exception as e:
            logger.error(f"Error saving report: {e}")
            return None
            
    def get_report_status(self, stock_symbol: str) -> Dict[str, Any]:
        """
        Get the status of a report generation process.
        
        Args:
            stock_symbol: Stock symbol to check status for
            
        Returns:
            Dictionary containing status information
        """
        try:
            if not self.workflow_graph:
                return {
                    "success": False,
                    "error": "Workflow not initialized"
                }
                
            status = self.workflow_graph.get_workflow_status(stock_symbol)
            return {
                "success": True,
                "status": status
            }
            
        except Exception as e:
            logger.error(f"Error getting report status: {e}")
            return {
                "success": False,
                "error": str(e)
            }
            
    def export_workflow_data(self, stock_symbol: str) -> Dict[str, Any]:
        """
        Export workflow data for a stock symbol.
        
        Args:
            stock_symbol: Stock symbol to export data for
            
        Returns:
            Dictionary containing exported data
        """
        try:
            if not self.workflow_graph:
                return {
                    "success": False,
                    "error": "Workflow not initialized"
                }
                
            export_data = self.workflow_graph.export_workflow_data(stock_symbol)
            return {
                "success": True,
                "export_data": export_data
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
        description="Stock Report Generator for NSE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --symbol RELIANCE
  python main.py --symbol TCS
  python main.py --symbol HDFCBANK
  python main.py --symbol INFY --output-dir my_reports
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
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Update output directory in config
    Config.OUTPUT_DIR = args.output_dir
    
    # Initialize the generator
    generator = StockReportGenerator()
    
    if not generator.initialize():
        logger.error("Failed to initialize Stock Report Generator")
        sys.exit(1)
        
    try:
        # Generate the report
        logger.info(f"Generating report for {args.symbol}")
        result = await generator.generate_report(
            stock_symbol=args.symbol.upper()
        )
        
        if result["success"]:
            print(f"\n✅ Report generated successfully!")
            print(f"📊 Stock: {result['stock_symbol']}")
            print(f"🏢 Company: {result.get('company_name', 'N/A')}")
            print(f"🏭 Sector: {result.get('sector', 'N/A')}")
            print(f"📄 Markdown report: {result['report_path']}")
            print(f"📋 PDF report: {result['report_path'].replace('.md', '.pdf')}")
            print(f"⏱️  Generated at: {result['generated_at']}")
            
            # Export data if requested
            if args.export_data:
                export_result = generator.export_workflow_data(args.symbol.upper())
                if export_result["success"]:
                    export_filename = f"workflow_data_{args.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    export_path = os.path.join(args.output_dir, export_filename)
                    
                    with open(export_path, 'w', encoding='utf-8') as f:
                        json.dump(export_result["export_data"], f, indent=2, default=str)
                        
                    print(f"📊 Workflow data exported to: {export_path}")
                else:
                    print(f"⚠️  Failed to export workflow data: {export_result['error']}")
                    
        else:
            print(f"\n❌ Report generation failed!")
            print(f"🔍 Error: {result.get('error', 'Unknown error')}")
            if 'workflow_results' in result:
                print(f"📋 Workflow status: {result['workflow_results'].get('workflow_status', 'Unknown')}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Report generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
