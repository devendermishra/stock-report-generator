"""
Main entry point for the Agentic Stock Research Report Generator.
Autonomous Multi-Agent System using LangGraph + LangChain.

This system uses three autonomous agents that collaborate to generate
comprehensive stock research reports for NSE stocks.
"""

import asyncio
import logging
import sys
import os
import argparse
from typing import Dict, Any, Optional

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Try relative imports first (when run as module)
    from .config import Config
    from .graph.multi_agent_graph import MultiAgentOrchestrator
    from .tools.stock_data_tool import validate_symbol as tool_validate_symbol
    from .tools.stock_data_tool import get_company_info as tool_get_company_info
except ImportError:
    # Fall back to absolute imports (when run as script)
    from config import Config
    from graph.multi_agent_graph import MultiAgentOrchestrator
    from tools.stock_data_tool import validate_symbol as tool_validate_symbol
    from tools.stock_data_tool import get_company_info as tool_get_company_info

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
    Main class for the Agentic Stock Research Report Generator.
    
    This system uses three autonomous agents:
    1. ResearchAgent - Gathers company information, sector overview, and peer data
    2. AnalysisAgent - Performs financial, management, and technical analysis  
    3. ReportAgent - Synthesizes all data into comprehensive reports
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the Stock Report Generator.
        
        Args:
            openai_api_key: OpenAI API key (if not provided, will use config)
        """
        self.openai_api_key = openai_api_key or Config.OPENAI_API_KEY
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it directly.")
        
        # Initialize the multi-agent orchestrator
        self.orchestrator = MultiAgentOrchestrator(self.openai_api_key)
        
        logger.info("Stock Report Generator initialized successfully")
    
    async def generate_report(
        self,
        stock_symbol: str,
        company_name: Optional[str] = None,
        sector: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive stock research report.
        
        Args:
            stock_symbol: NSE stock symbol (e.g., 'RELIANCE', 'TCS')
            company_name: Full company name (optional, will be fetched if not provided)
            sector: Sector name (optional, will be fetched if not provided)
            
        Returns:
            Dictionary containing the generated report and metadata
        """
        try:
            logger.info(f"Starting report generation for {stock_symbol}")
            
            # Validate inputs
            if not stock_symbol:
                raise ValueError("Stock symbol is required")
            
            # Clean stock symbol (remove .NS suffix if present)
            if stock_symbol.endswith('.NS'):
                stock_symbol = stock_symbol[:-3]
            
            # Validate symbol against NSE via tool
            validation = tool_validate_symbol.invoke({"symbol": stock_symbol})
            if not validation or not validation.get("valid", False):
                raise ValueError(validation.get("error", "Symbol not found on NSE"))
            # Fetch company info from tool to populate missing fields
            info = tool_get_company_info.invoke({"symbol": stock_symbol}) or {}
            if not company_name:
                company_name = info.get("company_name") or info.get("short_name") or validation.get("company_name") or f"Company {stock_symbol}"
            if not sector:
                sector = info.get("sector") or validation.get("sector") or "Unknown"
            
            # Run the multi-agent workflow
            results = await self.orchestrator.run_workflow(
                stock_symbol=stock_symbol,
                company_name=company_name,
                sector=sector
            )
            
            # Log results
            if results["workflow_status"] == "completed":
                logger.info(f"Successfully generated report for {stock_symbol}")
                if results.get("pdf_path"):
                    logger.info(f"PDF report saved to: {results['pdf_path']}")
            else:
                logger.warning(f"Report generation completed with errors for {stock_symbol}")
                if results.get("errors"):
                    logger.warning(f"Errors: {results['errors']}")
            
            return results
            
        except Exception as e:
            logger.error(f"Report generation failed for {stock_symbol}: {e}")
            return {
                "stock_symbol": stock_symbol,
                "workflow_status": "failed",
                "error": str(e),
                "errors": [str(e)]
            }
    
    def generate_report_sync(
        self,
        stock_symbol: str,
        company_name: Optional[str] = None,
        sector: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for generate_report.
        
        Args:
            stock_symbol: NSE stock symbol
            company_name: Full company name (optional)
            sector: Sector name (optional)
            
        Returns:
            Dictionary containing the generated report and metadata
        """
        return asyncio.run(self.generate_report(stock_symbol, company_name, sector))
    
    # Deprecated hard-coded helpers removed in favor of live NSE validation via tools

def export_graph_diagram(orchestrator, output_path: str) -> bool:
    """
    Export the MultiAgentOrchestrator graph diagram to an image file.
    
    Args:
        orchestrator: MultiAgentOrchestrator instance
        output_path: Path where the graph image should be saved
        
    Returns:
        True if successful, False otherwise
    """
    try:
        graph = orchestrator.graph.get_graph()
        
        # Try to export as PNG using graphviz
        try:
            png_data = graph.draw_png()
            if png_data:
                with open(output_path, 'wb') as f:
                    f.write(png_data)
                logger.info(f"Graph diagram exported to {output_path}")
                print(f"✅ Graph diagram exported to {output_path}")
                return True
        except Exception as e:
            logger.warning(f"Failed to export as PNG (graphviz may not be installed): {e}")
        
        # Fallback to Mermaid format
        try:
            mermaid_diagram = graph.draw_mermaid()
            if mermaid_diagram:
                # Save as .mmd file (mermaid format)
                if '.' in output_path:
                    mmd_path = output_path.rsplit('.', 1)[0] + '.mmd'
                else:
                    mmd_path = output_path + '.mmd'
                with open(mmd_path, 'w') as f:
                    f.write(mermaid_diagram)
                logger.info(f"Graph diagram exported as Mermaid format to {mmd_path}")
                print(f"✅ Graph diagram exported as Mermaid format to {mmd_path}")
                print(f"   (Install graphviz to export as PNG. Mermaid can be viewed at https://mermaid.live/)")
                return True
        except Exception as e:
            logger.warning(f"Failed to export as Mermaid: {e}")
        
        # Last resort: ASCII representation
        try:
            ascii_diagram = graph.draw_ascii()
            if ascii_diagram:
                if '.' in output_path:
                    txt_path = output_path.rsplit('.', 1)[0] + '.txt'
                else:
                    txt_path = output_path + '.txt'
                with open(txt_path, 'w') as f:
                    f.write(ascii_diagram)
                logger.info(f"Graph diagram exported as ASCII to {txt_path}")
                print(f"✅ Graph diagram exported as ASCII to {txt_path}")
                return True
        except Exception as e:
            logger.error(f"Failed to export graph diagram: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Error exporting graph diagram: {e}")
        print(f"❌ Failed to export graph diagram: {e}")
        return False

async def main():
    """Main function for command-line usage."""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(
            description='Generate stock research reports using multi-agent AI system',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python main.py RELIANCE
  python main.py RELIANCE "Reliance Industries" "Oil & Gas"
  python main.py RELIANCE --export-graph graph.png
  python main.py --export-graph-only graph.png
            """
        )
        parser.add_argument('stock_symbol', nargs='?', help='NSE stock symbol (e.g., RELIANCE, TCS)')
        parser.add_argument('company_name', nargs='?', help='Full company name (optional, will be fetched if not provided)')
        parser.add_argument('sector', nargs='?', help='Sector name (optional, will be fetched if not provided)')
        parser.add_argument('--export-graph', '--graph-output', dest='graph_output', 
                          help='Export the multi-agent graph diagram to the specified file path (e.g., graph.png)')
        parser.add_argument('--export-graph-only', dest='export_graph_only',
                          help='Only export the graph diagram without generating a report (specify output path)')
        
        args = parser.parse_args()
        
        # Check configuration
        config_validation = Config.validate_config()
        if not config_validation["openai_key"]:
            print("Error: OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
            sys.exit(1)
        
        # Initialize the generator
        generator = StockReportGenerator()
        
        # Handle graph export only mode
        if args.export_graph_only:
            print(f"Exporting graph diagram to {args.export_graph_only}...")
            success = export_graph_diagram(generator.orchestrator, args.export_graph_only)
            sys.exit(0 if success else 1)
        
        # Check if stock symbol is provided
        if not args.stock_symbol:
            parser.print_help()
            sys.exit(1)
        
        stock_symbol = args.stock_symbol.upper()
        company_name = args.company_name
        sector = args.sector
        
        # Export graph if requested
        if args.graph_output:
            print(f"Exporting graph diagram to {args.graph_output}...")
            export_graph_diagram(generator.orchestrator, args.graph_output)
            print()
        
        # Leave company_name/sector None to auto-populate using tools in generate_report
        
        print(f"Generating report for {company_name or stock_symbol} ({stock_symbol}) in {sector or 'unknown'} sector...")
        print("This may take a few minutes as the AI agents gather and analyze data...")
        
        # Generate the report
        results = await generator.generate_report(stock_symbol, company_name, sector)
        
        # Display results
        print("\n" + "="*60)
        print("REPORT GENERATION COMPLETED")
        print("="*60)
        
        if results["workflow_status"] == "completed":
            print(f"✅ Successfully generated report for {stock_symbol}")
            print(f"📊 Company: {results.get('company_name', 'N/A')}")
            print(f"🏢 Sector: {results.get('sector', 'N/A')}")
            print(f"⏱️  Duration: {results.get('duration_seconds', 0):.2f} seconds")
            
            if results.get("pdf_path"):
                print(f"📄 PDF Report: {results['pdf_path']}")
            
            if results.get("final_report"):
                print("\n📋 Report Preview (first 500 characters):")
                print("-" * 40)
                print(results["final_report"][:500] + "..." if len(results["final_report"]) > 500 else results["final_report"])
                print("-" * 40)
            
        else:
            print(f"❌ Report generation failed for {stock_symbol}")
            if results.get("errors"):
                print("Errors encountered:")
                for error in results["errors"]:
                    print(f"  - {error}")
        
        print("\n" + "="*60)
        
    except KeyboardInterrupt:
        print("\n\nReport generation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        logger.error(f"Main function error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())
