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
import time
from typing import Dict, Any, Optional

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Try relative imports first (when run as module)
    from .config import Config
    from .graph.multi_agent_graph import MultiAgentOrchestrator
    from .tools.stock_data_tool import validate_symbol as tool_validate_symbol
    from .tools.stock_data_tool import get_company_info as tool_get_company_info
    from .utils.retry import retry_tool_call
except ImportError:
    # Fall back to absolute imports (when run as script)
    from config import Config
    from graph.multi_agent_graph import MultiAgentOrchestrator
    from tools.stock_data_tool import validate_symbol as tool_validate_symbol
    from tools.stock_data_tool import get_company_info as tool_get_company_info
    try:
        from utils.retry import retry_tool_call
    except ImportError:
        retry_tool_call = lambda func: func  # No-op if retry not available

# Import enhanced logging and session management
try:
    from .utils.logging_config import setup_logging
    from .utils.session_manager import SessionContext, set_session_context
except ImportError:
    from utils.logging_config import setup_logging
    from utils.session_manager import SessionContext, set_session_context

# Configure enhanced logging with MDC support
# Use Config to determine if prompts and outputs should be combined
try:
    from .config import Config
except ImportError:
    from config import Config

setup_logging(
    log_dir="logs",
    log_level="INFO",
    include_session_id=True,
    combine_prompts_and_outputs=Config.COMBINE_PROMPTS_AND_OUTPUTS
)

logger = logging.getLogger(__name__)

# Initialize metrics
try:
    from .utils.metrics import initialize_metrics, get_metrics_status
except ImportError:
    from utils.metrics import initialize_metrics, get_metrics_status

# Initialize Prometheus metrics if enabled
try:
    initialize_metrics()
except Exception as e:
    logger.debug(f"Metrics initialization: {e}")

# Log metrics status
try:
    metrics_status = get_metrics_status()
    if metrics_status.get("prometheus_enabled"):
        logger.info("Metrics collection enabled (Prometheus available)")
    else:
        logger.debug("Metrics collection enabled (in-memory only)")
except Exception as e:
    logger.debug(f"Could not get metrics status: {e}")

class StockReportGenerator:
    """
    Main class for the Agentic Stock Research Report Generator.

    This system uses three autonomous agents:
    1. ResearchAgent - Gathers company information, sector overview, and peer data
    2. AnalysisAgent - Performs financial, management, and technical analysis
    3. ReportAgent - Synthesizes all data into comprehensive reports
    """

    def __init__(self, openai_api_key: Optional[str] = None, use_ai_research: bool = True, use_ai_analysis: bool = True, skip_pdf: bool = False):
        """
        Initialize the Stock Report Generator.

        Args:
            openai_api_key: OpenAI API key (if not provided, will use config)
            use_ai_research: If True, use AIResearchAgent (iterative LLM-based research).
                            If False, use ResearchPlannerAgent + ResearchAgent (structured workflow).
            use_ai_analysis: If True, use AIAnalysisAgent (iterative LLM-based comprehensive analysis).
                           If False, use separate Financial, Management, Technical, Valuation agents.
            skip_pdf: If True, skip PDF generation and only return markdown content.
        """
        self.openai_api_key = openai_api_key or Config.OPENAI_API_KEY
        self.use_ai_research = use_ai_research
        self.use_ai_analysis = use_ai_analysis
        self.skip_pdf = skip_pdf

        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it directly.")

        # Initialize the multi-agent orchestrator
        self.orchestrator = MultiAgentOrchestrator(
            self.openai_api_key,
            use_ai_research=use_ai_research,
            use_ai_analysis=use_ai_analysis,
            skip_pdf=skip_pdf
        )

        research_mode = "AI Research Agent" if use_ai_research else "Research Planner + Research Agent"
        analysis_mode = "AI Analysis Agent" if use_ai_analysis else "Separate Analysis Agents"
        logger.info(f"Stock Report Generator initialized successfully (Research: {research_mode}, Analysis: {analysis_mode})")

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
        # Import metrics for report generation tracking
        try:
            from .utils.metrics import record_report, record_error
        except ImportError:
            from utils.metrics import record_report, record_error

        # Record start time for report generation metrics
        report_start_time = time.time()

        # Create session context for this invocation
        with SessionContext() as session_id:
            try:
                # Set session context with stock symbol
                set_session_context('stock_symbol', stock_symbol)

                logger.info(f"Starting report generation for {stock_symbol} [Session: {session_id}]")

                # Validate inputs
                if not stock_symbol:
                    raise ValueError("Stock symbol is required")

                # Clean stock symbol (remove .NS suffix if present)
                if stock_symbol.endswith('.NS'):
                    stock_symbol = stock_symbol[:-3]

                # Validate symbol against NSE via tool (with retry)
                @retry_tool_call()
                def _validate_symbol(symbol):
                    return tool_validate_symbol.invoke({"symbol": symbol})

                validation = _validate_symbol(stock_symbol)
                if not validation or not validation.get("valid", False):
                    error_msg = validation.get("error", "Symbol not found on NSE")
                    # Metrics already recorded in validate_symbol, but record error here too
                    try:
                        from .utils.metrics import record_error
                    except ImportError:
                        from utils.metrics import record_error
                    record_error("validation_failed", f"main.generate_report")
                    raise ValueError(error_msg)

                # Fetch company info from tool to populate missing fields (with retry)
                @retry_tool_call()
                def _get_company_info(symbol):
                    return tool_get_company_info.invoke({"symbol": symbol})

                info = _get_company_info(stock_symbol) or {}
                if not company_name:
                    company_name = info.get("company_name") or info.get("short_name") or validation.get("company_name") or f"Company {stock_symbol}"
                if not sector:
                    sector = info.get("sector") or validation.get("sector") or "Unknown"

                # Update session context
                set_session_context('company_name', company_name)
                set_session_context('sector', sector)

                # Run the multi-agent workflow
                results = await self.orchestrator.run_workflow(
                    stock_symbol=stock_symbol,
                    company_name=company_name,
                    sector=sector,
                    skip_pdf=self.skip_pdf
                )

                # Add session ID to results
                results['session_id'] = session_id

                # Calculate report generation duration
                report_duration = time.time() - report_start_time

                # Record report generation metrics
                try:
                    status = "completed" if results["workflow_status"] == "completed" else "failed"
                    record_report(stock_symbol, report_duration, status)
                except Exception as metrics_error:
                    logger.debug(f"Failed to record report generation metrics: {metrics_error}")

                # Log results
                if results["workflow_status"] == "completed":
                    logger.info(f"Successfully generated report for {stock_symbol} [Session: {session_id}]")
                    if results.get("pdf_path"):
                        logger.info(f"PDF report saved to: {results['pdf_path']}")
                else:
                    logger.warning(f"Report generation completed with errors for {stock_symbol} [Session: {session_id}]")
                    if results.get("errors"):
                        logger.warning(f"Errors: {results['errors']}")

                return results

            except Exception as e:
                # Calculate report generation duration even on failure
                report_duration = time.time() - report_start_time

                # Record failed report generation metrics
                try:
                    record_report(stock_symbol, report_duration, "failed")
                    record_error("report_generation_failed", f"main.generate_report.{type(e).__name__}")
                except Exception as metrics_error:
                    logger.debug(f"Failed to record report generation metrics: {metrics_error}")

                logger.error(f"Report generation failed for {stock_symbol} [Session: {session_id}]: {e}")
                return {
                    "stock_symbol": stock_symbol,
                    "session_id": session_id,
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
        try:
            # Check if there's a running event loop
            loop = asyncio.get_running_loop()
            # If we're in an async context, create a new event loop in a new thread
            # This allows the sync method to work even when called from async context
            import concurrent.futures
            import threading
            
            def run_in_new_loop():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(
                        self.generate_report(stock_symbol, company_name, sector)
                    )
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_new_loop)
                return future.result()
        except RuntimeError:
            # No running event loop, safe to use asyncio.run()
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
  python main.py RELIANCE --skip-ai
  python main.py RELIANCE -s
  python main.py RELIANCE --skip-pdf
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
        parser.add_argument('--skip-ai', '-s', dest='skip_ai', action='store_true',
                          help='Skip AI agents and use ResearchPlannerAgent + ResearchAgent (structured workflow) instead of AIResearchAgent and AIAnalysisAgent')
        parser.add_argument('--skip-pdf', dest='skip_pdf', action='store_true',
                          help='Skip PDF generation and output the full Markdown content instead')

        args = parser.parse_args()

        # Check configuration
        config_validation = Config.validate_config()
        if not config_validation["openai_key"]:
            print("Error: OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
            sys.exit(1)

        # Show metrics status
        try:
            metrics_status = get_metrics_status()
            if metrics_status.get("prometheus_enabled"):
                print(f"📊 Metrics collection enabled (Prometheus on port {Config.METRICS_PORT})")
            else:
                print("📊 Metrics collection enabled (in-memory)")
        except Exception:
            pass  # Metrics always work, so this is just for display

        # Default to AI agents, use structured workflow only if --skip-ai is passed
        use_ai_research = not args.skip_ai
        use_ai_analysis = not args.skip_ai

        # Initialize the generator with AI flags and skip_pdf flag
        generator = StockReportGenerator(
            use_ai_research=use_ai_research,
            use_ai_analysis=use_ai_analysis,
            skip_pdf=args.skip_pdf
        )

        # Log which mode is being used
        if use_ai_research and use_ai_analysis:
            logger.info("Using AI Research Agent (iterative LLM-based research)")
            logger.info("Using AI Analysis Agent (iterative LLM-based comprehensive analysis)")
        else:
            logger.info("Using Research Planner + Research Agent (structured workflow)")
            logger.info("Using separate Financial, Management, Technical, Valuation Analysis Agents")

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

            if args.skip_pdf:
                # Output full markdown content when --skip-pdf is used
                if results.get("final_report"):
                    print("\n📋 Full Markdown Report:")
                    print("=" * 60)
                    print(results["final_report"])
                    print("=" * 60)
            else:
                # Normal mode: show PDF path and preview
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

def cli_main():
    """CLI entry point for pip-installed package."""
    asyncio.run(main())

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())
