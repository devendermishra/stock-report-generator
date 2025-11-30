"""
FastAPI entry point for Stock Report Generator.
Provides REST API endpoints for generating stock research reports.
"""

import logging
import sys
import os
import tempfile
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Import from modules (we're now inside src/, so use relative imports when run as module)
try:
    # Try relative imports first (when run as python -m src.api)
    from .main import StockReportGenerator
    from .config import Config
    from .utils.logging_config import setup_logging
    from .utils.circuit_breaker import get_api_circuit_breaker
    from .tools.pdf_generator_tool import PDFGeneratorTool
except ImportError:
    # Fall back to absolute imports (when PYTHONPATH includes src/)
    from main import StockReportGenerator
    from config import Config
    from utils.logging_config import setup_logging
    from utils.circuit_breaker import get_api_circuit_breaker
    from tools.pdf_generator_tool import PDFGeneratorTool

setup_logging(
    log_dir="logs",
    log_level="INFO",
    include_session_id=True,
    combine_prompts_and_outputs=Config.COMBINE_PROMPTS_AND_OUTPUTS
)

logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize FastAPI app
app = FastAPI(
    title="Stock Report Generator API",
    description="API for generating comprehensive stock research reports",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (configure for production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Add rate limiter to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Global generator instances (initialized on startup)
generator_ai: Optional[StockReportGenerator] = None  # AI mode (default)
generator_structured: Optional[StockReportGenerator] = None  # Structured workflow mode

# Circuit breaker will be initialized lazily with Config values when first accessed
circuit_breaker = None


class ReportRequest(BaseModel):
    """Request model for generating a stock report."""
    stock_symbol: str = Field(..., description="NSE stock symbol (e.g., 'RELIANCE', 'TCS')")
    company_name: Optional[str] = Field(None, description="Full company name (optional, will be fetched if not provided)")
    sector: Optional[str] = Field(None, description="Sector name (optional, will be fetched if not provided)")
    use_ai_research: bool = Field(True, description="Use AI Research Agent (iterative LLM-based research)")
    use_ai_analysis: bool = Field(True, description="Use AI Analysis Agent (iterative LLM-based comprehensive analysis)")
    skip_pdf: bool = Field(True, description="Skip PDF generation and return markdown only (default: True for API)")

    class Config:
        json_schema_extra = {
            "example": {
                "stock_symbol": "RELIANCE",
                "company_name": "Reliance Industries Limited",
                "sector": "Oil & Gas",
                "use_ai_research": True,
                "use_ai_analysis": True,
                "skip_pdf": True
            }
        }


class ReportResponse(BaseModel):
    """Response model for stock report generation."""
    success: bool
    stock_symbol: str
    company_name: Optional[str] = None
    sector: Optional[str] = None
    markdown_report: Optional[str] = None
    session_id: Optional[str] = None
    duration_seconds: Optional[float] = None
    workflow_status: str
    error: Optional[str] = None
    errors: list = []


class PDFRequest(BaseModel):
    """Request model for PDF generation from markdown."""
    markdown_content: str = Field(..., description="Markdown content to convert to PDF")
    stock_symbol: Optional[str] = Field(None, description="Stock symbol for filename (optional)")

    class Config:
        json_schema_extra = {
            "example": {
                "markdown_content": "# Stock Report\n\nThis is a sample report...",
                "stock_symbol": "RELIANCE"
            }
        }


@app.on_event("startup")
async def startup_event():
    """Initialize the Stock Report Generator on startup."""
    global generator_ai, generator_structured, circuit_breaker
    try:
        # Validate configuration
        config_validation = Config.validate_config()
        if not config_validation["openai_key"]:
            logger.error("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
            raise ValueError("OpenAI API key is required")

        # Initialize circuit breaker with Config values
        circuit_breaker = get_api_circuit_breaker()
        logger.info(
            f"Circuit breaker initialized: threshold={Config.CIRCUIT_BREAKER_FAILURE_THRESHOLD}, "
            f"window={Config.CIRCUIT_BREAKER_TIME_WINDOW_SECONDS}s, "
            f"recovery={Config.CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS}s"
        )

        # Initialize AI mode generator (default)
        generator_ai = StockReportGenerator(
            use_ai_research=True,
            use_ai_analysis=True,
            skip_pdf=True  # Default to markdown only for API
        )
        logger.info("Stock Report Generator (AI mode) initialized successfully")

        # Initialize structured workflow generator (for skip-ai mode)
        generator_structured = StockReportGenerator(
            use_ai_research=False,
            use_ai_analysis=False,
            skip_pdf=True  # Default to markdown only for API
        )
        logger.info("Stock Report Generator (Structured mode) initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Stock Report Generator: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Stock Report Generator API",
        "version": "1.0.0",
        "description": "API for generating comprehensive stock research reports",
        "endpoints": {
            "/": "API information (this endpoint)",
            "/health": "Health check endpoint",
            "/report/{symbol}": "Generate stock research report (GET) - Returns markdown",
            "/report/{symbol}?s=true": "Generate report with skip-ai mode (structured workflow)",
            "/pdf": "Convert markdown to PDF (POST) - Returns PDF file"
        },
        "usage": {
            "example": "GET /report/RELIANCE",
            "skip_ai": "GET /report/RELIANCE?s=true"
        },
        "rate_limit": {
            "default": f"{Config.API_RATE_LIMIT_PER_MINUTE} requests per minute",
            "configurable": "Set API_RATE_LIMIT_PER_MINUTE environment variable to customize"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global circuit_breaker
    # Ensure circuit breaker is initialized
    if circuit_breaker is None:
        circuit_breaker = get_api_circuit_breaker()

    circuit_state = circuit_breaker.get_state()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "generator_ai_initialized": generator_ai is not None,
        "generator_structured_initialized": generator_structured is not None,
        "circuit_breaker": {
            "state": circuit_state.value,
            "failure_count": circuit_breaker.get_failure_count(),
            "config": {
                "failure_threshold": Config.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
                "time_window_seconds": Config.CIRCUIT_BREAKER_TIME_WINDOW_SECONDS,
                "recovery_timeout_seconds": Config.CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS
            }
        }
    }


@app.get("/report/{symbol}", response_class=PlainTextResponse)
@limiter.limit(f"{Config.API_RATE_LIMIT_PER_MINUTE}/minute")
async def generate_report(request: Request, symbol: str, s: bool = False):
    """
    Generate a stock research report and return as plain text markdown.

    Simple GET endpoint to generate stock reports.

    Args:
        request: FastAPI Request object (for rate limiting)
        symbol: NSE stock symbol (e.g., 'RELIANCE', 'TCS')
        s: Skip AI mode flag. If True, uses structured workflow instead of AI agents (default: False)

    Returns:
        Plain text markdown report

    Raises:
        HTTPException: If report generation fails, rate limit exceeded, or circuit breaker is open

    Rate Limit:
        Configurable via API_RATE_LIMIT_PER_MINUTE environment variable (default: 2 per minute)

    Examples:
        GET /report/RELIANCE
        GET /report/RELIANCE?s=true
    """
    global circuit_breaker
    # Ensure circuit breaker is initialized
    if circuit_breaker is None:
        circuit_breaker = get_api_circuit_breaker()

    # Check circuit breaker state
    if circuit_breaker.is_open():
        logger.warning("Circuit breaker is OPEN - rejecting request")
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable - Circuit breaker is open due to repeated failures"
        )

    # Select the appropriate generator based on skip-ai mode
    generator = generator_structured if s else generator_ai

    if generator is None:
        circuit_breaker.record_failure()
        mode = "structured" if s else "AI"
        raise HTTPException(
            status_code=503,
            detail=f"Stock Report Generator ({mode} mode) not initialized"
        )

    try:
        logger.info(f"Received report generation request for {symbol} (skip_ai={s}, mode={'structured' if s else 'AI'})")

        # Clean stock symbol (remove .NS suffix if present)
        stock_symbol = symbol.upper()
        if stock_symbol.endswith('.NS'):
            stock_symbol = stock_symbol[:-3]

        # Generate report using the appropriate generator
        results = await generator.generate_report(
            stock_symbol=stock_symbol,
            company_name=None,  # Auto-fetch
            sector=None  # Auto-fetch
        )

        # Check if generation was successful
        if results.get("workflow_status") == "failed":
            error_msg = results.get("error", "Unknown error occurred")
            errors = results.get("errors", [])
            logger.error(f"Report generation failed for {stock_symbol}: {error_msg}")
            circuit_breaker.record_failure()
            raise HTTPException(
                status_code=500,
                detail=f"Report generation failed: {error_msg}"
            )

        # Extract markdown report
        markdown_report = results.get("final_report", "")

        if not markdown_report:
            logger.warning(f"No markdown report generated for {stock_symbol}")
            circuit_breaker.record_failure()
            raise HTTPException(
                status_code=500,
                detail="Report generation completed but no markdown content was produced"
            )

        # Record success if we got here
        circuit_breaker.record_success()

        # Return as plain text markdown
        return markdown_report

    except HTTPException as e:
        # Record failure for HTTP exceptions (except 503 from circuit breaker itself)
        if e.status_code != 503 or "Circuit breaker" not in str(e.detail):
            circuit_breaker.record_failure()
        raise
    except Exception as e:
        logger.error(f"Unexpected error generating report: {e}")
        circuit_breaker.record_failure()
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/pdf")
@limiter.limit("5/minute")
async def generate_pdf_from_markdown(request: Request, pdf_request: PDFRequest):
    """
    Convert markdown content to PDF format.

    Accepts markdown content and returns a PDF file.

    Args:
        request: FastAPI Request object (for rate limiting)
        pdf_request: PDFRequest containing markdown content and optional stock symbol

    Returns:
        PDF file as Response with PDF content

    Raises:
        HTTPException: If PDF generation fails or rate limit exceeded

    Rate Limit:
        5 requests per minute

    Examples:
        POST /pdf
        {
            "markdown_content": "# Report\n\nContent here...",
            "stock_symbol": "RELIANCE"
        }
    """
    try:
        logger.info(f"Received PDF generation request (stock_symbol={pdf_request.stock_symbol})")

        # Generate filename for download
        if pdf_request.stock_symbol:
            filename = f"{pdf_request.stock_symbol}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        else:
            filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        # Create a temporary directory for PDF generation
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize PDF generator with temp directory
            pdf_generator = PDFGeneratorTool(output_dir=temp_dir)

            # Generate PDF
            pdf_path = pdf_generator.generate_pdf(
                markdown_content=pdf_request.markdown_content,
                stock_symbol=pdf_request.stock_symbol
            )

            # Read PDF file content before temp directory is cleaned up
            with open(pdf_path, 'rb') as pdf_file:
                pdf_content = pdf_file.read()

            # Return PDF content as response
            return Response(
                content=pdf_content,
                media_type="application/pdf",
                headers={
                    "Content-Disposition": f"attachment; filename={filename}"
                }
            )

    except Exception as e:
        logger.error(f"Error generating PDF: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"PDF generation failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    # Run the API server
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

