# Stock Report Generator for NSE

A sophisticated multi-agent AI system that generates comprehensive equity research reports for NSE stocks using **LangGraph** orchestration with **7 specialized autonomous agents** and **15+ integrated tools** for collaborative reasoning.

## 🎯 Overview

This system demonstrates advanced AI collaboration by using multiple specialized agents that work together to analyze stocks from different perspectives:

- **Research Planning** - Creates structured research plans with ordered tool calls
- **Data Gathering** - Company information, sector analysis, peer comparison
- **Financial Analysis** - Comprehensive financial statement analysis
- **Management Analysis** - Management effectiveness and governance assessment
- **Technical Analysis** - Technical indicators and trend analysis
- **Valuation Analysis** - Valuation metrics and target price calculation
- **Report Synthesis** - Final professional report generation

## 🏗️ Architecture

### Multi-Agent System (7 Agents)
```
User Input (Stock Symbol)
     │
     ▼
┌─────────────────────┐
│ ResearchPlanner     │ Creates structured research plan
│      Agent          │
└──────────┬──────────┘
           │
     ┌─────▼─────┐
     │ Research  │ Gathers company & sector data
     │  Agent    │
     └─────┬─────┘
           │
     ┌─────┼─────┬─────────────┐
     │     │     │             │
┌────▼─┐ ┌─▼──┐ ┌─▼──┐ ┌──────▼─┐
│Finan-│ │Mgmt│ │Tech│ │Valuat- │ 4 Analysis Agents
│cial  │ │Anal│ │Anal│ │ion     │ (Run in Parallel)
│Anal  │ │    │ │    │ │Anal    │
└──┬───┘ └─┬──┘ └─┬──┘ └───┬────┘
   └───────┼──────┼────────┘
           │      │
     ┌─────▼──────▼─────┐
     │   Report Agent   │ Synthesizes all results
     └──────────────────┘
           │
     ┌─────▼─────┐
     │ Final     │ Markdown + PDF
     │ Report    │
     └───────────┘
```

### Key Components

#### 🤖 Agents (7 Total)
1. **ResearchPlannerAgent** - Creates structured research plans with ordered tool call sequences
2. **ResearchAgent** - Gathers company information, sector overview, and peer data
3. **FinancialAnalysisAgent** - Performs comprehensive financial statement analysis
4. **ManagementAnalysisAgent** - Analyzes management effectiveness and governance
5. **TechnicalAnalysisAgent** - Performs technical analysis with indicators
6. **ValuationAnalysisAgent** - Performs valuation analysis and target price calculation
7. **ReportAgent** - Synthesizes all data into comprehensive reports

#### 🛠️ Tools (15+ Total)

**Stock Data Tools:**
- **get_stock_metrics** - Retrieves stock price data, financial metrics, market data (yfinance)
- **get_company_info** - Fetches company information, business details, fundamentals
- **validate_symbol** - Validates stock symbols against NSE

**Web Search Tools:**
- **search_sector_news** - Searches for sector-specific news and trends (DuckDuckGo)
- **search_company_news** - Searches for company-specific news
- **search_market_trends** - Searches for market trends and analysis
- **generic_web_search** - Generic web search capability

**Analysis & Calculation Tools:**
- **TechnicalAnalysisFormatter** - Formats and processes technical analysis data
- **StockDataCalculator** - Performs financial calculations and ratios

**Report Generation Tools:**
- **PDFGeneratorTool** - Generates professional PDF reports
- **ReportFormatterTool** - Formats reports in markdown/professional format

**Text Processing Tools:**
- **SummarizerTool** - AI-powered text summarization and insight extraction
  - `summarize_text()` - Summarizes documents with key points
  - `extract_insights()` - Extracts structured insights from text

**Additional Tools:**
- **PDFParserTool** - Extracts text from PDF documents
- **ReportFetcherTool** - Downloads financial reports and transcripts

#### 🔄 LangGraph Orchestration
- **MultiAgentOrchestrator** - Manages workflow and agent coordination
- **StateGraph** - LangGraph StateGraph for workflow management
- **Parallel Execution** - Analysis agents run concurrently for efficiency
- **Structured State** - MultiAgentState (Pydantic model) for communication
- **Error Handling** - Graceful failure recovery with error propagation
- **Conditional Logic** - Smart routing based on results and errors
- **State Management** - Tracks progress across all 7 agents

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- OpenAI API key
- **Optional**: NVIDIA GPU with CUDA support for accelerated AI processing

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd stock-report-generator
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp env.example .env
# Edit .env with your API keys
```

4. **Run the system**
```bash
cd src
python main.py RELIANCE "Reliance Industries Limited" "Oil & Gas"
```

### Example Usage

```bash
# Generate report for Reliance Industries
python src/main.py RELIANCE "Reliance Industries Limited" "Oil & Gas"

# Generate report for TCS (company name and sector auto-detected if not provided)
python src/main.py TCS

# Generate report for HDFC Bank
python src/main.py HDFCBANK "HDFC Bank Limited" "Banking"

# Export graph diagram
python src/main.py RELIANCE --export-graph graph.png

# Export graph only (without generating report)
python src/main.py --export-graph-only graph.png
```

## 🚀 GPU Acceleration (Optional)

For enhanced performance with AI models, you can enable GPU acceleration:

### GPU Setup
```bash
# Install GPU dependencies
pip install -r requirements-gpu.txt

# Verify GPU support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Docker with GPU Support
```bash
# Build GPU-enabled image
docker build -f Dockerfile.gpu -t stock-report-generator:gpu .

# Run with GPU support
docker run --gpus all -it stock-report-generator:gpu
```

### GPU Requirements
- NVIDIA GPU with CUDA Compute Capability 6.0+
- CUDA Toolkit 11.8 or 12.0
- 4GB+ GPU memory (8GB+ recommended)

📖 **Detailed GPU setup guide**: [docs/GPU_SETUP.md](docs/GPU_SETUP.md)

## 📊 Generated Report Structure

The system generates comprehensive reports in both **Markdown** and **PDF** formats with:

### 📋 Executive Summary
- Key highlights and metrics
- Investment thesis
- Management outlook
- Recommendation summary

### 🏢 Company Overview
- Basic information and trading data
- Financial metrics (P/E, P/B, EPS, etc.)
- Market cap and valuation ratios

### 🏭 Sector Analysis
- Sector trends and outlook
- Peer comparison
- Regulatory environment
- Growth opportunities

### 📈 Financial Performance
- Revenue and profit growth
- Valuation metrics
- Technical analysis
- Performance indicators

### 💼 Management Discussion
- Strategic initiatives
- Growth opportunities
- Risk factors
- Management outlook

### 🎯 Investment Recommendation
- Investment thesis
- Target price and valuation
- Risk-reward profile
- Time horizon

### ⚠️ Risk Factors
- Sector-specific risks
- Company risks
- Market risks
- Regulatory risks

## 📄 PDF Generation

The system automatically generates professional PDF reports alongside markdown files. PDF reports feature:

### 🎨 Professional Styling
- Clean, professional layout with proper typography
- Color-coded sections for easy navigation
- Consistent formatting and spacing
- Executive summary highlighting
- Financial metrics emphasis

### 📋 PDF Features
- **Automatic Generation**: PDFs are created automatically with each report
- **Professional Layout**: A4 format with proper margins and headers
- **Styled Sections**: Different visual treatments for various report sections
- **Bold Text Support**: Proper formatting of bold text and emphasis
- **List Formatting**: Clean bullet points and numbered lists

### 🛠️ PDF Tools
- **PDFGeneratorTool**: Core PDF generation functionality
- **Batch Conversion**: Convert multiple markdown files to PDF
- **Custom Styling**: Professional financial report styling
- **Utility Script**: Standalone PDF conversion tool

### 📝 Usage Examples

```bash
# Generate PDF from existing markdown report
python generate_pdf_from_markdown.py reports/stock_report_ICICIBANK_20251021_200913.md

# Batch convert all markdown files to PDF
python generate_pdf_from_markdown.py --batch reports/

# Convert with custom output directory
python generate_pdf_from_markdown.py --output-dir my_pdfs reports/stock_report_ICICIBANK_20251021_200913.md
```

## 🔧 Configuration

### Environment Variables
```bash
# Required
OPENAI_API_KEY=your_openai_api_key
# Note: TAVILY_API_KEY removed - now using free DuckDuckGo search

# Optional
DEFAULT_MODEL=gpt-4o-mini
OUTPUT_DIR=reports
```

### Model Configuration
- **Default Model**: GPT-4o-mini (cost-effective)
- **Max Tokens**: 4000 per request
- **Temperature**: 0.1 (consistent outputs)
- **Context Size**: 10,000 entries

## 🏛️ Project Structure

```
src/
├── agents/                           # AI Agents (7 agents)
│   ├── __init__.py
│   ├── base_agent.py                 # Base agent class
│   ├── research_planner_agent.py    # Research planner agent
│   ├── research_agent.py             # Research agent
│   ├── financial_analysis_agent.py  # Financial analysis agent
│   ├── management_analysis_agent.py # Management analysis agent
│   ├── technical_analysis_agent.py  # Technical analysis agent
│   ├── valuation_analysis_agent.py  # Valuation analysis agent
│   └── report_agent.py               # Report synthesis agent
├── tools/                            # Tool Implementations (15+ tools)
│   ├── __init__.py
│   ├── stock_data_tool.py            # Stock data retrieval (yfinance, NSE)
│   ├── web_search_tool.py            # Web search (DuckDuckGo)
│   ├── generic_web_search_tool.py     # Generic web search
│   ├── summarizer_tool.py            # Text summarization
│   ├── pdf_generator_tool.py          # PDF generation
│   ├── report_formatter_tool.py      # Report formatting
│   ├── technical_analysis_formatter.py # Technical analysis
│   ├── stock_data_calculator.py      # Financial calculations
│   ├── pdf_parser_tool.py            # PDF parsing
│   └── report_fetcher_tool.py       # Report fetching
├── graph/                            # LangGraph Orchestration
│   └── multi_agent_graph.py          # MultiAgentOrchestrator
├── main.py                            # Entry point
└── config.py                          # Configuration

REQUIREMENTS_CHECKLIST.md              # Requirements verification
IMPLEMENTATION_SUMMARY.md               # Implementation details
MULTI_AGENT_README.md                  # Comprehensive documentation
```

## 🔍 How It Works

### 1. **Research Planning Phase** (ResearchPlannerAgent)
- Analyzes stock symbol, company, and sector context
- Reviews available tools
- Creates structured research plan with ordered tool calls
- Outputs executable plan for data gathering

### 2. **Data Gathering Phase** (ResearchAgent)
- Executes research plan from planner
- Retrieves real-time stock data (yfinance, NSE)
- Searches for company and sector news
- Gathers peer comparison data
- Collects comprehensive research data

### 3. **Parallel Analysis Phase** (4 Agents run concurrently)
- **FinancialAnalysisAgent**: Financial statement analysis, ratios, health assessment
- **ManagementAnalysisAgent**: Management effectiveness, governance analysis
- **TechnicalAnalysisAgent**: Technical indicators, trends, support/resistance
- **ValuationAnalysisAgent**: Valuation metrics, target price calculation

### 4. **Report Synthesis Phase** (ReportAgent)
- Receives all analysis results from 4 analysis agents
- Synthesizes comprehensive report
- Formats professional markdown document
- Generates PDF report with styling
- Ensures all required sections are included

## 🛡️ Error Handling & Quality Assurance

### Robust Error Handling
- **Graceful Degradation** - System continues with partial data
- **Fallback Analysis** - AI-generated content when data unavailable
- **Error Recovery** - Automatic retry mechanisms
- **Quality Validation** - Consistency checking across agents

### Data Quality Assurance
- **Source Validation** - Verifies data sources
- **Consistency Checking** - Identifies conflicting information
- **Confidence Scoring** - Rates analysis reliability
- **Quality Metrics** - Tracks report completeness

## 📈 Performance & Scalability

### Optimization Features
- **Parallel Processing** - Agents can run concurrently
- **Caching** - Reuses previously fetched data
- **Rate Limiting** - Respects API limits
- **Memory Management** - Efficient context storage

### Monitoring & Logging
- **Comprehensive Logging** - Tracks all operations
- **Performance Metrics** - Measures execution time
- **Error Tracking** - Monitors failures
- **Context Inspection** - Debug workflow state

## 🔧 Advanced Usage

### Programmatic Usage
```python
import asyncio
from src.main import StockReportGenerator

# Initialize generator
generator = StockReportGenerator()

# Generate report programmatically
async def main():
    result = await generator.generate_report(
        stock_symbol="RELIANCE",
        company_name="Reliance Industries Limited",  # Optional, auto-detected
        sector="Oil & Gas"  # Optional, auto-detected
    )
    
    print(f"Status: {result['workflow_status']}")
    print(f"PDF Path: {result.get('pdf_path')}")
    print(f"Errors: {result.get('errors', [])}")

# Run async function
asyncio.run(main())
```

### Synchronous Usage
```python
from src.main import StockReportGenerator

generator = StockReportGenerator()

# Synchronous wrapper
result = generator.generate_report_sync("RELIANCE")
print(f"Report generated: {result['workflow_status']}")
```

## 🧪 Testing & Development

### Running Tests
```bash
pytest tests/
```

### Code Quality
```bash
# Format code
black src/

# Lint code
flake8 src/

# Type checking
mypy src/
```

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt

# Set up pre-commit hooks
pre-commit install
```

## 📚 API Documentation

### Core Classes

#### `StockReportGenerator`
Main orchestrator class for the system. Initializes and manages the multi-agent workflow.

#### `MultiAgentOrchestrator`
LangGraph-based orchestrator that manages workflow and agent coordination.
- Manages 7 agents with distinct roles
- Coordinates parallel execution of analysis agents
- Handles state management and error propagation

### Key Methods

#### `generate_report(stock_symbol, company_name=None, sector=None)`
Generates a comprehensive stock report using all 7 agents.
- **stock_symbol**: NSE stock symbol (required)
- **company_name**: Full company name (optional, auto-detected)
- **sector**: Sector name (optional, auto-detected)
- **Returns**: Dictionary with workflow results, PDF path, errors, etc.

#### `generate_report_sync(stock_symbol, company_name=None, sector=None)`
Synchronous wrapper for `generate_report()`.

### Agent Classes

All agents inherit from `BaseAgent` and implement:
- `execute_task()` - Main task execution
- `execute_task_partial()` - Partial state update execution
- `select_tools()` - Autonomous tool selection

## 🤝 Contributing

We welcome contributions to the Stock Report Generator! Please follow these guidelines:

### Getting Started
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add type hints to all functions
- Include comprehensive docstrings
- Write unit tests for new features
- Update documentation as needed
- Keep functions under 100 lines
- Keep classes under 20 methods

### Code Quality Standards
- All functions must have type hints
- All classes and functions must have docstrings
- Functions should not exceed 100 lines
- Classes should not exceed 20 methods
- Scripts should not exceed 500 lines

### Testing
- Run unit tests: `python -m pytest tests/unit/`
- Run integration tests: `python -m pytest tests/integration/`
- Ensure test coverage is maintained

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support & Contact

### Maintainer Contact Information
- **GitHub**: [@devendermishra]
- **Issues**: [GitHub Issues](https://github.com/devendermishra/stock-report-generator/issues)

### Getting Help
- Check the [documentation](docs/)
- Search [existing issues](https://github.com/devendermishra/stock-report-generator/issues)
- Create a [new issue](https://github.com/devendermishra/stock-report-generator/issues/new) for bugs or feature requests
- Join our [discussions](https://github.com/devendermishra/stock-report-generator/discussions) for questions

## 🆘 Troubleshooting

For issues and questions:
1. Check the logs in `stock_report_generator.log`
2. Review the error messages
3. Verify API keys are correct
4. Ensure all dependencies are installed

## ✅ Requirements Compliance

This system meets and exceeds multi-agent system requirements:

- ✅ **7 specialized agents** with distinct roles (exceeds minimum of 3)
- ✅ **15+ integrated tools** (exceeds minimum of 3)
- ✅ **LangGraph orchestration** framework with parallel execution
- ✅ Clear communication via structured `MultiAgentState`
- ✅ Tools extend beyond basic LLM responses (API calls, file processing, calculations)

**See `REQUIREMENTS_CHECKLIST.md` for detailed verification.**

## 🔮 Future Enhancements

- **Real-time Data Integration** - Live market data feeds
- **Additional Agents** - News analysis, risk assessment, ESG analysis
- **Advanced Analytics** - Machine learning models for predictions
- **Portfolio Analysis** - Multi-stock comparison
- **Custom Templates** - Configurable report formats
- **API Endpoints** - REST API for integration
- **Dashboard Interface** - Web-based UI
- **Visualization Tools** - Charts, graphs, interactive reports

## 📖 Additional Documentation

- **REQUIREMENTS_CHECKLIST.md** - Detailed requirements verification
- **IMPLEMENTATION_SUMMARY.md** - Comprehensive implementation details
- **MULTI_AGENT_README.md** - Agent-specific documentation

---

**Built with ❤️ using LangGraph, LangChain, and modern AI technologies.**

**Key Technologies:**
- LangGraph for multi-agent orchestration
- LangChain for tool integration
- OpenAI GPT models for agent reasoning
- yfinance for stock data
- DuckDuckGo for web search
- ReportLab for PDF generation