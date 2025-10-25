# Stock Report Generator for NSE

A sophisticated multi-agent AI system that generates comprehensive equity research reports for NSE stocks using **LangGraph** orchestration and **Model Context Protocol (MCP)** for collaborative reasoning.

## 🎯 Overview

This system demonstrates advanced AI collaboration by using multiple specialized agents that work together to analyze stocks from different perspectives:

- **Sector Analysis** - Market trends and peer comparison
- **Stock Research** - Financial metrics and technical analysis  
- **Management Analysis** - Strategic insights from reports and calls
- **Report Review** - Final synthesis and professional formatting

## 🏗️ Architecture

### Multi-Agent System
```
User Input (Stock Symbol)
     │
     ▼
[ SectorResearcherAgent ] ─► sector_summary
     │
     ▼
[ StockResearcherAgent ] ─► stock_summary
     │
     ▼
[ ManagementAnalysisAgent ] ─► management_summary
     │
     ▼
[ ReportReviewerAgent ] ─► final_report.md
```

### Key Components

#### 🤖 Agents (4 Total)
1. **SectorResearcherAgent** - Analyzes sector trends, peer comparison, regulatory environment
2. **StockResearcherAgent** - Retrieves financial data, technical analysis, valuation metrics
3. **ManagementAnalysisAgent** - Processes reports, extracts management insights
4. **ReportReviewerAgent** - Synthesizes all outputs into final professional report

#### 🛠️ Tools (7 Total)
1. **WebSearchTool** - Fetches sector news and market trends (DuckDuckGo search - free, no API key required)
2. **StockDataTool** - Retrieves stock data and metrics (yfinance, NSE API)
3. **ReportFetcherTool** - Downloads financial reports and transcripts
4. **PDFParserTool** - Extracts and processes text from PDF documents
5. **SummarizerTool** - AI-powered text summarization and insight extraction
6. **ReportFormatterTool** - Generates professional markdown reports
7. **PDFGeneratorTool** - Converts markdown reports to professional PDF format

#### 🔄 LangGraph Orchestration
- **Workflow Management** - Coordinates agent execution
- **Error Handling** - Graceful failure recovery
- **State Management** - Tracks progress across agents
- **Conditional Logic** - Smart routing based on results

#### 🧠 MCP Context Sharing
- **Shared Memory** - Agents access previous outputs
- **Data Persistence** - Maintains context across workflow
- **Conflict Resolution** - Handles data inconsistencies
- **Quality Assurance** - Validates and enhances outputs

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
python main.py --symbol RELIANCE --company "Reliance Industries Limited" --sector "Oil & Gas"
```

### Example Usage

```bash
# Generate report for Reliance Industries
python main.py --symbol RELIANCE --company "Reliance Industries Limited" --sector "Oil & Gas"

# Generate report for TCS
python main.py --symbol TCS --company "Tata Consultancy Services" --sector "IT"

# Generate report for HDFC Bank
python main.py --symbol HDFCBANK --company "HDFC Bank Limited" --sector "Banking"
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
├── agents/                    # AI Agents
│   ├── sector_researcher.py   # Sector analysis agent
│   ├── stock_researcher.py    # Stock research agent
│   ├── management_analysis.py # Management analysis agent
│   └── report_reviewer.py     # Final report agent
├── tools/                     # MCP Tools
│   ├── web_search_tool.py     # Web search capabilities
│   ├── stock_data_tool.py     # Stock data retrieval
│   ├── report_fetcher_tool.py # Report downloading
│   ├── pdf_parser_tool.py     # PDF processing
│   ├── summarizer_tool.py     # Text summarization
│   ├── report_formatter_tool.py # Report formatting
│   └── pdf_generator_tool.py  # PDF generation
├── graph/                     # LangGraph Orchestration
│   ├── context_manager_mcp.py # MCP context management
│   └── stock_report_graph.py  # Workflow orchestration
├── main.py                    # Entry point
├── config.py                  # Configuration
└── generate_pdf_from_markdown.py # PDF conversion utility
```

## 🔍 How It Works

### 1. **Sector Research Phase**
- Searches for sector news and trends
- Analyzes peer company performance
- Researches regulatory environment
- Uses AI to synthesize insights

### 2. **Stock Research Phase**
- Retrieves real-time stock data
- Calculates technical indicators
- Performs valuation analysis
- Generates investment rating

### 3. **Management Analysis Phase**
- Downloads financial reports
- Extracts management insights
- Analyzes strategic initiatives
- Identifies risks and opportunities

### 4. **Report Review Phase**
- Combines all agent outputs
- Checks for consistency
- Resolves conflicts
- Formats final report

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

### Custom Configuration
```python
from src.main import StockReportGenerator

generator = StockReportGenerator()
generator.initialize()

# Generate report programmatically
result = await generator.generate_report(
    stock_symbol="RELIANCE",
    company_name="Reliance Industries Limited",
    sector="Oil & Gas"
)
```

### Workflow Status Monitoring
```python
# Check workflow status
status = generator.get_report_status("RELIANCE")
print(f"Status: {status['status']['workflow_status']}")

# Export workflow data
export_data = generator.export_workflow_data("RELIANCE")
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
Main orchestrator class for the system.

#### `MCPContextManager`
Manages shared memory and context between agents.

#### `StockReportGraph`
LangGraph workflow orchestrator.

### Key Methods

#### `generate_report(stock_symbol, company_name, sector)`
Generates a comprehensive stock report.

#### `get_report_status(stock_symbol)`
Gets the current status of report generation.

#### `export_workflow_data(stock_symbol)`
Exports complete workflow data for analysis.

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

## 🔮 Future Enhancements

- **Real-time Data Integration** - Live market data feeds
- **Advanced Analytics** - Machine learning models
- **Portfolio Analysis** - Multi-stock comparison
- **Custom Templates** - Configurable report formats
- **API Endpoints** - REST API for integration
- **Dashboard Interface** - Web-based UI

---

**Built with ❤️ using LangGraph, MCP, and modern AI technologies.**