# LangGraph Integration for Dynamic Tool Execution

## 🎯 Overview

I've successfully refactored the dynamic agent system to use **LangGraph** and **LangChain** frameworks instead of a custom execution framework. This provides better tool orchestration, intelligent tool selection, and robust error handling.

## 🚀 Key Changes

### 1. **LangGraph Dynamic Agent** (`src/agents/langgraph_dynamic_agent.py`)
- Uses LangGraph framework for workflow orchestration
- LLM-driven tool selection and execution
- Intelligent error handling and recovery
- State management with TypedDict

### 2. **LangChain Tools Integration** (`src/tools/langchain_tools_integration.py`)
- Converts all existing tools to LangChain-compatible format
- Provides unified interface for tool execution
- Supports all tool categories: stock data, web search, report processing, analysis

### 3. **Updated Main Integration** (`src/main.py`)
- Added `--langgraph` command line option
- Integrated LangGraph agent system
- Enhanced report generation with LangGraph framework

## 🔧 How It Works

### **LangGraph Workflow**
```
User Request → LangGraph Agent → LLM Analysis → Tool Selection → Execution → Synthesis → Results
```

### **Tool Execution Flow**
1. **LLM Analyzes Context**: Determines what analysis is needed
2. **Tool Selection**: LLM chooses appropriate tools based on context
3. **LangChain Execution**: Tools are executed using LangChain framework
4. **Result Synthesis**: LLM synthesizes results into final analysis

## 📋 Usage Examples

### **Command Line Usage**

```bash
# Use LangGraph framework for intelligent tool execution
python src/main.py --symbol RELIANCE --langgraph

# LangGraph with verbose logging
python src/main.py --symbol TCS --langgraph --verbose

# LangGraph with custom output directory
python src/main.py --symbol HDFCBANK --langgraph --output-dir my_reports
```

### **Python API Usage**

```python
import asyncio
from src.main import StockReportGenerator

async def main():
    # Initialize generator
    generator = StockReportGenerator()
    generator.initialize()
    
    # Generate LangGraph report
    result = await generator.generate_langgraph_report("RELIANCE")
    
    if result["success"]:
        print(f"✅ Report: {result['report_path']}")
        print(f"🎯 Rating: {result['analysis_result']['investment_rating']}")
        print(f"📊 Confidence: {result['analysis_result']['confidence_score']}")

asyncio.run(main())
```

## 🎯 Available Report Types

| Report Type | Command | Framework | Description |
|-------------|---------|-----------|-------------|
| **Traditional** | `--symbol RELIANCE` | Fixed Sequence | Original fixed-action approach |
| **Dynamic** | `--symbol RELIANCE --dynamic` | Custom Framework | LLM-driven action planning |
| **LangGraph** | `--symbol RELIANCE --langgraph` | LangGraph + LangChain | Framework-based tool execution |

## 🔧 LangGraph Agent Architecture

### **State Management**
```python
class AgentState(TypedDict):
    messages: List[Any]                    # Conversation history
    stock_symbol: str                      # Stock symbol
    company_name: str                      # Company name
    analysis_context: Dict[str, Any]       # Analysis context
    current_step: str                      # Current workflow step
    analysis_results: Dict[str, Any]       # Analysis results
    recommendations: Dict[str, Any]        # Investment recommendations
    errors: List[str]                      # Error tracking
    session_id: str                        # Session identifier
```

### **Workflow Nodes**
1. **Analyze Node**: LLM analyzes request and determines approach
2. **Execute Tools Node**: LangChain executes selected tools
3. **Synthesize Node**: LLM synthesizes results into final analysis

### **Tool Integration**
All existing tools are converted to LangChain format:
- **Stock Data Tools**: `get_stock_metrics`, `get_historical_data`, `calculate_technical_indicators`
- **Web Search Tools**: `search_market_news`, `analyze_sector_trends`
- **Report Processing Tools**: `fetch_financial_reports`, `parse_pdf_document`
- **Analysis Tools**: `summarize_text`, `extract_insights`, `analyze_sentiment`

## 🎯 Key Benefits

### **1. Framework-Based Execution**
- Uses proven LangGraph and LangChain frameworks
- Better error handling and recovery
- Robust tool orchestration

### **2. Intelligent Tool Selection**
- LLM decides which tools to use based on context
- Dynamic tool selection and execution
- Optimal workflow orchestration

### **3. Enhanced Reliability**
- Framework-level error handling
- Automatic retry mechanisms
- Better state management

### **4. Extensibility**
- Easy to add new tools
- Framework supports complex workflows
- Better integration with LangChain ecosystem

## 📊 Example Output

### **LangGraph Report Structure**
```
# LangGraph Dynamic Stock Analysis Report

## 📊 Stock Information
- Symbol: RELIANCE
- Company: Reliance Industries Limited
- Sector: Oil & Gas
- Analysis Type: LangGraph LLM-Driven Analysis

## 🎯 Investment Analysis
- Investment Rating: BUY
- Confidence Score: 0.85
- Market Sentiment: Positive

## 📈 Key Metrics
- Current Price: ₹2,456.75
- Market Cap: ₹16,600,000,000,000
- PE Ratio: 12.5

## 🤖 LangGraph Analysis Details
- Agent Type: LangGraph Dynamic Agent
- Tool Execution: LLM-driven tool selection and execution
- Analysis Method: Intelligent workflow orchestration
```

## 🔄 Comparison: Custom vs LangGraph

| Aspect | Custom Framework | LangGraph Framework |
|--------|------------------|---------------------|
| **Tool Execution** | Custom orchestration | LangChain framework |
| **Error Handling** | Basic retry logic | Framework-level handling |
| **State Management** | Manual tracking | TypedDict state |
| **Workflow Control** | Custom logic | LangGraph nodes |
| **Tool Integration** | Custom wrappers | Native LangChain tools |
| **Reliability** | Basic | Enterprise-grade |
| **Extensibility** | Limited | Framework-supported |

## 🚀 Getting Started

### **1. Install Dependencies**
```bash
pip install langgraph langchain langchain-openai
```

### **2. Set Environment Variables**
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

### **3. Run LangGraph Analysis**
```bash
python src/main.py --symbol RELIANCE --langgraph --verbose
```

### **4. Check Results**
```bash
# Reports are saved in the output directory
ls reports/langgraph_stock_report_RELIANCE_*.md
```

## 🎯 Advanced Usage

### **Custom Tool Integration**
```python
# Add custom tools to LangChain integration
@tool
def custom_analysis_tool(symbol: str) -> Dict[str, Any]:
    """Custom analysis tool."""
    # Your custom logic here
    return {"result": "analysis"}

# Register with LangChain tools integration
langchain_tools_integration.langchain_tools.append(custom_analysis_tool)
```

### **Workflow Customization**
```python
# Customize LangGraph workflow
workflow = StateGraph(AgentState)
workflow.add_node("custom_analysis", custom_analysis_node)
workflow.add_edge("analyze", "custom_analysis")
```

## 🎉 Summary

The LangGraph integration provides:

1. **Framework-Based Execution**: Uses proven LangGraph and LangChain frameworks
2. **Intelligent Tool Selection**: LLM-driven tool selection and execution
3. **Enhanced Reliability**: Better error handling and state management
4. **Easy Integration**: Seamless integration with existing tools
5. **Extensibility**: Easy to add new tools and workflows

The system now uses industry-standard frameworks for tool execution while maintaining the intelligent, LLM-driven approach to dynamic action planning! 🚀

