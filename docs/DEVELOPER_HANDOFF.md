# Developer Handoff Documentation

**Purpose**: This document provides essential information for developers taking over or contributing to the Stock Report Generator project. It focuses on practical development aspects, code organization, and common workflows.

**Last Updated**: 2024

---

## Table of Contents

1. [Quick Start for Developers](#quick-start-for-developers)
2. [Project Structure](#project-structure)
3. [Architecture Overview](#architecture-overview)
4. [Development Workflow](#development-workflow)
5. [Key Components Deep Dive](#key-components-deep-dive)
6. [Adding New Features](#adding-new-features)
7. [Testing Guide](#testing-guide)
8. [Common Issues & Solutions](#common-issues--solutions)
9. [Code Organization Principles](#code-organization-principles)
10. [Deployment & Operations](#deployment--operations)

---

## Quick Start for Developers

### Prerequisites

- **Python 3.10+** (3.11+ recommended)
- **OpenAI API Key** ([Get one here](https://platform.openai.com/api-keys))
- **Git** for version control
- **Virtual Environment** (venv or conda)

### Initial Setup

```bash
# 1. Clone the repository
git clone <repository-url>
cd stock-report-generator

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate  # Windows

# 3. Install in editable mode (recommended for development)
pip install -e .

# 4. Set up environment variables
cp env.example .env
# Edit .env and add your OPENAI_API_KEY

# 5. Verify installation
python -c "import langchain, langgraph, openai; print('✅ Setup complete!')"

# 6. Run a test report
cd src
python main.py RELIANCE
```

### Development Dependencies

```bash
# Install all dependencies including dev tools
pip install -r requirements.txt

# For code quality tools
pip install black flake8 mypy pytest pytest-cov
```

### First Code Change

1. **Make a small change** (e.g., update a log message)
2. **Run tests**: `pytest tests/unit/`
3. **Check formatting**: `black --check src/`
4. **Run linter**: `flake8 src/`
5. **Test your change**: `python src/main.py RELIANCE`

---

## Project Structure

```
stock-report-generator/
├── src/                          # Main source code
│   ├── agents/                   # AI Agent implementations
│   │   ├── base_agent.py         # Base class for all agents
│   │   ├── ai_research_agent.py  # AI-powered research (default mode)
│   │   ├── ai_analysis_agent.py  # AI-powered analysis (default mode)
│   │   ├── ai_report_agent.py    # AI-powered report generation
│   │   ├── research_planner_agent.py  # Structured research planning
│   │   ├── research_agent.py     # Structured research execution
│   │   ├── financial_analysis_agent.py
│   │   ├── management_analysis_agent.py
│   │   ├── technical_analysis_agent.py
│   │   ├── valuation_analysis_agent.py
│   │   └── report_agent.py       # Structured report synthesis
│   │
│   ├── tools/                    # Tool implementations (15+ tools)
│   │   ├── stock_data_tool.py    # Yahoo Finance integration
│   │   ├── web_search_tool.py     # DuckDuckGo search
│   │   ├── summarizer_tool.py    # Text summarization
│   │   ├── pdf_generator_tool.py # PDF generation
│   │   ├── report_formatter_tool.py
│   │   └── ...                   # Other tools
│   │
│   ├── graph/                    # LangGraph orchestration
│   │   └── multi_agent_graph.py  # Main orchestrator
│   │
│   ├── utils/                    # Utility modules
│   │   ├── logging_config.py     # Enhanced logging setup
│   │   ├── session_manager.py    # Session context management
│   │   ├── retry.py              # Retry mechanisms
│   │   ├── metrics.py            # Performance metrics
│   │   └── circuit_breaker.py   # Circuit breaker pattern
│   │
│   ├── main.py                   # CLI entry point
│   └── config.py                 # Configuration management
│
├── tests/                        # Test suite
│   ├── unit/                     # Unit tests
│   └── integration/              # Integration tests
│
├── docs/                         # Documentation
│   ├── DOCUMENTATION.md          # Comprehensive system docs
│   ├── AGENT_SPECIALIZATION.md   # Agent details
│   ├── DEPLOYMENT.md             # Deployment guide
│   └── DEVELOPER_HANDOFF.md      # This file
│
├── data/                         # Data directories
│   ├── inputs/                   # Input files
│   ├── outputs/                  # Output files
│   └── processed/                # Processed data
│
├── reports/                      # Generated reports (output)
├── logs/                         # Application logs
├── requirements.txt              # Python dependencies
├── pyproject.toml                # Project configuration
└── README.md                     # Project overview
```

### Key Directories Explained

- **`src/agents/`**: All agent implementations. Each agent inherits from `BaseAgent` and implements specific analysis tasks.
- **`src/tools/`**: LangChain-compatible tools that agents can call. Tools are stateless functions that perform specific operations.
- **`src/graph/`**: LangGraph orchestration logic. Defines workflow, state management, and agent coordination.
- **`src/utils/`**: Shared utilities for logging, retries, metrics, and error handling.
- **`tests/`**: Comprehensive test suite. Unit tests for individual components, integration tests for workflows.

---

## Architecture Overview

### System Design

The system uses a **multi-agent architecture** orchestrated by **LangGraph**:

```
┌─────────────────────────────────────────────────────────┐
│              StockReportGenerator (main.py)              │
│                  Entry point & CLI                       │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│         MultiAgentOrchestrator (multi_agent_graph.py)   │
│              LangGraph StateGraph                       │
│         - State management                              │
│         - Agent coordination                            │
│         - Parallel execution                           │
└──────────────────────┬──────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│   Research  │ │  Analysis   │ │   Report    │
│   Agents    │ │   Agents    │ │   Agent     │
└─────────────┘ └─────────────┘ └─────────────┘
        │              │              │
        └──────────────┼──────────────┘
                       │
                       ▼
              ┌────────────────┐
              │  Tools (15+)   │
              │  - Stock Data  │
              │  - Web Search  │
              │  - Analysis    │
              │  - PDF Gen     │
              └────────────────┘
```

### Two Operational Modes

#### 1. AI-Powered Iterative Mode (Default)

**Flow**: `AIResearchAgent` → `AIAnalysisAgent` → `AIReportAgent`

- **AIResearchAgent**: Iteratively decides which tools to call for research
- **AIAnalysisAgent**: Comprehensively analyzes across all dimensions
- **AIReportAgent**: Generates report content using LLM

**When to use**: Default mode, best for comprehensive analysis

#### 2. Structured Workflow Mode (`--skip-ai`)

**Flow**: `ResearchPlannerAgent` → `ResearchAgent` → `[4 Analysis Agents in Parallel]` → `ReportAgent`

- **ResearchPlannerAgent**: Creates structured research plan
- **ResearchAgent**: Executes plan sequentially
- **4 Analysis Agents**: Run in parallel (Financial, Management, Technical, Valuation)
- **ReportAgent**: Synthesizes all results

**When to use**: More predictable, faster execution, lower token usage

### State Management

The system uses `MultiAgentState` (Pydantic model) to share data between agents:

```python
class MultiAgentState(BaseModel):
    stock_symbol: str
    company_name: str
    sector: str
    research_results: Optional[Dict[str, Any]] = None
    analysis_results: Optional[Dict[str, Any]] = None
    final_report: Optional[str] = None
    pdf_path: Optional[str] = None
    errors: Annotated[List[str], reduce_errors] = []
    # ... more fields
```

**Key Points**:
- State is immutable - agents return updates, not modify directly
- LangGraph merges state updates automatically
- Error lists use reducer functions for concurrent updates

---

## Development Workflow

### Making Code Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following code organization principles (see below)

3. **Run tests**:
   ```bash
   pytest tests/unit/          # Unit tests
   pytest tests/integration/   # Integration tests
   pytest tests/ -v            # All tests with verbose output
   ```

4. **Check code quality**:
   ```bash
   black --check src/          # Formatting
   flake8 src/                 # Linting
   mypy src/                   # Type checking (optional)
   ```

5. **Test your changes**:
   ```bash
   python src/main.py RELIANCE --skip-pdf  # Quick test
   ```

6. **Commit and push**:
   ```bash
   git add .
   git commit -m "feat: description of changes"
   git push origin feature/your-feature-name
   ```

### Running the Application

**Basic usage**:
```bash
cd src
python main.py RELIANCE
```

**With options**:
```bash
# Use structured workflow (skip AI agents)
python main.py RELIANCE --skip-ai

# Skip PDF generation (faster for testing)
python main.py RELIANCE --skip-pdf

# Export workflow graph
python main.py RELIANCE --export-graph graph.png
```

**Programmatic usage**:
```python
from src.main import StockReportGenerator

generator = StockReportGenerator()
result = generator.generate_report_sync("RELIANCE")
print(result["workflow_status"])
```

### Debugging

**Enable debug logging**:
```python
# In src/config.py or .env
LOG_LEVEL=DEBUG
```

**View logs**:
```bash
tail -f logs/stock_report_generator.log
tail -f logs/prompts.log  # LLM prompts and responses
```

**Common debugging approaches**:
1. Check logs in `logs/` directory
2. Use `--skip-pdf` for faster iteration
3. Test individual agents in isolation
4. Use Python debugger: `import pdb; pdb.set_trace()`

---

## Key Components Deep Dive

### 1. BaseAgent Architecture

All agents inherit from `BaseAgent`:

```python
class BaseAgent(ABC):
    def __init__(self, agent_id: str, openai_api_key: str, available_tools: List[Any]):
        # Initialization
    
    @abstractmethod
    async def execute_task(self, ...) -> AgentState:
        # Main task execution
    
    def select_tools(self, task: str, context: Dict) -> List[Any]:
        # Autonomous tool selection
```

**Key Methods**:
- `execute_task()`: Main execution logic (must be implemented)
- `select_tools()`: Choose which tools to use (can be overridden)
- `execute_task_partial()`: Partial state updates for LangGraph

### 2. Tool Implementation Pattern

Tools are LangChain-compatible callables:

```python
from langchain_core.tools import tool

@tool
def get_stock_metrics(symbol: str) -> Dict[str, Any]:
    """Retrieve stock metrics from Yahoo Finance."""
    # Implementation
    return {"price": 100, "volume": 1000000}
```

**Tool Registration**:
Tools are registered in `MultiAgentOrchestrator.__init__()` and passed to agents.

### 3. LangGraph Workflow

The workflow is defined in `MultiAgentOrchestrator`:

```python
# Create graph
graph = StateGraph(MultiAgentState)

# Add nodes (agents)
graph.add_node("research", research_node)
graph.add_node("analysis", analysis_node)

# Add edges
graph.add_edge("research", "analysis")
graph.add_edge("analysis", END)

# Compile
self.graph = graph.compile()
```

**Node Functions**:
Each node is a function that:
1. Receives current state
2. Calls agent's `execute_task()`
3. Returns state updates

### 4. Error Handling Pattern

**Graceful Degradation**:
- Agents catch exceptions and add to `errors` list
- Workflow continues with partial data
- Final report includes error summary

**Retry Mechanisms**:
- Tools use `@retry_tool_call()` decorator
- Configurable retry attempts and delays
- Circuit breaker for repeated failures

**Example**:
```python
@retry_tool_call(max_attempts=3, initial_delay=1.0)
def fetch_data():
    # Tool implementation
    pass
```

---

## Adding New Features

### Adding a New Agent

1. **Create agent file** in `src/agents/`:
   ```python
   from .base_agent import BaseAgent, AgentState
   
   class MyNewAgent(BaseAgent):
       async def execute_task(self, stock_symbol, company_name, sector, context):
           # Implementation
           return AgentState(...)
   ```

2. **Register in orchestrator** (`src/graph/multi_agent_graph.py`):
   ```python
   my_agent = MyNewAgent(...)
   graph.add_node("my_agent", my_agent_node)
   ```

3. **Add to workflow**:
   ```python
   graph.add_edge("previous_node", "my_agent")
   graph.add_edge("my_agent", "next_node")
   ```

4. **Update state model** if needed:
   ```python
   class MultiAgentState(BaseModel):
       my_agent_results: Optional[Dict[str, Any]] = None
   ```

5. **Write tests** in `tests/unit/test_my_new_agent.py`

### Adding a New Tool

1. **Create tool file** in `src/tools/`:
   ```python
   from langchain_core.tools import tool
   
   @tool
   def my_new_tool(param: str) -> Dict[str, Any]:
       """Tool description for LLM."""
       # Implementation
       return {"result": "data"}
   ```

2. **Register in orchestrator**:
   ```python
   from tools.my_new_tool import my_new_tool
   
   available_tools = [
       # ... existing tools
       my_new_tool,
   ]
   ```

3. **Agents can now use it** - they'll discover it automatically via tool descriptions

4. **Write tests** in `tests/unit/test_my_new_tool.py`

### Modifying Workflow

To change the workflow order or add conditional logic:

1. **Edit** `src/graph/multi_agent_graph.py`
2. **Modify graph structure**:
   ```python
   # Add conditional edge
   def should_continue(state):
       return "next_node" if condition else "alternative_node"
   
   graph.add_conditional_edges("node", should_continue)
   ```
3. **Test thoroughly** - workflow changes affect all agents

---

## Testing Guide

### Test Structure

```
tests/
├── unit/                    # Unit tests (fast, isolated)
│   ├── test_agents/         # Agent tests
│   ├── test_tools/          # Tool tests
│   └── test_utils/           # Utility tests
└── integration/             # Integration tests (slower, end-to-end)
    └── test_full_workflow.py
```

### Running Tests

```bash
# All tests
pytest tests/

# Unit tests only
pytest tests/unit/

# Specific test file
pytest tests/unit/test_stock_data_tool.py

# With coverage
pytest tests/ --cov=src --cov-report=html

# Verbose output
pytest tests/ -v

# Stop on first failure
pytest tests/ -x
```

### Writing Tests

**Unit Test Example**:
```python
import pytest
from src.tools.stock_data_tool import get_stock_metrics

def test_get_stock_metrics():
    result = get_stock_metrics.invoke({"symbol": "RELIANCE"})
    assert result is not None
    assert "price" in result
```

**Agent Test Example**:
```python
import pytest
from src.agents.research_agent import ResearchAgent

@pytest.mark.asyncio
async def test_research_agent():
    agent = ResearchAgent(...)
    state = await agent.execute_task("RELIANCE", "Reliance", "Oil & Gas", {})
    assert state.results is not None
```

### Test Fixtures

Common fixtures in `tests/conftest.py`:
- `mock_openai_client`: Mocked OpenAI client
- `sample_state`: Sample MultiAgentState
- `sample_tools`: List of test tools

---

## Common Issues & Solutions

### Issue: "OpenAI API key not found"

**Solution**:
```bash
# Set in .env file
echo "OPENAI_API_KEY=sk-..." >> .env

# Or export as environment variable
export OPENAI_API_KEY=sk-...
```

### Issue: "ModuleNotFoundError: No module named 'langchain'"

**Solution**:
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Or reinstall in editable mode
pip install -e .
```

### Issue: "Rate limit exceeded"

**Solution**:
- Increase `REQUEST_DELAY` in `src/config.py`
- Decrease `MAX_REQUESTS_PER_MINUTE`
- Use `--skip-ai` mode (fewer API calls)

### Issue: "PDF generation fails"

**Solution**:
- Check if `reportlab` is installed: `pip install reportlab`
- Verify output directory exists: `mkdir -p reports`
- Check logs for specific error: `tail -f logs/stock_report_generator.log`

### Issue: "Agent stuck in loop"

**Solution**:
- Check loop caps in agent files (e.g., `max_iterations=5`)
- Review agent logs for repeated tool calls
- Consider using structured mode (`--skip-ai`) for debugging

### Issue: "State conflicts in parallel execution"

**Solution**:
- Ensure reducer functions are properly defined for list fields
- Check that agents don't modify state directly (return updates instead)
- Review `reduce_errors()` function in `multi_agent_graph.py`

### Issue: "Import errors when running as script"

**Solution**:
- Use `python -m src.main` instead of `python src/main.py`
- Or ensure you're in the project root directory
- Check that `src/` is in Python path

---

## Code Organization Principles

### 1. Agent Design

- **Single Responsibility**: Each agent handles one aspect of analysis
- **Autonomous Tool Selection**: Agents decide which tools to use
- **State Immutability**: Agents return state updates, don't modify directly
- **Error Handling**: Catch exceptions, add to errors list, continue

### 2. Tool Design

- **Stateless**: Tools should not maintain internal state
- **Idempotent**: Same inputs should produce same outputs
- **Descriptive**: Tool descriptions help LLMs select appropriate tools
- **Error Handling**: Return error dicts, don't raise exceptions

### 3. Configuration Management

- **Environment Variables**: Use `.env` file for secrets
- **Config Class**: Centralize config in `src/config.py`
- **Defaults**: Provide sensible defaults for all settings
- **Validation**: Validate config on startup

### 4. Logging

- **Structured Logging**: Use logging module, not print statements
- **Session Context**: Include session IDs in logs for traceability
- **Log Levels**: Use appropriate levels (DEBUG, INFO, WARNING, ERROR)
- **Log Files**: Separate files for different log types

### 5. Error Handling

- **Graceful Degradation**: Continue with partial data when possible
- **Error Collection**: Collect errors, don't fail immediately
- **User-Friendly Messages**: Provide clear error messages
- **Retry Logic**: Use retry decorators for transient failures

### 6. Type Hints

- **Always Use Type Hints**: All functions should have type hints
- **Pydantic Models**: Use Pydantic for data validation
- **Optional Types**: Use `Optional[T]` for nullable values

### 7. Documentation

- **Docstrings**: All classes and functions should have docstrings
- **Type Information**: Include parameter and return types in docstrings
- **Examples**: Provide usage examples in docstrings
- **Comments**: Explain "why", not "what" (code should be self-explanatory)

---

## Deployment & Operations

### Environment Setup

**Development**:
```bash
LOG_LEVEL=DEBUG
DEFAULT_MODEL=gpt-4o-mini
OUTPUT_DIR=reports
```

**Production**:
```bash
LOG_LEVEL=INFO
DEFAULT_MODEL=gpt-4o-mini
OUTPUT_DIR=/var/reports
MAX_REQUESTS_PER_MINUTE=30
REQUEST_DELAY=2.0
```

### Monitoring

**Logs**:
- Application logs: `logs/stock_report_generator.log`
- Prompt logs: `logs/prompts.log`
- Rotate logs regularly to prevent disk space issues

**Metrics** (if enabled):
- Prometheus metrics endpoint: `http://localhost:8000/metrics`
- Track: report generation time, success rate, API call counts

### Performance Optimization

1. **Use structured mode** (`--skip-ai`) for faster execution
2. **Parallel execution**: Analysis agents run concurrently
3. **Caching**: Implement caching for frequently accessed data
4. **Rate limiting**: Adjust based on API limits
5. **Token optimization**: Use appropriate `MAX_TOKENS` values

### Scaling Considerations

- **Batch Processing**: Process multiple stocks in parallel
- **API Rate Limits**: Distribute requests across time
- **Resource Limits**: Monitor memory and CPU usage
- **Database**: Consider storing results in database for large-scale usage

---

## Additional Resources

### Documentation Files

- **`docs/DOCUMENTATION.md`**: Comprehensive system documentation
- **`docs/AGENT_SPECIALIZATION.md`**: Detailed agent information
- **`docs/DEPLOYMENT.md`**: Deployment guide
- **`docs/IMPLEMENTATION_SUMMARY.md`**: Implementation details
- **`README.md`**: Quick start and overview

### External Dependencies

- **LangGraph**: [Documentation](https://langchain-ai.github.io/langgraph/)
- **LangChain**: [Documentation](https://python.langchain.com/)
- **OpenAI API**: [Documentation](https://platform.openai.com/docs)

### Getting Help

1. **Check logs**: `logs/stock_report_generator.log`
2. **Review documentation**: See `docs/` directory
3. **Search issues**: Check GitHub issues
4. **Create issue**: For bugs or feature requests

---

## Quick Reference

### Common Commands

```bash
# Setup
pip install -e .
cp env.example .env

# Development
pytest tests/
black src/
flake8 src/

# Running
python src/main.py RELIANCE
python src/main.py RELIANCE --skip-ai
python src/main.py RELIANCE --skip-pdf

# Debugging
tail -f logs/stock_report_generator.log
```

### Key Files

- **Entry Point**: `src/main.py`
- **Orchestrator**: `src/graph/multi_agent_graph.py`
- **Base Agent**: `src/agents/base_agent.py`
- **Config**: `src/config.py`
- **State Model**: `src/graph/multi_agent_graph.py` (MultiAgentState)

### Important Constants

- **Max Iterations**: AIResearchAgent=5, AIAnalysisAgent=12, AIReportAgent=15
- **Default Model**: gpt-4o-mini
- **Max Tokens**: 4000
- **Temperature**: 0.1

---

## Conclusion

This handoff document provides the essential information for developers to understand, modify, and extend the Stock Report Generator system. For detailed information on specific components, refer to the comprehensive documentation in the `docs/` directory.

**Key Takeaways**:
1. System uses LangGraph for multi-agent orchestration
2. Two modes: AI-powered (default) and structured workflow
3. All agents inherit from BaseAgent
4. Tools are LangChain-compatible callables
5. State is managed through Pydantic models
6. Error handling follows graceful degradation pattern

**Next Steps**:
1. Read `docs/DOCUMENTATION.md` for comprehensive system overview
2. Review `docs/AGENT_SPECIALIZATION.md` for agent details
3. Explore codebase starting with `src/main.py`
4. Run tests to understand system behavior
5. Make a small change and test it

Good luck with your development! 🚀

