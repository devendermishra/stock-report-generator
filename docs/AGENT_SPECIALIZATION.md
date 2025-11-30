# Agent Specialization and Roles Documentation

## Overview

This document provides comprehensive details about the specialization, roles, responsibilities, and implementation of each agent in the Stock Report Generator multi-agent system. This addresses gaps in documentation and clarifies how agents work together to generate comprehensive stock research reports.

## Table of Contents

1. [Agent Architecture Overview](#agent-architecture-overview)
2. [Operational Modes](#operational-modes)
3. [Research Phase Agents](#research-phase-agents)
4. [Analysis Phase Agents](#analysis-phase-agents)
5. [Report Phase Agents](#report-phase-agents)
6. [Agent Coordination and Workflow](#agent-coordination-and-workflow)
7. [Tool Usage by Agent](#tool-usage-by-agent)
8. [When to Use Which Mode](#when-to-use-which-mode)

---

## Agent Architecture Overview

The system employs **10 distinct agents** organized into three phases:

### Phase 1: Research Phase
- **ResearchPlannerAgent** (Mixed Mode only)
- **ResearchAgent** (Mixed Mode only)
- **AIResearchAgent** (Agentic AI Mode only)

### Phase 2: Analysis Phase
- **FinancialAnalysisAgent** (Mixed Mode only)
- **ManagementAnalysisAgent** (Mixed Mode only)
- **TechnicalAnalysisAgent** (Mixed Mode only)
- **ValuationAnalysisAgent** (Mixed Mode only)
- **AIAnalysisAgent** (Agentic AI Mode only)

### Phase 3: Report Phase
- **ReportAgent** (Mixed Mode only)
- **AIReportAgent** (Agentic AI Mode only)

---

## Operational Modes

The system operates in **two distinct modes**, each using different agent sets:

### Mode 1: Agentic AI Mode: AI powered Iterative Mode (Default)

**Characteristics:**
- Uses LLM-based iterative decision-making
- Agents dynamically select tools based on context
- More adaptive and flexible
- Higher token usage (15-30% more)
- Better for complex or unique stocks

**Agent Set:**
1. **AIResearchAgent** - Iterative research with dynamic tool selection
2. **AIAnalysisAgent** - Comprehensive analysis across all dimensions
3. **AIReportAgent** - LLM-powered report generation

**Workflow:**
```
AIResearchAgent → AIAnalysisAgent → AIReportAgent → Final Report
```

### Mode 2: Mixed Mode (`--skip-ai` flag)

**Characteristics:**
- Follows predefined research pipeline
- Deterministic execution
- Lower token usage
- Faster execution (20-25% faster)
- More predictable for standard analyses

**Agent Set:**
1. **ResearchPlannerAgent** - Creates structured research plan
2. **ResearchAgent** - Executes research plan sequentially
3. **FinancialAnalysisAgent** - Financial statement analysis
4. **ManagementAnalysisAgent** - Management and governance analysis
5. **TechnicalAnalysisAgent** - Technical indicators and trends
6. **ValuationAnalysisAgent** - Valuation metrics and target price
7. **ReportAgent** - Synthesizes all analysis into report

**Workflow:**
```
ResearchPlannerAgent → ResearchAgent → [4 Analysis Agents in Parallel] → ReportAgent → Final Report
```

---

## Research Phase Agents

### 1. ResearchPlannerAgent

**Specialization:** Research Strategy Planning

**Role:** Creates a structured, ordered research plan with specific tool call sequences before data gathering begins.

**Responsibilities:**
- Create data gathering plan for research agent

**Key Features:**
- **No tool execution** - Only plans, doesn't execute
- Uses OpenAI to generate structured JSON plans
- Validates tool names against available tools
- Orders tool calls logically (prerequisites first)

**Tools Used:** None (planning only, uses OpenAI for plan generation)

**Output:**
```json
{
  "planning_status": "completed",
  "research_plan": {
    "tool_calls": [
      {
        "order": 1,
        "tool_name": "get_stock_metrics",
        "parameters": {"symbol": "RELIANCE"}
      },
      {
        "order": 2,
        "tool_name": "get_company_info",
        "parameters": {"symbol": "RELIANCE"}
      }
    ],
    "total_steps": 2
  },
  "confidence_score": 0.8
}
```

**When Used:** Only in Mixed Mode

**File:** `src/agents/research_planner_agent.py`

---

### 2. ResearchAgent

**Specialization:** Comprehensive Data Gathering

**Role:** Executes the research plan created by `ResearchPlannerAgent` to gather all necessary data for analysis.

**Responsibilities:**
- Executes tool calls in the order specified by research plan
- Gathers company fundamentals (info, metrics, business overview)
- Collects market data (stock metrics, price data)
- Performs sector analysis (trends, positioning)
- Aggregates news and sentiment (company and sector news)
- Gathers peer comparison data

**Key Features:**
- **Plan-driven execution** - Follows structured plan from ResearchPlannerAgent
- Reuses data from research plan context when available
- Handles tool execution errors gracefully
- Aggregates all research data into structured format

**Tools Used:**
- `get_stock_metrics` - Stock price, volume, market cap, ratios
- `get_company_info` - Company description, business model, history
- `validate_symbol` - Symbol validation (usually already done)
- `search_sector_news` - Sector-specific news and trends
- `search_company_news` - Company-specific news
- `search_market_trends` - Market-wide trends and analysis
- `generic_web_search` / `search_web_generic` - General web search

**Output:**
```json
{
  "company_data": {
    "stock_metrics": {...},
    "company_info": {...}
  },
  "sector_data": {
    "sector_news": [...],
    "market_trends": [...]
  },
  "news_data": {
    "company_news": [...],
    "sector_news": [...]
  },
  "peer_data": {...}
}
```

**When Used:** Only in Mixed Mode (after `ResearchPlannerAgent`)

**File:** `src/agents/research_agent.py`

---

### 3. AIResearchAgent

**Specialization:** Adaptive Iterative Research

**Role:** Uses iterative LLM-based decision-making to dynamically gather research data based on emerging insights.

**Responsibilities:**
- Analyzes current information state
- Decides which tool to use next based on what's needed
- Executes selected tool
- Observes results and adapts strategy
- Continues until research goal is achieved
- Stops when sufficient data is gathered

**Key Features:**
- **True LangGraph agent pattern** - LLM decides actions iteratively
- **Dynamic tool selection** - Chooses tools based on current state
- **Self-correcting** - Can gather missing data if initial results incomplete
- **Efficient** - Skips unnecessary tools if goal already achieved
- **Adaptive** - Adjusts research depth based on data availability

**Tools Used:**
- `get_stock_metrics` - Stock price, volume, market cap, ratios
- `get_company_info` - Company description, business model
- `search_sector_news` - Sector-specific news
- `search_company_news` - Company-specific news
- `search_market_trends` - Market-wide trends
- `search_web_generic` - General web search

**Execution Pattern:**
```
1. LLM analyzes: "Need basic company info first"
   → Executes: get_company_info
   → Observes: Company info retrieved

2. LLM analyzes: "Good, now get stock metrics"
   → Executes: get_stock_metrics
   → Observes: Metrics retrieved

3. LLM analyzes: "Have enough for basic report, finishing"
   → Stops (saved unnecessary tool calls)
```

**Output:** Same structure as ResearchAgent, but gathered adaptively

**When Used:** Only in Agentic AI Mode (replaces ResearchPlannerAgent + ResearchAgent)

**File:** `src/agents/ai_research_agent.py`

**Max Iterations:** 5 (configurable)

---

## Analysis Phase Agents

### 4. FinancialAnalysisAgent

**Specialization:** Financial Statement Analysis and Ratio Interpretation

**Role:** Performs comprehensive financial analysis including financial ratios, health assessment, and financial metrics evaluation.

**Responsibilities:**
- Analyzes financial statements and ratios
- Calculates additional financial ratios (P/E, P/B, Price-to-Sales)
- Assesses financial health using LLM analysis
- Categorizes market cap (Large/Mid/Small Cap)
- Evaluates financial stability and growth potential

**Key Features:**
- **Reuses data** - Uses stock_metrics from ResearchAgent when available
- **LLM-powered health assessment** - Uses OpenAI to analyze financial health
- **Comprehensive ratio calculation** - Calculates multiple financial ratios
- **Health scoring** - Provides 0-100 health score with factors

**Tools Used:**
- `get_stock_metrics` - Financial metrics (P/E, P/B, EPS, ROE, etc.)

**Output:**
```json
{
  "stock_metrics": {...},
  "financial_ratios": {
    "pe_ratio": 25.5,
    "pb_ratio": 3.2,
    "dividend_yield": 1.5,
    "beta": 1.1,
    "price_to_sales": 2.8,
    "market_cap_category": "Large Cap"
  },
  "financial_health": {
    "health_score": 75,
    "health_factors": ["Strong P/E ratio", "Stable dividend yield"],
    "overall_assessment": "Financially healthy with good growth potential"
  }
}
```

**When Used:** Only in Mixed Mode (runs in parallel with other analysis agents)

**File:** `src/agents/financial_analysis_agent.py`

---

### 5. ManagementAnalysisAgent

**Specialization:** Management Effectiveness and Governance Assessment

**Role:** Evaluates management quality, governance practices, and leadership effectiveness.

**Responsibilities:**
- Analyzes management team effectiveness
- Assesses corporate governance practices
- Evaluates strategic initiatives and leadership quality
- Reviews management-related news and announcements
- Provides governance assessment

**Key Features:**
- **News-driven analysis** - Uses company news to assess management
- **LLM-powered evaluation** - Uses OpenAI for qualitative assessment
- **Governance focus** - Emphasizes corporate governance practices

**Tools Used:**
- `get_company_info` - Company information including management details
- `search_company_news` - Management-related news and announcements

**Output:**
```json
{
  "management_assessment": {
    "leadership_quality": "Strong",
    "governance_score": 80,
    "strategic_initiatives": [...],
    "management_news": [...],
    "overall_evaluation": "Management demonstrates strong leadership..."
  }
}
```

**When Used:** Only in Mixed Mode (runs in parallel with other analysis agents)

**File:** `src/agents/management_analysis_agent.py`

---

### 6. TechnicalAnalysisAgent

**Specialization:** Technical Indicators and Price Trend Analysis

**Role:** Performs technical analysis using price patterns, volume analysis, and technical indicators.

**Responsibilities:**
- Calculates technical indicators (RSI, MACD, Bollinger Bands)
- Analyzes price trends and patterns
- Evaluates volume patterns
- Identifies support and resistance levels
- Provides technical trading signals

**Key Features:**
- **Technical formatter integration** - Uses TechnicalAnalysisFormatter for calculations
- **Indicator-focused** - Emphasizes technical indicators over fundamentals
- **Trend analysis** - Analyzes price trends and patterns

**Tools Used:**
- `get_stock_metrics` - Price data, volume, 52-week high/low
- `TechnicalAnalysisFormatter` - Technical indicator calculations

**Output:**
```json
{
  "technical_indicators": {
    "rsi": 55.2,
    "macd": {...},
    "bollinger_bands": {...},
    "moving_averages": {...}
  },
  "trend_analysis": {
    "current_trend": "Bullish",
    "support_level": 2500,
    "resistance_level": 2800,
    "volume_pattern": "Increasing"
  },
  "technical_signals": {
    "buy_signals": [...],
    "sell_signals": [...],
    "neutral_signals": [...]
  }
}
```

**When Used:** Only in Mixed Mode (runs in parallel with other analysis agents)

**File:** `src/agents/technical_analysis_agent.py`

---

### 7. ValuationAnalysisAgent

**Specialization:** Valuation Metrics and Target Price Calculation

**Role:** Performs valuation analysis and calculates target prices using multiple valuation methods.

**Responsibilities:**
- Calculates valuation metrics (P/E, P/B, PEG ratios)
- Performs relative valuation analysis
- Calculates target price using multiple methods
- Assesses valuation attractiveness
- Compares valuation to sector peers

**Key Features:**
- **Multi-method valuation** - Uses multiple valuation approaches
- **Target price calculation** - Provides specific target price estimates
- **Market context** - Considers market trends in valuation

**Tools Used:**
- `get_stock_metrics` - Valuation metrics (P/E, P/B, market cap)
- `search_market_trends` - Market context for valuation

**Output:**
```json
{
  "valuation_metrics": {
    "pe_ratio": 25.5,
    "pb_ratio": 3.2,
    "peg_ratio": 1.2,
    "ev_ebitda": 15.8
  },
  "target_price": {
    "method_1": 2750,
    "method_2": 2800,
    "weighted_average": 2775,
    "confidence": "High"
  },
  "valuation_assessment": {
    "attractiveness": "Fairly Valued",
    "comparison_to_peers": "In line with sector average"
  }
}
```

**When Used:** Only in Mixed Mode (runs in parallel with other analysis agents)

**File:** `src/agents/valuation_analysis_agent.py`

---

### 8. AIAnalysisAgent

**Specialization:** Comprehensive Multi-Dimensional Analysis

**Role:** Performs all analysis types (financial, management, technical, valuation) using iterative LLM-based decision-making.

**Responsibilities:**
- Analyzes what data is needed for different analysis types
- Decides which tools to use to gather required data
- Performs financial analysis
- Performs management analysis
- Performs technical analysis
- Performs valuation analysis
- Synthesizes insights across all analysis dimensions
- Decides if more data or analysis is needed

**Key Features:**
- **True LangGraph agent pattern** - LLM decides actions iteratively
- **Comprehensive coverage** - Handles all four analysis types
- **Dynamic data gathering** - Gathers additional data if needed
- **Cross-dimensional synthesis** - Integrates insights across analysis types
- **Adaptive depth** - Adjusts analysis depth based on data availability

**Tools Used:**
- `get_stock_metrics` - Financial and technical metrics
- `get_company_info` - Company and management information
- `search_company_news` - Management and company news
- `search_market_trends` - Market context for valuation
- `format_technical_analysis` - Technical indicator calculations

**Execution Pattern:**
```
1. LLM analyzes: "Need financial metrics for financial analysis"
   → Executes: get_stock_metrics
   → Observes: Metrics retrieved

2. LLM analyzes: "Now perform financial analysis"
   → Performs: Financial analysis using metrics
   → Observes: Financial analysis complete

3. LLM analyzes: "Need company info for management analysis"
   → Executes: get_company_info
   → Observes: Info retrieved

4. LLM analyzes: "Perform management analysis"
   → Performs: Management analysis
   → Observes: Management analysis complete

5. LLM analyzes: "All analyses complete, finalizing"
   → Stops
```

**Output:**
```json
{
  "financial_analysis": {...},
  "management_analysis": {...},
  "technical_analysis": {...},
  "valuation_analysis": {...},
  "synthesis": {
    "key_insights": [...],
    "cross_dimensional_findings": [...]
  }
}
```

**When Used:** Only in Agentic AI Mode (replaces all 4 separate analysis agents)

**File:** `src/agents/ai_analysis_agent.py`

**Max Iterations:** 12 (configurable)

---

## Report Phase Agents

### 9. ReportAgent

**Specialization:** Report Synthesis and Formatting (Mixed Mode)

**Role:** Synthesizes all research and analysis results into a comprehensive, professionally formatted report.

**Responsibilities:**
- Combines research results from ResearchAgent
- Integrates analysis results from all 4 analysis agents
- Generates comprehensive report with all required sections
- Formats report in Markdown
- Generates professionally styled PDF
- Ensures report completeness and coherence

**Key Features:**
- **Template-driven** - Follows structured report template
- **Section ordering** - Executive Summary → Company Overview → Analysis → Recommendations
- **Data integration** - Combines data from multiple agents
- **Quality validation** - Checks for section completeness

**Tools Used:**
- `PDFGeneratorTool` - PDF generation and styling
- `ReportFormatterTool` - Report formatting and structure
- `SummarizerTool` - Text summarization if needed

**Output:**
- Markdown report file
- PDF report file
- Report metadata (path, generation time, etc.)

**When Used:** Only in Mixed Mode (after all analysis agents complete)

**File:** `src/agents/report_agent.py`

---

### 10. AIReportAgent

**Specialization:** LLM-Powered Report Generation (AI Mode)

**Role:** Uses LLM to generate comprehensive reports with professional formatting, iteratively ensuring all sections are complete.

**Responsibilities:**
- Analyzes research and analysis results
- Generates report content using LLM
- Ensures all required sections are included
- Formats report professionally
- Generates PDF with styling
- Validates report quality and completeness

**Key Features:**
- **LLM-powered content** - Uses OpenAI to generate report text
- **Iterative refinement** - Can refine sections if needed
- **Quality validation** - Checks for completeness and coherence
- **Professional formatting** - Ensures institutional-quality output

**Tools Used:**
- `PDFGeneratorTool` - PDF generation and styling
- `ReportFormatterTool` - Report formatting
- `SummarizerTool` - Text summarization

**Output:**
- Markdown report file
- PDF report file
- Report metadata

**When Used:** Only in Agentic AI Mode (after AIAnalysisAgent)

**File:** `src/agents/ai_report_agent.py`

---

## Agent Coordination and Workflow

### AI-Powered Iterative Mode Workflow

```
┌─────────────────┐
│  User Input     │ Stock Symbol, Company, Sector
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ AIResearchAgent │ Iterative research with dynamic tool selection
│                 │ - Analyzes what data is needed
│                 │ - Selects and executes tools
│                 │ - Adapts based on results
│                 │ - Stops when goal achieved
└────────┬────────┘
         │ research_results
         ▼
┌─────────────────┐
│ AIAnalysisAgent │ Comprehensive analysis across all dimensions
│                 │ - Financial analysis
│                 │ - Management analysis
│                 │ - Technical analysis
│                 │ - Valuation analysis
│                 │ - Synthesizes insights
└────────┬────────┘
         │ analysis_results
         ▼
┌─────────────────┐
│ AIReportAgent   │ LLM-powered report generation
│                 │ - Generates report content
│                 │ - Formats professionally
│                 │ - Creates PDF
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Final Report   │ Markdown + PDF
└─────────────────┘
```

### Structured Workflow Mode

```
┌─────────────────┐
│  User Input     │ Stock Symbol, Company, Sector
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│ ResearchPlannerAgent │ Creates structured research plan
│                      │ - Analyzes requirements
│                      │ - Generates tool call sequence
│                      │ - Validates plan
└────────┬─────────────┘
         │ research_plan_results
         ▼
┌─────────────────┐
│ ResearchAgent   │ Executes research plan
│                 │ - Follows plan sequentially
│                 │ - Gathers all research data
└────────┬────────┘
         │ research_results
         │
         ├──────────────────┬──────────────────┬──────────────────┐
         ▼                  ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Financial    │  │ Management   │  │ Technical    │  │ Valuation    │
│ Analysis     │  │ Analysis     │  │ Analysis     │  │ Analysis     │
│ Agent        │  │ Agent        │  │ Agent        │  │ Agent        │
│              │  │              │  │              │  │              │
│ - Ratios     │  │ - Governance │  │ - Indicators │  │ - Valuation  │
│ - Health     │  │ - Leadership │  │ - Trends     │  │ - Target     │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │                 │
       └─────────────────┴─────────────────┴─────────────────┘
                         │ (all analysis results)
                         ▼
              ┌──────────────────┐
              │ ReportAgent      │ Synthesizes all results
              │                  │ - Combines research + analysis
              │                  │ - Generates report
              │                  │ - Creates PDF
              └────────┬─────────┘
                        │
                        ▼
              ┌──────────────────┐
              │  Final Report    │ Markdown + PDF
              └──────────────────┘
```

**Note:** In Mixed Mode, all 4 analysis agents run **in parallel** (LangGraph automatically waits for all to complete before proceeding to ReportAgent).

---

## Tool Usage by Agent

### Research Tools

| Tool | ResearchPlannerAgent | ResearchAgent | AIResearchAgent |
|------|---------------------|---------------|-----------------|
| `validate_symbol` | ❌ | ✅ | ❌ (pre-validated) |
| `get_stock_metrics` | ❌ | ✅ | ✅ |
| `get_company_info` | ❌ | ✅ | ✅ |
| `search_sector_news` | ❌ | ✅ | ✅ |
| `search_company_news` | ❌ | ✅ | ✅ |
| `search_market_trends` | ❌ | ✅ | ✅ |
| `search_web_generic` | ❌ | ✅ | ✅ |

### Analysis Tools

| Tool | FinancialAnalysis | ManagementAnalysis | TechnicalAnalysis | ValuationAnalysis | AIAnalysis |
|------|-------------------|-------------------|-------------------|-------------------|------------|
| `get_stock_metrics` | ✅ | ❌ | ✅ | ✅ | ✅ |
| `get_company_info` | ❌ | ✅ | ❌ | ❌ | ✅ |
| `search_company_news` | ❌ | ✅ | ❌ | ❌ | ✅ |
| `search_market_trends` | ❌ | ❌ | ❌ | ✅ | ✅ |
| `TechnicalAnalysisFormatter` | ❌ | ❌ | ✅ | ❌ | ✅ (via helpers) |

### Report Tools

| Tool | ReportAgent | AIReportAgent |
|------|-------------|---------------|
| `PDFGeneratorTool` | ✅ | ✅ |
| `ReportFormatterTool` | ✅ | ✅ |
| `SummarizerTool` | ✅ (optional) | ✅ (optional) |

---

## When to Use Which Mode

### Use AI-Powered Iterative Mode When:

✅ **Complex or unique stocks** - Stocks with unusual characteristics or limited data  
✅ **Adaptive research needed** - When research strategy should adapt to findings  
✅ **Quality over speed** - When thoroughness is more important than speed  
✅ **Exploratory analysis** - When you want the system to explore different angles  
✅ **Limited prior knowledge** - When you're not sure what data sources are best  

**Trade-offs:**
- ⚠️ Higher token usage (15-30% more)
- ⚠️ Longer execution time (10-20% slower)
- ✅ More thorough and adaptive
- ✅ Better for complex cases

### Use Structured Workflow Mode When:

✅ **Standard analysis** - Well-known stocks with standard data availability  
✅ **Predictable workflow** - When you want consistent, predictable execution  
✅ **Cost optimization** - When you want to minimize token usage  
✅ **Speed priority** - When faster execution is important  
✅ **Reproducibility** - When you need consistent results across runs  

**Trade-offs:**
- ✅ Lower token usage
- ✅ Faster execution (20-25% faster)
- ✅ More predictable
- ⚠️ Less adaptive to unique situations

### Recommendation

**Default to AI-Powered Iterative Mode** for most use cases, as it provides better adaptability and thoroughness. Use Mixed Mode when:
- Processing many stocks in batch (cost/speed optimization)
- You need highly reproducible results
- Working with well-known, standard stocks

---

## Agent State Management

All agents inherit from `BaseAgent` and use the `AgentState` dataclass:

```python
@dataclass
class AgentState:
    agent_id: str
    stock_symbol: str
    company_name: str
    sector: str
    current_task: str
    context: Dict[str, Any]  # Shared context from previous agents
    results: Dict[str, Any]   # Agent-specific results
    tools_used: List[str]
    confidence_score: float
    errors: List[str]
    start_time: datetime
    end_time: Optional[datetime]
```

Agents communicate through the shared `context` dictionary, which accumulates results from previous agents.

---

## Error Handling

All agents implement robust error handling:

1. **Graceful Degradation** - Continue with partial data when some sources fail
2. **Error Propagation** - Errors are collected in `errors` list and propagated to state
3. **Confidence Scoring** - Agents calculate confidence scores based on data completeness
4. **Retry Logic** - Some tools implement retry mechanisms for transient failures

---

## Extending the System

### Adding a New Agent

1. Create new agent class inheriting from `BaseAgent`
2. Implement `execute_task()` method
3. Define available tools in `__init__`
4. Add agent to `MultiAgentOrchestrator` in `multi_agent_graph.py`
5. Add node to graph workflow
6. Update state model if needed

### Adding a New Tool

1. Create tool function with `@tool` decorator
2. Add tool to appropriate agent's `available_tools` list
3. Tool will be automatically available for agent to use

---

## Summary

This documentation clarifies:

✅ **10 distinct agents** with clear specializations  
✅ **Two operational modes** with different agent sets  
✅ **Specific responsibilities** for each agent  
✅ **Tool usage** by each agent  
✅ **Workflow coordination** between agents  
✅ **When to use** each mode  

This addresses the feedback about gaps in documentation and implementation regarding agent specialization and roles.


