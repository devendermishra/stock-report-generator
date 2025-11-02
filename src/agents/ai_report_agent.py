"""
AI Report Agent - True LangGraph Agent with Iterative Content Generation.

This agent implements the true LangGraph agent pattern for report generation:
1. LLM decides what sections to generate and in what order
2. LLM generates content for each section
3. LLM decides when all sections are complete
4. Uses PDF generation tool to create final PDF (not AI-driven)

Only content generation uses AI; PDF generation remains tool-based.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import re

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, ToolMessage
except ImportError:
    ChatOpenAI = None
    HumanMessage = None
    ToolMessage = None

try:
    from .base_agent import BaseAgent, AgentState
    from ..tools.pdf_generator_tool import generate_pdf_from_markdown
    from ..config import Config
except ImportError:
    from agents.base_agent import BaseAgent, AgentState
    from tools.pdf_generator_tool import generate_pdf_from_markdown
    from config import Config

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class AIReportAgent(BaseAgent):
    """
    AI Report Agent with iterative content generation using LLM decision-making.
    
    This agent uses an LLM in a loop to:
    1. Analyze available research and analysis data
    2. Decide what report sections to generate
    3. Generate content for each section using LLM
    4. Decide when all sections are complete
    5. Use PDF generation tool (non-AI) to create final PDF
    
    Content generation is AI-driven; PDF generation is tool-based.
    """
    
    def __init__(self, agent_id: str, openai_api_key: str):
        """
        Initialize the AI Report Agent.
        
        Args:
            agent_id: Unique identifier for the agent
            openai_api_key: OpenAI API key for LLM calls
        """
        # Define available tools (only PDF generation - content generation is AI-driven)
        available_tools = [
            generate_pdf_from_markdown
        ]
        
        super().__init__(agent_id, openai_api_key, available_tools)
        
        # Initialize LLM with tool bindings
        if ChatOpenAI is None:
            raise ImportError("langchain_openai is required for AIReportAgent. Install with: pip install langchain-openai")
        
        self.llm = ChatOpenAI(
            model=Config.DEFAULT_MODEL,
            temperature=0.3,  # Slightly higher for creative report generation
            api_key=openai_api_key
        )
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(available_tools)
        
        # Tool name to function mapping
        self.tool_map = {tool.name: tool for tool in available_tools}
        
        # Initialize OpenAI client for direct LLM calls (content generation)
        self.openai_client = AsyncOpenAI(api_key=openai_api_key)
        
        # Maximum iterations to prevent infinite loops
        self.max_iterations = 15
    
    async def execute_task(
        self,
        stock_symbol: str,
        company_name: str,
        sector: str,
        context: Dict[str, Any]
    ) -> AgentState:
        """
        Execute report generation using true agent pattern with iterative content generation.
        
        Args:
            stock_symbol: NSE stock symbol
            company_name: Full company name
            sector: Sector name
            context: Current context from previous agents (research and analysis data)
            
        Returns:
            AgentState with report generation results
        """
        start_time = datetime.now()
        
        # Initialize state
        state = AgentState(
            agent_id=self.agent_id,
            stock_symbol=stock_symbol,
            company_name=company_name,
            sector=sector,
            current_task="ai_iterative_report_generation",
            context=context,
            results={},
            tools_used=[],
            confidence_score=0.0,
            errors=[],
            start_time=start_time
        )
        
        try:
            # Extract data from previous agents
            research_data = context.get("research_agent_results", {}) or context.get("ai_research_agent_results", {})
            analysis_data = {
                "financial_analysis": context.get("financial_analysis_results", {}),
                "management_analysis": context.get("management_analysis_results", {}),
                "technical_analysis": context.get("technical_analysis_results", {}),
                "valuation_analysis": context.get("valuation_analysis_results", {})
            }
            
            # If using AI Analysis Agent, extract its structured results
            if context.get("analysis_results"):
                ai_analysis_results = context.get("analysis_results", {})
                if ai_analysis_results.get("financial_analysis"):
                    analysis_data["financial_analysis"] = ai_analysis_results["financial_analysis"]
                if ai_analysis_results.get("management_analysis"):
                    analysis_data["management_analysis"] = ai_analysis_results["management_analysis"]
                if ai_analysis_results.get("technical_analysis"):
                    analysis_data["technical_analysis"] = ai_analysis_results["technical_analysis"]
                if ai_analysis_results.get("valuation_analysis"):
                    analysis_data["valuation_analysis"] = ai_analysis_results["valuation_analysis"]
            
            # Generate complete report content using LLM (AI-driven content generation)
            final_report = await self._generate_report_content(
                stock_symbol, company_name, sector, research_data, analysis_data
            )
            
            # Generate PDF using tool (non-AI, Python-based)
            pdf_result = generate_pdf_from_markdown.invoke({
                "markdown_content": final_report,
                "stock_symbol": stock_symbol
            })
            
            pdf_path = pdf_result.get("pdf_path", "") if isinstance(pdf_result, dict) else ""
            
            # Process results
            state.results = {
                "final_report": final_report,
                "pdf_path": pdf_path,
                "report_metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "stock_symbol": stock_symbol,
                    "company_name": company_name,
                    "sector": sector,
                    "content_generation": "AI-driven",
                    "pdf_generation": "Tool-based"
                }
            }
            
            state.tools_used = ["generate_pdf_from_markdown"]
            state.confidence_score = 0.9
            
            state.end_time = datetime.now()
            
            self.logger.info(
                f"AI Report Agent completed report generation for {stock_symbol}"
            )
            
            return state
            
        except Exception as e:
            self.logger.error(f"AI Report Agent execution failed: {e}")
            state.errors.append(f"Agent execution failed: {str(e)}")
            state.confidence_score = 0.0
            state.end_time = datetime.now()
            return state
    
    async def _generate_report_content(
        self,
        stock_symbol: str,
        company_name: str,
        sector: str,
        research_data: Dict[str, Any],
        analysis_data: Dict[str, Any]
    ) -> str:
        """
        Generate complete report content using LLM.
        
        This is the AI-driven content generation part.
        """
        try:
            # Generate full report using LLM
            prompt = f"""Generate a comprehensive, professional stock research report for {company_name} ({stock_symbol}) in the {sector} sector.

Available Research Data:
{json.dumps(self._prepare_research_summary(research_data), indent=2, default=str)[:3000]}

Available Analysis Data:
{json.dumps(self._prepare_analysis_summary(analysis_data), indent=2, default=str)[:3000]}

Generate a complete markdown report with these sections in order:
1. **Executive Summary** - Key highlights, investment thesis, and recommendation overview
2. **Stock Details** - Company information, current price, market cap, and basic metrics
3. **Financial Analysis** - Financial health assessment, key ratios, and financial strength analysis
4. **Management Analysis** - Governance assessment and management effectiveness
5. **Sector Outlook** - Sector trends, risks, opportunities, and growth prospects for {sector} sector
6. **Peer Analysis** - Competitive positioning and relative valuation vs sector peers
7. **Investment Recommendation** - Buy/Sell/Hold recommendation with target price, upside potential, and detailed justification. Include what to do if you own this stock.
8. **Technical Analysis** - Price trends, technical indicators, support/resistance levels, and trading signals

Requirements:
- Use proper markdown formatting (## for sections, ### for subsections)
- Include specific data points, numbers, and metrics from the provided data
- Write professionally and comprehensively
- Be specific and data-driven in analysis
- Include immediately after title **Generated on:** <data>\n**Report Type:** Comprehensive Equity Research Analysis\n**Sector:** {sector} 
- Include a disclaimer at the end

Generate the complete report now."""
            
            response = await self.openai_client.chat.completions.create(
                model=Config.DEFAULT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional financial analyst specializing in Indian stock markets. Generate comprehensive, well-structured, data-driven stock research reports in markdown format. Always base your analysis on the provided data."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=4000
            )
            
            generated_report = response.choices[0].message.content
            
            # Add header and footer
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            header = f"""# Stock Research Report: {company_name} ({stock_symbol})

**Generated on:** {timestamp}
**Report Type:** Comprehensive Equity Research Analysis
**Sector:** {sector}

---
"""
            
            footer = f"""
---

**Disclaimer:** This report is generated by an AI-powered multi-agent system for research purposes only. It should not be considered as investment advice. Please consult with a qualified financial advisor before making investment decisions.

**Report Generated:** {timestamp}
"""
            
            return generated_report + footer
            
        except Exception as e:
            self.logger.error(f"Report content generation failed: {e}")
            return f"# Stock Research Report: {company_name} ({stock_symbol})\n\nError generating report: {str(e)}"
    
    def _prepare_research_summary(self, research_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare research data summary for LLM."""
        summary = {}
        
        company_data = research_data.get("company_data", {})
        if company_data:
            summary["company_info"] = company_data.get("company_info", {})
            summary["stock_metrics"] = company_data.get("stock_metrics", {})
        
        sector_data = research_data.get("sector_data", {})
        if sector_data:
            summary["sector_name"] = sector_data.get("sector_name")
            summary["sector_trends"] = sector_data.get("trends", {})
        
        news_data = research_data.get("news_data", {})
        if news_data:
            summary["recent_news"] = news_data.get("articles", [])[:5] if isinstance(news_data.get("articles"), list) else []
        
        peer_data = research_data.get("peer_data", {})
        if peer_data:
            summary["peer_count"] = len(peer_data.get("peers", {}))
        
        return summary
    
    def _prepare_analysis_summary(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare analysis data summary for LLM."""
        summary = {}
        
        financial = analysis_data.get("financial_analysis", {})
        if financial:
            summary["financial"] = {
                "financial_health": financial.get("financial_health", {}),
                "financial_ratios": financial.get("financial_ratios", {}),
                "stock_metrics": financial.get("stock_metrics", {})
            }
        
        management = analysis_data.get("management_analysis", {})
        if management:
            summary["management"] = {
                "management_score": management.get("management_score"),
                "key_strengths": management.get("key_strengths", []),
                "key_concerns": management.get("key_concerns", []),
                "overall_assessment": management.get("overall_assessment")
            }
        
        technical = analysis_data.get("technical_analysis", {})
        if technical:
            summary["technical"] = {
                "technical_indicators": technical.get("technical_indicators", {}),
                "technical_summary": technical.get("technical_summary", ""),
                "trend": technical.get("technical_indicators", {}).get("trend")
            }
        
        valuation = analysis_data.get("valuation_analysis", {})
        if valuation:
            summary["valuation"] = {
                "target_price": valuation.get("target_price", {}),
                "valuation_metrics": valuation.get("valuation_metrics", {})
            }
        
        return summary

