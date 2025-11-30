"""
Report Agent for synthesizing all collected and analyzed data into comprehensive reports.
This agent autonomously selects and uses tools to generate professional stock research reports.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
import json
import re

try:
    from .base_agent import BaseAgent, AgentState
    from ..tools.pdf_generator_tool import PDFGeneratorTool
    from ..tools.report_formatter_tool import ReportFormatterTool
    from ..tools.summarizer_tool import SummarizerTool
    from ..config import Config
    from ..tools.openai_call_wrapper import logged_async_chat_completion
except ImportError:
    from agents.base_agent import BaseAgent, AgentState
    from tools.pdf_generator_tool import PDFGeneratorTool
    from tools.report_formatter_tool import ReportFormatterTool
    from tools.summarizer_tool import SummarizerTool
    from config import Config
    from tools.openai_call_wrapper import logged_async_chat_completion
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

class ReportAgent(BaseAgent):
    """
    Report Agent responsible for synthesizing all data into comprehensive reports.
    
    Tasks:
    - Synthesize research and analysis data
    - Generate comprehensive stock research report
    - Create markdown and PDF formats
    - Ensure report quality and consistency
    """
    
    def __init__(self, agent_id: str, openai_api_key: str, skip_pdf: bool = False):
        """
        Initialize the Report Agent.
        
        Args:
            agent_id: Unique identifier for the agent
            openai_api_key: OpenAI API key for LLM calls
            skip_pdf: If True, skip PDF generation and only return markdown content
        """
        self.skip_pdf = skip_pdf
        # Define available tools
        available_tools = [
            PDFGeneratorTool(),
            ReportFormatterTool(),
            SummarizerTool(openai_api_key)
        ]
        
        super().__init__(agent_id, openai_api_key, available_tools)
        
        # Initialize OpenAI client for LLM-based report generation
        self.openai_client = AsyncOpenAI(api_key=openai_api_key)
        
    async def execute_task(
        self,
        stock_symbol: str,
        company_name: str,
        sector: str,
        context: Dict[str, Any]
    ) -> AgentState:
        """
        Execute report generation tasks to create comprehensive stock research report.
        
        Args:
            stock_symbol: NSE stock symbol
            company_name: Full company name
            sector: Sector name
            context: Current context from previous agents
            
        Returns:
            AgentState with report generation results
        """
        start_time = datetime.now()
        self.logger.info(f"Starting report generation for {stock_symbol} ({company_name})")
        
        # Initialize state
        state = AgentState(
            agent_id=self.agent_id,
            stock_symbol=stock_symbol,
            company_name=company_name,
            sector=sector,
            current_task="comprehensive_report_generation",
            context=context,
            results={},
            tools_used=[],
            confidence_score=0.0,
            errors=[],
            start_time=start_time
        )
        
        try:
            # Extract data from previous agents
            research_data = context.get("research_agent_results", {})
            analysis_data = context.get("analysis_agent_results", {})
            
            # Execute report generation tasks
            report_tasks = [
                self._generate_executive_summary(stock_symbol, company_name, research_data, analysis_data),
                self._generate_stock_details_section(stock_symbol, company_name, research_data, analysis_data),
                self._generate_analysis_sections(research_data, analysis_data),
                self._generate_sector_outlook_section(research_data, sector),
                self._generate_peer_analysis_section(research_data, stock_symbol, sector),
                self._generate_recommendations_section(analysis_data),
                self._generate_technical_analysis_section(analysis_data)
            ]
            
            # Execute all report generation tasks in parallel
            results = await asyncio.gather(*report_tasks, return_exceptions=True)
            
            # Process results
            executive_summary, stock_details, analysis_sections, sector_outlook, peer_analysis, recommendations, technical_analysis = results
            
            # Handle exceptions
            task_results = [executive_summary, stock_details, analysis_sections, sector_outlook, peer_analysis, recommendations, technical_analysis]
            task_names = ["executive_summary", "stock_details", "analysis_sections", "sector_outlook", "peer_analysis", "recommendations", "technical_analysis"]
            
            for i, (result, name) in enumerate(zip(task_results, task_names)):
                if isinstance(result, Exception):
                    state.errors.append(f"{name} generation failed: {str(result)}")
                    task_results[i] = {}
                else:
                    state.tools_used.extend(result.get("tools_used", []))
            
            # Compile report sections
            report_sections = {
                "executive_summary": task_results[0].get("data", {}),
                "stock_details": task_results[1].get("data", {}),
                "analysis_sections": task_results[2].get("data", {}),
                "sector_outlook": task_results[3].get("data", {}),
                "peer_analysis": task_results[4].get("data", {}),
                "recommendations": task_results[5].get("data", {}),
                "technical_analysis": task_results[6].get("data", {})
            }
            
            # Generate final report (pass sector to ensure it's available in report)
            final_report = await self._generate_final_report(stock_symbol, company_name, sector, report_sections)
            
            # Generate PDF unless skip_pdf is True
            pdf_path = ""
            if not self.skip_pdf:
                pdf_path = await self._generate_pdf_report(stock_symbol, company_name, final_report)
            else:
                self.logger.info("Skipping PDF generation as requested")
            
            # Compile results
            state.results = {
                "report_sections": report_sections,
                "final_report": final_report,
                "pdf_path": pdf_path,
                "report_metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "stock_symbol": stock_symbol,
                    "company_name": company_name,
                    "sector": sector,
                    "sections_count": len([s for s in report_sections.values() if s]),
                    "pdf_generation": "Skipped" if self.skip_pdf else "Completed"
                }
            }
            
            # Calculate confidence score
            state.confidence_score = self.calculate_confidence_score(
                state.results, state.tools_used, state.errors
            )
            
            # Update context
            state.context = self.update_context(context, state.results, self.agent_id)
            
            state.end_time = datetime.now()
            duration = (state.end_time - state.start_time).total_seconds()
            self.log_execution(state, duration)
            
            return state
            
        except Exception as e:
            self.logger.error(f"Report generation execution failed: {e}")
            state.errors.append(f"Report generation execution failed: {str(e)}")
            state.end_time = datetime.now()
            state.confidence_score = 0.0
            return state
    
    async def _generate_executive_summary(
        self,
        stock_symbol: str,
        company_name: str,
        research_data: Dict[str, Any],
        analysis_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate executive summary section."""
        try:
            self.logger.info(f"Generating executive summary for {stock_symbol}")
            tools_used = []
            
            # Extract key information
            company_info = research_data.get("company_data", {}).get("company_info", {})
            stock_metrics = research_data.get("company_data", {}).get("stock_metrics", {})
            financial_analysis = analysis_data.get("financial_analysis", {})
            valuation_analysis = analysis_data.get("valuation_analysis", {})
            
            # Generate executive summary
            summary = self._create_executive_summary(
                stock_symbol, company_name, company_info, stock_metrics,
                financial_analysis, valuation_analysis
            )
            
            return {
                "data": {
                    "summary": summary,
                    "key_highlights": self._extract_key_highlights(research_data, analysis_data)
                },
                "tools_used": tools_used
            }
            
        except Exception as e:
            self.logger.error(f"Executive summary generation failed: {e}")
            return {"data": {}, "tools_used": [], "error": str(e)}
    
    async def _generate_stock_details_section(
        self,
        stock_symbol: str,
        company_name: str,
        research_data: Dict[str, Any],
        analysis_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate stock details section."""
        try:
            self.logger.info(f"Generating stock details for {stock_symbol}")
            tools_used = []
            
            company_info = research_data.get("company_data", {}).get("company_info", {})
            stock_metrics = research_data.get("company_data", {}).get("stock_metrics", {})
            
            stock_details = {
                "company_name": company_info.get("company_name", company_name),
                "symbol": stock_symbol,
                "sector": company_info.get("sector", "N/A"),
                "industry": company_info.get("industry", "N/A"),
                "description": company_info.get("description", "No description available"),
                "website": company_info.get("website", ""),
                "employees": company_info.get("employees", "N/A"),
                "location": f"{company_info.get('city', '')}, {company_info.get('state', '')}, {company_info.get('country', 'India')}",
                "current_price": stock_metrics.get("current_price", 0),
                "market_cap": stock_metrics.get("market_cap", 0),
                "currency": stock_metrics.get("currency", "INR"),
                "exchange": stock_metrics.get("exchange", "NSE")
            }
            
            return {
                "data": stock_details,
                "tools_used": tools_used
            }
            
        except Exception as e:
            self.logger.error(f"Stock details generation failed: {e}")
            return {"data": {}, "tools_used": [], "error": str(e)}
    
    async def _generate_analysis_sections(
        self,
        research_data: Dict[str, Any],
        analysis_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate financial and management analysis sections."""
        try:
            self.logger.info("Generating analysis sections")
            tools_used = []
            
            financial_analysis = analysis_data.get("financial_analysis", {})
            management_analysis = analysis_data.get("management_analysis", {})
            
            analysis_sections = {
                "financial_analysis": {
                    "summary": financial_analysis.get("financial_health", {}).get("overall_assessment", "N/A"),
                    "health_score": financial_analysis.get("financial_health", {}).get("health_score", 0),
                    "key_ratios": financial_analysis.get("financial_ratios", {}),
                    "health_factors": financial_analysis.get("financial_health", {}).get("health_factors", [])
                },
                "management_analysis": {
                    "summary": management_analysis.get("management_assessment", {}).get("key_factors", []),
                    "governance_score": management_analysis.get("management_assessment", {}).get("governance_score", 0),
                    "recent_developments": management_analysis.get("management_assessment", {}).get("recent_developments", [])
                }
            }
            
            return {
                "data": analysis_sections,
                "tools_used": tools_used
            }
            
        except Exception as e:
            self.logger.error(f"Analysis sections generation failed: {e}")
            return {"data": {}, "tools_used": [], "error": str(e)}
    
    async def _generate_sector_outlook_section(self, research_data: Dict[str, Any], sector: str = None) -> Dict[str, Any]:
        """Generate sector outlook section using LLM."""
        try:
            self.logger.info(f"Generating sector outlook section for sector: {sector}")
            tools_used = []
            
            sector_data = research_data.get("sector_data", {})
            # Prioritize sector parameter (most reliable), then sector_data, then company_info
            sector_name = (
                sector or  # Highest priority - from workflow parameter
                sector_data.get("sector_name") or 
                research_data.get("company_data", {}).get("company_info", {}).get("sector") or 
                "N/A"
            )
            
            if sector_name == "N/A" and sector:
                sector_name = sector  # Use provided sector even if not found in data
                self.logger.warning(f"Sector not found in data, using provided sector parameter: {sector}")
            
            sector_news = sector_data.get("sector_news", {})
            market_trends = sector_data.get("market_trends", {})
            
            # Prepare data for LLM analysis
            sector_info = {
                "sector_name": sector_name,
                "news_count": len(sector_news.get("results", [])),
                "trends_count": len(market_trends.get("results", [])),
                "recent_news": sector_news.get("results", [])[:5],  # Top 5 news items
                "market_trends": market_trends.get("results", [])[:5]  # Top 5 trends
            }
            
            # Generate sector outlook using LLM with explicit sector emphasis
            prompt = f"""You are analyzing the OUTLOOK for the {sector_name.upper()} SECTOR in India.

**CRITICAL**: Focus EXCLUSIVELY on the {sector_name.upper()} sector. Do NOT discuss other sectors like financial, insurance, or banking sectors unless they are directly related to {sector_name}.

<sector_data>
{json.dumps(sector_info, indent=2, default=str)}
</sector_data>

Analyze the outlook specifically for the {sector_name.upper()} sector based on the provided data. Generate a concise analysis report (2-3 paragraphs) that:
- Discusses trends, growth drivers, and challenges specific to the {sector_name.upper()} sector
- Is based on the sector news and market trends provided
- Does NOT mention unrelated sectors

Return JSON:
{{
    "outlook": "<comprehensive {sector_name} sector analysis and outlook (2-3 paragraphs), focusing ONLY on {sector_name}>",
    "key_trends": ["trend1 specific to {sector_name}", "trend2 specific to {sector_name}", "trend3"],
    "risk_factors": ["risk1 for {sector_name} sector", "risk2 for {sector_name} sector"],
    "growth_prospects": "<brief assessment specific to {sector_name} sector>"
}}
"""
            
            # Use logged wrapper for prompt logging
            messages = [
                {
                    "role": "system",
                    "content": f"You are a financial sector analyst specializing in Indian stock markets. You analyze sector data and provide comprehensive outlook reports. Focus exclusively on the specified sector and do not confuse it with other sectors."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            response = await logged_async_chat_completion(
                client=self.openai_client,
                model=Config.DEFAULT_MODEL,
                messages=messages,
                temperature=0.2,  # Lower temperature for more focused output
                max_tokens=1000,
                agent_name="ReportAgent"
            )
            
            # Parse response
            response_text = response.choices[0].message.content
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            if json_match:
                outlook_data = json.loads(json_match.group(0))
                sector_outlook = {
                    "sector_name": sector_name,
                    "outlook": outlook_data.get("outlook", "Sector analysis based on recent news and market trends"),
                    "key_trends": outlook_data.get("key_trends", []),
                    "risk_factors": outlook_data.get("risk_factors", []),
                    "growth_prospects": outlook_data.get("growth_prospects", "Moderate growth expected")
                }
            else:
                # Fallback
                sector_outlook = {
                    "sector_name": sector_name,
                    "outlook": "Sector analysis based on recent news and market trends",
                    "key_trends": [],
                    "risk_factors": [],
                    "growth_prospects": "Analysis completed"
                }
            
            return {
                "data": sector_outlook,
                "tools_used": tools_used
            }
            
        except Exception as e:
            self.logger.error(f"Sector outlook generation failed: {e}")
            return {"data": {}, "tools_used": [], "error": str(e)}
    
    async def _generate_peer_analysis_section(self, research_data: Dict[str, Any], stock_symbol: str = None, sector: str = None) -> Dict[str, Any]:
        """Generate peer analysis section using LLM with 4 selected peers."""
        try:
            self.logger.info(f"Generating peer analysis section for {stock_symbol}")
            tools_used = []
            
            peer_data = research_data.get("peer_data", {})
            all_peers = peer_data.get("peers", {})
            
            # Prioritize sector parameter (most reliable), then peer_data, then company_info
            peer_sector = (
                sector or  # Highest priority - from workflow parameter
                peer_data.get("sector") or 
                research_data.get("company_data", {}).get("company_info", {}).get("sector") or 
                "N/A"
            )
            
            if peer_sector == "N/A" and sector:
                peer_sector = sector  # Use provided sector even if not found in data
                self.logger.warning(f"Sector not found in peer data, using provided sector parameter: {sector}")
            
            # Get current stock metrics for comparison
            company_data = research_data.get("company_data", {})
            current_stock_metrics = company_data.get("stock_metrics", {})
            current_price = current_stock_metrics.get("current_price", 0) or 0
            current_market_cap = current_stock_metrics.get("market_cap", 0) or 0
            
            if not all_peers:
                self.logger.warning(f"No peer data found for {stock_symbol}")
                return {
                    "data": {
                        "peer_count": 0,
                        "sector": peer_sector,
                        "peer_comparison": f"No peer data available for comparison in {peer_sector} sector."
                    },
                    "tools_used": tools_used
                }
            
            self.logger.info(f"Found {len(all_peers)} peers, selecting up to 4 for comparison")
            
            # Select 4 peers: 2 top peers (by market cap) and 1 peer near to stock (similar market cap)
            selected_peers = self._select_peer_comparison_set(all_peers, current_market_cap, current_price)
            
            if not selected_peers:
                self.logger.warning("No peers selected for comparison")
                return {
                    "data": {
                        "peer_count": 0,
                        "sector": peer_sector,
                        "peer_comparison": "Unable to select peers for comparison due to missing market cap data."
                    },
                    "tools_used": tools_used
                }
            
            # Prepare peer comparison data for LLM
            peer_comparison_data = {}
            for symbol, data in selected_peers.items():
                try:
                    company_info = data.get("company_info", {}) or {}
                    stock_metrics = data.get("stock_metrics", {}) or {}
                    
                    # Extract values safely with defaults
                    peer_comparison_data[symbol] = {
                        "company_name": company_info.get("company_name") or company_info.get("name") or symbol,
                        "current_price": stock_metrics.get("current_price") or stock_metrics.get("price") or 0,
                        "market_cap": stock_metrics.get("market_cap") or stock_metrics.get("market_capitalization") or 0,
                        "pe_ratio": stock_metrics.get("pe_ratio") or stock_metrics.get("pe") or stock_metrics.get("price_to_earnings") or 0,
                        "pb_ratio": stock_metrics.get("pb_ratio") or stock_metrics.get("pb") or stock_metrics.get("price_to_book") or 0,
                        "dividend_yield": stock_metrics.get("dividend_yield") or stock_metrics.get("dividend") or 0
                    }
                except Exception as e:
                    self.logger.warning(f"Failed to extract data for peer {symbol}: {e}")
                    continue
            
            if not peer_comparison_data:
                self.logger.warning("No valid peer comparison data extracted")
                return {
                    "data": {
                        "peer_count": len(selected_peers),
                        "sector": peer_sector,
                        "peer_comparison": "Unable to generate detailed peer comparison due to incomplete data.",
                        "selected_peers": list(selected_peers.keys())
                    },
                    "tools_used": tools_used
                }
            
            # Add current stock for comparison
            company_info = company_data.get("company_info", {}) or {}
            current_stock_info = {
                "company_name": company_info.get("company_name") or company_info.get("name") or stock_symbol or "Current Stock",
                "current_price": current_price,
                "market_cap": current_market_cap,
                "pe_ratio": current_stock_metrics.get("pe_ratio") or current_stock_metrics.get("pe") or current_stock_metrics.get("price_to_earnings") or 0,
                "pb_ratio": current_stock_metrics.get("pb_ratio") or current_stock_metrics.get("pb") or current_stock_metrics.get("price_to_book") or 0,
                "dividend_yield": current_stock_metrics.get("dividend_yield") or current_stock_metrics.get("dividend") or 0
            }
            
            # Generate peer comparison using LLM
            prompt = f"""Compare the current stock with {len(selected_peers)} peers from the {peer_sector} sector. Generate a comprehensive peer analysis report (3-4 paragraphs).

<current_stock>
{json.dumps(current_stock_info, indent=2, default=str)}
</current_stock>

<peers>
{json.dumps(peer_comparison_data, indent=2, default=str)}
</peers>

**Instructions:**
- Analyze valuation metrics (P/E, P/B ratios) compared to peers
- Assess competitive positioning within the {peer_sector} sector
- Identify key differentiators and relative strengths/weaknesses
- Provide actionable insights for investment decision-making

Return JSON:
{{
    "comparison_report": "<comprehensive peer comparison analysis (3-4 paragraphs) focusing on {peer_sector} sector>",
    "relative_valuation": "<assessment of valuation vs peers (undervalued/overvalued/fairly valued)>",
    "competitive_position": "<assessment of competitive position within {peer_sector} sector>",
    "key_differentiators": ["differentiator1", "differentiator2", "differentiator3"]
}}
"""
            
            # Use logged wrapper for prompt logging
            messages = [
                {
                    "role": "system",
                    "content": f"You are a financial analyst specializing in peer comparison analysis. Compare stocks within the {peer_sector} sector and provide detailed, actionable insights. Focus on relative valuation, competitive positioning, and investment implications."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            response = await logged_async_chat_completion(
                client=self.openai_client,
                model=Config.DEFAULT_MODEL,
                messages=messages,
                temperature=0.2,  # Lower temperature for more focused analysis
                max_tokens=1200,
                agent_name="ReportAgent"
            )
            
            # Parse response
            response_text = response.choices[0].message.content
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            if json_match:
                comparison_data = json.loads(json_match.group(0))
                peer_comparison = comparison_data.get("comparison_report", "")
                
                # Build full peer comparison text
                comparison_text = comparison_data.get("comparison_report", "")
                if comparison_data.get("relative_valuation"):
                    comparison_text += f"\n\n**Relative Valuation:** {comparison_data.get('relative_valuation')}"
                if comparison_data.get("competitive_position"):
                    comparison_text += f"\n\n**Competitive Position:** {comparison_data.get('competitive_position')}"
                if comparison_data.get("key_differentiators"):
                    comparison_text += "\n\n**Key Differentiators:**"
                    for diff in comparison_data.get("key_differentiators", []):
                        comparison_text += f"\n- {diff}"
            else:
                # Fallback to simple comparison if LLM parsing fails
                self.logger.warning("Failed to parse LLM response for peer comparison, using fallback")
                comparison_text = self._create_peer_comparison(selected_peers)
                
                # Try to add basic analysis even in fallback
                if selected_peers:
                    avg_pe = sum(peer_comparison_data.get(s, {}).get("pe_ratio", 0) for s in selected_peers.keys()) / len(selected_peers)
                    current_pe = current_stock_info.get("pe_ratio", 0)
                    if current_pe > 0 and avg_pe > 0:
                        if current_pe < avg_pe * 0.9:
                            comparison_text += f"\n\n**Valuation Note:** The stock appears undervalued with P/E ratio of {current_pe:.2f} compared to peer average of {avg_pe:.2f}."
                        elif current_pe > avg_pe * 1.1:
                            comparison_text += f"\n\n**Valuation Note:** The stock appears overvalued with P/E ratio of {current_pe:.2f} compared to peer average of {avg_pe:.2f}."
                        else:
                            comparison_text += f"\n\n**Valuation Note:** The stock is fairly valued with P/E ratio of {current_pe:.2f} compared to peer average of {avg_pe:.2f}."
            
            peer_analysis = {
                "peer_count": len(selected_peers),
                "sector": peer_sector,
                "peer_comparison": comparison_text,
                "selected_peers": list(selected_peers.keys())
            }
            
            return {
                "data": peer_analysis,
                "tools_used": tools_used
            }
            
        except Exception as e:
            self.logger.error(f"Peer analysis generation failed: {e}")
            return {"data": {}, "tools_used": [], "error": str(e)}
    
    def _select_peer_comparison_set(
        self,
        all_peers: Dict[str, Any],
        current_market_cap: float,
        current_price: float
    ) -> Dict[str, Any]:
        """
        Select 4 peers: 2 top peers (by market cap) and 1 peer near to stock (similar market cap).
        
        Args:
            all_peers: Dictionary of all peer data
            current_market_cap: Market cap of current stock
            current_price: Current price of stock
            
        Returns:
            Dictionary with selected peers (up to 4)
        """
        if not all_peers:
            return {}
        
        # Sort peers by market cap
        peer_list = []
        for symbol, data in all_peers.items():
            metrics = data.get("stock_metrics", {})
            if not metrics:
                continue
            market_cap = metrics.get("market_cap", 0) or 0
            peer_list.append((symbol, data, market_cap))
        
        if not peer_list:
            return {}
        
        # Sort by market cap (descending)
        peer_list.sort(key=lambda x: x[2], reverse=True)
        
        selected = {}
        target_count = min(4, len(peer_list))  # Don't exceed available peers
        
        # Get top 2 peers by market cap
        for i, (symbol, data, _) in enumerate(peer_list[:2]):
            selected[symbol] = data
        
        # Find peer closest to current stock's market cap (excluding already selected)
        if current_market_cap > 0 and len(selected) < target_count:
            remaining_peers = [(s, d, m) for s, d, m in peer_list if s not in selected]
            if remaining_peers:
                closest_peer = min(remaining_peers, key=lambda x: abs(x[2] - current_market_cap) if x[2] > 0 else float('inf'))
                selected[closest_peer[0]] = closest_peer[1]
        
        # Fill up to 4 total if we have more peers available
        if len(selected) < target_count:
            remaining_peers = [(s, d, m) for s, d, m in peer_list if s not in selected]
            while len(selected) < target_count and remaining_peers:
                next_peer = remaining_peers.pop(0)
                selected[next_peer[0]] = next_peer[1]
        
        return selected
    
    async def _generate_recommendations_section(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate buy/sell/hold recommendations section."""
        try:
            self.logger.info("Generating recommendations section")
            tools_used = []
            
            valuation_analysis = analysis_data.get("valuation_analysis", {})
            target_price = valuation_analysis.get("target_price", {})
            financial_analysis = analysis_data.get("financial_analysis", {})
            
            recommendations = {
                "recommendation": target_price.get("recommendation", "HOLD"),
                "target_price": target_price.get("target_price", 0),
                "current_price": target_price.get("current_price", 0),
                "upside_potential": target_price.get("upside_potential", 0),
                "justification": self._create_recommendation_justification(
                    target_price, financial_analysis
                )
            }
            
            return {
                "data": recommendations,
                "tools_used": tools_used
            }
            
        except Exception as e:
            self.logger.error(f"Recommendations generation failed: {e}")
            return {"data": {}, "tools_used": [], "error": str(e)}
    
    async def _generate_technical_analysis_section(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate technical analysis section."""
        try:
            self.logger.info("Generating technical analysis section")
            tools_used = []
            
            technical_analysis = analysis_data.get("technical_analysis", {})
            technical_indicators = technical_analysis.get("technical_indicators", {})
            technical_summary = technical_analysis.get("technical_summary", "")
            
            tech_analysis = {
                "indicators": technical_indicators,
                "summary": technical_summary,
                "trend": technical_indicators.get("trend", "N/A"),
                "volume_trend": technical_indicators.get("volume_trend", "N/A"),
                "upside_potential": technical_indicators.get("upside_potential", 0),
                "downside_risk": technical_indicators.get("downside_risk", 0)
            }
            
            return {
                "data": tech_analysis,
                "tools_used": tools_used
            }
            
        except Exception as e:
            self.logger.error(f"Technical analysis generation failed: {e}")
            return {"data": {}, "tools_used": [], "error": str(e)}
    
    async def _generate_final_report(
        self,
        stock_symbol: str,
        company_name: str,
        sector: str,
        report_sections: Dict[str, Any]
    ) -> str:
        """Generate the final comprehensive report in markdown format."""
        try:
            self.logger.info(f"Generating final report for {stock_symbol}")
            
            # Create markdown report
            markdown_content = self._create_markdown_report(stock_symbol, company_name, sector, report_sections)
            
            return markdown_content
            
        except Exception as e:
            self.logger.error(f"Final report generation failed: {e}")
            return f"# Error generating report for {stock_symbol}\n\nError: {str(e)}"
    
    async def _generate_pdf_report(
        self,
        stock_symbol: str,
        company_name: str,
        markdown_content: str
    ) -> str:
        """Generate PDF version of the report."""
        try:
            self.logger.info(f"Generating PDF report for {stock_symbol}")
            tools_used = []
            
            # Use PDF generator tool
            pdf_generator = PDFGeneratorTool()
            pdf_path = pdf_generator.generate_pdf(
                markdown_content=markdown_content,
                stock_symbol=stock_symbol
            )
            tools_used.append("PDFGeneratorTool")
            
            return pdf_path
            
        except Exception as e:
            self.logger.error(f"PDF generation failed: {e}")
            return ""
    
    def _create_executive_summary(
        self,
        stock_symbol: str,
        company_name: str,
        company_info: Dict[str, Any],
        stock_metrics: Dict[str, Any],
        financial_analysis: Dict[str, Any],
        valuation_analysis: Dict[str, Any]
    ) -> str:
        """Create executive summary."""
        try:
            current_price = stock_metrics.get("current_price", 0)
            market_cap = stock_metrics.get("market_cap", 0)
            pe_ratio = stock_metrics.get("pe_ratio", 0)
            
            target_price = valuation_analysis.get("target_price", {})
            recommendation = target_price.get("recommendation", "HOLD")
            target_price_value = target_price.get("target_price", current_price)
            
            financial_health = financial_analysis.get("financial_health", {})
            health_assessment = financial_health.get("overall_assessment", "N/A")
            
            summary = f"""
## Executive Summary

**{company_name} ({stock_symbol})** is a {company_info.get('sector', 'N/A')} sector company trading at ₹{current_price:.2f} with a market capitalization of ₹{market_cap:,.0f}.

**Key Metrics:**
- Current Price: ₹{current_price:.2f}
- Market Cap: ₹{market_cap:,.0f}
- P/E Ratio: {pe_ratio:.2f}
- Target Price: ₹{target_price_value:.2f}
- Recommendation: **{recommendation}**

**Financial Health:** {health_assessment}

**Investment Thesis:** Based on comprehensive analysis of financial metrics, management effectiveness, and sector trends, {company_name} presents a {recommendation.lower()} opportunity with {target_price.get('upside_potential', 0):.1f}% upside potential to the target price.
"""
            return summary.strip()
            
        except Exception as e:
            self.logger.error(f"Executive summary creation failed: {e}")
            return f"Executive summary for {company_name} ({stock_symbol}) - Analysis completed."
    
    def _extract_key_highlights(
        self,
        research_data: Dict[str, Any],
        analysis_data: Dict[str, Any]
    ) -> List[str]:
        """Extract key highlights from research and analysis data."""
        highlights = []
        
        try:
            # Financial highlights
            financial_analysis = analysis_data.get("financial_analysis", {})
            if financial_analysis.get("financial_health"):
                health_score = financial_analysis["financial_health"].get("health_score", 0)
                highlights.append(f"Financial health score: {health_score}/100")
            
            # Management highlights
            management_analysis = analysis_data.get("management_analysis", {})
            if management_analysis.get("management_assessment"):
                governance_score = management_analysis["management_assessment"].get("governance_score", 0)
                highlights.append(f"Management governance score: {governance_score}/100")
            
            # Valuation highlights
            valuation_analysis = analysis_data.get("valuation_analysis", {})
            if valuation_analysis.get("target_price"):
                target = valuation_analysis["target_price"]
                highlights.append(f"Target price: ₹{target.get('target_price', 'N/A')} ({target.get('recommendation', 'N/A')})")
            
            # Technical highlights
            technical_analysis = analysis_data.get("technical_analysis", {})
            if technical_analysis.get("technical_indicators"):
                tech = technical_analysis["technical_indicators"]
                highlights.append(f"Technical trend: {tech.get('trend', 'N/A')}")
            
        except Exception as e:
            self.logger.error(f"Key highlights extraction failed: {e}")
            highlights.append("Analysis completed with comprehensive data collection")
        
        return highlights
    
    def _create_peer_comparison(self, peers: Dict[str, Any]) -> str:
        """Create peer comparison analysis."""
        try:
            if not peers:
                return "No peer data available for comparison."
            
            comparison_parts = []
            comparison_parts.append("**Peer Comparison:**")
            
            for symbol, data in list(peers.items())[:3]:  # Top 3 peers
                company_info = data.get("company_info", {})
                stock_metrics = data.get("stock_metrics", {})
                
                name = company_info.get("company_name", symbol)
                price = stock_metrics.get("current_price", 0)
                pe_ratio = stock_metrics.get("pe_ratio", 0)
                
                comparison_parts.append(f"- **{name} ({symbol}):** ₹{price:.2f}, P/E: {pe_ratio:.2f}")
            
            return "\n".join(comparison_parts)
            
        except Exception as e:
            self.logger.error(f"Peer comparison creation failed: {e}")
            return "Peer comparison analysis completed."
    
    def _create_recommendation_justification(
        self,
        target_price: Dict[str, Any],
        financial_analysis: Dict[str, Any]
    ) -> str:
        """Create recommendation justification."""
        try:
            recommendation = target_price.get("recommendation", "HOLD")
            upside_potential = target_price.get("upside_potential", 0)
            
            financial_health = financial_analysis.get("financial_health", {})
            health_score = financial_health.get("health_score", 0)
            
            justification_parts = []
            
            if recommendation == "BUY":
                justification_parts.append("**Buy Recommendation:** Based on attractive valuation and strong fundamentals.")
            elif recommendation == "SELL":
                justification_parts.append("**Sell Recommendation:** Based on overvaluation concerns and weak fundamentals.")
            else:
                justification_parts.append("**Hold Recommendation:** Based on fair valuation and mixed fundamentals.")
            
            justification_parts.append(f"- Upside potential: {upside_potential:.1f}%")
            justification_parts.append(f"- Financial health score: {health_score}/100")
            
            if health_score >= 70:
                justification_parts.append("- Strong financial position supports investment thesis")
            elif health_score >= 50:
                justification_parts.append("- Moderate financial position with some concerns")
            else:
                justification_parts.append("- Weak financial position requires careful consideration")
            
            # Add guidance for existing holders
            justification_parts.append("")
            justification_parts.append("**What to Do If You Have This Stock:**")
            
            if recommendation == "BUY":
                justification_parts.append("- **Continue Holding:** Stock shows strong fundamentals and positive outlook")
                justification_parts.append("- **Consider Adding:** If portfolio allows, consider averaging up on dips")
                justification_parts.append("- **Set Stop-Loss:** Protect gains with stop-loss at 10-15% below current price or key support")
                justification_parts.append("- **Take Partial Profits:** If stock has run up significantly, consider partial profit booking while maintaining core position")
                justification_parts.append("- **Monitor Catalysts:** Watch for key catalysts that could drive further upside")
            elif recommendation == "SELL":
                justification_parts.append("- **Consider Exiting:** Given identified concerns, consider reducing or exiting position")
                justification_parts.append("- **Staggered Exit:** If holding substantial position, use staggered exit to avoid market impact")
                justification_parts.append("- **Review Tax Implications:** Consider tax consequences before selling, especially for long-term holdings")
                justification_parts.append("- **Set Stop-Loss:** If holding temporarily, set tight stop-loss to protect capital")
                justification_parts.append("- **Review Thesis:** Reassess original investment thesis - if it no longer holds, exit")
                justification_parts.append("- **Consider Alternatives:** Look for better opportunities in same sector or other sectors")
            else:  # HOLD
                justification_parts.append("- **Maintain Position:** Hold current position without major changes")
                justification_parts.append("- **Avoid Averaging:** Given mixed signals, avoid aggressive averaging down unless high conviction")
                justification_parts.append("- **Regular Monitoring:** Keep close watch on key factors for any changes")
                justification_parts.append("- **Partial Profit Taking:** If sitting on gains, consider partial profits while keeping core holdings")
                justification_parts.append("- **Rebalance if Needed:** If position has become too large in portfolio, consider rebalancing")
                justification_parts.append("- **Stay Disciplined:** Stick to investment plan and avoid emotional decisions based on volatility")
            
            return "\n".join(justification_parts)
            
        except Exception as e:
            self.logger.error(f"Recommendation justification creation failed: {e}")
            return "Recommendation based on comprehensive analysis of financial metrics and market conditions."
    
    def _create_markdown_report(
        self,
        stock_symbol: str,
        company_name: str,
        sector: str,
        report_sections: Dict[str, Any]
    ) -> str:
        """Create comprehensive markdown report."""
        try:
            # Get current timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Start building the report
            report_parts = []
            
            # Title and header
            report_parts.append(f"# Stock Research Report: {company_name} ({stock_symbol})")
            report_parts.append(f"**Generated on:** {timestamp}")
            report_parts.append(f"**Report Type:** Comprehensive Equity Research Analysis")
            report_parts.append("---")
            
            # Executive Summary
            executive_summary = report_sections.get("executive_summary", {})
            if executive_summary.get("summary"):
                report_parts.append(executive_summary["summary"])
                report_parts.append("")
            
            # Stock Details
            stock_details = report_sections.get("stock_details", {})
            if stock_details:
                report_parts.append("## Stock Details")
                report_parts.append(f"**Company Name:** {stock_details.get('company_name', 'N/A')}")
                report_parts.append(f"**Symbol:** {stock_details.get('symbol', 'N/A')}")
                report_parts.append(f"**Sector:** {stock_details.get('sector', 'N/A')}")
                report_parts.append(f"**Industry:** {stock_details.get('industry', 'N/A')}")
                report_parts.append(f"**Current Price:** ₹{stock_details.get('current_price', 0):.2f}")
                report_parts.append(f"**Market Cap:** ₹{stock_details.get('market_cap', 0):,.0f}")
                report_parts.append(f"**Exchange:** {stock_details.get('exchange', 'NSE')}")
                report_parts.append("")
                
                if stock_details.get('description'):
                    report_parts.append("**Company Description:**")
                    report_parts.append(stock_details['description'])
                    report_parts.append("")
            
            # Financial Analysis
            analysis_sections = report_sections.get("analysis_sections", {})
            financial_analysis = analysis_sections.get("financial_analysis", {})
            if financial_analysis:
                report_parts.append("## Financial Analysis")
                report_parts.append(f"**Overall Assessment:** {financial_analysis.get('summary', 'N/A')}")
                report_parts.append(f"**Health Score:** {financial_analysis.get('health_score', 0)}/100")
                
                if financial_analysis.get('health_factors'):
                    report_parts.append("**Key Factors:**")
                    for factor in financial_analysis['health_factors']:
                        report_parts.append(f"- {factor}")
                report_parts.append("")
            
            # Management Analysis
            management_analysis = analysis_sections.get("management_analysis", {})
            if management_analysis:
                report_parts.append("## Management Analysis")
                report_parts.append(f"**Governance Score:** {management_analysis.get('governance_score', 0)}/100")
                
                if management_analysis.get('summary'):
                    report_parts.append("**Key Factors:**")
                    for factor in management_analysis['summary']:
                        report_parts.append(f"- {factor}")
                report_parts.append("")
            
            # Sector Outlook
            sector_outlook = report_sections.get("sector_outlook", {})
            # Get sector from multiple sources - prioritize provided sector parameter
            displayed_sector = (
                sector or  # Highest priority - from workflow parameter
                sector_outlook.get('sector_name') or
                stock_details.get('sector') or
                'N/A'
            )
            
            if sector_outlook:
                report_parts.append("## Sector Outlook")
                report_parts.append(f"**Sector:** {displayed_sector}")
                report_parts.append("")
                report_parts.append(sector_outlook.get('outlook', 'N/A'))
                report_parts.append("")
                
                if sector_outlook.get('key_trends'):
                    report_parts.append("**Key Trends:**")
                    for trend in sector_outlook['key_trends']:
                        report_parts.append(f"- {trend}")
                    report_parts.append("")
                
                if sector_outlook.get('risk_factors'):
                    report_parts.append("**Risk Factors:**")
                    for risk in sector_outlook['risk_factors']:
                        report_parts.append(f"- {risk}")
                    report_parts.append("")
                
                if sector_outlook.get('growth_prospects'):
                    report_parts.append(f"**Growth Prospects:** {sector_outlook.get('growth_prospects')}")
                    report_parts.append("")
            
            # Peer Analysis
            peer_analysis = report_sections.get("peer_analysis", {})
            if peer_analysis:
                report_parts.append("## Peer Analysis")
                report_parts.append(f"**Peers Analyzed:** {peer_analysis.get('peer_count', 0)}")
                if peer_analysis.get('selected_peers'):
                    report_parts.append(f"**Selected Peers:** {', '.join(peer_analysis['selected_peers'])}")
                report_parts.append("")
                if peer_analysis.get('peer_comparison'):
                    report_parts.append(peer_analysis['peer_comparison'])
                    report_parts.append("")
            
            # Recommendations
            recommendations = report_sections.get("recommendations", {})
            if recommendations:
                report_parts.append("## Investment Recommendation")
                report_parts.append(f"**Recommendation:** {recommendations.get('recommendation', 'HOLD')}")
                report_parts.append(f"**Target Price:** ₹{recommendations.get('target_price', 0):.2f}")
                report_parts.append(f"**Current Price:** ₹{recommendations.get('current_price', 0):.2f}")
                report_parts.append(f"**Upside Potential:** {recommendations.get('upside_potential', 0):.1f}%")
                report_parts.append("")
                
                if recommendations.get('justification'):
                    report_parts.append("**Justification:**")
                    report_parts.append(recommendations['justification'])
                    report_parts.append("")
            
            # Technical Analysis
            technical_analysis = report_sections.get("technical_analysis", {})
            if technical_analysis:
                report_parts.append("## Technical Analysis")
                report_parts.append(f"**Trend:** {technical_analysis.get('trend', 'N/A')}")
                report_parts.append(f"**Volume Trend:** {technical_analysis.get('volume_trend', 'N/A')}")
                report_parts.append(f"**Upside Potential:** {technical_analysis.get('upside_potential', 0):.1f}%")
                report_parts.append(f"**Downside Risk:** {technical_analysis.get('downside_risk', 0):.1f}%")
                report_parts.append("")
                
                if technical_analysis.get('summary'):
                    report_parts.append("**Technical Summary:**")
                    report_parts.append(technical_analysis['summary'])
                    report_parts.append("")
            
            # Footer
            report_parts.append("---")
            report_parts.append("**Disclaimer:** This report is generated by an AI-powered multi-agent system for research purposes only. It should not be considered as investment advice. Please consult with a qualified financial advisor before making investment decisions.")
            report_parts.append(f"**Report Generated:** {timestamp}")
            
            return "\n".join(report_parts)
            
        except Exception as e:
            self.logger.error(f"Markdown report creation failed: {e}")
            return f"# Error generating report for {stock_symbol}\n\nError: {str(e)}"
