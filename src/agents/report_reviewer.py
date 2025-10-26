"""
Report Reviewer Agent for combining all agent outputs into a final report.
Specializes in report synthesis, consistency checking, and final formatting.
"""

import logging
from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
import openai

try:
    # Try relative imports first (when run as module)
    from ..tools.report_formatter_tool import ReportFormatterTool, FormattedReport, ConsistencyIssue
    from ..graph.context_manager_mcp import MCPContextManager, ContextType
except ImportError:
    # Fall back to absolute imports (when run as script)
    from tools.report_formatter_tool import ReportFormatterTool, FormattedReport, ConsistencyIssue
    from graph.context_manager_mcp import MCPContextManager, ContextType

logger = logging.getLogger(__name__)

@dataclass
class FinalReport:
    """Represents the final comprehensive report."""
    stock_symbol: str
    company_name: str
    report_content: str
    sections: List[Dict[str, Any]]
    data_sources: List[str]
    confidence_score: float
    consistency_issues: List[ConsistencyIssue]
    recommendations: List[str]
    created_at: datetime

class ReportReviewerAgent:
    """
    Report Reviewer Agent for combining all agent outputs into a final report.
    
    This agent specializes in:
    - Report synthesis and integration
    - Consistency checking
    - Final formatting and presentation
    - Quality assurance
    - Recommendation generation
    """
    
    def __init__(
        self,
        agent_id: str,
        mcp_context: MCPContextManager,
        report_formatter_tool: ReportFormatterTool,
        openai_api_key: str
    ):
        """
        Initialize the Report Reviewer Agent.
        
        Args:
            agent_id: Unique identifier for this agent
            mcp_context: MCP context manager for shared memory
            report_formatter_tool: Tool for formatting final reports
            openai_api_key: OpenAI API key for reasoning
        """
        self.agent_id = agent_id
        self.mcp_context = mcp_context
        self.report_formatter_tool = report_formatter_tool
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        
    def create_final_report(
        self,
        stock_symbol: str,
        company_name: str
    ) -> FinalReport:
        """
        Create the final comprehensive report by combining all agent outputs.
        
        Args:
            stock_symbol: NSE stock symbol
            company_name: Full company name
            
        Returns:
            FinalReport object
        """
        try:
            logger.info(f"Starting final report creation for {stock_symbol}")
            
            # Step 1: Retrieve all agent outputs from MCP context
            agent_outputs = self._retrieve_agent_outputs()
            
            # Step 2: Validate and clean data
            cleaned_outputs = self._validate_and_clean_outputs(agent_outputs)
            
            # Step 3: Check for consistency issues
            consistency_issues = self._check_consistency(cleaned_outputs)
            
            # Step 4: Resolve conflicts and inconsistencies
            resolved_outputs = self._resolve_conflicts(cleaned_outputs, consistency_issues)
            
            # Step 5: Generate investment recommendations
            recommendations = self._generate_recommendations(resolved_outputs)
            
            # Step 6: Format the final report
            formatted_report = self._format_final_report(
                stock_symbol, company_name, resolved_outputs, recommendations
            )
            
            # Step 7: Perform final quality check
            quality_issues = self._perform_quality_check(formatted_report)
            
            # Step 8: Create final report object
            final_report = self._create_final_report_object(
                stock_symbol, company_name, formatted_report, 
                consistency_issues, recommendations, quality_issues
            )
            
            # Step 9: Store final report in MCP context
            self._store_final_report(final_report)
            
            logger.info(f"Completed final report creation for {stock_symbol}")
            return final_report
            
        except Exception as e:
            logger.error(f"Error creating final report: {e}")
            return self._create_fallback_report(stock_symbol, company_name)
            
    def _retrieve_agent_outputs(self) -> Dict[str, Any]:
        """Retrieve all agent outputs from MCP context."""
        try:
            outputs = {}
            
            # Get sector summary
            sector_summary = self.mcp_context.get_latest_context(ContextType.SECTOR_SUMMARY)
            if sector_summary:
                outputs["sector_summary"] = sector_summary["data"]
                
            # Get stock summary
            stock_summary = self.mcp_context.get_latest_context(ContextType.STOCK_SUMMARY)
            if stock_summary:
                outputs["stock_summary"] = stock_summary["data"]
                
            # Get management summary
            management_summary = self.mcp_context.get_latest_context(ContextType.MANAGEMENT_SUMMARY)
            if management_summary:
                outputs["management_summary"] = management_summary["data"]
                
            # Get SWOT summary
            swot_summary = self.mcp_context.get_latest_context(ContextType.SWOT_SUMMARY)
            if swot_summary:
                outputs["swot_summary"] = swot_summary["data"]
                
            logger.info(f"Retrieved outputs from {len(outputs)} agents")
            return outputs
            
        except Exception as e:
            logger.error(f"Error retrieving agent outputs: {e}")
            return {}
            
    def _validate_and_clean_outputs(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean agent outputs."""
        try:
            cleaned_outputs = {}
            
            for output_type, data in outputs.items():
                if data and isinstance(data, dict):
                    # Basic validation
                    if self._is_valid_output(data):
                        cleaned_outputs[output_type] = data
                    else:
                        logger.warning(f"Invalid output for {output_type}")
                        cleaned_outputs[output_type] = self._create_fallback_output(output_type)
                else:
                    logger.warning(f"Missing or invalid data for {output_type}")
                    cleaned_outputs[output_type] = self._create_fallback_output(output_type)
                    
            logger.info(f"Validated and cleaned {len(cleaned_outputs)} outputs")
            return cleaned_outputs
            
        except Exception as e:
            logger.error(f"Error validating outputs: {e}")
            return {}
            
    def _check_consistency(self, outputs: Dict[str, Any]) -> List[ConsistencyIssue]:
        """Check for consistency issues across agent outputs."""
        try:
            issues = []
            
            # Check for conflicting data
            if "sector_summary" in outputs and "stock_summary" in outputs:
                sector_data = outputs["sector_summary"]
                stock_data = outputs["stock_summary"]
                
                # Check for price inconsistencies
                if "current_price" in stock_data and "sector_avg_price" in sector_data:
                    price_diff = abs(stock_data["current_price"] - sector_data.get("sector_avg_price", 0))
                    if price_diff > stock_data["current_price"] * 0.5:  # 50% difference
                        issues.append(ConsistencyIssue(
                            issue_type="price_inconsistency",
                            description="Significant price difference between stock and sector data",
                            severity="medium",
                            location="price_data",
                            suggestion="Verify price data sources"
                        ))
                        
            # Check for missing critical data
            for output_type, data in outputs.items():
                if not data or len(data) < 3:
                    issues.append(ConsistencyIssue(
                        issue_type="insufficient_data",
                        description=f"Insufficient data in {output_type}",
                        severity="high",
                        location=output_type,
                        suggestion="Review agent outputs for completeness"
                    ))
                    
            logger.info(f"Found {len(issues)} consistency issues")
            return issues
            
        except Exception as e:
            logger.error(f"Error checking consistency: {e}")
            return []
            
    def _resolve_conflicts(
        self,
        outputs: Dict[str, Any],
        consistency_issues: List[ConsistencyIssue]
    ) -> Dict[str, Any]:
        """Resolve conflicts and inconsistencies in outputs."""
        try:
            resolved_outputs = outputs.copy()
            
            for issue in consistency_issues:
                if issue.issue_type == "price_inconsistency":
                    # Use stock data as primary source for price
                    if "stock_summary" in resolved_outputs:
                        stock_price = resolved_outputs["stock_summary"].get("current_price")
                        if stock_price:
                            # Update sector data to match stock price
                            if "sector_summary" in resolved_outputs:
                                resolved_outputs["sector_summary"]["sector_avg_price"] = stock_price
                                
                elif issue.issue_type == "insufficient_data":
                    # Enhance data with AI-generated content
                    output_type = issue.location
                    if output_type in resolved_outputs:
                        enhanced_data = self._enhance_data_with_ai(resolved_outputs[output_type], output_type)
                        resolved_outputs[output_type] = enhanced_data
                        
            logger.info("Resolved conflicts in agent outputs")
            return resolved_outputs
            
        except Exception as e:
            logger.error(f"Error resolving conflicts: {e}")
            return outputs
            
    def _generate_recommendations(self, outputs: Dict[str, Any]) -> List[str]:
        """Generate investment recommendations based on all agent outputs."""
        try:
            # Prepare context for AI analysis
            context = {
                "sector_analysis": outputs.get("sector_summary", {}),
                "stock_analysis": outputs.get("stock_summary", {}),
                "management_analysis": outputs.get("management_summary", {})
            }
            
            prompt = f"""
            Based on the following comprehensive analysis, provide investment recommendations:
            
            Sector Analysis: {context['sector_analysis']}
            Stock Analysis: {context['stock_analysis']}
            Management Analysis: {context['management_analysis']}
            
            Provide 3-5 specific investment recommendations focusing on:
            1. Investment thesis
            2. Key risks to monitor
            3. Growth catalysts
            4. Valuation considerations
            5. Time horizon
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a senior investment analyst providing final recommendations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.1
            )
            
            recommendations_text = response.choices[0].message.content
            
            # Parse recommendations into list
            recommendations = []
            for line in recommendations_text.split('\n'):
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or line.startswith('1.') or line.startswith('2.') or line.startswith('3.')):
                    recommendations.append(line)
                    
            if not recommendations:
                recommendations = ["Investment recommendation analysis pending"]
                
            logger.info(f"Generated {len(recommendations)} recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Investment recommendation analysis pending"]
            
    def _format_final_report(
        self,
        stock_symbol: str,
        company_name: str,
        outputs: Dict[str, Any],
        recommendations: List[str]
    ) -> FormattedReport:
        """Format the final report using the report formatter tool."""
        try:
            # Prepare data for formatter
            sector_summary = outputs.get("sector_summary", {})
            stock_summary = outputs.get("stock_summary", {})
            management_summary = outputs.get("management_summary", {})
            swot_summary = outputs.get("swot_summary", {})
            
            # Add recommendations to stock summary
            if stock_summary:
                stock_summary["recommendations"] = recommendations
                
            # Format the report
            formatted_report = self.report_formatter_tool.format_report(
                stock_symbol=stock_symbol,
                sector_summary=sector_summary,
                stock_summary=stock_summary,
                management_summary=management_summary,
                swot_summary=swot_summary,
                additional_data={"recommendations": recommendations}
            )
            
            logger.info(f"Formatted final report for {stock_symbol}")
            return formatted_report
            
        except Exception as e:
            logger.error(f"Error formatting final report: {e}")
            return self._create_fallback_formatted_report(stock_symbol, company_name)
            
    def _perform_quality_check(self, formatted_report: FormattedReport) -> List[ConsistencyIssue]:
        """Perform final quality check on the formatted report."""
        try:
            # Use the report formatter's consistency checker
            quality_issues = self.report_formatter_tool.check_consistency(formatted_report)
            
            # Add additional quality checks
            additional_issues = []
            
            # Check report length
            if formatted_report.word_count < 500:
                additional_issues.append(ConsistencyIssue(
                    issue_type="insufficient_content",
                    description="Report is too short",
                    severity="medium",
                    location="overall",
                    suggestion="Add more detailed analysis"
                ))
                
            # Check for placeholder content
            if "N/A" in formatted_report.markdown_content or "pending" in formatted_report.markdown_content.lower():
                additional_issues.append(ConsistencyIssue(
                    issue_type="placeholder_content",
                    description="Report contains placeholder content",
                    severity="high",
                    location="content",
                    suggestion="Replace placeholder content with actual analysis"
                ))
                
            all_issues = quality_issues + additional_issues
            logger.info(f"Found {len(all_issues)} quality issues")
            return all_issues
            
        except Exception as e:
            logger.error(f"Error performing quality check: {e}")
            return []
            
    def _create_final_report_object(
        self,
        stock_symbol: str,
        company_name: str,
        formatted_report: FormattedReport,
        consistency_issues: List[ConsistencyIssue],
        recommendations: List[str],
        quality_issues: List[ConsistencyIssue]
    ) -> FinalReport:
        """Create the final report object."""
        try:
            # Calculate overall confidence score
            confidence_score = self._calculate_overall_confidence(
                formatted_report, consistency_issues, quality_issues
            )
            
            # Extract data sources
            data_sources = formatted_report.metadata.get("data_sources", [])
            
            # Create sections summary
            sections = []
            for section in formatted_report.sections:
                sections.append({
                    "title": section.title,
                    "type": section.metadata.get("section_type", "unknown"),
                    "word_count": len(section.content.split())
                })
                
            final_report = FinalReport(
                stock_symbol=stock_symbol,
                company_name=company_name,
                report_content=formatted_report.markdown_content,
                sections=sections,
                data_sources=data_sources,
                confidence_score=confidence_score,
                consistency_issues=consistency_issues + quality_issues,
                recommendations=recommendations,
                created_at=datetime.now()
            )
            
            logger.info(f"Created final report object for {stock_symbol}")
            return final_report
            
        except Exception as e:
            logger.error(f"Error creating final report object: {e}")
            return self._create_fallback_report(stock_symbol, company_name)
            
    def _store_final_report(self, final_report: FinalReport) -> None:
        """Store the final report in MCP context."""
        try:
            report_data = {
                "stock_symbol": final_report.stock_symbol,
                "company_name": final_report.company_name,
                "report_content": final_report.report_content,
                "sections": final_report.sections,
                "data_sources": final_report.data_sources,
                "confidence_score": final_report.confidence_score,
                "consistency_issues": [issue.__dict__ for issue in final_report.consistency_issues],
                "recommendations": final_report.recommendations,
                "created_at": final_report.created_at.isoformat()
            }
            
            self.mcp_context.store_context(
                context_id=f"final_report_{final_report.stock_symbol}",
                context_type=ContextType.FINAL_REPORT,
                data=report_data,
                agent_id=self.agent_id,
                metadata={"report_type": "final_comprehensive_report"}
            )
            
            logger.info(f"Stored final report for {final_report.stock_symbol}")
            
        except Exception as e:
            logger.error(f"Error storing final report: {e}")
            
    # Helper methods
    def _is_valid_output(self, data: Dict[str, Any]) -> bool:
        """Check if agent output data is valid."""
        return bool(data and len(data) > 0)
        
    def _create_fallback_output(self, output_type: str) -> Dict[str, Any]:
        """Create fallback output when data is missing."""
        fallback_outputs = {
            "sector_summary": {
                "summary": "Sector analysis pending",
                "trends": ["Analysis pending"],
                "outlook": "Sector outlook pending"
            },
            "stock_summary": {
                "current_price": 0,
                "market_cap": 0,
                "pe_ratio": None,
                "performance_summary": "Stock analysis pending"
            },
            "management_summary": {
                "summary": "Management analysis pending",
                "strategic_initiatives": ["Analysis pending"],
                "management_outlook": "Management outlook pending"
            }
        }
        return fallback_outputs.get(output_type, {"summary": "Analysis pending"})
        
    def _enhance_data_with_ai(self, data: Dict[str, Any], output_type: str) -> Dict[str, Any]:
        """Enhance data using AI when insufficient."""
        try:
            prompt = f"""
            Enhance the following {output_type} data with additional insights:
            
            Current data: {data}
            
            Provide enhanced data in the same format with additional fields and insights.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a financial analyst enhancing data quality."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            # For now, return original data (in a real implementation, parse the response)
            return data
            
        except Exception as e:
            logger.error(f"Error enhancing data: {e}")
            return data
            
    def _calculate_overall_confidence(
        self,
        formatted_report: FormattedReport,
        consistency_issues: List[ConsistencyIssue],
        quality_issues: List[ConsistencyIssue]
    ) -> float:
        """Calculate overall confidence score."""
        base_score = 0.7
        
        # Reduce score based on issues
        issue_penalty = len(consistency_issues) * 0.05 + len(quality_issues) * 0.1
        
        # Increase score based on report quality
        quality_bonus = 0
        if formatted_report.word_count > 1000:
            quality_bonus += 0.1
        if len(formatted_report.sections) > 5:
            quality_bonus += 0.1
            
        final_score = base_score - issue_penalty + quality_bonus
        return max(0.0, min(1.0, final_score))
        
    def _create_fallback_formatted_report(
        self,
        stock_symbol: str,
        company_name: str
    ) -> FormattedReport:
        """Create fallback formatted report when formatting fails."""
        try:
            from ..tools.report_formatter_tool import ReportSection
        except ImportError:
            from tools.report_formatter_tool import ReportSection
        
        sections = [
            ReportSection(
                title="Executive Summary",
                content=f"Analysis for {stock_symbol} is pending due to data limitations.",
                level=1,
                order=1,
                metadata={"section_type": "executive_summary"}
            )
        ]
        
        return FormattedReport(
            title=f"Equity Research Report: {stock_symbol}",
            sections=sections,
            metadata={"stock_symbol": stock_symbol, "status": "pending"},
            markdown_content=f"# Equity Research Report: {stock_symbol}\n\nAnalysis pending.",
            word_count=10,
            creation_timestamp=datetime.now()
        )
        
    def _create_fallback_report(
        self,
        stock_symbol: str,
        company_name: str
    ) -> FinalReport:
        """Create fallback report when main process fails."""
        return FinalReport(
            stock_symbol=stock_symbol,
            company_name=company_name,
            report_content=f"# Equity Research Report: {stock_symbol}\n\nAnalysis pending due to system limitations.",
            sections=[],
            data_sources=[],
            confidence_score=0.3,
            consistency_issues=[],
            recommendations=["Analysis pending"],
            created_at=datetime.now()
        )
