"""
Stock Researcher Agent for retrieving and analyzing stock data.
Specializes in financial metrics, technical analysis, and valuation.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import openai

try:
    # Try relative imports first (when run as module)
    from ..tools.stock_data_tool import StockDataTool, StockMetrics, PriceData
    from ..graph.context_manager_mcp import MCPContextManager, ContextType
except ImportError:
    # Fall back to absolute imports (when run as script)
    from tools.stock_data_tool import StockDataTool, StockMetrics, PriceData
    from graph.context_manager_mcp import MCPContextManager, ContextType

logger = logging.getLogger(__name__)

@dataclass
class StockAnalysis:
    """Represents the output of stock analysis."""
    symbol: str
    company_name: str
    current_price: float
    market_cap: float
    financial_metrics: Dict[str, Any]
    technical_analysis: Dict[str, Any]
    valuation_metrics: Dict[str, Any]
    performance_summary: str
    investment_rating: str
    target_price: Optional[float]
    confidence_score: float

class StockResearcherAgent:
    """
    Stock Researcher Agent for retrieving and analyzing stock data.
    
    This agent specializes in:
    - Financial metrics analysis
    - Technical analysis
    - Valuation assessment
    - Performance evaluation
    - Investment rating
    """
    
    def __init__(
        self,
        agent_id: str,
        mcp_context: MCPContextManager,
        stock_data_tool: StockDataTool,
        openai_api_key: str
    ):
        """
        Initialize the Stock Researcher Agent.
        
        Args:
            agent_id: Unique identifier for this agent
            mcp_context: MCP context manager for shared memory
            stock_data_tool: Stock data tool for data retrieval
            openai_api_key: OpenAI API key for reasoning
        """
        self.agent_id = agent_id
        self.mcp_context = mcp_context
        self.stock_data_tool = stock_data_tool
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        
    async def analyze_stock(
        self,
        stock_symbol: str,
        company_name: str
    ) -> StockAnalysis:
        """
        Perform comprehensive stock analysis.
        
        Args:
            stock_symbol: NSE stock symbol
            company_name: Full company name
            
        Returns:
            StockAnalysis object
        """
        try:
            logger.info(f"Starting stock analysis for {stock_symbol}")
            
            # Step 1: Get basic stock metrics
            stock_metrics = self._get_stock_metrics(stock_symbol)
            
            # Step 2: Get historical data
            historical_data = self._get_historical_data(stock_symbol)
            
            # Step 3: Calculate technical indicators
            technical_analysis = self._calculate_technical_analysis(historical_data)
            
            # Step 4: Analyze financial metrics
            financial_analysis = self._analyze_financial_metrics(stock_metrics)
            
            # Step 5: Perform valuation analysis
            valuation_analysis = self._perform_valuation_analysis(stock_metrics, technical_analysis)
            
            # Step 6: Generate investment rating
            investment_rating = self._generate_investment_rating(
                stock_metrics, financial_analysis, technical_analysis, valuation_analysis
            )
            
            # Step 7: Synthesize all analysis
            analysis = self._synthesize_stock_analysis(
                stock_symbol, company_name, stock_metrics, financial_analysis,
                technical_analysis, valuation_analysis, investment_rating
            )
            
            # Step 8: Store results in MCP context
            self._store_analysis_results(analysis)
            
            logger.info(f"Completed stock analysis for {stock_symbol}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in stock analysis: {e}")
            return self._create_fallback_analysis(stock_symbol)
            
    def _get_stock_metrics(self, stock_symbol: str) -> Optional[StockMetrics]:
        """Get comprehensive stock metrics."""
        try:
            metrics = self.stock_data_tool.get_stock_metrics(stock_symbol)
            if metrics:
                logger.info(f"Retrieved metrics for {stock_symbol}")
            else:
                logger.warning(f"No metrics available for {stock_symbol}")
            return metrics
        except Exception as e:
            logger.error(f"Error getting stock metrics: {e}")
            return None
            
    def _get_historical_data(self, stock_symbol: str) -> List[PriceData]:
        """Get historical price data."""
        try:
            historical_data = self.stock_data_tool.get_historical_data(
                symbol=stock_symbol,
                period="1y",
                interval="1d"
            )
            logger.info(f"Retrieved {len(historical_data)} historical data points for {stock_symbol}")
            return historical_data
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return []
            
    def _calculate_technical_analysis(self, historical_data: List[PriceData]) -> Dict[str, Any]:
        """Calculate technical analysis indicators."""
        try:
            if not historical_data:
                return {}
                
            technical_indicators = self.stock_data_tool.calculate_technical_indicators(historical_data)
            
            # Add additional technical analysis
            analysis = {
                "indicators": technical_indicators,
                "trend_analysis": self._analyze_trend(historical_data),
                "support_resistance": self._find_support_resistance(historical_data),
                "momentum": self._calculate_momentum(historical_data)
            }
            
            logger.info("Calculated technical analysis indicators")
            return analysis
            
        except Exception as e:
            logger.error(f"Error calculating technical analysis: {e}")
            return {}
            
    def _analyze_financial_metrics(self, stock_metrics: Optional[StockMetrics]) -> Dict[str, Any]:
        """Analyze financial metrics and ratios."""
        try:
            if not stock_metrics:
                return {}
                
            analysis = {
                "valuation_ratios": {
                    "pe_ratio": stock_metrics.pe_ratio,
                    "pb_ratio": stock_metrics.pb_ratio,
                    "dividend_yield": stock_metrics.dividend_yield
                },
                "risk_metrics": {
                    "beta": stock_metrics.beta,
                    "volatility": self._calculate_volatility(stock_metrics)
                },
                "performance_metrics": {
                    "current_price": stock_metrics.current_price,
                    "change_percent": stock_metrics.change_percent,
                    "high_52w": stock_metrics.high_52w,
                    "low_52w": stock_metrics.low_52w
                },
                "trading_metrics": {
                    "volume": stock_metrics.volume,
                    "avg_volume": stock_metrics.avg_volume,
                    "volume_ratio": stock_metrics.volume / stock_metrics.avg_volume if stock_metrics.avg_volume > 0 else 0
                }
            }
            
            logger.info("Analyzed financial metrics")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing financial metrics: {e}")
            return {}
            
    def _perform_valuation_analysis(
        self,
        stock_metrics: Optional[StockMetrics],
        technical_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform valuation analysis."""
        try:
            if not stock_metrics:
                return {}
                
            valuation = {
                "current_valuation": {
                    "market_cap": stock_metrics.market_cap,
                    "pe_ratio": stock_metrics.pe_ratio,
                    "pb_ratio": stock_metrics.pb_ratio
                },
                "valuation_assessment": self._assess_valuation(stock_metrics),
                "price_targets": self._calculate_price_targets(stock_metrics, technical_analysis),
                "upside_potential": self._calculate_upside_potential(stock_metrics)
            }
            
            logger.info("Performed valuation analysis")
            return valuation
            
        except Exception as e:
            logger.error(f"Error performing valuation analysis: {e}")
            return {}
            
    def _generate_investment_rating(
        self,
        stock_metrics: Optional[StockMetrics],
        financial_analysis: Dict[str, Any],
        technical_analysis: Dict[str, Any],
        valuation_analysis: Dict[str, Any]
    ) -> str:
        """Generate investment rating using AI reasoning."""
        try:
            # Prepare context for AI analysis
            context = {
                "metrics": stock_metrics.__dict__ if stock_metrics else {},
                "financial": financial_analysis,
                "technical": technical_analysis,
                "valuation": valuation_analysis
            }
            
            prompt = f"""
            Based on the following stock analysis data, provide an investment rating:
            
            Stock Metrics: {context['metrics']}
            Financial Analysis: {context['financial']}
            Technical Analysis: {context['technical']}
            Valuation Analysis: {context['valuation']}
            
            Provide a rating from: STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL
            Also provide a brief justification for the rating.
            
            Format your response as:
            RATING: [RATING]
            JUSTIFICATION: [Brief explanation]
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a senior equity analyst with expertise in Indian markets."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.1
            )
            
            rating_text = response.choices[0].message.content
            
            # Extract rating
            if "STRONG_BUY" in rating_text:
                return "STRONG_BUY"
            elif "BUY" in rating_text:
                return "BUY"
            elif "HOLD" in rating_text:
                return "HOLD"
            elif "SELL" in rating_text:
                return "SELL"
            elif "STRONG_SELL" in rating_text:
                return "STRONG_SELL"
            else:
                return "HOLD"
                
        except Exception as e:
            logger.error(f"Error generating investment rating: {e}")
            return "HOLD"
            
    def _synthesize_stock_analysis(
        self,
        stock_symbol: str,
        company_name: str,
        stock_metrics: Optional[StockMetrics],
        financial_analysis: Dict[str, Any],
        technical_analysis: Dict[str, Any],
        valuation_analysis: Dict[str, Any],
        investment_rating: str
    ) -> StockAnalysis:
        """Synthesize all stock analysis components."""
        try:
            # Generate performance summary using AI
            summary = self._generate_performance_summary(
                stock_metrics, financial_analysis, technical_analysis, valuation_analysis
            )
            
            # Calculate target price
            target_price = self._calculate_target_price(stock_metrics, valuation_analysis)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                stock_metrics, financial_analysis, technical_analysis
            )
            
            analysis = StockAnalysis(
                symbol=stock_symbol,
                company_name=company_name,
                current_price=stock_metrics.current_price if stock_metrics else 0,
                market_cap=stock_metrics.market_cap if stock_metrics else 0,
                financial_metrics=financial_analysis,
                technical_analysis=technical_analysis,
                valuation_metrics=valuation_analysis,
                performance_summary=summary,
                investment_rating=investment_rating,
                target_price=target_price,
                confidence_score=confidence_score
            )
            
            logger.info(f"Synthesized stock analysis for {stock_symbol}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error synthesizing stock analysis: {e}")
            return self._create_fallback_analysis(stock_symbol)
            
    def _store_analysis_results(self, analysis: StockAnalysis) -> None:
        """Store analysis results in MCP context."""
        try:
            # Get the original stock metrics for individual fields
            stock_metrics = self._get_stock_metrics(analysis.symbol)
            
            analysis_data = {
                "symbol": analysis.symbol,
                "company_name": analysis.company_name,
                "current_price": analysis.current_price,
                "market_cap": analysis.market_cap,
                "pe_ratio": stock_metrics.pe_ratio if stock_metrics else None,
                "pb_ratio": stock_metrics.pb_ratio if stock_metrics else None,
                "eps": stock_metrics.eps if stock_metrics else None,
                "dividend_yield": stock_metrics.dividend_yield if stock_metrics else None,
                "beta": stock_metrics.beta if stock_metrics else None,
                "volume": stock_metrics.volume if stock_metrics else None,
                "avg_volume": stock_metrics.avg_volume if stock_metrics else None,
                "change_percent": stock_metrics.change_percent if stock_metrics else None,
                "high_52w": stock_metrics.high_52w if stock_metrics else None,
                "low_52w": stock_metrics.low_52w if stock_metrics else None,
                "revenue_growth": stock_metrics.revenue_growth if stock_metrics else None,
                "profit_growth": stock_metrics.profit_growth if stock_metrics else None,
                "roe": stock_metrics.roe if stock_metrics else None,
                "roa": stock_metrics.roa if stock_metrics else None,
                "ev_ebitda": stock_metrics.ev_ebitda if stock_metrics else None,
                "debt_to_equity": stock_metrics.debt_to_equity if stock_metrics else None,
                "current_ratio": stock_metrics.current_ratio if stock_metrics else None,
                "financial_metrics": analysis.financial_metrics,
                "technical_analysis": analysis.technical_analysis,
                "valuation_metrics": analysis.valuation_metrics,
                "performance_summary": analysis.performance_summary,
                "investment_rating": analysis.investment_rating,
                "target_price": analysis.target_price,
                "confidence_score": analysis.confidence_score,
                "timestamp": datetime.now().isoformat()
            }
            
            self.mcp_context.store_context(
                context_id=f"stock_analysis_{analysis.symbol}",
                context_type=ContextType.STOCK_SUMMARY,
                data=analysis_data,
                agent_id=self.agent_id,
                metadata={"analysis_type": "stock_research"}
            )
            
            logger.info(f"Stored stock analysis results for {analysis.symbol}")
            
        except Exception as e:
            logger.error(f"Error storing analysis results: {e}")
            
    # Helper methods for technical analysis
    def _analyze_trend(self, historical_data: List[PriceData]) -> str:
        """Analyze price trend."""
        if len(historical_data) < 2:
            return "Insufficient data"
            
        recent_prices = [data.close for data in historical_data[-20:]]
        if len(recent_prices) < 2:
            return "Insufficient data"
            
        if recent_prices[-1] > recent_prices[0]:
            return "Uptrend"
        elif recent_prices[-1] < recent_prices[0]:
            return "Downtrend"
        else:
            return "Sideways"
            
    def _find_support_resistance(self, historical_data: List[PriceData]) -> Dict[str, float]:
        """Find support and resistance levels."""
        if not historical_data:
            return {}
            
        prices = [data.close for data in historical_data]
        return {
            "support": min(prices),
            "resistance": max(prices),
            "current": prices[-1] if prices else 0
        }
        
    def _calculate_momentum(self, historical_data: List[PriceData]) -> float:
        """Calculate price momentum."""
        if len(historical_data) < 2:
            return 0.0
            
        recent_prices = [data.close for data in historical_data[-10:]]
        if len(recent_prices) < 2:
            return 0.0
            
        return (recent_prices[-1] - recent_prices[0]) / recent_prices[0] * 100
        
    def _calculate_volatility(self, stock_metrics: StockMetrics) -> float:
        """Calculate stock volatility."""
        if not stock_metrics or not stock_metrics.beta:
            return 0.0
        return stock_metrics.beta * 15  # Simplified volatility calculation
        
    def _assess_valuation(self, stock_metrics: StockMetrics) -> str:
        """Assess if stock is overvalued, undervalued, or fairly valued."""
        if not stock_metrics or not stock_metrics.pe_ratio:
            return "Unknown"
            
        pe = stock_metrics.pe_ratio
        if pe < 15:
            return "Undervalued"
        elif pe > 25:
            return "Overvalued"
        else:
            return "Fairly valued"
            
    def _calculate_price_targets(
        self,
        stock_metrics: Optional[StockMetrics],
        technical_analysis: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate price targets."""
        if not stock_metrics:
            return {}
            
        current_price = stock_metrics.current_price
        return {
            "conservative": current_price * 1.1,
            "moderate": current_price * 1.2,
            "optimistic": current_price * 1.3
        }
        
    def _calculate_upside_potential(self, stock_metrics: Optional[StockMetrics]) -> float:
        """Calculate upside potential."""
        if not stock_metrics:
            return 0.0
            
        return (stock_metrics.high_52w - stock_metrics.current_price) / stock_metrics.current_price * 100
        
    def _generate_performance_summary(
        self,
        stock_metrics: Optional[StockMetrics],
        financial_analysis: Dict[str, Any],
        technical_analysis: Dict[str, Any],
        valuation_analysis: Dict[str, Any]
    ) -> str:
        """Generate performance summary using AI."""
        try:
            prompt = f"""
            Provide a brief performance summary for this stock based on:
            
            Stock Metrics: {stock_metrics.__dict__ if stock_metrics else 'N/A'}
            Financial Analysis: {financial_analysis}
            Technical Analysis: {technical_analysis}
            Valuation Analysis: {valuation_analysis}
            
            Keep it concise (2-3 sentences) and focus on key performance indicators.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a financial analyst."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating performance summary: {e}")
            return "Performance analysis pending"
            
    def _calculate_target_price(
        self,
        stock_metrics: Optional[StockMetrics],
        valuation_analysis: Dict[str, Any]
    ) -> Optional[float]:
        """Calculate target price."""
        if not stock_metrics:
            return None
            
        # Simple target price calculation
        current_price = stock_metrics.current_price
        pe_ratio = stock_metrics.pe_ratio or 20
        
        # Adjust based on PE ratio
        if pe_ratio < 15:
            return current_price * 1.2  # 20% upside for undervalued stocks
        elif pe_ratio > 25:
            return current_price * 0.9  # 10% downside for overvalued stocks
        else:
            return current_price * 1.1  # 10% upside for fairly valued stocks
            
    def _calculate_confidence_score(
        self,
        stock_metrics: Optional[StockMetrics],
        financial_analysis: Dict[str, Any],
        technical_analysis: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for the analysis."""
        score = 0.5  # Base score
        
        if stock_metrics:
            score += 0.2
        if financial_analysis:
            score += 0.2
        if technical_analysis:
            score += 0.1
            
        return min(score, 1.0)
        
    def _create_fallback_analysis(self, stock_symbol: str) -> StockAnalysis:
        """Create fallback analysis when main analysis fails."""
        return StockAnalysis(
            symbol=stock_symbol,
            company_name=stock_symbol,  # Fallback to symbol if no company name
            current_price=0.0,
            market_cap=0.0,
            financial_metrics={},
            technical_analysis={},
            valuation_metrics={},
            performance_summary="Analysis pending due to data limitations",
            investment_rating="HOLD",
            target_price=None,
            confidence_score=0.3
        )
