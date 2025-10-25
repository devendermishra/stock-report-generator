"""
Data models for stock data operations.
Contains dataclasses and type definitions for stock-related data structures.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class StockMetrics:
    """Represents key stock metrics and ratios."""
    symbol: str
    current_price: float
    market_cap: float
    pe_ratio: Optional[float]
    pb_ratio: Optional[float]
    eps: Optional[float]
    dividend_yield: Optional[float]
    beta: Optional[float]
    volume: int
    avg_volume: int
    high_52w: float
    low_52w: float
    change_percent: float
    last_updated: datetime
    # Additional financial metrics
    revenue_growth: Optional[float] = None
    profit_growth: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    ev_ebitda: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    return_on_equity: Optional[float] = None
    return_on_assets: Optional[float] = None


@dataclass
class PriceData:
    """Represents historical price data."""
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class CompanyInfo:
    """Represents company information from NSE."""
    symbol: str
    company_name: str
    sector: str
    industry: str
    market_cap: Optional[float] = None
    isin: Optional[str] = None
    listing_date: Optional[str] = None
