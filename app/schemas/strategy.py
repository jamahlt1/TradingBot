from pydantic import BaseModel, validator
from typing import Optional, List
from datetime import datetime
import json

class StrategyBase(BaseModel):
    name: str
    description: Optional[str] = None
    strategy_type: str  # straddle_hedging, trend_trading, swing_trading, scalping, crypto_trend, news_trading, sentiment_trading, pairs_trading, stat_arb, ict_concepts
    allocation: float
    risk_per_trade: float
    rr_target: float
    trailing_stop: Optional[float] = None
    active: bool = True
    
    # Strategy-specific parameters
    entry_conditions: Optional[str] = None  # JSON string
    exit_conditions: Optional[str] = None   # JSON string
    timeframes: Optional[str] = None        # Multiple timeframes
    instruments: Optional[str] = None       # JSON string
    max_positions: Optional[int] = 1
    correlation_threshold: Optional[float] = None
    volatility_threshold: Optional[float] = None
    news_sources: Optional[str] = None      # JSON string
    sentiment_sources: Optional[str] = None # JSON string

    @validator('strategy_type')
    def validate_strategy_type(cls, v):
        valid_types = [
            'straddle_hedging', 'trend_trading', 'swing_trading', 'scalping',
            'crypto_trend', 'news_trading', 'sentiment_trading', 'pairs_trading',
            'stat_arb', 'ict_concepts'
        ]
        if v not in valid_types:
            raise ValueError(f'strategy_type must be one of {valid_types}')
        return v

    @validator('entry_conditions', 'exit_conditions', 'instruments', 'news_sources', 'sentiment_sources')
    def validate_json_fields(cls, v):
        if v is not None:
            try:
                json.loads(v)
            except json.JSONDecodeError:
                raise ValueError('Must be valid JSON string')
        return v

class StrategyCreate(StrategyBase):
    pass

class StrategyUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    strategy_type: Optional[str] = None
    allocation: Optional[float] = None
    risk_per_trade: Optional[float] = None
    rr_target: Optional[float] = None
    trailing_stop: Optional[float] = None
    active: Optional[bool] = None
    entry_conditions: Optional[str] = None
    exit_conditions: Optional[str] = None
    timeframes: Optional[str] = None
    instruments: Optional[str] = None
    max_positions: Optional[int] = None
    correlation_threshold: Optional[float] = None
    volatility_threshold: Optional[float] = None
    news_sources: Optional[str] = None
    sentiment_sources: Optional[str] = None

class StrategyResponse(StrategyBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True