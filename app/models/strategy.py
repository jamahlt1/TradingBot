from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Strategy(Base):
    __tablename__ = "strategies"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(Text, nullable=True)
    strategy_type = Column(String, index=True)  # straddle_hedging, trend_trading, swing_trading, scalping, crypto_trend, news_trading, sentiment_trading, pairs_trading, stat_arb, ict_concepts
    allocation = Column(Float)
    risk_per_trade = Column(Float)
    rr_target = Column(Float)
    trailing_stop = Column(Float, nullable=True)
    active = Column(Boolean, default=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    
    # Strategy-specific parameters
    entry_conditions = Column(Text, nullable=True)  # JSON string for entry conditions
    exit_conditions = Column(Text, nullable=True)   # JSON string for exit conditions
    timeframes = Column(String, nullable=True)      # Multiple timeframes for analysis
    instruments = Column(Text, nullable=True)       # JSON string for trading instruments
    max_positions = Column(Integer, default=1)      # Maximum concurrent positions
    correlation_threshold = Column(Float, nullable=True)  # For pairs trading
    volatility_threshold = Column(Float, nullable=True)   # For straddle strategies
    news_sources = Column(Text, nullable=True)      # JSON string for news sources
    sentiment_sources = Column(Text, nullable=True) # JSON string for sentiment sources
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)