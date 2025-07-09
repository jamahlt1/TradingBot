from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class TradeBase(BaseModel):
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    status: str  # 'pending', 'filled', 'cancelled'
    strategy_id: Optional[int] = None

class TradeCreate(TradeBase):
    pass

class TradeUpdate(BaseModel):
    symbol: Optional[str] = None
    side: Optional[str] = None
    quantity: Optional[float] = None
    price: Optional[float] = None
    status: Optional[str] = None
    strategy_id: Optional[int] = None

class TradeResponse(TradeBase):
    id: int
    timestamp: datetime
    user_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True