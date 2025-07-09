from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class StrategyBase(BaseModel):
    name: str
    description: Optional[str] = None
    allocation: float
    risk_per_trade: float
    rr_target: float
    trailing_stop: Optional[float] = None
    active: bool = True

class StrategyCreate(StrategyBase):
    pass

class StrategyUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    allocation: Optional[float] = None
    risk_per_trade: Optional[float] = None
    rr_target: Optional[float] = None
    trailing_stop: Optional[float] = None
    active: Optional[bool] = None

class StrategyResponse(StrategyBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True