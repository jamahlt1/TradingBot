from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter(prefix="/strategies", tags=["strategies"])

class Strategy(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    allocation: float
    risk_per_trade: float
    rr_target: float
    trailing_stop: Optional[float] = None
    active: bool = True

# In-memory store for now
strategies_db = []

@router.get("/", response_model=List[Strategy])
def list_strategies():
    return strategies_db

@router.post("/", response_model=Strategy)
def create_strategy(strategy: Strategy):
    strategies_db.append(strategy)
    return strategy

@router.get("/{strategy_id}", response_model=Strategy)
def get_strategy(strategy_id: int):
    for s in strategies_db:
        if s.id == strategy_id:
            return s
    raise HTTPException(status_code=404, detail="Strategy not found")

@router.put("/{strategy_id}", response_model=Strategy)
def update_strategy(strategy_id: int, strategy: Strategy):
    for i, s in enumerate(strategies_db):
        if s.id == strategy_id:
            strategies_db[i] = strategy
            return strategy
    raise HTTPException(status_code=404, detail="Strategy not found")

@router.delete("/{strategy_id}")
def delete_strategy(strategy_id: int):
    for i, s in enumerate(strategies_db):
        if s.id == strategy_id:
            del strategies_db[i]
            return {"ok": True}
    raise HTTPException(status_code=404, detail="Strategy not found")