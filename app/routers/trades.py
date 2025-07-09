from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

router = APIRouter(prefix="/trades", tags=["trades"])

class Trade(BaseModel):
    id: int
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    timestamp: datetime
    status: str  # 'pending', 'filled', 'cancelled'
    strategy_id: Optional[int] = None

trades_db = []

@router.get("/", response_model=List[Trade])
def list_trades():
    return trades_db

@router.post("/", response_model=Trade)
def create_trade(trade: Trade):
    trades_db.append(trade)
    return trade

@router.get("/{trade_id}", response_model=Trade)
def get_trade(trade_id: int):
    for t in trades_db:
        if t.id == trade_id:
            return t
    raise HTTPException(status_code=404, detail="Trade not found")

@router.put("/{trade_id}", response_model=Trade)
def update_trade(trade_id: int, trade: Trade):
    for i, t in enumerate(trades_db):
        if t.id == trade_id:
            trades_db[i] = trade
            return trade
    raise HTTPException(status_code=404, detail="Trade not found")

@router.delete("/{trade_id}")
def delete_trade(trade_id: int):
    for i, t in enumerate(trades_db):
        if t.id == trade_id:
            del trades_db[i]
            return {"ok": True}
    raise HTTPException(status_code=404, detail="Trade not found")