from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from app.database.session import get_db
from app.crud import trade as crud_trade
from app.schemas.trade import TradeCreate, TradeUpdate, TradeResponse
from app.routers.auth import get_current_user
from app.models.user import User

router = APIRouter(prefix="/trades", tags=["trades"])

@router.get("/", response_model=List[TradeResponse])
def list_trades(
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    trades = crud_trade.get_trades(db, user_id=current_user.id, skip=skip, limit=limit)
    return trades

@router.post("/", response_model=TradeResponse)
def create_trade(
    trade: TradeCreate, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    return crud_trade.create_trade(db=db, trade=trade, user_id=current_user.id)

@router.get("/{trade_id}", response_model=TradeResponse)
def get_trade(
    trade_id: int, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    trade = crud_trade.get_trade(db, trade_id=trade_id, user_id=current_user.id)
    if trade is None:
        raise HTTPException(status_code=404, detail="Trade not found")
    return trade

@router.put("/{trade_id}", response_model=TradeResponse)
def update_trade(
    trade_id: int, 
    trade: TradeUpdate, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    db_trade = crud_trade.update_trade(db, trade_id=trade_id, trade=trade, user_id=current_user.id)
    if db_trade is None:
        raise HTTPException(status_code=404, detail="Trade not found")
    return db_trade

@router.delete("/{trade_id}")
def delete_trade(
    trade_id: int, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    db_trade = crud_trade.delete_trade(db, trade_id=trade_id, user_id=current_user.id)
    if db_trade is None:
        raise HTTPException(status_code=404, detail="Trade not found")
    return {"ok": True}

@router.get("/strategy/{strategy_id}", response_model=List[TradeResponse])
def get_trades_by_strategy(
    strategy_id: int,
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    trades = crud_trade.get_trades_by_strategy(db, strategy_id=strategy_id, user_id=current_user.id, skip=skip, limit=limit)
    return trades