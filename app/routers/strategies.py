from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import and_
from typing import List
from app.database.session import get_db
from app.crud import strategy as crud_strategy
from app.schemas.strategy import StrategyCreate, StrategyUpdate, StrategyResponse
from app.routers.auth import get_current_user
from app.models.user import User
from app.models.strategy import Strategy
from app.models.trade import Trade

router = APIRouter(prefix="/strategies", tags=["strategies"])

@router.get("/", response_model=List[StrategyResponse])
def list_strategies(
    skip: int = 0, 
    limit: int = 100, 
    strategy_type: str = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    strategies = crud_strategy.get_strategies(db, user_id=current_user.id, skip=skip, limit=limit)
    if strategy_type:
        strategies = [s for s in strategies if s.strategy_type == strategy_type]
    return strategies

@router.get("/types")
def get_strategy_types():
    """Get all available strategy types"""
    return {
        "strategy_types": [
            "straddle_hedging",
            "trend_trading", 
            "swing_trading",
            "scalping",
            "crypto_trend",
            "news_trading",
            "sentiment_trading",
            "pairs_trading",
            "stat_arb",
            "ict_concepts"
        ]
    }

@router.get("/type/{strategy_type}", response_model=List[StrategyResponse])
def get_strategies_by_type(
    strategy_type: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get strategies by specific type"""
    strategies = db.query(Strategy).filter(
        and_(
            Strategy.user_id == current_user.id,
            Strategy.strategy_type == strategy_type
        )
    ).all()
    return strategies

@router.post("/", response_model=StrategyResponse)
def create_strategy(
    strategy: StrategyCreate, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    return crud_strategy.create_strategy(db=db, strategy=strategy, user_id=current_user.id)

@router.get("/{strategy_id}", response_model=StrategyResponse)
def get_strategy(
    strategy_id: int, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    strategy = crud_strategy.get_strategy(db, strategy_id=strategy_id, user_id=current_user.id)
    if strategy is None:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return strategy

@router.put("/{strategy_id}", response_model=StrategyResponse)
def update_strategy(
    strategy_id: int, 
    strategy: StrategyUpdate, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    db_strategy = crud_strategy.update_strategy(db, strategy_id=strategy_id, strategy=strategy, user_id=current_user.id)
    if db_strategy is None:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return db_strategy

@router.delete("/{strategy_id}")
def delete_strategy(
    strategy_id: int, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    db_strategy = crud_strategy.delete_strategy(db, strategy_id=strategy_id, user_id=current_user.id)
    if db_strategy is None:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return {"ok": True}

@router.get("/{strategy_id}/performance")
def get_strategy_performance(
    strategy_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get detailed performance metrics for a specific strategy"""
    strategy = crud_strategy.get_strategy(db, strategy_id=strategy_id, user_id=current_user.id)
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    # Get all trades for this strategy
    trades = db.query(Trade).filter(
        and_(
            Trade.strategy_id == strategy_id,
            Trade.user_id == current_user.id
        )
    ).all()
    
    if not trades:
        return {
            "strategy_id": strategy_id,
            "strategy_name": strategy.name,
            "strategy_type": strategy.strategy_type,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0,
            "total_pnl": 0,
            "avg_trade_size": 0,
            "max_drawdown": 0
        }
    
    # Calculate metrics
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t.side == "sell" and t.price > 0])
    losing_trades = len([t for t in trades if t.side == "buy" and t.price > 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    total_pnl = sum([t.price for t in trades if t.side == "sell"])
    avg_trade_size = total_pnl / total_trades if total_trades > 0 else 0
    
    return {
        "strategy_id": strategy_id,
        "strategy_name": strategy.name,
        "strategy_type": strategy.strategy_type,
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": round(win_rate, 2),
        "total_pnl": round(total_pnl, 2),
        "avg_trade_size": round(avg_trade_size, 2),
        "max_drawdown": 0,  # TODO: Implement proper drawdown calculation
        "allocation": strategy.allocation,
        "risk_per_trade": strategy.risk_per_trade,
        "rr_target": strategy.rr_target
    }

@router.post("/{strategy_id}/activate")
def activate_strategy(
    strategy_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Activate a strategy"""
    strategy = crud_strategy.get_strategy(db, strategy_id=strategy_id, user_id=current_user.id)
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    strategy.active = True
    db.commit()
    return {"message": "Strategy activated"}

@router.post("/{strategy_id}/deactivate")
def deactivate_strategy(
    strategy_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Deactivate a strategy"""
    strategy = crud_strategy.get_strategy(db, strategy_id=strategy_id, user_id=current_user.id)
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    strategy.active = False
    db.commit()
    return {"message": "Strategy deactivated"}