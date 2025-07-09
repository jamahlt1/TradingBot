from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from app.database.session import get_db
from app.models.trade import Trade
from app.models.strategy import Strategy
from app.routers.auth import get_current_user
from app.models.user import User
from typing import Dict, Any
from datetime import datetime, timedelta

router = APIRouter(prefix="/analytics", tags=["analytics"])

@router.get("/overview")
def get_analytics_overview(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Get total trades
    total_trades = db.query(Trade).filter(Trade.user_id == current_user.id).count()
    
    # Get winning and losing trades
    winning_trades = db.query(Trade).filter(
        and_(Trade.user_id == current_user.id, Trade.side == "sell")
    ).count()
    
    losing_trades = db.query(Trade).filter(
        and_(Trade.user_id == current_user.id, Trade.side == "buy")
    ).count()
    
    # Calculate win rate
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    # Calculate total P&L
    total_pnl = db.query(func.sum(Trade.price)).filter(
        and_(Trade.user_id == current_user.id, Trade.side == "sell")
    ).scalar() or 0
    
    # Calculate max drawdown (simplified)
    max_drawdown = 0  # TODO: Implement proper drawdown calculation
    
    return {
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": round(win_rate, 2),
        "total_pnl": round(total_pnl, 2),
        "max_drawdown": max_drawdown
    }

@router.get("/performance")
def get_performance_metrics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Get trades from last 30 days
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    
    recent_trades = db.query(Trade).filter(
        and_(
            Trade.user_id == current_user.id,
            Trade.timestamp >= thirty_days_ago
        )
    ).all()
    
    # Calculate daily returns
    daily_returns = {}
    for trade in recent_trades:
        date = trade.timestamp.date()
        if date not in daily_returns:
            daily_returns[date] = 0
        if trade.side == "sell":
            daily_returns[date] += trade.price
        else:
            daily_returns[date] -= trade.price
    
    return {
        "daily_returns": daily_returns,
        "total_trades_30d": len(recent_trades),
        "period": "30 days"
    }

@router.get("/strategy/{strategy_id}")
def get_strategy_analytics(
    strategy_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Verify strategy belongs to user
    strategy = db.query(Strategy).filter(
        and_(Strategy.id == strategy_id, Strategy.user_id == current_user.id)
    ).first()
    
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    # Get trades for this strategy
    strategy_trades = db.query(Trade).filter(
        and_(
            Trade.strategy_id == strategy_id,
            Trade.user_id == current_user.id
        )
    ).all()
    
    total_trades = len(strategy_trades)
    winning_trades = len([t for t in strategy_trades if t.side == "sell"])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    total_pnl = sum([t.price for t in strategy_trades if t.side == "sell"])
    
    return {
        "strategy_id": strategy_id,
        "strategy_name": strategy.name,
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "win_rate": round(win_rate, 2),
        "total_pnl": round(total_pnl, 2),
        "allocation": strategy.allocation,
        "risk_per_trade": strategy.risk_per_trade
    }