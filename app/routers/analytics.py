from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from app.database.session import get_db
from app.models.trade import Trade
from app.models.strategy import Strategy
from app.routers.auth import get_current_user
from app.models.user import User
from typing import Dict, Any
from datetime import datetime, timedelta
import numpy as np

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

@router.get("/capm/asset")
def get_capm_for_asset(
    symbol: str = Query(..., description="Asset symbol, e.g. AAPL, BTCUSDT"),
    market_symbol: str = Query("SPY", description="Market index symbol, e.g. SPY, BTCUSD, etc."),
    risk_free_rate: float = Query(0.03, description="Annual risk-free rate as decimal, e.g. 0.03 for 3%"),
    lookback_days: int = Query(252, description="Number of days for historical returns"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Calculate CAPM metrics for a single asset.
    """
    # Fetch historical daily returns for asset and market
    asset_returns = db.execute(
        f"""
        SELECT date, daily_return FROM asset_returns
        WHERE symbol = :symbol
        ORDER BY date DESC LIMIT :lookback
        """, {"symbol": symbol, "lookback": lookback_days}
    ).fetchall()
    market_returns = db.execute(
        f"""
        SELECT date, daily_return FROM asset_returns
        WHERE symbol = :symbol
        ORDER BY date DESC LIMIT :lookback
        """, {"symbol": market_symbol, "lookback": lookback_days}
    ).fetchall()
    if not asset_returns or not market_returns:
        raise HTTPException(status_code=404, detail="Insufficient return data for CAPM calculation.")
    # Align by date
    asset_dict = {r[0]: r[1] for r in asset_returns}
    market_dict = {r[0]: r[1] for r in market_returns}
    common_dates = sorted(set(asset_dict.keys()) & set(market_dict.keys()))
    asset_series = np.array([asset_dict[d] for d in common_dates])
    market_series = np.array([market_dict[d] for d in common_dates])
    # Calculate beta
    beta = float(np.cov(asset_series, market_series)[0, 1] / np.var(market_series))
    # Calculate expected return
    avg_market_return = float(np.mean(market_series)) * 252  # annualized
    expected_return = risk_free_rate + beta * (avg_market_return - risk_free_rate)
    # Calculate realized return
    realized_return = float(np.mean(asset_series)) * 252
    # Calculate alpha
    alpha = realized_return - (risk_free_rate + beta * (avg_market_return - risk_free_rate))
    # Risk premium
    risk_premium = avg_market_return - risk_free_rate
    return {
        "symbol": symbol,
        "market_symbol": market_symbol,
        "beta": round(beta, 4),
        "expected_return": round(expected_return, 4),
        "realized_return": round(realized_return, 4),
        "alpha": round(alpha, 4),
        "risk_premium": round(risk_premium, 4),
        "risk_free_rate": risk_free_rate,
        "lookback_days": lookback_days
    }

@router.get("/capm/portfolio")
def get_capm_for_portfolio(
    market_symbol: str = Query("SPY", description="Market index symbol, e.g. SPY, BTCUSD, etc."),
    risk_free_rate: float = Query(0.03, description="Annual risk-free rate as decimal, e.g. 0.03 for 3%"),
    lookback_days: int = Query(252, description="Number of days for historical returns"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Calculate CAPM metrics for the user's portfolio (weighted by current holdings).
    """
    # Get user's current holdings (symbol, weight)
    holdings = db.execute(
        """
        SELECT symbol, weight FROM portfolio_holdings
        WHERE user_id = :user_id
        """, {"user_id": current_user.id}
    ).fetchall()
    if not holdings:
        raise HTTPException(status_code=404, detail="No portfolio holdings found.")
    # For each asset, get returns
    asset_returns_dict = {}
    for symbol, weight in holdings:
        asset_returns = db.execute(
            f"""
            SELECT date, daily_return FROM asset_returns
            WHERE symbol = :symbol
            ORDER BY date DESC LIMIT :lookback
            """, {"symbol": symbol, "lookback": lookback_days}
        ).fetchall()
        asset_returns_dict[symbol] = {r[0]: r[1] for r in asset_returns}
    # Get market returns
    market_returns = db.execute(
        f"""
        SELECT date, daily_return FROM asset_returns
        WHERE symbol = :symbol
        ORDER BY date DESC LIMIT :lookback
        """, {"symbol": market_symbol, "lookback": lookback_days}
    ).fetchall()
    market_dict = {r[0]: r[1] for r in market_returns}
    # Align dates
    common_dates = set(market_dict.keys())
    for symbol in asset_returns_dict:
        common_dates &= set(asset_returns_dict[symbol].keys())
    common_dates = sorted(common_dates)
    # Build portfolio return series
    weights = {symbol: float(weight) for symbol, weight in holdings}
    portfolio_series = []
    market_series = []
    for d in common_dates:
        port_ret = sum(asset_returns_dict[s][d] * weights[s] for s in weights)
        portfolio_series.append(port_ret)
        market_series.append(market_dict[d])
    portfolio_series = np.array(portfolio_series)
    market_series = np.array(market_series)
    # Calculate beta
    beta = float(np.cov(portfolio_series, market_series)[0, 1] / np.var(market_series))
    # Calculate expected return
    avg_market_return = float(np.mean(market_series)) * 252
    expected_return = risk_free_rate + beta * (avg_market_return - risk_free_rate)
    # Realized return
    realized_return = float(np.mean(portfolio_series)) * 252
    # Alpha
    alpha = realized_return - (risk_free_rate + beta * (avg_market_return - risk_free_rate))
    # Risk premium
    risk_premium = avg_market_return - risk_free_rate
    return {
        "portfolio": [{"symbol": s, "weight": weights[s]} for s in weights],
        "market_symbol": market_symbol,
        "beta": round(beta, 4),
        "expected_return": round(expected_return, 4),
        "realized_return": round(realized_return, 4),
        "alpha": round(alpha, 4),
        "risk_premium": round(risk_premium, 4),
        "risk_free_rate": risk_free_rate,
        "lookback_days": lookback_days
    }