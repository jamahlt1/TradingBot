from sqlalchemy.orm import Session
from app.models.trade import Trade
from app.schemas.trade import TradeCreate, TradeUpdate

def get_trade(db: Session, trade_id: int, user_id: int):
    return db.query(Trade).filter(Trade.id == trade_id, Trade.user_id == user_id).first()

def get_trades(db: Session, user_id: int, skip: int = 0, limit: int = 100):
    return db.query(Trade).filter(Trade.user_id == user_id).offset(skip).limit(limit).all()

def create_trade(db: Session, trade: TradeCreate, user_id: int):
    db_trade = Trade(**trade.dict(), user_id=user_id)
    db.add(db_trade)
    db.commit()
    db.refresh(db_trade)
    return db_trade

def update_trade(db: Session, trade_id: int, trade: TradeUpdate, user_id: int):
    db_trade = get_trade(db, trade_id, user_id)
    if not db_trade:
        return None
    
    update_data = trade.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_trade, field, value)
    
    db.commit()
    db.refresh(db_trade)
    return db_trade

def delete_trade(db: Session, trade_id: int, user_id: int):
    db_trade = get_trade(db, trade_id, user_id)
    if db_trade:
        db.delete(db_trade)
        db.commit()
    return db_trade

def get_trades_by_strategy(db: Session, strategy_id: int, user_id: int, skip: int = 0, limit: int = 100):
    return db.query(Trade).filter(
        Trade.strategy_id == strategy_id, 
        Trade.user_id == user_id
    ).offset(skip).limit(limit).all()