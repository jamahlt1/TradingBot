from sqlalchemy.orm import Session
from app.models.strategy import Strategy
from app.schemas.strategy import StrategyCreate, StrategyUpdate

def get_strategy(db: Session, strategy_id: int, user_id: int):
    return db.query(Strategy).filter(Strategy.id == strategy_id, Strategy.user_id == user_id).first()

def get_strategies(db: Session, user_id: int, skip: int = 0, limit: int = 100):
    return db.query(Strategy).filter(Strategy.user_id == user_id).offset(skip).limit(limit).all()

def create_strategy(db: Session, strategy: StrategyCreate, user_id: int):
    db_strategy = Strategy(**strategy.dict(), user_id=user_id)
    db.add(db_strategy)
    db.commit()
    db.refresh(db_strategy)
    return db_strategy

def update_strategy(db: Session, strategy_id: int, strategy: StrategyUpdate, user_id: int):
    db_strategy = get_strategy(db, strategy_id, user_id)
    if not db_strategy:
        return None
    
    update_data = strategy.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_strategy, field, value)
    
    db.commit()
    db.refresh(db_strategy)
    return db_strategy

def delete_strategy(db: Session, strategy_id: int, user_id: int):
    db_strategy = get_strategy(db, strategy_id, user_id)
    if db_strategy:
        db.delete(db_strategy)
        db.commit()
    return db_strategy