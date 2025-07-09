from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from app.database.session import get_db
from app.crud import strategy as crud_strategy
from app.schemas.strategy import StrategyCreate, StrategyUpdate, StrategyResponse
from app.routers.auth import get_current_user
from app.models.user import User

router = APIRouter(prefix="/strategies", tags=["strategies"])

@router.get("/", response_model=List[StrategyResponse])
def list_strategies(
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    strategies = crud_strategy.get_strategies(db, user_id=current_user.id, skip=skip, limit=limit)
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