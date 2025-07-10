from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from app.database.session import get_db
from app.crud import account as crud_account
from app.schemas.account import AccountCreate, AccountUpdate, AccountResponse
from app.routers.auth import get_current_user
from app.models.user import User

router = APIRouter(prefix="/accounts", tags=["accounts"])

@router.get("/", response_model=List[AccountResponse])
def list_accounts(
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    accounts = crud_account.get_accounts(db, user_id=current_user.id, skip=skip, limit=limit)
    return accounts

@router.post("/", response_model=AccountResponse)
def create_account(
    account: AccountCreate, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    return crud_account.create_account(db=db, account=account, user_id=current_user.id)

@router.get("/{account_id}", response_model=AccountResponse)
def get_account(
    account_id: int, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    account = crud_account.get_account(db, account_id=account_id, user_id=current_user.id)
    if account is None:
        raise HTTPException(status_code=404, detail="Account not found")
    return account

@router.put("/{account_id}", response_model=AccountResponse)
def update_account(
    account_id: int, 
    account: AccountUpdate, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    db_account = crud_account.update_account(db, account_id=account_id, account=account, user_id=current_user.id)
    if db_account is None:
        raise HTTPException(status_code=404, detail="Account not found")
    return db_account

@router.delete("/{account_id}")
def delete_account(
    account_id: int, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    db_account = crud_account.delete_account(db, account_id=account_id, user_id=current_user.id)
    if db_account is None:
        raise HTTPException(status_code=404, detail="Account not found")
    return {"ok": True}