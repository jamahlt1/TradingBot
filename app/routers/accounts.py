from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter(prefix="/accounts", tags=["accounts"])

class Account(BaseModel):
    id: int
    name: str
    type: str  # e.g., 'forex', 'crypto', 'stocks', etc.
    balance: float
    active: bool = True

accounts_db = []

@router.get("/", response_model=List[Account])
def list_accounts():
    return accounts_db

@router.post("/", response_model=Account)
def create_account(account: Account):
    accounts_db.append(account)
    return account

@router.get("/{account_id}", response_model=Account)
def get_account(account_id: int):
    for a in accounts_db:
        if a.id == account_id:
            return a
    raise HTTPException(status_code=404, detail="Account not found")

@router.put("/{account_id}", response_model=Account)
def update_account(account_id: int, account: Account):
    for i, a in enumerate(accounts_db):
        if a.id == account_id:
            accounts_db[i] = account
            return account
    raise HTTPException(status_code=404, detail="Account not found")

@router.delete("/{account_id}")
def delete_account(account_id: int):
    for i, a in enumerate(accounts_db):
        if a.id == account_id:
            del accounts_db[i]
            return {"ok": True}
    raise HTTPException(status_code=404, detail="Account not found")