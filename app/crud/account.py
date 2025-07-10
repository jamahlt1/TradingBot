from sqlalchemy.orm import Session
from app.models.account import Account
from app.schemas.account import AccountCreate, AccountUpdate

def get_account(db: Session, account_id: int, user_id: int):
    return db.query(Account).filter(Account.id == account_id, Account.user_id == user_id).first()

def get_accounts(db: Session, user_id: int, skip: int = 0, limit: int = 100):
    return db.query(Account).filter(Account.user_id == user_id).offset(skip).limit(limit).all()

def create_account(db: Session, account: AccountCreate, user_id: int):
    db_account = Account(**account.dict(), user_id=user_id)
    db.add(db_account)
    db.commit()
    db.refresh(db_account)
    return db_account

def update_account(db: Session, account_id: int, account: AccountUpdate, user_id: int):
    db_account = get_account(db, account_id, user_id)
    if not db_account:
        return None
    
    update_data = account.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_account, field, value)
    
    db.commit()
    db.refresh(db_account)
    return db_account

def delete_account(db: Session, account_id: int, user_id: int):
    db_account = get_account(db, account_id, user_id)
    if db_account:
        db.delete(db_account)
        db.commit()
    return db_account