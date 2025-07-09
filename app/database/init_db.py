from sqlalchemy.orm import Session
from app.database.session import engine, Base
from app.models import user, strategy, account, trade
from app.core.security import get_password_hash
from app.crud import user as crud_user
from app.schemas.user import UserCreate

def init_db():
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    # Create sample user if none exists
    from app.database.session import SessionLocal
    db = SessionLocal()
    
    try:
        # Check if any users exist
        existing_user = db.query(user.User).first()
        if not existing_user:
            # Create sample user
            sample_user = UserCreate(
                email="test@example.com",
                password="testpassword123"
            )
            crud_user.create_user(db=db, user=sample_user)
            print("Sample user created: test@example.com / testpassword123")
        else:
            print("Database already initialized")
    except Exception as e:
        print(f"Error initializing database: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    init_db()