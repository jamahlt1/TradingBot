import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import jwt
import bcrypt
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import json
import uuid

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Float, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)

Base = declarative_base()

class UserStatus(Enum):
    """User Account Status"""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    PENDING = "pending"
    VERIFIED = "verified"

class NotificationType(Enum):
    """Notification Types"""
    TRADE_EXECUTED = "trade_executed"
    PROFIT_TARGET = "profit_target"
    LOSS_ALERT = "loss_alert"
    STRATEGY_UPDATE = "strategy_update"
    DAILY_REPORT = "daily_report"
    WEEKLY_REPORT = "weekly_report"
    MONTHLY_REPORT = "monthly_report"
    SYSTEM_ALERT = "system_alert"

@dataclass
class UserProfile:
    """User Profile Data"""
    user_id: str
    username: str
    email: str
    first_name: str
    last_name: str
    phone: Optional[str] = None
    country: Optional[str] = None
    timezone: str = "UTC"
    status: UserStatus = UserStatus.PENDING
    created_at: datetime = None
    last_login: datetime = None
    preferences: Dict[str, Any] = None
    trading_experience: str = "beginner"
    risk_tolerance: str = "moderate"
    notification_settings: Dict[str, bool] = None

@dataclass
class TradingAccount:
    """Trading Account Configuration"""
    account_id: str
    user_id: str
    account_type: str  # "metatrader", "crypto", "forex", "stock"
    broker: str
    account_number: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    server: Optional[str] = None
    is_active: bool = True
    balance: float = 0.0
    equity: float = 0.0
    margin: float = 0.0
    free_margin: float = 0.0
    leverage: int = 100
    currency: str = "USD"
    created_at: datetime = None
    last_sync: datetime = None

@dataclass
class TradingSettings:
    """Trading Settings Configuration"""
    user_id: str
    max_risk_per_trade: float = 0.02  # 2% per trade
    max_daily_loss: float = 0.05  # 5% daily loss
    max_weekly_loss: float = 0.15  # 15% weekly loss
    max_monthly_loss: float = 0.30  # 30% monthly loss
    profit_target_daily: float = 0.03  # 3% daily target
    profit_target_weekly: float = 0.10  # 10% weekly target
    profit_target_monthly: float = 0.25  # 25% monthly target
    compounding_enabled: bool = True
    compounding_rate: float = 0.1  # 10% compounding
    auto_rebalance: bool = True
    rebalance_frequency: str = "weekly"
    stop_loss_type: str = "fixed"  # "fixed", "trailing", "atr"
    take_profit_type: str = "fixed"  # "fixed", "trailing", "atr"
    max_open_positions: int = 10
    position_sizing_method: str = "kelly"  # "fixed", "kelly", "martingale"
    risk_reward_ratio: float = 2.0
    slippage_tolerance: float = 0.001  # 0.1%
    execution_speed_priority: bool = True
    news_filter_enabled: bool = True
    sentiment_filter_enabled: bool = True
    correlation_filter_enabled: bool = True

@dataclass
class StrategyAllocation:
    """Strategy Allocation Configuration"""
    allocation_id: str
    user_id: str
    strategy_name: str
    account_id: str
    allocation_percentage: float
    max_allocation: float
    min_allocation: float
    is_active: bool = True
    created_at: datetime = None
    updated_at: datetime = None
    performance_metrics: Dict[str, Any] = None

class UserManagement:
    """
    Comprehensive User Management System
    - User authentication and authorization
    - Profile management
    - Trading account integration
    - Settings and preferences
    - Notification system
    - SMTP email notifications
    """
    
    def __init__(self, database_url: str, smtp_config: Dict[str, Any] = None):
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # JWT configuration
        self.jwt_secret = "your-super-secret-jwt-key-change-in-production"
        self.jwt_algorithm = "HS256"
        self.jwt_expiry_hours = 24
        
        # SMTP configuration
        self.smtp_config = smtp_config or {
            'host': 'smtp.gmail.com',
            'port': 587,
            'username': 'your-email@gmail.com',
            'password': 'your-app-password',
            'use_tls': True
        }
        
        # User sessions
        self.active_sessions = {}
        
    async def create_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new user account"""
        try:
            session = self.SessionLocal()
            
            # Check if user already exists
            existing_user = session.query(User).filter(
                User.email == user_data['email']
            ).first()
            
            if existing_user:
                return {'success': False, 'error': 'User already exists'}
            
            # Hash password
            salt = bcrypt.gensalt()
            hashed_password = bcrypt.hashpw(user_data['password'].encode('utf-8'), salt)
            
            # Create user
            user = User(
                user_id=str(uuid.uuid4()),
                username=user_data['username'],
                email=user_data['email'],
                password_hash=hashed_password.decode('utf-8'),
                first_name=user_data.get('first_name', ''),
                last_name=user_data.get('last_name', ''),
                phone=user_data.get('phone'),
                country=user_data.get('country'),
                timezone=user_data.get('timezone', 'UTC'),
                status=UserStatus.PENDING.value,
                created_at=datetime.now(),
                trading_experience=user_data.get('trading_experience', 'beginner'),
                risk_tolerance=user_data.get('risk_tolerance', 'moderate')
            )
            
            session.add(user)
            session.commit()
            
            # Create default trading settings
            trading_settings = TradingSettings(
                user_id=user.user_id,
                max_risk_per_trade=0.02,
                max_daily_loss=0.05,
                max_weekly_loss=0.15,
                max_monthly_loss=0.30,
                profit_target_daily=0.03,
                profit_target_weekly=0.10,
                profit_target_monthly=0.25,
                compounding_enabled=True,
                compounding_rate=0.1,
                auto_rebalance=True,
                rebalance_frequency="weekly",
                stop_loss_type="fixed",
                take_profit_type="fixed",
                max_open_positions=10,
                position_sizing_method="kelly",
                risk_reward_ratio=2.0,
                slippage_tolerance=0.001,
                execution_speed_priority=True,
                news_filter_enabled=True,
                sentiment_filter_enabled=True,
                correlation_filter_enabled=True
            )
            
            session.add(trading_settings)
            session.commit()
            
            # Send welcome email
            await self.send_welcome_email(user.email, user.first_name)
            
            return {
                'success': True,
                'user_id': user.user_id,
                'message': 'User created successfully'
            }
            
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return {'success': False, 'error': str(e)}
        finally:
            session.close()
    
    async def authenticate_user(self, email: str, password: str) -> Dict[str, Any]:
        """Authenticate user and generate JWT token"""
        try:
            session = self.SessionLocal()
            
            user = session.query(User).filter(User.email == email).first()
            
            if not user:
                return {'success': False, 'error': 'Invalid credentials'}
            
            # Check password
            if not bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
                return {'success': False, 'error': 'Invalid credentials'}
            
            # Check account status
            if user.status != UserStatus.ACTIVE.value:
                return {'success': False, 'error': 'Account not active'}
            
            # Update last login
            user.last_login = datetime.now()
            session.commit()
            
            # Generate JWT token
            token = self.generate_jwt_token(user.user_id)
            
            # Store session
            self.active_sessions[user.user_id] = {
                'token': token,
                'created_at': datetime.now(),
                'last_activity': datetime.now()
            }
            
            return {
                'success': True,
                'token': token,
                'user_id': user.user_id,
                'username': user.username,
                'email': user.email
            }
            
        except Exception as e:
            logger.error(f"Error authenticating user: {e}")
            return {'success': False, 'error': str(e)}
        finally:
            session.close()
    
    def generate_jwt_token(self, user_id: str) -> str:
        """Generate JWT token for user"""
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(hours=self.jwt_expiry_hours),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    def verify_jwt_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token and return user info"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            return {'success': True, 'user_id': payload['user_id']}
        except jwt.ExpiredSignatureError:
            return {'success': False, 'error': 'Token expired'}
        except jwt.InvalidTokenError:
            return {'success': False, 'error': 'Invalid token'}
    
    async def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user profile information"""
        try:
            session = self.SessionLocal()
            
            user = session.query(User).filter(User.user_id == user_id).first()
            
            if not user:
                return {'success': False, 'error': 'User not found'}
            
            profile = UserProfile(
                user_id=user.user_id,
                username=user.username,
                email=user.email,
                first_name=user.first_name,
                last_name=user.last_name,
                phone=user.phone,
                country=user.country,
                timezone=user.timezone,
                status=UserStatus(user.status),
                created_at=user.created_at,
                last_login=user.last_login,
                preferences=json.loads(user.preferences) if user.preferences else {},
                trading_experience=user.trading_experience,
                risk_tolerance=user.risk_tolerance,
                notification_settings=json.loads(user.notification_settings) if user.notification_settings else {}
            )
            
            return {'success': True, 'profile': asdict(profile)}
            
        except Exception as e:
            logger.error(f"Error getting user profile: {e}")
            return {'success': False, 'error': str(e)}
        finally:
            session.close()
    
    async def update_user_profile(self, user_id: str, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update user profile"""
        try:
            session = self.SessionLocal()
            
            user = session.query(User).filter(User.user_id == user_id).first()
            
            if not user:
                return {'success': False, 'error': 'User not found'}
            
            # Update profile fields
            for field, value in profile_data.items():
                if hasattr(user, field):
                    setattr(user, field, value)
            
            user.updated_at = datetime.now()
            session.commit()
            
            return {'success': True, 'message': 'Profile updated successfully'}
            
        except Exception as e:
            logger.error(f"Error updating user profile: {e}")
            return {'success': False, 'error': str(e)}
        finally:
            session.close()
    
    async def add_trading_account(self, user_id: str, account_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add trading account for user"""
        try:
            session = self.SessionLocal()
            
            # Check if account already exists
            existing_account = session.query(TradingAccount).filter(
                TradingAccount.user_id == user_id,
                TradingAccount.account_number == account_data['account_number'],
                TradingAccount.broker == account_data['broker']
            ).first()
            
            if existing_account:
                return {'success': False, 'error': 'Account already exists'}
            
            # Create trading account
            account = TradingAccount(
                account_id=str(uuid.uuid4()),
                user_id=user_id,
                account_type=account_data['account_type'],
                broker=account_data['broker'],
                account_number=account_data['account_number'],
                api_key=account_data.get('api_key'),
                api_secret=account_data.get('api_secret'),
                server=account_data.get('server'),
                leverage=account_data.get('leverage', 100),
                currency=account_data.get('currency', 'USD'),
                created_at=datetime.now()
            )
            
            session.add(account)
            session.commit()
            
            return {
                'success': True,
                'account_id': account.account_id,
                'message': 'Trading account added successfully'
            }
            
        except Exception as e:
            logger.error(f"Error adding trading account: {e}")
            return {'success': False, 'error': str(e)}
        finally:
            session.close()
    
    async def get_trading_accounts(self, user_id: str) -> Dict[str, Any]:
        """Get user's trading accounts"""
        try:
            session = self.SessionLocal()
            
            accounts = session.query(TradingAccount).filter(
                TradingAccount.user_id == user_id,
                TradingAccount.is_active == True
            ).all()
            
            account_list = []
            for account in accounts:
                account_list.append({
                    'account_id': account.account_id,
                    'account_type': account.account_type,
                    'broker': account.broker,
                    'account_number': account.account_number,
                    'balance': account.balance,
                    'equity': account.equity,
                    'margin': account.margin,
                    'free_margin': account.free_margin,
                    'leverage': account.leverage,
                    'currency': account.currency,
                    'is_active': account.is_active,
                    'last_sync': account.last_sync
                })
            
            return {'success': True, 'accounts': account_list}
            
        except Exception as e:
            logger.error(f"Error getting trading accounts: {e}")
            return {'success': False, 'error': str(e)}
        finally:
            session.close()
    
    async def get_trading_settings(self, user_id: str) -> Dict[str, Any]:
        """Get user's trading settings"""
        try:
            session = self.SessionLocal()
            
            settings = session.query(TradingSettings).filter(
                TradingSettings.user_id == user_id
            ).first()
            
            if not settings:
                return {'success': False, 'error': 'Trading settings not found'}
            
            return {'success': True, 'settings': asdict(settings)}
            
        except Exception as e:
            logger.error(f"Error getting trading settings: {e}")
            return {'success': False, 'error': str(e)}
        finally:
            session.close()
    
    async def update_trading_settings(self, user_id: str, settings_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update user's trading settings"""
        try:
            session = self.SessionLocal()
            
            settings = session.query(TradingSettings).filter(
                TradingSettings.user_id == user_id
            ).first()
            
            if not settings:
                return {'success': False, 'error': 'Trading settings not found'}
            
            # Update settings fields
            for field, value in settings_data.items():
                if hasattr(settings, field):
                    setattr(settings, field, value)
            
            session.commit()
            
            return {'success': True, 'message': 'Trading settings updated successfully'}
            
        except Exception as e:
            logger.error(f"Error updating trading settings: {e}")
            return {'success': False, 'error': str(e)}
        finally:
            session.close()
    
    async def add_strategy_allocation(self, user_id: str, allocation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add strategy allocation for user"""
        try:
            session = self.SessionLocal()
            
            # Check if allocation already exists
            existing_allocation = session.query(StrategyAllocation).filter(
                StrategyAllocation.user_id == user_id,
                StrategyAllocation.strategy_name == allocation_data['strategy_name'],
                StrategyAllocation.account_id == allocation_data['account_id']
            ).first()
            
            if existing_allocation:
                return {'success': False, 'error': 'Strategy allocation already exists'}
            
            # Create strategy allocation
            allocation = StrategyAllocation(
                allocation_id=str(uuid.uuid4()),
                user_id=user_id,
                strategy_name=allocation_data['strategy_name'],
                account_id=allocation_data['account_id'],
                allocation_percentage=allocation_data['allocation_percentage'],
                max_allocation=allocation_data.get('max_allocation', 100.0),
                min_allocation=allocation_data.get('min_allocation', 0.0),
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            session.add(allocation)
            session.commit()
            
            return {
                'success': True,
                'allocation_id': allocation.allocation_id,
                'message': 'Strategy allocation added successfully'
            }
            
        except Exception as e:
            logger.error(f"Error adding strategy allocation: {e}")
            return {'success': False, 'error': str(e)}
        finally:
            session.close()
    
    async def get_strategy_allocations(self, user_id: str) -> Dict[str, Any]:
        """Get user's strategy allocations"""
        try:
            session = self.SessionLocal()
            
            allocations = session.query(StrategyAllocation).filter(
                StrategyAllocation.user_id == user_id,
                StrategyAllocation.is_active == True
            ).all()
            
            allocation_list = []
            for allocation in allocations:
                allocation_list.append({
                    'allocation_id': allocation.allocation_id,
                    'strategy_name': allocation.strategy_name,
                    'account_id': allocation.account_id,
                    'allocation_percentage': allocation.allocation_percentage,
                    'max_allocation': allocation.max_allocation,
                    'min_allocation': allocation.min_allocation,
                    'is_active': allocation.is_active,
                    'performance_metrics': json.loads(allocation.performance_metrics) if allocation.performance_metrics else {}
                })
            
            return {'success': True, 'allocations': allocation_list}
            
        except Exception as e:
            logger.error(f"Error getting strategy allocations: {e}")
            return {'success': False, 'error': str(e)}
        finally:
            session.close()
    
    async def send_notification(self, user_id: str, notification_type: NotificationType, 
                               subject: str, message: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send notification to user"""
        try:
            session = self.SessionLocal()
            
            user = session.query(User).filter(User.user_id == user_id).first()
            
            if not user:
                return {'success': False, 'error': 'User not found'}
            
            # Check notification settings
            notification_settings = json.loads(user.notification_settings) if user.notification_settings else {}
            
            if notification_type.value not in notification_settings or not notification_settings[notification_type.value]:
                return {'success': False, 'error': 'Notification type disabled'}
            
            # Send email notification
            if notification_settings.get('email_enabled', True):
                await self.send_email_notification(user.email, subject, message, data)
            
            # Store notification in database
            notification = Notification(
                notification_id=str(uuid.uuid4()),
                user_id=user_id,
                notification_type=notification_type.value,
                subject=subject,
                message=message,
                data=json.dumps(data) if data else None,
                sent_at=datetime.now(),
                is_read=False
            )
            
            session.add(notification)
            session.commit()
            
            return {'success': True, 'message': 'Notification sent successfully'}
            
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
            return {'success': False, 'error': str(e)}
        finally:
            session.close()
    
    async def send_email_notification(self, email: str, subject: str, message: str, data: Dict[str, Any] = None) -> bool:
        """Send email notification via SMTP"""
        try:
            msg = MimeMultipart()
            msg['From'] = self.smtp_config['username']
            msg['To'] = email
            msg['Subject'] = subject
            
            # Create HTML message
            html_content = f"""
            <html>
            <body>
                <h2>{subject}</h2>
                <p>{message}</p>
            """
            
            if data:
                html_content += "<h3>Details:</h3><ul>"
                for key, value in data.items():
                    html_content += f"<li><strong>{key}:</strong> {value}</li>"
                html_content += "</ul>"
            
            html_content += """
                <br><br>
                <p>Best regards,<br>Trading Bot Platform</p>
            </body>
            </html>
            """
            
            msg.attach(MimeText(html_content, 'html'))
            
            # Send email
            with smtplib.SMTP(self.smtp_config['host'], self.smtp_config['port']) as server:
                if self.smtp_config['use_tls']:
                    server.starttls()
                server.login(self.smtp_config['username'], self.smtp_config['password'])
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
            return False
    
    async def send_welcome_email(self, email: str, first_name: str) -> bool:
        """Send welcome email to new user"""
        subject = "Welcome to Trading Bot Platform"
        message = f"Hi {first_name}, welcome to our advanced trading platform!"
        
        data = {
            'platform_features': [
                'Multi-strategy trading',
                'Advanced ML algorithms',
                'Real-time market analysis',
                'Risk management tools',
                'Performance tracking'
            ]
        }
        
        return await self.send_email_notification(email, subject, message, data)
    
    async def logout_user(self, user_id: str) -> Dict[str, Any]:
        """Logout user and invalidate session"""
        try:
            if user_id in self.active_sessions:
                del self.active_sessions[user_id]
            
            return {'success': True, 'message': 'Logged out successfully'}
            
        except Exception as e:
            logger.error(f"Error logging out user: {e}")
            return {'success': False, 'error': str(e)}


# Database Models
class User(Base):
    __tablename__ = "users"
    
    user_id = Column(String, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    first_name = Column(String)
    last_name = Column(String)
    phone = Column(String)
    country = Column(String)
    timezone = Column(String, default="UTC")
    status = Column(String, default=UserStatus.PENDING.value)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    last_login = Column(DateTime)
    preferences = Column(Text)  # JSON string
    trading_experience = Column(String, default="beginner")
    risk_tolerance = Column(String, default="moderate")
    notification_settings = Column(Text)  # JSON string

class TradingAccount(Base):
    __tablename__ = "trading_accounts"
    
    account_id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    account_type = Column(String, nullable=False)
    broker = Column(String, nullable=False)
    account_number = Column(String, nullable=False)
    api_key = Column(String)
    api_secret = Column(String)
    server = Column(String)
    is_active = Column(Boolean, default=True)
    balance = Column(Float, default=0.0)
    equity = Column(Float, default=0.0)
    margin = Column(Float, default=0.0)
    free_margin = Column(Float, default=0.0)
    leverage = Column(Integer, default=100)
    currency = Column(String, default="USD")
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    last_sync = Column(DateTime)

class TradingSettings(Base):
    __tablename__ = "trading_settings"
    
    user_id = Column(String, primary_key=True)
    max_risk_per_trade = Column(Float, default=0.02)
    max_daily_loss = Column(Float, default=0.05)
    max_weekly_loss = Column(Float, default=0.15)
    max_monthly_loss = Column(Float, default=0.30)
    profit_target_daily = Column(Float, default=0.03)
    profit_target_weekly = Column(Float, default=0.10)
    profit_target_monthly = Column(Float, default=0.25)
    compounding_enabled = Column(Boolean, default=True)
    compounding_rate = Column(Float, default=0.1)
    auto_rebalance = Column(Boolean, default=True)
    rebalance_frequency = Column(String, default="weekly")
    stop_loss_type = Column(String, default="fixed")
    take_profit_type = Column(String, default="fixed")
    max_open_positions = Column(Integer, default=10)
    position_sizing_method = Column(String, default="kelly")
    risk_reward_ratio = Column(Float, default=2.0)
    slippage_tolerance = Column(Float, default=0.001)
    execution_speed_priority = Column(Boolean, default=True)
    news_filter_enabled = Column(Boolean, default=True)
    sentiment_filter_enabled = Column(Boolean, default=True)
    correlation_filter_enabled = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

class StrategyAllocation(Base):
    __tablename__ = "strategy_allocations"
    
    allocation_id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    strategy_name = Column(String, nullable=False)
    account_id = Column(String, nullable=False)
    allocation_percentage = Column(Float, nullable=False)
    max_allocation = Column(Float, default=100.0)
    min_allocation = Column(Float, default=0.0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    performance_metrics = Column(Text)  # JSON string

class Notification(Base):
    __tablename__ = "notifications"
    
    notification_id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    notification_type = Column(String, nullable=False)
    subject = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    data = Column(Text)  # JSON string
    sent_at = Column(DateTime, default=datetime.now)
    is_read = Column(Boolean, default=False)