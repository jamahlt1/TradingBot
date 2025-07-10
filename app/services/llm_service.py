#!/usr/bin/env python3
"""
LLM Service Layer
Integrates OpenRouter LLM with the trading system
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import uuid
from dataclasses import dataclass, asdict

from app.integrations.openrouter_client import OpenRouterClient, ChatMessage
from app.database.database import get_db
from app.models.models import User, Strategy, Account, Trade
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

@dataclass
class ChatSession:
    id: str
    user_id: int
    messages: List[ChatMessage]
    context: Dict
    created_at: datetime
    updated_at: datetime
    
class LLMService:
    """
    LLM Service for trading system integration
    Provides AI-powered trading assistance and automation
    """
    
    def __init__(self, openrouter_client: OpenRouterClient, trading_app_state: Dict):
        self.llm_client = openrouter_client
        self.app_state = trading_app_state
        self.chat_sessions: Dict[str, ChatSession] = {}
        self.active_automations: Dict[str, Dict] = {}
        
    async def create_chat_session(self, user_id: int, initial_context: Dict = None) -> str:
        """Create new chat session"""
        session_id = str(uuid.uuid4())
        
        # System message with trading context
        system_message = ChatMessage(
            role='system',
            content=self._build_system_prompt(initial_context or {})
        )
        
        session = ChatSession(
            id=session_id,
            user_id=user_id,
            messages=[system_message],
            context=initial_context or {},
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.chat_sessions[session_id] = session
        return session_id
    
    def _build_system_prompt(self, context: Dict) -> str:
        """Build comprehensive system prompt with trading context"""
        available_strategies = list(self.app_state.get('strategies', {}).keys())
        available_integrations = list(self.app_state.get('integrations', {}).keys())
        
        return f"""You are an advanced AI trading assistant with full access to a comprehensive trading platform. You have the following capabilities:

TRADING SYSTEM ACCESS:
- Available Strategies: {', '.join(available_strategies)}
- Broker Integrations: {', '.join(available_integrations)}
- Risk Management: Real-time risk monitoring and controls
- Portfolio Management: Position tracking and optimization
- Market Data: Real-time and historical data access

EXECUTION CAPABILITIES:
- Place, modify, and cancel orders across all connected brokers
- Execute complex multi-leg strategies
- Manage position sizing and risk parameters
- Run backtests and optimizations
- Generate performance reports

ANALYSIS CAPABILITIES:
- Technical and fundamental analysis
- Market sentiment analysis
- Strategy performance evaluation
- Risk assessment and monitoring
- Forecast generation using ML and statistical models

AUTOMATION FEATURES:
- Strategy building through conversation
- Parameter optimization based on performance
- Automated trade execution with risk controls
- Real-time monitoring and alerts
- Document analysis for trading plans

USER CONTEXT:
{json.dumps(context, indent=2, default=str)}

GUIDELINES:
1. Always prioritize risk management and user safety
2. Confirm high-risk actions before execution
3. Provide clear reasoning for all recommendations
4. Use available data sources for accurate analysis
5. Be transparent about limitations and uncertainties
6. Maintain professional trading standards

You can execute trades, build strategies, analyze markets, and automate trading processes. Always ask for confirmation on significant actions that could impact account value."""

    async def chat(self, session_id: str, message: str, user_context: Dict = None) -> Dict:
        """Process chat message and return response"""
        try:
            session = self.chat_sessions.get(session_id)
            if not session:
                return {'error': 'Session not found'}
            
            # Add user message
            user_message = ChatMessage(role='user', content=message)
            session.messages.append(user_message)
            
            # Update context if provided
            if user_context:
                session.context.update(user_context)
            
            # Check if this is a trade command
            trade_command = await self._detect_trade_command(message)
            if trade_command:
                return await self._handle_trade_command(session, trade_command)
            
            # Check if this is a strategy building request
            strategy_request = await self._detect_strategy_request(message)
            if strategy_request:
                return await self._handle_strategy_request(session, strategy_request)
            
            # Regular chat completion
            response = await self.llm_client.chat_completion(
                messages=session.messages[-10:],  # Keep last 10 messages for context
                model='claude-3-sonnet'
            )
            
            if response:
                assistant_message = ChatMessage(role='assistant', content=response.content)
                session.messages.append(assistant_message)
                session.updated_at = datetime.now()
                
                return {
                    'response': response.content,
                    'model': response.model,
                    'tokens_used': response.tokens_used,
                    'cost': response.cost,
                    'session_id': session_id,
                    'timestamp': response.timestamp.isoformat()
                }
            
            return {'error': 'Failed to generate response'}
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return {'error': str(e)}
    
    async def _detect_trade_command(self, message: str) -> Optional[str]:
        """Detect if message contains trade commands"""
        trade_keywords = [
            'buy', 'sell', 'close', 'order', 'position', 'trade',
            'long', 'short', 'execute', 'place order', 'market order',
            'limit order', 'stop loss', 'take profit'
        ]
        
        message_lower = message.lower()
        for keyword in trade_keywords:
            if keyword in message_lower:
                return 'trade_command'
        
        return None
    
    async def _detect_strategy_request(self, message: str) -> Optional[str]:
        """Detect if message is a strategy building request"""
        strategy_keywords = [
            'strategy', 'build strategy', 'create strategy', 'new strategy',
            'optimize', 'backtest', 'parameters', 'trading plan'
        ]
        
        message_lower = message.lower()
        for keyword in strategy_keywords:
            if keyword in message_lower:
                return 'strategy_request'
        
        return None
    
    async def _handle_trade_command(self, session: ChatSession, command_type: str) -> Dict:
        """Handle trade execution commands"""
        try:
            # Get current user context
            context = await self._get_user_trading_context(session.user_id)
            
            # Parse trade command
            trade_instruction = await self.llm_client.execute_trade_command(
                session.messages[-1].content,
                context
            )
            
            if not trade_instruction:
                return {'error': 'Could not parse trade command'}
            
            # Risk check
            if trade_instruction.get('risk_check') == 'rejected':
                return {
                    'response': f"Trade command rejected due to risk concerns: {trade_instruction.get('reasoning')}",
                    'trade_rejected': True
                }
            
            # Execute if approved or request confirmation
            if trade_instruction.get('risk_check') == 'approved':
                execution_result = await self._execute_trade(trade_instruction, session.user_id)
                response_text = f"Trade executed: {trade_instruction.get('reasoning')}\nResult: {execution_result}"
            else:
                response_text = f"Trade command parsed. Confirm execution:\n{json.dumps(trade_instruction, indent=2)}\nReply 'confirm' to execute."
            
            # Add response to session
            assistant_message = ChatMessage(role='assistant', content=response_text)
            session.messages.append(assistant_message)
            
            return {
                'response': response_text,
                'trade_instruction': trade_instruction,
                'session_id': session.id
            }
            
        except Exception as e:
            logger.error(f"Trade command handling error: {e}")
            return {'error': str(e)}
    
    async def _handle_strategy_request(self, session: ChatSession, request_type: str) -> Dict:
        """Handle strategy building and optimization requests"""
        try:
            # Extract strategy requirements from conversation
            requirements = await self._extract_strategy_requirements(session.messages)
            
            # Build strategy using LLM
            strategy_data = await self.llm_client.build_trading_strategy(requirements)
            
            if strategy_data:
                # Save strategy to database
                strategy_id = await self._save_strategy(strategy_data, session.user_id)
                
                response_text = f"Strategy created successfully!\n\nStrategy: {strategy_data.get('strategy_name')}\nDescription: {strategy_data.get('description')}\n\nStrategy ID: {strategy_id}\n\nWould you like to backtest this strategy or modify any parameters?"
                
                assistant_message = ChatMessage(role='assistant', content=response_text)
                session.messages.append(assistant_message)
                
                return {
                    'response': response_text,
                    'strategy_data': strategy_data,
                    'strategy_id': strategy_id,
                    'session_id': session.id
                }
            
            return {'error': 'Failed to build strategy'}
            
        except Exception as e:
            logger.error(f"Strategy request handling error: {e}")
            return {'error': str(e)}
    
    async def _get_user_trading_context(self, user_id: int) -> Dict:
        """Get comprehensive trading context for user"""
        try:
            db = next(get_db())
            
            # Get user accounts
            accounts = db.query(Account).filter(Account.user_id == user_id).all()
            
            # Get active positions
            positions = []
            orders = []
            
            # Calculate total balance
            total_balance = sum(acc.balance for acc in accounts)
            
            # Get recent trades
            recent_trades = db.query(Trade).filter(
                Trade.user_id == user_id,
                Trade.created_at >= datetime.now() - timedelta(days=7)
            ).all()
            
            return {
                'user_id': user_id,
                'accounts': [{'id': acc.id, 'name': acc.name, 'balance': acc.balance, 'type': acc.account_type} for acc in accounts],
                'total_balance': total_balance,
                'positions': positions,
                'orders': orders,
                'recent_trades': [{'symbol': t.symbol, 'side': t.side, 'quantity': t.quantity, 'price': t.price} for t in recent_trades[-10:]],
                'risk_limits': {
                    'max_position_size': total_balance * 0.1,  # 10% max position
                    'daily_loss_limit': total_balance * 0.02,  # 2% daily loss limit
                    'max_leverage': 3.0
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting user context: {e}")
            return {}
    
    async def _execute_trade(self, trade_instruction: Dict, user_id: int) -> Dict:
        """Execute trade through appropriate broker integration"""
        try:
            # This would integrate with the actual broker clients
            # For now, return a simulation
            
            return {
                'status': 'executed',
                'order_id': f"ORDER_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'symbol': trade_instruction.get('symbol'),
                'action': trade_instruction.get('action'),
                'quantity': trade_instruction.get('quantity'),
                'price': trade_instruction.get('price'),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def _extract_strategy_requirements(self, messages: List[ChatMessage]) -> Dict:
        """Extract strategy requirements from conversation"""
        # Analyze recent messages to extract requirements
        recent_messages = messages[-5:]  # Last 5 messages
        
        conversation_text = "\n".join([f"{msg.role}: {msg.content}" for msg in recent_messages])
        
        # Default requirements that can be overridden
        requirements = {
            'risk_tolerance': 'medium',
            'timeframe': '1d',
            'market_type': 'stocks',
            'strategy_type': 'trend_following',
            'max_drawdown': 15,
            'target_return': 20,
            'capital_allocation': 100
        }
        
        # Parse specific requirements from conversation
        # This could be enhanced with better NLP
        if 'scalping' in conversation_text.lower():
            requirements['strategy_type'] = 'scalping'
            requirements['timeframe'] = '1m'
        elif 'swing' in conversation_text.lower():
            requirements['strategy_type'] = 'swing_trading'
            requirements['timeframe'] = '1d'
        elif 'crypto' in conversation_text.lower():
            requirements['market_type'] = 'crypto'
        elif 'forex' in conversation_text.lower():
            requirements['market_type'] = 'forex'
        
        return requirements
    
    async def _save_strategy(self, strategy_data: Dict, user_id: int) -> str:
        """Save strategy to database"""
        try:
            db = next(get_db())
            
            strategy = Strategy(
                user_id=user_id,
                name=strategy_data.get('strategy_name', 'AI Generated Strategy'),
                description=strategy_data.get('description', ''),
                parameters=json.dumps(strategy_data.get('parameters', {})),
                strategy_type=strategy_data.get('strategy_type', 'custom'),
                is_active=False,
                created_at=datetime.now()
            )
            
            db.add(strategy)
            db.commit()
            db.refresh(strategy)
            
            return str(strategy.id)
            
        except Exception as e:
            logger.error(f"Error saving strategy: {e}")
            return None
    
    async def analyze_market_data_for_user(self, user_id: int, symbol: str, context: str = "") -> Dict:
        """Analyze market data for specific user"""
        try:
            # Get market data from integrations
            market_data = {}
            
            # Try different data sources
            for integration_name, integration in self.app_state.get('integrations', {}).items():
                try:
                    if hasattr(integration, 'get_market_data'):
                        data = await integration.get_market_data(symbol)
                        if data:
                            market_data[integration_name] = data
                except Exception as e:
                    logger.warning(f"Failed to get data from {integration_name}: {e}")
            
            if not market_data:
                return {'error': 'No market data available'}
            
            # Analyze with LLM
            analysis = await self.llm_client.analyze_market_data(market_data, context)
            
            return {
                'symbol': symbol,
                'analysis': analysis,
                'data_sources': list(market_data.keys()),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Market analysis error: {e}")
            return {'error': str(e)}
    
    async def forecast_market_direction_for_user(self, user_id: int, symbol: str, 
                                                external_factors: Dict = None) -> Dict:
        """Generate market forecast for user"""
        try:
            # Get historical data
            historical_data = []
            
            for integration_name, integration in self.app_state.get('integrations', {}).items():
                try:
                    if hasattr(integration, 'get_historical_data'):
                        data = await integration.get_historical_data(symbol, '1d', 100)
                        if data:
                            historical_data = data
                            break
                except Exception as e:
                    logger.warning(f"Failed to get historical data from {integration_name}: {e}")
            
            if not historical_data:
                return {'error': 'No historical data available'}
            
            # Generate forecast
            forecast = await self.llm_client.forecast_market_direction(
                historical_data, external_factors
            )
            
            return {
                'symbol': symbol,
                'forecast': forecast,
                'data_points': len(historical_data),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Forecasting error: {e}")
            return {'error': str(e)}
    
    async def optimize_user_strategy(self, user_id: int, strategy_id: str) -> Dict:
        """Optimize existing strategy for user"""
        try:
            db = next(get_db())
            
            # Get strategy
            strategy = db.query(Strategy).filter(
                Strategy.id == strategy_id,
                Strategy.user_id == user_id
            ).first()
            
            if not strategy:
                return {'error': 'Strategy not found'}
            
            # Get strategy performance data
            performance_data = await self._get_strategy_performance(strategy_id)
            
            # Optimize with LLM
            strategy_dict = {
                'name': strategy.name,
                'parameters': json.loads(strategy.parameters),
                'type': strategy.strategy_type
            }
            
            optimization_result = await self.llm_client.optimize_strategy_parameters(
                strategy_dict, performance_data
            )
            
            return {
                'strategy_id': strategy_id,
                'optimization_result': optimization_result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Strategy optimization error: {e}")
            return {'error': str(e)}
    
    async def _get_strategy_performance(self, strategy_id: str) -> Dict:
        """Get strategy performance metrics"""
        # This would get actual performance data from backtesting/live trading
        # For now, return sample data
        return {
            'total_return': 15.5,
            'sharpe_ratio': 1.2,
            'max_drawdown': -8.3,
            'win_rate': 0.65,
            'profit_factor': 1.8,
            'total_trades': 45,
            'avg_trade_duration': '2.3 days'
        }
    
    async def analyze_uploaded_plan(self, user_id: int, plan_text: str) -> Dict:
        """Analyze uploaded trading plan"""
        try:
            analysis = await self.llm_client.analyze_trading_plan(plan_text)
            
            return {
                'user_id': user_id,
                'analysis': analysis,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Plan analysis error: {e}")
            return {'error': str(e)}
    
    async def generate_user_report(self, user_id: int, period: str = "daily") -> Dict:
        """Generate comprehensive trading report for user"""
        try:
            # Get user account data
            context = await self._get_user_trading_context(user_id)
            
            # Generate report with LLM
            report = await self.llm_client.generate_trading_report(context, period)
            
            return {
                'user_id': user_id,
                'period': period,
                'report': report,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Report generation error: {e}")
            return {'error': str(e)}
    
    async def start_automation(self, user_id: int, automation_config: Dict) -> str:
        """Start automated trading based on LLM instructions"""
        automation_id = str(uuid.uuid4())
        
        self.active_automations[automation_id] = {
            'user_id': user_id,
            'config': automation_config,
            'status': 'active',
            'created_at': datetime.now(),
            'last_action': None
        }
        
        # Start automation task
        asyncio.create_task(self._run_automation(automation_id))
        
        return automation_id
    
    async def _run_automation(self, automation_id: str):
        """Run automated trading logic"""
        try:
            automation = self.active_automations.get(automation_id)
            if not automation:
                return
            
            # This would implement the actual automation logic
            # For now, just log that automation is running
            logger.info(f"Running automation {automation_id}")
            
            # Automation would continuously:
            # 1. Analyze market conditions
            # 2. Check strategy signals
            # 3. Execute trades based on LLM recommendations
            # 4. Monitor and adjust parameters
            
        except Exception as e:
            logger.error(f"Automation error: {e}")
    
    def get_chat_session(self, session_id: str) -> Optional[ChatSession]:
        """Get chat session by ID"""
        return self.chat_sessions.get(session_id)
    
    def get_user_sessions(self, user_id: int) -> List[ChatSession]:
        """Get all chat sessions for user"""
        return [session for session in self.chat_sessions.values() if session.user_id == user_id]