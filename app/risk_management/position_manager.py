import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class PositionStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"
    HEDGED = "hedged"
    EMERGENCY_CLOSED = "emergency_closed"

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Position:
    """Position information"""
    id: str
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    current_price: float
    entry_time: datetime
    status: PositionStatus
    pnl: float = 0.0
    pnl_percentage: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_level: RiskLevel = RiskLevel.MEDIUM
    hedging_position: Optional[str] = None

class PositionManager:
    """
    Advanced Position Management System
    - Real-time position monitoring
    - Emergency position closing
    - Risk-based position sizing
    - Hedging strategies
    - Portfolio risk management
    """
    
    def __init__(self,
                 max_portfolio_risk: float = 0.02,  # 2% max portfolio risk
                 max_position_risk: float = 0.01,   # 1% max position risk
                 emergency_stop_loss: float = 0.05, # 5% emergency stop loss
                 max_correlation: float = 0.8,      # Max correlation between positions
                 hedging_enabled: bool = True,
                 auto_hedging: bool = False):
        
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_risk = max_position_risk
        self.emergency_stop_loss = emergency_stop_loss
        self.max_correlation = max_correlation
        self.hedging_enabled = hedging_enabled
        self.auto_hedging = auto_hedging
        
        self.positions = {}
        self.risk_alerts = []
        self.hedging_positions = {}
        self.portfolio_value = 0.0
        self.cash_balance = 0.0
        
    def add_position(self, position: Position) -> bool:
        """Add a new position with risk checks"""
        try:
            # Check portfolio risk limits
            if not self._check_portfolio_risk(position):
                logger.warning(f"Position {position.id} exceeds portfolio risk limits")
                return False
            
            # Check correlation with existing positions
            if not self._check_correlation_risk(position):
                logger.warning(f"Position {position.id} has high correlation with existing positions")
                return False
            
            # Add position
            self.positions[position.id] = position
            logger.info(f"Added position {position.id}: {position.symbol} {position.side} {position.size}")
            
            # Update portfolio metrics
            self._update_portfolio_metrics()
            
            # Check for auto-hedging
            if self.auto_hedging:
                self._check_auto_hedging(position)
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding position: {e}")
            return False
    
    def update_position(self, position_id: str, current_price: float) -> Dict[str, Any]:
        """Update position with current price and check risk"""
        try:
            if position_id not in self.positions:
                return {'status': 'error', 'message': 'Position not found'}
            
            position = self.positions[position_id]
            position.current_price = current_price
            
            # Calculate P&L
            if position.side == 'long':
                position.pnl = (current_price - position.entry_price) * position.size
            else:
                position.pnl = (position.entry_price - current_price) * position.size
            
            position.pnl_percentage = (position.pnl / (position.entry_price * position.size)) * 100
            
            # Check risk levels
            risk_check = self._check_position_risk(position)
            
            # Update portfolio metrics
            self._update_portfolio_metrics()
            
            return {
                'status': 'updated',
                'position': position,
                'risk_check': risk_check
            }
            
        except Exception as e:
            logger.error(f"Error updating position: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def close_position(self, position_id: str, reason: str = "manual") -> Dict[str, Any]:
        """Close a position"""
        try:
            if position_id not in self.positions:
                return {'status': 'error', 'message': 'Position not found'}
            
            position = self.positions[position_id]
            position.status = PositionStatus.CLOSED
            
            # Close hedging position if exists
            if position.hedging_position:
                self._close_hedging_position(position.hedging_position)
            
            # Remove from active positions
            del self.positions[position_id]
            
            logger.info(f"Closed position {position_id}: {reason}")
            
            return {
                'status': 'closed',
                'position': position,
                'reason': reason
            }
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def emergency_close_all(self, reason: str = "emergency") -> Dict[str, Any]:
        """Emergency close all positions"""
        try:
            closed_positions = []
            
            for position_id, position in list(self.positions.items()):
                result = self.close_position(position_id, f"{reason}_emergency")
                if result['status'] == 'closed':
                    closed_positions.append(result['position'])
            
            # Close all hedging positions
            for hedge_id in list(self.hedging_positions.keys()):
                self._close_hedging_position(hedge_id)
            
            logger.warning(f"Emergency closed {len(closed_positions)} positions: {reason}")
            
            return {
                'status': 'emergency_closed',
                'closed_positions': closed_positions,
                'reason': reason
            }
            
        except Exception as e:
            logger.error(f"Error in emergency close: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def create_hedge(self, position_id: str, hedge_type: str = "correlation") -> Dict[str, Any]:
        """Create a hedging position"""
        try:
            if position_id not in self.positions:
                return {'status': 'error', 'message': 'Position not found'}
            
            position = self.positions[position_id]
            
            # Calculate hedge parameters
            hedge_params = self._calculate_hedge_parameters(position, hedge_type)
            
            if not hedge_params:
                return {'status': 'error', 'message': 'Unable to calculate hedge parameters'}
            
            # Create hedging position
            hedge_position = Position(
                id=f"hedge_{position_id}",
                symbol=hedge_params['symbol'],
                side='short' if position.side == 'long' else 'long',
                size=hedge_params['size'],
                entry_price=hedge_params['entry_price'],
                current_price=hedge_params['entry_price'],
                entry_time=datetime.now(),
                status=PositionStatus.OPEN,
                risk_level=RiskLevel.LOW
            )
            
            # Add hedging position
            self.hedging_positions[hedge_position.id] = hedge_position
            position.hedging_position = hedge_position.id
            
            logger.info(f"Created hedge for position {position_id}: {hedge_params}")
            
            return {
                'status': 'hedged',
                'hedge_position': hedge_position,
                'hedge_params': hedge_params
            }
            
        except Exception as e:
            logger.error(f"Error creating hedge: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _calculate_hedge_parameters(self, position: Position, hedge_type: str) -> Optional[Dict[str, Any]]:
        """Calculate hedge parameters"""
        try:
            if hedge_type == "correlation":
                # Find correlated asset for hedging
                correlated_symbol = self._find_correlated_asset(position.symbol)
                if correlated_symbol:
                    return {
                        'symbol': correlated_symbol,
                        'size': position.size * 0.8,  # 80% hedge ratio
                        'entry_price': self._get_current_price(correlated_symbol)
                    }
            
            elif hedge_type == "beta":
                # Beta-based hedging
                beta = self._calculate_beta(position.symbol)
                if beta is not None:
                    return {
                        'symbol': 'SPY',  # Market proxy
                        'size': position.size * beta * 0.9,
                        'entry_price': self._get_current_price('SPY')
                    }
            
            elif hedge_type == "options":
                # Options-based hedging
                return {
                    'symbol': f"{position.symbol}_PUT",
                    'size': position.size * 0.1,  # 10% options hedge
                    'entry_price': self._get_option_price(position.symbol, 'PUT')
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error calculating hedge parameters: {e}")
            return None
    
    def _check_portfolio_risk(self, new_position: Position) -> bool:
        """Check if new position exceeds portfolio risk limits"""
        try:
            # Calculate current portfolio risk
            total_risk = sum(pos.size * pos.entry_price for pos in self.positions.values())
            new_risk = new_position.size * new_position.entry_price
            
            # Check if adding new position exceeds limits
            if (total_risk + new_risk) / self.portfolio_value > self.max_portfolio_risk:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking portfolio risk: {e}")
            return False
    
    def _check_correlation_risk(self, new_position: Position) -> bool:
        """Check correlation risk with existing positions"""
        try:
            for position in self.positions.values():
                correlation = self._calculate_correlation(new_position.symbol, position.symbol)
                if abs(correlation) > self.max_correlation:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking correlation risk: {e}")
            return True  # Allow if correlation calculation fails
    
    def _check_position_risk(self, position: Position) -> Dict[str, Any]:
        """Check individual position risk"""
        risk_check = {
            'risk_level': RiskLevel.LOW,
            'alerts': [],
            'recommendations': []
        }
        
        try:
            # Check stop loss
            if position.stop_loss:
                if (position.side == 'long' and position.current_price <= position.stop_loss) or \
                   (position.side == 'short' and position.current_price >= position.stop_loss):
                    risk_check['alerts'].append('Stop loss triggered')
                    risk_check['recommendations'].append('Close position immediately')
                    risk_check['risk_level'] = RiskLevel.CRITICAL
            
            # Check emergency stop loss
            if abs(position.pnl_percentage) > self.emergency_stop_loss * 100:
                risk_check['alerts'].append('Emergency stop loss triggered')
                risk_check['recommendations'].append('Emergency close position')
                risk_check['risk_level'] = RiskLevel.CRITICAL
            
            # Check for large losses
            if position.pnl_percentage < -10:
                risk_check['alerts'].append('Large loss detected')
                risk_check['recommendations'].append('Consider closing or hedging')
                risk_check['risk_level'] = RiskLevel.HIGH
            
            # Check for large gains
            if position.pnl_percentage > 20:
                risk_check['alerts'].append('Large gain detected')
                risk_check['recommendations'].append('Consider taking profits')
                risk_check['risk_level'] = RiskLevel.MEDIUM
            
            # Update position risk level
            position.risk_level = risk_check['risk_level']
            
            return risk_check
            
        except Exception as e:
            logger.error(f"Error checking position risk: {e}")
            return risk_check
    
    def _update_portfolio_metrics(self):
        """Update portfolio metrics"""
        try:
            total_value = sum(pos.size * pos.current_price for pos in self.positions.values())
            total_pnl = sum(pos.pnl for pos in self.positions.values())
            
            self.portfolio_value = total_value + self.cash_balance
            
            # Calculate portfolio risk metrics
            if self.portfolio_value > 0:
                portfolio_return = (total_pnl / self.portfolio_value) * 100
                portfolio_risk = self._calculate_portfolio_risk()
                
                logger.info(f"Portfolio: Value=${self.portfolio_value:.2f}, PnL=${total_pnl:.2f}, Return={portfolio_return:.2f}%")
            
        except Exception as e:
            logger.error(f"Error updating portfolio metrics: {e}")
    
    def _calculate_portfolio_risk(self) -> float:
        """Calculate portfolio risk using VaR"""
        try:
            if not self.positions:
                return 0.0
            
            # Calculate position weights and returns
            weights = []
            returns = []
            
            for position in self.positions.values():
                weight = (position.size * position.current_price) / self.portfolio_value
                weights.append(weight)
                returns.append(position.pnl_percentage / 100)
            
            # Calculate portfolio variance
            portfolio_variance = np.sum(np.array(weights) ** 2 * np.array(returns) ** 2)
            
            # Calculate VaR (95% confidence)
            var_95 = np.sqrt(portfolio_variance) * 1.645
            
            return var_95
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            return 0.0
    
    def _find_correlated_asset(self, symbol: str) -> Optional[str]:
        """Find correlated asset for hedging"""
        # This would use real correlation data
        # For now, return common hedging pairs
        hedging_pairs = {
            'BTC': 'ETH',
            'ETH': 'BTC',
            'AAPL': 'SPY',
            'GOOGL': 'SPY',
            'TSLA': 'SPY'
        }
        
        return hedging_pairs.get(symbol.split('/')[0])
    
    def _calculate_beta(self, symbol: str) -> Optional[float]:
        """Calculate beta for a symbol"""
        # This would calculate real beta
        # For now, return estimated betas
        betas = {
            'BTC': 2.5,
            'ETH': 2.0,
            'AAPL': 1.2,
            'GOOGL': 1.1,
            'TSLA': 2.0
        }
        
        return betas.get(symbol.split('/')[0])
    
    def _calculate_correlation(self, symbol1: str, symbol2: str) -> float:
        """Calculate correlation between two symbols"""
        # This would use real correlation data
        # For now, return estimated correlations
        correlations = {
            ('BTC', 'ETH'): 0.8,
            ('AAPL', 'GOOGL'): 0.7,
            ('AAPL', 'SPY'): 0.6,
            ('GOOGL', 'SPY'): 0.6
        }
        
        key = (symbol1.split('/')[0], symbol2.split('/')[0])
        return correlations.get(key, 0.0)
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        # This would fetch real-time prices
        # For now, return simulated prices
        base_prices = {
            'BTC': 50000,
            'ETH': 3000,
            'AAPL': 150,
            'GOOGL': 2800,
            'TSLA': 200,
            'SPY': 400
        }
        
        return base_prices.get(symbol.split('/')[0], 100)
    
    def _get_option_price(self, symbol: str, option_type: str) -> float:
        """Get option price"""
        # This would fetch real option prices
        # For now, return estimated prices
        base_price = self._get_current_price(symbol)
        return base_price * 0.05  # 5% of underlying price
    
    def _close_hedging_position(self, hedge_id: str):
        """Close a hedging position"""
        try:
            if hedge_id in self.hedging_positions:
                hedge_position = self.hedging_positions[hedge_id]
                hedge_position.status = PositionStatus.CLOSED
                del self.hedging_positions[hedge_id]
                logger.info(f"Closed hedging position {hedge_id}")
        except Exception as e:
            logger.error(f"Error closing hedging position: {e}")
    
    def _check_auto_hedging(self, position: Position):
        """Check if auto-hedging is needed"""
        try:
            # Auto-hedge large positions
            position_value = position.size * position.entry_price
            if position_value / self.portfolio_value > 0.1:  # 10% of portfolio
                self.create_hedge(position.id, "correlation")
        except Exception as e:
            logger.error(f"Error in auto-hedging check: {e}")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary"""
        try:
            total_positions = len(self.positions)
            total_hedges = len(self.hedging_positions)
            total_pnl = sum(pos.pnl for pos in self.positions.values())
            total_pnl_percentage = (total_pnl / self.portfolio_value) * 100 if self.portfolio_value > 0 else 0
            
            risk_levels = {
                RiskLevel.LOW: len([p for p in self.positions.values() if p.risk_level == RiskLevel.LOW]),
                RiskLevel.MEDIUM: len([p for p in self.positions.values() if p.risk_level == RiskLevel.MEDIUM]),
                RiskLevel.HIGH: len([p for p in self.positions.values() if p.risk_level == RiskLevel.HIGH]),
                RiskLevel.CRITICAL: len([p for p in self.positions.values() if p.risk_level == RiskLevel.CRITICAL])
            }
            
            return {
                'total_positions': total_positions,
                'total_hedges': total_hedges,
                'portfolio_value': self.portfolio_value,
                'cash_balance': self.cash_balance,
                'total_pnl': total_pnl,
                'total_pnl_percentage': total_pnl_percentage,
                'risk_levels': risk_levels,
                'alerts': len(self.risk_alerts)
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return {}
    
    def get_risk_alerts(self) -> List[Dict[str, Any]]:
        """Get current risk alerts"""
        alerts = []
        
        try:
            for position in self.positions.values():
                risk_check = self._check_position_risk(position)
                if risk_check['alerts']:
                    alerts.append({
                        'position_id': position.id,
                        'symbol': position.symbol,
                        'alerts': risk_check['alerts'],
                        'recommendations': risk_check['recommendations'],
                        'risk_level': risk_check['risk_level']
                    })
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error getting risk alerts: {e}")
            return []