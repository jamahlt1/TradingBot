import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class HedgingStrategy(BaseStrategy):
    """
    Advanced Hedging Strategy
    - Portfolio risk management
    - Correlation analysis
    - Dynamic hedging ratios
    - Beta calculation
    - Risk-adjusted hedging
    """
    
    def __init__(self,
                 correlation_threshold: float = 0.7,
                 beta_threshold: float = 0.8,
                 max_hedge_ratio: float = 1.0,
                 min_hedge_ratio: float = 0.1,
                 risk_adjustment_factor: float = 0.5,
                 rebalance_frequency_hours: int = 4,
                 volatility_lookback: int = 20,
                 correlation_lookback: int = 60,
                 hedge_instruments: List[str] = None,
                 max_portfolio_risk: float = 0.05):
        
        super().__init__()
        self.correlation_threshold = correlation_threshold
        self.beta_threshold = beta_threshold
        self.max_hedge_ratio = max_hedge_ratio
        self.min_hedge_ratio = min_hedge_ratio
        self.risk_adjustment_factor = risk_adjustment_factor
        self.rebalance_frequency_hours = rebalance_frequency_hours
        self.volatility_lookback = volatility_lookback
        self.correlation_lookback = correlation_lookback
        self.hedge_instruments = hedge_instruments or ['SPY', 'QQQ', 'VIX']
        self.max_portfolio_risk = max_portfolio_risk
        
        self.hedge_positions = {}
        self.portfolio_positions = {}
        self.last_rebalance = None
        self.correlation_matrix = {}
        self.beta_cache = {}
        
    def get_parameter_space(self) -> Dict[str, Any]:
        """Get parameter space for Bayesian optimization"""
        return {
            'correlation_threshold': {'type': 'real', 'min': 0.5, 'max': 0.9},
            'beta_threshold': {'type': 'real', 'min': 0.5, 'max': 1.2},
            'max_hedge_ratio': {'type': 'real', 'min': 0.5, 'max': 1.5},
            'min_hedge_ratio': {'type': 'real', 'min': 0.05, 'max': 0.3},
            'risk_adjustment_factor': {'type': 'real', 'min': 0.2, 'max': 0.8},
            'rebalance_frequency_hours': {'type': 'integer', 'min': 1, 'max': 24},
            'volatility_lookback': {'type': 'integer', 'min': 10, 'max': 50},
            'correlation_lookback': {'type': 'integer', 'min': 30, 'max': 120},
            'max_portfolio_risk': {'type': 'real', 'min': 0.02, 'max': 0.1}
        }
    
    def calculate_portfolio_risk(self, positions: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate portfolio risk metrics"""
        try:
            if not positions:
                return {
                    'total_value': 0,
                    'total_risk': 0,
                    'diversification_score': 0,
                    'concentration_risk': 0,
                    'correlation_risk': 0
                }
            
            # Calculate portfolio metrics
            total_value = sum(pos.get('value', 0) for pos in positions.values())
            position_weights = {}
            
            for symbol, pos in positions.items():
                weight = pos.get('value', 0) / total_value if total_value > 0 else 0
                position_weights[symbol] = weight
            
            # Calculate concentration risk (Herfindahl index)
            concentration_risk = sum(weight ** 2 for weight in position_weights.values())
            
            # Calculate diversification score
            diversification_score = 1 - concentration_risk
            
            # Calculate total risk (simplified)
            total_risk = sum(pos.get('risk', 0) * weight for symbol, pos in positions.items() 
                           for weight in [position_weights.get(symbol, 0)])
            
            return {
                'total_value': total_value,
                'total_risk': total_risk,
                'diversification_score': diversification_score,
                'concentration_risk': concentration_risk,
                'position_weights': position_weights
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            return {
                'total_value': 0,
                'total_risk': 0,
                'diversification_score': 0,
                'concentration_risk': 0
            }
    
    def calculate_correlations(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Calculate correlations between portfolio positions and hedge instruments"""
        try:
            correlations = {}
            
            for portfolio_symbol in self.portfolio_positions.keys():
                if portfolio_symbol not in market_data:
                    continue
                
                portfolio_returns = market_data[portfolio_symbol]['close'].pct_change().dropna()
                
                for hedge_symbol in self.hedge_instruments:
                    if hedge_symbol not in market_data:
                        continue
                    
                    hedge_returns = market_data[hedge_symbol]['close'].pct_change().dropna()
                    
                    # Align time series
                    common_index = portfolio_returns.index.intersection(hedge_returns.index)
                    if len(common_index) < self.correlation_lookback:
                        continue
                    
                    portfolio_aligned = portfolio_returns.loc[common_index]
                    hedge_aligned = hedge_returns.loc[common_index]
                    
                    # Calculate correlation
                    correlation = portfolio_aligned.corr(hedge_aligned)
                    
                    if not pd.isna(correlation):
                        correlations[f"{portfolio_symbol}_{hedge_symbol}"] = correlation
            
            return correlations
            
        except Exception as e:
            logger.error(f"Error calculating correlations: {e}")
            return {}
    
    def calculate_beta(self, portfolio_symbol: str, hedge_symbol: str, market_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate beta between portfolio position and hedge instrument"""
        try:
            if portfolio_symbol not in market_data or hedge_symbol not in market_data:
                return 0.0
            
            portfolio_returns = market_data[portfolio_symbol]['close'].pct_change().dropna()
            hedge_returns = market_data[hedge_symbol]['close'].pct_change().dropna()
            
            # Align time series
            common_index = portfolio_returns.index.intersection(hedge_returns.index)
            if len(common_index) < self.correlation_lookback:
                return 0.0
            
            portfolio_aligned = portfolio_returns.loc[common_index]
            hedge_aligned = hedge_returns.loc[common_index]
            
            # Calculate beta
            covariance = np.cov(portfolio_aligned, hedge_aligned)[0, 1]
            hedge_variance = np.var(hedge_aligned)
            
            beta = covariance / hedge_variance if hedge_variance > 0 else 0.0
            
            return beta
            
        except Exception as e:
            logger.error(f"Error calculating beta: {e}")
            return 0.0
    
    def calculate_hedge_ratio(self, portfolio_symbol: str, hedge_symbol: str, 
                            market_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate optimal hedge ratio"""
        try:
            # Get position value
            position_value = self.portfolio_positions.get(portfolio_symbol, {}).get('value', 0)
            if position_value <= 0:
                return 0.0
            
            # Calculate beta
            beta = self.calculate_beta(portfolio_symbol, hedge_symbol, market_data)
            
            # Calculate volatility ratio
            portfolio_vol = self.calculate_volatility(portfolio_symbol, market_data)
            hedge_vol = self.calculate_volatility(hedge_symbol, market_data)
            
            vol_ratio = portfolio_vol / hedge_vol if hedge_vol > 0 else 1.0
            
            # Calculate base hedge ratio
            base_ratio = abs(beta) * vol_ratio
            
            # Apply risk adjustment
            risk_adjustment = self.calculate_risk_adjustment(portfolio_symbol, market_data)
            adjusted_ratio = base_ratio * risk_adjustment
            
            # Apply bounds
            final_ratio = max(self.min_hedge_ratio, 
                            min(self.max_hedge_ratio, adjusted_ratio))
            
            return final_ratio
            
        except Exception as e:
            logger.error(f"Error calculating hedge ratio: {e}")
            return 0.0
    
    def calculate_volatility(self, symbol: str, market_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate volatility for a symbol"""
        try:
            if symbol not in market_data:
                return 0.0
            
            returns = market_data[symbol]['close'].pct_change().dropna()
            if len(returns) < self.volatility_lookback:
                return 0.0
            
            # Use rolling volatility
            volatility = returns.tail(self.volatility_lookback).std()
            return volatility
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.0
    
    def calculate_risk_adjustment(self, symbol: str, market_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate risk adjustment factor"""
        try:
            # Get current market conditions
            volatility = self.calculate_volatility(symbol, market_data)
            
            # Adjust based on volatility
            if volatility > 0.05:  # High volatility
                adjustment = 1.0 + self.risk_adjustment_factor
            elif volatility < 0.01:  # Low volatility
                adjustment = 1.0 - self.risk_adjustment_factor
            else:
                adjustment = 1.0
            
            return max(0.5, min(1.5, adjustment))
            
        except Exception as e:
            logger.error(f"Error calculating risk adjustment: {e}")
            return 1.0
    
    def identify_hedging_opportunities(self, market_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Identify hedging opportunities"""
        try:
            opportunities = []
            
            # Calculate correlations
            correlations = self.calculate_correlations(market_data)
            
            for portfolio_symbol in self.portfolio_positions.keys():
                position_info = self.portfolio_positions[portfolio_symbol]
                position_value = position_info.get('value', 0)
                
                if position_value <= 0:
                    continue
                
                # Find best hedge instrument
                best_hedge = None
                best_correlation = 0
                best_beta = 0
                
                for hedge_symbol in self.hedge_instruments:
                    correlation_key = f"{portfolio_symbol}_{hedge_symbol}"
                    correlation = correlations.get(correlation_key, 0)
                    
                    if abs(correlation) > abs(best_correlation):
                        best_correlation = correlation
                        best_hedge = hedge_symbol
                        best_beta = self.calculate_beta(portfolio_symbol, hedge_symbol, market_data)
                
                # Check if hedging is needed
                if (abs(best_correlation) > self.correlation_threshold and 
                    abs(best_beta) > self.beta_threshold):
                    
                    hedge_ratio = self.calculate_hedge_ratio(portfolio_symbol, best_hedge, market_data)
                    
                    if hedge_ratio > self.min_hedge_ratio:
                        opportunities.append({
                            'portfolio_symbol': portfolio_symbol,
                            'hedge_symbol': best_hedge,
                            'correlation': best_correlation,
                            'beta': best_beta,
                            'hedge_ratio': hedge_ratio,
                            'position_value': position_value,
                            'hedge_value': position_value * hedge_ratio,
                            'risk_score': abs(best_correlation) * abs(best_beta)
                        })
            
            # Sort by risk score
            opportunities.sort(key=lambda x: x['risk_score'], reverse=True)
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error identifying hedging opportunities: {e}")
            return []
    
    def should_rebalance(self) -> bool:
        """Check if rebalancing is needed"""
        try:
            if self.last_rebalance is None:
                return True
            
            time_since_rebalance = datetime.now() - self.last_rebalance
            return time_since_rebalance.total_seconds() / 3600 >= self.rebalance_frequency_hours
            
        except Exception as e:
            logger.error(f"Error checking rebalance: {e}")
            return True
    
    def calculate_portfolio_beta(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate portfolio beta against market"""
        try:
            if not self.portfolio_positions:
                return 0.0
            
            portfolio_beta = 0.0
            total_value = sum(pos.get('value', 0) for pos in self.portfolio_positions.values())
            
            for symbol, position in self.portfolio_positions.items():
                position_value = position.get('value', 0)
                weight = position_value / total_value if total_value > 0 else 0
                
                # Calculate beta against SPY (market proxy)
                beta = self.calculate_beta(symbol, 'SPY', market_data)
                portfolio_beta += beta * weight
            
            return portfolio_beta
            
        except Exception as e:
            logger.error(f"Error calculating portfolio beta: {e}")
            return 0.0
    
    def generate_signals(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate hedging signals"""
        try:
            signals = []
            current_time = datetime.now()
            
            # Check if rebalancing is needed
            if not self.should_rebalance():
                return signals
            
            # Simulate market data for all symbols
            market_data = {
                symbol: data for symbol in list(self.portfolio_positions.keys()) + self.hedge_instruments
            }
            
            # Identify hedging opportunities
            opportunities = self.identify_hedging_opportunities(market_data)
            
            # Generate hedge signals
            for opp in opportunities:
                # Check if we need to establish or adjust hedge
                current_hedge = self.hedge_positions.get(opp['hedge_symbol'])
                
                if current_hedge is None:
                    # Establish new hedge
                    signals.append({
                        'signal': 'sell' if opp['correlation'] > 0 else 'buy',
                        'price': data['close'].iloc[-1],
                        'timestamp': current_time,
                        'confidence': min(1.0, opp['risk_score']),
                        'reason': f"Hedge {opp['portfolio_symbol']} with {opp['hedge_symbol']}",
                        'size': opp['hedge_value'],
                        'hedge_ratio': opp['hedge_ratio'],
                        'correlation': opp['correlation'],
                        'beta': opp['beta'],
                        'setup_type': 'hedge_establish'
                    })
                
                else:
                    # Check if hedge needs adjustment
                    current_ratio = current_hedge.get('ratio', 0)
                    target_ratio = opp['hedge_ratio']
                    
                    if abs(current_ratio - target_ratio) > 0.1:  # 10% threshold
                        adjustment_size = opp['position_value'] * (target_ratio - current_ratio)
                        
                        if adjustment_size > 0:
                            signals.append({
                                'signal': 'sell' if opp['correlation'] > 0 else 'buy',
                                'price': data['close'].iloc[-1],
                                'timestamp': current_time,
                                'confidence': 0.8,
                                'reason': f"Adjust hedge {opp['hedge_symbol']}",
                                'size': abs(adjustment_size),
                                'hedge_ratio': target_ratio,
                                'setup_type': 'hedge_adjust'
                            })
            
            # Check for hedge removal
            for hedge_symbol, hedge_info in self.hedge_positions.items():
                # Check if hedge is still needed
                portfolio_beta = self.calculate_portfolio_beta(market_data)
                
                if abs(portfolio_beta) < self.beta_threshold * 0.5:  # Reduced threshold
                    signals.append({
                        'signal': 'close_hedge',
                        'price': data['close'].iloc[-1],
                        'timestamp': current_time,
                        'confidence': 0.9,
                        'reason': f"Remove hedge {hedge_symbol} - low portfolio beta",
                        'hedge_symbol': hedge_symbol,
                        'setup_type': 'hedge_remove'
                    })
            
            # Update last rebalance time
            if signals:
                self.last_rebalance = current_time
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating hedging signals: {e}")
            return []
    
    def update_position(self, signal: Dict[str, Any], account_balance: float):
        """Update hedge positions based on signal"""
        try:
            if signal['setup_type'] == 'hedge_establish':
                # Establish new hedge position
                hedge_symbol = signal.get('hedge_symbol', 'UNKNOWN')
                self.hedge_positions[hedge_symbol] = {
                    'symbol': hedge_symbol,
                    'size': signal['size'],
                    'ratio': signal['hedge_ratio'],
                    'entry_price': signal['price'],
                    'entry_time': signal['timestamp'],
                    'correlation': signal['correlation'],
                    'beta': signal['beta']
                }
            
            elif signal['setup_type'] == 'hedge_adjust':
                # Adjust existing hedge
                hedge_symbol = signal.get('hedge_symbol', 'UNKNOWN')
                if hedge_symbol in self.hedge_positions:
                    self.hedge_positions[hedge_symbol]['ratio'] = signal['hedge_ratio']
                    self.hedge_positions[hedge_symbol]['size'] += signal['size']
            
            elif signal['setup_type'] == 'hedge_remove':
                # Remove hedge position
                hedge_symbol = signal.get('hedge_symbol')
                if hedge_symbol in self.hedge_positions:
                    del self.hedge_positions[hedge_symbol]
                    
        except Exception as e:
            logger.error(f"Error updating hedge position: {e}")
    
    def get_latest_data(self) -> pd.DataFrame:
        """Get latest market data"""
        # This would be implemented to fetch real-time data
        return pd.DataFrame()
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information"""
        return {
            'name': 'Hedging Strategy',
            'description': 'Portfolio risk management with dynamic hedging',
            'parameters': {
                'correlation_threshold': self.correlation_threshold,
                'beta_threshold': self.beta_threshold,
                'max_hedge_ratio': self.max_hedge_ratio,
                'min_hedge_ratio': self.min_hedge_ratio,
                'rebalance_frequency_hours': self.rebalance_frequency_hours,
                'max_portfolio_risk': self.max_portfolio_risk
            },
            'hedge_positions': self.hedge_positions,
            'portfolio_positions': self.portfolio_positions,
            'hedge_instruments': self.hedge_instruments,
            'last_rebalance': self.last_rebalance
        }