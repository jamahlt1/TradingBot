import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import ta
from .base_strategy import BaseStrategy

class SwingTradingStrategy(BaseStrategy):
    """
    Advanced Swing Trading Strategy
    - Multi-timeframe analysis
    - Support/resistance levels
    - Swing high/low detection
    - Risk-adjusted position sizing
    - Trend confirmation
    """
    
    def __init__(self,
                 swing_period: int = 20,
                 rsi_period: int = 14,
                 rsi_overbought: float = 70,
                 rsi_oversold: float = 30,
                 atr_period: int = 14,
                 risk_per_trade: float = 0.02,
                 max_position_size: float = 0.1,
                 stop_loss_atr: float = 2.0,
                 take_profit_atr: float = 4.0,
                 trend_confirmation_periods: int = 3,
                 min_swing_strength: float = 0.6):
        
        super().__init__()
        self.swing_period = swing_period
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        self.stop_loss_atr = stop_loss_atr
        self.take_profit_atr = take_profit_atr
        self.trend_confirmation_periods = trend_confirmation_periods
        self.min_swing_strength = min_swing_strength
        
        self.position = None
        self.swing_points = []
        self.support_resistance_levels = []
        
    def get_parameter_space(self) -> Dict[str, Any]:
        """Get parameter space for Bayesian optimization"""
        return {
            'swing_period': {'type': 'integer', 'min': 10, 'max': 50},
            'rsi_period': {'type': 'integer', 'min': 10, 'max': 30},
            'rsi_overbought': {'type': 'real', 'min': 60, 'max': 80},
            'rsi_oversold': {'type': 'real', 'min': 20, 'max': 40},
            'atr_period': {'type': 'integer', 'min': 10, 'max': 30},
            'risk_per_trade': {'type': 'real', 'min': 0.01, 'max': 0.05},
            'stop_loss_atr': {'type': 'real', 'min': 1.0, 'max': 5.0},
            'take_profit_atr': {'type': 'real', 'min': 2.0, 'max': 8.0},
            'trend_confirmation_periods': {'type': 'integer', 'min': 2, 'max': 10},
            'min_swing_strength': {'type': 'real', 'min': 0.4, 'max': 0.8}
        }
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators for swing trading"""
        df = data.copy()
        
        # Basic indicators
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
        
        # RSI
        df['rsi'] = ta.momentum.rsi(df['close'], window=self.rsi_period)
        
        # ATR for volatility
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=self.atr_period)
        
        # Bollinger Bands
        df['bb_upper'] = ta.volatility.bollinger_hband(df['close'])
        df['bb_lower'] = ta.volatility.bollinger_lband(df['close'])
        df['bb_middle'] = ta.volatility.bollinger_mavg(df['close'])
        
        # MACD
        df['macd'] = ta.trend.macd(df['close'])
        df['macd_signal'] = ta.trend.macd_signal(df['close'])
        df['macd_histogram'] = ta.trend.macd_diff(df['close'])
        
        # Stochastic
        df['stoch'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
        df['stoch_signal'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'])
        
        # Williams %R
        df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
        
        # CCI
        df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'])
        
        return df
    
    def detect_swing_points(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect swing highs and lows"""
        swing_points = []
        
        for i in range(self.swing_period, len(data) - self.swing_period):
            high = data['high'].iloc[i]
            low = data['low'].iloc[i]
            
            # Check for swing high
            if all(data['high'].iloc[i-self.swing_period:i] < high) and \
               all(data['high'].iloc[i+1:i+self.swing_period+1] < high):
                swing_points.append({
                    'type': 'high',
                    'index': i,
                    'price': high,
                    'timestamp': data.index[i] if hasattr(data.index[i], 'timestamp') else i,
                    'strength': self._calculate_swing_strength(data, i, 'high')
                })
            
            # Check for swing low
            if all(data['low'].iloc[i-self.swing_period:i] > low) and \
               all(data['low'].iloc[i+1:i+self.swing_period+1] > low):
                swing_points.append({
                    'type': 'low',
                    'index': i,
                    'price': low,
                    'timestamp': data.index[i] if hasattr(data.index[i], 'timestamp') else i,
                    'strength': self._calculate_swing_strength(data, i, 'low')
                })
        
        return swing_points
    
    def _calculate_swing_strength(self, data: pd.DataFrame, index: int, swing_type: str) -> float:
        """Calculate the strength of a swing point"""
        try:
            if swing_type == 'high':
                swing_price = data['high'].iloc[index]
                left_prices = data['high'].iloc[index-self.swing_period:index]
                right_prices = data['high'].iloc[index+1:index+self.swing_period+1]
            else:
                swing_price = data['low'].iloc[index]
                left_prices = data['low'].iloc[index-self.swing_period:index]
                right_prices = data['low'].iloc[index+1:index+self.swing_period+1]
            
            # Calculate how much the swing point stands out
            left_avg = left_prices.mean()
            right_avg = right_prices.mean()
            
            if swing_type == 'high':
                strength = (swing_price - max(left_avg, right_avg)) / swing_price
            else:
                strength = (min(left_avg, right_avg) - swing_price) / swing_price
            
            return max(0, min(1, strength * 10))  # Normalize to 0-1
            
        except Exception as e:
            return 0.5
    
    def find_support_resistance_levels(self, swing_points: List[Dict]) -> Dict[str, List[float]]:
        """Find support and resistance levels from swing points"""
        highs = [point['price'] for point in swing_points if point['type'] == 'high']
        lows = [point['price'] for point in swing_points if point['type'] == 'low']
        
        # Cluster similar levels
        resistance_levels = self._cluster_levels(highs, tolerance=0.02)
        support_levels = self._cluster_levels(lows, tolerance=0.02)
        
        return {
            'resistance': resistance_levels,
            'support': support_levels
        }
    
    def _cluster_levels(self, levels: List[float], tolerance: float) -> List[float]:
        """Cluster similar price levels"""
        if not levels:
            return []
        
        levels = sorted(levels)
        clustered = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if abs(level - current_cluster[0]) / current_cluster[0] <= tolerance:
                current_cluster.append(level)
            else:
                # Average the cluster and add to results
                clustered.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [level]
        
        # Add the last cluster
        if current_cluster:
            clustered.append(sum(current_cluster) / len(current_cluster))
        
        return clustered
    
    def detect_swing_setup(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect swing trading setup"""
        df = self.calculate_technical_indicators(data)
        swing_points = self.detect_swing_points(df)
        levels = self.find_support_resistance_levels(swing_points)
        
        current_price = df['close'].iloc[-1]
        current_rsi = df['rsi'].iloc[-1]
        current_atr = df['atr'].iloc[-1]
        
        # Find nearest support and resistance
        nearest_support = min(levels['support'], key=lambda x: abs(x - current_price)) if levels['support'] else None
        nearest_resistance = min(levels['resistance'], key=lambda x: abs(x - current_price)) if levels['resistance'] else None
        
        # Calculate setup score
        setup_score = 0
        setup_type = None
        setup_reason = []
        
        # Check for oversold conditions (potential buy)
        if current_rsi < self.rsi_oversold:
            if nearest_support and current_price > nearest_support * 0.98:
                setup_score += 0.3
                setup_reason.append('Oversold near support')
        
        # Check for overbought conditions (potential sell)
        if current_rsi > self.rsi_overbought:
            if nearest_resistance and current_price < nearest_resistance * 1.02:
                setup_score += 0.3
                setup_reason.append('Overbought near resistance')
        
        # Check trend confirmation
        if len(df) >= self.trend_confirmation_periods:
            recent_trend = self._calculate_recent_trend(df)
            if recent_trend['direction'] == 'up' and setup_score > 0:
                setup_score += 0.2
                setup_reason.append('Uptrend confirmation')
            elif recent_trend['direction'] == 'down' and setup_score > 0:
                setup_score += 0.2
                setup_reason.append('Downtrend confirmation')
        
        # Check MACD confirmation
        if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1]:
            setup_score += 0.1
            setup_reason.append('MACD bullish')
        elif df['macd'].iloc[-1] < df['macd_signal'].iloc[-1]:
            setup_score += 0.1
            setup_reason.append('MACD bearish')
        
        # Determine setup type
        if setup_score >= self.min_swing_strength:
            if current_rsi < self.rsi_oversold:
                setup_type = 'buy'
            elif current_rsi > self.rsi_overbought:
                setup_type = 'sell'
        
        return {
            'setup_type': setup_type,
            'setup_score': setup_score,
            'setup_reason': setup_reason,
            'current_price': current_price,
            'current_rsi': current_rsi,
            'current_atr': current_atr,
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'swing_points': swing_points[-5:],  # Last 5 swing points
            'levels': levels
        }
    
    def _calculate_recent_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate recent trend direction"""
        recent_data = data.tail(self.trend_confirmation_periods)
        
        if len(recent_data) < 2:
            return {'direction': 'neutral', 'strength': 0}
        
        # Calculate trend using linear regression
        x = np.arange(len(recent_data))
        y = recent_data['close'].values
        
        slope = np.polyfit(x, y, 1)[0]
        trend_strength = abs(slope) / recent_data['close'].mean()
        
        if slope > 0:
            direction = 'up'
        elif slope < 0:
            direction = 'down'
        else:
            direction = 'neutral'
        
        return {
            'direction': direction,
            'strength': trend_strength
        }
    
    def calculate_position_size(self, data: pd.DataFrame, account_balance: float) -> float:
        """Calculate position size based on risk management"""
        try:
            current_price = data['close'].iloc[-1]
            atr = data['atr'].iloc[-1] if 'atr' in data.columns else current_price * 0.02
            
            # Risk-based position sizing
            risk_amount = account_balance * self.risk_per_trade
            stop_loss_distance = atr * self.stop_loss_atr
            
            if stop_loss_distance == 0:
                return 0
            
            position_size = risk_amount / stop_loss_distance
            
            # Apply maximum position size limit
            max_position_value = account_balance * self.max_position_size
            max_position_size = max_position_value / current_price
            
            return min(position_size, max_position_size)
            
        except Exception as e:
            return 0
    
    def generate_signals(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate swing trading signals"""
        if len(data) < self.swing_period * 2:
            return []
        
        df = self.calculate_technical_indicators(data)
        setup = self.detect_swing_setup(df)
        
        signals = []
        current_time = datetime.now()
        
        if setup['setup_type'] and setup['setup_score'] >= self.min_swing_strength:
            # Entry signal
            if self.position is None:
                signals.append({
                    'signal': setup['setup_type'],
                    'price': setup['current_price'],
                    'timestamp': current_time,
                    'confidence': setup['setup_score'],
                    'reason': ', '.join(setup['setup_reason']),
                    'indicators': {
                        'rsi': setup['current_rsi'],
                        'atr': setup['current_atr'],
                        'nearest_support': setup['nearest_support'],
                        'nearest_resistance': setup['nearest_resistance']
                    },
                    'setup_type': 'swing_entry'
                })
        
        # Exit signals for existing positions
        elif self.position is not None:
            # Check stop loss and take profit
            if self.position['type'] == 'long':
                if setup['current_price'] <= self.position.get('stop_loss', 0):
                    signals.append({
                        'signal': 'sell',
                        'price': setup['current_price'],
                        'timestamp': current_time,
                        'confidence': 1.0,
                        'reason': 'Stop loss triggered',
                        'position_id': self.position['id']
                    })
                elif setup['current_price'] >= self.position.get('take_profit', float('inf')):
                    signals.append({
                        'signal': 'sell',
                        'price': setup['current_price'],
                        'timestamp': current_time,
                        'confidence': 0.9,
                        'reason': 'Take profit reached',
                        'position_id': self.position['id']
                    })
                elif setup['setup_type'] == 'sell' and setup['setup_score'] >= 0.7:
                    # Swing exit signal
                    signals.append({
                        'signal': 'sell',
                        'price': setup['current_price'],
                        'timestamp': current_time,
                        'confidence': setup['setup_score'],
                        'reason': 'Swing exit signal',
                        'position_id': self.position['id']
                    })
            
            elif self.position['type'] == 'short':
                if setup['current_price'] >= self.position.get('stop_loss', float('inf')):
                    signals.append({
                        'signal': 'buy',
                        'price': setup['current_price'],
                        'timestamp': current_time,
                        'confidence': 1.0,
                        'reason': 'Stop loss triggered (short)',
                        'position_id': self.position['id']
                    })
                elif setup['current_price'] <= self.position.get('take_profit', 0):
                    signals.append({
                        'signal': 'buy',
                        'price': setup['current_price'],
                        'timestamp': current_time,
                        'confidence': 0.9,
                        'reason': 'Take profit reached (short)',
                        'position_id': self.position['id']
                    })
                elif setup['setup_type'] == 'buy' and setup['setup_score'] >= 0.7:
                    # Swing exit signal
                    signals.append({
                        'signal': 'buy',
                        'price': setup['current_price'],
                        'timestamp': current_time,
                        'confidence': setup['setup_score'],
                        'reason': 'Swing exit signal (short)',
                        'position_id': self.position['id']
                    })
        
        return signals
    
    def update_position(self, signal: Dict[str, Any], account_balance: float):
        """Update position based on signal"""
        if signal['signal'] == 'buy' and self.position is None:
            # Open long position
            position_size = self.calculate_position_size(
                self.get_latest_data(), account_balance
            )
            
            self.position = {
                'id': f"swing_{datetime.now().timestamp()}",
                'type': 'long',
                'entry_price': signal['price'],
                'size': position_size,
                'entry_time': signal['timestamp'],
                'stop_loss': signal['price'] * (1 - self.stop_loss_atr * 0.01),
                'take_profit': signal['price'] * (1 + self.take_profit_atr * 0.01)
            }
            
        elif signal['signal'] == 'sell' and self.position is not None:
            # Close position
            self.position = None
    
    def get_latest_data(self) -> pd.DataFrame:
        """Get latest market data"""
        # This would be implemented to fetch real-time data
        return pd.DataFrame()
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information"""
        return {
            'name': 'Swing Trading Strategy',
            'description': 'Advanced swing trading with support/resistance analysis',
            'parameters': {
                'swing_period': self.swing_period,
                'rsi_period': self.rsi_period,
                'atr_period': self.atr_period,
                'risk_per_trade': self.risk_per_trade,
                'stop_loss_atr': self.stop_loss_atr,
                'take_profit_atr': self.take_profit_atr,
                'trend_confirmation_periods': self.trend_confirmation_periods,
                'min_swing_strength': self.min_swing_strength
            },
            'current_position': self.position,
            'swing_points_count': len(self.swing_points)
        }