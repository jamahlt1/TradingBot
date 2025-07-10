import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import ta
from .base_strategy import BaseStrategy

class TrendFollowingStrategy(BaseStrategy):
    """
    Advanced Trend Following Strategy
    - Multiple timeframe analysis
    - Dynamic trend detection
    - Risk-adjusted position sizing
    - Stop-loss and take-profit management
    """
    
    def __init__(self, 
                 short_window: int = 20,
                 long_window: int = 50,
                 rsi_period: int = 14,
                 rsi_overbought: float = 70,
                 rsi_oversold: float = 30,
                 atr_period: int = 14,
                 risk_per_trade: float = 0.02,
                 max_position_size: float = 0.1,
                 stop_loss_atr: float = 2.0,
                 take_profit_atr: float = 4.0,
                 timeframe: str = '1d',
                 trend_strength_threshold: float = 0.6):
        
        super().__init__()
        self.short_window = short_window
        self.long_window = long_window
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        self.stop_loss_atr = stop_loss_atr
        self.take_profit_atr = take_profit_atr
        self.timeframe = timeframe
        self.trend_strength_threshold = trend_strength_threshold
        
        self.position = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        
    def get_parameter_space(self) -> Dict[str, Any]:
        """Get parameter space for Bayesian optimization"""
        return {
            'short_window': {'type': 'integer', 'min': 10, 'max': 50},
            'long_window': {'type': 'integer', 'min': 30, 'max': 100},
            'rsi_period': {'type': 'integer', 'min': 10, 'max': 30},
            'rsi_overbought': {'type': 'real', 'min': 60, 'max': 80},
            'rsi_oversold': {'type': 'real', 'min': 20, 'max': 40},
            'atr_period': {'type': 'integer', 'min': 10, 'max': 30},
            'risk_per_trade': {'type': 'real', 'min': 0.01, 'max': 0.05},
            'stop_loss_atr': {'type': 'real', 'min': 1.0, 'max': 5.0},
            'take_profit_atr': {'type': 'real', 'min': 2.0, 'max': 8.0},
            'trend_strength_threshold': {'type': 'real', 'min': 0.4, 'max': 0.8}
        }
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        df = data.copy()
        
        # Moving averages
        df['sma_short'] = ta.trend.sma_indicator(df['close'], window=self.short_window)
        df['sma_long'] = ta.trend.sma_indicator(df['close'], window=self.long_window)
        df['ema_short'] = ta.trend.ema_indicator(df['close'], window=self.short_window)
        df['ema_long'] = ta.trend.ema_indicator(df['close'], window=self.long_window)
        
        # Trend indicators
        df['macd'] = ta.trend.macd(df['close'])
        df['macd_signal'] = ta.trend.macd_signal(df['close'])
        df['macd_histogram'] = ta.trend.macd_diff(df['close'])
        
        # Momentum indicators
        df['rsi'] = ta.momentum.rsi(df['close'], window=self.rsi_period)
        df['stoch'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
        df['stoch_signal'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'])
        
        # Volatility indicators
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=self.atr_period)
        df['bb_upper'] = ta.volatility.bollinger_hband(df['close'])
        df['bb_lower'] = ta.volatility.bollinger_lband(df['close'])
        df['bb_middle'] = ta.volatility.bollinger_mavg(df['close'])
        
        # Volume indicators
        df['volume_sma'] = ta.volume.volume_sma(df['close'], df['volume'])
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        
        # Trend strength
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
        df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'])
        
        return df
    
    def detect_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect trend direction and strength"""
        df = data.tail(50)  # Use last 50 periods for trend analysis
        
        # Multiple timeframe trend analysis
        trends = {}
        
        # Short-term trend (5-20 periods)
        short_trend = self._calculate_trend_strength(df.tail(20))
        trends['short_term'] = short_trend
        
        # Medium-term trend (20-50 periods)
        medium_trend = self._calculate_trend_strength(df.tail(50))
        trends['medium_term'] = medium_trend
        
        # Long-term trend (50+ periods)
        long_trend = self._calculate_trend_strength(df)
        trends['long_term'] = long_trend
        
        # Overall trend consensus
        bullish_count = sum(1 for t in trends.values() if t['direction'] == 'bullish')
        bearish_count = sum(1 for t in trends.values() if t['direction'] == 'bearish')
        
        if bullish_count > bearish_count:
            overall_direction = 'bullish'
            strength = bullish_count / len(trends)
        elif bearish_count > bullish_count:
            overall_direction = 'bearish'
            strength = bearish_count / len(trends)
        else:
            overall_direction = 'neutral'
            strength = 0.5
        
        return {
            'overall_direction': overall_direction,
            'overall_strength': strength,
            'timeframe_trends': trends,
            'consensus': 'bullish' if bullish_count > bearish_count else 'bearish'
        }
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trend strength for a specific timeframe"""
        if len(data) < 10:
            return {'direction': 'neutral', 'strength': 0.5}
        
        # Price action analysis
        price_change = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]
        
        # Moving average analysis
        ma_trend = 1 if data['close'].iloc[-1] > data['sma_short'].iloc[-1] > data['sma_long'].iloc[-1] else -1
        
        # MACD analysis
        macd_trend = 1 if data['macd'].iloc[-1] > data['macd_signal'].iloc[-1] else -1
        
        # RSI analysis
        rsi = data['rsi'].iloc[-1]
        rsi_trend = 1 if 40 < rsi < 80 else -1
        
        # ADX for trend strength
        adx = data['adx'].iloc[-1] if 'adx' in data.columns else 25
        trend_strength = min(adx / 50, 1.0)  # Normalize ADX
        
        # Consensus calculation
        signals = [price_change, ma_trend, macd_trend, rsi_trend]
        bullish_signals = sum(1 for s in signals if s > 0)
        bearish_signals = sum(1 for s in signals if s < 0)
        
        if bullish_signals > bearish_signals:
            direction = 'bullish'
            strength = bullish_signals / len(signals) * trend_strength
        elif bearish_signals > bullish_signals:
            direction = 'bearish'
            strength = bearish_signals / len(signals) * trend_strength
        else:
            direction = 'neutral'
            strength = 0.5
        
        return {
            'direction': direction,
            'strength': strength,
            'price_change': price_change,
            'ma_trend': ma_trend,
            'macd_trend': macd_trend,
            'rsi_trend': rsi_trend,
            'adx': adx
        }
    
    def calculate_position_size(self, data: pd.DataFrame, account_balance: float) -> float:
        """Calculate position size based on risk management"""
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
    
    def generate_signals(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate trading signals based on trend analysis"""
        if len(data) < self.long_window:
            return []
        
        df = self.calculate_technical_indicators(data)
        trend_analysis = self.detect_trend(df)
        
        signals = []
        current_price = df['close'].iloc[-1]
        current_time = datetime.now()
        
        # Entry signals
        if self.position is None:  # No current position
            if trend_analysis['overall_direction'] == 'bullish' and trend_analysis['overall_strength'] > self.trend_strength_threshold:
                # Bullish trend entry
                rsi = df['rsi'].iloc[-1]
                if rsi < self.rsi_overbought:  # Not overbought
                    signals.append({
                        'signal': 'buy',
                        'price': current_price,
                        'timestamp': current_time,
                        'confidence': trend_analysis['overall_strength'],
                        'reason': f"Bullish trend detected (strength: {trend_analysis['overall_strength']:.2f})",
                        'indicators': {
                            'rsi': rsi,
                            'macd': df['macd'].iloc[-1],
                            'adx': df['adx'].iloc[-1] if 'adx' in df.columns else None,
                            'atr': df['atr'].iloc[-1] if 'atr' in df.columns else None
                        }
                    })
            
            elif trend_analysis['overall_direction'] == 'bearish' and trend_analysis['overall_strength'] > self.trend_strength_threshold:
                # Bearish trend entry (short)
                rsi = df['rsi'].iloc[-1]
                if rsi > self.rsi_oversold:  # Not oversold
                    signals.append({
                        'signal': 'sell',
                        'price': current_price,
                        'timestamp': current_time,
                        'confidence': trend_analysis['overall_strength'],
                        'reason': f"Bearish trend detected (strength: {trend_analysis['overall_strength']:.2f})",
                        'indicators': {
                            'rsi': rsi,
                            'macd': df['macd'].iloc[-1],
                            'adx': df['adx'].iloc[-1] if 'adx' in df.columns else None,
                            'atr': df['atr'].iloc[-1] if 'atr' in df.columns else None
                        }
                    })
        
        # Exit signals for existing positions
        elif self.position is not None:
            # Check stop loss and take profit
            if self.position['type'] == 'long':
                if current_price <= self.stop_loss:
                    signals.append({
                        'signal': 'sell',
                        'price': current_price,
                        'timestamp': current_time,
                        'confidence': 1.0,
                        'reason': "Stop loss triggered",
                        'position_id': self.position['id']
                    })
                elif current_price >= self.take_profit:
                    signals.append({
                        'signal': 'sell',
                        'price': current_price,
                        'timestamp': current_time,
                        'confidence': 0.9,
                        'reason': "Take profit reached",
                        'position_id': self.position['id']
                    })
                elif trend_analysis['overall_direction'] == 'bearish':
                    # Trend reversal exit
                    signals.append({
                        'signal': 'sell',
                        'price': current_price,
                        'timestamp': current_time,
                        'confidence': trend_analysis['overall_strength'],
                        'reason': "Trend reversal exit",
                        'position_id': self.position['id']
                    })
            
            elif self.position['type'] == 'short':
                if current_price >= self.stop_loss:
                    signals.append({
                        'signal': 'buy',
                        'price': current_price,
                        'timestamp': current_time,
                        'confidence': 1.0,
                        'reason': "Stop loss triggered (short)",
                        'position_id': self.position['id']
                    })
                elif current_price <= self.take_profit:
                    signals.append({
                        'signal': 'buy',
                        'price': current_price,
                        'timestamp': current_time,
                        'confidence': 0.9,
                        'reason': "Take profit reached (short)",
                        'position_id': self.position['id']
                    })
                elif trend_analysis['overall_direction'] == 'bullish':
                    # Trend reversal exit
                    signals.append({
                        'signal': 'buy',
                        'price': current_price,
                        'timestamp': current_time,
                        'confidence': trend_analysis['overall_strength'],
                        'reason': "Trend reversal exit (short)",
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
                'id': f"pos_{datetime.now().timestamp()}",
                'type': 'long',
                'entry_price': signal['price'],
                'size': position_size,
                'entry_time': signal['timestamp']
            }
            
            # Set stop loss and take profit
            atr = self.get_latest_data()['atr'].iloc[-1] if 'atr' in self.get_latest_data().columns else signal['price'] * 0.02
            self.stop_loss = signal['price'] - (atr * self.stop_loss_atr)
            self.take_profit = signal['price'] + (atr * self.take_profit_atr)
            
        elif signal['signal'] == 'sell' and self.position is not None:
            # Close position
            self.position = None
            self.stop_loss = None
            self.take_profit = None
    
    def get_latest_data(self) -> pd.DataFrame:
        """Get latest market data"""
        # This would be implemented to fetch real-time data
        # For now, return empty DataFrame
        return pd.DataFrame()
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information"""
        return {
            'name': 'Trend Following Strategy',
            'description': 'Advanced trend following with multiple timeframe analysis',
            'parameters': {
                'short_window': self.short_window,
                'long_window': self.long_window,
                'rsi_period': self.rsi_period,
                'atr_period': self.atr_period,
                'risk_per_trade': self.risk_per_trade,
                'stop_loss_atr': self.stop_loss_atr,
                'take_profit_atr': self.take_profit_atr,
                'timeframe': self.timeframe
            },
            'current_position': self.position,
            'risk_metrics': {
                'stop_loss': self.stop_loss,
                'take_profit': self.take_profit
            }
        }