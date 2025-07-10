import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class ICTStrategy(BaseStrategy):
    """
    Advanced ICT (Inner Circle Trader) Strategy
    - Market structure analysis
    - Order block identification
    - Fair value gaps
    - Liquidity zones
    - Institutional order flow
    """
    
    def __init__(self,
                 order_block_lookback: int = 50,
                 fair_value_gap_threshold: float = 0.002,
                 liquidity_zone_threshold: float = 0.005,
                 market_structure_periods: int = 20,
                 risk_per_trade: float = 0.02,
                 max_position_size: float = 0.1,
                 stop_loss_atr: float = 2.0,
                 take_profit_atr: float = 4.0,
                 order_block_min_size: int = 3,
                 fair_value_gap_min_size: int = 2):
        
        super().__init__()
        self.order_block_lookback = order_block_lookback
        self.fair_value_gap_threshold = fair_value_gap_threshold
        self.liquidity_zone_threshold = liquidity_zone_threshold
        self.market_structure_periods = market_structure_periods
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        self.stop_loss_atr = stop_loss_atr
        self.take_profit_atr = take_profit_atr
        self.order_block_min_size = order_block_min_size
        self.fair_value_gap_min_size = fair_value_gap_min_size
        
        self.position = None
        self.order_blocks = []
        self.fair_value_gaps = []
        self.liquidity_zones = []
        self.market_structure = []
        
    def get_parameter_space(self) -> Dict[str, Any]:
        """Get parameter space for Bayesian optimization"""
        return {
            'order_block_lookback': {'type': 'integer', 'min': 20, 'max': 100},
            'fair_value_gap_threshold': {'type': 'real', 'min': 0.001, 'max': 0.01},
            'liquidity_zone_threshold': {'type': 'real', 'min': 0.002, 'max': 0.02},
            'market_structure_periods': {'type': 'integer', 'min': 10, 'max': 50},
            'risk_per_trade': {'type': 'real', 'min': 0.01, 'max': 0.05},
            'stop_loss_atr': {'type': 'real', 'min': 1.0, 'max': 5.0},
            'take_profit_atr': {'type': 'real', 'min': 2.0, 'max': 8.0},
            'order_block_min_size': {'type': 'integer', 'min': 2, 'max': 10},
            'fair_value_gap_min_size': {'type': 'integer', 'min': 1, 'max': 5}
        }
    
    def identify_market_structure(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Identify market structure (higher highs, lower lows, etc.)"""
        try:
            if len(data) < self.market_structure_periods:
                return {'structure': 'undefined', 'trend': 'neutral'}
            
            highs = data['high'].rolling(window=self.market_structure_periods).max()
            lows = data['low'].rolling(window=self.market_structure_periods).min()
            
            # Identify swing highs and lows
            swing_highs = []
            swing_lows = []
            
            for i in range(self.market_structure_periods, len(data) - self.market_structure_periods):
                if data['high'].iloc[i] == highs.iloc[i]:
                    swing_highs.append({
                        'index': i,
                        'price': data['high'].iloc[i],
                        'timestamp': data.index[i] if hasattr(data.index[i], 'timestamp') else i
                    })
                
                if data['low'].iloc[i] == lows.iloc[i]:
                    swing_lows.append({
                        'index': i,
                        'price': data['low'].iloc[i],
                        'timestamp': data.index[i] if hasattr(data.index[i], 'timestamp') else i
                    })
            
            # Analyze structure
            if len(swing_highs) >= 2 and len(swing_lows) >= 2:
                recent_highs = swing_highs[-2:]
                recent_lows = swing_lows[-2:]
                
                # Check for higher highs and higher lows (uptrend)
                if (recent_highs[-1]['price'] > recent_highs[-2]['price'] and 
                    recent_lows[-1]['price'] > recent_lows[-2]['price']):
                    structure = 'uptrend'
                    trend = 'bullish'
                
                # Check for lower highs and lower lows (downtrend)
                elif (recent_highs[-1]['price'] < recent_highs[-2]['price'] and 
                      recent_lows[-1]['price'] < recent_lows[-2]['price']):
                    structure = 'downtrend'
                    trend = 'bearish'
                
                # Check for consolidation
                elif (abs(recent_highs[-1]['price'] - recent_highs[-2]['price']) < 
                      recent_highs[-2]['price'] * 0.01):
                    structure = 'consolidation'
                    trend = 'neutral'
                
                else:
                    structure = 'mixed'
                    trend = 'neutral'
            else:
                structure = 'undefined'
                trend = 'neutral'
            
            return {
                'structure': structure,
                'trend': trend,
                'swing_highs': swing_highs[-5:],  # Last 5 swing highs
                'swing_lows': swing_lows[-5:],    # Last 5 swing lows
                'current_high': data['high'].iloc[-1],
                'current_low': data['low'].iloc[-1]
            }
            
        except Exception as e:
            logger.error(f"Error identifying market structure: {e}")
            return {'structure': 'undefined', 'trend': 'neutral'}
    
    def find_order_blocks(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find order blocks (institutional order zones)"""
        try:
            order_blocks = []
            
            for i in range(self.order_block_min_size, len(data) - self.order_block_min_size):
                # Look for strong moves followed by consolidation
                move_size = abs(data['close'].iloc[i] - data['open'].iloc[i])
                avg_move = data['close'].pct_change().abs().rolling(20).mean().iloc[i]
                
                # Check if this is a significant move
                if move_size > avg_move * 2:
                    # Look for consolidation after the move
                    consolidation_start = i + 1
                    consolidation_end = min(i + self.order_block_lookback, len(data))
                    
                    consolidation_range = data.iloc[consolidation_start:consolidation_end]
                    
                    if len(consolidation_range) >= self.order_block_min_size:
                        # Calculate consolidation metrics
                        consolidation_high = consolidation_range['high'].max()
                        consolidation_low = consolidation_range['low'].min()
                        consolidation_range_size = consolidation_high - consolidation_low
                        
                        # Check if consolidation is tight
                        if consolidation_range_size < move_size * 0.5:
                            # Determine order block type
                            if data['close'].iloc[i] > data['open'].iloc[i]:
                                # Bullish order block
                                order_block_type = 'bullish'
                                order_block_high = consolidation_high
                                order_block_low = consolidation_low
                            else:
                                # Bearish order block
                                order_block_type = 'bearish'
                                order_block_high = consolidation_high
                                order_block_low = consolidation_low
                            
                            order_blocks.append({
                                'type': order_block_type,
                                'start_index': consolidation_start,
                                'end_index': consolidation_end - 1,
                                'high': order_block_high,
                                'low': order_block_low,
                                'mid': (order_block_high + order_block_low) / 2,
                                'strength': move_size / avg_move,
                                'timestamp': data.index[consolidation_start] if hasattr(data.index[consolidation_start], 'timestamp') else consolidation_start
                            })
            
            return order_blocks
            
        except Exception as e:
            logger.error(f"Error finding order blocks: {e}")
            return []
    
    def find_fair_value_gaps(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find fair value gaps (FVG)"""
        try:
            fair_value_gaps = []
            
            for i in range(1, len(data) - 1):
                current_high = data['high'].iloc[i]
                current_low = data['low'].iloc[i]
                prev_low = data['low'].iloc[i-1]
                next_high = data['high'].iloc[i+1]
                
                # Bullish FVG: gap between current low and previous low
                if current_low > prev_low:
                    gap_size = current_low - prev_low
                    if gap_size > self.fair_value_gap_threshold * prev_low:
                        fair_value_gaps.append({
                            'type': 'bullish',
                            'index': i,
                            'gap_high': current_low,
                            'gap_low': prev_low,
                            'gap_size': gap_size,
                            'strength': gap_size / prev_low,
                            'timestamp': data.index[i] if hasattr(data.index[i], 'timestamp') else i
                        })
                
                # Bearish FVG: gap between current high and next high
                if current_high < next_high:
                    gap_size = next_high - current_high
                    if gap_size > self.fair_value_gap_threshold * current_high:
                        fair_value_gaps.append({
                            'type': 'bearish',
                            'index': i,
                            'gap_high': next_high,
                            'gap_low': current_high,
                            'gap_size': gap_size,
                            'strength': gap_size / current_high,
                            'timestamp': data.index[i] if hasattr(data.index[i], 'timestamp') else i
                        })
            
            return fair_value_gaps
            
        except Exception as e:
            logger.error(f"Error finding fair value gaps: {e}")
            return []
    
    def find_liquidity_zones(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find liquidity zones (areas of high volume/activity)"""
        try:
            liquidity_zones = []
            
            # Calculate volume profile
            volume_profile = data.groupby(pd.cut(data['close'], bins=20))['volume'].sum()
            
            # Find high volume zones
            volume_threshold = volume_profile.quantile(0.8)
            high_volume_zones = volume_profile[volume_profile > volume_threshold]
            
            for zone_range, volume in high_volume_zones.items():
                zone_low = zone_range.left
                zone_high = zone_range.right
                
                # Check if current price is near this zone
                current_price = data['close'].iloc[-1]
                distance_to_zone = min(abs(current_price - zone_low), abs(current_price - zone_high))
                
                if distance_to_zone < current_price * self.liquidity_zone_threshold:
                    liquidity_zones.append({
                        'type': 'volume_zone',
                        'low': zone_low,
                        'high': zone_high,
                        'volume': volume,
                        'strength': volume / volume_profile.max(),
                        'distance': distance_to_zone
                    })
            
            # Also look for recent swing highs/lows as liquidity zones
            market_structure = self.identify_market_structure(data)
            
            for swing_high in market_structure.get('swing_highs', []):
                current_price = data['close'].iloc[-1]
                distance = abs(current_price - swing_high['price'])
                
                if distance < current_price * self.liquidity_zone_threshold:
                    liquidity_zones.append({
                        'type': 'swing_high',
                        'price': swing_high['price'],
                        'strength': 0.8,
                        'distance': distance
                    })
            
            for swing_low in market_structure.get('swing_lows', []):
                current_price = data['close'].iloc[-1]
                distance = abs(current_price - swing_low['price'])
                
                if distance < current_price * self.liquidity_zone_threshold:
                    liquidity_zones.append({
                        'type': 'swing_low',
                        'price': swing_low['price'],
                        'strength': 0.8,
                        'distance': distance
                    })
            
            return liquidity_zones
            
        except Exception as e:
            logger.error(f"Error finding liquidity zones: {e}")
            return []
    
    def calculate_ict_score(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate ICT trading score"""
        try:
            market_structure = self.identify_market_structure(data)
            order_blocks = self.find_order_blocks(data)
            fair_value_gaps = self.find_fair_value_gaps(data)
            liquidity_zones = self.find_liquidity_zones(data)
            
            current_price = data['close'].iloc[-1]
            score = 0.0
            signals = []
            
            # Score based on market structure
            if market_structure['trend'] == 'bullish':
                score += 0.3
                signals.append('Bullish market structure')
            elif market_structure['trend'] == 'bearish':
                score -= 0.3
                signals.append('Bearish market structure')
            
            # Score based on order blocks
            for ob in order_blocks[-3:]:  # Check last 3 order blocks
                if ob['type'] == 'bullish' and current_price > ob['low']:
                    score += 0.2
                    signals.append(f"Bullish order block at {ob['low']:.2f}")
                elif ob['type'] == 'bearish' and current_price < ob['high']:
                    score -= 0.2
                    signals.append(f"Bearish order block at {ob['high']:.2f}")
            
            # Score based on fair value gaps
            for fvg in fair_value_gaps[-5:]:  # Check last 5 FVGs
                if fvg['type'] == 'bullish' and current_price > fvg['gap_low']:
                    score += 0.15
                    signals.append(f"Bullish FVG at {fvg['gap_low']:.2f}")
                elif fvg['type'] == 'bearish' and current_price < fvg['gap_high']:
                    score -= 0.15
                    signals.append(f"Bearish FVG at {fvg['gap_high']:.2f}")
            
            # Score based on liquidity zones
            for zone in liquidity_zones:
                if zone['type'] == 'swing_high' and current_price < zone['price']:
                    score += 0.1  # Potential resistance
                    signals.append(f"Liquidity zone at {zone['price']:.2f}")
                elif zone['type'] == 'swing_low' and current_price > zone['price']:
                    score += 0.1  # Potential support
                    signals.append(f"Liquidity zone at {zone['price']:.2f}")
            
            # Normalize score
            score = max(-1.0, min(1.0, score))
            
            return {
                'score': score,
                'signals': signals,
                'market_structure': market_structure,
                'order_blocks': order_blocks[-3:],
                'fair_value_gaps': fair_value_gaps[-3:],
                'liquidity_zones': liquidity_zones
            }
            
        except Exception as e:
            logger.error(f"Error calculating ICT score: {e}")
            return {
                'score': 0.0,
                'signals': ['Error in ICT analysis'],
                'market_structure': {'structure': 'undefined', 'trend': 'neutral'},
                'order_blocks': [],
                'fair_value_gaps': [],
                'liquidity_zones': []
            }
    
    def calculate_position_size(self, data: pd.DataFrame, account_balance: float) -> float:
        """Calculate position size based on ICT analysis"""
        try:
            current_price = data['close'].iloc[-1]
            atr = data['close'].pct_change().rolling(14).std().iloc[-1] * current_price
            
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
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    def generate_signals(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate ICT trading signals"""
        if len(data) < self.market_structure_periods:
            return []
        
        try:
            ict_analysis = self.calculate_ict_score(data)
            signals = []
            current_time = datetime.now()
            
            # Generate entry signals
            if self.position is None:
                if ict_analysis['score'] > 0.5:  # Strong bullish signal
                    signals.append({
                        'signal': 'buy',
                        'price': data['close'].iloc[-1],
                        'timestamp': current_time,
                        'confidence': min(1.0, ict_analysis['score']),
                        'reason': f"ICT bullish signal (score: {ict_analysis['score']:.2f})",
                        'ict_analysis': ict_analysis,
                        'setup_type': 'ict_entry'
                    })
                
                elif ict_analysis['score'] < -0.5:  # Strong bearish signal
                    signals.append({
                        'signal': 'sell',
                        'price': data['close'].iloc[-1],
                        'timestamp': current_time,
                        'confidence': min(1.0, abs(ict_analysis['score'])),
                        'reason': f"ICT bearish signal (score: {ict_analysis['score']:.2f})",
                        'ict_analysis': ict_analysis,
                        'setup_type': 'ict_entry'
                    })
            
            # Generate exit signals for existing positions
            elif self.position is not None:
                # Check for ICT reversal signals
                if self.position['type'] == 'long' and ict_analysis['score'] < -0.3:
                    signals.append({
                        'signal': 'sell',
                        'price': data['close'].iloc[-1],
                        'timestamp': current_time,
                        'confidence': min(1.0, abs(ict_analysis['score'])),
                        'reason': "ICT reversal signal",
                        'position_id': self.position['id']
                    })
                
                elif self.position['type'] == 'short' and ict_analysis['score'] > 0.3:
                    signals.append({
                        'signal': 'buy',
                        'price': data['close'].iloc[-1],
                        'timestamp': current_time,
                        'confidence': min(1.0, ict_analysis['score']),
                        'reason': "ICT reversal signal",
                        'position_id': self.position['id']
                    })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating ICT signals: {e}")
            return []
    
    def update_position(self, signal: Dict[str, Any], account_balance: float):
        """Update position based on signal"""
        if signal['signal'] == 'buy' and self.position is None:
            # Open long position
            position_size = self.calculate_position_size(
                self.get_latest_data(), account_balance
            )
            
            self.position = {
                'id': f"ict_{datetime.now().timestamp()}",
                'type': 'long',
                'entry_price': signal['price'],
                'size': position_size,
                'entry_time': signal['timestamp'],
                'stop_loss': signal['price'] * (1 - self.stop_loss_atr * 0.01),
                'take_profit': signal['price'] * (1 + self.take_profit_atr * 0.01),
                'ict_score': signal.get('ict_analysis', {}).get('score', 0.0)
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
            'name': 'ICT Trading Strategy',
            'description': 'Institutional trading based on market structure and order flow',
            'parameters': {
                'order_block_lookback': self.order_block_lookback,
                'fair_value_gap_threshold': self.fair_value_gap_threshold,
                'liquidity_zone_threshold': self.liquidity_zone_threshold,
                'market_structure_periods': self.market_structure_periods,
                'risk_per_trade': self.risk_per_trade,
                'stop_loss_atr': self.stop_loss_atr,
                'take_profit_atr': self.take_profit_atr
            },
            'current_position': self.position,
            'order_blocks_count': len(self.order_blocks),
            'fair_value_gaps_count': len(self.fair_value_gaps),
            'liquidity_zones_count': len(self.liquidity_zones)
        }