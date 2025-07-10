import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import logging
from dataclasses import dataclass
from enum import Enum

from app.ml.enhanced_ml_engine import EnhancedMLEngine, MLModelType, MLPrediction
from app.core.strategy_base import StrategyBase
from app.core.risk_manager import RiskManager
from app.core.position_manager import PositionManager

logger = logging.getLogger(__name__)

class SwingSignal(Enum):
    """Swing Trading Signals"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"

@dataclass
class SwingAnalysis:
    """Swing Trading Analysis Result"""
    signal: SwingSignal
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    swing_high: float
    swing_low: float
    support_level: float
    resistance_level: float
    ml_prediction: MLPrediction
    technical_analysis: Dict[str, Any]
    openrouter_analysis: Dict[str, Any]
    swing_pattern: str
    risk_reward_ratio: float

class EnhancedSwingTradingStrategy(StrategyBase):
    """
    Enhanced Swing Trading Strategy with Advanced ML and OpenRouter
    - Swing high/low detection
    - ML-powered entry/exit signals
    - OpenRouter AI analysis
    - Pattern recognition
    - Risk-reward optimization
    """
    
    def __init__(self,
                 symbol: str,
                 timeframe: str = '4h',
                 ml_engine: EnhancedMLEngine = None,
                 risk_manager: RiskManager = None,
                 position_manager: PositionManager = None,
                 openrouter_api_key: str = None):
        
        super().__init__(symbol, timeframe)
        
        # Initialize ML engine
        self.ml_engine = ml_engine or EnhancedMLEngine(
            openrouter_api_key=openrouter_api_key,
            model_types=[
                MLModelType.XGBOOST,
                MLModelType.LIGHTGBM,
                MLModelType.LSTM,
                MLModelType.ENSEMBLE
            ],
            feature_engineering=True,
            ensemble_method='weighted'
        )
        
        # Risk and position management
        self.risk_manager = risk_manager
        self.position_manager = position_manager
        
        # Swing trading parameters
        self.swing_period = 20
        self.confirmation_period = 3
        self.min_swing_size = 0.02  # 2% minimum swing
        self.risk_reward_min = 2.0
        
        # ML parameters
        self.ml_confidence_threshold = 0.65
        self.ml_prediction_weight = 0.35
        self.technical_weight = 0.45
        self.ai_analysis_weight = 0.20
        
        # Pattern recognition
        self.patterns = {
            'double_top': self.detect_double_top,
            'double_bottom': self.detect_double_bottom,
            'head_shoulders': self.detect_head_shoulders,
            'inverse_head_shoulders': self.detect_inverse_head_shoulders,
            'triangle': self.detect_triangle,
            'flag': self.detect_flag,
            'wedge': self.detect_wedge
        }
        
        # Performance tracking
        self.swing_history = []
        self.ml_predictions = []
        self.ai_analyses = []
        self.pattern_detections = []
        
    async def initialize(self):
        """Initialize the strategy"""
        try:
            logger.info(f"Initializing Enhanced Swing Trading Strategy for {self.symbol}")
            
            # Get historical data for ML training
            historical_data = await self.get_historical_data(days=365)
            
            if historical_data is not None and len(historical_data) > 100:
                # Prepare target variable (swing direction)
                historical_data['target'] = self.prepare_swing_target(historical_data)
                
                # Initialize and train ML models
                await self.ml_engine.initialize_models(historical_data, 'target')
                await self.ml_engine.train_models(historical_data, 'target')
                
                logger.info("ML models trained successfully")
            else:
                logger.warning("Insufficient historical data for ML training")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing strategy: {e}")
            return False
    
    def prepare_swing_target(self, data: pd.DataFrame) -> pd.Series:
        """Prepare target variable for swing trading"""
        try:
            # Calculate swing highs and lows
            swing_highs = self.find_swing_highs(data)
            swing_lows = self.find_swing_lows(data)
            
            # Create target based on swing direction
            target = pd.Series(index=data.index, dtype=int)
            
            for i in range(len(data)):
                if i in swing_highs:
                    target.iloc[i] = 0  # Sell signal at swing high
                elif i in swing_lows:
                    target.iloc[i] = 1  # Buy signal at swing low
                else:
                    target.iloc[i] = 2  # Hold
            
            return target
            
        except Exception as e:
            logger.error(f"Error preparing swing target: {e}")
            return pd.Series(2, index=data.index)
    
    async def analyze_swing(self, data: pd.DataFrame) -> SwingAnalysis:
        """Analyze swing trading opportunities"""
        try:
            if data is None or len(data) < 50:
                return self.create_hold_analysis()
            
            # Find swing points
            swing_highs = self.find_swing_highs(data)
            swing_lows = self.find_swing_lows(data)
            
            # Technical analysis
            technical_analysis = self.perform_swing_technical_analysis(data, swing_highs, swing_lows)
            
            # ML prediction
            ml_prediction = await self.get_ml_prediction(data)
            
            # OpenRouter AI analysis
            ai_analysis = await self.get_ai_analysis(data, technical_analysis, ml_prediction)
            
            # Pattern detection
            patterns = self.detect_patterns(data)
            
            # Combine signals
            final_signal = self.combine_swing_signals(technical_analysis, ml_prediction, ai_analysis, patterns)
            
            # Create swing analysis
            swing_analysis = self.create_swing_analysis(final_signal, technical_analysis, ml_prediction, ai_analysis, patterns)
            
            # Store for tracking
            self.swing_history.append(swing_analysis)
            self.ml_predictions.append(ml_prediction)
            self.ai_analyses.append(ai_analysis)
            self.pattern_detections.append(patterns)
            
            return swing_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing swing: {e}")
            return self.create_hold_analysis()
    
    def find_swing_highs(self, data: pd.DataFrame) -> List[int]:
        """Find swing highs"""
        try:
            swing_highs = []
            highs = data['high'].values
            
            for i in range(self.swing_period, len(highs) - self.swing_period):
                # Check if current point is higher than surrounding points
                left_max = max(highs[i-self.swing_period:i])
                right_max = max(highs[i+1:i+self.swing_period+1])
                
                if highs[i] > left_max and highs[i] > right_max:
                    swing_highs.append(i)
            
            return swing_highs
            
        except Exception as e:
            logger.error(f"Error finding swing highs: {e}")
            return []
    
    def find_swing_lows(self, data: pd.DataFrame) -> List[int]:
        """Find swing lows"""
        try:
            swing_lows = []
            lows = data['low'].values
            
            for i in range(self.swing_period, len(lows) - self.swing_period):
                # Check if current point is lower than surrounding points
                left_min = min(lows[i-self.swing_period:i])
                right_min = min(lows[i+1:i+self.swing_period+1])
                
                if lows[i] < left_min and lows[i] < right_min:
                    swing_lows.append(i)
            
            return swing_lows
            
        except Exception as e:
            logger.error(f"Error finding swing lows: {e}")
            return []
    
    def perform_swing_technical_analysis(self, data: pd.DataFrame, swing_highs: List[int], swing_lows: List[int]) -> Dict[str, Any]:
        """Perform comprehensive swing trading technical analysis"""
        try:
            analysis = {}
            
            # Current swing levels
            current_price = data['close'].iloc[-1]
            analysis['current_price'] = current_price
            
            # Find nearest swing points
            if swing_highs:
                nearest_high_idx = min(swing_highs, key=lambda x: abs(x - len(data) + 1))
                analysis['nearest_swing_high'] = data['high'].iloc[nearest_high_idx]
                analysis['distance_to_high'] = (analysis['nearest_swing_high'] - current_price) / current_price
            else:
                analysis['nearest_swing_high'] = current_price * 1.05
                analysis['distance_to_high'] = 0.05
            
            if swing_lows:
                nearest_low_idx = min(swing_lows, key=lambda x: abs(x - len(data) + 1))
                analysis['nearest_swing_low'] = data['low'].iloc[nearest_low_idx]
                analysis['distance_to_low'] = (current_price - analysis['nearest_swing_low']) / current_price
            else:
                analysis['nearest_swing_low'] = current_price * 0.95
                analysis['distance_to_low'] = 0.05
            
            # Support and resistance levels
            analysis['support_levels'] = self.find_support_levels(data)
            analysis['resistance_levels'] = self.find_resistance_levels(data)
            
            # Momentum indicators
            analysis['momentum'] = self.calculate_swing_momentum(data)
            
            # Volatility analysis
            analysis['volatility'] = self.calculate_swing_volatility(data)
            
            # Volume analysis
            if 'volume' in data.columns:
                analysis['volume_analysis'] = self.analyze_swing_volume(data)
            
            # Swing strength
            analysis['swing_strength'] = self.calculate_swing_strength(data, swing_highs, swing_lows)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in swing technical analysis: {e}")
            return {}
    
    def calculate_swing_momentum(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate momentum indicators for swing trading"""
        try:
            momentum = {}
            
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            momentum['rsi'] = rsi.iloc[-1]
            
            # Stochastic
            low_min = data['low'].rolling(window=14).min()
            high_max = data['high'].rolling(window=14).max()
            k = 100 * (data['close'] - low_min) / (high_max - low_min)
            d = k.rolling(window=3).mean()
            momentum['stoch_k'] = k.iloc[-1]
            momentum['stoch_d'] = d.iloc[-1]
            
            # Williams %R
            williams_r = ((data['high'].rolling(window=14).max() - data['close']) / 
                         (data['high'].rolling(window=14).max() - data['low'].rolling(window=14).min())) * -100
            momentum['williams_r'] = williams_r.iloc[-1]
            
            # CCI (Commodity Channel Index)
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            sma_tp = typical_price.rolling(window=20).mean()
            mean_deviation = abs(typical_price - sma_tp).rolling(window=20).mean()
            cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
            momentum['cci'] = cci.iloc[-1]
            
            return momentum
            
        except Exception as e:
            logger.error(f"Error calculating swing momentum: {e}")
            return {}
    
    def calculate_swing_volatility(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate volatility indicators for swing trading"""
        try:
            volatility = {}
            
            # ATR
            high = data['high']
            low = data['low']
            close = data['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()
            volatility['atr'] = atr.iloc[-1]
            
            # Bollinger Bands
            sma = data['close'].rolling(window=20).mean()
            std = data['close'].rolling(window=20).std()
            bb_upper = sma + (std * 2)
            bb_lower = sma - (std * 2)
            bb_width = (bb_upper - bb_lower) / sma
            volatility['bb_width'] = bb_width.iloc[-1]
            volatility['bb_position'] = (data['close'].iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
            
            # Historical volatility
            returns = data['close'].pct_change()
            volatility['historical_vol'] = returns.rolling(window=20).std().iloc[-1] * np.sqrt(252)
            
            # Volatility ratio
            volatility['vol_ratio'] = volatility['historical_vol'] / volatility['historical_vol'] if volatility['historical_vol'] > 0 else 1
            
            return volatility
            
        except Exception as e:
            logger.error(f"Error calculating swing volatility: {e}")
            return {}
    
    def analyze_swing_volume(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume patterns for swing trading"""
        try:
            volume_analysis = {}
            
            # Volume moving average
            volume_sma = data['volume'].rolling(window=20).mean()
            current_volume = data['volume'].iloc[-1]
            volume_analysis['volume_ratio'] = current_volume / volume_sma.iloc[-1]
            
            # Volume trend
            volume_trend = data['volume'].rolling(window=5).mean()
            volume_analysis['volume_trend'] = volume_trend.iloc[-1] > volume_trend.iloc[-5]
            
            # Volume confirmation
            price_change = data['close'].pct_change()
            volume_change = data['volume'].pct_change()
            volume_analysis['volume_confirmation'] = (price_change > 0) & (volume_change > 0)
            
            # Volume divergence
            price_high = data['close'].rolling(window=10).max()
            volume_high = data['volume'].rolling(window=10).max()
            volume_analysis['volume_divergence'] = (data['close'].iloc[-1] >= price_high.iloc[-1] * 0.95) and (data['volume'].iloc[-1] < volume_high.iloc[-1] * 0.8)
            
            return volume_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing swing volume: {e}")
            return {}
    
    def calculate_swing_strength(self, data: pd.DataFrame, swing_highs: List[int], swing_lows: List[int]) -> float:
        """Calculate swing strength"""
        try:
            if not swing_highs and not swing_lows:
                return 0.0
            
            # Calculate average swing size
            swing_sizes = []
            
            for high_idx in swing_highs:
                if high_idx > 0:
                    swing_sizes.append(data['high'].iloc[high_idx] - data['low'].iloc[high_idx-1])
            
            for low_idx in swing_lows:
                if low_idx > 0:
                    swing_sizes.append(data['high'].iloc[low_idx-1] - data['low'].iloc[low_idx])
            
            if swing_sizes:
                avg_swing_size = np.mean(swing_sizes)
                current_price = data['close'].iloc[-1]
                swing_strength = avg_swing_size / current_price
                return min(1.0, swing_strength)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating swing strength: {e}")
            return 0.0
    
    def detect_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect swing trading patterns"""
        try:
            patterns = {}
            
            for pattern_name, pattern_func in self.patterns.items():
                try:
                    pattern_result = pattern_func(data)
                    if pattern_result['detected']:
                        patterns[pattern_name] = pattern_result
                except Exception as e:
                    logger.error(f"Error detecting {pattern_name}: {e}")
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            return {}
    
    def detect_double_top(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect double top pattern"""
        try:
            # Find swing highs
            swing_highs = self.find_swing_highs(data)
            
            if len(swing_highs) < 2:
                return {'detected': False}
            
            # Check last two swing highs
            high1_idx = swing_highs[-2]
            high2_idx = swing_highs[-1]
            
            high1_price = data['high'].iloc[high1_idx]
            high2_price = data['high'].iloc[high2_idx]
            
            # Check if highs are similar (within 2%)
            price_diff = abs(high1_price - high2_price) / high1_price
            
            if price_diff < 0.02 and high2_idx > high1_idx + 5:
                return {
                    'detected': True,
                    'type': 'bearish',
                    'high1': high1_price,
                    'high2': high2_price,
                    'neckline': data['low'].iloc[high1_idx:high2_idx].min()
                }
            
            return {'detected': False}
            
        except Exception as e:
            logger.error(f"Error detecting double top: {e}")
            return {'detected': False}
    
    def detect_double_bottom(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect double bottom pattern"""
        try:
            # Find swing lows
            swing_lows = self.find_swing_lows(data)
            
            if len(swing_lows) < 2:
                return {'detected': False}
            
            # Check last two swing lows
            low1_idx = swing_lows[-2]
            low2_idx = swing_lows[-1]
            
            low1_price = data['low'].iloc[low1_idx]
            low2_price = data['low'].iloc[low2_idx]
            
            # Check if lows are similar (within 2%)
            price_diff = abs(low1_price - low2_price) / low1_price
            
            if price_diff < 0.02 and low2_idx > low1_idx + 5:
                return {
                    'detected': True,
                    'type': 'bullish',
                    'low1': low1_price,
                    'low2': low2_price,
                    'neckline': data['high'].iloc[low1_idx:low2_idx].max()
                }
            
            return {'detected': False}
            
        except Exception as e:
            logger.error(f"Error detecting double bottom: {e}")
            return {'detected': False}
    
    def detect_head_shoulders(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect head and shoulders pattern"""
        try:
            # Find swing highs
            swing_highs = self.find_swing_highs(data)
            
            if len(swing_highs) < 3:
                return {'detected': False}
            
            # Check last three swing highs
            left_shoulder_idx = swing_highs[-3]
            head_idx = swing_highs[-2]
            right_shoulder_idx = swing_highs[-1]
            
            left_shoulder_price = data['high'].iloc[left_shoulder_idx]
            head_price = data['high'].iloc[head_idx]
            right_shoulder_price = data['high'].iloc[right_shoulder_idx]
            
            # Check pattern conditions
            if (head_price > left_shoulder_price and 
                head_price > right_shoulder_price and
                abs(left_shoulder_price - right_shoulder_price) / left_shoulder_price < 0.03):
                
                return {
                    'detected': True,
                    'type': 'bearish',
                    'left_shoulder': left_shoulder_price,
                    'head': head_price,
                    'right_shoulder': right_shoulder_price,
                    'neckline': data['low'].iloc[left_shoulder_idx:right_shoulder_idx].min()
                }
            
            return {'detected': False}
            
        except Exception as e:
            logger.error(f"Error detecting head and shoulders: {e}")
            return {'detected': False}
    
    def detect_inverse_head_shoulders(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect inverse head and shoulders pattern"""
        try:
            # Find swing lows
            swing_lows = self.find_swing_lows(data)
            
            if len(swing_lows) < 3:
                return {'detected': False}
            
            # Check last three swing lows
            left_shoulder_idx = swing_lows[-3]
            head_idx = swing_lows[-2]
            right_shoulder_idx = swing_lows[-1]
            
            left_shoulder_price = data['low'].iloc[left_shoulder_idx]
            head_price = data['low'].iloc[head_idx]
            right_shoulder_price = data['low'].iloc[right_shoulder_idx]
            
            # Check pattern conditions
            if (head_price < left_shoulder_price and 
                head_price < right_shoulder_price and
                abs(left_shoulder_price - right_shoulder_price) / left_shoulder_price < 0.03):
                
                return {
                    'detected': True,
                    'type': 'bullish',
                    'left_shoulder': left_shoulder_price,
                    'head': head_price,
                    'right_shoulder': right_shoulder_price,
                    'neckline': data['high'].iloc[left_shoulder_idx:right_shoulder_idx].max()
                }
            
            return {'detected': False}
            
        except Exception as e:
            logger.error(f"Error detecting inverse head and shoulders: {e}")
            return {'detected': False}
    
    def detect_triangle(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect triangle pattern"""
        try:
            # Find swing highs and lows
            swing_highs = self.find_swing_highs(data)
            swing_lows = self.find_swing_lows(data)
            
            if len(swing_highs) < 3 or len(swing_lows) < 3:
                return {'detected': False}
            
            # Get recent swing points
            recent_highs = [data['high'].iloc[i] for i in swing_highs[-3:]]
            recent_lows = [data['low'].iloc[i] for i in swing_lows[-3:]]
            
            # Check for converging trendlines
            high_slope = (recent_highs[-1] - recent_highs[0]) / len(recent_highs)
            low_slope = (recent_lows[-1] - recent_lows[0]) / len(recent_lows)
            
            if abs(high_slope) < 0.01 and abs(low_slope) < 0.01:
                return {
                    'detected': True,
                    'type': 'neutral',
                    'highs': recent_highs,
                    'lows': recent_lows
                }
            
            return {'detected': False}
            
        except Exception as e:
            logger.error(f"Error detecting triangle: {e}")
            return {'detected': False}
    
    def detect_flag(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect flag pattern"""
        try:
            # Simple flag detection based on price consolidation
            recent_data = data.tail(20)
            price_range = recent_data['high'].max() - recent_data['low'].min()
            avg_price = recent_data['close'].mean()
            
            if price_range / avg_price < 0.05:  # Less than 5% range
                return {
                    'detected': True,
                    'type': 'continuation',
                    'range': price_range,
                    'avg_price': avg_price
                }
            
            return {'detected': False}
            
        except Exception as e:
            logger.error(f"Error detecting flag: {e}")
            return {'detected': False}
    
    def detect_wedge(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect wedge pattern"""
        try:
            # Find swing highs and lows
            swing_highs = self.find_swing_highs(data)
            swing_lows = self.find_swing_lows(data)
            
            if len(swing_highs) < 3 or len(swing_lows) < 3:
                return {'detected': False}
            
            # Get recent swing points
            recent_highs = [data['high'].iloc[i] for i in swing_highs[-3:]]
            recent_lows = [data['low'].iloc[i] for i in swing_lows[-3:]]
            
            # Check for wedge pattern (converging lines)
            high_slope = (recent_highs[-1] - recent_highs[0]) / len(recent_highs)
            low_slope = (recent_lows[-1] - recent_lows[0]) / len(recent_lows)
            
            if high_slope < 0 and low_slope > 0:  # Falling wedge
                return {
                    'detected': True,
                    'type': 'bullish',
                    'highs': recent_highs,
                    'lows': recent_lows
                }
            elif high_slope > 0 and low_slope < 0:  # Rising wedge
                return {
                    'detected': True,
                    'type': 'bearish',
                    'highs': recent_highs,
                    'lows': recent_lows
                }
            
            return {'detected': False}
            
        except Exception as e:
            logger.error(f"Error detecting wedge: {e}")
            return {'detected': False}
    
    async def get_ml_prediction(self, data: pd.DataFrame) -> MLPrediction:
        """Get ML prediction for swing trading"""
        try:
            if self.ml_engine and len(data) > 50:
                return await self.ml_engine.predict(data)
            else:
                return MLPrediction(
                    prediction=0.5,
                    confidence=0.0,
                    model_type="none",
                    features_used=[],
                    timestamp=datetime.now(),
                    metadata={}
                )
                
        except Exception as e:
            logger.error(f"Error getting ML prediction: {e}")
            return MLPrediction(
                prediction=0.5,
                confidence=0.0,
                model_type="error",
                features_used=[],
                timestamp=datetime.now(),
                metadata={'error': str(e)}
            )
    
    async def get_ai_analysis(self, data: pd.DataFrame, technical_analysis: Dict, ml_prediction: MLPrediction) -> Dict[str, Any]:
        """Get AI analysis for swing trading"""
        try:
            if not self.ml_engine.openrouter_client:
                return {}
            
            # Prepare analysis context
            context = {
                'symbol': self.symbol,
                'current_price': data['close'].iloc[-1],
                'technical_analysis': technical_analysis,
                'ml_prediction': ml_prediction.prediction,
                'ml_confidence': ml_prediction.confidence
            }
            
            # Create prompt for swing analysis
            prompt = f"""
            Analyze the swing trading strategy for {self.symbol}:
            
            Current Price: ${context['current_price']:.2f}
            ML Prediction: {context['ml_prediction']:.3f} (confidence: {context['ml_confidence']:.3f})
            
            Technical Analysis:
            - RSI: {technical_analysis.get('momentum', {}).get('rsi', 50):.2f}
            - Swing Strength: {technical_analysis.get('swing_strength', 0):.3f}
            - Distance to Swing High: {technical_analysis.get('distance_to_high', 0):.3f}
            - Distance to Swing Low: {technical_analysis.get('distance_to_low', 0):.3f}
            
            Provide:
            1. Swing trading opportunity assessment
            2. Entry/exit recommendations
            3. Risk level assessment
            4. Position sizing advice
            5. Key swing levels to watch
            """
            
            # Get AI analysis
            analysis = await self.ml_engine.openrouter_client.analyze_market(prompt)
            
            return {
                'ai_analysis': analysis,
                'context': context
            }
            
        except Exception as e:
            logger.error(f"Error getting AI analysis: {e}")
            return {}
    
    def combine_swing_signals(self, technical_analysis: Dict, ml_prediction: MLPrediction, ai_analysis: Dict, patterns: Dict) -> Dict[str, Any]:
        """Combine all signals for swing trading"""
        try:
            combined_signal = {
                'signal': SwingSignal.HOLD,
                'confidence': 0.0,
                'reasoning': []
            }
            
            # Technical signal
            technical_score = self.calculate_swing_technical_score(technical_analysis)
            combined_signal['technical_score'] = technical_score
            
            # ML signal
            ml_score = ml_prediction.prediction
            combined_signal['ml_score'] = ml_score
            
            # AI analysis signal
            ai_score = self.extract_swing_ai_score(ai_analysis)
            combined_signal['ai_score'] = ai_score
            
            # Pattern signal
            pattern_score = self.calculate_pattern_score(patterns)
            combined_signal['pattern_score'] = pattern_score
            
            # Weighted combination
            final_score = (
                technical_score * self.technical_weight +
                ml_score * self.ml_prediction_weight +
                ai_score * self.ai_analysis_weight +
                pattern_score * 0.1  # Pattern weight
            )
            
            # Determine signal
            if final_score > 0.7:
                combined_signal['signal'] = SwingSignal.STRONG_BUY
                combined_signal['confidence'] = final_score
            elif final_score > 0.6:
                combined_signal['signal'] = SwingSignal.BUY
                combined_signal['confidence'] = final_score
            elif final_score < 0.3:
                combined_signal['signal'] = SwingSignal.STRONG_SELL
                combined_signal['confidence'] = 1 - final_score
            elif final_score < 0.4:
                combined_signal['signal'] = SwingSignal.SELL
                combined_signal['confidence'] = 1 - final_score
            else:
                combined_signal['signal'] = SwingSignal.HOLD
                combined_signal['confidence'] = 0.5
            
            return combined_signal
            
        except Exception as e:
            logger.error(f"Error combining swing signals: {e}")
            return {'signal': SwingSignal.HOLD, 'confidence': 0.0}
    
    def calculate_swing_technical_score(self, technical_analysis: Dict) -> float:
        """Calculate technical analysis score for swing trading"""
        try:
            score = 0.5  # Neutral starting point
            
            # Swing strength
            swing_strength = technical_analysis.get('swing_strength', 0)
            if swing_strength > 0.05:
                score += 0.2
            elif swing_strength < 0.02:
                score -= 0.2
            
            # Distance to swing levels
            distance_to_high = technical_analysis.get('distance_to_high', 0)
            distance_to_low = technical_analysis.get('distance_to_low', 0)
            
            if distance_to_low < 0.02:  # Near swing low
                score += 0.3
            elif distance_to_high < 0.02:  # Near swing high
                score -= 0.3
            
            # Momentum
            momentum = technical_analysis.get('momentum', {})
            rsi = momentum.get('rsi', 50)
            if rsi < 30:
                score += 0.2  # Oversold
            elif rsi > 70:
                score -= 0.2  # Overbought
            
            # Volume confirmation
            volume_analysis = technical_analysis.get('volume_analysis', {})
            if volume_analysis.get('volume_confirmation', False):
                score += 0.1
            
            return max(0, min(1, score))
            
        except Exception as e:
            logger.error(f"Error calculating swing technical score: {e}")
            return 0.5
    
    def extract_swing_ai_score(self, ai_analysis: Dict) -> float:
        """Extract score from AI analysis for swing trading"""
        try:
            if not ai_analysis or 'ai_analysis' not in ai_analysis:
                return 0.5
            
            analysis_text = ai_analysis['ai_analysis'].get('analysis', '').lower()
            
            # Swing trading specific sentiment analysis
            bullish_words = ['bullish', 'buy', 'long', 'uptrend', 'positive', 'swing low', 'bounce']
            bearish_words = ['bearish', 'sell', 'short', 'downtrend', 'negative', 'swing high', 'drop']
            
            bullish_count = sum(1 for word in bullish_words if word in analysis_text)
            bearish_count = sum(1 for word in bearish_words if word in analysis_text)
            
            if bullish_count > bearish_count:
                return 0.7
            elif bearish_count > bullish_count:
                return 0.3
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error extracting swing AI score: {e}")
            return 0.5
    
    def calculate_pattern_score(self, patterns: Dict) -> float:
        """Calculate pattern score"""
        try:
            score = 0.5
            
            for pattern_name, pattern_data in patterns.items():
                if pattern_data.get('detected', False):
                    pattern_type = pattern_data.get('type', 'neutral')
                    
                    if pattern_type == 'bullish':
                        score += 0.2
                    elif pattern_type == 'bearish':
                        score -= 0.2
                    elif pattern_type == 'continuation':
                        score += 0.1
            
            return max(0, min(1, score))
            
        except Exception as e:
            logger.error(f"Error calculating pattern score: {e}")
            return 0.5
    
    def create_swing_analysis(self, combined_signal: Dict, technical_analysis: Dict, ml_prediction: MLPrediction, ai_analysis: Dict, patterns: Dict) -> SwingAnalysis:
        """Create swing analysis with all information"""
        try:
            current_price = technical_analysis.get('current_price', 0)
            
            # Calculate stop loss and take profit
            atr = technical_analysis.get('volatility', {}).get('atr', current_price * 0.02)
            
            if combined_signal['signal'] in [SwingSignal.STRONG_BUY, SwingSignal.BUY]:
                stop_loss = current_price - (atr * 2)
                take_profit = current_price + (atr * 4)  # Higher risk-reward for swing trading
            elif combined_signal['signal'] in [SwingSignal.STRONG_SELL, SwingSignal.SELL]:
                stop_loss = current_price + (atr * 2)
                take_profit = current_price - (atr * 4)
            else:
                stop_loss = take_profit = current_price
            
            # Calculate risk-reward ratio
            if stop_loss != take_profit:
                risk = abs(current_price - stop_loss)
                reward = abs(take_profit - current_price)
                risk_reward_ratio = reward / risk if risk > 0 else 0
            else:
                risk_reward_ratio = 0
            
            return SwingAnalysis(
                signal=combined_signal['signal'],
                confidence=combined_signal['confidence'],
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                swing_high=technical_analysis.get('nearest_swing_high', current_price),
                swing_low=technical_analysis.get('nearest_swing_low', current_price),
                support_level=technical_analysis.get('support_levels', [current_price])[0] if technical_analysis.get('support_levels') else current_price,
                resistance_level=technical_analysis.get('resistance_levels', [current_price])[0] if technical_analysis.get('resistance_levels') else current_price,
                ml_prediction=ml_prediction,
                technical_analysis=technical_analysis,
                openrouter_analysis=ai_analysis,
                swing_pattern=self.get_dominant_pattern(patterns),
                risk_reward_ratio=risk_reward_ratio
            )
            
        except Exception as e:
            logger.error(f"Error creating swing analysis: {e}")
            return self.create_hold_analysis()
    
    def get_dominant_pattern(self, patterns: Dict) -> str:
        """Get the most significant pattern"""
        try:
            if not patterns:
                return "none"
            
            # Find the most recent pattern
            for pattern_name in ['double_top', 'double_bottom', 'head_shoulders', 'inverse_head_shoulders', 'triangle', 'flag', 'wedge']:
                if pattern_name in patterns and patterns[pattern_name].get('detected', False):
                    return pattern_name
            
            return "none"
            
        except Exception as e:
            logger.error(f"Error getting dominant pattern: {e}")
            return "none"
    
    def create_hold_analysis(self) -> SwingAnalysis:
        """Create hold analysis"""
        return SwingAnalysis(
            signal=SwingSignal.HOLD,
            confidence=0.0,
            entry_price=0.0,
            stop_loss=0.0,
            take_profit=0.0,
            swing_high=0.0,
            swing_low=0.0,
            support_level=0.0,
            resistance_level=0.0,
            ml_prediction=MLPrediction(
                prediction=0.5,
                confidence=0.0,
                model_type="none",
                features_used=[],
                timestamp=datetime.now(),
                metadata={}
            ),
            technical_analysis={},
            openrouter_analysis={},
            swing_pattern="none",
            risk_reward_ratio=0.0
        )
    
    async def execute_strategy(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Execute the enhanced swing trading strategy"""
        try:
            # Analyze swing
            swing_analysis = await self.analyze_swing(data)
            
            # Risk management check
            if self.risk_manager:
                risk_check = await self.risk_manager.check_risk(swing_analysis)
                if not risk_check['approved']:
                    logger.warning(f"Risk check failed: {risk_check['reason']}")
                    return {'action': 'hold', 'reason': 'risk_check_failed'}
            
            # Position management
            if self.position_manager:
                position_check = await self.position_manager.check_position(swing_analysis)
                if position_check['action'] == 'close':
                    return {'action': 'close', 'reason': 'position_management'}
            
            # Check risk-reward ratio
            if swing_analysis.risk_reward_ratio < self.risk_reward_min:
                return {'action': 'hold', 'reason': 'insufficient_risk_reward'}
            
            # Execute based on signal
            if swing_analysis.signal in [SwingSignal.STRONG_BUY, SwingSignal.BUY] and swing_analysis.confidence > self.ml_confidence_threshold:
                return {
                    'action': 'buy',
                    'signal': swing_analysis,
                    'reason': 'swing_trading_buy',
                    'confidence': swing_analysis.confidence
                }
            elif swing_analysis.signal in [SwingSignal.STRONG_SELL, SwingSignal.SELL] and swing_analysis.confidence > self.ml_confidence_threshold:
                return {
                    'action': 'sell',
                    'signal': swing_analysis,
                    'reason': 'swing_trading_sell',
                    'confidence': swing_analysis.confidence
                }
            else:
                return {
                    'action': 'hold',
                    'signal': swing_analysis,
                    'reason': 'insufficient_confidence'
                }
                
        except Exception as e:
            logger.error(f"Error executing strategy: {e}")
            return {'action': 'hold', 'reason': 'error', 'error': str(e)}
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get strategy summary"""
        try:
            return {
                'strategy_name': 'Enhanced Swing Trading',
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'ml_models_trained': len(self.ml_engine.models) if self.ml_engine else 0,
                'swing_signals_generated': len(self.swing_history),
                'ml_predictions': len(self.ml_predictions),
                'ai_analyses': len(self.ai_analyses),
                'patterns_detected': len(self.pattern_detections),
                'parameters': {
                    'swing_period': self.swing_period,
                    'confirmation_period': self.confirmation_period,
                    'min_swing_size': self.min_swing_size,
                    'risk_reward_min': self.risk_reward_min,
                    'ml_confidence_threshold': self.ml_confidence_threshold,
                    'ml_prediction_weight': self.ml_prediction_weight,
                    'technical_weight': self.technical_weight,
                    'ai_analysis_weight': self.ai_analysis_weight
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting strategy summary: {e}")
            return {}