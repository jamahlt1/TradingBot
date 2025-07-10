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

class TrendStrength(Enum):
    """Trend Strength Levels"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

@dataclass
class TrendSignal:
    """Trend Signal Result"""
    signal: str  # 'buy', 'sell', 'hold'
    strength: TrendStrength
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    ml_prediction: MLPrediction
    technical_analysis: Dict[str, Any]
    openrouter_analysis: Dict[str, Any]

class EnhancedTrendFollowingStrategy(StrategyBase):
    """
    Enhanced Trend Following Strategy with Advanced ML and OpenRouter
    - Multiple trend detection methods
    - ML-powered signal confirmation
    - OpenRouter AI analysis
    - Dynamic position sizing
    - Advanced risk management
    """
    
    def __init__(self,
                 symbol: str,
                 timeframe: str = '1h',
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
        
        # Strategy parameters
        self.trend_periods = [20, 50, 200]
        self.momentum_periods = [14, 21]
        self.volatility_periods = [20, 50]
        
        # ML parameters
        self.ml_confidence_threshold = 0.7
        self.ml_prediction_weight = 0.4
        self.technical_weight = 0.4
        self.ai_analysis_weight = 0.2
        
        # Performance tracking
        self.signal_history = []
        self.ml_predictions = []
        self.ai_analyses = []
        
    async def initialize(self):
        """Initialize the strategy"""
        try:
            logger.info(f"Initializing Enhanced Trend Following Strategy for {self.symbol}")
            
            # Get historical data for ML training
            historical_data = await self.get_historical_data(days=365)
            
            if historical_data is not None and len(historical_data) > 100:
                # Prepare target variable (future returns)
                historical_data['target'] = historical_data['close'].shift(-1) / historical_data['close'] - 1
                historical_data['target'] = (historical_data['target'] > 0).astype(int)
                
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
    
    async def analyze_trend(self, data: pd.DataFrame) -> TrendSignal:
        """Analyze trend with ML and AI"""
        try:
            if data is None or len(data) < 50:
                return self.create_hold_signal()
            
            # Technical analysis
            technical_analysis = self.perform_technical_analysis(data)
            
            # ML prediction
            ml_prediction = await self.get_ml_prediction(data)
            
            # OpenRouter AI analysis
            ai_analysis = await self.get_ai_analysis(data, technical_analysis, ml_prediction)
            
            # Combine signals
            final_signal = self.combine_signals(technical_analysis, ml_prediction, ai_analysis)
            
            # Create trend signal
            trend_signal = self.create_trend_signal(final_signal, technical_analysis, ml_prediction, ai_analysis)
            
            # Store for tracking
            self.signal_history.append(trend_signal)
            self.ml_predictions.append(ml_prediction)
            self.ai_analyses.append(ai_analysis)
            
            return trend_signal
            
        except Exception as e:
            logger.error(f"Error analyzing trend: {e}")
            return self.create_hold_signal()
    
    def perform_technical_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive technical analysis"""
        try:
            analysis = {}
            
            # Moving averages
            for period in self.trend_periods:
                sma = data['close'].rolling(window=period).mean()
                ema = data['close'].ewm(span=period).mean()
                
                analysis[f'sma_{period}'] = sma.iloc[-1]
                analysis[f'ema_{period}'] = ema.iloc[-1]
                analysis[f'price_vs_sma_{period}'] = data['close'].iloc[-1] / sma.iloc[-1]
                analysis[f'price_vs_ema_{period}'] = data['close'].iloc[-1] / ema.iloc[-1]
            
            # Trend strength indicators
            analysis['trend_strength'] = self.calculate_trend_strength(data)
            analysis['momentum'] = self.calculate_momentum(data)
            analysis['volatility'] = self.calculate_volatility(data)
            
            # Support and resistance
            analysis['support_levels'] = self.find_support_levels(data)
            analysis['resistance_levels'] = self.find_resistance_levels(data)
            
            # Breakout detection
            analysis['breakout_signal'] = self.detect_breakout(data)
            
            # Volume analysis
            if 'volume' in data.columns:
                analysis['volume_trend'] = self.analyze_volume_trend(data)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in technical analysis: {e}")
            return {}
    
    def calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength using multiple indicators"""
        try:
            # ADX for trend strength
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Calculate +DM and -DM
            plus_dm = high.diff()
            minus_dm = low.diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            minus_dm = abs(minus_dm)
            
            # Calculate TR
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate smoothed values
            period = 14
            atr = tr.rolling(window=period).mean()
            plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
            
            # Calculate ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=period).mean()
            
            return adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0
            
        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            return 0
    
    def calculate_momentum(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate momentum indicators"""
        try:
            momentum = {}
            
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            momentum['rsi'] = rsi.iloc[-1]
            
            # MACD
            ema12 = data['close'].ewm(span=12).mean()
            ema26 = data['close'].ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            momentum['macd'] = macd.iloc[-1]
            momentum['macd_signal'] = signal.iloc[-1]
            momentum['macd_histogram'] = macd.iloc[-1] - signal.iloc[-1]
            
            # Stochastic
            low_min = data['low'].rolling(window=14).min()
            high_max = data['high'].rolling(window=14).max()
            k = 100 * (data['close'] - low_min) / (high_max - low_min)
            d = k.rolling(window=3).mean()
            momentum['stoch_k'] = k.iloc[-1]
            momentum['stoch_d'] = d.iloc[-1]
            
            return momentum
            
        except Exception as e:
            logger.error(f"Error calculating momentum: {e}")
            return {}
    
    def calculate_volatility(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate volatility indicators"""
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
            
            return volatility
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return {}
    
    def find_support_levels(self, data: pd.DataFrame) -> List[float]:
        """Find support levels"""
        try:
            levels = []
            low_prices = data['low'].values
            
            # Find local minima
            for i in range(2, len(low_prices) - 2):
                if (low_prices[i] < low_prices[i-1] and 
                    low_prices[i] < low_prices[i-2] and
                    low_prices[i] < low_prices[i+1] and
                    low_prices[i] < low_prices[i+2]):
                    levels.append(low_prices[i])
            
            # Return recent levels
            return sorted(levels)[-3:] if levels else []
            
        except Exception as e:
            logger.error(f"Error finding support levels: {e}")
            return []
    
    def find_resistance_levels(self, data: pd.DataFrame) -> List[float]:
        """Find resistance levels"""
        try:
            levels = []
            high_prices = data['high'].values
            
            # Find local maxima
            for i in range(2, len(high_prices) - 2):
                if (high_prices[i] > high_prices[i-1] and 
                    high_prices[i] > high_prices[i-2] and
                    high_prices[i] > high_prices[i+1] and
                    high_prices[i] > high_prices[i+2]):
                    levels.append(high_prices[i])
            
            # Return recent levels
            return sorted(levels)[-3:] if levels else []
            
        except Exception as e:
            logger.error(f"Error finding resistance levels: {e}")
            return []
    
    def detect_breakout(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect breakout patterns"""
        try:
            breakout = {}
            
            # Price vs moving averages
            sma20 = data['close'].rolling(window=20).mean()
            sma50 = data['close'].rolling(window=50).mean()
            
            current_price = data['close'].iloc[-1]
            
            # Breakout above/below moving averages
            breakout['above_sma20'] = current_price > sma20.iloc[-1]
            breakout['above_sma50'] = current_price > sma50.iloc[-1]
            
            # Volume breakout
            if 'volume' in data.columns:
                avg_volume = data['volume'].rolling(window=20).mean()
                current_volume = data['volume'].iloc[-1]
                breakout['volume_breakout'] = current_volume > avg_volume.iloc[-1] * 1.5
            
            # Price breakout
            recent_high = data['high'].rolling(window=20).max()
            recent_low = data['low'].rolling(window=20).min()
            
            breakout['price_breakout_high'] = current_price > recent_high.iloc[-2]
            breakout['price_breakout_low'] = current_price < recent_low.iloc[-2]
            
            return breakout
            
        except Exception as e:
            logger.error(f"Error detecting breakout: {e}")
            return {}
    
    def analyze_volume_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume trends"""
        try:
            volume_analysis = {}
            
            # Volume moving average
            volume_sma = data['volume'].rolling(window=20).mean()
            current_volume = data['volume'].iloc[-1]
            volume_analysis['volume_ratio'] = current_volume / volume_sma.iloc[-1]
            
            # Volume trend
            volume_trend = data['volume'].rolling(window=5).mean()
            volume_analysis['volume_trend'] = volume_trend.iloc[-1] > volume_trend.iloc[-5]
            
            # Price-volume relationship
            price_change = data['close'].pct_change()
            volume_change = data['volume'].pct_change()
            volume_analysis['price_volume_correlation'] = price_change.corr(volume_change)
            
            return volume_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing volume trend: {e}")
            return {}
    
    async def get_ml_prediction(self, data: pd.DataFrame) -> MLPrediction:
        """Get ML prediction"""
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
        """Get AI analysis from OpenRouter"""
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
            
            # Create prompt for trend analysis
            prompt = f"""
            Analyze the trend following strategy for {self.symbol}:
            
            Current Price: ${context['current_price']:.2f}
            ML Prediction: {context['ml_prediction']:.3f} (confidence: {context['ml_confidence']:.3f})
            
            Technical Analysis:
            - Trend Strength: {technical_analysis.get('trend_strength', 0):.2f}
            - RSI: {technical_analysis.get('momentum', {}).get('rsi', 50):.2f}
            - MACD: {technical_analysis.get('momentum', {}).get('macd', 0):.4f}
            - Volatility: {technical_analysis.get('volatility', {}).get('historical_vol', 0):.4f}
            
            Provide:
            1. Trend direction assessment
            2. Entry/exit recommendations
            3. Risk level assessment
            4. Position sizing advice
            5. Key levels to watch
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
    
    def combine_signals(self, technical_analysis: Dict, ml_prediction: MLPrediction, ai_analysis: Dict) -> Dict[str, Any]:
        """Combine all signals into final decision"""
        try:
            combined_signal = {
                'signal': 'hold',
                'confidence': 0.0,
                'strength': TrendStrength.WEAK,
                'reasoning': []
            }
            
            # Technical signal
            technical_score = self.calculate_technical_score(technical_analysis)
            combined_signal['technical_score'] = technical_score
            
            # ML signal
            ml_score = ml_prediction.prediction
            combined_signal['ml_score'] = ml_score
            
            # AI analysis signal
            ai_score = self.extract_ai_score(ai_analysis)
            combined_signal['ai_score'] = ai_score
            
            # Weighted combination
            final_score = (
                technical_score * self.technical_weight +
                ml_score * self.ml_prediction_weight +
                ai_score * self.ai_analysis_weight
            )
            
            # Determine signal
            if final_score > 0.6:
                combined_signal['signal'] = 'buy'
                combined_signal['confidence'] = final_score
                combined_signal['strength'] = self.get_trend_strength(final_score)
            elif final_score < 0.4:
                combined_signal['signal'] = 'sell'
                combined_signal['confidence'] = 1 - final_score
                combined_signal['strength'] = self.get_trend_strength(1 - final_score)
            else:
                combined_signal['signal'] = 'hold'
                combined_signal['confidence'] = 0.5
            
            return combined_signal
            
        except Exception as e:
            logger.error(f"Error combining signals: {e}")
            return {'signal': 'hold', 'confidence': 0.0, 'strength': TrendStrength.WEAK}
    
    def calculate_technical_score(self, technical_analysis: Dict) -> float:
        """Calculate technical analysis score"""
        try:
            score = 0.5  # Neutral starting point
            
            # Trend strength
            trend_strength = technical_analysis.get('trend_strength', 0)
            if trend_strength > 25:
                score += 0.2
            elif trend_strength < 15:
                score -= 0.2
            
            # Momentum
            momentum = technical_analysis.get('momentum', {})
            rsi = momentum.get('rsi', 50)
            if rsi > 70:
                score -= 0.1  # Overbought
            elif rsi < 30:
                score += 0.1  # Oversold
            elif 40 < rsi < 60:
                score += 0.1  # Neutral
            
            # MACD
            macd = momentum.get('macd', 0)
            macd_signal = momentum.get('macd_signal', 0)
            if macd > macd_signal:
                score += 0.1
            else:
                score -= 0.1
            
            # Breakout signals
            breakout = technical_analysis.get('breakout_signal', {})
            if breakout.get('above_sma20', False):
                score += 0.1
            if breakout.get('above_sma50', False):
                score += 0.1
            
            return max(0, min(1, score))
            
        except Exception as e:
            logger.error(f"Error calculating technical score: {e}")
            return 0.5
    
    def extract_ai_score(self, ai_analysis: Dict) -> float:
        """Extract score from AI analysis"""
        try:
            if not ai_analysis or 'ai_analysis' not in ai_analysis:
                return 0.5
            
            analysis_text = ai_analysis['ai_analysis'].get('analysis', '').lower()
            
            # Simple sentiment analysis
            bullish_words = ['bullish', 'buy', 'long', 'uptrend', 'positive']
            bearish_words = ['bearish', 'sell', 'short', 'downtrend', 'negative']
            
            bullish_count = sum(1 for word in bullish_words if word in analysis_text)
            bearish_count = sum(1 for word in bearish_words if word in analysis_text)
            
            if bullish_count > bearish_count:
                return 0.7
            elif bearish_count > bullish_count:
                return 0.3
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error extracting AI score: {e}")
            return 0.5
    
    def get_trend_strength(self, confidence: float) -> TrendStrength:
        """Get trend strength based on confidence"""
        if confidence > 0.8:
            return TrendStrength.VERY_STRONG
        elif confidence > 0.6:
            return TrendStrength.STRONG
        elif confidence > 0.4:
            return TrendStrength.MODERATE
        else:
            return TrendStrength.WEAK
    
    def create_trend_signal(self, combined_signal: Dict, technical_analysis: Dict, ml_prediction: MLPrediction, ai_analysis: Dict) -> TrendSignal:
        """Create trend signal with all information"""
        try:
            current_price = technical_analysis.get('current_price', 0)
            
            # Calculate stop loss and take profit
            atr = technical_analysis.get('volatility', {}).get('atr', current_price * 0.02)
            
            if combined_signal['signal'] == 'buy':
                stop_loss = current_price - (atr * 2)
                take_profit = current_price + (atr * 3)
            elif combined_signal['signal'] == 'sell':
                stop_loss = current_price + (atr * 2)
                take_profit = current_price - (atr * 3)
            else:
                stop_loss = take_profit = current_price
            
            return TrendSignal(
                signal=combined_signal['signal'],
                strength=combined_signal['strength'],
                confidence=combined_signal['confidence'],
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                ml_prediction=ml_prediction,
                technical_analysis=technical_analysis,
                openrouter_analysis=ai_analysis
            )
            
        except Exception as e:
            logger.error(f"Error creating trend signal: {e}")
            return self.create_hold_signal()
    
    def create_hold_signal(self) -> TrendSignal:
        """Create hold signal"""
        return TrendSignal(
            signal='hold',
            strength=TrendStrength.WEAK,
            confidence=0.0,
            entry_price=0.0,
            stop_loss=0.0,
            take_profit=0.0,
            ml_prediction=MLPrediction(
                prediction=0.5,
                confidence=0.0,
                model_type="none",
                features_used=[],
                timestamp=datetime.now(),
                metadata={}
            ),
            technical_analysis={},
            openrouter_analysis={}
        )
    
    async def execute_strategy(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Execute the enhanced trend following strategy"""
        try:
            # Analyze trend
            trend_signal = await self.analyze_trend(data)
            
            # Risk management check
            if self.risk_manager:
                risk_check = await self.risk_manager.check_risk(trend_signal)
                if not risk_check['approved']:
                    logger.warning(f"Risk check failed: {risk_check['reason']}")
                    return {'action': 'hold', 'reason': 'risk_check_failed'}
            
            # Position management
            if self.position_manager:
                position_check = await self.position_manager.check_position(trend_signal)
                if position_check['action'] == 'close':
                    return {'action': 'close', 'reason': 'position_management'}
            
            # Execute based on signal
            if trend_signal.signal == 'buy' and trend_signal.confidence > self.ml_confidence_threshold:
                return {
                    'action': 'buy',
                    'signal': trend_signal,
                    'reason': 'trend_following_buy',
                    'confidence': trend_signal.confidence
                }
            elif trend_signal.signal == 'sell' and trend_signal.confidence > self.ml_confidence_threshold:
                return {
                    'action': 'sell',
                    'signal': trend_signal,
                    'reason': 'trend_following_sell',
                    'confidence': trend_signal.confidence
                }
            else:
                return {
                    'action': 'hold',
                    'signal': trend_signal,
                    'reason': 'insufficient_confidence'
                }
                
        except Exception as e:
            logger.error(f"Error executing strategy: {e}")
            return {'action': 'hold', 'reason': 'error', 'error': str(e)}
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get strategy summary"""
        try:
            return {
                'strategy_name': 'Enhanced Trend Following',
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'ml_models_trained': len(self.ml_engine.models) if self.ml_engine else 0,
                'signals_generated': len(self.signal_history),
                'ml_predictions': len(self.ml_predictions),
                'ai_analyses': len(self.ai_analyses),
                'parameters': {
                    'trend_periods': self.trend_periods,
                    'momentum_periods': self.momentum_periods,
                    'volatility_periods': self.volatility_periods,
                    'ml_confidence_threshold': self.ml_confidence_threshold,
                    'ml_prediction_weight': self.ml_prediction_weight,
                    'technical_weight': self.technical_weight,
                    'ai_analysis_weight': self.ai_analysis_weight
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting strategy summary: {e}")
            return {}