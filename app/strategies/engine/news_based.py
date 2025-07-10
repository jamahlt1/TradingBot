import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import re
import logging
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class NewsBasedStrategy(BaseStrategy):
    """
    Advanced News-Based Trading Strategy
    - Real-time news sentiment analysis
    - Event-driven trading signals
    - Risk-adjusted position sizing
    - News impact assessment
    - Market reaction prediction
    """
    
    def __init__(self,
                 sentiment_threshold: float = 0.6,
                 impact_threshold: float = 0.5,
                 news_lookback_hours: int = 24,
                 position_hold_hours: int = 4,
                 risk_per_trade: float = 0.02,
                 max_position_size: float = 0.1,
                 stop_loss_percentage: float = 0.05,
                 take_profit_percentage: float = 0.15,
                 min_news_volume: int = 10,
                 sentiment_weight: float = 0.4,
                 impact_weight: float = 0.3,
                 technical_weight: float = 0.3):
        
        super().__init__()
        self.sentiment_threshold = sentiment_threshold
        self.impact_threshold = impact_threshold
        self.news_lookback_hours = news_lookback_hours
        self.position_hold_hours = position_hold_hours
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        self.stop_loss_percentage = stop_loss_percentage
        self.take_profit_percentage = take_profit_percentage
        self.min_news_volume = min_news_volume
        self.sentiment_weight = sentiment_weight
        self.impact_weight = impact_weight
        self.technical_weight = technical_weight
        
        self.position = None
        self.news_cache = {}
        self.sentiment_cache = {}
        
    def get_parameter_space(self) -> Dict[str, Any]:
        """Get parameter space for Bayesian optimization"""
        return {
            'sentiment_threshold': {'type': 'real', 'min': 0.3, 'max': 0.8},
            'impact_threshold': {'type': 'real', 'min': 0.2, 'max': 0.8},
            'news_lookback_hours': {'type': 'integer', 'min': 6, 'max': 48},
            'position_hold_hours': {'type': 'integer', 'min': 1, 'max': 24},
            'risk_per_trade': {'type': 'real', 'min': 0.01, 'max': 0.05},
            'stop_loss_percentage': {'type': 'real', 'min': 0.02, 'max': 0.10},
            'take_profit_percentage': {'type': 'real', 'min': 0.05, 'max': 0.30},
            'min_news_volume': {'type': 'integer', 'min': 5, 'max': 50},
            'sentiment_weight': {'type': 'real', 'min': 0.2, 'max': 0.6},
            'impact_weight': {'type': 'real', 'min': 0.2, 'max': 0.6},
            'technical_weight': {'type': 'real', 'min': 0.2, 'max': 0.6}
        }
    
    def analyze_news_sentiment(self, news_data: List[Dict]) -> Dict[str, Any]:
        """Analyze news sentiment and impact"""
        try:
            if not news_data:
                return {
                    'overall_sentiment': 0.0,
                    'sentiment_score': 0.0,
                    'impact_score': 0.0,
                    'news_volume': 0,
                    'key_events': [],
                    'confidence': 0.0
                }
            
            # Calculate sentiment scores
            sentiment_scores = []
            impact_scores = []
            key_events = []
            
            for news in news_data:
                # Extract sentiment from news data
                sentiment = news.get('sentiment', 0.0)
                impact = news.get('impact', 0.0)
                title = news.get('title', '')
                
                sentiment_scores.append(sentiment)
                impact_scores.append(impact)
                
                # Identify key events
                if impact > self.impact_threshold:
                    key_events.append({
                        'title': title,
                        'sentiment': sentiment,
                        'impact': impact,
                        'timestamp': news.get('timestamp', datetime.now())
                    })
            
            # Calculate weighted scores
            avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
            avg_impact = np.mean(impact_scores) if impact_scores else 0.0
            
            # Calculate confidence based on news volume and consistency
            news_volume = len(news_data)
            sentiment_std = np.std(sentiment_scores) if len(sentiment_scores) > 1 else 0.0
            confidence = min(1.0, news_volume / self.min_news_volume) * (1 - sentiment_std)
            
            return {
                'overall_sentiment': avg_sentiment,
                'sentiment_score': avg_sentiment,
                'impact_score': avg_impact,
                'news_volume': news_volume,
                'key_events': key_events,
                'confidence': confidence,
                'sentiment_std': sentiment_std
            }
            
        except Exception as e:
            logger.error(f"Error analyzing news sentiment: {e}")
            return {
                'overall_sentiment': 0.0,
                'sentiment_score': 0.0,
                'impact_score': 0.0,
                'news_volume': 0,
                'key_events': [],
                'confidence': 0.0
            }
    
    def detect_news_events(self, news_data: List[Dict]) -> List[Dict[str, Any]]:
        """Detect significant news events"""
        events = []
        
        try:
            # Keywords for different event types
            event_keywords = {
                'earnings': ['earnings', 'revenue', 'profit', 'quarterly', 'annual'],
                'merger': ['merger', 'acquisition', 'buyout', 'takeover'],
                'regulatory': ['regulation', 'fda', 'sec', 'government', 'policy'],
                'product': ['launch', 'release', 'announcement', 'product'],
                'partnership': ['partnership', 'collaboration', 'alliance', 'deal'],
                'legal': ['lawsuit', 'legal', 'court', 'settlement'],
                'management': ['ceo', 'executive', 'resignation', 'appointment']
            }
            
            for news in news_data:
                title = news.get('title', '').lower()
                content = news.get('content', '').lower()
                text = f"{title} {content}"
                
                # Detect event type
                detected_events = []
                for event_type, keywords in event_keywords.items():
                    if any(keyword in text for keyword in keywords):
                        detected_events.append(event_type)
                
                if detected_events:
                    events.append({
                        'title': news.get('title', ''),
                        'event_types': detected_events,
                        'sentiment': news.get('sentiment', 0.0),
                        'impact': news.get('impact', 0.0),
                        'timestamp': news.get('timestamp', datetime.now()),
                        'source': news.get('source', 'unknown')
                    })
            
            return events
            
        except Exception as e:
            logger.error(f"Error detecting news events: {e}")
            return []
    
    def calculate_news_score(self, sentiment_analysis: Dict, events: List[Dict]) -> float:
        """Calculate overall news score"""
        try:
            # Base score from sentiment
            sentiment_score = sentiment_analysis.get('sentiment_score', 0.0)
            impact_score = sentiment_analysis.get('impact_score', 0.0)
            confidence = sentiment_analysis.get('confidence', 0.0)
            
            # Event impact
            event_score = 0.0
            if events:
                event_impacts = [event.get('impact', 0.0) for event in events]
                event_sentiments = [event.get('sentiment', 0.0) for event in events]
                event_score = np.mean(event_impacts) * np.mean(event_sentiments)
            
            # Weighted combination
            news_score = (
                self.sentiment_weight * sentiment_score +
                self.impact_weight * impact_score +
                (1 - self.sentiment_weight - self.impact_weight) * event_score
            ) * confidence
            
            return max(-1.0, min(1.0, news_score))  # Normalize to [-1, 1]
            
        except Exception as e:
            logger.error(f"Error calculating news score: {e}")
            return 0.0
    
    def predict_market_reaction(self, news_score: float, technical_data: pd.DataFrame) -> Dict[str, Any]:
        """Predict market reaction to news"""
        try:
            if technical_data.empty:
                return {
                    'direction': 'neutral',
                    'magnitude': 0.0,
                    'confidence': 0.0,
                    'timeframe': 'short'
                }
            
            # Get recent price action
            recent_prices = technical_data['close'].tail(20)
            current_price = recent_prices.iloc[-1]
            price_change = (current_price - recent_prices.iloc[0]) / recent_prices.iloc[0]
            
            # Calculate volatility
            returns = recent_prices.pct_change().dropna()
            volatility = returns.std()
            
            # Predict reaction
            if abs(news_score) < 0.3:
                direction = 'neutral'
                magnitude = 0.0
            elif news_score > 0.3:
                direction = 'bullish'
                magnitude = min(0.1, abs(news_score) * 0.2)
            else:
                direction = 'bearish'
                magnitude = min(0.1, abs(news_score) * 0.2)
            
            # Adjust for volatility
            magnitude *= (1 + volatility * 10)
            
            # Calculate confidence
            confidence = min(1.0, abs(news_score) * 2)
            
            return {
                'direction': direction,
                'magnitude': magnitude,
                'confidence': confidence,
                'timeframe': 'short',
                'volatility_adjustment': volatility
            }
            
        except Exception as e:
            logger.error(f"Error predicting market reaction: {e}")
            return {
                'direction': 'neutral',
                'magnitude': 0.0,
                'confidence': 0.0,
                'timeframe': 'short'
            }
    
    def get_recent_news(self, symbol: str) -> List[Dict]:
        """Get recent news for a symbol"""
        try:
            # Simulate news data - in real implementation, this would fetch from news APIs
            mock_news = [
                {
                    'title': f'Strong earnings report for {symbol}',
                    'content': f'{symbol} reported better-than-expected earnings...',
                    'sentiment': 0.8,
                    'impact': 0.7,
                    'timestamp': datetime.now() - timedelta(hours=2),
                    'source': 'Financial Times'
                },
                {
                    'title': f'{symbol} announces new product launch',
                    'content': f'{symbol} has announced a revolutionary new product...',
                    'sentiment': 0.6,
                    'impact': 0.5,
                    'timestamp': datetime.now() - timedelta(hours=4),
                    'source': 'Reuters'
                },
                {
                    'title': f'Analyst downgrades {symbol}',
                    'content': f'Leading analyst firm has downgraded {symbol}...',
                    'sentiment': -0.4,
                    'impact': 0.3,
                    'timestamp': datetime.now() - timedelta(hours=6),
                    'source': 'Bloomberg'
                }
            ]
            
            # Filter by lookback period
            cutoff_time = datetime.now() - timedelta(hours=self.news_lookback_hours)
            recent_news = [news for news in mock_news if news['timestamp'] > cutoff_time]
            
            return recent_news
            
        except Exception as e:
            logger.error(f"Error getting recent news: {e}")
            return []
    
    def calculate_position_size(self, data: pd.DataFrame, account_balance: float, news_score: float) -> float:
        """Calculate position size based on news impact"""
        try:
            current_price = data['close'].iloc[-1]
            
            # Base position size
            risk_amount = account_balance * self.risk_per_trade
            
            # Adjust for news impact
            news_multiplier = 1.0 + abs(news_score) * 0.5  # Increase size for strong news
            adjusted_risk = risk_amount * news_multiplier
            
            # Calculate position size
            stop_loss_distance = current_price * self.stop_loss_percentage
            position_size = adjusted_risk / stop_loss_distance
            
            # Apply maximum position size limit
            max_position_value = account_balance * self.max_position_size
            max_position_size = max_position_value / current_price
            
            return min(position_size, max_position_size)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    def generate_signals(self, data: pd.DataFrame, symbol: str) -> List[Dict[str, Any]]:
        """Generate news-based trading signals"""
        if data.empty:
            return []
        
        try:
            # Get recent news
            news_data = self.get_recent_news(symbol)
            
            # Analyze sentiment
            sentiment_analysis = self.analyze_news_sentiment(news_data)
            
            # Detect events
            events = self.detect_news_events(news_data)
            
            # Calculate news score
            news_score = self.calculate_news_score(sentiment_analysis, events)
            
            # Predict market reaction
            reaction = self.predict_market_reaction(news_score, data)
            
            signals = []
            current_time = datetime.now()
            
            # Generate entry signals
            if self.position is None and abs(news_score) > self.sentiment_threshold:
                if news_score > self.sentiment_threshold and reaction['direction'] == 'bullish':
                    signals.append({
                        'signal': 'buy',
                        'price': data['close'].iloc[-1],
                        'timestamp': current_time,
                        'confidence': reaction['confidence'],
                        'reason': f"Positive news sentiment (score: {news_score:.2f})",
                        'news_analysis': {
                            'sentiment_score': sentiment_analysis['sentiment_score'],
                            'impact_score': sentiment_analysis['impact_score'],
                            'news_volume': sentiment_analysis['news_volume'],
                            'key_events': len(events),
                            'predicted_reaction': reaction
                        },
                        'setup_type': 'news_entry'
                    })
                
                elif news_score < -self.sentiment_threshold and reaction['direction'] == 'bearish':
                    signals.append({
                        'signal': 'sell',
                        'price': data['close'].iloc[-1],
                        'timestamp': current_time,
                        'confidence': reaction['confidence'],
                        'reason': f"Negative news sentiment (score: {news_score:.2f})",
                        'news_analysis': {
                            'sentiment_score': sentiment_analysis['sentiment_score'],
                            'impact_score': sentiment_analysis['impact_score'],
                            'news_volume': sentiment_analysis['news_volume'],
                            'key_events': len(events),
                            'predicted_reaction': reaction
                        },
                        'setup_type': 'news_entry'
                    })
            
            # Generate exit signals for existing positions
            elif self.position is not None:
                # Check position hold time
                if hasattr(self.position, 'entry_time'):
                    hold_time = current_time - self.position['entry_time']
                    if hold_time.total_seconds() / 3600 > self.position_hold_hours:
                        signals.append({
                            'signal': 'sell' if self.position['type'] == 'long' else 'buy',
                            'price': data['close'].iloc[-1],
                            'timestamp': current_time,
                            'confidence': 0.8,
                            'reason': f"Position hold time exceeded ({self.position_hold_hours}h)",
                            'position_id': self.position['id']
                        })
                
                # Check for news reversal
                if self.position['type'] == 'long' and news_score < -self.sentiment_threshold:
                    signals.append({
                        'signal': 'sell',
                        'price': data['close'].iloc[-1],
                        'timestamp': current_time,
                        'confidence': reaction['confidence'],
                        'reason': "News sentiment reversal",
                        'position_id': self.position['id']
                    })
                
                elif self.position['type'] == 'short' and news_score > self.sentiment_threshold:
                    signals.append({
                        'signal': 'buy',
                        'price': data['close'].iloc[-1],
                        'timestamp': current_time,
                        'confidence': reaction['confidence'],
                        'reason': "News sentiment reversal",
                        'position_id': self.position['id']
                    })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating news signals: {e}")
            return []
    
    def update_position(self, signal: Dict[str, Any], account_balance: float):
        """Update position based on signal"""
        if signal['signal'] == 'buy' and self.position is None:
            # Open long position
            news_score = signal.get('news_analysis', {}).get('sentiment_score', 0.0)
            position_size = self.calculate_position_size(
                self.get_latest_data(), account_balance, news_score
            )
            
            self.position = {
                'id': f"news_{datetime.now().timestamp()}",
                'type': 'long',
                'entry_price': signal['price'],
                'size': position_size,
                'entry_time': signal['timestamp'],
                'stop_loss': signal['price'] * (1 - self.stop_loss_percentage),
                'take_profit': signal['price'] * (1 + self.take_profit_percentage),
                'news_score': news_score
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
            'name': 'News-Based Trading Strategy',
            'description': 'Event-driven trading based on news sentiment analysis',
            'parameters': {
                'sentiment_threshold': self.sentiment_threshold,
                'impact_threshold': self.impact_threshold,
                'news_lookback_hours': self.news_lookback_hours,
                'position_hold_hours': self.position_hold_hours,
                'risk_per_trade': self.risk_per_trade,
                'stop_loss_percentage': self.stop_loss_percentage,
                'take_profit_percentage': self.take_profit_percentage
            },
            'current_position': self.position,
            'news_cache_size': len(self.news_cache),
            'sentiment_cache_size': len(self.sentiment_cache)
        }