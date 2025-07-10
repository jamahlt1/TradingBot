import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import re
from textblob import TextBlob
import requests
from datetime import datetime, timedelta

class SentimentTradingStrategy:
    def __init__(self, 
                 sentiment_threshold: float = 0.2,
                 news_weight: float = 0.4,
                 social_weight: float = 0.3,
                 technical_weight: float = 0.3,
                 lookback_hours: int = 24,
                 allocation: float = 0.1,
                 ml_model: Optional[Any] = None):
        self.sentiment_threshold = sentiment_threshold
        self.news_weight = news_weight
        self.social_weight = social_weight
        self.technical_weight = technical_weight
        self.lookback_hours = lookback_hours
        self.allocation = allocation
        self.ml_model = ml_model

    def analyze_text_sentiment(self, text: str) -> float:
        """Analyze sentiment of text using TextBlob"""
        blob = TextBlob(text)
        return blob.sentiment.polarity

    def fetch_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Fetch news sentiment for a symbol"""
        # Example: using a news API (replace with actual API)
        try:
            # This is a placeholder - replace with actual news API
            news_data = {
                'articles': [
                    {'title': f'Positive news about {symbol}', 'sentiment': 0.3},
                    {'title': f'Neutral news about {symbol}', 'sentiment': 0.0},
                    {'title': f'Negative news about {symbol}', 'sentiment': -0.2}
                ]
            }
            
            sentiments = [article['sentiment'] for article in news_data['articles']]
            avg_sentiment = np.mean(sentiments) if sentiments else 0.0
            
            return {
                'avg_sentiment': avg_sentiment,
                'article_count': len(news_data['articles']),
                'articles': news_data['articles']
            }
        except Exception as e:
            return {'avg_sentiment': 0.0, 'article_count': 0, 'articles': []}

    def fetch_social_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Fetch social media sentiment for a symbol"""
        # Example: using Twitter/Reddit APIs (replace with actual APIs)
        try:
            # This is a placeholder - replace with actual social media APIs
            social_data = {
                'twitter': {'sentiment': 0.1, 'volume': 1000},
                'reddit': {'sentiment': -0.05, 'volume': 500},
                'telegram': {'sentiment': 0.2, 'volume': 200}
            }
            
            total_sentiment = sum(data['sentiment'] * data['volume'] for data in social_data.values())
            total_volume = sum(data['volume'] for data in social_data.values())
            avg_sentiment = total_sentiment / total_volume if total_volume > 0 else 0.0
            
            return {
                'avg_sentiment': avg_sentiment,
                'total_volume': total_volume,
                'platforms': social_data
            }
        except Exception as e:
            return {'avg_sentiment': 0.0, 'total_volume': 0, 'platforms': {}}

    def calculate_technical_sentiment(self, prices: pd.DataFrame) -> float:
        """Calculate technical sentiment based on price action"""
        if len(prices) < 20:
            return 0.0
            
        close_prices = prices['close'].values
        returns = np.diff(close_prices) / close_prices[:-1]
        
        # Calculate various technical indicators
        sma_20 = np.mean(close_prices[-20:])
        sma_50 = np.mean(close_prices[-50:]) if len(close_prices) >= 50 else sma_20
        
        # Price momentum
        momentum = (close_prices[-1] - close_prices[-5]) / close_prices[-5] if len(close_prices) >= 5 else 0
        
        # Volatility
        volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0
        
        # Technical sentiment score
        price_sentiment = 1.0 if close_prices[-1] > sma_20 else -1.0
        trend_sentiment = 1.0 if sma_20 > sma_50 else -1.0
        momentum_sentiment = np.tanh(momentum * 10)  # Normalize momentum
        volatility_sentiment = -np.tanh(volatility * 100)  # High volatility = negative sentiment
        
        technical_sentiment = (price_sentiment + trend_sentiment + momentum_sentiment + volatility_sentiment) / 4
        return technical_sentiment

    def generate_signals(self, symbol: str, prices: pd.DataFrame = None) -> List[Dict[str, Any]]:
        """Generate trading signals based on sentiment analysis"""
        signals = []
        
        # Fetch news sentiment
        news_sentiment = self.fetch_news_sentiment(symbol)
        
        # Fetch social sentiment
        social_sentiment = self.fetch_social_sentiment(symbol)
        
        # Calculate technical sentiment
        technical_sentiment = self.calculate_technical_sentiment(prices) if prices is not None else 0.0
        
        # Combine sentiments with weights
        combined_sentiment = (
            news_sentiment['avg_sentiment'] * self.news_weight +
            social_sentiment['avg_sentiment'] * self.social_weight +
            technical_sentiment * self.technical_weight
        )
        
        # Generate signal based on combined sentiment
        signal = None
        confidence = abs(combined_sentiment)
        
        if combined_sentiment > self.sentiment_threshold:
            signal = 'buy'
        elif combined_sentiment < -self.sentiment_threshold:
            signal = 'sell'
        
        if signal and confidence >= 0.3:
            signals.append({
                'symbol': symbol,
                'signal': signal,
                'confidence': confidence,
                'combined_sentiment': combined_sentiment,
                'news_sentiment': news_sentiment['avg_sentiment'],
                'social_sentiment': social_sentiment['avg_sentiment'],
                'technical_sentiment': technical_sentiment,
                'timestamp': datetime.now().isoformat()
            })
        
        return signals

    def backtest(self, symbol: str, prices: pd.DataFrame) -> Dict[str, Any]:
        """Backtest sentiment strategy"""
        signals = self.generate_signals(symbol, prices)
        # TODO: Implement full backtest logic with PnL, drawdown, etc.
        return {'signals': signals, 'pnl': 0, 'drawdown': 0}

    def ml_predict(self, features: np.ndarray) -> np.ndarray:
        """ML prediction for sentiment analysis"""
        if self.ml_model:
            return self.ml_model.predict(features)
        return np.zeros(features.shape[0])