from .pairs_trading import PairsTradingStrategy
from .scalping import ScalpingStrategy
from .sentiment import SentimentStrategy
from .trend_following import TrendFollowingStrategy
from .swing_trading import SwingTradingStrategy
from .crypto_arbitrage import CryptoArbitrageStrategy
from .news_based import NewsBasedStrategy
from .ict import ICTStrategy
from .twap import TWAPStrategy
from .hedging import HedgingStrategy

__all__ = [
    'PairsTradingStrategy',
    'ScalpingStrategy', 
    'SentimentStrategy',
    'TrendFollowingStrategy',
    'SwingTradingStrategy',
    'CryptoArbitrageStrategy',
    'NewsBasedStrategy',
    'ICTStrategy',
    'TWAPStrategy',
    'HedgingStrategy'
]