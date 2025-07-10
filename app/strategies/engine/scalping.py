import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

class ScalpingStrategy:
    def __init__(self, 
                 rsi_period: int = 14, 
                 rsi_oversold: float = 30, 
                 rsi_overbought: float = 70,
                 macd_fast: int = 12, 
                 macd_slow: int = 26, 
                 macd_signal: int = 9,
                 bb_period: int = 20, 
                 bb_std: float = 2.0,
                 atr_period: int = 14,
                 timeframe: str = '1m',
                 allocation: float = 0.05,
                 ml_model: Optional[Any] = None):
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period
        self.timeframe = timeframe
        self.allocation = allocation
        self.ml_model = ml_model

    def calculate_rsi(self, prices: np.ndarray) -> np.ndarray:
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gains = pd.Series(gains).rolling(window=self.rsi_period).mean()
        avg_losses = pd.Series(losses).rolling(window=self.rsi_period).mean()
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        return rsi.values

    def calculate_macd(self, prices: np.ndarray) -> tuple:
        ema_fast = pd.Series(prices).ewm(span=self.macd_fast).mean()
        ema_slow = pd.Series(prices).ewm(span=self.macd_slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal).mean()
        histogram = macd_line - signal_line
        return macd_line.values, signal_line.values, histogram.values

    def calculate_bollinger_bands(self, prices: np.ndarray) -> tuple:
        sma = pd.Series(prices).rolling(window=self.bb_period).mean()
        std = pd.Series(prices).rolling(window=self.bb_period).std()
        upper_band = sma + (std * self.bb_std)
        lower_band = sma - (std * self.bb_std)
        return upper_band.values, sma.values, lower_band.values

    def calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(tr).rolling(window=self.atr_period).mean()
        return atr.values

    def generate_signals(self, prices: pd.DataFrame) -> List[Dict[str, Any]]:
        signals = []
        close_prices = prices['close'].values
        high_prices = prices['high'].values
        low_prices = prices['low'].values
        
        # Calculate indicators
        rsi = self.calculate_rsi(close_prices)
        macd_line, signal_line, histogram = self.calculate_macd(close_prices)
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close_prices)
        atr = self.calculate_atr(high_prices, low_prices, close_prices)
        
        # Generate signals
        for i in range(len(close_prices)):
            if i < max(self.rsi_period, self.bb_period, self.atr_period):
                continue
                
            signal = None
            confidence = 0
            
            # RSI signals
            if rsi[i] < self.rsi_oversold:
                signal = 'buy'
                confidence += 0.3
            elif rsi[i] > self.rsi_overbought:
                signal = 'sell'
                confidence += 0.3
                
            # MACD signals
            if macd_line[i] > signal_line[i] and macd_line[i-1] <= signal_line[i-1]:
                if signal == 'buy':
                    confidence += 0.3
                else:
                    signal = 'buy'
                    confidence += 0.3
            elif macd_line[i] < signal_line[i] and macd_line[i-1] >= signal_line[i-1]:
                if signal == 'sell':
                    confidence += 0.3
                else:
                    signal = 'sell'
                    confidence += 0.3
                    
            # Bollinger Bands signals
            if close_prices[i] < bb_lower[i]:
                if signal == 'buy':
                    confidence += 0.2
                else:
                    signal = 'buy'
                    confidence += 0.2
            elif close_prices[i] > bb_upper[i]:
                if signal == 'sell':
                    confidence += 0.2
                else:
                    signal = 'sell'
                    confidence += 0.2
                    
            if signal and confidence >= 0.5:
                signals.append({
                    'timestamp': i,
                    'signal': signal,
                    'confidence': confidence,
                    'price': close_prices[i],
                    'atr': atr[i],
                    'rsi': rsi[i],
                    'macd': macd_line[i],
                    'bb_position': (close_prices[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i])
                })
                
        return signals

    def backtest(self, prices: pd.DataFrame) -> Dict[str, Any]:
        signals = self.generate_signals(prices)
        # TODO: Implement full backtest logic with PnL, drawdown, etc.
        return {'signals': signals, 'pnl': 0, 'drawdown': 0}

    def ml_predict(self, features: np.ndarray) -> np.ndarray:
        if self.ml_model:
            return self.ml_model.predict(features)
        return np.zeros(features.shape[0])