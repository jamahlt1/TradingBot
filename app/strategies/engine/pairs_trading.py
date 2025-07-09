import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from typing import Dict, Any, List, Optional

class PairsTradingStrategy:
    def __init__(self, 
                 lookback: int = 120, 
                 entry_z: float = 2.0, 
                 exit_z: float = 0.5, 
                 max_positions: int = 1, 
                 allocation: float = 0.1, 
                 ml_model: Optional[Any] = None):
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.max_positions = max_positions
        self.allocation = allocation
        self.ml_model = ml_model

    def find_cointegrated_pairs(self, prices: pd.DataFrame) -> List[Dict[str, Any]]:
        n = prices.shape[1]
        pairs = []
        for i in range(n):
            for j in range(i+1, n):
                score, pvalue, _ = coint(prices.iloc[:, i], prices.iloc[:, j])
                if pvalue < 0.05:
                    pairs.append({
                        'pair': (prices.columns[i], prices.columns[j]),
                        'pvalue': pvalue
                    })
        return pairs

    def compute_zscore(self, spread: np.ndarray) -> np.ndarray:
        mean = np.mean(spread)
        std = np.std(spread)
        return (spread - mean) / std

    def generate_signals(self, prices: pd.DataFrame) -> List[Dict[str, Any]]:
        signals = []
        pairs = self.find_cointegrated_pairs(prices)
        for pair_info in pairs:
            a, b = pair_info['pair']
            spread = prices[a] - prices[b]
            zscores = self.compute_zscore(spread.values[-self.lookback:])
            if zscores[-1] > self.entry_z:
                signals.append({'pair': (a, b), 'signal': 'short_spread', 'zscore': zscores[-1]})
            elif zscores[-1] < -self.entry_z:
                signals.append({'pair': (a, b), 'signal': 'long_spread', 'zscore': zscores[-1]})
            elif abs(zscores[-1]) < self.exit_z:
                signals.append({'pair': (a, b), 'signal': 'exit', 'zscore': zscores[-1]})
        return signals

    def backtest(self, prices: pd.DataFrame) -> Dict[str, Any]:
        # Example: simple backtest logic
        signals = self.generate_signals(prices)
        # TODO: Implement full backtest logic with PnL, drawdown, etc.
        return {'signals': signals, 'pnl': 0, 'drawdown': 0}

    def ml_predict(self, features: np.ndarray) -> np.ndarray:
        if self.ml_model:
            return self.ml_model.predict(features)
        return np.zeros(features.shape[0])