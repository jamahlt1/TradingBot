import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import aiohttp
import logging
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class CryptoArbitrageStrategy(BaseStrategy):
    """
    Advanced Crypto Arbitrage Strategy
    - Multi-exchange arbitrage
    - Correlation analysis
    - Risk-adjusted position sizing
    - Real-time opportunity detection
    """
    
    def __init__(self,
                 min_spread: float = 0.005,  # 0.5% minimum spread
                 max_position_size: float = 0.1,
                 correlation_threshold: float = 0.7,
                 max_slippage: float = 0.002,  # 0.2% max slippage
                 execution_delay: float = 0.1,  # 100ms max execution delay
                 risk_per_trade: float = 0.02,
                 max_concurrent_positions: int = 5,
                 exchanges: List[str] = None):
        
        super().__init__()
        self.min_spread = min_spread
        self.max_position_size = max_position_size
        self.correlation_threshold = correlation_threshold
        self.max_slippage = max_slippage
        self.execution_delay = execution_delay
        self.risk_per_trade = risk_per_trade
        self.max_concurrent_positions = max_concurrent_positions
        
        self.exchanges = exchanges or ['binance', 'coinbase', 'kraken', 'kucoin', 'bybit']
        self.positions = []
        self.opportunities = []
        self.correlation_matrix = {}
        
        # Exchange clients (would be initialized with real clients)
        self.exchange_clients = {}
        
    def get_parameter_space(self) -> Dict[str, Any]:
        """Get parameter space for Bayesian optimization"""
        return {
            'min_spread': {'type': 'real', 'min': 0.001, 'max': 0.02},
            'max_position_size': {'type': 'real', 'min': 0.05, 'max': 0.2},
            'correlation_threshold': {'type': 'real', 'min': 0.5, 'max': 0.9},
            'max_slippage': {'type': 'real', 'min': 0.001, 'max': 0.01},
            'execution_delay': {'type': 'real', 'min': 0.05, 'max': 0.5},
            'risk_per_trade': {'type': 'real', 'min': 0.01, 'max': 0.05}
        }
    
    async def get_exchange_prices(self, symbol: str) -> Dict[str, Dict]:
        """Get prices from multiple exchanges"""
        prices = {}
        
        for exchange in self.exchanges:
            try:
                # Simulate getting prices from different exchanges
                # In real implementation, this would call actual exchange APIs
                base_price = 50000  # Simulated BTC price
                spread = np.random.uniform(-0.01, 0.01)  # Random spread
                
                prices[exchange] = {
                    'bid': base_price * (1 - abs(spread)),
                    'ask': base_price * (1 + abs(spread)),
                    'volume': np.random.uniform(1000, 10000),
                    'timestamp': datetime.now(),
                    'latency': np.random.uniform(0.01, 0.1)
                }
            except Exception as e:
                logger.error(f"Error getting price from {exchange}: {e}")
        
        return prices
    
    def calculate_arbitrage_opportunities(self, prices: Dict[str, Dict]) -> List[Dict]:
        """Calculate arbitrage opportunities between exchanges"""
        opportunities = []
        
        exchanges = list(prices.keys())
        
        for i, exchange1 in enumerate(exchanges):
            for j, exchange2 in enumerate(exchanges[i+1:], i+1):
                try:
                    price1 = prices[exchange1]
                    price2 = prices[exchange2]
                    
                    # Calculate spread
                    spread1 = (price2['ask'] - price1['bid']) / price1['bid']
                    spread2 = (price1['ask'] - price2['bid']) / price2['bid']
                    
                    # Check if spread is profitable after fees and slippage
                    total_costs = self.max_slippage * 2 + self.execution_delay * 0.001
                    
                    if spread1 > self.min_spread + total_costs:
                        opportunities.append({
                            'type': 'arbitrage',
                            'buy_exchange': exchange1,
                            'sell_exchange': exchange2,
                            'symbol': 'BTC/USD',
                            'spread': spread1,
                            'buy_price': price1['bid'],
                            'sell_price': price2['ask'],
                            'profit_potential': spread1 - total_costs,
                            'volume': min(price1['volume'], price2['volume']),
                            'timestamp': datetime.now()
                        })
                    
                    elif spread2 > self.min_spread + total_costs:
                        opportunities.append({
                            'type': 'arbitrage',
                            'buy_exchange': exchange2,
                            'sell_exchange': exchange1,
                            'symbol': 'BTC/USD',
                            'spread': spread2,
                            'buy_price': price2['bid'],
                            'sell_price': price1['ask'],
                            'profit_potential': spread2 - total_costs,
                            'volume': min(price1['volume'], price2['volume']),
                            'timestamp': datetime.now()
                        })
                
                except Exception as e:
                    logger.error(f"Error calculating arbitrage: {e}")
        
        return sorted(opportunities, key=lambda x: x['profit_potential'], reverse=True)
    
    def calculate_correlation_matrix(self, symbols: List[str], data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate correlation matrix for symbols"""
        try:
            # Prepare price data for correlation calculation
            price_data = {}
            
            for symbol in symbols:
                if symbol in data and len(data[symbol]) > 0:
                    price_data[symbol] = data[symbol]['close'].values
            
            if len(price_data) < 2:
                return pd.DataFrame()
            
            # Create DataFrame for correlation
            df = pd.DataFrame(price_data)
            
            # Calculate correlation matrix
            correlation_matrix = df.corr()
            
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return pd.DataFrame()
    
    def find_correlated_pairs(self, correlation_matrix: pd.DataFrame) -> List[Tuple[str, str, float]]:
        """Find highly correlated trading pairs"""
        correlated_pairs = []
        
        try:
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    symbol1 = correlation_matrix.columns[i]
                    symbol2 = correlation_matrix.columns[j]
                    correlation = correlation_matrix.iloc[i, j]
                    
                    if abs(correlation) > self.correlation_threshold:
                        correlated_pairs.append((symbol1, symbol2, correlation))
            
            return sorted(correlated_pairs, key=lambda x: abs(x[2]), reverse=True)
            
        except Exception as e:
            logger.error(f"Error finding correlated pairs: {e}")
            return []
    
    def calculate_pair_trading_opportunities(self, correlated_pairs: List[Tuple], data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Calculate pair trading opportunities"""
        opportunities = []
        
        for symbol1, symbol2, correlation in correlated_pairs:
            try:
                if symbol1 not in data or symbol2 not in data:
                    continue
                
                df1 = data[symbol1]
                df2 = data[symbol2]
                
                if len(df1) < 20 or len(df2) < 20:
                    continue
                
                # Calculate spread between correlated assets
                price1 = df1['close'].iloc[-1]
                price2 = df2['close'].iloc[-1]
                
                # Normalize prices for comparison
                norm_price1 = price1 / df1['close'].mean()
                norm_price2 = price2 / df2['close'].mean()
                
                spread = abs(norm_price1 - norm_price2)
                
                if spread > self.min_spread:
                    # Determine which asset to buy/sell
                    if norm_price1 > norm_price2:
                        buy_symbol = symbol2
                        sell_symbol = symbol1
                        buy_price = price2
                        sell_price = price1
                    else:
                        buy_symbol = symbol1
                        sell_symbol = symbol2
                        buy_price = price1
                        sell_price = price2
                    
                    opportunities.append({
                        'type': 'pair_trading',
                        'buy_symbol': buy_symbol,
                        'sell_symbol': sell_symbol,
                        'correlation': correlation,
                        'spread': spread,
                        'buy_price': buy_price,
                        'sell_price': sell_price,
                        'profit_potential': spread - self.max_slippage * 2,
                        'timestamp': datetime.now()
                    })
            
            except Exception as e:
                logger.error(f"Error calculating pair trading opportunity: {e}")
        
        return sorted(opportunities, key=lambda x: x['profit_potential'], reverse=True)
    
    def calculate_position_size(self, opportunity: Dict, account_balance: float) -> float:
        """Calculate position size based on risk management"""
        try:
            # Risk-based position sizing
            risk_amount = account_balance * self.risk_per_trade
            
            if opportunity['type'] == 'arbitrage':
                # For arbitrage, use the spread as risk metric
                risk_per_unit = opportunity['spread']
            else:
                # For pair trading, use correlation-adjusted risk
                risk_per_unit = opportunity['spread'] * (1 - abs(opportunity['correlation']))
            
            if risk_per_unit <= 0:
                return 0
            
            position_size = risk_amount / risk_per_unit
            
            # Apply maximum position size limit
            max_position_value = account_balance * self.max_position_size
            max_position_size = max_position_value / opportunity['buy_price']
            
            return min(position_size, max_position_size)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    async def execute_arbitrage_trade(self, opportunity: Dict, position_size: float) -> Dict:
        """Execute arbitrage trade"""
        try:
            start_time = datetime.now()
            
            # Simulate buying on first exchange
            buy_order = {
                'exchange': opportunity['buy_exchange'],
                'symbol': opportunity['symbol'],
                'side': 'buy',
                'amount': position_size,
                'price': opportunity['buy_price'],
                'timestamp': start_time
            }
            
            # Simulate selling on second exchange
            sell_order = {
                'exchange': opportunity['sell_exchange'],
                'symbol': opportunity['symbol'],
                'side': 'sell',
                'amount': position_size,
                'price': opportunity['sell_price'],
                'timestamp': start_time + timedelta(milliseconds=50)
            }
            
            # Calculate actual profit
            buy_cost = buy_order['amount'] * buy_order['price']
            sell_revenue = sell_order['amount'] * sell_order['price']
            profit = sell_revenue - buy_cost
            
            trade_result = {
                'trade_id': f"arb_{int(start_time.timestamp())}",
                'opportunity': opportunity,
                'buy_order': buy_order,
                'sell_order': sell_order,
                'position_size': position_size,
                'profit': profit,
                'profit_percentage': (profit / buy_cost) * 100,
                'execution_time': (sell_order['timestamp'] - start_time).total_seconds(),
                'status': 'executed',
                'timestamp': datetime.now()
            }
            
            # Add to positions
            self.positions.append(trade_result)
            
            return trade_result
            
        except Exception as e:
            logger.error(f"Error executing arbitrage trade: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    async def execute_pair_trade(self, opportunity: Dict, position_size: float) -> Dict:
        """Execute pair trading trade"""
        try:
            start_time = datetime.now()
            
            # Simulate buying the undervalued asset
            buy_order = {
                'symbol': opportunity['buy_symbol'],
                'side': 'buy',
                'amount': position_size,
                'price': opportunity['buy_price'],
                'timestamp': start_time
            }
            
            # Simulate selling the overvalued asset
            sell_order = {
                'symbol': opportunity['sell_symbol'],
                'side': 'sell',
                'amount': position_size,
                'price': opportunity['sell_price'],
                'timestamp': start_time + timedelta(milliseconds=50)
            }
            
            trade_result = {
                'trade_id': f"pair_{int(start_time.timestamp())}",
                'opportunity': opportunity,
                'buy_order': buy_order,
                'sell_order': sell_order,
                'position_size': position_size,
                'status': 'executed',
                'timestamp': datetime.now()
            }
            
            # Add to positions
            self.positions.append(trade_result)
            
            return trade_result
            
        except Exception as e:
            logger.error(f"Error executing pair trade: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def generate_signals(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate trading signals based on arbitrage opportunities"""
        signals = []
        
        try:
            # Get current prices from all exchanges
            symbol = 'BTC/USD'  # Example symbol
            prices = asyncio.run(self.get_exchange_prices(symbol))
            
            # Calculate arbitrage opportunities
            arbitrage_opportunities = self.calculate_arbitrage_opportunities(prices)
            
            # Calculate correlation-based opportunities
            symbols = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'ADA/USD']
            correlation_matrix = self.calculate_correlation_matrix(symbols, {})
            correlated_pairs = self.find_correlated_pairs(correlation_matrix)
            pair_opportunities = self.calculate_pair_trading_opportunities(correlated_pairs, {})
            
            # Combine all opportunities
            all_opportunities = arbitrage_opportunities + pair_opportunities
            
            # Generate signals for top opportunities
            for opportunity in all_opportunities[:3]:  # Top 3 opportunities
                if opportunity['profit_potential'] > self.min_spread:
                    signals.append({
                        'signal': 'arbitrage',
                        'opportunity': opportunity,
                        'confidence': min(opportunity['profit_potential'] / 0.01, 1.0),
                        'timestamp': datetime.now(),
                        'type': opportunity['type']
                    })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return []
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information"""
        return {
            'name': 'Crypto Arbitrage Strategy',
            'description': 'Multi-exchange arbitrage with correlation analysis',
            'parameters': {
                'min_spread': self.min_spread,
                'max_position_size': self.max_position_size,
                'correlation_threshold': self.correlation_threshold,
                'max_slippage': self.max_slippage,
                'execution_delay': self.execution_delay,
                'risk_per_trade': self.risk_per_trade
            },
            'exchanges': self.exchanges,
            'active_positions': len(self.positions),
            'total_opportunities': len(self.opportunities)
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get strategy performance metrics"""
        if not self.positions:
            return {}
        
        profits = [pos['profit'] for pos in self.positions if 'profit' in pos]
        
        return {
            'total_trades': len(self.positions),
            'profitable_trades': len([p for p in profits if p > 0]),
            'total_profit': sum(profits),
            'average_profit': np.mean(profits) if profits else 0,
            'profit_std': np.std(profits) if profits else 0,
            'win_rate': len([p for p in profits if p > 0]) / len(profits) if profits else 0
        }