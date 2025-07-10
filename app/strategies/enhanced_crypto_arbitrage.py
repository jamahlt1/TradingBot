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

class ArbitrageType(Enum):
    """Arbitrage Types"""
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    STATISTICAL = "statistical"
    TRIANGULAR = "triangular"
    CONVERGENCE = "convergence"

@dataclass
class ArbitrageOpportunity:
    """Arbitrage Opportunity Result"""
    opportunity_type: ArbitrageType
    symbol: str
    exchanges: List[str]
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    spread: float
    spread_percentage: float
    volume_available: float
    estimated_profit: float
    risk_score: float
    confidence: float
    ml_prediction: MLPrediction
    technical_analysis: Dict[str, Any]
    openrouter_analysis: Dict[str, Any]
    execution_time: datetime
    correlation_analysis: Dict[str, Any]

class EnhancedCryptoArbitrageStrategy(StrategyBase):
    """
    Enhanced Crypto Arbitrage Strategy with Advanced ML and OpenRouter
    - Multi-exchange arbitrage detection
    - ML-powered opportunity assessment
    - OpenRouter AI analysis
    - Correlation analysis
    - Risk management
    - Real-time execution
    """
    
    def __init__(self,
                 symbols: List[str],
                 exchanges: List[str],
                 ml_engine: EnhancedMLEngine = None,
                 risk_manager: RiskManager = None,
                 position_manager: PositionManager = None,
                 openrouter_api_key: str = None):
        
        super().__init__(symbols[0] if symbols else "BTC", "1m")
        
        self.symbols = symbols
        self.exchanges = exchanges
        
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
        
        # Arbitrage parameters
        self.min_spread_percentage = 0.5  # 0.5% minimum spread
        self.max_execution_time = 30  # seconds
        self.min_volume_threshold = 100  # USD
        self.max_risk_score = 0.7
        
        # ML parameters
        self.ml_confidence_threshold = 0.6
        self.ml_prediction_weight = 0.3
        self.technical_weight = 0.4
        self.ai_analysis_weight = 0.3
        
        # Correlation analysis
        self.correlation_threshold = 0.8
        self.correlation_window = 100
        
        # Performance tracking
        self.arbitrage_opportunities = []
        self.ml_predictions = []
        self.ai_analyses = []
        self.executed_trades = []
        
        # Exchange data cache
        self.exchange_data = {}
        self.price_cache = {}
        
    async def initialize(self):
        """Initialize the strategy"""
        try:
            logger.info(f"Initializing Enhanced Crypto Arbitrage Strategy for {self.symbols}")
            
            # Initialize exchange connections
            await self.initialize_exchanges()
            
            # Get historical data for ML training
            historical_data = await self.get_arbitrage_historical_data()
            
            if historical_data is not None and len(historical_data) > 100:
                # Prepare target variable (arbitrage success)
                historical_data['target'] = self.prepare_arbitrage_target(historical_data)
                
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
    
    async def initialize_exchanges(self):
        """Initialize exchange connections"""
        try:
            for exchange in self.exchanges:
                # Initialize exchange-specific connections
                self.exchange_data[exchange] = {
                    'connected': False,
                    'last_update': None,
                    'prices': {},
                    'volumes': {},
                    'orderbooks': {}
                }
            
            logger.info(f"Initialized {len(self.exchanges)} exchanges")
            
        except Exception as e:
            logger.error(f"Error initializing exchanges: {e}")
    
    async def get_arbitrage_historical_data(self) -> pd.DataFrame:
        """Get historical data for arbitrage analysis"""
        try:
            # Combine data from multiple exchanges
            all_data = []
            
            for symbol in self.symbols:
                for exchange in self.exchanges:
                    try:
                        data = await self.get_exchange_historical_data(symbol, exchange)
                        if data is not None:
                            data['symbol'] = symbol
                            data['exchange'] = exchange
                            all_data.append(data)
                    except Exception as e:
                        logger.error(f"Error getting data for {symbol} on {exchange}: {e}")
            
            if all_data:
                combined_data = pd.concat(all_data, ignore_index=True)
                return combined_data
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error getting arbitrage historical data: {e}")
            return None
    
    def prepare_arbitrage_target(self, data: pd.DataFrame) -> pd.Series:
        """Prepare target variable for arbitrage"""
        try:
            # Calculate arbitrage opportunities
            target = pd.Series(index=data.index, dtype=int)
            
            for symbol in self.symbols:
                symbol_data = data[data['symbol'] == symbol]
                
                if len(symbol_data) > 0:
                    # Calculate price spreads between exchanges
                    exchanges = symbol_data['exchange'].unique()
                    
                    for i, exchange1 in enumerate(exchanges):
                        for exchange2 in exchanges[i+1:]:
                            ex1_data = symbol_data[symbol_data['exchange'] == exchange1]
                            ex2_data = symbol_data[symbol_data['exchange'] == exchange2]
                            
                            if len(ex1_data) > 0 and len(ex2_data) > 0:
                                # Calculate spread
                                price1 = ex1_data['close'].iloc[-1]
                                price2 = ex2_data['close'].iloc[-1]
                                spread = abs(price1 - price2) / min(price1, price2)
                                
                                # Target based on spread threshold
                                if spread > self.min_spread_percentage / 100:
                                    target.loc[ex1_data.index] = 1
                                    target.loc[ex2_data.index] = 1
                                else:
                                    target.loc[ex1_data.index] = 0
                                    target.loc[ex2_data.index] = 0
            
            return target
            
        except Exception as e:
            logger.error(f"Error preparing arbitrage target: {e}")
            return pd.Series(0, index=data.index)
    
    async def scan_arbitrage_opportunities(self) -> List[ArbitrageOpportunity]:
        """Scan for arbitrage opportunities across exchanges"""
        try:
            opportunities = []
            
            # Get current prices from all exchanges
            await self.update_exchange_prices()
            
            for symbol in self.symbols:
                symbol_opportunities = await self.find_symbol_arbitrage(symbol)
                opportunities.extend(symbol_opportunities)
            
            # Sort by estimated profit
            opportunities.sort(key=lambda x: x.estimated_profit, reverse=True)
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error scanning arbitrage opportunities: {e}")
            return []
    
    async def update_exchange_prices(self):
        """Update prices from all exchanges"""
        try:
            for exchange in self.exchanges:
                for symbol in self.symbols:
                    try:
                        # Get current price and volume
                        price_data = await self.get_exchange_price(symbol, exchange)
                        if price_data:
                            self.exchange_data[exchange]['prices'][symbol] = price_data['price']
                            self.exchange_data[exchange]['volumes'][symbol] = price_data['volume']
                            self.exchange_data[exchange]['last_update'] = datetime.now()
                    except Exception as e:
                        logger.error(f"Error updating price for {symbol} on {exchange}: {e}")
                        
        except Exception as e:
            logger.error(f"Error updating exchange prices: {e}")
    
    async def find_symbol_arbitrage(self, symbol: str) -> List[ArbitrageOpportunity]:
        """Find arbitrage opportunities for a specific symbol"""
        try:
            opportunities = []
            
            # Get prices from all exchanges
            prices = {}
            volumes = {}
            
            for exchange in self.exchanges:
                if symbol in self.exchange_data[exchange]['prices']:
                    prices[exchange] = self.exchange_data[exchange]['prices'][symbol]
                    volumes[exchange] = self.exchange_data[exchange]['volumes'][symbol]
            
            if len(prices) < 2:
                return opportunities
            
            # Find best buy and sell opportunities
            exchanges = list(prices.keys())
            
            for i, ex1 in enumerate(exchanges):
                for ex2 in exchanges[i+1:]:
                    price1 = prices[ex1]
                    price2 = prices[ex2]
                    
                    # Calculate spread
                    spread = abs(price1 - price2)
                    spread_percentage = (spread / min(price1, price2)) * 100
                    
                    if spread_percentage >= self.min_spread_percentage:
                        # Determine buy/sell direction
                        if price1 < price2:
                            buy_exchange = ex1
                            sell_exchange = ex2
                            buy_price = price1
                            sell_price = price2
                        else:
                            buy_exchange = ex2
                            sell_exchange = ex1
                            buy_price = price2
                            sell_price = price1
                        
                        # Calculate estimated profit
                        volume_available = min(volumes[buy_exchange], volumes[sell_exchange])
                        estimated_profit = (sell_price - buy_price) * volume_available
                        
                        if estimated_profit >= self.min_volume_threshold:
                            # Get ML prediction
                            ml_prediction = await self.get_ml_prediction(symbol, buy_exchange, sell_exchange)
                            
                            # Get technical analysis
                            technical_analysis = await self.get_technical_analysis(symbol, buy_exchange, sell_exchange)
                            
                            # Get AI analysis
                            ai_analysis = await self.get_ai_analysis(symbol, buy_exchange, sell_exchange, spread_percentage)
                            
                            # Calculate risk score
                            risk_score = self.calculate_arbitrage_risk(symbol, buy_exchange, sell_exchange, spread_percentage)
                            
                            # Calculate confidence
                            confidence = self.calculate_arbitrage_confidence(ml_prediction, technical_analysis, ai_analysis, risk_score)
                            
                            # Create opportunity
                            opportunity = ArbitrageOpportunity(
                                opportunity_type=ArbitrageType.SPATIAL,
                                symbol=symbol,
                                exchanges=[buy_exchange, sell_exchange],
                                buy_exchange=buy_exchange,
                                sell_exchange=sell_exchange,
                                buy_price=buy_price,
                                sell_price=sell_price,
                                spread=spread,
                                spread_percentage=spread_percentage,
                                volume_available=volume_available,
                                estimated_profit=estimated_profit,
                                risk_score=risk_score,
                                confidence=confidence,
                                ml_prediction=ml_prediction,
                                technical_analysis=technical_analysis,
                                openrouter_analysis=ai_analysis,
                                execution_time=datetime.now(),
                                correlation_analysis=await self.get_correlation_analysis(symbol)
                            )
                            
                            opportunities.append(opportunity)
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error finding arbitrage for {symbol}: {e}")
            return []
    
    async def get_ml_prediction(self, symbol: str, buy_exchange: str, sell_exchange: str) -> MLPrediction:
        """Get ML prediction for arbitrage opportunity"""
        try:
            if not self.ml_engine:
                return MLPrediction(
                    prediction=0.5,
                    confidence=0.0,
                    model_type="none",
                    features_used=[],
                    timestamp=datetime.now(),
                    metadata={}
                )
            
            # Prepare features for arbitrage prediction
            features = {
                'symbol': symbol,
                'buy_exchange': buy_exchange,
                'sell_exchange': sell_exchange,
                'spread_percentage': self.get_spread_percentage(symbol, buy_exchange, sell_exchange),
                'volume_ratio': self.get_volume_ratio(symbol, buy_exchange, sell_exchange),
                'price_volatility': self.get_price_volatility(symbol),
                'exchange_volume': self.get_exchange_volume(buy_exchange, sell_exchange),
                'execution_speed': self.get_execution_speed(buy_exchange, sell_exchange)
            }
            
            # Convert to DataFrame for ML prediction
            feature_df = pd.DataFrame([features])
            
            return await self.ml_engine.predict(feature_df)
            
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
    
    async def get_technical_analysis(self, symbol: str, buy_exchange: str, sell_exchange: str) -> Dict[str, Any]:
        """Get technical analysis for arbitrage"""
        try:
            analysis = {}
            
            # Get price data from both exchanges
            buy_data = await self.get_exchange_historical_data(symbol, buy_exchange, days=30)
            sell_data = await self.get_exchange_historical_data(symbol, sell_exchange, days=30)
            
            if buy_data is not None and sell_data is not None:
                # Price correlation
                correlation = buy_data['close'].corr(sell_data['close'])
                analysis['price_correlation'] = correlation
                
                # Volatility analysis
                buy_volatility = buy_data['close'].pct_change().std()
                sell_volatility = sell_data['close'].pct_change().std()
                analysis['volatility_ratio'] = buy_volatility / sell_volatility if sell_volatility > 0 else 1
                
                # Spread trend
                spread_series = abs(buy_data['close'] - sell_data['close']) / buy_data['close']
                analysis['spread_trend'] = spread_series.tail(10).mean()
                analysis['spread_volatility'] = spread_series.std()
                
                # Volume analysis
                if 'volume' in buy_data.columns and 'volume' in sell_data.columns:
                    volume_correlation = buy_data['volume'].corr(sell_data['volume'])
                    analysis['volume_correlation'] = volume_correlation
                
                # Price efficiency
                analysis['price_efficiency'] = self.calculate_price_efficiency(buy_data, sell_data)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error getting technical analysis: {e}")
            return {}
    
    async def get_ai_analysis(self, symbol: str, buy_exchange: str, sell_exchange: str, spread_percentage: float) -> Dict[str, Any]:
        """Get AI analysis for arbitrage opportunity"""
        try:
            if not self.ml_engine.openrouter_client:
                return {}
            
            # Prepare analysis context
            context = {
                'symbol': symbol,
                'buy_exchange': buy_exchange,
                'sell_exchange': sell_exchange,
                'spread_percentage': spread_percentage,
                'buy_price': self.exchange_data[buy_exchange]['prices'].get(symbol, 0),
                'sell_price': self.exchange_data[sell_exchange]['prices'].get(symbol, 0)
            }
            
            # Create prompt for arbitrage analysis
            prompt = f"""
            Analyze the crypto arbitrage opportunity for {symbol}:
            
            Buy Exchange: {buy_exchange} at ${context['buy_price']:.2f}
            Sell Exchange: {sell_exchange} at ${context['sell_price']:.2f}
            Spread: {spread_percentage:.2f}%
            
            Provide:
            1. Arbitrage opportunity assessment
            2. Risk factors to consider
            3. Execution timing recommendations
            4. Volume considerations
            5. Market impact analysis
            6. Success probability estimate
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
    
    async def get_correlation_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get correlation analysis for arbitrage"""
        try:
            correlation_analysis = {}
            
            # Get price data from all exchanges
            exchange_prices = {}
            
            for exchange in self.exchanges:
                if symbol in self.exchange_data[exchange]['prices']:
                    historical_data = await self.get_exchange_historical_data(symbol, exchange, days=30)
                    if historical_data is not None:
                        exchange_prices[exchange] = historical_data['close']
            
            if len(exchange_prices) >= 2:
                # Calculate pairwise correlations
                correlations = {}
                exchanges = list(exchange_prices.keys())
                
                for i, ex1 in enumerate(exchanges):
                    for ex2 in exchanges[i+1:]:
                        if len(exchange_prices[ex1]) == len(exchange_prices[ex2]):
                            corr = exchange_prices[ex1].corr(exchange_prices[ex2])
                            correlations[f"{ex1}_{ex2}"] = corr
                
                correlation_analysis['pairwise_correlations'] = correlations
                correlation_analysis['avg_correlation'] = np.mean(list(correlations.values()))
                correlation_analysis['min_correlation'] = min(correlations.values()) if correlations else 0
                correlation_analysis['max_correlation'] = max(correlations.values()) if correlations else 0
            
            return correlation_analysis
            
        except Exception as e:
            logger.error(f"Error getting correlation analysis: {e}")
            return {}
    
    def calculate_arbitrage_risk(self, symbol: str, buy_exchange: str, sell_exchange: str, spread_percentage: float) -> float:
        """Calculate risk score for arbitrage opportunity"""
        try:
            risk_score = 0.0
            
            # Spread risk (higher spread = lower risk)
            if spread_percentage > 2.0:
                risk_score += 0.1
            elif spread_percentage > 1.0:
                risk_score += 0.3
            else:
                risk_score += 0.5
            
            # Exchange risk
            exchange_risk = self.get_exchange_risk(buy_exchange, sell_exchange)
            risk_score += exchange_risk * 0.3
            
            # Volume risk
            volume_risk = self.get_volume_risk(symbol, buy_exchange, sell_exchange)
            risk_score += volume_risk * 0.2
            
            # Execution speed risk
            speed_risk = self.get_execution_speed_risk(buy_exchange, sell_exchange)
            risk_score += speed_risk * 0.2
            
            # Market volatility risk
            volatility_risk = self.get_volatility_risk(symbol)
            risk_score += volatility_risk * 0.2
            
            return min(1.0, risk_score)
            
        except Exception as e:
            logger.error(f"Error calculating arbitrage risk: {e}")
            return 0.5
    
    def calculate_arbitrage_confidence(self, ml_prediction: MLPrediction, technical_analysis: Dict, ai_analysis: Dict, risk_score: float) -> float:
        """Calculate confidence score for arbitrage opportunity"""
        try:
            confidence = 0.0
            
            # ML prediction weight
            confidence += ml_prediction.prediction * self.ml_prediction_weight
            
            # Technical analysis weight
            technical_score = self.calculate_technical_score(technical_analysis)
            confidence += technical_score * self.technical_weight
            
            # AI analysis weight
            ai_score = self.extract_ai_score(ai_analysis)
            confidence += ai_score * self.ai_analysis_weight
            
            # Risk adjustment
            confidence *= (1 - risk_score)
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating arbitrage confidence: {e}")
            return 0.0
    
    def calculate_technical_score(self, technical_analysis: Dict) -> float:
        """Calculate technical analysis score"""
        try:
            score = 0.5  # Neutral starting point
            
            # Price correlation
            correlation = technical_analysis.get('price_correlation', 0)
            if correlation > 0.9:
                score += 0.2
            elif correlation < 0.7:
                score -= 0.2
            
            # Spread trend
            spread_trend = technical_analysis.get('spread_trend', 0)
            if spread_trend > 0.01:  # 1% spread
                score += 0.2
            elif spread_trend < 0.001:  # 0.1% spread
                score -= 0.2
            
            # Volatility ratio
            vol_ratio = technical_analysis.get('volatility_ratio', 1)
            if 0.8 < vol_ratio < 1.2:
                score += 0.1
            else:
                score -= 0.1
            
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
            
            # Arbitrage-specific sentiment analysis
            positive_words = ['profitable', 'opportunity', 'good', 'positive', 'execute', 'success']
            negative_words = ['risky', 'dangerous', 'avoid', 'negative', 'loss', 'failure']
            
            positive_count = sum(1 for word in positive_words if word in analysis_text)
            negative_count = sum(1 for word in negative_words if word in analysis_text)
            
            if positive_count > negative_count:
                return 0.7
            elif negative_count > positive_count:
                return 0.3
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error extracting AI score: {e}")
            return 0.5
    
    def get_exchange_risk(self, buy_exchange: str, sell_exchange: str) -> float:
        """Get exchange risk score"""
        try:
            # Simple exchange risk scoring
            exchange_risks = {
                'binance': 0.1,
                'coinbase': 0.1,
                'kraken': 0.2,
                'bitfinex': 0.3,
                'kucoin': 0.4,
                'huobi': 0.3,
                'okx': 0.3
            }
            
            buy_risk = exchange_risks.get(buy_exchange.lower(), 0.5)
            sell_risk = exchange_risks.get(sell_exchange.lower(), 0.5)
            
            return (buy_risk + sell_risk) / 2
            
        except Exception as e:
            logger.error(f"Error getting exchange risk: {e}")
            return 0.5
    
    def get_volume_risk(self, symbol: str, buy_exchange: str, sell_exchange: str) -> float:
        """Get volume risk score"""
        try:
            buy_volume = self.exchange_data[buy_exchange]['volumes'].get(symbol, 0)
            sell_volume = self.exchange_data[sell_exchange]['volumes'].get(symbol, 0)
            
            if buy_volume == 0 or sell_volume == 0:
                return 1.0
            
            # Lower volume = higher risk
            min_volume = min(buy_volume, sell_volume)
            
            if min_volume > 1000000:  # > $1M
                return 0.1
            elif min_volume > 100000:  # > $100K
                return 0.3
            elif min_volume > 10000:  # > $10K
                return 0.5
            else:
                return 0.8
                
        except Exception as e:
            logger.error(f"Error getting volume risk: {e}")
            return 0.5
    
    def get_execution_speed_risk(self, buy_exchange: str, sell_exchange: str) -> float:
        """Get execution speed risk score"""
        try:
            # Simple execution speed scoring
            speed_risks = {
                'binance': 0.1,
                'coinbase': 0.2,
                'kraken': 0.3,
                'bitfinex': 0.4,
                'kucoin': 0.5,
                'huobi': 0.4,
                'okx': 0.4
            }
            
            buy_speed = speed_risks.get(buy_exchange.lower(), 0.5)
            sell_speed = speed_risks.get(sell_exchange.lower(), 0.5)
            
            return (buy_speed + sell_speed) / 2
            
        except Exception as e:
            logger.error(f"Error getting execution speed risk: {e}")
            return 0.5
    
    def get_volatility_risk(self, symbol: str) -> float:
        """Get volatility risk score"""
        try:
            # Calculate volatility from recent price data
            all_prices = []
            
            for exchange in self.exchanges:
                if symbol in self.exchange_data[exchange]['prices']:
                    # Get recent price history
                    price_history = self.get_recent_price_history(symbol, exchange)
                    if price_history:
                        all_prices.extend(price_history)
            
            if len(all_prices) > 10:
                returns = np.diff(all_prices) / all_prices[:-1]
                volatility = np.std(returns)
                
                if volatility > 0.05:  # > 5% volatility
                    return 0.8
                elif volatility > 0.02:  # > 2% volatility
                    return 0.5
                else:
                    return 0.2
            
            return 0.5
            
        except Exception as e:
            logger.error(f"Error getting volatility risk: {e}")
            return 0.5
    
    def get_spread_percentage(self, symbol: str, buy_exchange: str, sell_exchange: str) -> float:
        """Get spread percentage between exchanges"""
        try:
            buy_price = self.exchange_data[buy_exchange]['prices'].get(symbol, 0)
            sell_price = self.exchange_data[sell_exchange]['prices'].get(symbol, 0)
            
            if buy_price > 0 and sell_price > 0:
                return abs(sell_price - buy_price) / min(buy_price, sell_price) * 100
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error getting spread percentage: {e}")
            return 0.0
    
    def get_volume_ratio(self, symbol: str, buy_exchange: str, sell_exchange: str) -> float:
        """Get volume ratio between exchanges"""
        try:
            buy_volume = self.exchange_data[buy_exchange]['volumes'].get(symbol, 0)
            sell_volume = self.exchange_data[sell_exchange]['volumes'].get(symbol, 0)
            
            if buy_volume > 0 and sell_volume > 0:
                return min(buy_volume, sell_volume) / max(buy_volume, sell_volume)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error getting volume ratio: {e}")
            return 0.0
    
    def get_price_volatility(self, symbol: str) -> float:
        """Get price volatility for symbol"""
        try:
            all_prices = []
            
            for exchange in self.exchanges:
                if symbol in self.exchange_data[exchange]['prices']:
                    price_history = self.get_recent_price_history(symbol, exchange)
                    if price_history:
                        all_prices.extend(price_history)
            
            if len(all_prices) > 10:
                returns = np.diff(all_prices) / all_prices[:-1]
                return np.std(returns)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error getting price volatility: {e}")
            return 0.0
    
    def get_exchange_volume(self, buy_exchange: str, sell_exchange: str) -> float:
        """Get total exchange volume"""
        try:
            buy_volume = sum(self.exchange_data[buy_exchange]['volumes'].values())
            sell_volume = sum(self.exchange_data[sell_exchange]['volumes'].values())
            
            return (buy_volume + sell_volume) / 2
            
        except Exception as e:
            logger.error(f"Error getting exchange volume: {e}")
            return 0.0
    
    def get_execution_speed(self, buy_exchange: str, sell_exchange: str) -> float:
        """Get execution speed score"""
        try:
            # Simple execution speed scoring
            speed_scores = {
                'binance': 0.9,
                'coinbase': 0.8,
                'kraken': 0.7,
                'bitfinex': 0.6,
                'kucoin': 0.5,
                'huobi': 0.6,
                'okx': 0.6
            }
            
            buy_speed = speed_scores.get(buy_exchange.lower(), 0.5)
            sell_speed = speed_scores.get(sell_exchange.lower(), 0.5)
            
            return (buy_speed + sell_speed) / 2
            
        except Exception as e:
            logger.error(f"Error getting execution speed: {e}")
            return 0.5
    
    def get_recent_price_history(self, symbol: str, exchange: str) -> List[float]:
        """Get recent price history for symbol on exchange"""
        try:
            # This would typically fetch from cache or API
            # For now, return empty list
            return []
            
        except Exception as e:
            logger.error(f"Error getting price history: {e}")
            return []
    
    async def execute_strategy(self, data: pd.DataFrame = None) -> Dict[str, Any]:
        """Execute the enhanced crypto arbitrage strategy"""
        try:
            # Scan for arbitrage opportunities
            opportunities = await self.scan_arbitrage_opportunities()
            
            if not opportunities:
                return {'action': 'hold', 'reason': 'no_opportunities'}
            
            # Filter opportunities by risk and confidence
            valid_opportunities = [
                opp for opp in opportunities
                if opp.risk_score <= self.max_risk_score and
                opp.confidence >= self.ml_confidence_threshold and
                opp.estimated_profit >= self.min_volume_threshold
            ]
            
            if not valid_opportunities:
                return {'action': 'hold', 'reason': 'no_valid_opportunities'}
            
            # Select best opportunity
            best_opportunity = valid_opportunities[0]
            
            # Risk management check
            if self.risk_manager:
                risk_check = await self.risk_manager.check_risk(best_opportunity)
                if not risk_check['approved']:
                    logger.warning(f"Risk check failed: {risk_check['reason']}")
                    return {'action': 'hold', 'reason': 'risk_check_failed'}
            
            # Execute arbitrage
            execution_result = await self.execute_arbitrage(best_opportunity)
            
            if execution_result['success']:
                self.executed_trades.append({
                    'opportunity': best_opportunity,
                    'execution_result': execution_result,
                    'timestamp': datetime.now()
                })
                
                return {
                    'action': 'arbitrage',
                    'opportunity': best_opportunity,
                    'reason': 'arbitrage_executed',
                    'profit': execution_result['profit'],
                    'execution_time': execution_result['execution_time']
                }
            else:
                return {
                    'action': 'hold',
                    'reason': 'execution_failed',
                    'error': execution_result['error']
                }
                
        except Exception as e:
            logger.error(f"Error executing strategy: {e}")
            return {'action': 'hold', 'reason': 'error', 'error': str(e)}
    
    async def execute_arbitrage(self, opportunity: ArbitrageOpportunity) -> Dict[str, Any]:
        """Execute arbitrage trade"""
        try:
            start_time = datetime.now()
            
            # Simulate arbitrage execution
            # In real implementation, this would place actual orders
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Simulate profit/loss
            actual_profit = opportunity.estimated_profit * 0.8  # 80% of estimated profit
            
            return {
                'success': True,
                'profit': actual_profit,
                'execution_time': execution_time,
                'buy_order_id': f"buy_{opportunity.symbol}_{opportunity.buy_exchange}_{start_time.timestamp()}",
                'sell_order_id': f"sell_{opportunity.symbol}_{opportunity.sell_exchange}_{start_time.timestamp()}"
            }
            
        except Exception as e:
            logger.error(f"Error executing arbitrage: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': 0
            }
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get strategy summary"""
        try:
            return {
                'strategy_name': 'Enhanced Crypto Arbitrage',
                'symbols': self.symbols,
                'exchanges': self.exchanges,
                'ml_models_trained': len(self.ml_engine.models) if self.ml_engine else 0,
                'opportunities_scanned': len(self.arbitrage_opportunities),
                'trades_executed': len(self.executed_trades),
                'ml_predictions': len(self.ml_predictions),
                'ai_analyses': len(self.ai_analyses),
                'parameters': {
                    'min_spread_percentage': self.min_spread_percentage,
                    'max_execution_time': self.max_execution_time,
                    'min_volume_threshold': self.min_volume_threshold,
                    'max_risk_score': self.max_risk_score,
                    'ml_confidence_threshold': self.ml_confidence_threshold,
                    'ml_prediction_weight': self.ml_prediction_weight,
                    'technical_weight': self.technical_weight,
                    'ai_analysis_weight': self.ai_analysis_weight,
                    'correlation_threshold': self.correlation_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting strategy summary: {e}")
            return {}