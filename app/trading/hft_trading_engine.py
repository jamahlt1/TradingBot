import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import socket
import struct

from app.core.strategy_base import StrategyBase
from app.core.risk_manager import RiskManager
from app.core.position_manager import PositionManager
from app.ml.enhanced_ml_engine import EnhancedMLEngine, MLPrediction

logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Order Types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"

class OrderSide(Enum):
    """Order Sides"""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """Order Status"""
    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

@dataclass
class HFTOrder:
    """HFT Order Structure"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float]
    stop_price: Optional[float]
    time_in_force: str = "IOC"  # Immediate or Cancel
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_price: float = 0.0
    commission: float = 0.0
    created_at: datetime = None
    updated_at: datetime = None
    expires_at: Optional[datetime] = None
    parent_order_id: Optional[str] = None
    strategy_id: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class MarketData:
    """Real-time Market Data"""
    symbol: str
    timestamp: datetime
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    last_price: float
    last_size: float
    volume: float
    high: float
    low: float
    open: float
    close: float
    spread: float
    mid_price: float
    tick_size: float
    lot_size: float

@dataclass
class HFTMetrics:
    """HFT Performance Metrics"""
    total_orders: int
    filled_orders: int
    cancelled_orders: int
    rejected_orders: int
    fill_rate: float
    avg_fill_time: float
    avg_spread_capture: float
    total_pnl: float
    total_volume: float
    latency_stats: Dict[str, float]
    slippage_stats: Dict[str, float]
    market_impact: float
    alpha_generation: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float

class HFTTradingEngine:
    """
    High-Frequency Trading Engine
    - Ultra-low latency execution
    - Market making strategies
    - Statistical arbitrage
    - Order book analysis
    - Real-time risk management
    - Multi-venue trading
    - Co-location support
    """
    
    def __init__(self,
                 exchange_configs: Dict[str, Any],
                 risk_manager: RiskManager,
                 position_manager: PositionManager,
                 ml_engine: EnhancedMLEngine = None,
                 max_positions: int = 100,
                 max_order_size: float = 1000000,
                 max_daily_loss: float = 0.05,
                 latency_threshold: float = 0.001,  # 1ms
                 enable_co_location: bool = False,
                 enable_dma: bool = True):
        
        self.exchange_configs = exchange_configs
        self.risk_manager = risk_manager
        self.position_manager = position_manager
        self.ml_engine = ml_engine
        self.max_positions = max_positions
        self.max_order_size = max_order_size
        self.max_daily_loss = max_daily_loss
        self.latency_threshold = latency_threshold
        self.enable_co_location = enable_co_location
        self.enable_dma = enable_dma
        
        # HFT state
        self.active_orders = {}
        self.order_history = []
        self.market_data_cache = {}
        self.position_cache = {}
        self.risk_limits = {}
        self.performance_metrics = {}
        
        # Latency tracking
        self.latency_stats = {
            'order_submission': [],
            'market_data_processing': [],
            'signal_generation': [],
            'risk_check': []
        }
        
        # Threading and concurrency
        self.order_queue = queue.Queue()
        self.market_data_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.running = False
        
        # Initialize exchanges
        self.exchanges = self._initialize_exchanges()
        
        # Initialize strategies
        self.strategies = {}
        
        logger.info("HFT Trading Engine initialized")
    
    def _initialize_exchanges(self) -> Dict[str, Any]:
        """Initialize exchange connections"""
        exchanges = {}
        
        try:
            for exchange_name, config in self.exchange_configs.items():
                # Create exchange connection
                exchange = self._create_exchange_connection(exchange_name, config)
                exchanges[exchange_name] = exchange
                
                logger.info(f"Initialized {exchange_name} connection")
            
            return exchanges
            
        except Exception as e:
            logger.error(f"Error initializing exchanges: {e}")
            return {}
    
    def _create_exchange_connection(self, exchange_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create exchange connection with low latency setup"""
        try:
            connection = {
                'name': exchange_name,
                'config': config,
                'connected': False,
                'last_heartbeat': None,
                'order_count': 0,
                'volume_today': 0.0,
                'latency': 0.0
            }
            
            # Setup co-location if enabled
            if self.enable_co_location and config.get('co_location_enabled', False):
                connection['co_location'] = self._setup_co_location(config)
            
            # Setup DMA if enabled
            if self.enable_dma and config.get('dma_enabled', False):
                connection['dma'] = self._setup_dma(config)
            
            return connection
            
        except Exception as e:
            logger.error(f"Error creating exchange connection: {e}")
            return {}
    
    def _setup_co_location(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Setup co-location for ultra-low latency"""
        try:
            co_location = {
                'enabled': True,
                'server_location': config.get('server_location'),
                'network_latency': config.get('network_latency', 0.0001),
                'order_routing': config.get('order_routing'),
                'market_data_feed': config.get('market_data_feed')
            }
            
            # Setup direct market data feed
            if co_location['market_data_feed']:
                self._setup_market_data_feed(co_location['market_data_feed'])
            
            return co_location
            
        except Exception as e:
            logger.error(f"Error setting up co-location: {e}")
            return {'enabled': False}
    
    def _setup_dma(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Setup Direct Market Access"""
        try:
            dma = {
                'enabled': True,
                'order_routing': config.get('dma_order_routing'),
                'market_data_routing': config.get('dma_market_data_routing'),
                'risk_checks': config.get('dma_risk_checks', True),
                'pre_trade_checks': config.get('dma_pre_trade_checks', True)
            }
            
            return dma
            
        except Exception as e:
            logger.error(f"Error setting up DMA: {e}")
            return {'enabled': False}
    
    def _setup_market_data_feed(self, feed_config: Dict[str, Any]):
        """Setup direct market data feed"""
        try:
            # Setup UDP multicast for market data
            if feed_config.get('protocol') == 'udp':
                self._setup_udp_feed(feed_config)
            elif feed_config.get('protocol') == 'tcp':
                self._setup_tcp_feed(feed_config)
            
            logger.info("Market data feed setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up market data feed: {e}")
    
    async def start_engine(self):
        """Start HFT trading engine"""
        try:
            logger.info("Starting HFT Trading Engine")
            
            self.running = True
            
            # Start background tasks
            asyncio.create_task(self._market_data_processor())
            asyncio.create_task(self._order_processor())
            asyncio.create_task(self._risk_monitor())
            asyncio.create_task(self._performance_monitor())
            
            # Connect to exchanges
            await self._connect_exchanges()
            
            logger.info("HFT Trading Engine started successfully")
            
        except Exception as e:
            logger.error(f"Error starting HFT engine: {e}")
            raise
    
    async def stop_engine(self):
        """Stop HFT trading engine"""
        try:
            logger.info("Stopping HFT Trading Engine")
            
            self.running = False
            
            # Cancel all active orders
            await self._cancel_all_orders()
            
            # Close exchange connections
            await self._disconnect_exchanges()
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            logger.info("HFT Trading Engine stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping HFT engine: {e}")
            raise
    
    async def _connect_exchanges(self):
        """Connect to all exchanges"""
        try:
            for exchange_name, exchange in self.exchanges.items():
                # Connect to exchange
                connected = await self._connect_exchange(exchange_name, exchange)
                
                if connected:
                    exchange['connected'] = True
                    exchange['last_heartbeat'] = datetime.now()
                    logger.info(f"Connected to {exchange_name}")
                else:
                    logger.error(f"Failed to connect to {exchange_name}")
            
        except Exception as e:
            logger.error(f"Error connecting to exchanges: {e}")
    
    async def _connect_exchange(self, exchange_name: str, exchange: Dict[str, Any]) -> bool:
        """Connect to specific exchange"""
        try:
            config = exchange['config']
            
            # Setup connection based on exchange type
            if config.get('type') == 'rest':
                return await self._connect_rest_exchange(exchange_name, config)
            elif config.get('type') == 'websocket':
                return await self._connect_websocket_exchange(exchange_name, config)
            elif config.get('type') == 'fix':
                return await self._connect_fix_exchange(exchange_name, config)
            else:
                logger.error(f"Unsupported exchange type: {config.get('type')}")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to {exchange_name}: {e}")
            return False
    
    async def _connect_rest_exchange(self, exchange_name: str, config: Dict[str, Any]) -> bool:
        """Connect to REST API exchange"""
        try:
            # Test connection
            response = await self._make_request(
                config['base_url'],
                config['endpoints']['ping'],
                headers=config.get('headers', {})
            )
            
            return response is not None
            
        except Exception as e:
            logger.error(f"Error connecting to REST exchange {exchange_name}: {e}")
            return False
    
    async def _connect_websocket_exchange(self, exchange_name: str, config: Dict[str, Any]) -> bool:
        """Connect to WebSocket exchange"""
        try:
            # Setup WebSocket connection
            websocket = await self._create_websocket_connection(
                config['websocket_url'],
                config.get('auth_headers', {})
            )
            
            if websocket:
                # Subscribe to market data
                await self._subscribe_market_data(websocket, config.get('subscriptions', []))
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to WebSocket exchange {exchange_name}: {e}")
            return False
    
    async def _connect_fix_exchange(self, exchange_name: str, config: Dict[str, Any]) -> bool:
        """Connect to FIX protocol exchange"""
        try:
            # Setup FIX connection
            fix_connection = await self._create_fix_connection(
                config['fix_host'],
                config['fix_port'],
                config['sender_comp_id'],
                config['target_comp_id']
            )
            
            if fix_connection:
                # Logon to FIX session
                await self._fix_logon(fix_connection, config)
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to FIX exchange {exchange_name}: {e}")
            return False
    
    async def submit_order(self, order: HFTOrder) -> Dict[str, Any]:
        """Submit order with ultra-low latency"""
        try:
            start_time = time.time()
            
            # Pre-trade risk checks
            risk_check = await self._pre_trade_risk_check(order)
            if not risk_check['approved']:
                return {
                    'success': False,
                    'error': 'Risk check failed',
                    'details': risk_check['reasons']
                }
            
            # Route order to appropriate exchange
            exchange_name = self._select_exchange(order.symbol)
            exchange = self.exchanges.get(exchange_name)
            
            if not exchange or not exchange['connected']:
                return {
                    'success': False,
                    'error': f'Exchange {exchange_name} not available'
                }
            
            # Submit order
            order_result = await self._submit_order_to_exchange(exchange_name, order)
            
            # Track latency
            latency = time.time() - start_time
            self.latency_stats['order_submission'].append(latency)
            
            if order_result['success']:
                # Add to active orders
                self.active_orders[order.order_id] = order
                
                # Update metrics
                exchange['order_count'] += 1
                exchange['volume_today'] += order.quantity * (order.price or 0)
            
            return order_result
            
        except Exception as e:
            logger.error(f"Error submitting order: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _pre_trade_risk_check(self, order: HFTOrder) -> Dict[str, Any]:
        """Perform pre-trade risk checks"""
        try:
            reasons = []
            
            # Position limit check
            current_position = self.position_cache.get(order.symbol, 0)
            if order.side == OrderSide.BUY:
                new_position = current_position + order.quantity
            else:
                new_position = current_position - order.quantity
            
            if abs(new_position) > self.max_positions:
                reasons.append(f"Position limit exceeded: {new_position}")
            
            # Order size check
            if order.quantity > self.max_order_size:
                reasons.append(f"Order size too large: {order.quantity}")
            
            # Daily loss check
            daily_pnl = self._calculate_daily_pnl()
            if daily_pnl < -(self.max_daily_loss * self._get_account_value()):
                reasons.append(f"Daily loss limit exceeded: {daily_pnl}")
            
            # Market impact check
            market_impact = self._calculate_market_impact(order)
            if market_impact > 0.001:  # 0.1% impact limit
                reasons.append(f"Market impact too high: {market_impact}")
            
            # Latency check
            avg_latency = np.mean(self.latency_stats['order_submission'][-100:]) if self.latency_stats['order_submission'] else 0
            if avg_latency > self.latency_threshold:
                reasons.append(f"Latency too high: {avg_latency}")
            
            return {
                'approved': len(reasons) == 0,
                'reasons': reasons
            }
            
        except Exception as e:
            logger.error(f"Error in pre-trade risk check: {e}")
            return {
                'approved': False,
                'reasons': [f"Risk check error: {str(e)}"]
            }
    
    def _select_exchange(self, symbol: str) -> str:
        """Select best exchange for order routing"""
        try:
            # Simple routing based on symbol
            if symbol.endswith('USDT'):
                return 'binance'
            elif symbol.endswith('USD'):
                return 'coinbase'
            else:
                return 'binance'  # Default
                
        except Exception as e:
            logger.error(f"Error selecting exchange: {e}")
            return 'binance'
    
    async def _submit_order_to_exchange(self, exchange_name: str, order: HFTOrder) -> Dict[str, Any]:
        """Submit order to specific exchange"""
        try:
            exchange = self.exchanges[exchange_name]
            config = exchange['config']
            
            # Prepare order data
            order_data = {
                'symbol': order.symbol,
                'side': order.side.value,
                'type': order.order_type.value,
                'quantity': order.quantity,
                'price': order.price,
                'time_in_force': order.time_in_force
            }
            
            # Submit based on exchange type
            if config.get('type') == 'rest':
                return await self._submit_rest_order(exchange_name, order_data)
            elif config.get('type') == 'websocket':
                return await self._submit_websocket_order(exchange_name, order_data)
            elif config.get('type') == 'fix':
                return await self._submit_fix_order(exchange_name, order_data)
            else:
                return {
                    'success': False,
                    'error': f'Unsupported exchange type: {config.get("type")}'
                }
                
        except Exception as e:
            logger.error(f"Error submitting order to {exchange_name}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _submit_rest_order(self, exchange_name: str, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submit order via REST API"""
        try:
            exchange = self.exchanges[exchange_name]
            config = exchange['config']
            
            # Make REST request
            response = await self._make_request(
                config['base_url'],
                config['endpoints']['order'],
                method='POST',
                data=order_data,
                headers=config.get('headers', {})
            )
            
            if response and response.get('status') == 'success':
                return {
                    'success': True,
                    'order_id': response.get('order_id'),
                    'exchange': exchange_name
                }
            else:
                return {
                    'success': False,
                    'error': response.get('message', 'Unknown error')
                }
                
        except Exception as e:
            logger.error(f"Error submitting REST order: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel active order"""
        try:
            order = self.active_orders.get(order_id)
            if not order:
                return {
                    'success': False,
                    'error': 'Order not found'
                }
            
            # Cancel order on exchange
            exchange_name = self._get_order_exchange(order_id)
            cancel_result = await self._cancel_order_on_exchange(exchange_name, order_id)
            
            if cancel_result['success']:
                # Update order status
                order.status = OrderStatus.CANCELLED
                order.updated_at = datetime.now()
                
                # Remove from active orders
                del self.active_orders[order_id]
                
                # Add to history
                self.order_history.append(order)
            
            return cancel_result
            
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _market_data_processor(self):
        """Process market data in real-time"""
        try:
            while self.running:
                try:
                    # Get market data from queue
                    market_data = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            self.executor, self.market_data_queue.get
                        ),
                        timeout=0.001
                    )
                    
                    # Process market data
                    await self._process_market_data(market_data)
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error processing market data: {e}")
                    
        except Exception as e:
            logger.error(f"Error in market data processor: {e}")
    
    async def _process_market_data(self, market_data: MarketData):
        """Process incoming market data"""
        try:
            start_time = time.time()
            
            # Update cache
            self.market_data_cache[market_data.symbol] = market_data
            
            # Update strategies
            await self._update_strategies(market_data)
            
            # Track latency
            latency = time.time() - start_time
            self.latency_stats['market_data_processing'].append(latency)
            
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
    
    async def _update_strategies(self, market_data: MarketData):
        """Update all active strategies with market data"""
        try:
            for strategy_id, strategy in self.strategies.items():
                if strategy.is_active:
                    # Generate signals
                    signal = await strategy.process_market_data(market_data)
                    
                    if signal and signal.get('action') in ['buy', 'sell']:
                        # Create order
                        order = self._create_order_from_signal(signal, market_data)
                        
                        # Submit order
                        await self.submit_order(order)
            
        except Exception as e:
            logger.error(f"Error updating strategies: {e}")
    
    def _create_order_from_signal(self, signal: Dict[str, Any], market_data: MarketData) -> HFTOrder:
        """Create HFT order from strategy signal"""
        try:
            order_id = f"order_{int(time.time() * 1000000)}"
            
            # Determine order type and price
            if signal.get('order_type') == 'market':
                order_type = OrderType.MARKET
                price = None
            else:
                order_type = OrderType.LIMIT
                price = signal.get('price', market_data.mid_price)
            
            # Determine side
            side = OrderSide.BUY if signal['action'] == 'buy' else OrderSide.SELL
            
            # Calculate quantity
            quantity = signal.get('quantity', 1.0)
            
            order = HFTOrder(
                order_id=order_id,
                symbol=market_data.symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=signal.get('stop_price'),
                time_in_force=signal.get('time_in_force', 'IOC'),
                strategy_id=signal.get('strategy_id'),
                created_at=datetime.now(),
                metadata=signal.get('metadata', {})
            )
            
            return order
            
        except Exception as e:
            logger.error(f"Error creating order from signal: {e}")
            raise
    
    async def _order_processor(self):
        """Process order updates and executions"""
        try:
            while self.running:
                try:
                    # Get order update from queue
                    order_update = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            self.executor, self.order_queue.get
                        ),
                        timeout=0.001
                    )
                    
                    # Process order update
                    await self._process_order_update(order_update)
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error processing order update: {e}")
                    
        except Exception as e:
            logger.error(f"Error in order processor: {e}")
    
    async def _process_order_update(self, order_update: Dict[str, Any]):
        """Process order update from exchange"""
        try:
            order_id = order_update['order_id']
            order = self.active_orders.get(order_id)
            
            if not order:
                return
            
            # Update order status
            new_status = OrderStatus(order_update['status'])
            order.status = new_status
            order.updated_at = datetime.now()
            
            # Update fill information
            if 'filled_quantity' in order_update:
                order.filled_quantity = order_update['filled_quantity']
            
            if 'average_price' in order_update:
                order.average_price = order_update['average_price']
            
            # Handle order completion
            if new_status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                # Remove from active orders
                del self.active_orders[order_id]
                
                # Add to history
                self.order_history.append(order)
                
                # Update position
                if new_status == OrderStatus.FILLED:
                    await self._update_position(order)
            
        except Exception as e:
            logger.error(f"Error processing order update: {e}")
    
    async def _update_position(self, order: HFTOrder):
        """Update position after order fill"""
        try:
            symbol = order.symbol
            current_position = self.position_cache.get(symbol, 0)
            
            if order.side == OrderSide.BUY:
                new_position = current_position + order.filled_quantity
            else:
                new_position = current_position - order.filled_quantity
            
            self.position_cache[symbol] = new_position
            
            # Update position manager
            await self.position_manager.update_position(
                symbol=symbol,
                quantity=new_position,
                average_price=order.average_price
            )
            
        except Exception as e:
            logger.error(f"Error updating position: {e}")
    
    async def _risk_monitor(self):
        """Monitor risk in real-time"""
        try:
            while self.running:
                try:
                    # Check position limits
                    await self._check_position_limits()
                    
                    # Check daily loss limits
                    await self._check_daily_loss_limits()
                    
                    # Check latency limits
                    await self._check_latency_limits()
                    
                    # Sleep for monitoring interval
                    await asyncio.sleep(0.1)  # 100ms monitoring interval
                    
                except Exception as e:
                    logger.error(f"Error in risk monitor: {e}")
                    
        except Exception as e:
            logger.error(f"Error in risk monitor: {e}")
    
    async def _check_position_limits(self):
        """Check position limits"""
        try:
            for symbol, position in self.position_cache.items():
                if abs(position) > self.max_positions:
                    logger.warning(f"Position limit exceeded for {symbol}: {position}")
                    
                    # Close excess position
                    await self._close_excess_position(symbol, position)
            
        except Exception as e:
            logger.error(f"Error checking position limits: {e}")
    
    async def _check_daily_loss_limits(self):
        """Check daily loss limits"""
        try:
            daily_pnl = self._calculate_daily_pnl()
            max_daily_loss = self.max_daily_loss * self._get_account_value()
            
            if daily_pnl < -max_daily_loss:
                logger.warning(f"Daily loss limit exceeded: {daily_pnl}")
                
                # Stop all trading
                await self._emergency_stop()
            
        except Exception as e:
            logger.error(f"Error checking daily loss limits: {e}")
    
    async def _check_latency_limits(self):
        """Check latency limits"""
        try:
            for metric, latencies in self.latency_stats.items():
                if latencies:
                    avg_latency = np.mean(latencies[-100:])
                    if avg_latency > self.latency_threshold:
                        logger.warning(f"Latency limit exceeded for {metric}: {avg_latency}")
            
        except Exception as e:
            logger.error(f"Error checking latency limits: {e}")
    
    async def _performance_monitor(self):
        """Monitor HFT performance metrics"""
        try:
            while self.running:
                try:
                    # Calculate performance metrics
                    metrics = self._calculate_hft_metrics()
                    
                    # Update performance cache
                    self.performance_metrics = metrics
                    
                    # Log performance
                    if metrics['total_orders'] % 100 == 0:
                        logger.info(f"HFT Performance: {metrics}")
                    
                    # Sleep for monitoring interval
                    await asyncio.sleep(1.0)  # 1 second monitoring interval
                    
                except Exception as e:
                    logger.error(f"Error in performance monitor: {e}")
                    
        except Exception as e:
            logger.error(f"Error in performance monitor: {e}")
    
    def _calculate_hft_metrics(self) -> HFTMetrics:
        """Calculate HFT performance metrics"""
        try:
            # Order metrics
            total_orders = len(self.order_history)
            filled_orders = len([o for o in self.order_history if o.status == OrderStatus.FILLED])
            cancelled_orders = len([o for o in self.order_history if o.status == OrderStatus.CANCELLED])
            rejected_orders = len([o for o in self.order_history if o.status == OrderStatus.REJECTED])
            
            fill_rate = filled_orders / total_orders if total_orders > 0 else 0
            
            # Latency metrics
            latency_stats = {}
            for metric, latencies in self.latency_stats.items():
                if latencies:
                    latency_stats[metric] = {
                        'mean': np.mean(latencies[-100:]),
                        'median': np.median(latencies[-100:]),
                        'p95': np.percentile(latencies[-100:], 95),
                        'p99': np.percentile(latencies[-100:], 99)
                    }
            
            # PnL metrics
            total_pnl = sum(o.average_price * o.filled_quantity for o in self.order_history if o.status == OrderStatus.FILLED)
            total_volume = sum(o.filled_quantity for o in self.order_history if o.status == OrderStatus.FILLED)
            
            # Calculate other metrics
            win_rate = self._calculate_win_rate()
            profit_factor = self._calculate_profit_factor()
            max_drawdown = self._calculate_max_drawdown()
            sharpe_ratio = self._calculate_sharpe_ratio()
            
            return HFTMetrics(
                total_orders=total_orders,
                filled_orders=filled_orders,
                cancelled_orders=cancelled_orders,
                rejected_orders=rejected_orders,
                fill_rate=fill_rate,
                avg_fill_time=0.0,  # Would need to track fill times
                avg_spread_capture=0.0,  # Would need to track spread capture
                total_pnl=total_pnl,
                total_volume=total_volume,
                latency_stats=latency_stats,
                slippage_stats={},  # Would need to track slippage
                market_impact=0.0,  # Would need to calculate market impact
                alpha_generation=0.0,  # Would need benchmark
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor
            )
            
        except Exception as e:
            logger.error(f"Error calculating HFT metrics: {e}")
            return HFTMetrics(
                total_orders=0,
                filled_orders=0,
                cancelled_orders=0,
                rejected_orders=0,
                fill_rate=0.0,
                avg_fill_time=0.0,
                avg_spread_capture=0.0,
                total_pnl=0.0,
                total_volume=0.0,
                latency_stats={},
                slippage_stats={},
                market_impact=0.0,
                alpha_generation=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=0.0
            )
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate"""
        try:
            filled_orders = [o for o in self.order_history if o.status == OrderStatus.FILLED]
            if not filled_orders:
                return 0.0
            
            winning_orders = 0
            for order in filled_orders:
                # Calculate PnL for this order
                if order.side == OrderSide.BUY:
                    # Would need to track exit price
                    pass
                else:
                    # Would need to track exit price
                    pass
            
            return winning_orders / len(filled_orders)
            
        except Exception as e:
            logger.error(f"Error calculating win rate: {e}")
            return 0.0
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor"""
        try:
            # Would need to track total wins and losses
            return 1.0
            
        except Exception as e:
            logger.error(f"Error calculating profit factor: {e}")
            return 1.0
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        try:
            # Would need to track equity curve
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        try:
            # Would need to track returns
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def _calculate_daily_pnl(self) -> float:
        """Calculate daily PnL"""
        try:
            # Would need to track daily PnL
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating daily PnL: {e}")
            return 0.0
    
    def _get_account_value(self) -> float:
        """Get current account value"""
        try:
            # Would need to get from exchange
            return 100000.0
            
        except Exception as e:
            logger.error(f"Error getting account value: {e}")
            return 100000.0
    
    def _calculate_market_impact(self, order: HFTOrder) -> float:
        """Calculate estimated market impact"""
        try:
            # Simple market impact calculation
            order_size = order.quantity * (order.price or 1.0)
            market_volume = 1000000.0  # Would need real market volume
            
            impact = order_size / market_volume
            return min(impact, 0.01)  # Cap at 1%
            
        except Exception as e:
            logger.error(f"Error calculating market impact: {e}")
            return 0.0
    
    async def _emergency_stop(self):
        """Emergency stop all trading"""
        try:
            logger.warning("EMERGENCY STOP: Stopping all trading")
            
            # Cancel all active orders
            await self._cancel_all_orders()
            
            # Close all positions
            await self._close_all_positions()
            
            # Stop engine
            await self.stop_engine()
            
        except Exception as e:
            logger.error(f"Error in emergency stop: {e}")
    
    async def _cancel_all_orders(self):
        """Cancel all active orders"""
        try:
            for order_id in list(self.active_orders.keys()):
                await self.cancel_order(order_id)
                
        except Exception as e:
            logger.error(f"Error cancelling all orders: {e}")
    
    async def _close_all_positions(self):
        """Close all positions"""
        try:
            for symbol, position in self.position_cache.items():
                if position != 0:
                    # Create closing order
                    side = OrderSide.SELL if position > 0 else OrderSide.BUY
                    quantity = abs(position)
                    
                    order = HFTOrder(
                        order_id=f"close_{int(time.time() * 1000000)}",
                        symbol=symbol,
                        side=side,
                        order_type=OrderType.MARKET,
                        quantity=quantity,
                        price=None,
                        created_at=datetime.now()
                    )
                    
                    await self.submit_order(order)
                    
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return asdict(self.performance_metrics) if self.performance_metrics else {}
    
    def get_active_orders(self) -> List[Dict[str, Any]]:
        """Get all active orders"""
        return [asdict(order) for order in self.active_orders.values()]
    
    def get_order_history(self) -> List[Dict[str, Any]]:
        """Get order history"""
        return [asdict(order) for order in self.order_history]
    
    def get_positions(self) -> Dict[str, float]:
        """Get current positions"""
        return self.position_cache.copy()
    
    def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get latest market data for symbol"""
        return self.market_data_cache.get(symbol)
    
    def get_latency_stats(self) -> Dict[str, Any]:
        """Get latency statistics"""
        stats = {}
        for metric, latencies in self.latency_stats.items():
            if latencies:
                stats[metric] = {
                    'mean': np.mean(latencies[-100:]),
                    'median': np.median(latencies[-100:]),
                    'p95': np.percentile(latencies[-100:], 95),
                    'p99': np.percentile(latencies[-100:], 99),
                    'min': min(latencies[-100:]),
                    'max': max(latencies[-100:])
                }
        return stats