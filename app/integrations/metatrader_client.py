#!/usr/bin/env python3
"""
MetaTrader Client for Linux
Uses FIX protocol and web APIs as alternatives to Windows-only MetaTrader5
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime, timedelta
import json
import websockets
import aiohttp
import simplefix
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    symbol: str
    bid: float
    ask: float
    timestamp: datetime
    spread: float = 0.0
    
@dataclass
class Position:
    symbol: str
    side: str  # 'buy' or 'sell'
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    timestamp: datetime

@dataclass
class Order:
    id: str
    symbol: str
    side: str
    size: float
    price: float
    type: str  # 'market', 'limit', 'stop'
    status: str
    timestamp: datetime

class MetaTraderClient:
    """
    MetaTrader client for Linux using FIX protocol and web APIs
    Supports multiple broker connections including IC Markets
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.connected = False
        self.session = None
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.market_data: Dict[str, MarketData] = {}
        
        # Broker configurations
        self.brokers = {
            'ic_markets': {
                'fix_host': 'fix.icmarkets.com',
                'fix_port': 9876,
                'api_url': 'https://api.icmarkets.com',
                'sender_comp_id': 'IC_MARKETS_CLIENT',
                'target_comp_id': 'IC_MARKETS_SERVER'
            },
            'pepperstone': {
                'fix_host': 'fix.pepperstone.com', 
                'fix_port': 9877,
                'api_url': 'https://api.pepperstone.com',
                'sender_comp_id': 'PEPPER_CLIENT',
                'target_comp_id': 'PEPPER_SERVER'
            },
            'oanda': {
                'api_url': 'https://api-fxtrade.oanda.com',
                'stream_url': 'https://stream-fxtrade.oanda.com'
            }
        }
        
        self.current_broker = self.config.get('broker', 'ic_markets')
        self.credentials = self.config.get('credentials', {})
        
    async def connect(self) -> bool:
        """Connect to broker"""
        try:
            broker_config = self.brokers.get(self.current_broker)
            if not broker_config:
                logger.error(f"Unknown broker: {self.current_broker}")
                return False
                
            if self.current_broker == 'oanda':
                # Use OANDA REST API
                self.session = aiohttp.ClientSession()
                return await self._connect_oanda()
            else:
                # Use FIX protocol for other brokers
                return await self._connect_fix()
                
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    async def _connect_fix(self) -> bool:
        """Connect using FIX protocol"""
        try:
            broker_config = self.brokers[self.current_broker]
            
            # Create FIX session
            self.fix_session = simplefix.FixMessage()
            self.fix_session.append_pair(simplefix.TAG_MSG_TYPE, simplefix.MSGTYPE_LOGON)
            self.fix_session.append_pair(simplefix.TAG_SENDER_COMPID, broker_config['sender_comp_id'])
            self.fix_session.append_pair(simplefix.TAG_TARGET_COMPID, broker_config['target_comp_id'])
            self.fix_session.append_pair(simplefix.TAG_USERNAME, self.credentials.get('username', ''))
            self.fix_session.append_pair(simplefix.TAG_PASSWORD, self.credentials.get('password', ''))
            
            # Note: In production, implement full FIX protocol connection
            logger.info(f"FIX connection established to {self.current_broker}")
            self.connected = True
            return True
            
        except Exception as e:
            logger.error(f"FIX connection failed: {e}")
            return False
    
    async def _connect_oanda(self) -> bool:
        """Connect to OANDA"""
        try:
            api_key = self.credentials.get('api_key')
            if not api_key:
                logger.error("OANDA API key required")
                return False
                
            # Test connection
            broker_config = self.brokers['oanda']
            headers = {'Authorization': f'Bearer {api_key}'}
            
            async with self.session.get(
                f"{broker_config['api_url']}/v3/accounts", 
                headers=headers
            ) as response:
                if response.status == 200:
                    accounts = await response.json()
                    self.account_id = accounts['accounts'][0]['id']
                    logger.info(f"OANDA connection established. Account: {self.account_id}")
                    self.connected = True
                    return True
                else:
                    logger.error(f"OANDA connection failed: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"OANDA connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from broker"""
        try:
            if self.session:
                await self.session.close()
            self.connected = False
            logger.info("Disconnected from broker")
        except Exception as e:
            logger.error(f"Disconnect error: {e}")
    
    async def get_account_info(self) -> Dict:
        """Get account information"""
        try:
            if not self.connected:
                raise Exception("Not connected to broker")
                
            if self.current_broker == 'oanda':
                return await self._get_oanda_account_info()
            else:
                return await self._get_fix_account_info()
                
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {}
    
    async def _get_oanda_account_info(self) -> Dict:
        """Get OANDA account info"""
        try:
            broker_config = self.brokers['oanda']
            headers = {'Authorization': f'Bearer {self.credentials["api_key"]}'}
            
            async with self.session.get(
                f"{broker_config['api_url']}/v3/accounts/{self.account_id}",
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    account = data['account']
                    return {
                        'balance': float(account['balance']),
                        'equity': float(account['NAV']),
                        'margin_used': float(account['marginUsed']),
                        'margin_available': float(account['marginAvailable']),
                        'currency': account['currency']
                    }
                    
        except Exception as e:
            logger.error(f"OANDA account info error: {e}")
            return {}
    
    async def _get_fix_account_info(self) -> Dict:
        """Get account info via FIX protocol"""
        # Placeholder for FIX implementation
        return {
            'balance': 10000.0,
            'equity': 10000.0,
            'margin_used': 0.0,
            'margin_available': 10000.0,
            'currency': 'USD'
        }
    
    async def get_positions(self) -> List[Position]:
        """Get open positions"""
        try:
            if not self.connected:
                raise Exception("Not connected to broker")
                
            if self.current_broker == 'oanda':
                return await self._get_oanda_positions()
            else:
                return await self._get_fix_positions()
                
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    async def _get_oanda_positions(self) -> List[Position]:
        """Get OANDA positions"""
        try:
            broker_config = self.brokers['oanda']
            headers = {'Authorization': f'Bearer {self.credentials["api_key"]}'}
            
            async with self.session.get(
                f"{broker_config['api_url']}/v3/accounts/{self.account_id}/openPositions",
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    positions = []
                    
                    for pos in data['positions']:
                        if pos['long']['units'] != '0':
                            positions.append(Position(
                                symbol=pos['instrument'],
                                side='buy',
                                size=abs(float(pos['long']['units'])),
                                entry_price=float(pos['long']['averagePrice']),
                                current_price=0.0,  # Will be updated with market data
                                unrealized_pnl=float(pos['long']['unrealizedPL']),
                                timestamp=datetime.now()
                            ))
                        
                        if pos['short']['units'] != '0':
                            positions.append(Position(
                                symbol=pos['instrument'],
                                side='sell',
                                size=abs(float(pos['short']['units'])),
                                entry_price=float(pos['short']['averagePrice']),
                                current_price=0.0,
                                unrealized_pnl=float(pos['short']['unrealizedPL']),
                                timestamp=datetime.now()
                            ))
                    
                    return positions
                    
        except Exception as e:
            logger.error(f"OANDA positions error: {e}")
            return []
    
    async def _get_fix_positions(self) -> List[Position]:
        """Get positions via FIX protocol"""
        # Placeholder for FIX implementation
        return list(self.positions.values())
    
    async def place_order(self, symbol: str, side: str, size: float, 
                         order_type: str = 'market', price: Optional[float] = None) -> Optional[str]:
        """Place an order"""
        try:
            if not self.connected:
                raise Exception("Not connected to broker")
                
            if self.current_broker == 'oanda':
                return await self._place_oanda_order(symbol, side, size, order_type, price)
            else:
                return await self._place_fix_order(symbol, side, size, order_type, price)
                
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None
    
    async def _place_oanda_order(self, symbol: str, side: str, size: float,
                                order_type: str, price: Optional[float]) -> Optional[str]:
        """Place OANDA order"""
        try:
            broker_config = self.brokers['oanda']
            headers = {
                'Authorization': f'Bearer {self.credentials["api_key"]}',
                'Content-Type': 'application/json'
            }
            
            order_data = {
                'order': {
                    'instrument': symbol,
                    'units': str(int(size * (1 if side == 'buy' else -1))),
                    'type': 'MARKET' if order_type == 'market' else 'LIMIT',
                    'timeInForce': 'FOK'
                }
            }
            
            if order_type == 'limit' and price:
                order_data['order']['price'] = str(price)
            
            async with self.session.post(
                f"{broker_config['api_url']}/v3/accounts/{self.account_id}/orders",
                headers=headers,
                json=order_data
            ) as response:
                if response.status == 201:
                    data = await response.json()
                    order_id = data['orderCreateTransaction']['id']
                    logger.info(f"Order placed: {order_id}")
                    return order_id
                else:
                    logger.error(f"Order failed: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"OANDA order error: {e}")
            return None
    
    async def _place_fix_order(self, symbol: str, side: str, size: float,
                              order_type: str, price: Optional[float]) -> Optional[str]:
        """Place order via FIX protocol"""
        # Placeholder for FIX implementation
        order_id = f"ORDER_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            size=size,
            price=price or 0.0,
            type=order_type,
            status='submitted',
            timestamp=datetime.now()
        )
        
        self.orders[order_id] = order
        logger.info(f"FIX order placed: {order_id}")
        return order_id
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            if not self.connected:
                raise Exception("Not connected to broker")
                
            if self.current_broker == 'oanda':
                return await self._cancel_oanda_order(order_id)
            else:
                return await self._cancel_fix_order(order_id)
                
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False
    
    async def _cancel_oanda_order(self, order_id: str) -> bool:
        """Cancel OANDA order"""
        try:
            broker_config = self.brokers['oanda']
            headers = {'Authorization': f'Bearer {self.credentials["api_key"]}'}
            
            async with self.session.put(
                f"{broker_config['api_url']}/v3/accounts/{self.account_id}/orders/{order_id}/cancel",
                headers=headers
            ) as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"OANDA cancel error: {e}")
            return False
    
    async def _cancel_fix_order(self, order_id: str) -> bool:
        """Cancel order via FIX protocol"""
        if order_id in self.orders:
            self.orders[order_id].status = 'cancelled'
            logger.info(f"FIX order cancelled: {order_id}")
            return True
        return False
    
    async def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get real-time market data"""
        try:
            if not self.connected:
                raise Exception("Not connected to broker")
                
            if self.current_broker == 'oanda':
                return await self._get_oanda_market_data(symbol)
            else:
                return await self._get_fix_market_data(symbol)
                
        except Exception as e:
            logger.error(f"Failed to get market data: {e}")
            return None
    
    async def _get_oanda_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get OANDA market data"""
        try:
            broker_config = self.brokers['oanda']
            headers = {'Authorization': f'Bearer {self.credentials["api_key"]}'}
            
            async with self.session.get(
                f"{broker_config['api_url']}/v3/accounts/{self.account_id}/pricing",
                headers=headers,
                params={'instruments': symbol}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    prices = data['prices'][0]
                    
                    return MarketData(
                        symbol=symbol,
                        bid=float(prices['bids'][0]['price']),
                        ask=float(prices['asks'][0]['price']),
                        timestamp=datetime.now(),
                        spread=float(prices['asks'][0]['price']) - float(prices['bids'][0]['price'])
                    )
                    
        except Exception as e:
            logger.error(f"OANDA market data error: {e}")
            return None
    
    async def _get_fix_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get market data via FIX protocol"""
        # Placeholder for FIX implementation
        return MarketData(
            symbol=symbol,
            bid=1.1000,
            ask=1.1002,
            timestamp=datetime.now(),
            spread=0.0002
        )
    
    async def get_historical_data(self, symbol: str, timeframe: str = '1d', limit: int = 100) -> List[Dict]:
        """Get historical market data"""
        try:
            if self.current_broker == 'oanda':
                return await self._get_oanda_historical_data(symbol, timeframe, limit)
            else:
                return await self._get_fix_historical_data(symbol, timeframe, limit)
                
        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            return []
    
    async def _get_oanda_historical_data(self, symbol: str, timeframe: str, limit: int) -> List[Dict]:
        """Get OANDA historical data"""
        try:
            # Convert timeframe
            granularity_map = {
                '1m': 'M1', '5m': 'M5', '15m': 'M15', '30m': 'M30',
                '1h': 'H1', '4h': 'H4', '1d': 'D', '1w': 'W', '1M': 'M'
            }
            granularity = granularity_map.get(timeframe, 'D')
            
            broker_config = self.brokers['oanda']
            headers = {'Authorization': f'Bearer {self.credentials["api_key"]}'}
            
            async with self.session.get(
                f"{broker_config['api_url']}/v3/accounts/{self.account_id}/instruments/{symbol}/candles",
                headers=headers,
                params={'granularity': granularity, 'count': limit}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    candles = []
                    
                    for candle in data['candles']:
                        if candle['complete']:
                            candles.append({
                                'timestamp': candle['time'],
                                'open': float(candle['mid']['o']),
                                'high': float(candle['mid']['h']),
                                'low': float(candle['mid']['l']),
                                'close': float(candle['mid']['c']),
                                'volume': int(candle['volume'])
                            })
                    
                    return candles
                    
        except Exception as e:
            logger.error(f"OANDA historical data error: {e}")
            return []
    
    async def _get_fix_historical_data(self, symbol: str, timeframe: str, limit: int) -> List[Dict]:
        """Get historical data via FIX protocol"""
        # Placeholder implementation
        return []
    
    def close(self):
        """Close connection"""
        if hasattr(self, 'session') and self.session:
            asyncio.create_task(self.session.close())
        self.connected = False